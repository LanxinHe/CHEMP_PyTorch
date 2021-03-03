import torch
import torch.nn as nn


class InputLayer(nn.Module):
    def __init__(self, length, sigma_square_v):
        super(InputLayer, self).__init__()
        self.length = length
        self.symbols = torch.linspace(1-length, length-1, length)
        self.sigma_square_v = sigma_square_v

    def forward(self, inputs, p_pre):
        # z_i (batch_size, 1);  J_i (batch_size, 2tx+1)
        z_i, J_i = inputs
        # p_pre (batch_size, length)
        expectation = torch.matmul(p_pre, self.symbols)  # (batch_size)
        var = torch.matmul(p_pre, torch.square(self.symbols)) - torch.square(expectation)   # (batch_size)
        mu = torch.einsum('ij,i->i', J_i[:, :-1], expectation) - torch.multiply(J_i[:, -1], expectation)  # (batch_size)
        sigma_square = torch.einsum('ij,i->i', torch.square(J_i[:, :-1]), var) -\
            torch.multiply(torch.square(J_i[:, -1]), var) + self.sigma_square_v     # (batch_size)
        s_add = (self.symbols + self.symbols[0]).unsqueeze(0).repeat(z_i.shape[0], 1)      # shape: (batch_size, length)
        s_minor = (self.symbols - self.symbols[0]).unsqueeze(0).repeat(z_i.shape[0], 1)   # shape: (batch_size, length)
        """
        In the sequel, we are trying to use RNN to predict LLR, in the hope that under the bad-conditional circumstance,
        we can guess the distribution of g_i better 
        """
        rnn_inputs = torch.cat([z_i, mu, sigma_square, s_add, s_minor, J_i])
        return rnn_inputs


class InputLayerWhole(nn.Module):
    def __init__(self, length, sigma_square_v):
        super(InputLayerWhole, self).__init__()
        self.length = length
        self.symbols = torch.linspace(1-length, length-1, length)
        self.sigma_square_v = sigma_square_v

    def forward(self, inputs, p_pre):
        z, j_matrix = inputs    # z(batch_size, 2tx) j_matrix(batch_size, 2tx, 2tx)
        # let p_pre be in shape of (batch_size, length, 2tx)
        expectation = torch.einsum('ijk,j->ik', p_pre, self.symbols)  # (batch_size, 2tx)
        # var has the same shape as expectation: (batch_size, 2tx)
        var = torch.einsum('ijk,j->k', p_pre, torch.square(self.symbols)) - torch.square(expectation)
        # mu should be in shape of (batch_size, 2tx)
        # Here dim1 and dim2 are used to take a batch diagonal
        j_diagonal = torch.diagonal(j_matrix, dim1=-2, dim2=-1)     # shape: (batch_size, 2tx)
        mu = torch.einsum('ijk,ik->ij', j_matrix, expectation) -\
            torch.multiply(expectation, j_diagonal)
        # sigma_square should be in shape of (batch_size, 2tx)
        sigma_square = torch.einsum('ijk, ik->ij', torch.square(j_matrix), var) + self.sigma_square_v -\
            torch.multiply(var, j_diagonal)
        s_add = (self.symbols + self.symbols[0]).unsqueeze(0).unsqueeze(0).repeat(z.shape[0], z.shape[1], 1)
        # shape: (batch_size, 2tx, length)
        s_minor = (self.symbols - self.symbols[0]).unsqueeze(0).unsqueeze(0).repeat(z.shape[0], z.shape[1], 1)

        rnn_inputs = torch.cat([z.unsqueeze(-1), expectation.unsqueeze(-1), var.unsqueeze(-1),
                                mu.unsqueeze(-1), sigma_square.unsqueeze(-1), j_matrix, s_add, s_minor], dim=2)
        """(batch_size, 2tx, input_size), input_size = 2tx + 2length + 5"""
        return rnn_inputs


class RecurentPart(nn.Module):
    def __init__(self, tx, length, lstm_layers):
        super(RecurentPart, self).__init__()
        self.tx = tx
        self.length = length
        input_size = 2*(tx + length) + 5
        # Here output_size is length + 1, '1' for the delta
        self.lstm = nn.LSTM(input_size, length+1, bidirectional=True, num_layers=lstm_layers)
        # Here weights and bias are different from those in Tan's paper.
        # They are used to recover the information from bi-directional LSTM.
        self.w = nn.Parameter(torch.randn([length+1, 2*(length+1)]), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([length+1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.permute(1, 0, 2)    # (2tx, batch_size, input_size)
        outputs, (_, _) = self.lstm(inputs)
        outputs = torch.matmul(outputs, self.w.T) + self.b    # likelihood (2tx, batch_size, hidden_size)
        split = torch.split(outputs, [self.length, 1], dim=-1)
        likelihood, delta = split[0], split[1]
        delta = self.sigmoid(delta.squeeze(-1)).permute(1, 0)   # (batch_size, 2tx)
        p_tilde = self.softmax(likelihood.permute(1, 2, 0))  # (batch_size, length, 2tx)
        return p_tilde, delta


class RecurrentCHEMP(nn.Module):
    def __init__(self, iterations, tx, length, sigma_square_v, lstm_layers):
        super(RecurrentCHEMP, self).__init__()
        for i in range(iterations):
            setattr(self, 'input_layer'+str(i), InputLayerWhole(length, sigma_square_v))
            setattr(self, 'rnn'+str(i), RecurentPart(tx, length, lstm_layers))
        self.iterations = iterations
        self.length = length
        self.tx = tx

    def forward(self, inputs):
        z, J = inputs
        batch_size = z.shape[0]
        p = torch.ones([batch_size, self.length, 2 * self.tx]) / self.length
        for i in range(self.iterations):
            rnn_inputs = getattr(self, 'input_layer'+str(i))(inputs, p)
            p_tilde, delta = getattr(self, 'rnn'+str(i))(rnn_inputs)
            p = torch.einsum('ijk,ik->ijk', p_tilde, 1 - delta) + torch.einsum('ijk,ik->ijk', p, delta)
        return p




