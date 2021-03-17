import torch
import torch.nn as nn


"""
If we are trying to use RNN in CHEMPLayer
"""


class CHEMPLayer(nn.Module):
    def __init__(self, length, input_size, hidden_size, rnn_layers):
        """
        :param length:
        :param input_size: 2tx
        :param hidden_size: for RNN, the feature dim of the hidden state
        """
        super(CHEMPLayer, self).__init__()
        self.length = length
        self.input_size = input_size  # 2tx

        self.symbols = torch.linspace(1-length, length-1, length)
        # If delta is in shape of (length, input_size), then the sum of probability would be greater than 1
        # self.delta = nn.Parameter(torch.ones([input_size]) * 0.28, requires_grad=True)
        # Here weights and bias are different from those in Tan's paper.
        # They are used to recover the information from (bi-directional) LSTM.
        self.w_likelihood = nn.Parameter(torch.randn([length, hidden_size]), requires_grad=True)
        self.b_likelihood = nn.Parameter(torch.zeros([length]), requires_grad=True)

        self.w_delta = nn.Parameter(torch.randn([1, hidden_size]))
        self.b_delta = nn.Parameter(torch.zeros([1]))
        self.gru = nn.GRU(length, hidden_size, num_layers=rnn_layers)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, p_pre, h_pre, sigma_square_v):
        z, j_matrix = inputs    # z(batch_size, 2tx) j_matrix(batch_size, 2tx, 2tx)
        # let p_pre be in shape of (batch_size, length, 2tx)
        expectation = torch.einsum('ijk,j->ik', p_pre, self.symbols)  # (batch_size, 2tx)
        # var has the same shape as expectation: (batch_size, 2tx)
        var = torch.einsum('ijk,j->ik', p_pre, torch.square(self.symbols)) - torch.square(expectation)
        # mu should be in shape of (batch_size, 2tx)
        # Here dim1 and dim2 are used to take a batch diagonal
        j_diagonal = torch.diagonal(j_matrix, dim1=-2, dim2=-1)     # shape: (batch_size, 2tx)
        mu = torch.einsum('ijk,ik->ij', j_matrix, expectation) -\
            torch.multiply(expectation, j_diagonal)
        # sigma_square should be in shape of (batch_size, 2tx)
        sigma_square = torch.einsum('ijk, ik->ij', torch.square(j_matrix), var) + sigma_square_v -\
            torch.multiply(var, torch.square(j_diagonal))

        """L_i(s_k) = [(2(z_i-mu_i)-J_ii(s_k+s_1))(J_ii(s_k-s_1))]/(2sigma_i^2)"""
        # Therefore, we had better to calculate (s_k + s_1) and (s_k - s_1) in the beginning
        s_add = self.symbols + self.symbols[0]      # shape: (length)
        s_minor = self.symbols - self.symbols[0]    # shape: (length)
        # Meanwhile we also need to expand the dimension (use torch.repeat())
        temp1 = 2*(z-mu).unsqueeze(1).repeat(1, self.length, 1)   # 2(z - mu)
        temp2 = torch.einsum('ij,k->ikj', j_diagonal, s_add)    # J_ii(s_k + s_1)
        temp3 = torch.einsum('ij,k->ikj', j_diagonal, s_minor)  # J_ii(s_k - s_1)
        temp4 = sigma_square.unsqueeze(1).repeat(1, self.length, 1)     # sigma_square
        # log-likelihood should be in shape of (batch_size, length, 2tx)
        likelihood = ((temp1 - temp2) * temp3) / 2 / temp4
        # Bound the log-likelihood lower than 88 anyway to avoid Inf.
        likelihood = torch.where(likelihood > 88., torch.full_like(likelihood, 88), likelihood)

        gru_outputs, h = self.gru(likelihood.permute(2, 0, 1), h_pre)
        # (2tx, batch_size, length)
        likelihood_tilde = torch.matmul(gru_outputs, self.w_likelihood.T) + self.b_likelihood
        likelihood_tilde = torch.where(likelihood_tilde > 88., torch.full_like(likelihood_tilde, 88), likelihood_tilde)
        delta = self.sigmoid(torch.matmul(gru_outputs, self.w_delta.T) + self.b_delta)  # (2tx, batch_size, 1)
        delta = delta.permute(1, 2, 0).repeat([1, self.length, 1])  # (batch_size, length, 2tx)

        p_tilde = self.softmax(likelihood_tilde.permute(1, 2, 0))
        p = torch.multiply(p_tilde, 1 - delta) + torch.multiply(p_pre, delta)
        return p, h


class CHEMPModel(nn.Module):
    def __init__(self, length, input_size, iterations, hidden_size, rnn_layers):
        super(CHEMPModel, self).__init__()
        self.length = length
        self.input_size = input_size
        self.iterations = iterations
        for layer in range(iterations):
            setattr(self, 'layer_'+str(layer), CHEMPLayer(length, input_size, hidden_size, rnn_layers))

    def forward(self, inputs, p, h, sigma_square_v):
        for i in range(self.iterations):
            p, h = getattr(self, 'layer_'+str(i))(inputs, p, h, sigma_square_v)
        return p, h




