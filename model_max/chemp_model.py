import torch
import torch.nn as nn


"""
This is the modified CHEMP model proposed by TAN Xiaosi  
"""


class CHEMPLayer(nn.Module):
    def __init__(self, length, input_size):
        super(CHEMPLayer, self).__init__()
        self.length = length
        self.input_size = input_size  # 2tx

        self.symbols = torch.linspace(1-length, length-1, length)
        # If delta is in shape of (length, input_size), then the sum of probability would be greater than 1
        self.delta = nn.Parameter(torch.ones([input_size]) * 0.28, requires_grad=True)
        self.w = nn.Parameter(torch.ones([length, input_size]), requires_grad=True)
        self.b = nn.Parameter(torch.zeros([length, input_size]), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, p_pre, sigma_square_v):
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
        temp1 = (2*(z-mu)).unsqueeze(1).repeat(1, self.length, 1)   # 2(z - mu)
        temp2 = torch.einsum('ij,k->ikj', j_diagonal, s_add)    # J_ii(s_k + s_1)
        temp3 = torch.einsum('ij,k->ikj', j_diagonal, s_minor)  # J_ii(s_k - s_1)
        temp4 = sigma_square.unsqueeze(1).repeat(1, self.length, 1)     # sigma_square
        # log-likelihood should be in shape of (batch_size, length, 2tx)
        likelihood = ((temp1 - temp2) * temp3) / 2 / temp4
        likelihood_tilde = torch.multiply(likelihood, self.w) + self.b
        # Use max trick to avoid overflow
        log_max, _ = torch.max(likelihood_tilde, dim=1)
        log_max = log_max.unsqueeze(1).repeat([1, self.length, 1])
        p_tilde = self.softmax(likelihood_tilde - log_max)
        p = torch.multiply(p_tilde, 1 - self.delta) + torch.multiply(p_pre, self.delta)
        return p


class CHEMPModel(nn.Module):
    def __init__(self, length, input_size, n_layers):
        super(CHEMPModel, self).__init__()
        self.length = length
        self.input_size = input_size
        self.n_layers = n_layers
        for layer in range(n_layers):
            setattr(self, 'chemp_layer_'+str(layer), CHEMPLayer(length, input_size))

    def forward(self, inputs, sigma_square_v):
        z, _ = inputs
        batch_size = z.shape[0]
        p = torch.ones([batch_size, self.length, self.input_size]) / self.length    # initial probability
        for layer in range(self.n_layers):
            p = getattr(self, 'chemp_layer_'+str(layer))(inputs, p, sigma_square_v)
        return p

