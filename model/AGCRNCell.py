import torch
import torch.nn as nn
from model.AGCN import AVWGCN

class AGCRNCell(nn.Module):
    """Modified GRU cell."""
    
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(self, x, state, node_embeddings):
        # Equation 7
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)  # previous hidden state
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))  # z and r
        # The r and z are swapped, compared to the paper
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))  # new candidate hidden state
        h = r*state + (1-r)*hc  # new hidden state
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)