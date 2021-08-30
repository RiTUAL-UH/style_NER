import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()

        # Attention module
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, dec_h_prev, enc_h_all, epsilon=1e-8):
        w = self.W(dec_h_prev).unsqueeze(1)  # (batch, 1, hidden)
        u = self.U(enc_h_all)  # (batch, seq, hidden)

        s = self.v(torch.tanh(w + u))

        # Masked softmax
        m, _ = s.max(dim=1, keepdim=True)
        s = torch.exp(s - m)  # num stable softmax to avoid overflow
        a = s / (torch.sum(s, dim=1, keepdim=True) + epsilon)  # avoid underflow with epsilon

        # Context vector: weighted sum of the input vectors
        c = torch.sum(enc_h_all * a, dim=1)

        return c, a

