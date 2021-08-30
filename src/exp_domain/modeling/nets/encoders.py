import torch
import torch.nn as nn

from src.exp_domain.modeling.nets.layers import LSTMLayer


class LSTMEncoder(nn.Module):
    def __init__(self, src_embedder, tgt_embedder, input_dim, hidden_dim, latent_dim, bidirectional, num_layers, dropout):
        super(LSTMEncoder, self).__init__()

        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder

        self.encoder = LSTMLayer(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                bidirectional=bidirectional,
                                num_layers=num_layers,
                                drop_prob=dropout)

        self.h2mu = nn.Linear(hidden_dim, latent_dim)
        self.h2logvar = nn.Linear(hidden_dim, latent_dim)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.src_embedder.embedding.weight.data.uniform_(-0.1, 0.1)
        self.tgt_embedder.embedding.weight.data.uniform_(-0.1, 0.1)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, domain, noise=False):
        embedder = self.src_embedder if domain == 'src' else self.tgt_embedder

        embedded = embedder(x, noise=noise)
        embedded = self.dropout(embedded)
        
        encoded = self.encoder(embedded)
        encoded['outs'] = self.dropout(encoded['outs'])
        encoded['last'] = self.dropout(encoded['last'])
        
        mu = self.h2mu(encoded['last'])
        logvar = self.h2logvar(encoded['last'])
        
        z = self._reparameterize(mu, logvar)
        
        results = {'mu': mu, 'logvar': logvar, 'z': z, 'outs': encoded['outs'], 'last': encoded['last']}

        return results

