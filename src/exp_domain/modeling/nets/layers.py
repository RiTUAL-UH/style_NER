import torch
import torch.nn as nn


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional, num_layers, drop_prob=0.3):
        super(LSTMLayer, self).__init__()

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim // 2 if bidirectional else hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=drop_prob if num_layers > 1 else 0,
                            batch_first=True)

        self._flatten_parameters()

    def _flatten_parameters(self):
        self.lstm.flatten_parameters()

    def forward(self, vectors, mask=None, hidden=None):
        batch_size = vectors.size(0)
        max_length = vectors.size(1)

        if mask is None:
            mask = torch.ones(batch_size, max_length).long().to(vectors.device)

        lengths = mask.view(batch_size, max_length).long().sum(-1)

        if hidden is not None:
            directions = 2 if self.lstm.bidirectional else 1
            (h0, c0) = hidden  # (num_layers * num_directions, batch, hidden_size)
            assert h0.size(0) == c0.size(0) == self.num_layers * directions
            assert h0.size(1) == c0.size(1) == batch_size
            assert h0.size(2) == c0.size(2) == self.hidden_dim // directions

        lstm_outs, (hn, cn) = self.lstm(vectors, hidden)  # (batch, seq_len, num_directions * hidden_size)

        assert lstm_outs.size(0) == batch_size
        assert lstm_outs.size(1) == max_length
        assert lstm_outs.size(2) == self.hidden_dim

        if self.bidirectional:
            # Separate the directions of the LSTM
            lstm_outs = lstm_outs.view(batch_size, max_length, 2, self.hidden_dim // 2)

            # Pick up the last hidden state per direction
            fw_last_hn = lstm_outs[range(batch_size), lengths - 1, 0]   # (batch, hidden // 2)
            bw_last_hn = lstm_outs[range(batch_size), 0, 1]             # (batch, hidden // 2)

            lstm_outs = lstm_outs.view(batch_size, max_length, self.hidden_dim)

            last_hn = torch.cat([fw_last_hn, bw_last_hn], dim=1)        # (batch, hidden // 2) -> (batch, hidden)
        else:
            last_hn = lstm_outs[range(batch_size), lengths - 1]         # (batch, hidden)
        
        return {'last': last_hn, 'outs': lstm_outs, 'hidden': (hn, cn)}

