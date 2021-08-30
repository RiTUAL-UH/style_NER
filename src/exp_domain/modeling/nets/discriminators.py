import torch
import torch.nn as nn

from src.exp_domain.modeling.nets.layers import LSTMLayer


class MLPDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, drop_prob, smooth_labels):
        super(MLPDiscriminator, self).__init__()

        self.input_dim = input_dim
        self.smooth_labels = smooth_labels

        self.features = []
        for i in range(num_layers):
            input_dim = input_dim if i == 0 else hidden_dim
            self.features += [
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(drop_prob)
            ]
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.binary_xentropy = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets=None):
        features = self.features(inputs)
        logits = self.classifier(features)

        loss = None
        if targets is not None:
            if self.smooth_labels is not None:
                targets = targets * self.smooth_labels

            loss = self.binary_xentropy(logits, targets)

        result = {'logits': logits, 'loss': loss}

        return result


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, drop_prob, smooth_labels):
        super(LSTMDiscriminator, self).__init__()

        self.features = LSTMLayer(input_dim=input_dim, hidden_dim=hidden_dim,
                                  bidirectional=True, num_layers=num_layers,
                                  drop_prob=drop_prob)
                                  
        self.drop = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.binary_xentropy = nn.BCEWithLogitsLoss()
        self.smooth_labels = smooth_labels

    def forward(self, inputs, targets=None):
        features = self.features(inputs)
        features = self.drop(features['last'])
        logits = self.classifier(features)

        loss = None
        if targets is not None:
            if self.smooth_labels is not None:
                targets = targets * self.smooth_labels

            loss = self.binary_xentropy(logits, targets)

        result = {'logits': logits, 'loss': loss}
        
        return result

