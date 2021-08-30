import random
import torch
import torch.nn as nn

from src.exp_domain.modeling.nets.layers import LSTMLayer
from src.exp_domain.modeling.nets.attention import BahdanauAttention


class LSTMDecoder(nn.Module):
    def __init__(self, src_embedder, tgt_embedder, input_dim, hidden_dim, latent_dim, bidirectional, num_layers, src_vocab_size, tgt_vocab_size, dropout):
        super().__init__()

        self.pad_ix = src_embedder.get_pad_ix()
        self.unk_ix = src_embedder.get_unk_ix()
        self.bos_ix = src_embedder.get_bos_ix()
        self.eos_ix = src_embedder.get_eos_ix()
        self.msk_ix = src_embedder.get_msk_ix()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder

        self.src_z2emb = nn.Linear(latent_dim, input_dim)
        self.tgt_z2emb = nn.Linear(latent_dim, input_dim)
        
        self.decoder = LSTMLayer(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                bidirectional=bidirectional,
                                num_layers=num_layers,
                                drop_prob=dropout)
        
        self.src_projector = nn.Linear(hidden_dim, src_vocab_size)
        self.tgt_projector = nn.Linear(hidden_dim, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        self.src_embedder.embedding.weight.data.uniform_(-0.1, 0.1)
        self.tgt_embedder.embedding.weight.data.uniform_(-0.1, 0.1)

        self.src_projector.bias.data.zero_()
        self.tgt_projector.bias.data.zero_()

        self.src_projector.weight.data.uniform_(-0.1, 0.1)
        self.tgt_projector.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden(self, batch_size):
        directions = 2 if self.bidirectional else 1

        hx = torch.zeros(self.num_layers * directions, batch_size, self.hidden_dim)
        cx = torch.zeros(self.num_layers * directions, batch_size, self.hidden_dim)

        return hx, cx
    
    def forward(self, x, z, domain, noise=False):
        embedder = self.src_embedder if domain == 'src' else self.tgt_embedder
        z2emb = self.src_z2emb if domain == 'src' else self.tgt_z2emb
        projector = self.src_projector if domain == 'src' else self.tgt_projector

        embedded = embedder(x, noise=noise)
        embedded = self.dropout(embedded)

        if z is not None:
            embedded = embedded + z2emb(z).unsqueeze(1)

        decoded = self.decoder(embedded)
        decoded['outs'] = self.dropout(decoded['outs'])

        logits = projector(decoded['outs'])

        results = {'logits': logits}

        return results
    
    def generate(self, z, max_sent_length, temperature, algorithm, domain, device, strip=True):
        embedder = self.src_embedder if domain == 'src' else self.tgt_embedder
        z2emb = self.src_z2emb if domain == 'src' else self.tgt_z2emb
        projector = self.src_projector if domain == 'src' else self.tgt_projector

        batch_size = z.size(0)

        hidden = None

        inp = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.bos_ix).to(device)
        out = torch.empty(batch_size, max_sent_length, dtype=torch.long).fill_(self.pad_ix)

        for t in range(max_sent_length):
            out[:, t] = inp.squeeze(1).cpu().data
            
            embedded = embedder(inp) + z2emb(z).unsqueeze(1)
            decoded = self.decoder(embedded, hidden=hidden)
            logits = projector(decoded['outs'])

            # do no sample <UNK> and <MSK> words
            logits.index_fill_(2, torch.tensor([self.unk_ix]).to(device), -1e30)
            logits.index_fill_(2, torch.tensor([self.msk_ix]).to(device), -1e30)

            if algorithm == 'greedy':
                inds = logits.argmax(dim=-1)
            elif algorithm == 'sample':
                inds = torch.multinomial((logits.squeeze() / temperature).exp(), num_samples=1)
            elif algorithm == 'top5':
                logits_exp = (logits / temperature).exp()
                not_top5_indices = logits_exp.topk(logits_exp.size(-1) - 10,dim=2,largest=False).indices
                for i in range(logits_exp.size(0)):
                    logits_exp[i, :, not_top5_indices[i]] = 0.
                inds = torch.multinomial(logits_exp.squeeze(), num_samples=1)
            
            # prepare the input to the decoder for next timestep
            inp = inds.to(device)
            hidden = decoded['hidden']

        out = out.tolist()

        if strip:
            # remove <BOS> and <END> words
            out = [s[1:s.index(self.eos_ix)] if self.eos_ix in s else s[1:] for s in out]
            return out
        else:
            # add <PAD> words
            for i, s in enumerate(out):
                if self.eos_ix in s:
                    eos_idx = s.index(self.eos_ix)
                    out[i] = s[:eos_idx+1] + [self.pad_ix] * (len(s) - eos_idx - 1)
        
            return torch.tensor(out).to(device)


class AttentionDecoder(nn.Module):
    def __init__(self, src_embedder, tgt_embedder, input_dim, hidden_dim, src_vocab_size, tgt_vocab_size, dropout):
        super().__init__()

        self.pad_ix = src_embedder.get_pad_ix()
        self.unk_ix = src_embedder.get_unk_ix()
        self.bos_ix = src_embedder.get_bos_ix()
        self.eos_ix = src_embedder.get_eos_ix()
        self.msk_ix = src_embedder.get_msk_ix()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder

        self.decoder = nn.LSTMCell(input_dim, hidden_dim)
        self.attention = BahdanauAttention(hidden_dim)

        self.src_projector = nn.Linear(hidden_dim * 2, src_vocab_size)
        self.tgt_projector = nn.Linear(hidden_dim * 2, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size, device):
        hx = torch.zeros(batch_size, self.hidden_dim).to(device)
        cx = torch.zeros(batch_size, self.hidden_dim).to(device)
        return hx, cx

    def forward(self, x, z, teacher, domain, teacher_forcing_ratio=0.5):
        embedder = self.src_embedder if domain == 'src' else self.tgt_embedder
        projector = self.src_projector if domain == 'src' else self.tgt_projector

        batch_size = x.size(0)
        seq_length = x.size(1)

        hx, cx = self.init_hidden(batch_size, x.device)  # the initial hidden state

        decoded_logits = []  # track decoded embeddings

        inputs = x[:, 0]

        for i in range(seq_length):
            embedded = embedder(inputs)

            hx, cx = self.decoder(embedded, (hx, cx))
            hx = self.dropout(hx)

            # get context vector based on attention
            cxt, _ = self.attention(hx, z)
            cxt = self.dropout(cxt)

            logits = projector(torch.cat((cxt, hx), dim=1))

            guess = logits.argmax(dim=-1).view(-1)

            inputs = teacher[:, i] if random.random() < teacher_forcing_ratio else guess

            decoded_logits.append(logits)

        decoded_logits = torch.cat(decoded_logits, dim=1).view(batch_size, seq_length, -1)

        result = {'logits': decoded_logits}

        return result

    def generate(self, z, max_sent_length, temperature, algorithm, domain, device, strip=True):
        embedder = self.src_embedder if domain == 'src' else self.tgt_embedder
        projector = self.src_projector if domain == 'src' else self.tgt_projector

        batch_size = z.size(0)

        inp = torch.empty(batch_size, dtype=torch.long).fill_(self.bos_ix).to(device)
        out = torch.empty(batch_size, max_sent_length, dtype=torch.long).fill_(self.eos_ix)

        hx, cx = self.init_hidden(batch_size, z.device)  # the initial hidden state

        for t in range(max_sent_length):
            out[:, t] = inp.cpu().data

            embedded = embedder(inp)

            # hx, cx = self.decoder(embedded, (hx, cx))
            hx, cx = self.decoder(embedded, (hx, cx))
            hx = self.dropout(hx)

            # get context vector based on attention
            cxt, _ = self.attention(hx, z)
            cxt = self.dropout(cxt)

            logits = projector(torch.cat((cxt, hx), dim=1))

            # do no sample <UNK> words
            logits.index_fill_(1, torch.tensor([self.unk_ix]).to(device), -1e30)

            logits = logits.view(-1, logits.shape[-1])

            if algorithm == 'greedy':
                inds = logits.argmax(dim=-1)
            elif algorithm == 'sample':
                inds = torch.multinomial((logits.squeeze() / temperature).exp(), num_samples=1)
            elif algorithm == 'top5':
                logits_exp = (logits / temperature).exp()
                not_top5_indices = logits_exp.topk(logits_exp.size(-1) - 10, dim=-1,largest=False).indices
                for i in range(logits_exp.size(0)):
                    logits_exp[i, not_top5_indices[i]] = 0.
                inds = torch.multinomial(logits_exp.squeeze(), num_samples=1)
            
            # prepare the input to the decoder for next timestep
            inp = inds.view(-1)

        out = out.tolist()

        if strip:
            # remove <BOS> and <END> words
            out = [s[1:s.index(self.eos_ix)] if self.eos_ix in s else s[1:] for s in out]
            return out
        else:
            # add <PAD> words
            for i, s in enumerate(out):
                if self.eos_ix in s:
                    eos_idx = s.index(self.eos_ix)
                    out[i] = s[:eos_idx+1] + [self.pad_ix] * (len(s) - eos_idx - 1)
        
            return torch.tensor(out).to(device)

