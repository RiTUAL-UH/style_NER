import torch
import numpy as np
import torch.nn as nn

from typing import List, Tuple


class TokenEmbedder(nn.Module):
    def __init__(self, vocab: List[Tuple[str, int]], vocab_size, embedding_dim, shuffle_param, dropout_param, masking_param):
        super(TokenEmbedder, self).__init__()

        self.PAD_WORD = '<PAD>'
        self.UNK_WORD = '<UNK>'
        self.BOS_WORD = '<BOS>'
        self.EOS_WORD = '<EOS>'
        self.MSK_WORD = '<MSK>'

        self.index2word = [self.PAD_WORD, self.UNK_WORD, self.BOS_WORD, self.EOS_WORD, self.MSK_WORD]

        token_set = set(self.index2word)

        for word, freq in vocab[:vocab_size]:
            if word not in token_set:
                self.index2word.append(word)
                token_set.add(word)
        
        self.index2word = dict(enumerate(self.index2word))
        self.word2index = {w: i for i, w in self.index2word.items()}

        self.vocab_size = len(self.index2word)
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.word2index[self.PAD_WORD])

        self.num_embeddings = self.embedding.num_embeddings

        self.shuffle_param = shuffle_param
        self.dropout_param = dropout_param
        self.masking_param = masking_param

    @staticmethod
    def _map_terms(mapper, to_map, UNK_IX, BOS_IX=None, EOS_IX=None, do_padding=False, do_normalize=False):
        mapped = []
        for terms in to_map:
            if do_normalize:
                mapped_terms = [mapper.get(normalize(t), UNK_IX) for t in terms]
            else:
                mapped_terms = [mapper.get(t, UNK_IX) for t in terms]

            if do_padding == True:
                mapped_terms = [BOS_IX] + mapped_terms + [EOS_IX]

            mapped.append(mapped_terms)

        return mapped

    def encode(self, sentences, do_padding=True, do_normalize=True):
        BOS_IX = self.get_bos_ix()
        EOS_IX = self.get_eos_ix()
        UNK_IX = self.get_unk_ix()
        return self._map_terms(self.word2index, sentences, UNK_IX, BOS_IX, EOS_IX, do_padding=do_padding, do_normalize=do_normalize)

    def decode(self, encodings, do_padding=False, do_normalize=False):
        return self._map_terms(self.index2word, encodings, self.UNK_WORD, do_padding=do_padding, do_normalize=do_normalize)

    def word_shuffle(self, words, BOS_IX, EOS_IX):
        """
        Randomly shuffle the words in the sentences
        """
        if self.shuffle_param == 0 :
            return words

        # do not shuffle <BOS> and <EOS>
        _words = words[BOS_IX+1:EOS_IX]
        lengths = len(_words)
        word_idx = np.arange(lengths)

        noise = np.random.uniform(0, self.shuffle_param, size=(lengths))
        scores = word_idx + noise
        scores += 1e-6 * word_idx # ensure no reordering inside a word
        permutation = scores.argsort()

        shuffled_words = _words[permutation]
        shuffled_words = np.concatenate([[words[BOS_IX]], shuffled_words, [words[EOS_IX]]])
        if EOS_IX != len(words):
            # add <PAD> to the sentences
            shuffled_words = np.concatenate([shuffled_words, words[EOS_IX+1:]])

        return shuffled_words
    
    def word_dropout(self, words, BOS_IX, EOS_IX):
        """
        Randomly drop some words from the sentences
        """
        if self.dropout_param == 0:
            return words
        
        assert 0 < self.dropout_param < 1

        keep = np.random.rand(len(words)) >= self.dropout_param
        
        # do not drop <BOS> and <EOS>
        keep[BOS_IX] = True
        keep[EOS_IX] = True

        dropped_words = list(words[keep])
        
        # we need to append <PAD> to the sentences
        padding_length = len(words) - keep.sum()
        dropped_words += [self.get_pad_ix()] * padding_length

        return dropped_words
    
    def word_masking(self, words, BOS_IX, EOS_IX):
        """
        Randomly mask words in the sentences with <MSK>
        """
        if self.masking_param == 0:
            return words
        
        assert 0 < self.masking_param < 1

        mask_word = self.word2index[self.MSK_WORD]

        keep = np.random.rand(len(words)) >= self.masking_param

        # do not mask <BOS> and <EOS>
        keep[BOS_IX] = True
        keep[EOS_IX] = True

        masked_words = [x if keep[i] else mask_word for i, x in enumerate(words)]

        return masked_words
    
    def inject_noise(self, words, device):
        """
        Add noise to the inputs
        """
        words = np.array(words)

        # do not shuffle/drop/mask <BOS> and <EOS>
        BOS_IX = np.where(words == self.get_bos_ix())[0][0]
        EOS_IX = np.where(words == self.get_eos_ix())[0][0]

        words = self.word_shuffle(words, BOS_IX, EOS_IX)
        words = self.word_dropout(words, BOS_IX, EOS_IX)
        words = self.word_masking(words, BOS_IX, EOS_IX)

        return torch.tensor(words).to(device)

    def get_bos_ix(self):
        return self.word2index[self.BOS_WORD]
    
    def get_eos_ix(self):
        return self.word2index[self.EOS_WORD]
    
    def get_pad_ix(self):
        return self.word2index[self.PAD_WORD]

    def get_unk_ix(self):
        return self.word2index[self.UNK_WORD]
    
    def get_msk_ix(self):
        return self.word2index[self.MSK_WORD]

    def forward(self, inputs, noise=False):
        if noise:
            for i, input in enumerate(inputs):
                inputs[i] = self.inject_noise(input.cpu(), inputs.device)

        embeded = self.embedding(inputs)
        return embeded


def normalize(word):
    return '<number>' if word.isnumeric() else word
