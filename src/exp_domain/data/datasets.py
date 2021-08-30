import os
import copy
import torch
import src.commons.utilities as utils

from collections import Counter
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler


class DomainDataset(Dataset):
    def __init__(self, domain_path):

        assert os.path.exists(domain_path), domain_path

        self.domain_path = domain_path
        self._init_data_fields()

    def merge(self, dataset):
        self.data += dataset.data

    def encode(self, word_to_index):
        self.data = _map_terms(self.data, word_to_index)

    def _init_data_fields(self):
        self.data = utils.load_from_json(self.domain_path, json_by_line=True)

        self.tokens = []

        for i in range(len(self.data)):
            tokens = self.data[i]['tokens']
            
            self.data[i] = tokens
            self.tokens += tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
    def collate_fn(self, batch, pad_id=0):
        sentences = batch

        # Padded variables
        p_sentences = []

        # How much padding do we need?
        max_seq_length = max(map(len, sentences)) + 1

        for i in range(len(sentences)):
            padding_length = max_seq_length - len(sentences[i])
            p_sentences.append(sentences[i] + [pad_id] * padding_length)
        
        batch = torch.tensor(p_sentences, dtype=torch.long)

        return batch


def get_dataloader(dataset, batch_size, shuffle=False):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    dloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate_fn)
    return dloader


def create_dataloaders(datasets, train_batch_size, eval_batch_size, args):
    dataloaders = dict()
    for dataset in datasets.keys():
        if dataset == 'train':
            dataloaders[dataset] = get_dataloader(datasets[dataset], batch_size=train_batch_size * args.n_gpu, shuffle=True)
        else:
            dataloaders[dataset] = get_dataloader(datasets[dataset], batch_size=eval_batch_size * args.n_gpu, shuffle=False)
    return dataloaders


def create_domain_dataset(domain_paths, merge_dev=False, verbose=None):
    datasets = dict()
    for dataset, datapath in vars(domain_paths).items():
        datasets[dataset] = DomainDataset(datapath)

    if merge_dev and 'test' in datasets:
        datasets['train'].merge(copy.deepcopy(datasets['dev']))
        datasets['dev'] = copy.deepcopy(datasets['test'])
        del datasets['test']

    if verbose:
        print("[LOG] {}:".format(verbose), end=' ')
        for key in datasets.keys():
            print("{:}:\t{:,}".format(key, len(datasets[key])), end=' ')
        print()

    return datasets


def build_vocab(dataset, min_freq=1):
    all_tokens = []
    sentences = dataset.data
    for sentence in sentences:
        for token in sentence:
            if token.isnumeric():
                token = '<number>'
            all_tokens.append(token)

    tok_counter = Counter(all_tokens)
    vocab = [token for token in all_tokens if tok_counter.get(token) > min_freq]
    vocab = Counter(vocab).most_common()
    return vocab


def encode_sentences(embedder, datasets):
    for dataset in datasets:
        datasets[dataset].data = embedder.encode(datasets[dataset].data)
    return datasets


def _map_terms(terms, mapper):
    for i in range(len(terms)):
        for j in range(len(terms[i])):
            terms[i][j] = mapper[terms[i][j]]
    return terms

