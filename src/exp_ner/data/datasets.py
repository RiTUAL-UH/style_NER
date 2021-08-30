import os
import torch
import src.commons.utilities as utils

from typing import List, Dict
from collections import Counter
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class NERDatasetBase(Dataset):
    def __init__(self,
                 dataset_file,
                 dataset_cols,
                 label_scheme: List[str],
                 tokenizer: BertTokenizer,
                 partition: str):

        self.tokenizer = tokenizer
        self.index_map = dict(enumerate(label_scheme))
        self.label_map = {l: i for i, l in self.index_map.items()}

        self.dataset_file = dataset_file
        self.dataset_cols = dataset_cols

        self.partition = partition

    def _init_data_fields(self, dataset=None):
        if dataset is None:
            dataset = utils.read_conll(self.dataset_file, columns=self.dataset_cols)

        self.tokens = dataset['tokens']
        self.labels = dataset['labels']

    def _prepare_encoding_fields_from_start(self, max_length=512):
        """
        Only call this method if you want everything tokenized and encoded from the very beginning.
        This is not the case when there is dynamic masking.
        """
        self.tokenized = []
        self.input_ids = []
        self.label_ids = []
        self.label_mask = []

        dataset = utils.read_conll(self.dataset_file, columns=self.dataset_cols)

        if self.partition == 'train':
            # dataset['tokens'], dataset['labels'] = shuffle(dataset['tokens'], dataset['labels'])

            num_samples = 499

            dataset['tokens'] = dataset['tokens'][:num_samples+15000]
            dataset['labels'] = dataset['labels'][:num_samples+15000]
            

        for i in range(len(dataset['tokens'])):
            tokens = dataset['tokens'][i]
            labels = dataset['labels'][i]

            tokenized, input_ids, label_ids, label_mask = process_sample(self.tokenizer, tokens, labels, self.label_map)

            if len(tokenized) >= 512:
                continue

            self.tokenized.append(tokenized)
            self.input_ids.append(input_ids)
            self.label_ids.append(label_ids)
            self.label_mask.append(label_mask)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def collate(self, batch, pad_tok=0):
        raise NotImplementedError()


class NERDataset(NERDatasetBase):
    """
    This class encodes the data from the beginning.
    """
    def __init__(self,
                 dataset_file,
                 dataset_cols,
                 label_scheme: List[str],
                 tokenizer: BertTokenizer,
                 partition: str):

        super().__init__(dataset_file, dataset_cols, label_scheme, tokenizer, partition)

        # Always encodes the data from the beginning, regardless the partition
        self._prepare_encoding_fields_from_start()

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        label_ids = self.label_ids[index]
        label_msk = self.label_mask[index]
        
        return input_ids, label_ids, label_msk

    def collate(self, batch, pad_tok=0):
        # Unwrap the batch into every field
        input_ids, label_ids, label_mask = map(list, zip(*batch))

        # Padded variables
        p_input_ids, p_input_mask, p_token_type, p_label_ids, p_label_mask = [], [], [], [], []

        # How much padding do we need?
        max_seq_length = max(map(len, input_ids))

        for i in range(len(input_ids)):
            padding_length = max_seq_length - len(input_ids[i])

            p_input_ids.append(input_ids[i] + [pad_tok] * padding_length)
            p_input_mask.append([1] * len(input_ids[i]) + [pad_tok] * padding_length)
            p_token_type.append([0] * len(input_ids[i]) + [pad_tok] * padding_length)
            p_label_ids.append(label_ids[i] + [pad_tok] * padding_length)
            p_label_mask.append(label_mask[i] + [pad_tok] * padding_length)

        input_dict = {
            'input_ids': torch.tensor(p_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(p_input_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(p_token_type, dtype=torch.long),
            'label_mask': torch.tensor(p_label_mask, dtype=torch.long),
            'labels': torch.tensor(p_label_ids, dtype=torch.long)
        }

        return input_dict


def process_sample(tokenizer, tokens, labels, label_map):
    tokenized = []
    label_ids = []
    label_msk = []

    for i, (token, label) in enumerate(zip(tokens, labels)):
        word_tokens = tokenizer.tokenize(token)
        if len(word_tokens) == 0:
            word_tokens = [tokenizer.unk_token]
        num_subtoks = len(word_tokens) - 1

        tokenized.extend(word_tokens)
        label_ids.extend([label_map[label]] + [0] * num_subtoks)
        label_msk.extend([1] + [0] * num_subtoks)

    tokenized = [tokenizer.cls_token] + tokenized + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(tokenized)
    label_ids = [0] + label_ids + [0]
    label_msk = [0] + label_msk + [0]

    return tokenized, input_ids, label_ids, label_msk


def get_overall_entity_frequency(tokens, labels):
    entities = []
    for i in range(len(tokens)):
        tokens_i = tokens[i]
        labels_i = labels[i]

        for j in range(len(tokens_i)):
            if labels_i[j] == 'O':
                continue
            entities.append(tokens_i[j].lower())

    return Counter(entities)


def get_dataloaders(args, tokenizer):
    dargs = args.data
    pargs = args.preproc
    oargs = args.optim

    corpus = dict()
    for split in dargs.partitions:
        if split == 'train' or split == 'dev':
            splits, fnames = [split], [dargs.partitions[split]]
        else:
            fnames = dargs.partitions[split]
            splits = [os.path.splitext(fname)[0] for fname in fnames]

        for split, fname in zip(splits, fnames):
            fpath = os.path.join(dargs.directory, fname)

            if pargs.dataset_class == 'ner':
                dataset = NERDataset(
                    fpath, dargs.colnames, dargs.label_scheme, tokenizer, split
                )
            else:
                raise NotImplementedError("Unexpected dataset class")

            if split == 'train':
                oargs.train_batch_size = oargs.per_gpu_train_batch_size * max(1, oargs.n_gpu)
                batch_size = oargs.train_batch_size
                sampler = RandomSampler(dataset)
            else:
                oargs.eval_batch_size = oargs.per_gpu_eval_batch_size * max(1, oargs.n_gpu)
                batch_size = oargs.eval_batch_size
                sampler = SequentialSampler(dataset)
            corpus[split] = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=dataset.collate)

    return corpus

