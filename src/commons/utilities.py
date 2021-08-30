import os
import csv
import json
import torch
import random
import signal
import numpy as np

from itertools import groupby
from typing import List, Dict


def read_conll(filename, columns, delimiter='\t'):
    def is_empty_line(line_pack):
        return all(field.strip() == '' for field in line_pack)

    data = []
    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        groups = groupby(reader, is_empty_line)

        for is_empty, pack in groups:
            if is_empty is False:
                data.append([list(field) for field in zip(*pack)])
    
    data = list(zip(*data))
    dataset = {colname: list(data[columns[colname]]) for colname in columns}

    return dataset


def write_conll(filename, data, colnames: List[str] = None, delimiter='\t'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if colnames is None:
        colnames = list(data.keys())

    any_key = colnames[0]

    with open(filename, 'w') as fp:
        for sample_i in range(len(data[any_key])):
            for token_i in range(len(data[any_key][sample_i])):
                row = [data[col][sample_i][token_i] for col in colnames]
                fp.write(delimiter.join(row) + '\n')
            fp.write('\n')


def read_conll_corpus(corpus_dir, filenames, columns, delimiter='\t'):
    corpus = {}
    for datafile in filenames:
        dataset = os.path.splitext(datafile)[0]
        datafile = os.path.join(corpus_dir, datafile)
        corpus[dataset] = read_conll(datafile, columns, delimiter=delimiter)
    return corpus


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten(nested_elems):
    return [elem for elems in nested_elems for elem in elems]


def input_with_timeout(prompt, timeout, default=''):
    def alarm_handler(signum, frame):
        raise Exception("Time is up!")
    try:
        # set signal handler
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout)  # produce SIGALRM in `timeout` seconds

        return input(prompt)
    except Exception as ex:
        return default
    finally:
        signal.alarm(0)  # cancel alarm


def load_from_json(filepath, json_by_line=False):
    with open(filepath, 'r') as fp:
        if json_by_line:
            data = []
            for line in fp:
                data.append(json.loads(line.strip()))
        else:
            data = json.load(fp)
    return data


def save_as_json(filepath, data, json_by_line=False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as fp:
        if json_by_line:
            for sample in data:
                fp.write(json.dumps(sample) + '\n')
        else:
            json.dump(data, fp)


def save_as_txt(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as fp:
        for sample in data:
            fp.write(sample)
            fp.write('\n')


def count_params(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])

