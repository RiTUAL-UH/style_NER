import os
import json
import src.commons.utilities as utils
import src.commons.globals as glb


def process_json(corpus_dir, save_dir, data_colnames, save_colnames):
    """
    corpus_dir: str
    save_dir: str
    data_colnames: List[str]
    save_colnames: List[str]
    """

    corpus_dir = os.path.join(glb.DATA_DIR, corpus_dir)
    save_dir = os.path.join(glb.DATA_DIR, save_dir)

    datasets=['train.json', 'dev.json', 'test.json']
    
    for dataset in datasets:
        data_path = os.path.join(corpus_dir, dataset)
        save_path = os.path.join(corpus_dir, save_dir + dataset)

        save_data = dict()
        with open(data_path, 'r') as json_file:
            for line in json_file:
                data = json.loads(line)
                for colname in data_colnames:
                    if colname not in save_data.keys():
                        save_data[colname] = [data[colname]]
                    else:
                        save_data[colname].append(data[colname])

        utils.write_conll(save_path, save_data)


def save_ner(data_path, save_path, data_colnames, save_colnames):
    """
    data_path: str
    save_path: str
    data_colnames: List[str]
    save_colnames: List[str]
    """

    dataset = utils.read_conll(data_path, data_colnames)

    save_data = {}
    for colname in save_colnames:
        save_data[colname] = dataset[colname]

    utils.write_conll(save_path, save_data)


def process_ner(corpus_dir, save_dir, data_colnames, save_colnames):
    """
    corpus_dir: str
    save_dir: str
    data_colnames: List[str]
    save_colnames: List[str]
    """

    corpus_dir = os.path.join(glb.DATA_DIR, corpus_dir)
    save_dir = os.path.join(glb.DATA_DIR, save_dir)

    datasets = ['train.txt', 'dev.txt', 'test.txt']
    
    for dataset in datasets:
        data_path = os.path.join(corpus_dir, dataset)
        save_path = os.path.join(corpus_dir, save_dir + dataset)
        save_ner(data_path, save_path, data_colnames, save_colnames)


def process_labels(corpus_dir, save_dir, data_colnames, save_colnames):
    """
    corpus_dir: str
    save_dir: str
    data_colnames: List[str]
    save_colnames: List[str]
    """

    NER_LABELS = {'B-person': 'B-PER', 'I-person': 'I-PER', 'B-geo-loc': 'B-LOC', 'I-geo-loc': 'I-LOC', 'B-company': 'B-ORG', 'I-company': 'I-ORG'}

    corpus_dir = os.path.join(glb.DATA_DIR, corpus_dir)
    save_dir = os.path.join(glb.DATA_DIR, save_dir)

    datasets = ['train.txt', 'dev.txt', 'test.txt']
    
    for dataset in datasets:
        data_path = os.path.join(corpus_dir, dataset)
        save_path = os.path.join(corpus_dir, save_dir + dataset)

        dataset = utils.read_conll(data_path, data_colnames)
        for i in range(len(dataset['labels'])):
            for j in range(len(dataset['labels'][i])):
                if dataset['labels'][i][j] in NER_LABELS.keys():
                    dataset['labels'][i][j] = NER_LABELS[dataset['labels'][i][j]]
                else:
                    dataset['labels'][i][j] = 'O'

        utils.write_conll(save_path, dataset)


if __name__ == '__main__':
    corpus_dir = 'ner/sm'
    save_dir = 'ner/sm/new/'
    data_colnames = {'tokens': 0, 'labels': 1}
    save_colnames = {'tokens': 0, 'labels': 1}
    
    process_labels(corpus_dir, save_dir, data_colnames, save_colnames)
