import os
import src.commons.utilities as utils
import src.commons.globals as glb


def ner2domain(corpus_dir, save_dir, colnames, datasets=['train.txt', 'dev.txt', 'test.txt']):
    """
    corpus_dir: str
    save_dir: str
    colnames: List[str]
    datasets: List[str]
    """

    corpus_dir = os.path.join(glb.DATA_DIR, corpus_dir)
    save_dir = os.path.join(glb.DATA_DIR, save_dir)

    for dataset in datasets:
        data_path = os.path.join(corpus_dir, dataset)
        save_path = os.path.join(save_dir, dataset)
        save_domain(data_path, save_path, colnames)

    trn_data = utils.load_from_json(os.path.join(save_dir, 'train.json'), json_by_line=True)
    dev_data = utils.load_from_json(os.path.join(save_dir, 'dev.json'), json_by_line=True)
    tst_data = utils.load_from_json(os.path.join(save_dir, 'test.json'), json_by_line=True)
    print(len(trn_data))
    print(len(dev_data))
    print(len(tst_data))


def domain2ner(corpus_dir, save_dir, datasets=['train.json', 'dev.json', 'test.json']):
    """
    corpus_dir: str
    save_dir: str
    datasets: List[str]
    """

    corpus_dir = os.path.join(glb.DATA_DIR, corpus_dir)
    save_dir = os.path.join(glb.DATA_DIR, save_dir)

    for dataset in datasets:
        data_path = os.path.join(corpus_dir, dataset)
        save_path = os.path.join(save_dir, dataset.replace('.json', '.txt'))

        data = utils.load_from_json(data_path, json_by_line=True)

        save_data = {'tokens': [], 'labels': []}

        for i in range(len(data)):
            sentence = data[i]['tokens']
            tokens, labels = unlinearize_sentence(sentence)
            save_data['tokens'].append(tokens)
            save_data['labels'].append(labels)
        
        utils.write_conll(save_path, save_data)
    
    return


def linearize_sentence(tokens, labels, linearize=True):
    sentence = []
    for i in range(len(tokens)):
        if linearize and labels[i] != 'O':
            sentence.append(labels[i])

        sentence.append(tokens[i])

    return sentence


def unlinearize_sentence(sentence):
    tokens, labels = [], []

    is_entity = False
    for token in sentence:
        if token.startswith('B-') or token.startswith('I-'):
            labels.append(token)
            is_entity = True
        else:
            tokens.append(token)
            if not is_entity:
                labels.append('O')
            else:
                is_entity = False
    
    assert len(tokens) == len(labels)

    return tokens, labels


def save_domain(data_path, save_path, colnames, sentence_linearization=True, json_format=True):
    """
    data_path: str
    save_path: str
    colnames: List[str]
    """
    
    data = utils.read_conll(data_path, colnames)

    zipped = list(zip(*data.values()))

    save_data = []
    for i in range(len(zipped)):
        if sentence_linearization:
            tokens, labels = zipped[i]
            lineazed_tokens = linearize_sentence(tokens, labels)
            tmp_data = {'tokens': lineazed_tokens}
        else:
            tmp_data = {'tokens': zipped[i][0]}
        save_data.append(tmp_data)
    
    if json_format:
        save_path = save_path.replace('.txt', '.json')

    utils.save_as_json(save_path, save_data, json_by_line=True)


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


if __name__ == '__main__':
    corpus_dir = 'ner/sm'
    save_dir = 'linearized_domain/sm'
    colnames = {'tokens': 0, 'labels': 1}
    
    ner2domain(corpus_dir, save_dir, colnames)

