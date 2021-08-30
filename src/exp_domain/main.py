import os
import re
import json
import torch
import random
import argparse

import numpy as np
import src.commons.globals as glb
import src.exp_domain.experiment_domain as exp_domain

from types import SimpleNamespace as Namespace


class Arguments(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='configs/exp1/exp1.1.json', help='The path to the JSON experiment config file')
        parser.add_argument('--mode', choices=['train', 'eval', 'generate', 'debug'], default='train')
        parser.add_argument('--replicable', action='store_true', help='If provided, a seed will be used to allow replicability')

        args = parser.parse_args()

        # Fields expected from the command line
        self.config = os.path.join(glb.PROJ_DIR, args.config)
        self.mode = args.mode
        self.replicable = args.replicable

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpu = torch.cuda.device_count()

        assert os.path.exists(self.config) and self.config.endswith('.json'), 'The config path provided does not exist or is not a JSON file'

        # Read the parameters from the JSON file and skip comments
        with open(self.config, 'r') as f:
            params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])

        arguments = json.loads(params, object_hook=lambda d: Namespace(**d))

        # Must-have fields expected from the JSON config file
        self.experiment_id = arguments.experiment_id
        self.experiment_type = arguments.experiment_type
        self.description = arguments.description
        self.datasets = arguments.datasets
        self.model = arguments.model
        self.training = arguments.training
        self.evaluation = arguments.evaluation

        # Checking that the JSON contains at least the fixed fields
        assert all([hasattr(self.model, name) for name in {'name'}])
        assert all([hasattr(self.training, name) for name in {'epochs', 'optimizer', 'clip_grad'}])
        assert any([hasattr(self.training, name) for name in {'batch_size', 'per_gpu_train_batch_size'}])
        assert any([hasattr(self.evaluation, name) for name in {'batch_size', 'per_gpu_eval_batch_size'}])

        self._format_datapaths()
        self._add_extra_fields()

    def _format_datapaths(self):
        assert all([hasattr(self.datasets.source, name) for name in {'train', 'dev', 'test'}])
        assert all([hasattr(self.datasets.target, name) for name in {'train', 'dev', 'test'}])

        for split, path in vars(self.datasets.source).items():
            if not os.path.isabs(path):
                self.datasets.source.__dict__[split] = os.path.join(glb.DATA_DIR, path)

        for split, path in vars(self.datasets.target).items():
            if not os.path.isabs(path):
                self.datasets.target.__dict__[split] = os.path.join(glb.DATA_DIR, path)

    def _add_extra_fields(self):
        self.checkpoints = os.path.join(glb.CHECKPOINT_DIR, self.experiment_id)
        self.figures = os.path.join(glb.FIGURE_DIR, self.experiment_id)
        self.history = os.path.join(glb.HISTORY_DIR, self.experiment_id)
        self.predictions = os.path.join(glb.PREDICTIONS_DIR, self.experiment_id)


def main():
    args = Arguments()

    if args.replicable:
        seed_num = 1111
        random.seed(seed_num)
        np.random.seed(seed_num)
        torch.manual_seed(seed_num)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_num)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

    print("[LOG] {}".format('=' * 40))
    print("[LOG] {: >15}: '{}'".format("Experiment ID", args.experiment_id))
    print("[LOG] {: >15}: '{}'".format("Description", args.description))

    print("[LOG] {: >15}:".format("Datasets"))
    for domain, splits in vars(args.datasets).items():
        for split, datapath in vars(splits).items():
            print("[LOG] {: >15}: {}".format(domain.title() + ' ' + split, datapath))

    print("[LOG] {: >15}:".format("Modeling"))
    for key, val in vars(args.model).items():
        print("[LOG] {: >15}: {}".format(key, val))

    print("[LOG] {: >15}: {}".format("Training", args.training))
    print("[LOG] {: >15}: {}".format("Evaluation", args.evaluation))
    print("[LOG] {: >15}: {}".format("Device", args.device))
    print("[LOG] {}".format('=' * 40))

    exp_domain.main(args)


if __name__ == '__main__':
    main()
