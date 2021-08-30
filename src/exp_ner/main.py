import os
import re
import json
import torch
import argparse

import src.commons.globals as glb
import src.commons.utilities as utils
import src.exp_ner.experiment_ner as exp_ner


class Arguments(dict):
    def __init__(self, *args, **kwargs):
        super(Arguments, self).__init__(*args, **kwargs)
        self.__dict__ = self

    @staticmethod
    def from_nested_dict(data):
        if not isinstance(data, dict):
              return data
        else: return Arguments({key: Arguments.from_nested_dict(data[key]) for key in data})


def load_args(default_config=None, verbose=False):
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--config', default=default_config, type=str, required=default_config is None, help='Provide the JSON config file with the experiment parameters')

    if default_config is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args("")

    # Override the default values with the JSON arguments
    with open(os.path.join(glb.PROJ_DIR, args.config)) as f:
        params = ''.join([re.sub(r"//.*$", "", line, flags=re.M) for line in f])  # Remove comments from the JSON config
        args = Arguments.from_nested_dict(json.loads(params))

    # Data Args
    args.data.directory = os.path.join(glb.PROJ_DIR, args.data.directory)

    # Exp Args
    args.experiment.output_dir = os.path.join(glb.PROJ_DIR, "results", args.experiment.id)
    args.experiment.checkpoint_dir = os.path.join(args.experiment.output_dir, "checkpoint")

    # Optim Args
    args.optim.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.optim.n_gpu = torch.cuda.device_count()

    if verbose:
        for main_field in ['experiment', 'data', 'preproc', 'model', 'optim']:
            assert hasattr(args, main_field)
            print(f"{main_field.title()} Args:")
            for k,v in args[main_field].items():
                print(f"\t{k}: {v}")
            print()
            
    return args


def main():
    args = load_args()

    print("[LOG] {}".format('=' * 40))
    print("[LOG] {: >15}: {}".format("Experiment ID", args.experiment.id))
    print("[LOG] {: >15}: {}".format("GPUs avaliable", args.optim.n_gpu))

    utils.set_seed(args.experiment.seed)

    exp_ner.main(args)


if __name__ == '__main__':
    main()
