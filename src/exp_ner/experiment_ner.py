import os
import re
import argparse
import json
import torch
import src.exp_ner.data.datasets as ds
import src.exp_ner.modeling.nets as nets
import src.commons.utilities as utils
import src.commons.globals as glb

from transformers import BertTokenizer, BertConfig
from src.exp_ner.modeling.trainer import *


def get_model_class(model_name):
    if model_name == 'ner':
        model_class = nets.NERModel

    elif model_name == 'ner_devlin':
        model_class = nets.NERDevlinModel

    else:
        raise NotImplementedError(f'Unknown model name: {model_name}')

    return model_class


def prepare_model_config(args):
    config = BertConfig.from_pretrained(args.model.pretrained)
    config.pretrained_frozen = args.model.pretrained_frozen
    config.model_name_or_path = args.model.pretrained
    config.num_labels = len(args.data.label_scheme)
    config.output_hidden_states = False
    config.output_attentions = False

    if args.model.name == 'ner_devlin':
        config.output_attentions = False
        config.lstm_dropout = args.model.lstm_dropout
        config.lstm_layers = args.model.lstm_layers
        config.lstm_bidirectional = args.model.lstm_bidirectional
        config.use_lstm = args.model.use_lstm

    elif args.model.name == 'ner':
        pass

    else:
        raise NotImplementedError(f"Unexpected model name: {args.model.name}")

    return config


def main(args):
    tokenizer = BertTokenizer.from_pretrained(args.model.pretrained, do_lower_case=args.preproc.do_lowercase)

    print("[LOG] {: >15}: {}".format("Vocab size", len(tokenizer)))
    if len(args.preproc.new_tokens) > 0:
        tokenizer.add_tokens(args.preproc.new_tokens)
        print("[LOG] {: >15}: {}".format("Adding new tokens. New vocab size", len(tokenizer)))

    dataloaders = ds.get_dataloaders(args, tokenizer)

    print()
    print(f"[LOG] Reading dataset from '{args.data.directory.replace(glb.PROJ_DIR, '$PROJECT')}'")
    for split in dataloaders:
        print("[LOG] {: >15}: {:,}".format(split.upper() + ' data size', len(dataloaders[split].dataset)))
    
    print("[LOG] {}".format('=' * 40))
    print()

    config = prepare_model_config(args)
    model = get_model_class(args.model.name)(config)
    model.to(args.optim.device)

    if len(args.preproc.new_tokens) > 0:
        model.resize_token_embeddings(len(tokenizer))
        assert len(tokenizer) == model.config.vocab_size
        assert len(tokenizer) == config.vocab_size

    if args.experiment.do_training:
        confirm = 'y'
        if os.path.exists(args.experiment.checkpoint_dir):
            confirm = input('A checkpoint was detected. Do you really want to train again and override the model? [y/n]: ').strip()

        if confirm != 'y':
            print("Skipping training")
            stats = {
                'train': torch.load(os.path.join(args.experiment.output_dir, 'train_preds_across_epochs.bin')),
                'dev': torch.load(os.path.join(args.experiment.output_dir, 'dev_preds_across_epochs.bin'))
            }
            print_stats(stats, args.data.label_scheme)
        else:
            # with torch.autograd.detect_anomaly():
            stats, f1, global_step = train(args, model, dataloaders)
            print_stats(stats, args.data.label_scheme)

            print(f"\nBest dev F1: {f1:.5f}")
            print(f"Best global step: {global_step}")

    # Load the best checkpoint according to dev
    if os.path.exists(args.experiment.checkpoint_dir):
        print(f"Loading model from pretrained checkpoint at {args.experiment.checkpoint_dir}")
        model = get_model_class(args.model.name).from_pretrained(args.experiment.checkpoint_dir)
        model.to(args.optim.device)

    elif not args.experiment.do_training:
        print("[WARNING] The model is not fine-tuned; 'do_training' was not provided and there is no checkpoint either...")

    if utils.input_with_timeout("Do you want to evaluate the model? [y/n]:", 15, "y").strip() == 'y':
        # Perform evaluation over the dev and test sets with the best checkpoint
        for split in dataloaders.keys():
            if split == 'train':
                continue

            stats = predict(args, model, dataloaders[split])
            # torch.save(stats, os.path.join(args.experiment.output_dir, f'{split}_best_preds.bin'))

            f1, prec, recall = stats.metrics(args.data.label_scheme)
            loss, _, _ = stats.loss()
            ner_loss, _, _ = stats.loss(loss_type='ner')
            lm_loss, _, _ = stats.loss(loss_type='lm')

            report = stats.get_classification_report(args.data.label_scheme)
            classes = sorted(set([label[2:] for label in args.data.label_scheme if label != 'O']))

            print(f"\n********** {split.upper()} RESULTS **********\n")
            print('\t'.join(["LLoss", "NLoss", "Loss"] + ["Prec", "Recall", "F1"]), end='\n')
            print('\t'.join([f"{l:.4f}" for l in [lm_loss, ner_loss, loss, prec * 100, recall * 100, f1 * 100]]), end='\t')

            # f1_scores = []
            # for c in classes + ["micro avg"]:
            #     if 'f1-score' in report[c].keys():
            #         f1_scores.append(report[c]['f1-score'])
            #     else:
            #         f1_scores.append(0)
            # print('\t'.join([f"{score * 100:.3f}" for score in f1_scores]))
            # print('\t'.join([f"{report[c]['f1-score'] * 100:.3f}" for c in classes + ["micro avg"]]))
            print()

            if utils.input_with_timeout("Print class-level results? [y/n]:", 5, "n").strip() == 'y':
                stats.print_classification_report(report=report)
        print()

if __name__ == '__main__':
    main()

