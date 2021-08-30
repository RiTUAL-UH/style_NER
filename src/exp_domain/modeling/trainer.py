import os
import gc
import time
import math
import torch
import numpy as np
import src.commons.utilities as utils

from tqdm import tqdm


def train(model, src_dataloaders, tgt_dataloaders, g_optimizer, d_optimizer, args):
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    best_ppl = np.Inf
    for epoch in range(1, args.training.epochs + 1):
        epoch_msg = f'Epoch {epoch:03d}'
        epoch_track = ''

        for dataset in ['train', 'dev']:
            if dataset == 'train':
                model.train()
                model.zero_grad()
            else:
                model.eval()

            epoch_toks = 0
            epoch_loss = {'loss': 0, 'auto_loss': 0, 'cross_loss': 0, 'adv_loss': 0}
            epoch_time = time.time()
            epoch_step = min(len(src_dataloaders[dataset]), len(tgt_dataloaders[dataset]))
            # ========================================================================
            for batch_i, (src_batch, tgt_batch) in enumerate(tqdm(zip(src_dataloaders[dataset], tgt_dataloaders[dataset]), total=epoch_step)):

                src_batch = src_batch.to(args.device)
                tgt_batch = tgt_batch.to(args.device)
                
                result = model(src_batch, tgt_batch, noise=(dataset == 'train'), wrap_scalars=args.n_gpu > 1)

                if args.n_gpu > 1:
                    for loss_key in result:
                        if loss_key in epoch_loss.keys():
                            result[loss_key] = result[loss_key].mean()

                loss = result['loss']

                # L2 regularization
                if args.training.l2 != 0:
                    loss = loss + l2_reg(model, args.training.l2)

                if dataset == 'train':
                    loss.backward()

                    # Clipping the norm ||g|| of gradient g before the optmizer's step
                    if args.training.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.training.clip_grad)

                    d_optimizer.step()
                    g_optimizer.step()

                    d_optimizer.zero_grad()
                    g_optimizer.zero_grad()

                    model.zero_grad()

                batch_toks = result['num_toks'].sum().item()

                epoch_toks += batch_toks
                for loss_key in epoch_loss.keys():
                    epoch_loss[loss_key] += batch_toks * result[loss_key].detach().data.item()
                
                if batch_i % 20 == 0:
                    torch.cuda.empty_cache()
                    _ = gc.collect()

            # ========================================================================
            epoch_time = time.time() - epoch_time
            epoch_msg += ' [{}] time: {:.1f}s'.format(dataset.upper(), epoch_time)

            for loss_key in epoch_loss.keys():
                epoch_loss[loss_key] /= epoch_toks
                epoch_msg += ' {}: {:.3f}'.format(loss_key, epoch_loss[loss_key])
            
            epoch_ppl = compute_ppl(epoch_loss['loss'])
            epoch_msg += ' ppl: {:.3f}'.format(epoch_ppl)

            if dataset == 'dev':
                best_ppl, epoch_track = track_best_model(args.checkpoints, model, epoch, best_ppl, epoch_ppl)
            
        print('[LOG]', epoch_msg + epoch_track)

    print("[LOG] Done training!")

    return model


def evaluate(model, dataloaders, domain, args):
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    for dataset in ['dev', 'test']:
        dataset_info = ''
        dataset_toks = 0
        dataset_loss = {'loss': 0, 'auto_loss': 0, 'cross_loss': 0, 'adv_loss': 0}
        dataset_time = time.time()
        # ========================================================================
        for batch_i, batch in enumerate(tqdm(dataloaders[dataset])):

            batch = batch.to(args.device)

            if domain == 'src':
                result = model(batch, None, wrap_scalars=args.n_gpu > 1)
            else:
                result = model(None, batch, wrap_scalars=args.n_gpu > 1)

            if args.n_gpu > 1:
                for loss_key in result:
                    if loss_key in dataset_loss.keys():
                        result[loss_key] = result[loss_key].mean()

            batch_toks = result['num_toks'].sum().item()

            dataset_toks += batch_toks
            for loss_key in dataset_loss.keys():
                dataset_loss[loss_key] += batch_toks * result[loss_key].detach().data.item()
            
            if batch_i % 20 == 0:
                torch.cuda.empty_cache()
                _ = gc.collect()

        # ========================================================================
        dataset_time = time.time() - dataset_time
        dataset_info += ' [{}] time: {:.1f}s'.format(dataset.upper(), dataset_time)

        for loss_key in dataset_loss.keys():
            dataset_loss[loss_key] /= dataset_toks
            dataset_info += ' {}: {:.3f}'.format(loss_key, dataset_loss[loss_key])
        
        dataset_ppl = compute_ppl(dataset_loss['loss'])
        dataset_info += ' ppl: {:.3f}'.format(dataset_ppl)
        
        print('[{}] {}'.format(domain.upper(), dataset_info))


def generate(model, dataloaders, src_domain, tgt_domain, max_len, temperature, algorithm, args):
    model.to(args.device)
    model.eval()
    
    prediction_path = os.path.join(args.checkpoints, 'predictions')
    
    for dataset in ['train', 'dev', 'test']:
        fake_samples = []

        for _, batch in enumerate(tqdm(dataloaders[dataset])):
            batch = batch.to(args.device)

            encoded = model.encoder(batch, src_domain)
            outputs = model.decoder.generate(encoded['outs'], max_len, temperature, algorithm, tgt_domain, args.device, strip=True)

            if tgt_domain == 'src':
                batch_samples = model.decoder.src_embedder.decode(outputs)
            else:
                batch_samples = model.decoder.tgt_embedder.decode(outputs)

            fake_samples += batch_samples
        
        filepath = os.path.join(prediction_path, src_domain + '_' + dataset + '_' + src_domain + '2' + tgt_domain + '_' + algorithm + '.txt')
        fake_samples = [' '.join(s) for s in fake_samples]
        utils.save_as_txt(filepath, fake_samples)
        print('[{} --> {}] Done! The predictions are saved in {}!'.format(src_domain, tgt_domain, filepath))


def compute_ppl(loss):
    return math.exp(loss)


def l2_reg(model, l2_lambda):
    l2 = 0
    for W in model.parameters():
        l2 = l2 + W.norm(2)
    return l2_lambda * l2


def track_best_model(model_path, model, epoch, best_ppl, dev_ppl):
    if dev_ppl != dev_ppl:
        print("[WARNING] The loss is NaN. The model won't be tracked this epoch")
        return best_ppl, ''

    if best_ppl < dev_ppl:
        return best_ppl, ''

    state = {
        'epoch': epoch,
        'ppl': dev_ppl,
        'model': model.state_dict()
    }

    os.makedirs(model_path, exist_ok=True)
    torch.save(state, os.path.join(model_path, 'model.pt'))

    return dev_ppl, ' * '


def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path)
    state['model'] = {key.replace('module.', ''): value for key, value in state['model'].items()}
    model.load_state_dict(state['model'])
    print('[LOG] Loading model from epoch {} with ppl {:.5f}'.format(state['epoch'], state['ppl']))
    return model

