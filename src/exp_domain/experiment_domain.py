import os
import torch.optim as optim
import torch.autograd as autograd

from src.exp_domain.data.datasets import create_domain_dataset, create_dataloaders, encode_sentences, build_vocab
from src.exp_domain.modeling.nets.embedding import TokenEmbedder
from src.exp_domain.modeling.nets.encoders import LSTMEncoder
from src.exp_domain.modeling.nets.decoders import LSTMDecoder, AttentionDecoder
from src.exp_domain.modeling.nets.discriminators import LSTMDiscriminator, MLPDiscriminator
from src.exp_domain.modeling.nets.models import ARModel, AdversarialARModel, CrossDomainARModel
from src.exp_domain.modeling.trainer import train, evaluate, generate, load_checkpoint


def choose_embedder(emb_args, src_vocab, tgt_vocab):
    if emb_args.name == 'token_embedder':
        shuffle_param = emb_args.noise.shuffle_param
        dropout_param = emb_args.noise.dropout_param
        masking_param = emb_args.noise.masking_param

        src_embedder = TokenEmbedder(src_vocab, emb_args.src_vocab_size, emb_args.embedding_dim, shuffle_param, dropout_param, masking_param)
        tgt_embedder = TokenEmbedder(tgt_vocab, emb_args.tgt_vocab_size, emb_args.embedding_dim, shuffle_param, dropout_param, masking_param)

        src_vocab_size = src_embedder.num_embeddings
        tgt_vocab_size = tgt_embedder.num_embeddings

        assert src_embedder.embedding_dim == tgt_embedder.embedding_dim, "Embedding tables must be the same dimension"

        return (src_embedder, tgt_embedder), src_embedder.embedding_dim, (src_vocab_size, tgt_vocab_size)
    
    else:
        raise NotImplementedError('Unknown encoder: {}'.format(emb_args.name))


def choose_encoder(enc_args, src_embedder, tgt_embedder, input_dim):
    if enc_args.name == 'lstm':
        encoder = LSTMEncoder(src_embedder=src_embedder,
                            tgt_embedder=tgt_embedder,
                            input_dim=input_dim,
                            hidden_dim=enc_args.hidden_dim,
                            latent_dim=enc_args.latent_dim,
                            bidirectional=enc_args.bidirectional,
                            num_layers=enc_args.num_layers,
                            dropout=enc_args.dropout)

    else:
        raise NotImplementedError('Unknown encoder: {}'.format(enc_args.name))

    return encoder


def choose_decoder(dec_args, src_embedder, tgt_embedder, input_dim, src_vocab_size, tgt_vocab_size):
    if dec_args.name == 'lstm':
        decoder = LSTMDecoder(src_embedder=src_embedder,
                            tgt_embedder=tgt_embedder,
                            input_dim=input_dim,
                            hidden_dim=dec_args.hidden_dim,
                            latent_dim=dec_args.latent_dim,
                            bidirectional=dec_args.bidirectional,
                            num_layers=dec_args.num_layers,
                            src_vocab_size=src_vocab_size,
                            tgt_vocab_size=tgt_vocab_size,
                            dropout=dec_args.dropout)

    elif dec_args.name == 'attention':
        decoder = AttentionDecoder(src_embedder=src_embedder,
                            tgt_embedder=tgt_embedder,
                            input_dim=input_dim,
                            hidden_dim=dec_args.hidden_dim,
                            src_vocab_size=src_vocab_size,
                            tgt_vocab_size=tgt_vocab_size,
                            dropout=dec_args.dropout)

    else:
        raise NotImplementedError('Unknown decoder: {}'.format(dec_args.name))

    return decoder


def choose_discriminator(disc_args):
    if disc_args.name == 'lstm':
        discriminator = LSTMDiscriminator(input_dim=disc_args.input_dim,
                                 hidden_dim=disc_args.hidden_dim,
                                 num_layers=disc_args.num_layers,
                                 drop_prob=disc_args.dropout,
                                 smooth_labels=disc_args.smooth_labels)

    elif disc_args.name == 'mlp':
        discriminator = MLPDiscriminator(input_dim=disc_args.input_dim,
                                hidden_dim=disc_args.hidden_dim,
                                num_layers=disc_args.num_layers,
                                drop_prob=disc_args.dropout,
                                smooth_labels=disc_args.smooth_labels)

    else:
        raise NotImplementedError('Unknown discriminator: {}'.format(disc_args.name))

    return discriminator


def choose_model(model_args, src_vocab, tgt_vocab):
    (src_embedder, tgt_embedder), embedding_dim, (src_vocab_size, tgt_vocab_size) = choose_embedder(model_args.embedder, src_vocab, tgt_vocab)

    encoder = choose_encoder(model_args.encoder, src_embedder, tgt_embedder, embedding_dim)
    decoder = choose_decoder(model_args.decoder, src_embedder, tgt_embedder, embedding_dim, src_vocab_size, tgt_vocab_size)

    discriminator = choose_discriminator(model_args.discriminator)

    lambda_auto = model_args.lambda_coef.lambda_auto
    lambda_adv = model_args.lambda_coef.lambda_adv
    lambda_cross = model_args.lambda_coef.lambda_cross

    if model_args.name == 'ARModel':
        model = ARModel(encoder, decoder, discriminator, lambda_auto, lambda_adv, lambda_cross)

    elif model_args.name == 'AdversarialARModel':
        model = AdversarialARModel(encoder, decoder, discriminator, lambda_auto, lambda_adv, lambda_cross)

    elif model_args.name == 'CrossDomainARModel':
        model = CrossDomainARModel(encoder, decoder, discriminator, lambda_auto, lambda_adv, lambda_cross)
        
    else:
        raise NotImplementedError('Unknown model: {}'.format(model_args.name))

    return model


def choose_optimizer(model_params, optim_args):
    params = filter(lambda p: p.requires_grad, model_params)

    if optim_args.name == 'sgd':
        if hasattr(optim_args, 'momentum'):
            optimizer = optim.SGD(params, lr=optim_args.lr, momentum=optim_args.momentum)
        else:
            optimizer = optim.SGD(params, lr=optim_args.lr)

    elif optim_args.name == 'adam':
        if hasattr(optim_args, 'beta_1') and hasattr(optim_args, 'beta_2'):
            optimizer = optim.Adam(params, lr=optim_args.lr, betas=(optim_args.beta_1, optim_args.beta_2))
        else:
            optimizer = optim.Adam(params, lr=optim_args.lr)

    elif optim_args.name == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=optim_args.lr)

    else:
        raise NotImplementedError('Optimizer not implemented: {}'.format(optim_args.name))
        
    return optimizer


def main(args):
    src_datasets = create_domain_dataset(args.datasets.source, merge_dev=False, verbose='Source domain')
    tgt_datasets = create_domain_dataset(args.datasets.target, merge_dev=False, verbose='Target domain')
    print("[LOG] {}".format('=' * 40))

    src_vocab = build_vocab(src_datasets['train'])
    tgt_vocab = build_vocab(tgt_datasets['train'])

    model = choose_model(args.model, src_vocab, tgt_vocab)

    src_datasets = encode_sentences(model.encoder.src_embedder, src_datasets)
    tgt_datasets = encode_sentences(model.encoder.tgt_embedder, tgt_datasets)

    src_dataloaders = create_dataloaders(src_datasets, args.training.per_gpu_train_batch_size, args.evaluation.per_gpu_eval_batch_size, args)
    tgt_dataloaders = create_dataloaders(tgt_datasets, args.training.per_gpu_train_batch_size, args.evaluation.per_gpu_eval_batch_size, args)

    print("[LOG] {}".format(model))
    print("[LOG] {}".format('=' * 40))

    d_optimizer = choose_optimizer(model.dis_params(), args.training.optimizer.discriminator)
    g_optimizer = choose_optimizer(model.gen_params(), args.training.optimizer.generator)

    # ===============================================================
    checkpoint_path = os.path.join(args.checkpoints, 'model.pt')
    if os.path.exists(checkpoint_path):
        option = input("[LOG] Found a checkpoint! Choose an option:\n"
                       "\t0) Ignore the checkpoint (NOTE: training will ovewrite the checkpoint)\n"
                       "\t1) Load the checkpoint and train from there\nYour choice: ").strip()
        assert option in {"0", "1"}, "Unexpected choice"

        if option == "1":
            model = load_checkpoint(model, checkpoint_path)
    # ===============================================================

    if args.mode == 'train':
        model = train(model, src_dataloaders, tgt_dataloaders, g_optimizer, d_optimizer, args)
    
    if args.mode == 'eval':
        evaluate(model, src_dataloaders, 'src', args)
        evaluate(model, tgt_dataloaders, 'tgt', args)

    elif args.mode == 'generate':
        generate(model, src_dataloaders, src_domain='src', tgt_domain='src', max_len=50, temperature=1, algorithm='greedy', args=args)
        generate(model, src_dataloaders, src_domain='src', tgt_domain='tgt', max_len=50, temperature=1, algorithm='greedy', args=args)
        generate(model, tgt_dataloaders, src_domain='tgt', tgt_domain='tgt', max_len=50, temperature=1, algorithm='greedy', args=args)
        generate(model, tgt_dataloaders, src_domain='tgt', tgt_domain='src', max_len=50, temperature=1, algorithm='greedy', args=args)

        generate(model, src_dataloaders, src_domain='src', tgt_domain='src', max_len=50, temperature=1, algorithm='top5', args=args)
        generate(model, src_dataloaders, src_domain='src', tgt_domain='tgt', max_len=50, temperature=1, algorithm='top5', args=args)
        generate(model, tgt_dataloaders, src_domain='tgt', tgt_domain='tgt', max_len=50, temperature=1, algorithm='top5', args=args)
        generate(model, tgt_dataloaders, src_domain='tgt', tgt_domain='src', max_len=50, temperature=1, algorithm='top5', args=args)

    elif args.mode == 'debug':
        with autograd.detect_anomaly():
            model = train(model, src_dataloaders, tgt_dataloaders, g_optimizer, d_optimizer, args)

