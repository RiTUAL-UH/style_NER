import torch
import torch.nn as nn


class ARModelBase(nn.Module):

    def __init__(self, encoder, decoder, discriminator, lambda_auto=1.0, lambda_adv=1.0, lambda_cross=1.0):
        super().__init__()

        self.bos_ix = encoder.src_embedder.get_bos_ix()
        self.eos_ix = encoder.src_embedder.get_eos_ix()
        self.pad_ix = encoder.src_embedder.get_pad_ix()

        self.encoder = encoder
        self.decoder = decoder
        
        self.discriminator = discriminator

        self.lambda_auto  = lambda_auto
        self.lambda_adv   = lambda_adv
        self.lambda_cross = lambda_cross
        
        self.xentropy = nn.CrossEntropyLoss(ignore_index=self.pad_ix, reduction='mean')

    def gen_params(self):
        return set(self.encoder.parameters()) | set(self.decoder.parameters())

    def dis_params(self):
        return set(self.discriminator.parameters())
    
    def get_token_num(self, sentences):
        return torch.sum(sentences != self.pad_ix)

    def cross_entropy_loss(self, logits, targets):
        logits_ = logits.reshape(-1, logits.size(-1))
        targets_ = targets.reshape(-1)
        loss = self.xentropy(logits_, targets_)
        return loss

    def auto_reconst(self, x, y, teacher, domain, noise=False):
        encoded = self.encoder(x, domain=domain, noise=noise)
        decoded = self.decoder(x, encoded['outs'], teacher=teacher, domain=domain, teacher_forcing_ratio=0.5)

        loss = self.cross_entropy_loss(decoded['logits'], y)

        return encoded['outs'], loss

    def cross_reconst(self, x, y, teacher, domain, noise=False):
        src_domain = 'src' if domain == 'src' else 'tgt'
        tgt_domain = 'tgt' if domain == 'src' else 'src'

        encoded = self.encoder(x, domain=src_domain, noise=noise)
        decoded = self.decoder(x, encoded['outs'], teacher=teacher, domain=tgt_domain, teacher_forcing_ratio=0.5)

        loss = self.cross_entropy_loss(decoded['logits'], y)

        return encoded['outs'], loss

    def adversarial(self, z, domain=None):
        batch_size = z.size(0)

        if domain == 'tgt':
            targets = torch.zeros(batch_size, 1).float().to(z.device)
        elif domain == 'src':
            targets = torch.ones(batch_size, 1).float().to(z.device)
        else:
            targets = None

        result = self.discriminator(z, targets)

        return result['loss']

    def forward(self, src_sentences, tgt_sentences, noise=False, wrap_scalars=False):
        raise NotImplementedError('The AutoReconstructionModelBase class should never execute forward')


class ARModel(ARModelBase):

    def __init__(self, encoder, decoder, discriminator, lambda_auto=1.0, lambda_adv=1.0, lambda_cross=1.0):
        super().__init__(encoder, decoder, discriminator, lambda_auto, lambda_adv, lambda_cross)

    def forward(self, src_sentences, tgt_sentences, noise=False, wrap_scalars=False):
        device = src_sentences.device if src_sentences is not None else tgt_sentences.device

        auto_loss  = torch.tensor(0., device=device).float()
        adv_loss   = torch.tensor(0., device=device).float()
        cross_loss = torch.tensor(0., device=device).float()
        num_toks   = torch.tensor(0., device=device).float()

        if src_sentences is not None:
            x_src, y_src = src_sentences[:, :-1], src_sentences[:, 1:]
        
            # Reconstructing the original source sentences
            _, auto_loss_src = self.auto_reconst(x_src, y_src, domain='src', noise=noise)

            # Weighting the losses
            auto_loss  += self.lambda_auto * auto_loss_src
            num_toks   += self.get_token_num(src_sentences)

        if tgt_sentences is not None:
            x_tgt, y_tgt = tgt_sentences[:, :-1], tgt_sentences[:, 1:]

            # Reconstructing the original target sentences
            _, auto_loss_tgt = self.auto_reconst(x_tgt, y_tgt, domain='tgt', noise=noise)
            
            # Weighting the losses
            auto_loss += self.lambda_auto * auto_loss_tgt
            num_toks  += self.get_token_num(tgt_sentences)

        if wrap_scalars:
            auto_loss  = auto_loss.unsqueeze(0)
            adv_loss   = adv_loss.unsqueeze(0)
            cross_loss = cross_loss.unsqueeze(0)
            num_toks   = num_toks.unsqueeze(0)

        result = {
            'loss':       auto_loss + adv_loss + cross_loss,
            'auto_loss':  auto_loss,
            'adv_loss':   adv_loss,
            'cross_loss': cross_loss,
            'num_toks':   num_toks
        }

        return result


class AdversarialARModel(ARModelBase):

    def __init__(self, encoder, decoder, discriminator, lambda_auto=1.0, lambda_adv=1.0, lambda_cross=1.0):
        super().__init__(encoder, decoder, discriminator, lambda_auto, lambda_adv, lambda_cross)

    def forward(self, src_sentences, tgt_sentences, noise=False, wrap_scalars=False):
        device = src_sentences.device if src_sentences is not None else tgt_sentences.device

        auto_loss  = torch.tensor(0., device=device).float()
        adv_loss   = torch.tensor(0., device=device).float()
        cross_loss = torch.tensor(0., device=device).float()
        num_toks   = torch.tensor(0., device=device).float()

        if src_sentences is not None:
            x_src, y_src = src_sentences[:, :-1], src_sentences[:, 1:]

            # Reconstructing the original source sentences
            z_src, auto_loss_src = self.auto_reconst(x_src, y_src, domain='src', noise=noise)

            # Adversarial loss for auto reconstruction on source sentences
            adv_loss_src = self.adversarial(z_src, domain='src')

            # Weighting the losses
            auto_loss  += self.lambda_auto * auto_loss_src
            adv_loss   += self.lambda_adv * adv_loss_src
            num_toks   += self.get_token_num(src_sentences)

        if tgt_sentences is not None:
            x_tgt, y_tgt = tgt_sentences[:, :-1], tgt_sentences[:, 1:]

            # Reconstructing the original target sentences
            z_tgt, auto_loss_tgt = self.auto_reconst(x_tgt, y_tgt, domain='tgt', noise=noise)

            # Adversarial loss for auto reconstruction on target sentences
            adv_loss_tgt = self.adversarial(z_tgt, domain='tgt')

            # Weighting the losses
            auto_loss += self.lambda_auto * auto_loss_tgt
            adv_loss  += self.lambda_adv * adv_loss_tgt
            num_toks  += self.get_token_num(tgt_sentences)

        if wrap_scalars:
            auto_loss  = auto_loss.unsqueeze(0)
            adv_loss   = adv_loss.unsqueeze(0)
            cross_loss = cross_loss.unsqueeze(0)
            num_toks   = num_toks.unsqueeze(0)

        result = {
            'loss': auto_loss + adv_loss + cross_loss,
            'auto_loss': auto_loss,
            'adv_loss': adv_loss,
            'cross_loss': cross_loss,
            'num_toks': num_toks
        }

        return result


class CrossDomainARModel(ARModelBase):

    def __init__(self, encoder, decoder, discriminator, lambda_auto=1.0, lambda_adv=1.0, lambda_cross=1.0):
        super().__init__(encoder, decoder, discriminator, lambda_auto, lambda_adv, lambda_cross)

    def forward(self, src_sentences, tgt_sentences, noise=False, wrap_scalars=False):
        device = src_sentences.device

        auto_loss  = torch.tensor(0., device=device).float()
        adv_loss   = torch.tensor(0., device=device).float()
        cross_loss = torch.tensor(0., device=device).float()
        num_toks   = torch.tensor(0., device=device).float()

        x_src, y_src = src_sentences[:, :-1], src_sentences[:, 1:]
        x_tgt, y_tgt = tgt_sentences[:, :-1], tgt_sentences[:, 1:]

        # Reconstructing the original sentences
        z_src, auto_loss_src = self.auto_reconst(x_src, y_src, teacher=x_src, domain='src', noise=noise)
        z_tgt, auto_loss_tgt = self.auto_reconst(x_tgt, y_tgt, teacher=x_tgt, domain='tgt', noise=noise)

        # Adversarial loss for auto reconstruction
        adv_loss_src = self.adversarial(z_src, domain='src')
        adv_loss_tgt = self.adversarial(z_tgt, domain='tgt')

        # Weighting the losses
        auto_loss += self.lambda_auto * (auto_loss_src + auto_loss_tgt)
        adv_loss  += self.lambda_adv * (adv_loss_src + adv_loss_tgt)
        num_toks  += self.get_token_num(src_sentences) + self.get_token_num(tgt_sentences)

        if self.lambda_cross != 0:
            # Getting the mappings between domains to do cross-domain reconstruction
            fake_x_tgt = self.decoder.generate(z_src, x_src.size(1), 1.0, 'greedy', 'tgt', device, strip=False)
            fake_x_src = self.decoder.generate(z_tgt, x_tgt.size(1), 1.0, 'greedy', 'src', device, strip=False)

            # Reconstructing the mapped fake_x to the original domain
            cross_z_tgt, cross_loss_src = self.cross_reconst(fake_x_tgt, y_src, teacher=x_src, domain='tgt')
            cross_z_src, cross_loss_tgt = self.cross_reconst(fake_x_src, y_tgt, teacher=x_tgt, domain='src')

            # Adversarial loss for cross-domain reconstruction
            adv_cross_loss_src = self.adversarial(cross_z_tgt, domain='tgt')
            adv_cross_loss_tgt = self.adversarial(cross_z_src, domain='src')

            # Weighting the losses
            adv_loss   += self.lambda_adv * (adv_cross_loss_src + adv_cross_loss_tgt)
            cross_loss += self.lambda_cross * (cross_loss_src + cross_loss_tgt)

        if wrap_scalars:
            auto_loss  = auto_loss.unsqueeze(0)
            adv_loss   = adv_loss.unsqueeze(0)
            cross_loss = cross_loss.unsqueeze(0)
            num_toks   = num_toks.unsqueeze(0)

        result = {
            'loss': auto_loss + adv_loss + cross_loss,
            'auto_loss': auto_loss,
            'adv_loss': adv_loss,
            'cross_loss': cross_loss,
            'num_toks': num_toks
        }

        return result

