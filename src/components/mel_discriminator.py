import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm



# ------------------------------------------
# PatchGAN Discriminator over (freq x time)
# ------------------------------------------

class NLayerPatchDiscriminator2D(nn.Module):
    """
    2D PatchGAN over log-mels: input (B, 1, F, T).
    Returns final score map + intermediate features for FM loss.
    """
    def __init__(self, in_ch=1, base_ch=64, n_layers=4, norm='in'):
        super().__init__()
        feat_layers = []
        ch = base_ch

        def norm2d(c):
            if norm == 'in':
                return nn.InstanceNorm2d(c, affine=True)
            elif norm == 'bn':
                return nn.BatchNorm2d(c)
            else:
                return nn.Identity()

        # First layer (no norm per usual PatchGAN)
        feat_layers += [
            spectral_norm(nn.Conv2d(in_ch, ch, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Pyramid
        for i in range(1, n_layers):
            prev_ch = ch
            ch = min(prev_ch * 2, 512)
            feat_layers += [
                spectral_norm(nn.Conv2d(prev_ch, ch, kernel_size=4, stride=2, padding=1)),
                norm2d(ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # One more conv (stride 1) for patch decisions
        feat_layers += [
            spectral_norm(nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        self.features = nn.ModuleList(feat_layers)
        self.final = spectral_norm(nn.Conv2d(ch, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x):  # x: (B,1,F,T)
        feats = []
        h = x
        for layer in self.features:
            h = layer(h)
            feats.append(h)
        out = self.final(h)
        return out, feats  # out: (B,1,f',t'), feats: list of tensors


class MultiScaleMelDiscriminator(nn.Module):
    """
    Multi-scale: evaluate (F,T), (F,T/2), (F,T/4) via avg-pooling on time.
    More stable GAN + better FM. You can add more scales if helpful.
    """
    def __init__(self, in_ch=1, base_ch=64, n_layers=4, n_scales=3):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # halve time only
        self.discriminators = nn.ModuleList([
            NLayerPatchDiscriminator2D(in_ch=in_ch, base_ch=base_ch, n_layers=n_layers)
            for _ in range(n_scales)
        ])

    def forward(self, x):  # x: (B,1,F,T)
        outs = []
        feats_all = []
        h = x
        for d in self.discriminators:
            out, feats = d(h)
            outs.append(out)
            feats_all.append(feats)
            # downscale time dimension for next scale
            if h.shape[-1] >= 4:  # avoid over-pooling on very short sequences
                h = self.pool(h)
        return outs, feats_all  # lists length = n_scales


# -------------------------------
# Losses: LSGAN, FM, Charbonnier
# -------------------------------

class CharbonnierLoss(nn.Module):
    """
    Robust L1: sqrt((x - y)^2 + eps^2). Use reduction='mean' or 'sum'.
    """
    def __init__(self, eps=1e-3, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def lsgan_d_loss(real_scores, fake_scores):
    """
    LSGAN discriminator loss: (D(real)-1)^2 + (D(fake))^2, averaged over scales.
    real_scores/fake_scores: lists of tensors from MultiScaleMelDiscriminator
    """
    loss = 0.0
    n = len(real_scores)
    for r, f in zip(real_scores, fake_scores):
        loss += ((r - 1.0) ** 2).mean() + ((f - 0.0) ** 2).mean()
    return loss / n

def lsgan_g_loss(fake_scores):
    """
    LSGAN generator (post-net) loss: (D(fake)-1)^2, averaged over scales.
    """
    loss = 0.0
    n = len(fake_scores)
    for f in fake_scores:
        loss += ((f - 1.0) ** 2).mean()
    return loss / n

class FeatureMatchingLoss(nn.Module):
    """
    Sum over scales and layers of L1 distance between discriminator features.
    Normalized by element count per feature map.
    """
    def __init__(self):
        super().__init__()

    def forward(self, real_feats_all, fake_feats_all):
        """
        real_feats_all / fake_feats_all: lists over scales, 
        each a list of feature maps per layer.
        """
        total = 0.0
        count = 0
        for real_feats, fake_feats in zip(real_feats_all, fake_feats_all):
            for r, f in zip(real_feats, fake_feats):
                # L1, normalized
                total += (r - f).abs().mean()
                count += 1
        if count == 0:
            return torch.tensor(0.0, device=real_feats_all[0][0].device)
        return total / count


# # ------------------------------------------
# # Wiring: generator wrapper + loss assembly
# # ------------------------------------------

# class MelRefinerGAN(nn.Module):
#     """
#     Wraps a post-net refiner and a multi-scale discriminator.
#     You provide pre-mel from your base model; this refines to post-mel.
#     """
#     def __init__(self, n_mels=80):
#         super().__init__()
#         self.postnet = MelPostNet(n_mels=n_mels, dim=256, layers=3, kernel_size=7, dilation=1, causal=True)
#         self.disc = MultiScaleMelDiscriminator(in_ch=1, base_ch=64, n_layers=4, n_scales=3)

#     @torch.no_grad()
#     def _to_disc_input(self, mel):  # (B,80,T) -> (B,1,80,T)
#         return mel.unsqueeze(1)

#     def forward_post(self, mel_pre):
#         return self.postnet(mel_pre)

#     def disc_forward(self, mel):  # log-mel
#         x = self._to_disc_input(mel)
#         outs, feats = self.disc(x)
#         return outs, feats
