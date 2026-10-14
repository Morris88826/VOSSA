import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

class ContrastiveHead(nn.Module):
    def __init__(self, in_dim=512, proj_dim=512, temperature=0.05):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, in_dim)
        )
        self.temperature = temperature

    def _all_gather(self, z):
        if not dist.is_available() or not dist.is_initialized():
            return z
        with torch.no_grad():
            gathered = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, z.detach().contiguous())
            z_all = torch.cat(gathered, dim=0)
        return z_all

    def forward(self, z_a, z_b, speaker_ids):
        # Normalize and project
        z_a = F.normalize(self.projector(z_a), dim=-1)
        z_b = F.normalize(self.projector(z_b), dim=-1)
        B = z_a.size(0)

        # Gather across GPUs (embeddings + speaker ids)
        z_a_all = self._all_gather(z_a)
        z_b_all = self._all_gather(z_b)
        ids_all = self._all_gather(speaker_ids)  # [B * world_size]

        # Compute similarities: each local batch vs. all global embeddings
        logits_ab = torch.matmul(z_a, z_b_all.T) / self.temperature  # [B, B*W]
        logits_ba = torch.matmul(z_b, z_a_all.T) / self.temperature  # [B, B*W]

        # Mask same-speaker negatives (keep the positive unmasked)
        same_spk_ab = speaker_ids.view(-1, 1) == ids_all.view(1, -1)
        same_spk_ba = speaker_ids.view(-1, 1) == ids_all.view(1, -1)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            start = rank * B
        else:
            start = 0

        target = torch.arange(start, start + B, device=z_a.device)

        # Do NOT mask out the true positive
        same_spk_ab[torch.arange(B, device=z_a.device), target] = False
        same_spk_ba[torch.arange(B, device=z_b.device), target] = False

        logits_ab[same_spk_ab] = -1e9
        logits_ba[same_spk_ba] = -1e9

        # Compute the positive similarity directly (avoid full diag)
        pos_ab = (z_a * z_b).sum(dim=-1) / self.temperature
        pos_ba = (z_b * z_a).sum(dim=-1) / self.temperature

        logits_ab[torch.arange(B, device=z_a.device), target] = pos_ab
        logits_ba[torch.arange(B, device=z_b.device), target] = pos_ba

        # Cross-entropy over global dimension
        loss_ab = F.cross_entropy(logits_ab, target)
        loss_ba = F.cross_entropy(logits_ba, target)

        return 0.5 * (loss_ab + loss_ba)
