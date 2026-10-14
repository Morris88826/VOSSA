import torch
from torch.nn import Conv1d


class SpeakerAdapter(torch.nn.Module):

    def __init__(self,
                 speaker_dim,
                 adapter_dim,
                 epsilon=1e-5
                 ):
        super(SpeakerAdapter, self).__init__()
        self.speaker_dim = speaker_dim
        self.adapter_dim = adapter_dim
        self.epsilon = epsilon
        self.W_scale = torch.nn.Conv1d(self.speaker_dim, self.adapter_dim, 1)
        self.W_bias = torch.nn.Conv1d(self.speaker_dim, self.adapter_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.W_scale.weight, 0.0)
        torch.nn.init.constant_(self.W_scale.bias, 1.0)
        torch.nn.init.constant_(self.W_bias.weight, 0.0)
        torch.nn.init.constant_(self.W_bias.bias, 0.0)

    def forward(self, x, speaker_embedding):
        x = x.transpose(1, -1)
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding.unsqueeze(-1))
        bias = self.W_bias(speaker_embedding.unsqueeze(-1))
        y *= scale.transpose(1, -1)
        y += bias.transpose(1, -1)
        y = y.transpose(1, -1)
        return y