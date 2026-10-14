import torch
import torch.nn as nn
from src.components.normalization import LayerNorm

class CodePredictor(nn.Module):
    def __init__(self, embed_dim, num_codes, apply_ln=False):
        super(CodePredictor, self).__init__()
        self.classifier = nn.Sequential(
            LayerNorm(embed_dim) if apply_ln else nn.Identity(),
            nn.Linear(embed_dim, num_codes)
        )

    def forward(self, x):
        ''' Forward function of Code Predictor:
            x = (B, embed_dim, len)
        '''
        # pass through classifier
        outputs = self.classifier(x.transpose(1,2)).transpose(1,2)
        
        return outputs