import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalTimbreEncoder(nn.Module):
    def __init__(self, input_dimension=512, n_layers=5, hidden_dimension=512):
        super().__init__()
        # Project high-dim spk_map to hidden dim
        self.proj = nn.Conv1d(input_dimension * n_layers, hidden_dimension, kernel_size=1)
        
        self.texture_extractor = nn.Sequential(
            nn.Conv1d(hidden_dimension, hidden_dimension, kernel_size=5, stride=1, padding=2, groups=hidden_dimension),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dimension, hidden_dimension, kernel_size=1),
            nn.InstanceNorm1d(hidden_dimension) # Better than LayerNorm for style
        )

    def forward(self, spk_map):
        B, d, T, L = spk_map.shape
        # spk_map is [B, 512, T, 5] -> Combine layers into channel dim
        x = spk_map.permute(0, 1, 3, 2).reshape(B, d * L, T) 
        
        z = self.proj(x)
        
        # Extract textures (raspy, breathy, etc.) instead of just averaging
        z = self.texture_extractor(z)

        # Final Soft-Clip for stability
        return torch.tanh(z)

class GlobalSpeakerEncoder(nn.Module):
    def __init__(self, input_dimension=512, n_layers=5, hidden_dimension=512, att=True):
        super().__init__()
        self.att = att
        in_dim = input_dimension * n_layers
        
        # Add an MLP Bottleneck to increase contrast
        self.encoder_mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dimension * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dimension * 2, hidden_dimension),
        )
        
        if att:
            self.att_fc1 = nn.Linear(in_dim, hidden_dimension)
            self.att_fc2 = nn.Linear(hidden_dimension, 1)

    def forward(self, spk_map: torch.Tensor):
        B, d, T, L = spk_map.shape
        # spk_map: [B, 512, T, 5]
        x = spk_map.transpose(2, 3).reshape(B, d * L, T) 

        if self.att:
            # Attend to parts of the audio that are "speaker-rich"
            att_weights = torch.tanh(self.att_fc1(x.transpose(1, 2)))
            att_weights = self.att_fc2(att_weights).transpose(1, 2)
            att_weights = F.softmax(att_weights, dim=-1)
            
            mu = (x * att_weights).sum(dim=-1)
            sigma = torch.sqrt(((x - mu.unsqueeze(-1))**2 * att_weights).sum(dim=-1) + 1e-9)
        else:
            mu = x.mean(dim=-1)
            sigma = x.std(dim=-1)
        
        stats = torch.cat([mu, sigma], dim=-1) # [B, d*L*2]
        
        # Apply the Non-Linear Bottleneck
        z = self.encoder_mlp(stats)
        
        # Final L2 normalization is critical for cosine similarity training
        return F.normalize(z, p=2, dim=-1)
    
class SpeakerModule(nn.Module):
    def __init__(self, spk_map_dim: int = 512, n_layers: int = 5, local_hidden_dim: int = 512, global_hidden_dim: int = 512, use_global_att: bool = True):
        super().__init__()
        self.local_encoder = LocalTimbreEncoder(input_dimension=spk_map_dim, n_layers=n_layers, hidden_dimension=local_hidden_dim)
        self.global_encoder = GlobalSpeakerEncoder(input_dimension=spk_map_dim, n_layers=n_layers, hidden_dimension=global_hidden_dim, att=use_global_att)
        self._init_weights()
        
    def forward(self, spk_map: torch.Tensor):
        local_features = self.local_encoder(spk_map)   # [B, local_dim, T']
        global_features = self.global_encoder(spk_map) # [B, global_dim]
        return global_features, local_features
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)