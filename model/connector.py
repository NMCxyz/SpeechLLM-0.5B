import torch
from torch import nn


def get_connector(name, audio_enc_dim, llm_dim, k, finetune_connector=False):
    connector = None
    if name == 'linear-pool':
        connector = LinearPoolConnector(audio_enc_dim, llm_dim, k)
    elif name == 'linear':
        connector = LinearConnector(audio_enc_dim, llm_dim, k)
    elif name == 'cnn':
        connector = CNNConnector(audio_enc_dim, llm_dim, k)
    elif name == 'conformer':
        connector = ConformerConnector(audio_enc_dim, llm_dim, k)
    elif name == 'transformer':
        connector = TransformerConnector(audio_enc_dim, llm_dim, k)
    elif name == 'minicpm-projection':
        connector = MultiModalConnector(audio_enc_dim, llm_dim, k)
    
    for param in connector.parameters():
        param.requires_grad = finetune_connector
        
    return connector



class ConformerBlock(nn.Module):
    def __init__(self, dim, ff_dim=None, heads=4, dropout=0.1):
        super().__init__()
        self.ff_dim = ff_dim or dim*4
        
        # Feed Forward 1
        self.ff1 = nn.Sequential(
            nn.Linear(dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(dim)
        
        # Convolution
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim*2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.conv_norm = nn.LayerNorm(dim)
        
        # Feed Forward 2
        self.ff2 = nn.Sequential(
            nn.Linear(dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, dim),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Feed Forward 1
        x = x + 0.5 * self.ff1(x)
        
        # Multi-Head Self-Attention
        attn_out = self.attention_norm(x)
        attn_out, _ = self.attention(attn_out, attn_out, attn_out)
        x = x + attn_out
        
        # Convolution
        conv_out = self.conv_norm(x)
        conv_out = self.conv(conv_out.transpose(-1, -2)).transpose(-1, -2)
        x = x + conv_out
        
        # Feed Forward 2
        x = x + 0.5 * self.ff2(x)
        
        return self.norm(x)

class ConformerConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k=None):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, out_channels)
        
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(out_channels) for _ in range(2)
        ])
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.conformer_blocks:
            x = block(x)
            
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, ff_dim=None, heads=8, dropout=0.1):
        super().__init__()
        self.ff_dim = ff_dim or dim*4
        
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, dim)
        )
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed Forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x

class TransformerConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k=None):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, out_channels)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(out_channels) for _ in range(4)
        ])
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.transformer_blocks:
            x = block(x)
            
        return x


class MultiModalConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k=2):
        super().__init__()
        self.avg_pooler = nn.AvgPool1d(k, stride=k) 
        self.projection_layer = MultiModalProjector(in_dim=in_dim, out_dim=out_dim)
        self.avg_pooler_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/speech_encoder/weights/audio_avg_pooler_weights.pth"
        self.projection_path = "/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/speech_encoder/weights/audio_projection_layer_weights.pth"
        self.avg_pooler.load_state_dict(torch.load(self.avg_pooler_path, map_location="cpu"))
        self.projection_layer.load_state_dict(torch.load(self.projection_path, map_location="cpu"))
        print("Loaded pretrained weights for connector components")

    def forward(self, x):
        x = self.projection_layer(x)  
        x = x.transpose(1, 2) 
        x = self.avg_pooler(x)  
        x = x.transpose(1, 2)  
        return x

class MultiModalProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear1 = nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=out_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    
    

class LinearConnector(nn.Module):
    def __init__(self, in_dim, out_dim, k):
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)

    def forward(self, x):
        x = self.layer(x)
        x = x.transpose(1, 2) 
        x = self.pool(x)  
        x = x.transpose(1, 2)
        return x


class LinearPoolConnector(nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super(LinearPoolConnector, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU())
        self.pool = nn.AvgPool1d(kernel_size=k, stride=k)
        self.linear2 = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim))

    def forward(self, x):
        # x: [B, T, d]
        x = self.linear1(x)  # x: [B, T, D]
        x = x.transpose(1, 2)  # x: [B, D, T]
        x = self.pool(x)  # x: [B, D, T']
        x = x.transpose(1, 2)  # x: [B, T', D]
        x = self.linear2(x)
        return x

# class CNNConnector(nn.Module):
#     def __init__(self, in_channels, out_channels, k):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv1d(in_channels, out_channels//2, kernel_size=5,
#                       stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv1d(out_channels//2, out_channels, kernel_size=5,
#                       stride=k, padding=0),
#             nn.ReLU(),
#             nn.Conv1d(out_channels, out_channels, kernel_size=5,
#                       stride=1, padding=0),
#         )

#     def forward(self, x):
#         return self.layer(x.transpose(1,2)).transpose(1,2)

class CNNConnector(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k, padding='same').to(dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=k, padding='same').to(dtype=torch.bfloat16)
        )

    def forward(self, x):
        return self.layer(x.transpose(1,2)).transpose(1,2)





if __name__ == "__main__":
    model = CNNConnector(128, 256)
    x = torch.randn(4, 50, 128)
    z = model(x)
    print(z.shape)