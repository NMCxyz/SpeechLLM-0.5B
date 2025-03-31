import torch
import json
import sys
from torch import nn
from transformers import AutoModel
sys.path.append("/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/speech_encoder")
from minicpmo_whisper import MiniCPMO_Whisper, MiniCPMWhisperEncoder, WhisperConfig
from speechtokenizer import SpeechTokenizer

def get_audio_encoder(name, finetune_encoder):
    if name == 'minicpm-o':
        return MiniCPMAudioEncoder(model_name="minicpmo", finetune=finetune_encoder)
    else:
        raise NotImplementedError
    

class MiniCPMAudioEncoder(nn.Module):
    def __init__(self, model_name = 'minicpmo', finetune=False):
        super().__init__()
        audio_config = json.load(open("/mnt/4T2/thuctap/cuongnm75/MultimodalVisualVideoStreaming/EOT/whisper_streaming/infer/audio_config.json", "r"))
        self.audio_config = WhisperConfig(**audio_config)
        self.encoder = MiniCPMWhisperEncoder(self.audio_config)
        
        for param in self.encoder.parameters():
            param.requires_grad = finetune

    def forward(self, x):
        outputs = self.encoder(x)
        return outputs.last_hidden_state


class TransformerAudioEnoder(nn.Module):
    def __init__(self, model_name='facebook/hubert-xlarge-ll60k', finetune=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for param in self.encoder.parameters():
            param.requires_grad = finetune
            
        for param in self.encoder.encoder.layers[-15:].parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.encoder(x).last_hidden_state


if __name__ == "__main__":
    model = SpeechTokenizerEnoder()
    # print(model)

    x = torch.randn(2, 1, 16000)
    z = model(x)
    print(z.shape)