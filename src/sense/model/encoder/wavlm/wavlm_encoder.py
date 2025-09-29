import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class WavLMEncoder(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    @classmethod
    def load(cls, model_path):
        from wavlm import WavLM, WavLMConfig
        checkpoint = torch.load(model_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        WavLM_model = WavLM(cfg)
        WavLM_model.load_state_dict(checkpoint['model'])
        
        try:
            from torch.nn.utils import remove_weight_norm
            if hasattr(WavLM_model, 'encoder') and hasattr(WavLM_model.encoder, 'pos_conv'):
                if isinstance(WavLM_model.encoder.pos_conv, nn.Sequential):
                    conv_layer = WavLM_model.encoder.pos_conv[0]
                    try:
                        remove_weight_norm(conv_layer)
                        print("Removed weight_norm from WavLM encoder pos_conv")
                    except ValueError:
                        print("No weight_norm found in pos_conv")
        except Exception as e:
            print(f"Warning: Could not remove weight_norm: {e}")
        
        return cls(cfg, WavLM_model)

    def extract_features(self, source, padding_mask):
        return self.model.extract_features(source, padding_mask)   # out: features, masks

if __name__ == "__main__":
    encoder = WavLMEncoder.load("src/sense/checkpoints/WavLM-Large/WavLM-Large.pt")
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'params: {total_params / 1e6} M')
    audio = torch.randn(2, 64000)
    padding_mask = torch.ones_like(audio, dtype=torch.bool)
    padding_mask[1, -32000:] = False
    padding_mask = ~padding_mask
    audio = torch.nn.functional.pad(audio,(160,160))
    out = encoder.extract_features(audio, padding_mask)
    mask = out[1]
    print(f'out shape: {out[0].shape}')
    print(f'mask shape: {mask.shape}')
    print(f'mask: {mask}')
