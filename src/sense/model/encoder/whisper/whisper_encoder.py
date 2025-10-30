import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import whisper

class WhisperWrappedEncoder:
    
    @classmethod
    def load(cls, model_path):
        
        def extract_variable_length_features(self, x: torch.Tensor):
            """
            x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
                the mel spectrogram of the audio
            """
            x = F.gelu(self.conv1(x))
            x = F.gelu(self.conv2(x))
            x = x.permute(0, 2, 1)

            # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
            # x = (x + self.positional_embedding).to(x.dtype)
            x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

            for block in self.blocks:
                x = block(x)

            x = self.ln_post(x)
            return x

        # encoder = whisper.load_model(name=model_path, device='cpu').encoder
        import os
        if os.path.isfile(model_path):
            encoder = whisper.load_model(name=model_path, device='cpu').encoder
        else:
            encoder = whisper.load_model(name="large-v3", device='cpu', download_root=None).encoder
        encoder.extract_variable_length_features = types.MethodType(extract_variable_length_features, encoder)
        return encoder

@staticmethod
def preprocess_audio(audio_path_or_array, sample_rate=16000, n_mels=128):
    if isinstance(audio_path_or_array, str):
        audio = whisper.load_audio(audio_path_or_array)
    else:
        audio = audio_path_or_array
        
    # Whisper mel spectrogram extraction
    mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
    return mel