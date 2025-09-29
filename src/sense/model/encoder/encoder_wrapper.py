import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_encoder(model_name, model_path=None, freeze_encoder=False, **kwargs):
    encoder_name = model_name.lower()
    if encoder_name == "conformer":
        from sense.model.encoder.conformer.encoder import ConformerEncoder
        encoder = ConformerEncoder(**kwargs)
    if encoder_name == "conformer2":
        from sense.model.encoder.conformer.encoder2 import ConformerEncoder
        encoder = ConformerEncoder(**kwargs)
    if encoder_name == "whisper" or encoder_name == "qwen-audio":
        from sense.model.encoder.whisper.whisper_encoder import WhisperWrappedEncoder
        encoder = WhisperWrappedEncoder.load(model_path)
    if encoder_name == "wavlm":
        from sense.model.encoder.wavlm.wavlm_encoder import WavLMEncoder
        encoder = WavLMEncoder.load(model_path)
    if encoder_name == "hubert":
        raise NotImplementedError("HuBERT encoder is not implemented")
    if encoder_name == "musicfm":
        raise NotImplementedError("MusicFM encoder is not implemented")
    if encoder_name == "emotion2vec":
        raise NotImplementedError("Emotion2vec encoder is not implemented")

    if freeze_encoder:
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()

    return encoder

if __name__=='__main__':
    from sense.model.encoder.whisper.whisper_encoder import preprocess_audio
    model_name = "whisper"
    model_path = "src/u3se/checkpoints/whisper-small"
    encoder = setup_encoder(model_name, "large")
    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f'params: {total_params / 1e6} M')
    audio = torch.randn(1, 16000)
    audio_mel = preprocess_audio(audio)
    print(f'audio_mel shape: {audio_mel.shape}')
    # audio_mel = torch.randn(2, 500, 80)  # Example mel spectrogram input
    # audio = torch.randn(2, 16000 * 4)
    # audio_mel = preprocess_audio(audio)
    # print(f'audio_mel shape: {audio_mel.shape}')
    encoder_outs = encoder.extract_variable_length_features(audio_mel) # bs*seq*dim
    print(f'encoder_outs shape: {encoder_outs.shape}')