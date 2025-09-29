import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from sense.model import WavLM, WavLMConfig
from sense.model.modules import MelSpec
from typing import List, Optional, Tuple

import s3tokenizer

from sense.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_llm_model,
    load_vocoder,
    mel_spec_type,
    nfe_step,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    speed,
    sway_sampling_coef,
    target_rms,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("sense").joinpath("infer/examples/basic"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: SenSE_Base | SenSE_Small | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to SenSE model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-nr",
    "--noisy_ref_audio",
    type=str,
    help="The noisy reference audio file.",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)
args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))

# command-line interface parameters

model = args.model or config.get("model", "SenSE_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/enhancement/clean_ref_en.wav")
noisy_ref_audio = args.noisy_ref_audio or config.get("noisy_ref_audio", "infer/examples/enhancement/noisy_ref_en.wav")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

device = args.device or config.get("device", device)

model_cfg = OmegaConf.load(
    args.model_cfg or config.get("model_cfg", str(files("sense").joinpath(f"configs/{model}.yaml")))
)
model_cls = get_class(f"sense.model.{model_cfg.model.backbone}")
model_arc = model_cfg.model.arch
llm_model = load_llm_model(model_cls, model_arc, ckpt_file, device=device)
llm_model.eval()

# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("sense").joinpath(f"{ref_audio}"))
if "infer/examples/" in noisy_ref_audio:
    noisy_ref_audio = str(files("sense").joinpath(f"{noisy_ref_audio}"))

print(f"Using {model}...")

# inference process
tokenizer = s3tokenizer.load_model("speech_tokenizer_v1", download_root="src/S3Tokenizer/s3tokenizer/ckpts").cuda()  # or "speech_tokenizer_v1_25hz speech_tokenizer_v2_25hz"

ref_audio_wav = s3tokenizer.load_audio(ref_audio).to(device)
noisy_audio_wav = s3tokenizer.load_audio(noisy_ref_audio).to(device)

ref_audio_mel = s3tokenizer.log_mel_spectrogram(ref_audio_wav)
noisy_audio_mel = s3tokenizer.log_mel_spectrogram(noisy_audio_wav)

mels, mel_lens = s3tokenizer.padding([ref_audio_mel, noisy_audio_mel])

codes, codes_lens = tokenizer.quantize(mels.cuda(), mel_lens.cuda())

ref_token = codes[0, :codes_lens[0].item()]
gen_token = codes[1, :codes_lens[1].item()]

def main():
    wav_to_mel = MelSpec(
                    n_fft=400,
                    hop_length=160,
                    win_length=400,
                    n_mel_channels=80,
                    target_sample_rate=16000,
                    mel_spec_type="whisper",
                )
    mels = wav_to_mel(ref_audio_wav.unsqueeze(0))
    mels = mels.permute(0, 2, 1).half()
    mels_len = torch.LongTensor([mels.shape[1]])
    pred_tokens = llm_model.generate(mels, mels_len)
    output_token_file = os.path.join(output_dir, output_file)
    os.makedirs(os.path.dirname(output_token_file), exist_ok=True)
    with open(output_token_file, "w", encoding="utf-8") as f:
        if pred_tokens.dim() == 1:
            f.write(" ".join([str(t.item()) for t in pred_tokens]))
        else:
            for row in pred_tokens:
                f.write(" ".join([str(t.item()) for t in row]) + "\n")

    print(f'pred_tokens.shape: {pred_tokens.shape}')
    print(f'ref_token.shape: {ref_token.shape}')
    print(f'pred_tokens: {pred_tokens}')
    print(f'ref_tokens: {ref_token}')

if __name__ == "__main__":
    main()