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

import s3tokenizer

from sense.infer.utils_infer import (
    cfg_strength,
    cross_fade_duration,
    device,
    fix_duration,
    infer_process,
    load_llm_model,
    load_fm_model,
    load_vocoder,
    reload_wavLM_large,
    get_ssl_features,
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
    "-fm",
    "--fm_model",
    type=str,
    help="The llm model name: LLM_AR_v1 | LLM_AR_v2 | etc.",
)
parser.add_argument(
    "-fm_mc",
    "--fm_model_cfg",
    type=str,
    help="The path to U3-SE model config file .yaml",
)
parser.add_argument(
    "-fm_ckpt",
    "--fm_ckpt_file",
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
    "-to",
    "--token_path",
    type=str,
    help="The path to token file, leave blank to use default",
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
    "--load_vocoder_from_local",
    action="store_true",
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--no_ref_audio",
    action="store_true",
    help="To denoise without reference audio, default to False",
)
parser.add_argument(
    "--no_noisy_ref_audio",
    action="store_true",
    help="To use TTS mode, default to False",
)
parser.add_argument(
    "--no_token",
    action="store_true",
    help="To ignore the token prompt, default to False",
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

fm_model = args.fm_model or config.get("fm_model", "SenSE_Base")
fm_ckpt_file = args.fm_ckpt_file or config.get("fm_ckpt_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/enhancement/clean_ref_en.wav")
noisy_ref_audio = args.noisy_ref_audio or config.get("noisy_ref_audio", "infer/examples/enhancement/noisy_ref_en.wav")
token_path = args.token_path

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_cli_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)
vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)

no_ref_audio = args.no_ref_audio or config.get("no_ref_audio", False)
no_noisy_ref_audio = args.no_noisy_ref_audio or config.get("no_noisy_ref_audio", False)
no_token = args.no_token or config.get("no_token", False)

device = args.device or config.get("device", device)

fm_model_cfg = OmegaConf.load(
    args.fm_model_cfg or config.get("fm_model_cfg", str(files("sense").joinpath(f"configs/{fm_model}.yaml")))
)
fm_model_cls = get_class(f"sense.model.{fm_model_cfg.model.backbone}")
fm_model_arc = fm_model_cfg.model.arch
mel_spec_cfg = fm_model_cfg.model.mel_spec

# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("sense").joinpath(f"{ref_audio}"))
if "infer/examples/" in noisy_ref_audio:
    noisy_ref_audio = str(files("sense").joinpath(f"{noisy_ref_audio}"))

print(f'Loading FM model {fm_model} from {fm_ckpt_file} ...')
fm_model = load_fm_model(
    fm_model_cls, fm_model_arc, mel_spec_cfg, fm_ckpt_file, device=device
)

# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "src/sense/checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "src/sense/checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)

# output path
wave_path = Path(output_dir) / output_file

# inference process
tokenizer = s3tokenizer.load_model("speech_tokenizer_v1_25hz", download_root="src/S3Tokenizer/s3tokenizer/ckpts").cuda()  # or "speech_tokenizer_v1_25hz speech_tokenizer_v2_25hz"

ref_audio_wav = s3tokenizer.load_audio(ref_audio).to(device)
noisy_audio_wav = s3tokenizer.load_audio(noisy_ref_audio).to(device)

ref_audio_mel = s3tokenizer.log_mel_spectrogram(ref_audio_wav)
noisy_audio_mel = s3tokenizer.log_mel_spectrogram(noisy_audio_wav)

mels, mel_lens = s3tokenizer.padding([ref_audio_mel, noisy_audio_mel])

codes, codes_lens = tokenizer.quantize(mels.cuda(), mel_lens.cuda())

ref_tokens = codes[0, :codes_lens[0].item()]
gen_tokens = codes[1, :codes_lens[1].item()]

def main():
    # Read pred_tokens from file
    pred_tokens_path = token_path
    with open(pred_tokens_path, "r", encoding="utf-8") as f:
        token_str = f.read().strip()
        token_list = [int(t) for t in token_str.split() if t.strip()]
        pred_tokens = torch.tensor(token_list, dtype=torch.long).to(device)
        pred_tokens = pred_tokens[:-1]   # Remove the trailing eos token

    gen_tokens = pred_tokens[:ref_tokens.shape[-1]]
    print(f'gen_tokens shape: {gen_tokens.shape}')
    print(f'pred_tokens.shape: {pred_tokens.shape}')

    main_voice = {"ref_audio": ref_audio, "ref_tokens": ref_tokens}
    if "voices" not in config:
        voices = {"main": main_voice}
    else:
        voices = config["voices"]
        voices["main"] = main_voice
    for voice in voices:
        # print("Voice:", voice)
        # print("ref_audio ", voices[voice]["ref_audio"])
        voices[voice]["ref_audio"], voices[voice]["ref_tokens"] = preprocess_ref_audio_text(
            voices[voice]["ref_audio"], voices[voice]["ref_tokens"], tokenizer
        )
        # print("ref_audio_", voices[voice]["ref_audio"], "\n\n")

    ref_audio_ = voices[voice]["ref_audio"]
    noisy_ref_audio_ = noisy_ref_audio
    ref_tokens_ = voices[voice]["ref_tokens"]
    gen_tokens_ = gen_tokens
    # print(f"Voice: {voice}")
    print("FM inference ...")
    audio_segment, final_sample_rate, spectrogram = infer_process(
        ref_audio_,
        noisy_ref_audio_,
        ref_tokens_,
        gen_tokens_,
        fm_model,
        vocoder,
        no_ref_audio=no_ref_audio,
        no_noisy_ref_audio=no_noisy_ref_audio,
        no_token=no_token,
        mel_spec_type=vocoder_name,
        target_rms=target_rms,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        cfg_strength=cfg_strength,
        sway_sampling_coef=sway_sampling_coef,
        speed=speed,
        fix_duration=fix_duration,
        device=device,
    )

    print(f'FM inference done')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(wave_path, "wb") as f:
        sf.write(f.name, audio_segment, final_sample_rate)
        print(f.name)

if __name__ == "__main__":
    main()