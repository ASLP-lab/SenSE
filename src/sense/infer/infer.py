import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from sense.model.modules import MelSpec

import s3tokenizer

from sense.infer.utils_infer import (
    device,
    load_llm_model,
    load_fm_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)


parser = argparse.ArgumentParser(
    prog="python3 infer.py",
    description="Commandline interface for SenSE with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default=os.path.join(files("sense").joinpath("infer/examples/enhancement"), "basic.toml"),
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file
parser.add_argument("-llm", "--llm_model", type=str, help="The llm model name: SenSE_LLM_Base | SenSE_LLM_Small | etc.")
parser.add_argument("-llm_ckpt", "--llm_ckpt_file", type=str, help="The path to model checkpoint .pt, leave blank to use default")
parser.add_argument("-fm", "--fm_model", type=str, help="The llm model name: SenSE_LLM_Base | SenSE_LLM_Small | etc.")
parser.add_argument("-fm_ckpt", "--fm_ckpt_file", type=str, help="The path to model checkpoint .pt, leave blank to use default")
parser.add_argument("-r", "--ref_audio", type=str, help="The reference audio file.")
parser.add_argument("-na", "--noisy_audio", type=str, help="The noisy reference audio file.")
parser.add_argument("-o", "--output_dir", type=str, help="The path to output folder")
parser.add_argument("-w", "--output_file", type=str, help="The name of output file")
parser.add_argument("--save_sample_rate", type=int, default=24000, help="The sample rate of output audio, default 24000")
parser.add_argument("--load_vocoder_from_local", action="store_true", help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz")
parser.add_argument("--vocoder_name", type=str, choices=["vocos", "bigvgan"], help=f"Used vocoder name: vocos | bigvgan")
parser.add_argument("--nfestep", default=32, type=int)
parser.add_argument("--odemethod", default="euler")
parser.add_argument("--cfg_strength", default=2.0, type=float)
parser.add_argument("--swaysampling", default=-1, type=float)
parser.add_argument("--no_ref_audio", action="store_true", help="To denoise without reference audio, default to False")
parser.add_argument("--no_noisy_audio", action="store_true", help="To use TTS mode, default to False")
parser.add_argument("--no_token", action="store_true", help="To ignore the token prompt, default to False")
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--device", type=str, help="Specify the device to run on")
args = parser.parse_args()


# config file
config = tomli.load(open(args.config, "rb"))

# command-line interface parameters

llm_model = args.llm_model or config.get("llm_model", "SenSE_LLM_Base")
llm_ckpt_file = args.llm_ckpt_file or config.get("llm_ckpt_file", "")
fm_model = args.fm_model or config.get("fm_model", "SenSE_CFM_Base")
fm_ckpt_file = args.fm_ckpt_file or config.get("fm_ckpt_file", "")

ref_audio = args.ref_audio or config.get("ref_audio", "infer/examples/enhancement/clean_ref.wav")
noisy_audio = args.noisy_audio or config.get("noisy_audio", "infer/examples/enhancement/noisy.wav")

output_dir = args.output_dir or config.get("output_dir", "tests")
output_file = args.output_file or config.get(
    "output_file", f"infer_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.wav"
)

load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)
vocoder_name = args.vocoder_name or config.get("vocoder_name", "bigvgan")

seed = args.seed if args.seed is not None else config.get("seed", 0)
no_ref_audio = args.no_ref_audio or config.get("no_ref_audio", False)
no_noisy_audio = args.no_noisy_audio or config.get("no_noisy_audio", False)
no_token = args.no_token or config.get("no_token", False)

device = args.device or config.get("device", device)

llm_model_cfg = OmegaConf.load(str(files("sense").joinpath(f"configs/{llm_model}.yaml")))
llm_model_cls = get_class(f"sense.model.{llm_model_cfg.model.backbone}")
llm_model_arc = llm_model_cfg.model.arch
llm_mel_spec_cfg = llm_model_cfg.model.mel_spec

fm_model_cfg = OmegaConf.load(str(files("sense").joinpath(f"configs/{fm_model}.yaml")))
fm_model_cls = get_class(f"sense.model.{fm_model_cfg.model.backbone}")
fm_model_arc = fm_model_cfg.model.arch
fm_mel_spec_cfg = fm_model_cfg.model.mel_spec

# patches for pip pkg user
if "infer/examples/" in ref_audio:
    ref_audio = str(files("sense").joinpath(f"{ref_audio}"))
if "infer/examples/" in noisy_audio:
    noisy_audio = str(files("sense").joinpath(f"{noisy_audio}"))

# print(f"Using {llm_model} and {fm_model} ...")
print(f'Loading LLM model {llm_model} from {llm_ckpt_file} ...')
llm_model = load_llm_model(
    llm_model_cls, llm_model_arc, llm_ckpt_file, device=device
)
print(f'Loading FM model {fm_model} from {fm_ckpt_file} ...')
fm_model = load_fm_model(
    fm_model_cls, fm_model_arc, fm_mel_spec_cfg, fm_ckpt_file, device=device
)

# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "src/sense/checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "src/sense/checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)

nfe_step = args.nfestep
odemethod = args.odemethod
cfg_strength = args.cfg_strength
sway_sampling_coef = args.swaysampling

# output path
wave_path = Path(output_dir) / output_file

# inference process
tokenizer = s3tokenizer.load_model("speech_tokenizer_v1", download_root="src/S3Tokenizer/s3tokenizer/ckpts").cuda()  # or "speech_tokenizer_v1_25hz speech_tokenizer_v2_25hz"

ref_audio_wav = s3tokenizer.load_audio(ref_audio).to(device)
noisy_audio_wav = s3tokenizer.load_audio(noisy_audio).to(device)

ref_audio_mel = s3tokenizer.log_mel_spectrogram(ref_audio_wav)
noisy_audio_mel = s3tokenizer.log_mel_spectrogram(noisy_audio_wav)

mels, mel_lens = s3tokenizer.padding([ref_audio_mel, noisy_audio_mel])

codes, codes_lens = tokenizer.quantize(mels.cuda(), mel_lens.cuda())

ref_tokens = codes[0, :codes_lens[0].item()]
gen_tokens = codes[1, :codes_lens[1].item()]

def main():
    print("LLM inference ...")
    wav_to_llm_mel = MelSpec(
        n_fft=llm_mel_spec_cfg.n_fft,
        hop_length=llm_mel_spec_cfg.hop_length,
        win_length=llm_mel_spec_cfg.win_length,
        n_mel_channels=llm_mel_spec_cfg.n_mel_channels,
        target_sample_rate=llm_mel_spec_cfg.target_sample_rate,
        mel_spec_type=llm_mel_spec_cfg.mel_spec_type,
    ).to(device)
    wav_to_fm_mel = MelSpec(
        n_fft=fm_mel_spec_cfg.n_fft,
        hop_length=fm_mel_spec_cfg.hop_length,
        win_length=fm_mel_spec_cfg.win_length,
        n_mel_channels=fm_mel_spec_cfg.n_mel_channels,
        target_sample_rate=fm_mel_spec_cfg.target_sample_rate,
        mel_spec_type=fm_mel_spec_cfg.mel_spec_type,
    ).to(device)
    mels = wav_to_llm_mel(noisy_audio_wav.unsqueeze(0))
    mels = mels.permute(0, 2, 1).half()
    mels_len = torch.LongTensor([mels.shape[1]])
    pred_tokens = llm_model.generate(mels, mels_len)

    gen_tokens = pred_tokens[0, :-1]

    print(f'LLM inference done')

    ref_audio_wav_fm, sr = torchaudio.load(ref_audio)
    noisy_audio_wav_fm, sr = torchaudio.load(noisy_audio)
    if sr != fm_mel_spec_cfg.target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, fm_mel_spec_cfg.target_sample_rate)
        ref_audio_wav_fm = resampler(ref_audio_wav_fm)
        noisy_audio_wav_fm = resampler(noisy_audio_wav_fm)
    ref_audio_wav_fm = ref_audio_wav_fm.to(device)
    noisy_audio_wav_fm = noisy_audio_wav_fm.to(device)
    ref_audio_mel_fm = wav_to_fm_mel(ref_audio_wav_fm).permute(0, 2, 1)
    noisy_audio_mel_fm = wav_to_fm_mel(noisy_audio_wav_fm).permute(0, 2, 1)
    ref_tokens_ = ref_tokens
    gen_tokens_ = gen_tokens
    ref_mel_lens = torch.LongTensor([ref_audio_mel_fm.shape[1]]).to(device)
    if no_ref_audio:
        total_mel_lens = torch.LongTensor([noisy_audio_mel_fm.shape[1]]).to(device)
        total_tokens = gen_tokens_.unsqueeze(0).to(device)
    else:
        total_mel_lens = torch.LongTensor([ref_audio_mel_fm.shape[1] + noisy_audio_mel_fm.shape[1]]).to(device)
        total_tokens = torch.cat([ref_tokens_, gen_tokens_], dim=-1).unsqueeze(0).to(device)


    print("FM inference ...")
    with torch.inference_mode():
        generated, _ = fm_model.sample(
            cond=ref_audio_mel_fm,
            cond_noisy=noisy_audio_mel_fm,
            text=total_tokens,
            duration=total_mel_lens,
            lens=ref_mel_lens,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            no_ref_audio=no_ref_audio,
            no_noisy_audio=no_noisy_audio,
            no_token=no_token,
            seed=seed,
        )
        # Final result
        assert len(generated) == 1
        gen = generated[0]
        target_rms = torch.sqrt(torch.mean(torch.square(noisy_audio_wav))).cpu()
        if not no_ref_audio:
            gen = gen[ref_mel_lens : total_mel_lens, :].unsqueeze(0)
        else:
            gen = gen.unsqueeze(0)
        gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
        if fm_mel_spec_cfg.mel_spec_type == "vocos":
            generated_wave = vocoder.decode(gen_mel_spec).cpu()
        elif fm_mel_spec_cfg.mel_spec_type == "bigvgan":
            generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()

        if hasattr(fm_mel_spec_cfg, 'output_sample_rate'):
            output_sample_rate = fm_mel_spec_cfg.output_sample_rate
        else:
            output_sample_rate = fm_mel_spec_cfg.target_sample_rate
        resampler = torchaudio.transforms.Resample(output_sample_rate, args.save_sample_rate)
        generated_wave = resampler(generated_wave)
        rms = torch.sqrt(torch.mean(torch.square(generated_wave)))
        generated_wave = generated_wave / rms * target_rms
        torchaudio.save(wave_path, generated_wave, args.save_sample_rate)

if __name__ == "__main__":
    main()
