import os
import sys


sys.path.append(os.getcwd())

import argparse
import time
from importlib.resources import files

import torch
import torchaudio
from accelerate import Accelerator
from hydra.utils import get_class
from omegaconf import OmegaConf
from tqdm import tqdm
import s3tokenizer

from sense.eval.utils_eval import (
    get_inference_prompt,
    get_librispeech_test_clean_metainfo,
    get_seedtts_testset_metainfo,
    get_dns_challenge_metainfo,
)
from sense.infer.utils_infer import (
    load_checkpoint,
    load_vocoder,
    load_llm_model,
    load_fm_model,
    reload_wavLM_large,
    get_ssl_features,
)
from sense.model import CFM
from sense.model.utils import get_tokenizer

accelerator = Accelerator()
device = f"cuda:{accelerator.process_index}"

rel_path = str(files("sense").joinpath("../../"))

def main():
    parser = argparse.ArgumentParser(description="batch inference")

    parser.add_argument("--fm_model", default="SenSE_Base", required=True)
    parser.add_argument("--fm_ckpt_file", default="ckpts/SenSE_Base_vocos_s3tokenizer_v1_[\'Emilia_ZH_EN\']/model_600000.pt", required=True)
    parser.add_argument("--save_sample_rate", default=24000, type=int)

    parser.add_argument("--seed", default=None, type=int)

    parser.add_argument("--nfestep", default=32, type=int)
    parser.add_argument("--odemethod", default="euler")
    parser.add_argument("--cfg_strength", default=2.0, type=float)
    parser.add_argument("--swaysampling", default=-1, type=float)

    parser.add_argument("--testset", required=True)
    parser.add_argument("--no_ref_audio", action="store_true", help="To denoise without reference audio, default to False")
    parser.add_argument("--no_noisy_ref_audio", action="store_true", help="To use TTS(token to speech) mode, default to False")
    parser.add_argument("--no_token", action="store_true", help="To ignore the token prompt, default to False")

    args = parser.parse_args()

    fm_model = args.fm_model
    fm_ckpt_file = args.fm_ckpt_file

    seed = args.seed
    exp_name = f"{fm_model}"
    # exp_name = args.expname
    # ckpt_step = args.ckptstep

    nfe_step = args.nfestep
    ode_method = args.odemethod
    sway_sampling_coef = args.swaysampling

    testset = args.testset

    infer_batch_size = 1  # max frames. 1 for ddp single inference (recommended)
    cfg_strength = args.cfg_strength
    speed = 1.0
    use_truth_duration = False
    no_ref_audio = args.no_ref_audio
    no_noisy_ref_audio = args.no_noisy_ref_audio
    no_token = args.no_token

    fm_model_cfg = OmegaConf.load(str(files("sense").joinpath(f"configs/{fm_model}.yaml")))
    fm_model_cls = get_class(f"sense.model.{fm_model_cfg.model.backbone}")
    fm_model_arc = fm_model_cfg.model.arch
    mel_spec_cfg = fm_model_cfg.model.mel_spec
    mel_spec_type = mel_spec_cfg.mel_spec_type

    # dataset_name = model_cfg.datasets.name
    tokenizer = fm_model_cfg.model.tokenizer

    # mel_spec_type = fm_model_cfg.model.mel_spec.mel_spec_type
    # target_sample_rate = fm_model_cfg.model.mel_spec.target_sample_rate
    # n_mel_channels = fm_model_cfg.model.mel_spec.n_mel_channels
    # hop_length = fm_model_cfg.model.mel_spec.hop_length
    # win_length = fm_model_cfg.model.mel_spec.win_length
    # n_fft = fm_model_cfg.model.mel_spec.n_fft

    if testset == "ls_pc_test_clean":
        metalst = rel_path + "/data/librispeech_pc_test_clean_cross_sentence.lst"
        librispeech_test_clean_path = "<SOME_PATH>/LibriSpeech/test-clean"  # test-clean path
        metainfo = get_librispeech_test_clean_metainfo(metalst, librispeech_test_clean_path)

    elif testset == "seedtts_test_zh":
        metalst = rel_path + "/data/seedtts_testset/zh/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)

    elif testset == "seedtts_test_en":
        metalst = rel_path + "/data/seedtts_testset/en/meta.lst"
        metainfo = get_seedtts_testset_metainfo(metalst)
    
    elif testset == "dns_challenge_no_reverb":
        metalst = rel_path + "/test_dataset/dns_challenge/no_reverb/no_reverb_meta.lst"
        metainfo = get_dns_challenge_metainfo(metalst)
    
    elif testset == "dns_challenge_with_reverb":
        metalst = rel_path + "/test_dataset/dns_challenge/with_reverb/with_reverb_meta_dereverb.lst"
        metainfo = get_dns_challenge_metainfo(metalst)

    # path to save genereted wavs

    output_dir = (
        f"{rel_path}/"
        f"results/{exp_name}/{testset}/"
        f"seed{seed}_{ode_method}_nfe{nfe_step}_{mel_spec_type}"
        f"{f'_ss{sway_sampling_coef}' if sway_sampling_coef else ''}"
        f"_cfg{cfg_strength}_speed{speed}"
        f"{'_gt-dur' if use_truth_duration else ''}"
        f"{'_no-ref-audio' if no_ref_audio else ''}"
    )

    # -------------------------------------------------#

    prompts_all = get_inference_prompt(
        metainfo,
        # speed=speed,
        # tokenizer=tokenizer,
        target_sample_rate=mel_spec_cfg.target_sample_rate,
        n_mel_channels=mel_spec_cfg.n_mel_channels,
        hop_length=mel_spec_cfg.hop_length,
        mel_spec_type=mel_spec_type,
        # target_rms=target_rms,
        # use_truth_duration=use_truth_duration,
        infer_batch_size=infer_batch_size,
    )

    # print(f'ref_mels: {prompts_all[0][1].shape}')
    # print(f'ref_mel_lens: {prompts_all[0][2]}')

    # Vocoder model
    local = True
    if mel_spec_type == "vocos":
        vocoder_local_path = "src/sense/checkpoints/vocos-mel-24khz"
    elif mel_spec_type == "bigvgan":
        vocoder_local_path = "src/sense/checkpoints/bigvgan_v2_24khz_100band_256x"
    elif mel_spec_type == "bigvgan_qwen":
        vocoder_local_path = "src/sense/checkpoints/bigvgan_qwen"
    else:
        raise ValueError(f"Unsupported mel_spec_type: {mel_spec_type}")
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=local, local_path=vocoder_local_path)

    # Tokenizer
    # print(f'tokenizer: {tokenizer}')
    # if tokenizer == "s3tokenizer_v1":
    #     vocab_char_map = None
    #     vocab_size = 4096
    # vocab_char_map, vocab_size = get_tokenizer(dataset_name, tokenizer)

    # Model
    print(f'Loading FM model {fm_model} from {fm_ckpt_file} ...')
    fm_model = load_fm_model(
        fm_model_cls, fm_model_arc, mel_spec_cfg, fm_ckpt_file, device=device
    )

    if not os.path.exists(output_dir) and accelerator.is_main_process:
        os.makedirs(output_dir)

    # start batch inference
    accelerator.wait_for_everyone()
    start = time.time()

    with accelerator.split_between_processes(prompts_all) as prompts:
        for prompt in tqdm(prompts, disable=not accelerator.is_local_main_process):
            utts, ref_wavs, ref_mels, ref_mel_lens, noisy_wavs, noisy_mels, noisy_mel_lens, noisy_rms_list, total_mel_lens = prompt
            ref_wavs = ref_wavs.to(device)
            ref_mels = ref_mels.to(device)
            ref_mel_lens = torch.tensor(ref_mel_lens, dtype=torch.long).to(device)
            noisy_wavs = noisy_wavs.to(device)
            noisy_mels = noisy_mels.to(device)
            noisy_mel_lens = torch.tensor(noisy_mel_lens, dtype=torch.long).to(device)
            total_mel_lens = torch.tensor(total_mel_lens, dtype=torch.long).to(device)

            assert noisy_mels.shape[0] == 1, "noisy_mels should be of shape (1, T, bins)"

            tokenizer = s3tokenizer.load_model("speech_tokenizer_v1", download_root="src/S3Tokenizer/s3tokenizer/ckpts").cuda()  # or "speech_tokenizer_v1_25hz speech_tokenizer_v2_25hz"
            # ref_audio_wav = s3tokenizer.load_audio(ref_wavs).to(device)
            resampler = torchaudio.transforms.Resample(mel_spec_cfg.target_sample_rate, 16000).to(device)
            ref_wavs = resampler(ref_wavs)[0]
            ref_mel = s3tokenizer.log_mel_spectrogram(ref_wavs)
            mels = ref_mel
            mel_lens = torch.tensor([ref_mel.shape[-1]])
            codes, codes_lens = tokenizer.quantize(mels.cuda(), mel_lens.cuda())
            ref_tokens = codes[0, :codes_lens[0].item()]
            gen_tokens = ref_tokens

            total_tokens = torch.cat([ref_tokens, gen_tokens], dim=0).unsqueeze(0)

            # Inference
            with torch.inference_mode():
                generated, _ = fm_model.sample(
                    cond=ref_mels,
                    cond_noisy=noisy_mels,
                    text=total_tokens,  # [1, L]
                    duration=total_mel_lens,
                    lens=ref_mel_lens,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    no_ref_audio=no_ref_audio,
                    no_noisy_ref_audio=no_noisy_ref_audio,
                    no_token=no_token,
                    seed=seed,
                )
                # Final result
                for i, gen in enumerate(generated):
                    target_rms = noisy_rms_list[i]
                    if not no_ref_audio:
                        gen = gen[ref_mel_lens[i] : total_mel_lens[i], :].unsqueeze(0)
                    else:
                        gen = gen.unsqueeze(0)
                    gen_mel_spec = gen.permute(0, 2, 1).to(torch.float32)
                    if mel_spec_type == "vocos":
                        generated_wave = vocoder.decode(gen_mel_spec).cpu()
                    elif mel_spec_type == "bigvgan":
                        generated_wave = vocoder(gen_mel_spec).squeeze(0).cpu()
                    elif mel_spec_type == "bigvgan_qwen":
                        generated_wave = vocoder(gen_mel_spec).unsqueeze(0).cpu()

                    if hasattr(mel_spec_cfg, 'output_sample_rate'):
                        output_sample_rate = mel_spec_cfg.output_sample_rate
                    else:
                        output_sample_rate = mel_spec_cfg.target_sample_rate
                    resampler = torchaudio.transforms.Resample(output_sample_rate, args.save_sample_rate)
                    generated_wave = resampler(generated_wave)
                    rms = torch.sqrt(torch.mean(torch.square(generated_wave)))
                    generated_wave = generated_wave / rms * target_rms
                    # if ref_rms_list[i] < target_rms:
                    #     generated_wave = generated_wave * ref_rms_list[i] / target_rms
                    torchaudio.save(f"{output_dir}/{utts[i]}.wav", generated_wave, args.save_sample_rate)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        timediff = time.time() - start
        print(f"Done batch inference in {timediff / 60:.2f} minutes.")


if __name__ == "__main__":
    main()
