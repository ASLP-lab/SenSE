import argparse
import json
import os
import sys


sys.path.append(os.getcwd())

import multiprocessing as mp
from importlib.resources import files

import numpy as np

from sense.eval.utils_eval import get_speech_enhancement_test, run_asr_wer, run_sim, run_wesim, run_speechbertscore


rel_path = str(files("sense").joinpath("../../"))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval_task", type=str, default="wer", choices=["sim", "wesim", "wer", "speechbertscore"])
    parser.add_argument("-l", "--lang", type=str, default="en")
    parser.add_argument("-m", "--metalst", type=str, required=True)
    parser.add_argument("-g", "--gen_wav_dir", type=str, required=True)
    parser.add_argument("--eval_ground_truth", action="store_true", help="Evaluate ground truth")
    parser.add_argument("-n", "--gpu_nums", type=int, default=8, help="Number of GPUs to use")
    parser.add_argument("--local", action="store_true", help="Use local custom checkpoint directory")
    return parser.parse_args()


def main():
    mp.set_start_method("spawn", force=True)
    args = get_args()
    eval_task = args.eval_task
    lang = args.lang
    gen_wav_dir = args.gen_wav_dir
    metalst = args.metalst
    eval_ground_truth = args.eval_ground_truth

    gpus = list(range(args.gpu_nums))
    test_set = get_speech_enhancement_test(metalst, gen_wav_dir, gpus, eval_ground_truth)

    local = args.local
    if local:  # use local custom checkpoint dir
        asr_ckpt_dir = "src/sense/checkpoints/faster-whisper-large-v3"
    else:
        asr_ckpt_dir = ""  # auto download to cache dir
    wavlm_ckpt_dir = "src/sense/checkpoints/wavlm_finetune/wavlm_large_finetune.pth"

    # --------------------------------------------------------------------------

    full_results = []
    metrics = []

    if eval_task == "wer":
        # rank, sub_test_set = test_set[0]
        # run_asr_wer((rank, lang, sub_test_set, asr_ckpt_dir))
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, lang, sub_test_set, asr_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_asr_wer, args)
            for r in results:
                full_results.extend(r)
    elif eval_task == "sim":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_sim, args)
            for r in results:
                full_results.extend(r)
    elif eval_task == "wesim":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_wesim, args)
            for r in results:
                full_results.extend(r)
    elif eval_task == "speechbertscore":
        with mp.Pool(processes=len(gpus)) as pool:
            args = [(rank, sub_test_set, wavlm_ckpt_dir) for (rank, sub_test_set) in test_set]
            results = pool.map(run_speechbertscore, args)
            for r in results:
                full_results.extend(r)
    else:
        raise ValueError(f"Unknown metric type: {eval_task}")

    result_path = f"{gen_wav_dir}/_{eval_task}_results.jsonl"
    error = 0
    with open(result_path, "w") as f:
        for line in full_results:
            if eval_task == "wer":
                if line["wer"] < 1.0:
                    metrics.append(line["wer"])
                else:
                    error += 1
            else:
                metrics.append(line[eval_task])
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        metric = round(np.mean(metrics), 5)
        f.write(f"\n{eval_task.upper()}: {metric}\n")

    print(f"\nTotal {len(metrics)} samples")
    print(f"{eval_task.upper()}: {metric}")
    print(f"{eval_task.upper()} results saved to {result_path}")
    if eval_task == "wer":
        print(f"Error num: {error}")


if __name__ == "__main__":
    main()
