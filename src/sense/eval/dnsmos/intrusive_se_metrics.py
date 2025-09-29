# from https://github.com/urgent-challenge/urgent2024_challenge/blob/main/evaluation_metrics/calculate_intrusive_se_metrics.py

import argparse
import concurrent.futures
import glob
import os
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pesq import PesqError, pesq
from pystoi import stoi
import soxr
import fast_bss_eval


METRICS = ("PESQ", "ESTOI")


class ComputeMetrics:
    def __init__(self):
        pass

    def estoi_metric(self, ref, inf, fs=16000):
        return stoi(ref, inf, fs_sig=fs, extended=True)

    def pesq_metric(self, ref, inf, fs=8000):
        assert ref.shape == inf.shape
        if fs == 8000:
            mode = "nb"
        elif fs == 16000:
            mode = "wb"
        elif fs > 16000:
            mode = "wb"
            ref = soxr.resample(ref, fs, 16000)
            inf = soxr.resample(inf, fs, 16000)
            fs = 16000
        else:
            raise ValueError(f"sample rate must be 8000 or 16000+ for PESQ evaluation, but got {fs}")
        pesq_score = pesq(fs, ref, inf, mode=mode, on_error=PesqError.RETURN_VALUES)
        if pesq_score == PesqError.NO_UTTERANCES_DETECTED:
            logging.warning(f"[PESQ] Error: No utterances detected. Skipping this sample.")
            return np.nan
        return pesq_score

    def __call__(self, ref_path, inf_path, metrics=METRICS):
        ref, fs = sf.read(ref_path, dtype="float32", always_2d=False)
        inf, fs2 = sf.read(inf_path, dtype="float32", always_2d=False)
        
        if ref.ndim == 2:
            ref = ref[:, 0]
        if inf.ndim == 2:
            inf = inf[:, 0]
        
        if fs2 != fs:
            inf = soxr.resample(inf, fs2, fs)
        
        if len(inf) > len(ref):
            inf = inf[:len(ref)]
        elif len(inf) < len(ref):
            inf = np.pad(inf, (0, len(ref) - len(inf)), mode='constant')
        
        assert ref.shape == inf.shape, f"Shape mismatch: ref {ref.shape} vs inf {inf.shape}"

        scores = {}
        for metric in metrics:
            if metric == "PESQ":
                scores[metric] = self.pesq_metric(ref, inf, fs=fs)
            elif metric == "ESTOI":
                scores[metric] = self.estoi_metric(ref, inf, fs=fs)
            else:
                raise NotImplementedError(metric)
        return scores

def calculate_intrusive_score(gt_dir, testset_dir, csv_path=None, json_path=None):
    compute_metrics = ComputeMetrics()

    ref_files = glob.glob(os.path.join(gt_dir, "*.wav"))
    ref_files.extend(glob.glob(os.path.join(gt_dir, "*.flac")))
    ref_files.extend(glob.glob(os.path.join(gt_dir, "*.mp3")))
    inf_files = glob.glob(os.path.join(testset_dir, "*.wav"))
    inf_files.extend(glob.glob(os.path.join(testset_dir, "*.flac")))
    inf_files.extend(glob.glob(os.path.join(testset_dir, "*.mp3")))

    data_pairs = [(os.path.basename(f), os.path.join(gt_dir, os.path.basename(f)), os.path.join(testset_dir, os.path.basename(f))) for f in ref_files if os.path.basename(f) in {os.path.basename(p) for p in inf_files}]

    print()
    print(f'Found {len(data_pairs)} pairs!')

    rows = []
    for uid, ref_path, inf_path in tqdm(data_pairs):
        try:
            scores = compute_metrics(ref_path, inf_path)
        except Exception as exc:
            print(f'{uid} generated an exception: {exc}')
        else:
            row = {'filename': uid}
            row.update(scores)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Add average row for each metric
    avg_row = {'filename': 'Average'}
    for metric in METRICS:
        avg_row[metric] = round(df[metric].mean(), 3)
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    if csv_path:
        df.to_csv(csv_path, index=False)

        # Dump averages to JSON
        avg_scores = {}
        json_path = json_path if json_path else os.path.join(os.path.dirname(csv_path), 'results.json')
        import json
        if os.path.exists(json_path):
            avg_scores.update(json.load(open(json_path)))
        avg_scores.update({metric: round(float(round(df[metric].mean(), 3)), 3) for metric in METRICS})
        with open(json_path, 'w') as f:
            json.dump(avg_scores, f, indent=4)
    else:
        print(df.describe())