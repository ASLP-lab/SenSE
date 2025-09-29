import argparse
import concurrent.futures
import glob
import os

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeScore:
    def __init__(self, primary_model_path, p808_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path)
        self.p808_onnx_sess = ort.InferenceSession(p808_model_path)
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr):
        p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
        p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
        p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, orig_sr=input_fs, target_sr=fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH)+1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict

def main(args):
    models = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []
    p808_model_path = os.path.join(args.dnsmos_path, 'model_v8.onnx')
    primary_model_path = os.path.join(args.dnsmos_path, 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(args.testset_dir, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip, desired_fs): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)
    # add average row on OVRL_raw SIG_raw BAK_raw OVRL SIG BAK P808_MOS
    avg_row = {
        'filename': 'Average',
        'len_in_sec': df['len_in_sec'].mean(),
        'sr': SAMPLING_RATE,
        'num_hops': df['num_hops'].mean(),
        'OVRL_raw': df['OVRL_raw'].mean(),
        'SIG_raw': df['SIG_raw'].mean(),
        'BAK_raw': df['BAK_raw'].mean(),
        'OVRL': round(df['OVRL'].mean(), 3),
        'SIG': round(df['SIG'].mean(), 3),
        'BAK': round(df['BAK'].mean(), 3),
        'P808_MOS': round(df['P808_MOS'].mean(), 3)
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    if args.csv_path:
        csv_path = args.csv_path
        df.to_csv(csv_path)
        # dump avg to json
        import json
        json_path = csv_path.replace('.csv', '.json')
        avg_row = {
            'SIG': round(df['SIG'].mean(), 3),
            'BAK': round(df['BAK'].mean(), 3),
            'OVRL': round(df['OVRL'].mean(), 3),
        }
        with open(json_path, 'w') as f:
            json.dump(avg_row, f, indent=4)
    else:
        print(df.describe())

def calculate_dnsmos_score(testset_dir, dnsmos_path, csv_path=None, json_path=None):
    models = glob.glob(os.path.join(testset_dir, "*"))
    audio_clips_list = []
    p808_model_path = os.path.join(dnsmos_path, 'model_v8.onnx')
    primary_model_path = os.path.join(dnsmos_path, 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(testset_dir, "*.wav"))
    clips.extend(glob.glob(os.path.join(testset_dir, "*.flac")))
    clips.extend(glob.glob(os.path.join(testset_dir, "*.mp3")))
    desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(testset_dir, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip, desired_fs): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)
    # add average row on OVRL_raw SIG_raw BAK_raw OVRL SIG BAK P808_MOS
    avg_row = {
        'filename': 'Average',
        'len_in_sec': 'N/A',
        'sr': SAMPLING_RATE,
        'num_hops': df['num_hops'].mean(),
        'OVRL_raw': df['OVRL_raw'].mean(),
        'SIG_raw': df['SIG_raw'].mean(),
        'BAK_raw': df['BAK_raw'].mean(),
        'OVRL': round(df['OVRL'].mean(), 3),
        'SIG': round(df['SIG'].mean(), 3),
        'BAK': round(df['BAK'].mean(), 3),
        'P808_MOS': round(df['P808_MOS'].mean(), 3)
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    if csv_path:
        csv_path = csv_path
        df.to_csv(csv_path)
        # dump avg to json
        import json
        json_path = json_path if json_path else os.path.join(os.path.dirname(csv_path), '_results.json')
        avg_row = {}
        if os.path.exists(json_path):
            avg_row.update(json.load(open(json_path)))
        avg_row.update({
            'SIG': round(df['SIG'].mean(), 3),
            'BAK': round(df['BAK'].mean(), 3),
            'OVRL': round(df['OVRL'].mean(), 3),
        })
        with open(json_path, 'w') as f:
            json.dump(avg_row, f, indent=4)
    else:
        print(f"SIG: {round(df['SIG'].mean(), 3)}")
        print(f"BAK: {round(df['BAK'].mean(), 3)}")
        print(f"OVRL: {round(df['OVRL'].mean(), 3)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.', 
                        help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('-d', "--dnsmos_path", default='src/sense/eval/dnsmos/DNSMOS', help='Path to the DNSMOS model directory')
    
    args = parser.parse_args()

    main(args)