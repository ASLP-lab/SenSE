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

import os

def extract_audio_paths_from_lst(lst_file):
    """
    读取 .lst 文件中的路径，并返回绝对路径的列表。

    参数:
        lst_file (str): .lst 文件路径，文件中每行是一个音频文件的路径。

    返回:
        List[str]: 音频文件的绝对路径列表。
    """
    audio_paths = []
    base_dir = os.path.dirname(os.path.abspath(lst_file))
    
    with open(lst_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            path = line.split(" ")[0]
            if not line or line.startswith("#"):
                continue
            abs_path = os.path.abspath(os.path.join(base_dir, path))
            if os.path.isfile(abs_path):
                audio_paths.append(abs_path)
            else:
                print(f"[警告] 找不到文件: {abs_path}")
    
    return audio_paths

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
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio)/fs) - INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_ovr_seg_raw = []
        predicted_mos_ovr_seg = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx * hop_len_samples): int((idx + INPUT_LENGTH) * hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis, :]
            oi = {'input_1': input_features}
            mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0][2]  # 第3个输出是 OVRL
            _, _, mos_ovr = self.get_polyfit_val(0, 0, mos_ovr_raw)  # 只保留 mos_ovr

            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_ovr_seg.append(mos_ovr)

        clip_dict = {
            'filename': fpath,
            'len_in_sec': actual_audio_len / fs,
            'sr': fs,
            'num_hops': num_hops,
            'OVRL_raw': np.mean(predicted_mos_ovr_seg_raw),
            'OVRL': np.mean(predicted_mos_ovr_seg),
        }
        return clip_dict

def main(args):
    import json

    # 加载模型
    p808_model_path = os.path.join(args.dnsmos_path, 'model_v8.onnx')
    primary_model_path = os.path.join(args.dnsmos_path, 'sig_bak_ovr.onnx')
    compute_score = ComputeScore(primary_model_path, p808_model_path)

    # 从 lst 文件提取音频路径
    clips = extract_audio_paths_from_lst(args.lst_file)
    desired_fs = SAMPLING_RATE
    rows = []

    # 并发处理每个音频文件
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_clip = {executor.submit(compute_score, clip, desired_fs): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_clip)):
            clip = future_to_clip[future]
            try:
                data = future.result()
                rows.append(data)
            except Exception as exc:
                print(f"[错误] 处理文件失败: {clip} - {exc}")

    # 转为 DataFrame
    df = pd.DataFrame(rows)

    # add average row on OVRL_raw SIG_raw BAK_raw OVRL SIG BAK P808_MOS
    avg_row = {
        'filename': 'Average',
        'len_in_sec': 'N/A',
        'sr': SAMPLING_RATE,
        'num_hops': df['num_hops'].mean(),
        'OVRL_raw': df['OVRL_raw'].mean(),
        'OVRL': round(df['OVRL'].mean(), 3),
    }
    df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
    if args.csv_path:
        csv_path = args.csv_path
        df.to_csv(csv_path)
        # dump avg to json
        import json
        json_path = csv_path.replace('.csv', '.json')
        avg_row = {
            'OVRL': round(df['OVRL'].mean(), 3),
        }
        with open(json_path, 'w') as f:
            json.dump(avg_row, f, indent=4)
    else:
        print(df.describe())

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', "--lst_file", required=True, help='Path to the .lst file containing audio paths')
    parser.add_argument('-o', "--csv_path", default=None, help='Dir to the csv that saves the results')
    parser.add_argument('-d', "--dnsmos_path", default='src/f5_tts/eval/dnsmos/DNSMOS', help='Path to the DNSMOS model directory')
    
    args = parser.parse_args()

    main(args)