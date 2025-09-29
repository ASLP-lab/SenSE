import os
import torchaudio
import torchaudio.transforms as T
import torch
from tqdm import tqdm

def resample_wav_dir(input_dir, output_dir, target_sr):
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有.wav文件
    for filename in tqdm(os.listdir(input_dir)):
        if not filename.lower().endswith('.wav'):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # 读取音频
        waveform, original_sr = torchaudio.load(input_path)

        if original_sr == target_sr:
            # 直接复制
            torchaudio.save(output_path, waveform, target_sr)
            continue

        # 重采样（使用 CPU）
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform_resampled = resampler(waveform)

        # 保存新音频
        torchaudio.save(output_path, waveform_resampled, target_sr)

    print(f"重采样完成，共处理 {len(os.listdir(input_dir))} 个文件。")

# 示例：使用方法
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量重采样wav音频文件")
    parser.add_argument("--input_dir", type=str, required=True, help="输入目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--target_sr", type=int, required=True, help="目标采样率（例如16000）")
    args = parser.parse_args()

    resample_wav_dir(args.input_dir, args.output_dir, args.target_sr)
