import os
import sys


sys.path.append(os.getcwd())

import json
from concurrent.futures import ProcessPoolExecutor
from importlib.resources import files
from pathlib import Path

import soundfile as sf
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def load_token_data(token_file_path):
    """
    加载合并的token文件数据
    
    Args:
        token_file_path (str): token文件路径
        
    Returns:
        dict: 以key为索引的token字典
    """
    token_dict = {}
    
    if not os.path.exists(token_file_path):
        print(f"警告: Token文件不存在: {token_file_path}")
        return token_dict
    
    print(f"加载token文件: {token_file_path}")
    
    with open(token_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading tokens"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                token_entry = json.loads(line)
                if "key" in token_entry and "code" in token_entry:
                    token_dict[token_entry["key"]] = token_entry["code"]
                else:
                    print(f"警告: 第{line_num}行格式不正确，缺少key或code字段")
            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行JSON解析错误: {e}")
    
    print(f"成功加载 {len(token_dict)} 个token条目")
    return token_dict


def deal_with_audio_dir(audio_dir, token_dict):
    sub_result, durations = [], []
    vocab_set = set()
    # audio_lists = list(audio_dir.rglob("*.wav"))
    audio_lists = [f for f in audio_dir.rglob("*.wav") if not f.stem.endswith("_noisy")]

    for line in audio_lists:
        noisy_path = str(line)[:-4] + "_noisy.wav"
        text_path = line.with_suffix(".normalized.txt")
        text = open(text_path, "r").read().strip()
        duration = sf.info(line).duration
        if duration < 0.4 or duration > 30:
            continue
        
        # 获取文件名（不含扩展名）作为key
        audio_key = Path(line).stem
        noisy_key = Path(noisy_path).stem
        
        # 从token字典中获取对应的token
        audio_token = token_dict.get(audio_key, [])
        noisy_token = token_dict.get(noisy_key, [])
        
        # 如果找不到对应的token，记录警告但继续处理
        if not audio_token:
            print(f"警告: 未找到音频文件的token: {audio_key}")
        if not noisy_token:
            print(f"警告: 未找到噪声音频文件的token: {noisy_key}")
        
        sub_result.append({
            "audio_path": str(line), 
            "noisy_audio_path": str(noisy_path), 
            "text": text, 
            "duration": duration,
            "audio_token": audio_token,
            "noisy_token": noisy_token
        })
        durations.append(duration)
        vocab_set.update(list(text))
    return sub_result, durations, vocab_set


def main():
    result = []
    duration_list = []
    text_vocab_set = set()
    
    # 加载token数据
    token_dict = load_token_data(token_file_path)
    if not token_dict:
        print("错误: 无法加载token数据，程序退出")
        return

    # process raw data
    executor = ProcessPoolExecutor(max_workers=max_workers)
    futures = []

    for subset in tqdm(SUB_SET):
        dataset_path = Path(os.path.join(dataset_dir, subset))
        [
            futures.append(executor.submit(deal_with_audio_dir, audio_dir, token_dict))
            for audio_dir in dataset_path.iterdir()
            if audio_dir.is_dir()
        ]
    for future in tqdm(futures, total=len(futures)):
        sub_result, durations, vocab_set = future.result()
        result.extend(sub_result)
        duration_list.extend(durations)
        text_vocab_set.update(vocab_set)
    executor.shutdown()

    # save preprocessed dataset to disk
    if not os.path.exists(f"{save_dir}"):
        os.makedirs(f"{save_dir}")
    print(f"\nSaving to {save_dir} ...")

    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # dup a json separately saving duration in case for DynamicBatchSampler ease
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # vocab map, i.e. tokenizer
    with open(f"{save_dir}/vocab.txt", "w") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    # 统计token匹配情况
    audio_token_found = sum(1 for item in result if item.get("audio_token"))
    noisy_token_found = sum(1 for item in result if item.get("noisy_token"))
    both_tokens_found = sum(1 for item in result if item.get("audio_token") and item.get("noisy_token"))
    
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")
    print(f"\nToken匹配统计:")
    print(f"  找到audio_token的样本: {audio_token_found}/{len(result)} ({audio_token_found/len(result)*100:.1f}%)")
    print(f"  找到noisy_token的样本: {noisy_token_found}/{len(result)} ({noisy_token_found/len(result)*100:.1f}%)")
    print(f"  两个token都找到的样本: {both_tokens_found}/{len(result)} ({both_tokens_found/len(result)*100:.1f}%)")


if __name__ == "__main__":
    max_workers = 36

    tokenizer = "char"  # "pinyin" | "char"

    SUB_SET = ["train-clean-100", "train-clean-360", "train-other-500"]
    dataset_dir = "/home/node56_tmpdata/xcli/data/LibriTTS_noisy"
    dataset_name = f"LibriTTS_noisy_token_{'_'.join(SUB_SET)}_{tokenizer}".replace("train-clean-", "").replace("train-other-", "")
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    
    # 设置token文件路径
    token_file_path = "/home/node56_tmpdata/xcli/data/LibriTTS_noisy_token/merged_tokens"
    
    print(f"\nPrepare for {dataset_name}, will save to {save_dir}")
    print(f"Token file: {token_file_path}\n")
    main()

    # For LibriTTS_100_360_500_char, sample count: 354218
    # For LibriTTS_100_360_500_char, vocab size is: 78
    # For LibriTTS_100_360_500_char, total 554.09 hours
