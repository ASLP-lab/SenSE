import os
import json
from pathlib import Path
from tqdm import tqdm
from datasets.arrow_writer import ArrowWriter


def load_token_data(token_file_path):
    """
    加载token文件，返回一个key->code的字典
    """
    token_dict = {}
    if not os.path.exists(token_file_path):
        raise FileNotFoundError(f"Token文件不存在: {token_file_path}")

    print(f"加载token文件: {token_file_path}")
    with open(token_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="加载token"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                token_entry = json.loads(line)
                if "key" in token_entry and "code" in token_entry:
                    token_dict[token_entry["key"]] = token_entry["code"]
                else:
                    print(f"⚠️ 第{line_num}行缺少 key 或 code 字段")
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_num}行解析错误: {e}")
    print(f"✅ 共加载 {len(token_dict)} 个 token 条目")
    return token_dict


def process_custom_jsonl(jsonl_path, token_dict):
    """
    处理自定义 jsonl 文件，返回处理结果
    """
    print(f"\n读取数据文件: {jsonl_path}")
    results = []
    durations = []
    missing_token_count = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="处理条目"), 1):
            try:
                obj = json.loads(line)
                audio_id = obj["id"]
                audio_path = obj["path"]
                duration = obj["duration"]

                # 基本过滤
                if duration < 0.4 or duration > 60:
                    continue

                audio_token = token_dict.get(audio_id, [])
                if not audio_token:
                    missing_token_count += 1
                    continue

                results.append({
                    "audio_path": audio_path,
                    "duration": duration,
                    "audio_token": audio_token
                })
                durations.append(duration)

            except Exception as e:
                print(f"⚠️ 第{line_num}行解析失败: {e}")

    print(f"\n✅ 成功处理样本数: {len(results)}")
    print(f"❌ 缺失token的样本数: {missing_token_count}")
    print(f"⏱️ 总时长: {sum(durations) / 3600:.2f} 小时")
    return results, durations


def save_dataset(results, durations, save_dir):
    """
    保存为 arrow 和 duration.json 文件
    """
    os.makedirs(save_dir, exist_ok=True)
    arrow_path = os.path.join(save_dir, "raw.arrow")
    duration_path = os.path.join(save_dir, "duration.json")

    print(f"\n💾 写入 arrow 文件: {arrow_path}")
    with ArrowWriter(path=arrow_path) as writer:
        for entry in tqdm(results, desc="写入 arrow"):
            writer.write(entry)

    print(f"💾 写入 duration 文件: {duration_path}")
    with open(duration_path, 'w', encoding='utf-8') as f:
        json.dump({"duration": durations}, f, ensure_ascii=False)


def main():
    # ==== 输入路径配置 ====
    jsonl_path = "data/aslp_hq_data/aslp_hq_data.jsonl"
    token_file_path = "/home/node56_tmpdata/xcli/data/ASLP_HQ_s3tokenizer_v1_50hz/merged_tokens.jsonl"
    save_dir = "data/aslp_hq_data/ASLP_HQ_s3tokenizer_v1_50hz"

    # ==== 处理流程 ====
    token_dict = load_token_data(token_file_path)
    results, durations = process_custom_jsonl(jsonl_path, token_dict)
    save_dataset(results, durations, save_dir)


if __name__ == "__main__":
    main()
