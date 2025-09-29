# 噪声数据集预处理脚本
# 处理.scp格式的音频文件列表，将音频路径和时长信息保存为Arrow格式

import os
import sys

sys.path.append(os.getcwd())

import json
from pathlib import Path
from importlib.resources import files

from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def read_scp_file(scp_file_path):
    """
    读取.scp文件，解析音频路径和时长信息
    
    Args:
        scp_file_path (str): .scp文件路径
        
    Returns:
        list: 包含音频路径和时长的字典列表
        list: 时长列表
    """
    if not os.path.exists(scp_file_path):
        raise FileNotFoundError(f"SCP文件不存在: {scp_file_path}")
    
    print(f"读取SCP文件: {scp_file_path}")
    
    result = []
    durations = []
    invalid_lines = 0
    
    with open(scp_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing SCP file"), 1):
            line = line.strip()
            if not line or line.startswith('#'):  # 跳过空行和注释行
                continue
            
            # 按空格分割，获取路径和时长
            parts = line.split()
            if len(parts) < 2:
                print(f"警告: 第{line_num}行格式不正确，跳过: {line}")
                invalid_lines += 1
                continue
            
            try:
                # 路径可能包含空格，所以取最后一个部分作为时长
                duration = float(parts[-1])
                audio_path = ' '.join(parts[:-1])  # 重新组合路径部分
                
                # 验证时长范围
                # if duration < 0.1 or duration > 60:  # 过滤过短或过长的音频
                #     print(f"警告: 第{line_num}行时长异常({duration}s)，跳过: {audio_path}")
                #     invalid_lines += 1
                #     continue
                
                # 验证文件路径
                if not os.path.exists(audio_path):
                    print(f"警告: 第{line_num}行音频文件不存在，跳过: {audio_path}")
                    invalid_lines += 1
                    continue
                
                result.append({
                    "audio_path": audio_path,
                    "duration": duration
                })
                durations.append(duration)
                
            except ValueError as e:
                print(f"警告: 第{line_num}行时长解析错误({e})，跳过: {line}")
                invalid_lines += 1
                continue
    
    print(f"成功读取 {len(result)} 个音频条目")
    if invalid_lines > 0:
        print(f"跳过 {invalid_lines} 行无效数据")
    
    return result, durations


def process_scp_files(scp_files):
    """
    处理多个SCP文件
    
    Args:
        scp_files (list): SCP文件路径列表
        
    Returns:
        list: 合并后的音频信息列表
        list: 合并后的时长列表
    """
    all_results = []
    all_durations = []
    
    for scp_file in scp_files:
        print(f"\n处理文件: {scp_file}")
        result, durations, sr = read_scp_file(scp_file)
        all_results.extend(result)
        all_durations.extend(durations)
        print(f"从 {Path(scp_file).name} 读取 {len(result)} 个条目")
    
    return all_results, all_durations


def save_to_arrow(result, duration_list, save_dir, dataset_name):
    """
    保存数据到Arrow格式文件
    
    Args:
        result (list): 音频信息列表
        duration_list (list): 时长列表
        save_dir (str): 保存目录
        dataset_name (str): 数据集名称
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"\n保存到 {save_dir} ...")
    
    # 保存为Arrow格式
    with ArrowWriter(path=f"{save_dir}/raw.arrow") as writer:
        for item in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(item)
    
    # 单独保存时长信息
    with open(f"{save_dir}/duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)
    
    print(f"\n数据集 {dataset_name} 处理完成:")
    print(f"  样本数量: {len(result):,}")
    print(f"  总时长: {sum(duration_list) / 3600:.2f} 小时")
    print(f"  平均时长: {sum(duration_list) / len(duration_list):.2f} 秒")
    print(f"  最短时长: {min(duration_list):.2f} 秒")
    print(f"  最长时长: {max(duration_list):.2f} 秒")


def main():
    """主函数"""
    # 处理SCP文件
    if isinstance(scp_files, str):
        scp_files_list = [scp_files]
    else:
        scp_files_list = scp_files
    
    print(f"开始处理 {len(scp_files_list)} 个SCP文件")
    
    # 读取和处理所有SCP文件
    result, duration_list = process_scp_files(scp_files_list)
    
    if not result:
        print("错误: 没有找到任何有效的音频数据")
        return
    
    # 保存数据
    save_to_arrow(result, duration_list, save_dir, dataset_name)


if __name__ == "__main__":
    # 配置参数
    dataset_name = "Noise_Dataset"
    
    # SCP文件路径配置
    # 单个文件示例:
    # scp_files = "/path/to/your/audio_list.scp"
    
    # 多个文件示例:
    scp_files = [
        "/home/work_nfs11/xcli/datalist/noise_tr_with_duration.scp",
        "/home/work_nfs11/xcli/work/F5-TTS/data/noise_all_filtered.scp"
        # 在这里添加更多的SCP文件路径
    ]
    
    # 保存目录
    save_dir = str(files("f5_tts").joinpath("../../")) + f"/data/{dataset_name}"
    
    print(f"\n准备处理数据集: {dataset_name}")
    print(f"输出目录: {save_dir}\n")
    
    main()
