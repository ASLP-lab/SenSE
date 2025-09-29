#!/usr/bin/env python3
"""
脚本用于修改raw.arrow文件中的音频路径前缀
将"/home/node25_tmpdata/data/LibriTTS/"修改为"/home/work_nfs16/fe_data/data/"
"""

import argparse
import os
from pathlib import Path

from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def modify_audio_paths(input_arrow_path, output_arrow_path, old_prefix, new_prefix):
    """
    修改Arrow文件中的音频路径前缀
    
    Args:
        input_arrow_path: 输入的raw.arrow文件路径
        output_arrow_path: 输出的raw.arrow文件路径
        old_prefix: 需要替换的旧路径前缀
        new_prefix: 新的路径前缀
    """
    
    print(f"正在加载数据集: {input_arrow_path}")
    dataset = Dataset.from_file(input_arrow_path)
    
    print(f"数据集总样本数: {len(dataset)}")
    print(f"将路径前缀从 '{old_prefix}' 修改为 '{new_prefix}'")
    
    # 准备修改后的数据
    modified_data = []
    
    for i, item in enumerate(tqdm(dataset, desc="处理音频路径")):
        modified_item = item.copy()
        
        # 修改主音频路径
        if 'audio_path' in modified_item:
            audio_path = modified_item['audio_path']
            if audio_path.startswith(old_prefix):
                modified_item['audio_path'] = audio_path.replace(old_prefix, new_prefix, 1)
            
        # 修改带噪音频路径（如果存在）
        if 'degraded_audio_path' in modified_item:
            noisy_path = modified_item['degraded_audio_path']
            if noisy_path.startswith(old_prefix):
                modified_item['degraded_audio_path'] = noisy_path.replace(old_prefix, new_prefix, 1)
        
        modified_data.append(modified_item)
    
    print(f"正在保存修改后的数据到: {output_arrow_path}")
    
    # 确保输出目录存在
    output_dir = Path(output_arrow_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 写入新的Arrow文件
    with ArrowWriter(path=output_arrow_path) as writer:
        for item in tqdm(modified_data, desc="写入新的Arrow文件"):
            writer.write(item)
    
    print("路径修改完成！")
    
    # 验证一些样本
    print("\n验证前几个样本的路径修改:")
    new_dataset = Dataset.from_file(output_arrow_path)
    for i in range(min(3, len(new_dataset))):
        sample = new_dataset[i]
        print(f"样本 {i+1}:")
        print(f"  音频路径: {sample['audio_path']}")
        if 'degraded_audio_path' in sample:
            print(f"  带噪音频路径: {sample['degraded_audio_path']}")
        print(f"  文本: {sample['text'][:50]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="修改raw.arrow文件中的音频路径前缀")
    parser.add_argument("--input", "-i", required=True, 
                       help="输入的raw.arrow文件路径")
    parser.add_argument("--output", "-o", required=True,
                       help="输出的raw.arrow文件路径")
    parser.add_argument("--old_prefix", default="/home/node25_tmpdata/data/LibriTTS/",
                       help="需要替换的旧路径前缀 (默认: /home/node25_tmpdata/data/LibriTTS/)")
    parser.add_argument("--new_prefix", default="/home/work_nfs16/fe_data/data/",
                       help="新的路径前缀 (默认: /home/work_nfs16/fe_data/data/)")
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 执行路径修改
    modify_audio_paths(args.input, args.output, args.old_prefix, args.new_prefix)


if __name__ == "__main__":
    main() 