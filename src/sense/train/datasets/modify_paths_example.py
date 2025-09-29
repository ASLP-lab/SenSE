#!/usr/bin/env python3
"""
简化的路径修改示例脚本
直接修改指定的raw.arrow文件中的音频路径前缀
"""

import os
from pathlib import Path
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


def modify_libritts_noisy_paths():
    """
    修改LibriTTS_noisy数据集中的音频路径前缀
    """
    
    # 配置路径
    old_prefix = "/home/node25_tmpdata/data/LibriTTS/"
    new_prefix = "/home/work_nfs16/fe_data/data/"
    
    # 根据您的数据集配置自动查找raw.arrow文件
    # 这里假设按照原脚本的输出目录结构
    dataset_name = "LibriTTS_noisy_100_360_500_custom"
    base_dir = Path("/home/work_nfs11/xcli/work/F5-TTS/data") / dataset_name
    input_arrow = base_dir / "raw.arrow"
    output_arrow = base_dir / "raw_modified.arrow"
    
    print(f"正在处理: {dataset_name}")
    print(f"输入文件: {input_arrow}")
    print(f"输出文件: {output_arrow}")
    print(f"路径前缀修改: {old_prefix} -> {new_prefix}")
    
    if not input_arrow.exists():
        print(f"错误: 找不到输入文件 {input_arrow}")
        print("请确保已经运行了prepare_libritts_noisy.py脚本生成raw.arrow文件")
        return
    
    # 加载数据集
    print("正在加载数据集...")
    dataset = Dataset.from_file(str(input_arrow))
    print(f"数据集总样本数: {len(dataset)}")
    
    # 修改路径
    modified_data = []
    modified_count = 0
    
    for item in tqdm(dataset, desc="修改音频路径"):
        modified_item = item.copy()
        
        # 修改主音频路径
        if 'audio_path' in modified_item:
            original_path = modified_item['audio_path']
            if original_path.startswith(old_prefix):
                modified_item['audio_path'] = original_path.replace(old_prefix, new_prefix, 1)
                modified_count += 1
        
        # 修改带噪音频路径
        if 'noisy_audio_path' in modified_item:
            original_noisy_path = modified_item['noisy_audio_path']
            if original_noisy_path.startswith(old_prefix):
                modified_item['noisy_audio_path'] = original_noisy_path.replace(old_prefix, new_prefix, 1)
        
        modified_data.append(modified_item)
    
    print(f"共修改了 {modified_count} 个音频路径")
    
    # 保存修改后的数据
    print("正在保存修改后的数据...")
    with ArrowWriter(path=str(output_arrow)) as writer:
        for item in tqdm(modified_data, desc="写入新文件"):
            writer.write(item)
    
    print("修改完成！")
    
    # 验证结果
    print("\n验证前3个样本:")
    new_dataset = Dataset.from_file(str(output_arrow))
    for i in range(min(3, len(new_dataset))):
        sample = new_dataset[i]
        print(f"\n样本 {i+1}:")
        print(f"  音频路径: {sample['audio_path']}")
        if 'noisy_audio_path' in sample:
            print(f"  带噪音频路径: {sample['noisy_audio_path']}")
        print(f"  时长: {sample['duration']:.2f}秒")
        print(f"  文本: {sample['text'][:50]}...")
    
    # 提示如何使用修改后的文件
    print(f"\n✅ 修改完成！")
    print(f"新的raw.arrow文件保存在: {output_arrow}")
    print(f"您可以将原文件备份后，用新文件替换:")
    print(f"  mv {input_arrow} {input_arrow}.backup")
    print(f"  mv {output_arrow} {input_arrow}")


if __name__ == "__main__":
    modify_libritts_noisy_paths() 
