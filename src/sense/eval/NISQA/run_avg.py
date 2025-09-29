# 文件路径: src/f5_tts/eval/NISQA/calc_nisqa_avg.py
import argparse
import pandas as pd

def process_csv(csv_path):
    """
    处理CSV文件并追加平均值行
    :param csv_path: 输入CSV文件路径
    :return: 平均值（保留3位小数）
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        
        # 验证必要列
        required_columns = ['deg', 'mos_pred', 'model']
        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV文件缺少必要列，需要包含：deg, mos_pred, model")
            
        # 计算平均值
        avg = round(df['mos_pred'].mean(), 3)
        
        # 构建新行数据
        new_row = {
            'deg': '[average]',
            'mos_pred': avg,
            'model': df['model'].iloc[0] if not df.empty else ''
        }
        
        # 追加新行
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # 保存文件（覆盖原文件）
        df.to_csv(csv_path, index=False)
        return avg
        
    except pd.errors.EmptyDataError:
        raise ValueError("CSV文件为空或格式错误")
    except PermissionError:
        raise PermissionError("文件被占用，请关闭后重试")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='计算并追加NISQA平均MOS值')
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件路径')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示处理详情')
    
    args = parser.parse_args()
    
    try:
        average = process_csv(args.input)
        if args.verbose:
            print(f"文件已更新: {args.input}")
            print(f"新增平均值行: [average], {average}")
        print(f"MOS平均值: {average}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        exit(1)