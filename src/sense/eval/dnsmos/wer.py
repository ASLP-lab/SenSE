import os
import glob
import pandas as pd
import json
from tqdm import tqdm
import torch
from torchmetrics import WordErrorRate, CharErrorRate
import whisper

def preprocess_text(text):
    # text = text.replace(" ", "")
    text = text.replace(".", "")
    text = text.replace("'", "")
    text = text.replace("-", "")
    text = text.replace(",", "")
    text = text.replace("!", "")
    text = text.lower()
    return text.strip()

def calculate_wer_score(gt_dir, testset_dir, csv_path, device='cpu', json_path=None):
    gt_files = glob.glob(os.path.join(gt_dir, "*.wav"))
    gt_files.extend(glob.glob(os.path.join(gt_dir, "*.flac")))
    gt_files.extend(glob.glob(os.path.join(gt_dir, "*.mp3")))
    
    test_files = glob.glob(os.path.join(testset_dir, "*.wav"))
    test_files.extend(glob.glob(os.path.join(testset_dir, "*.flac")))
    test_files.extend(glob.glob(os.path.join(testset_dir, "*.mp3")))

    wer_metric = WordErrorRate().to(device)
    cer_metric = CharErrorRate().to(device)
    model = whisper.load_model("large")
    model = model.to(device)
    
    gt_files_dict = {os.path.basename(f): f for f in gt_files}

    rows = []
    for test_file in tqdm(test_files):
        filename = os.path.basename(test_file)
        if filename in gt_files_dict:
            gt_file = gt_files_dict[filename]
            test_file = test_file

            result_gt = model.transcribe(gt_file)
            result_pred = model.transcribe(test_file)
            if result_gt['language'] == 'zh':
                result_gt = model.transcribe(gt_file, initial_prompt="以下是普通话的句子")
            if result_pred['language'] == 'zh':
                result_pred = model.transcribe(test_file, initial_prompt="以下是普通话的句子")
            
            content_gt = result_gt["text"]
            content_pred = result_pred["text"]

            content_gt = preprocess_text(content_gt)
            content_pred = preprocess_text(content_pred)
            
            if result_gt['language'] == 'zh' and result_pred['language'] == 'zh':
                wer_score = cer_metric(content_pred, content_gt).item()
                rows.append({"filename": filename, "WER": wer_score, "mode": "CER", "language": "zh", "gt": content_gt, "pred": content_pred})
            else:
                wer_score = wer_metric(content_pred, content_gt).item()
                rows.append({"filename": filename, "WER": wer_score, "mode": "WER", "language": result_gt['language'], "gt": content_gt, "pred": content_pred})
        else:
            print(f"Warning: No matching GT file found for {filename}")

    df = pd.DataFrame(rows)
    
    if csv_path:
        df.to_csv(csv_path, index=False)

        json_path = json_path if json_path else os.path.join(os.path.dirname(csv_path), 'results.json')
        avg_row = {}
        if os.path.exists(json_path):
            print(f"Updating {json_path} with new results")
            avg_row.update(json.load(open(json_path)))
        avg_row['WER'] = float(round(df['WER'].mean(), 3))

        with open(json_path, 'w') as f:
            json.dump(avg_row, f, indent=4)
    else:
        print(df.describe())

    print(f"Results saved to {csv_path} and {json_path}")