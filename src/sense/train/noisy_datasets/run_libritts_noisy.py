#!/usr/bin/env python3
"""
Example script to run LibriTTS noisy dataset preparation.
"""

import subprocess
import sys

def main():
    # Default parameters
    cmd = [
        sys.executable, 
        "src/f5_tts/train/noisy_datasets/prepare_libritts_noisy.py",
        "--input_dir", "/home/node25_tmpdata/data/LibriTTS/LibriTTS/",
        "--output_dir", "/home/node25_tmpdata/data/LibriTTS/LibriTTS_noisy/",
        "--noise_scp", "/home/node25_tmpdata/xcli/F5-TTS/data/noise_all_filtered.scp",
        "--snr_min", "-5.0",
        "--snr_max", "5.0",
        "--subsets", "train-clean-360",
        "--max_workers", "32",
        "--seed", "42"
    ]
    #    "--subsets", "train-clean-100", "train-clean-360", "train-other-500",
    
    print("Running LibriTTS noisy dataset preparation...")
    print("Command:", " ".join(cmd))
    print()
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print("Dataset preparation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: Dataset preparation failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)

if __name__ == "__main__":
    main() 