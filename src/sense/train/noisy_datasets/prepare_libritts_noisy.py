#!/usr/bin/env python3
"""
Script to create noisy versions of LibriTTS dataset by adding background noise.

This script processes the original LibriTTS dataset and creates a new dataset
with noise-augmented audio files while preserving the original directory structure.
"""

import os
import sys
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import shutil
import subprocess


def load_noise_list(noise_scp_path):
    """
    Load noise file list from .scp file.
    """
    noise_list = []
    with open(noise_scp_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                # Split from right to handle paths with spaces
                # The last element should be the duration
                parts = line.rsplit(' ', 1)  # Split only on the last space
                if len(parts) == 2:
                    audio_path = parts[0].strip()
                    try:
                        duration = float(parts[1].strip())
                        noise_list.append((audio_path, duration))
                    except ValueError:
                        print(f"Warning: Invalid duration format at line {line_num}: {line}")
                        continue
                else:
                    print(f"Warning: Invalid format at line {line_num}: {line}")
                    continue
    return noise_list

def adjust_noise_length(noise_audio, noise_sr, target_length_samples):
    """
    Adjust noise length to match target audio length.
    If noise is longer, truncate it.
    If noise is shorter, repeat it.
    """
    noise_length = len(noise_audio)
    
    if noise_length >= target_length_samples:
        # Truncate noise to match target length
        start_idx = random.randint(0, noise_length - target_length_samples)
        return noise_audio[start_idx:start_idx + target_length_samples]
    else:
        # Repeat noise to match target length
        repeat_times = (target_length_samples // noise_length) + 1
        repeated_noise = np.tile(noise_audio, repeat_times)
        return repeated_noise[:target_length_samples]


def add_noise_to_audio(clean_audio, noise_audio, snr_db):
    """
    Add noise to clean audio with specified SNR.
    
    Args:
        clean_audio: Clean speech signal
        noise_audio: Noise signal (same length as clean_audio)
        snr_db: Signal-to-noise ratio in dB
    
    Returns:
        Noisy audio signal
    """
    # Calculate signal and noise power
    signal_power = np.mean(clean_audio ** 2)
    noise_power = np.mean(noise_audio ** 2)
    
    # Calculate noise scaling factor based on desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_scaling = np.sqrt(signal_power / (noise_power * snr_linear + 1e-6))

    if np.isnan(noise_scaling) or np.isinf(noise_scaling):
        noise_scaling = 0.0
    
    # Scale noise and add to signal
    scaled_noise = noise_audio * noise_scaling
    noisy_audio = clean_audio + scaled_noise
    
    return noisy_audio


def process_audio_file(args):
    """Process a single audio file to create noisy version"""
    wav_file, noise_list, snr_range, target_dir, original_dir = args
    
    try:
        # Load clean audio
        clean_audio, sr = sf.read(wav_file)
        
        # Randomly select a noise file
        noise_path, noise_duration = random.choice(noise_list)
        
        # Check if noise file exists
        if not os.path.exists(noise_path):
            print(f"Warning: Noise file not found: {noise_path}")
            return False
        
        # Load noise audio
        try:
            noise_audio, noise_sr = sf.read(noise_path)
        except Exception as e:
            print(f"Warning: Cannot read noise file {noise_path}: {e}")
            return False
        
        # Resample noise if necessary
        if noise_sr != sr:
            # Simple resampling by linear interpolation
            from scipy.signal import resample
            target_length = int(len(noise_audio) * sr / noise_sr)
            noise_audio = resample(noise_audio, target_length)
        
        # Convert to mono if stereo
        if len(noise_audio.shape) > 1:
            noise_audio = np.mean(noise_audio, axis=1)
        if len(clean_audio.shape) > 1:
            clean_audio = np.mean(clean_audio, axis=1)
        
        # Adjust noise length to match clean audio
        target_length = len(clean_audio)
        adjusted_noise = adjust_noise_length(noise_audio, sr, target_length)
        
        # Randomly select SNR from range
        snr_db = random.uniform(snr_range[0], snr_range[1])
        
        # Add noise to clean audio
        noisy_audio = add_noise_to_audio(clean_audio, adjusted_noise, snr_db)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(noisy_audio))
        if max_val > 1.0:
            noisy_audio = noisy_audio / max_val * 0.95
        
        # Create output path
        relative_path = wav_file.relative_to(Path(original_dir))
        output_path = target_dir / relative_path
        
        # Create noisy filename
        noisy_filename = output_path.stem + "_noisy" + output_path.suffix
        noisy_output_path = output_path.parent / noisy_filename
        
        # Create output directory if it doesn't exist
        noisy_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save noisy audio
        sf.write(noisy_output_path, noisy_audio, sr)
        
        return True
        
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return False


def copy_directory_structure(src_dir, dst_dir, exclude_extensions=None, use_rsync=True):
    """Copy directory structure and files efficiently, skipping files that already exist"""
    # if exclude_extensions is None:
    #     exclude_extensions = {'.wav'}
    
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    
    # Create destination directory if it doesn't exist
    dst_path.mkdir(parents=True, exist_ok=True)
    
    if use_rsync and shutil.which('rsync'):
        # Use rsync for fast copying (Linux/Unix systems)
        print("Using rsync for fast directory copying...")
        try:
            # rsync options:
            # -a: archive mode (preserves permissions, timestamps, etc.)
            # -v: verbose
            # --ignore-existing: skip files that already exist in destination
            # --exclude: exclude audio files (we'll process them separately)
            cmd = [
                'rsync', '-av', '--ignore-existing', '--progress',
                f'{src_path}/', f'{dst_path}/'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Rsync completed successfully")
            if result.stdout:
                # Count transferred files from rsync output
                lines = result.stdout.strip().split('\n')
                transferred_files = [line for line in lines if not line.endswith('/') and line.strip() and not line.startswith('sending')]
                print(f"Transferred {len(transferred_files)} files")
            return
            
        except subprocess.CalledProcessError as e:
            print(f"Rsync failed: {e}")
            print("Falling back to Python copying...")
        except Exception as e:
            print(f"Rsync error: {e}")
            print("Falling back to Python copying...")
    
    # Fallback to optimized Python copying
    print("Using optimized Python copying...")
    copied_count = 0
    skipped_count = 0
    
    # Use os.scandir for better performance than os.walk
    def copy_tree_optimized(src, dst):
        nonlocal copied_count, skipped_count
        
        try:
            dst.mkdir(parents=True, exist_ok=True)
            
            with os.scandir(src) as entries:
                for entry in entries:
                    src_path = Path(entry.path)
                    dst_path = dst / entry.name
                    
                    if entry.is_dir():
                        # Recursively copy subdirectories
                        copy_tree_optimized(src_path, dst_path)
                    elif entry.is_file():
                        # Check if file already exists
                        if dst_path.exists():
                            skipped_count += 1
                            continue
                        
                        try:
                            # Use shutil.copy2 for metadata preservation
                            shutil.copy2(src_path, dst_path)
                            copied_count += 1
                        except Exception as e:
                            print(f"Warning: Failed to copy {src_path} to {dst_path}: {e}")
                            
        except Exception as e:
            print(f"Error processing directory {src}: {e}")
    
    # Start copying with progress indication
    with tqdm(desc="Copying directory structure") as pbar:
        copy_tree_optimized(src_path, dst_path)
        pbar.update(1)
    
    print(f"Directory structure copy complete: {copied_count} files copied, {skipped_count} files skipped (already exist)")

def main():
    parser = argparse.ArgumentParser(description='Create noisy LibriTTS dataset')
    parser.add_argument('--input_dir', type=str, 
                       default='/home/node25_tmpdata/data/LibriTTS/LibriTTS/',
                       help='Input LibriTTS directory')
    parser.add_argument('--output_dir', type=str,
                       default='/home/node25_tmpdata/data/LibriTTS/LibriTTS_noisy/',
                       help='Output directory for noisy dataset')
    parser.add_argument('--noise_scp', type=str,
                       default='/home/node25_tmpdata/xcli/F5-TTS/data/noise_all_filtered.scp',
                       help='Path to noise .scp file')
    parser.add_argument('--snr_min', type=float, default=0.0,
                       help='Minimum SNR in dB')
    parser.add_argument('--snr_max', type=float, default=20.0,
                       help='Maximum SNR in dB')
    parser.add_argument('--subsets', nargs='+', 
                       default=['train-clean-100', 'train-clean-360', 'train-other-500'],
                       help='LibriTTS subsets to process')
    parser.add_argument('--max_workers', type=int, default=16,
                       help='Number of parallel workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load noise list
    print(f"Loading noise list from {args.noise_scp}")
    noise_list = load_noise_list(args.noise_scp)
    print(f"Loaded {len(noise_list)} noise files")
    
    if not noise_list:
        print("Error: No noise files found in the .scp file!")
        return
    
    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing LibriTTS dataset from {input_dir}")
    print(f"Output will be saved to {output_dir}")
    print(f"SNR range: {args.snr_min} to {args.snr_max} dB")
    
    # First, copy directory structure and non-audio files
    print("Copying directory structure and files...")
    # copy_directory_structure(input_dir, output_dir)
    
    # Process each subset
    total_processed = 0
    total_files = 0
    
    for subset in args.subsets:
        subset_dir = input_dir / subset
        if not subset_dir.exists():
            print(f"Warning: Subset directory does not exist: {subset_dir}")
            continue
        
        print(f"\nProcessing subset: {subset}")
        
        # Find all .wav files in the subset
        wav_files = list(subset_dir.rglob("*.wav"))
        total_files += len(wav_files)
        
        print(f"Found {len(wav_files)} audio files in {subset}")
        
        if not wav_files:
            continue
        
        # Prepare arguments for parallel processing
        snr_range = (args.snr_min, args.snr_max)
        process_args = [(wav_file, noise_list, snr_range, output_dir, input_dir) 
                       for wav_file in wav_files]
        
        # Process files in parallel
        success_count = 0
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_audio_file, arg): arg[0] 
                             for arg in process_args}
            
            # Process completed tasks with progress bar
            with tqdm(total=len(wav_files), desc=f"Processing {subset}") as pbar:
                for future in as_completed(future_to_file):
                    wav_file = future_to_file[future]
                    try:
                        success = future.result()
                        if success:
                            success_count += 1
                    except Exception as e:
                        print(f"Error processing {wav_file}: {e}")
                    pbar.update(1)
        
        print(f"Successfully processed {success_count}/{len(wav_files)} files in {subset}")
        total_processed += success_count
    
    print(f"\n=== Processing Complete ===")
    print(f"Total files processed: {total_processed}/{total_files}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main() 