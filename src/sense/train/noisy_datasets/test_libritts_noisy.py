#!/usr/bin/env python3
"""
Test script for LibriTTS noisy dataset preparation.
Tests on a small subset of data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent.parent.parent.parent
sys.path.append(str(current_dir))

from src.f5_tts.train.datasets.prepare_libritts_noisy import (
    load_noise_list, 
    adjust_noise_length,
    add_noise_to_audio,
    copy_directory_structure
)

def test_load_noise_list():
    """Test loading noise list"""
    print("Testing load_noise_list...")
    noise_scp_path = "/home/node25_tmpdata/xcli/F5-TTS/data/noise_all_filtered.scp"
    
    if not os.path.exists(noise_scp_path):
        print(f"Warning: Noise file not found at {noise_scp_path}")
        return False
    
    noise_list = load_noise_list(noise_scp_path)
    print(f"Loaded {len(noise_list)} noise files")
    
    if noise_list:
        print(f"First noise file: {noise_list[0]}")
        return True
    return False

def test_audio_functions():
    """Test audio processing functions"""
    print("Testing audio processing functions...")
    
    # Test with dummy data
    import numpy as np
    
    # Create dummy audio
    clean_audio = np.random.randn(16000)  # 1 second at 16kHz
    noise_audio_short = np.random.randn(8000)  # 0.5 seconds
    noise_audio_long = np.random.randn(32000)  # 2 seconds
    
    # Test adjust_noise_length
    adjusted_short = adjust_noise_length(noise_audio_short, 16000, 16000)
    adjusted_long = adjust_noise_length(noise_audio_long, 16000, 16000)
    
    print(f"Original clean audio length: {len(clean_audio)}")
    print(f"Short noise adjusted length: {len(adjusted_short)}")
    print(f"Long noise adjusted length: {len(adjusted_long)}")
    
    # Test add_noise_to_audio
    noisy_audio = add_noise_to_audio(clean_audio, adjusted_short, snr_db=10.0)
    print(f"Noisy audio length: {len(noisy_audio)}")
    
    return True

def test_directory_structure():
    """Test directory structure copying"""
    print("Testing directory structure copying...")
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as tmp_dir:
        src_dir = Path(tmp_dir) / "src"
        dst_dir = Path(tmp_dir) / "dst"
        
        # Create test structure
        (src_dir / "subdir1" / "subdir2").mkdir(parents=True)
        (src_dir / "test.txt").write_text("test content")
        (src_dir / "test.wav").write_text("fake audio")
        (src_dir / "subdir1" / "another.txt").write_text("another file")
        
        # Copy structure
        copy_directory_structure(src_dir, dst_dir)
        
        # Check results
        assert (dst_dir / "subdir1" / "subdir2").exists()
        assert (dst_dir / "test.txt").exists()
        assert not (dst_dir / "test.wav").exists()  # Should be excluded
        assert (dst_dir / "subdir1" / "another.txt").exists()
        
        print("Directory structure copying test passed!")
        return True

def main():
    """Run all tests"""
    print("=== Testing LibriTTS Noisy Dataset Preparation ===\n")
    
    tests = [
        test_load_noise_list,
        test_audio_functions,
        test_directory_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                print(f"✓ {test.__name__} passed\n")
                passed += 1
            else:
                print(f"✗ {test.__name__} failed\n")
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}\n")
            failed += 1
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("All tests passed! The script should work correctly.")
    else:
        print("Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main() 