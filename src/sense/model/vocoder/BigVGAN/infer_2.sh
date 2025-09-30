#! /bin/bash

python inference.py \
    --checkpoint_file checkpoints/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt \
    --input_wavs_dir /home/node56_tmpdata/xcli/work/X-Codec-2.0/evaluation/ground_truth \
    --output_dir evaluation/bigvgan_v2_24khz_100band_256x \
    # --use_cuda_kernel