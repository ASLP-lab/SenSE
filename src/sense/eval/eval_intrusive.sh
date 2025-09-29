#!/bin/bash

# FRCRN
# tfgridnet
# MP-SENet
# CleanMel_mask
# voicefixer
# SELM
# GenSE
# LLaSE-G1
# U3-SE

# exp_dir=/home/node56_tmpdata/xcli/work/baseline/dns_challenge_no_reverb/FlowSE
# exp_dir=/home/node56_tmpdata/xcli/work/baseline/dns_challenge_hardset/FlowSE
# exp_dir=results/U3SE_LLM_Base_llama_U3SE_CFM_Base_nosample/dns_challenge_no_reverb/seed0_euler_nfe8_bigvgan_ss-1_cfg0.5_speed1.0_no-ref-audio
# exp_dir=results/U3SE_LLM_Base_whisper_U3SE_CFM_Base_115w_ft/dns_challenge_gsr/seed0_euler_nfe8_bigvgan_ss-1.0_cfg0.5_speed1.0_no-ref-audio
# exp_dir=/home/node55_tmpdata/xcli/work/anyenhance-v1-ccf-aatc/decode/anyenhance_40w_dns_gsr
exp_dir=results/U3SE_LLM_Base_whisper_U3SE_CFM_Base_ablation/dns_challenge_gsr/seed0_euler_nfe8_bigvgan_ss-1.0_cfg0.5_speed1.0

python src/u3se/eval/intrusive_se_metrics.py \
    --testset_dir $exp_dir \
    --gt_dir /home/node56_tmpdata/xcli/work/baseline/dns_challenge_gsr/clean \
    --json_path $exp_dir/_intrusive.json \
    --csv_path $exp_dir/_intrusive.csv