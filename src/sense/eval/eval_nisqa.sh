#! /bin/bash

# source activate nisqa

exp_dir=results/U3SE_LLM_Base_whisper_U3SE_CFM_Base_ablation/dns_challenge_hardset/seed0_euler_nfe8_bigvgan_ss-1.0_cfg0.5_speed1.0

python src/u3se/eval/NISQA/run_predict.py \
    --mode predict_dir \
    --pretrained_model src/u3se/eval/NISQA/weights/nisqa_mos_only.tar \
    --data_dir $exp_dir \
    --num_workers 0 \
    --bs 10 \
    --output_dir $exp_dir \

python src/u3se/eval/NISQA/run_avg.py --input $exp_dir/_NISQA_results.csv