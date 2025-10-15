# SenSE: Semantic-Aware High-Fidelity Universal Speech Enhancement

[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.24708)
[![demo](https://img.shields.io/badge/GitHub-Demo%20page-orange.svg)](https://stellanli.github.io/SenSE-demo/)
[![models](https://img.shields.io/badge/🤗-Models-yellow)](https://huggingface.co/ASLP-lab/SenSE)

<p align="center">
    <img src="figures/SenSE.png" width="900"/>
<p>

## Installation

### Create a separate environment if needed

```bash
# Create a python 3.10 conda env (you could also use virtualenv)
conda create -n sense python=3.10
conda activate sense
```

### Install PyTorch with matched device

```bash
> # Install pytorch with your CUDA version, e.g.
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```


### Then you can install the environment as follows:

```bash
cd SenSE
# git submodule update --init --recursive  # (optional, if need > bigvgan)
pip install -e .
```


## Inference

Please read the [Inference Guidance](https://github.com/ASLP-lab/SenSE/blob/main/src/sense/infer/REAMD.md)
 before running the inference code to ensure correct results.

Our models are available at https://huggingface.co/ASLP-lab/SenSE.

```bash
accelerate launch src/sense/eval/eval_infer_batch.py \
    --seed 0 \
    --llm_model SenSE_LLM_Base \
    --llm_ckpt_file "/path/to/llm_ckpt" \
    --fm_model SenSE_CFM_Base \
    --fm_ckpt_file "/path/to/cfm_ckpt" \
    --exp_name evaluation \
    --save_sample_rate 16000 \
    --testset dns_challenge_no_reverb \
    --nfestep 8 \
    --cfg_strength 0.5 \
    --swaysampling -1 \
    # --no_ref_audio
```


## Training

```bash
# for LLM stage training:
accelerate launch src/sense/train/train_llm.py \
    --config-name SenSE_LLM_Base.yaml

# for CFM stage training:
accelerate launch src/sense/train/train.py \
    --config-name SenSE_CFM_Base.yaml
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{li2025sense,
  title={SenSE: Semantic-Aware High-Fidelity Universal Speech Enhancement},
  author={Li, Xingchen and Xie, Hanke and Wang, Ziqian and Zhang, Zihan and Xiao, Longshuai and Xie, Lei},
  journal={arXiv preprint arXiv:2509.24708},
  year={2025}
}
```

## Acknowledgements

We sincerely thank all collaborators and Prof. [Shuai Wang](https://github.com/wsstriving) for their valuable contributions to this work and the accompanying paper.

In addition, this implementation is based on [F5-TTS](https://github.com/SWivid/F5-TTS). We appreciae their excellent work.

## Contact Us

If you are interested in leaving a message to our research team, feel free to email `lixingchen0126@gmail.com`.
<p align="center">
    <a href="http://www.nwpu-aslp.org/">
        <img src="figures/ASLP.jpg" width="400"/>
    </a>
</p>
