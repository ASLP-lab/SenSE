# Inference

The model checkpoints can be obtained from [huggingface](https://huggingface.co/ASLP-lab/SenSE).

Currently, the model supports **up to 30 seconds** of audio per inference, counting both the prompt audio and the degraded speech. For longer inputs, please split the audio into shorter segments before inference.

### ⚠️ Precision Issue in Whisper LayerNorm

Since the model uses **half-precision (FP16)**, you may encounter a **dtype mismatch error** in the Whisper library’s `LayerNorm` during inference.  
If this happens, please modify the source code of `LayerNorm` in your installed Whisper package as follows:

File path:
anaconda3/envs/sense/lib/python3.10/site-packages/whisper/model.py

Original implementation:
```python
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)
```

Modified implementation:
```python
class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps
        ).type(x.dtype)
```

### Vocoder Download

We use the open-source BigVGAN model as our vocoder. Please read the vocoder section’s [guidance](https://github.com/ASLP-lab/SenSE/blob/main/src/sense/model/vocoder/README.md) first to ensure correct inference.

### Batch Inference

You can perform batch inference using the code below.
Please modify the `test_dir` and `prompt_dir` parameters accordingly.
The degraded speech files in `test_dir` and the prompt audio files in `prompt_dir` should have matching filenames (one-to-one correspondence).
If no prompt audio is provided, set `prompt_dir=None` and add the `--no_ref_audio` flag.

```
accelerate launch src/sense/eval/eval_infer_batch.py \
    --seed 0 \
    --llm_model SenSE_LLM_Base \
    --llm_ckpt_file "/path/to/llm_ckpt" \
    --fm_model SenSE_CFM_Base \
    --fm_ckpt_file "/path/to/cfm_ckpt" \
    --exp_name test \
    --save_sample_rate 16000 \
    --testset custom \
    --test_dir "/path/to/degraded" \
    --prompt_dir None \
    --nfestep 8 \
    --cfg_strength 0.5 \
    --no_ref_audio
    # --swaysampling -1 \
```

### Single Inference

You can perform single-sample inference by specifying one degraded speech file and (optionally) one prompt audio file.

If you want to use a locally installed BigVGAN vocoder, please enable the `--load_vocoder_from_local` flag. Place the checkpoint file under the `src/sense/checkpoints` directory, for example:

```
src/sense/checkpoints/bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt
```

```
python src/sense/infer/infer.py \
    --config infer_config.toml \
    --llm_model SenSE_LLM_Base \
    --llm_ckpt_file "/path/to/llm_ckpt" \
    --fm_model SenSE_CFM_Base \
    --fm_ckpt_file "/path/to/cfm_ckpt" \
    --load_vocoder_from_local \
    --vocoder_name bigvgan \
    --nfestep 32 \
    --cfg_strength 0.5 \
    --swaysampling -1 \
    --output_dir tests/ \
    --output_file test.wav \
    --save_sample_rate 24000 \
    --ref_audio tests/noisy.wav \
    --noisy_audio tests/noisy.wav \
    --no_ref_audio  # if true, the reference audio is not used
```
