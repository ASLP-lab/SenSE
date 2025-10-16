# Vocoder

We use the open-source [BigVGAN](https://github.com/NVIDIA/BigVGAN) as our vocoder. Here, we include the external repository as a Git submodule.

Please first download the open-source BigVGAN model from [HuggingFace](https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x) and place it in the `src/sense/checkpoints` directory.

Please add the following code at the beginning of the bigvgan.py file in the BigVGAN repository to ensure correct imports:

```python
import sys
import os
bigvgan_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(bigvgan_path)
```