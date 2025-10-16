# Vocoder

We use the open-source BigVGAN as our vocoder. Here, we include the external repository as a Git submodule.

Please add the following code at the beginning of the bigvgan.py file in the BigVGAN repository to ensure correct imports:

```python
import sys
import os
bigvgan_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(bigvgan_path)
```