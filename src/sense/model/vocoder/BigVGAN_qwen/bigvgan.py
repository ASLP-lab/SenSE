import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import sys
import os
bigvgan_qwen_path= os.path.dirname(os.path.abspath(__file__))
sys.path.append(bigvgan_qwen_path)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import soundfile as sf
import torchaudio
from meldataset import get_mel_spectrogram, MAX_WAV_VALUE
from env import AttrDict
from mel_processing import mel_spectrogram_torch_aslp

import os
import librosa
import argparse
import json
from scipy.io.wavfile import write


class Qwen2_5OmniBigVGANConfig():
    r"""
    This is the configuration class to store the configuration of the Qwen2_5OmniToken2WavBigVGAN module used in the Qwen2.5-Omni-Token2Wav model.
    It defines the architecture of the BigVGAN model, which is used for converting mel-spectrograms to waveforms.

    Args:
        mel_dim (`int`, *optional*, defaults to 80):
            The dimension of the mel-spectrogram.
        upsample_initial_channel (`int`, *optional*, defaults to 1536):
            The number of channels in the initial upsampling layer.
        resblock_kernel_sizes (`List[int]`, *optional*, defaults to `[3, 7, 11]`):
            A list of kernel sizes for each residual block.
        resblock_dilation_sizes (`List[List[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`):
            A list of dilation sizes for each residual block.
        upsample_rates (`List[int]`, *optional*, defaults to `[5, 3, 2, 2, 2, 2]`):
            A list of upsampling rates for each upsampling layer.
        upsample_kernel_sizes (`List[int]`, *optional*, defaults to `[11, 7, 4, 4, 4, 4]`):
            A list of kernel sizes for each upsampling layer.
    """

    model_type = "qwen2_5_omni_bigvgan"

    def __init__(
        self,
        num_mels=80,
        num_freq=1025,
        n_fft=1024,
        hop_size=160,
        win_size=640,
        sampling_rate=16000,
        fmin=0,
        fmax=8000,
        upsample_initial_channel=1536,
        resblock_kernel_sizes=[3, 7, 11],
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        upsample_rates=[5, 3, 2, 2, 2, 2],
        upsample_kernel_sizes=[11, 7, 4, 4, 4, 4],
    ):
        self.num_mels = num_mels
        self.num_freq = num_freq
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.sampling_rate = sampling_rate
        self.fmin = fmin
        self.fmax = fmax
        self.upsample_initial_channel = upsample_initial_channel
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    """Generates a 1D Kaiser-windowed sinc filter.

    Args:
        cutoff (float): Normalized cutoff frequency (0 to 0.5).
        half_width (float): Transition bandwidth.
        kernel_size (int): Number of filter taps.

    Returns:
        torch.Tensor: A tensor of shape (1, 1, kernel_size) representing the filter.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    # Compute Kaiser window parameters
    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False, dtype=torch.float32)

    # Compute time indices
    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    # Compute sinc filter
    if cutoff == 0:
        return torch.zeros((1, 1, kernel_size), dtype=torch.float32)  # Ensures correct shape

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized_filter = 2 * cutoff * kaiser_window * sinc_filter

    # Normalize to ensure sum = 1 (avoid leakage of constant component)
    normalized_filter /= normalized_filter.sum()

    return normalized_filter.view(1, 1, kernel_size)


class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2

        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]

        hidden_states = F.pad(hidden_states, (self.pad, self.pad), mode="replicate")
        hidden_states = self.ratio * F.conv_transpose1d(
            hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        hidden_states = hidden_states[..., self.pad_left : -self.pad_right]

        return hidden_states


class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        cutoff = 0.5 / ratio
        half_width = 0.6 / ratio

        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")

        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = ratio
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter, persistent=False)

    def forward(self, hidden_states):
        channels = hidden_states.shape[1]
        hidden_states = F.pad(hidden_states, (self.pad_left, self.pad_right), mode="replicate")
        out = F.conv1d(hidden_states, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels)
        return out


class TorchActivation1d(nn.Module):
    def __init__(
        self,
        activation,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ):
        super().__init__()
        if not callable(activation):
            raise ValueError("Activation function must be callable")
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, hidden_states):
        hidden_states = self.upsample(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.downsample(hidden_states)

        return hidden_states


class AMPBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=(1, 3, 5),
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding=self._get_padding(kernel_size, dilation[0]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding=self._get_padding(kernel_size, dilation[1]),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding=self._get_padding(kernel_size, dilation[2]),
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=self._get_padding(kernel_size, 1),
                ),
            ]
        )

        self.num_layers = len(self.convs1) + len(self.convs2)  # total number of conv layers

        self.activations = nn.ModuleList(
            [TorchActivation1d(activation=SnakeBeta(channels)) for _ in range(self.num_layers)]
        )

    def _get_padding(self, kernel_size, dilation=1):
        return int((kernel_size * dilation - dilation) / 2)

    def forward(self, hidden_states):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for conv1, conv2, act1, act2 in zip(self.convs1, self.convs2, acts1, acts2):
            residual = hidden_states
            hidden_states = act1(hidden_states)
            hidden_states = conv1(hidden_states)
            hidden_states = act2(hidden_states)
            hidden_states = conv2(hidden_states)
            hidden_states = residual + hidden_states

        return hidden_states


class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    """

    def __init__(self, in_features, alpha=1.0):
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha = Parameter(torch.zeros(in_features) * alpha)
        self.beta = Parameter(torch.zeros(in_features) * alpha)

        self.no_div_by_zero = 0.000000001

    def forward(self, hidden_states):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        hidden_states = hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )

        return hidden_states

class Qwen2_5OmniToken2WavBigVGANModel(nn.Module):
    config_class = Qwen2_5OmniBigVGANConfig

    def __init__(self, config: Qwen2_5OmniBigVGANConfig):
        super().__init__() 
        self.h = config
        self.num_residual_blocks = len(config.resblock_kernel_sizes)
        self.num_upsample_layers = len(config.upsample_rates)

        self.conv_pre = nn.Conv1d(config.num_mels, config.upsample_initial_channel, 7, 1, padding=3)

        # Removing extra ModuleList breaks official state dict
        ups = [
            nn.ModuleList(
                [
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**layer_idx),
                        config.upsample_initial_channel // (2 ** (layer_idx + 1)),
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride) // 2,
                    )
                ]
            )
            for layer_idx, (stride, kernel_size) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes))
        ]
        self.ups = nn.ModuleList(ups)

        self.resblocks = nn.ModuleList(
            [
                AMPBlock(config.upsample_initial_channel // (2 ** (layer_idx + 1)), kernel_size, dilation)
                for layer_idx in range(self.num_upsample_layers)
                for kernel_size, dilation in zip(config.resblock_kernel_sizes, config.resblock_dilation_sizes)
            ]
        )

        self.activation_post = TorchActivation1d(
            activation=SnakeBeta(config.upsample_initial_channel // (2**self.num_upsample_layers))
        )
        self.conv_post = nn.Conv1d(
            config.upsample_initial_channel // (2**self.num_upsample_layers), 1, 7, 1, padding=3, bias=False
        )

    def normalize_spectrogram(self, spectrogram, max_value, min_db):
        return torch.clamp((2 * max_value) * ((spectrogram - min_db) / (-min_db)) - max_value, -max_value, max_value)

    def amplitude_to_db(self, amplitude, min_db_level):
        min_level = torch.exp(
            torch.tensor(min_db_level / 20.0 * np.log(10), device=amplitude.device, dtype=amplitude.dtype)
        )
        return 20 * torch.log10(torch.clamp(amplitude, min=min_level))

    def process_mel_spectrogram(self, mel_spectrogram):
        # amplitude_spectrum = torch.exp(mel_spectrogram)
        # decibel_spectrum = self.amplitude_to_db(amplitude_spectrum, -115) - 20
        return self.normalize_spectrogram(mel_spectrogram, 1, -115)

    def forward(self, mel_spectrogram):
        # processed_spectrogram = self.process_mel_spectrogram(mel_spectrogram)
        hidden_representation = self.conv_pre(mel_spectrogram)

        for layer_index in range(self.num_upsample_layers):
            hidden_representation = self.ups[layer_index][0](hidden_representation)
            residual_output = sum(
                self.resblocks[layer_index * self.num_residual_blocks + block_index](hidden_representation)
                for block_index in range(self.num_residual_blocks)
            )
            residual_output = residual_output / self.num_residual_blocks
            hidden_representation = residual_output

        hidden_representation = self.activation_post(hidden_representation)
        output_waveform = self.conv_post(hidden_representation)
        return torch.clamp(output_waveform, min=-1.0, max=1.0).squeeze().cpu()

def _denormalize(S_norm, max_abs_value=1.0, min_db=-115.0):
    return ((S_norm + max_abs_value) / (2 * max_abs_value)) * (-min_db) + min_db

def denorm_dbmel_to_logmel(dbmel_norm, min_db=-115.0, max_abs_value=1.0):
    # 反归一化
    dbmel = _denormalize(dbmel_norm, max_abs_value, min_db)
    # 恢复偏移（因为你在前面减了20）
    dbmel += 20
    # 还原为线性 Mel 幅度谱
    mel_linear = torch.pow(10.0, dbmel / 20.0)
    # 转换为 log-Mel
    logmel = torch.log(mel_linear + 1e-6)
    return logmel

def inference(a, h):
    # generator = Generator(h, use_cuda_kernel=a.use_cuda_kernel).to(device)

    # state_dict_g = load_checkpoint(a.checkpoint_file, device)
    # generator.load_state_dict(state_dict_g["generator"])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator = Qwen2_5OmniToken2WavBigVGANModel(Qwen2_5OmniBigVGANConfig())
    state_dict = torch.load(a.checkpoint_file, map_location="cpu")
    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator.eval()

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # Load the ground truth audio and resample if necessary
            wav, sr = librosa.load(
                os.path.join(a.input_wavs_dir, filname), sr=h.input_sampling_rate, mono=True
            )
            wav = torch.FloatTensor(wav).to(device)
            # Compute mel spectrogram from the ground truth audio
            # x = get_mel_spectrogram(wav.unsqueeze(0), generator.h)
            # print(f'mel shape: {x.shape}')
            x = mel_spectrogram_torch_aslp(
                y=wav.unsqueeze(0),
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=160,
                win_size=640,
                fmin=0,
                fmax=8000,
                center=False
            )

            y_g_hat = generator(x)

            resampler = torchaudio.transforms.Resample(h.output_sampling_rate, a.output_sample_rate)
            y_g_hat = resampler(y_g_hat)

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype("int16")

            output_file = os.path.join(
                a.output_dir, os.path.splitext(filname)[0] + "_generated.wav"
            )
            write(output_file, a.output_sample_rate, audio)
            print(output_file)


def main():
    print("Initializing Inference Process..")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wavs_dir", default="test_files")
    parser.add_argument("--output_dir", default="generated_files")
    parser.add_argument("--checkpoint_file", required=True)
    parser.add_argument("--output_sample_rate", default=16000)
    parser.add_argument("--use_cuda_kernel", action="store_true", default=False)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], "config.json")
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    inference(a, h)


if __name__ == "__main__":
    main()