import torch
import torch.utils.data
import numpy as np
import pyworld as pw
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

    #spec = _amp_to_db(spec, -115) - 20
    #spec = _normalize(spec, 1, -115)
    spec = spectral_normalize_torch(spec)

    return spec

# def mel_spectrogram_torch_aslp(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin=0, fmax=8000, center=False):
#     if torch.min(y) < -1.:
#         print('min value is ', torch.min(y))
#     if torch.max(y) > 1.:
#         print('max value is ', torch.max(y))

#     global mel_basis, hann_window
#     dtype_device = str(y.dtype) + '_' + str(y.device)
#     fmax_dtype_device = str(fmax) + '_' + dtype_device
#     wnsize_dtype_device = str(win_size) + '_' + dtype_device
#     if fmax_dtype_device not in mel_basis:
#         mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
#         mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
#     if wnsize_dtype_device not in hann_window:
#         hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

#     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
#                                 mode='reflect')
#     y = y.squeeze(1)

#     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
#                       center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

#     spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

#     spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

#     spec = _amp_to_db(spec, -115) - 20
#     spec = _normalize(spec, 1, -115)
#     return spec
def mel_spectrogram_torch_aslp(
    y,
    n_fft,
    num_mels,
    sampling_rate,
    hop_size,
    win_size,
    fmin=0,
    fmax=8000,
    center=False,
    min_db=-115.0,
    ref_level_db=20.0,
    max_abs_value=1.0
):
    """
    Compute normalized mel spectrogram (in dB scale) from waveform using PyTorch.

    Args:
        y (Tensor): shape [B, T], waveform normalized to [-1, 1]
        n_fft (int): FFT size
        num_mels (int): number of mel bins
        sampling_rate (int): sample rate of waveform
        hop_size (int): hop size between frames
        win_size (int): window size for STFT
        fmin (float): minimum frequency in mel filter
        fmax (float): maximum frequency in mel filter
        center (bool): whether to pad the input waveform
        min_db (float): minimum dB for amplitude to dB conversion
        ref_level_db (float): reference dB to subtract after amp_to_db
        max_abs_value (float): max abs value for normalization

    Returns:
        spec (Tensor): [B, num_mels, T'] normalized mel spectrogram
    """
    # Check range
    if torch.min(y) < -1.0 or torch.max(y) > 1.0:
        print(f"[Warning] Waveform out of range: min={torch.min(y)}, max={torch.max(y)}")

    # Cache keys for mel filter bank and Hann window
    global mel_basis, hann_window
    dtype_device = f"{y.dtype}_{y.device}"
    mel_key = f"{fmax}_{dtype_device}"
    win_key = f"{win_size}_{dtype_device}"

    # Create mel filter bank if not cached
    if mel_key not in mel_basis:
        mel_filter = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax
        )
        mel_basis[mel_key] = torch.from_numpy(mel_filter).to(dtype=y.dtype, device=y.device)

    # Create Hann window if not cached
    if win_key not in hann_window:
        hann_window[win_key] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # Apply STFT
    y_padded = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) // 2), int((n_fft - hop_size) // 2)),
        mode='reflect'
    ).squeeze(1)

    spec_complex = torch.stft(
        y_padded,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[win_key],
        center=center,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=False
    )

    magnitude = torch.sqrt(spec_complex.pow(2).sum(-1) + 1e-6)  # [B, F, T]

    # Convert to mel
    mel_spec = torch.matmul(mel_basis[mel_key], magnitude)  # [B, mel, T]

    # Amplitude to dB
    mel_db = _amp_to_db(mel_spec, min_level_db=min_db) - ref_level_db

    # Normalize
    mel_norm = _normalize(mel_db, max_abs_value=max_abs_value, min_db=min_db)

    return mel_norm



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


def mel_spectrogram_torch_nhv(spec_origin, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, min_db,
                              max_abs_value, min_level_db, ref_level_db, center=True):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='constant', normalized=False, onesided=True, return_complex=False)

    # spec = torch.abs(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1))
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spec[:, :, :spec_origin.shape[-1]]
    spec_origin = _db_to_amp(_denormalize(spec_origin, max_abs_value, min_db) + ref_level_db)

    spec_res = spec_origin - spec

    spec = _amp_to_db(spec, min_level_db) - ref_level_db
    spec = _normalize(spec, max_abs_value, min_db)

    spec_res = _amp_to_db(spec_res, min_level_db) - ref_level_db
    spec_res = _normalize(spec_res, max_abs_value, min_db)

    return spec, spec_res


def mel_spectrogram_torch_nhv2(spec_origin, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, min_db,
                               max_abs_value, min_level_db, ref_level_db, center=True):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    # y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                      center=center, pad_mode='constant', normalized=False, onesided=True, return_complex=False)

    # spec = torch.abs(spec)
    spec = torch.sqrt(spec.pow(2).sum(-1))
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spec[:, :, :spec_origin.shape[-1]]
    spec_origin = _db_to_amp(_denormalize(spec_origin, max_abs_value, min_db) + ref_level_db)

    spec_res = spec_origin - spec

    spec = _amp_to_db(spec, min_level_db) - ref_level_db
    spec = _normalize(spec, max_abs_value, min_db)

    return spec, spec_res


def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))


def _normalize(S, max_abs_value, min_db):
    return torch.clamp((2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value, -max_abs_value, max_abs_value)


def _db_to_amp(x):
    return torch.pow(10.0, (x) * 0.05)


def _denormalize(D, max_abs_value, min_db):
    return (((torch.clamp(D, -max_abs_value, max_abs_value) + max_abs_value) * -min_db / (2 * max_abs_value)) + min_db)


def extract_lf0(wav, sr, hop_size):

    #extract lf0
    f0,t = pw.harvest(wav.astype(np.float64), sr, frame_period=hop_size/sr*1000)
    f0 = pw.stonemask(wav.astype(np.float64),f0, t, sr)
    lf0 = np.log(f0 + 1e-8)
    lf0[lf0 < 1e-3] = 0
    lf0 = lf0.astype(np.float32)
    lf0 = lf0.reshape([-1])

    return lf0
