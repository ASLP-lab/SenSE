import json
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader, SequentialSampler
import scipy.signal as signal
from tqdm import tqdm
import os
import numpy as np
import random

from sense.model.modules import MelSpec
from sense.model.utils import default

def align_audio_duration(noise_audio, target_audio):
    
    noise_length = noise_audio.shape[-1]
    target_length = target_audio.shape[-1]
    
    if noise_length > target_length:
        start_idx = torch.randint(0, noise_length - target_length + 1, (1,)).item()
        aligned_noise = noise_audio[:, start_idx:start_idx + target_length]
    elif noise_length < target_length:
        repeat_times = (target_length + noise_length - 1) // noise_length  # Ceiling division
        repeated_noise = noise_audio.repeat(1, repeat_times)
        aligned_noise = repeated_noise[:, :target_length]
    else:
        aligned_noise = noise_audio
    
    return aligned_noise

def mix_audio_with_snr(clean_audio, noise_audio, snr_db_range=(-10, 10)):
    snr_db = torch.rand(1).item() * (snr_db_range[1] - snr_db_range[0]) + snr_db_range[0]
    
    clean_power = torch.mean(clean_audio ** 2)
    noise_power = torch.mean(noise_audio ** 2)
    
    if noise_power == 0:
        return clean_audio
    if clean_power == 0:
        return noise_audio
    
    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = clean_power / snr_linear
    noise_scale = torch.sqrt(target_noise_power / noise_power)
    
    scaled_noise = noise_audio * noise_scale
    mixed_audio = clean_audio + scaled_noise
    
    return mixed_audio

def random_amplitude_scaling(*audios, scale_range=(0.3, 1.0)):
    """
    Normalize and then scale multiple audio tensors by the same random factor.

    Args:
        *audios: input torch tensors (1D or 2D)
        scale_range: tuple (min_scale, max_scale), range of random scaling factor

    Returns:
        Tuple of scaled tensors
    """
    max_amp = max(torch.max(torch.abs(audio)).item() for audio in audios)
    
    if max_amp > 1.0:
        audios = tuple(audio / max_amp for audio in audios)

    scale = random.uniform(*scale_range)

    return tuple(audio * scale for audio in audios), scale

def add_reverberation_v2(speech_sample, rir_sample, fs):
    rir_wav = rir_sample
    wav_len = speech_sample.shape[1]
    delay_idx = np.argmax(np.abs(rir_wav[0]))  # get the delay index
    delay_before_num = int(0.001 * fs)
    delay_after_num = int(0.05 * fs)
    idx_start = delay_idx - delay_before_num
    idx_end = delay_idx + delay_after_num
    if idx_start < 0:
        idx_start = 0
    early_rir = rir_wav[:, idx_start:idx_end]
    
    reverbant_speech_early = signal.fftconvolve(speech_sample, early_rir, mode="full")
    reverbant_speech = signal.fftconvolve(speech_sample, rir_wav, mode="full")
    
    reverbant_speech = reverbant_speech[:, idx_start:idx_start + wav_len]
    reverbant_speech_early = reverbant_speech_early[:, :wav_len]
    scale = max(abs(reverbant_speech[0]))
    if scale == 0:
        scale = 1
    else:
        scale = 0.5 / scale
    reverbant_speech_early = reverbant_speech_early * scale
    reverbant_speech = reverbant_speech * scale
    return reverbant_speech, reverbant_speech_early

class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]


        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )

class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        output_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        pad=False,
        add_eos_token=False,
        eos_token=4098
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.output_sample_rate = output_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel
        self.pad = pad
        self.add_eos_token = add_eos_token
        self.eos_token = eos_token

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length
    
    def pad_to_multiple_of_320(wav):
        length = wav.shape[-1]
        pad_len = (320 - (length % 320)) % 320
        if pad_len > 0:
            wav = torch.nn.functional.pad(wav, (0, pad_len), value=0)
        return wav

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            audio_path = row["audio_path"]
            degraded_audio_path = row["degraded_audio_path"]
            duration = row["duration"]
            audio_token = row["audio_token"]

            # filter by given length
            if 0.3 <= duration <= 60:
                break  # valid

            index = (index + 1) % len(self.data)

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])
        else:
            audio, source_sample_rate = torchaudio.load(audio_path)
            degraded_audio, _ = torchaudio.load(degraded_audio_path)
            if self.pad:
                audio = torch.nn.functional.pad(audio,(160,160))
                degraded_audio = torch.nn.functional.pad(degraded_audio,(160,160))
            if duration > 30:
                audio = audio[:, :30 * source_sample_rate]
                degraded_audio = degraded_audio[:, :30 * source_sample_rate]

            # make sure mono input
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            if degraded_audio.shape[0] > 1:
                degraded_audio = torch.mean(degraded_audio, dim=0, keepdim=True)

            # resample if necessary
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)
                degraded_audio = resampler(degraded_audio)

            if degraded_audio.shape[-1] != audio.shape[-1]:
                degraded_audio = degraded_audio[:, :audio.shape[-1]] if degraded_audio.shape[-1] > audio.shape[-1] else degraded_audio
                audio = audio[:, :degraded_audio.shape[-1]] if audio.shape[-1] > degraded_audio.shape[-1] else audio

            (audio, degraded_audio), scale = random_amplitude_scaling(audio, degraded_audio, scale_range=(0.3, 1.0))

            # to mel spectrogram
            mel_spec = self.mel_spectrogram(audio)
            degraded_mel_spec = self.mel_spectrogram(degraded_audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'
            degraded_mel_spec = degraded_mel_spec.squeeze(0)  # '1 d t -> d t'
            if self.add_eos_token:
                audio_token.append(self.eos_token)
            
        return {
            "audio_path": audio_path,
            "clean_audio": audio,
            "noisy_audio": degraded_audio,
            "mel_spec": mel_spec,
            "noisy_mel_spec": degraded_mel_spec,
            "audio_token": audio_token
        }

# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 0:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str | list,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    pad: bool = False,
    add_eos_token: bool = False,
    eos_token : int = 4098
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    dataset_name    - 
    """

    # Compatible with OmegaConf's ListConfig type
    from omegaconf import ListConfig
    if isinstance(dataset_name, (list, tuple, ListConfig)):
        dataset_names = list(dataset_name)
    else:
        dataset_names = [dataset_name]
    
    print(f"Loading datasets: {dataset_names}")

    if dataset_type == "CustomDataset":
        combined_train_datasets = []
        combined_durations = []
        
        for i, single_dataset_name in enumerate(dataset_names):
            rel_data_path = str(files("sense").joinpath(f"../../data/{single_dataset_name}_{tokenizer}"))
            
            print(f"Loading dataset {i+1}/{len(dataset_names)}: {single_dataset_name}")
            
            if audio_type == "raw":
                try:
                    train_dataset = load_from_disk(f"{rel_data_path}/raw")
                except:  # noqa: E722
                    train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
                preprocessed_mel = False
            elif audio_type == "mel":
                train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
                preprocessed_mel = True
            
            with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            durations = data_dict["duration"]
            
            combined_train_datasets.append(train_dataset)
            combined_durations.extend(durations)
            
            print(f"  - Dataset {single_dataset_name}: {len(train_dataset)} samples, total duration: {sum(durations)/3600:.2f}h")
        
        if len(combined_train_datasets) == 1:
            merged_train_dataset = combined_train_datasets[0]
        else:
            from datasets import concatenate_datasets
            merged_train_dataset = concatenate_datasets(combined_train_datasets)
        
        train_dataset = CustomDataset(
            merged_train_dataset,
            durations=combined_durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
            pad=pad,
            add_eos_token=add_eos_token,
            eos_token=eos_token
        )

    elif dataset_type == "CustomDatasetPath":
        if isinstance(dataset_name, (list, tuple)):
            raise ValueError("CustomDatasetPath mode only supports single dataset, not multiple datasets")
        
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        if isinstance(dataset_name, (list, tuple)):
            raise ValueError("HFDataset mode only supports single dataset, not multiple datasets")
        
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("sense").joinpath("../../data"))),
        )

    return train_dataset


# collation


def collate_fn(batch):
    audio_paths = [item["audio_path"] for item in batch]

    clean_audios = [item["clean_audio"].squeeze(0) for item in batch]
    noisy_audios = [item["noisy_audio"].squeeze(0) for item in batch]
    audio_lengths = torch.LongTensor([audio.shape[-1] for audio in clean_audios])
    max_audio_length = audio_lengths.amax()

    padded_clean_audios = []
    padded_noisy_audios = []
    for idx, (clean_audio, noisy_audio) in enumerate(zip(clean_audios, noisy_audios)):
        if clean_audio.size(-1) != noisy_audio.size(-1):
            print(f'clean_audio length and noisy_audio are not equal: {audio_paths[idx]}')
            print(f'clean_audio length: {clean_audio.size(-1)}, noisy_audio length: {noisy_audio.size(-1)}')
            continue
        padding = (0, max_audio_length - clean_audio.size(-1))
        padded_clean_audio = F.pad(clean_audio, padding, value=0)
        padded_noisy_audio = F.pad(noisy_audio, padding, value=0)
        padded_clean_audios.append(padded_clean_audio)
        padded_noisy_audios.append(padded_noisy_audio)

    clean_audios = torch.stack(padded_clean_audios)
    noisy_audios = torch.stack(padded_noisy_audios)

    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    noisy_mel_specs = [item["noisy_mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()
    
    padded_mel_specs = []
    padded_noisy_mel_specs = []
    for idx, (spec, noisy_spec) in enumerate(zip(mel_specs, noisy_mel_specs)):  # TODO. maybe records mask for attention here
        if spec.size(-1) != noisy_spec.size(-1):
            print(f'spec length and noisy_spec are not equal: {audio_paths[idx]}')
            print(f'spec length: {spec.size(-1)}, noisy_spec length: {noisy_spec.size(-1)}')
            mel_lengths = torch.cat([mel_lengths[:idx], mel_lengths[idx+1:]])
            max_mel_length = mel_lengths.amax() if len(mel_lengths) > 0 else 0
            continue
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_noisy_spec = F.pad(noisy_spec, padding, value=0)
        padded_mel_specs.append(padded_spec)
        padded_noisy_mel_specs.append(padded_noisy_spec)

    mel_specs = torch.stack(padded_mel_specs)
    noisy_mel_specs = torch.stack(padded_noisy_mel_specs)

    audio_tokens = [item["audio_token"] for item in batch]
    audio_token_lengths = torch.LongTensor([len(item) for item in audio_tokens])
    
    # Convert audio_tokens and noisy_tokens to padded tensors [B, L]
    max_audio_token_length = max(len(tokens) for tokens in audio_tokens) if audio_tokens else 0
    
    # Pad audio_tokens to [B, L] with -1 padding
    padded_audio_tokens = []
    for tokens in audio_tokens:
        padded_tokens = tokens + [-1] * (max_audio_token_length - len(tokens))
        padded_audio_tokens.append(padded_tokens)
    audio_tokens_tensor = torch.LongTensor(padded_audio_tokens)
    
    return dict(
        clean_audio=clean_audios,
        noisy_audio=noisy_audios,
        audio_lengths=audio_lengths,
        mel=mel_specs,
        noisy_mel=noisy_mel_specs,
        mel_lengths=mel_lengths,
        audio_tokens=audio_tokens_tensor,
        audio_token_lengths=audio_token_lengths,
    )

if __name__ == "__main__":
    mel_config = {
        "target_sample_rate": 24000,
        "n_mel_channels": 100,
        "hop_length": 256,
        "win_length": 1024,
        "n_fft": 1024
    }
    
    train_dataset = load_dataset("DNS_Challenge", "Noise_Dataset", "RIR_Dataset", tokenizer="s3tokenizer_v1", mel_spec_kwargs=mel_config)

    sampler = SequentialSampler(train_dataset)
    batch_sampler = DynamicBatchSampler(
        sampler,
        19200,
        max_samples=32,
        random_seed=2,  # This enables reproducible shuffling
        drop_residual=False,
    )
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        batch_sampler=batch_sampler,
    )
    
    idx = 0
    for batch in train_dataloader:
        idx += 1
        if idx > 100:
            break
        print(f"batch {idx} shape: {batch['mel'].shape}")