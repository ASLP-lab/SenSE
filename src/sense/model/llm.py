import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from transformers import AutoModelForCausalLM, AutoConfig
from sense.model.utils import make_pad_mask
from sense.model.encoder.conformer.encoder2 import ConformerEncoder
from sense.model.encoder.encoder_wrapper import setup_encoder
from sense.model.projector_wrapper import setup_encoder_projector

_BACKEND_TO_MODEL = {
    "llama3.2-1b": "src/sense/checkpoints/meta-llama/Llama-3.2-1B",
    "qwen3-0.6b": "src/sense/checkpoints/Qwen/Qwen3-0.6B-Base",
}

ENCODER_PATH = {
    "whisper": "src/sense/checkpoints/whisper/large-v3.pt",
    "wavlm": "src/sense/checkpoints/WavLM-Large/WavLM-Large.pt"
}

""" Semantic-Aware Speech Language Model (SASLM)"""
class SASLM(nn.Module):
    def __init__(
        self,
        encoder_name: str = "conformer",
        freeze_encoder: bool = False,
        encoder_output_dim: int = 512,
        projector_name: str = "linear",
        vocab_size: int = 4096,
        llm_backend: str = "llama3.2-1b",
        model_name_or_path: str = None,
        dtype: str = None,
        low_cpu_mem_usage: bool = True,
        device_map=None,
        reinit_embeddings: bool = True,
    ):
        super().__init__()

        self.IGNORE_ID = -100
        self.SPEECH_BOS = vocab_size + 0
        self.SPEECH_EOS = vocab_size + 1
        self.TOKEN_EOS  = vocab_size + 2
        self.vocab_size = vocab_size + 3

        self.encoder_name = encoder_name
        if encoder_name == "conformer":   # 25hz
            self.encoder = setup_encoder(encoder_name, freeze_encoder=False, encoder_dim=encoder_output_dim, num_layers=8)
        elif encoder_name == "conformer2":   # 50hz
            self.encoder = setup_encoder(encoder_name, freeze_encoder=False, encoder_dim=encoder_output_dim, num_layers=8)
        elif encoder_name == "whisper":
            self.encoder = setup_encoder(encoder_name, model_path=ENCODER_PATH["whisper"], freeze_encoder=False)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        elif encoder_name == "wavlm":
            self.encoder = setup_encoder(encoder_name, model_path=ENCODER_PATH["wavlm"], freeze_encoder=False)
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder.eval()
        else:
            raise ValueError(f"Unknown encoder_name: {encoder_name}")

        if model_name_or_path is None:
            if llm_backend not in _BACKEND_TO_MODEL:
                raise ValueError(f"Unknown llm_backend: {llm_backend}. "
                                 f"Choose from {list(_BACKEND_TO_MODEL.keys())} or pass model_name_or_path.")
            model_name_or_path = _BACKEND_TO_MODEL[llm_backend]
        
        if dtype == "fp16":
            torch_dtype = torch.float16
        elif dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
        )
        self.llm.config.use_cache = False

        self.llm.resize_token_embeddings(self.vocab_size)
        self.llm.config.eos_token_id = self.TOKEN_EOS
        self.llm.config.pad_token_id = self.TOKEN_EOS

        hidden_size = self.llm.config.hidden_size
        # self.speech_llama_proj = nn.Linear(encoder_output_dim, hidden_size)
        self.projector_name = projector_name
        if projector_name == "original":
            self.projector = nn.Linear(encoder_output_dim, hidden_size)
        elif projector_name == "linear":
            self.projector = setup_encoder_projector(projector_name, ds_rate=1, encoder_dim=encoder_output_dim, llm_dim=hidden_size)
        elif projector_name == "cov1d-linear":
            self.projector = setup_encoder_projector(projector_name, ds_rate=1, encoder_dim=encoder_output_dim, llm_dim=hidden_size)
        else:
            raise ValueError(f"Unknown projector_name: {projector_name}")

        self.speech_token_embeds = self.llm.get_input_embeddings()
        self.speech_head = self.llm.get_output_embeddings()

        if reinit_embeddings:
            with torch.no_grad():
                std = getattr(self.llm.config, "initializer_range", 0.02)
                self.speech_token_embeds.weight.normal_(mean=0.0, std=std)
        
        self.top5 = Accuracy(task="multiclass", num_classes=self.vocab_size, top_k=5)

    def forward(self, mels, mels_len, speech_tokens, speech_tokens_lengths, wavs=None, wavs_len=None):
        device = mels.device

        speech_embeds, speech_masks = self.get_embedding_from_mel(mels, mels_len, wavs, wavs_len)
        speech_target = torch.full(speech_masks.shape, self.IGNORE_ID, device=device)

        speech_embeds, speech_masks, speech_target = self._add_bos_eos(
            self.SPEECH_BOS, self.SPEECH_EOS, speech_embeds, speech_masks, speech_target
        )

        speech_token_labels = speech_tokens.to(device)
        speech_tokens_lengths = speech_tokens_lengths.to(device)
        speech_token_embeds, speech_token_target, speech_token_masks = self.get_speech_token_label_embedding(
            speech_token_labels, speech_tokens_lengths
        )

        inputs_embeds = torch.cat([speech_embeds,  speech_token_embeds], dim=1)
        attention_mask = torch.cat([speech_masks,  speech_token_masks], dim=1)
        target         = torch.cat([speech_target, speech_token_target], dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=target,
            # use_cache=True,
        )
        loss = outputs.loss
        return {"loss": loss}

    @torch.no_grad()
    def generate(self, mels, mels_len,
                 wavs=None, wavs_len=None,
                 max_new_tokens: int = 2000,
                 do_sample: bool = True,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1):
        device = mels.device
        speech_embeds, speech_masks = self.get_embedding_from_mel(mels, mels_len, wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(
            self.SPEECH_BOS, self.SPEECH_EOS, speech_embeds, speech_masks, None
        )

        attn = torch.ones(speech_embeds.size()[:2], dtype=torch.long, device=device)

        output_ids = self.llm.generate(
            inputs_embeds=speech_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.TOKEN_EOS,
            pad_token_id=self.TOKEN_EOS,
            use_cache=True,
        )
        return output_ids

    def get_embedding_from_mel(self, mels, mels_len, wavs=None, wavs_len=None):
        # encoder_out, encoder_out_lengths = self.encoder(mels, mels_len)
        if self.encoder_name == "conformer" or self.encoder_name == "conformer2":
            encoder_out, encoder_out_lengths = self.encoder(mels, mels_len)
            encoder_pad_mask = make_pad_mask(encoder_out_lengths).to(mels.device)
        elif self.encoder_name == "whisper":
            encoder_out = self.encoder.extract_variable_length_features(mels.permute(0, 2, 1)) # bs*seq*dim
            encoder_out_lengths = mels_len // 2
            encoder_pad_mask = make_pad_mask(encoder_out_lengths).to(mels.device)
        elif self.encoder_name == "wavlm":
            padding_mask = make_pad_mask(wavs_len).to(wavs.device)
            encoder_out, encoder_pad_mask = self.encoder.extract_features(wavs, padding_mask)
        # speech_embeds = self.projector(encoder_out)
        speech_embeds = self.projector(encoder_out)

        if encoder_pad_mask.shape[1] < speech_embeds.shape[1]:
            speech_embeds = speech_embeds[:, :encoder_pad_mask.shape[1], :]
        speech_mask = ~encoder_pad_mask
        return speech_embeds, speech_mask

    def get_speech_token_label_embedding(self, speech_token_labels, speech_tokens_length):
        pad_mask = make_pad_mask(speech_tokens_length)  # True for pad
        speech_token_labels = speech_token_labels.masked_fill(pad_mask, 0)
        embeds = self.speech_token_embeds(speech_token_labels)
        target = speech_token_labels.masked_fill(pad_mask, self.IGNORE_ID)
        mask = ~pad_mask
        return embeds, target, mask

    def _add_bos_eos(self, bos, eos, inputs_embeds, attention_mask, target=None, ignore_target=True):
        B = inputs_embeds.size(0)
        bos_eos_mask = torch.ones((B, 1), dtype=torch.bool, device=inputs_embeds.device)

        if ignore_target:
            bos_target = torch.full((B, 1), self.IGNORE_ID, device=inputs_embeds.device) if bos is not None else None
            eos_target = torch.full((B, 1), self.IGNORE_ID, device=inputs_embeds.device) if eos is not None else None
        else:
            bos_target = torch.full((B, 1), bos, device=inputs_embeds.device) if bos is not None else None
            eos_target = torch.full((B, 1), eos, device=inputs_embeds.device) if eos is not None else None

        if bos is not None:
            bos_ids = torch.full((B, 1), bos, device=inputs_embeds.device, dtype=torch.long)
            bos_embed = self.speech_token_embeds(bos_ids)
            inputs_embeds = torch.cat([bos_embed, inputs_embeds], dim=1)
            attention_mask = torch.cat([bos_eos_mask, attention_mask], dim=1)
            if target is not None:
                target = torch.cat([bos_target, target], dim=1)

        if eos is not None:
            eos_ids = torch.full((B, 1), eos, device=inputs_embeds.device, dtype=torch.long)
            eos_embed = self.speech_token_embeds(eos_ids)
            inputs_embeds = torch.cat([inputs_embeds, eos_embed], dim=1)
            attention_mask = torch.cat([attention_mask, bos_eos_mask], dim=1)
            if target is not None:
                target = torch.cat([target, eos_target], dim=1)

        return inputs_embeds, attention_mask, target


if __name__ == "__main__":
    dtype = None
    model = SASLM(
        encoder_name="conformer",
        encoder_output_dim=512,
        vocab_size=4096,
        llm_backend="llama3.2-1b",
        dtype=dtype,
        reinit_embeddings=True,
        device_map=None,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    mels = torch.zeros(2, 40, 80)
    mels_len = torch.tensor([mels.shape[1], 30])
    speech_tokens = torch.zeros(2, 20).long()
    speech_tokens_length = torch.tensor([speech_tokens.shape[1], 15])

    out = model(mels, mels_len, speech_tokens, speech_tokens_length)
    print(out)
