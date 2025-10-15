import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy
from transformers import LlamaConfig, LlamaForCausalLM
from sense.model.utils import make_pad_mask
from sense.model.encoder.conformer.encoder2 import ConformerEncoder
from sense.model.encoder.encoder_wrapper import setup_encoder
from sense.model.projector_wrapper import setup_encoder_projector

# "envs/sense/lib/python3.10/site-packages/whisper/model.py"
ENCODER_PATH = {
    "whisper": "src/sense/checkpoints/whisper/large-v3.pt",
    "wavlm": "src/sense/checkpoints/WavLM-Large/WavLM-Large.pt"
}

class LLM_LLaMA(nn.Module):
    def __init__(
        self,
        encoder_name: str = "conformer",
        freeze_encoder: bool = False,
        encoder_output_dim: int = 512,
        projector_name: str = "linear",
        d_model: int = 1024,
        nhead: int = 16,
        num_layers: int = 12,
        vocab_size: int = 4096,
    ):
        super().__init__()

        self.IGNORE_ID = -100
        self.SPEECH_BOS = vocab_size + 0
        self.SPEECH_EOS = vocab_size + 1
        self.TOKEN_EOS = vocab_size + 2
        self.vocab_size = vocab_size + 3

        self.encoder_name = encoder_name
        self.freeze_encoder = freeze_encoder

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

        # self.speech_llama_proj = nn.Linear(encoder_output_dim, d_model * 2)
        self.projector_name = projector_name
        if projector_name == "original":
            self.speech_llama_proj = nn.Linear(encoder_output_dim, d_model * 2)
        elif projector_name == "linear":
            self.projector = setup_encoder_projector(projector_name, ds_rate=1, encoder_dim=encoder_output_dim, llm_dim=d_model * 2)
        elif projector_name == "cov1d-linear":
            self.projector = setup_encoder_projector(projector_name, ds_rate=1, encoder_dim=encoder_output_dim, llm_dim=d_model * 2)
        else:
            raise ValueError(f"Unknown projector_name: {projector_name}")
        
        # for param in self.projector.parameters():
        #     param.requires_grad = False
        # self.projector.eval()

        config = LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=d_model * 2,
            intermediate_size=d_model * 4,
            num_attention_heads=nhead,
            num_hidden_layers=num_layers,
            dropout_rate=0.1,
            attention_dropout=0.1,
            is_decoder=True,
            use_cache=True,
            max_position_embeddings=4096
        )

        self.llama = LlamaForCausalLM(config=config)

        self.speech_token_embeds = self.llama.model.embed_tokens
        self.speech_head = self.llama.lm_head

        # for param in self.llama.parameters():
        #     param.requires_grad = False
        # self.llama.eval()
        self.top5 = Accuracy(task="multiclass", num_classes=self.vocab_size + 3, top_k=5)

    def forward(self, mels, mels_len, speech_tokens, speech_tokens_lengths, wavs=None, wavs_len=None):
        device = mels.device

        speech_embeds, speech_masks = self.get_embedding_from_mel(mels, mels_len, wavs, wavs_len)
        speech_target = torch.full(speech_masks.shape, self.IGNORE_ID).to(device)
        length1 = speech_embeds.shape[1]

        # add bos and eos
        speech_embeds, speech_masks, speech_target = self._add_bos_eos(
            self.SPEECH_BOS,
            self.SPEECH_EOS,
            speech_embeds, speech_masks, speech_target
        )

        # Construct embedding and mask
        speech_token_labels = speech_tokens.to(device)
        speech_tokens_lengths = speech_tokens_lengths.to(device)
        speech_token_embeds, speech_token_target, speech_token_masks = self.get_speech_token_label_embedding(
                speech_token_labels, speech_tokens_lengths)
        length2 = speech_token_embeds.shape[1]
        # print(f'lengh1: {length1}, length2: {length2}')
        
        inputs_embeds_list = []
        attention_mask_list = []
        target_list = []
        inputs_embeds_list.extend([speech_embeds, speech_token_embeds])
        attention_mask_list.extend([speech_masks, speech_token_masks])
        target_list.extend([speech_target, speech_token_target])

        inputs_embeds = torch.cat(inputs_embeds_list, dim=1)
        attention_mask = torch.cat(attention_mask_list, dim=1)
        target = torch.cat(target_list, dim=1)

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # Input to LLaMA model (use inputs_embeds instead of input_ids)
        outputs = self.llama(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=target
        )

        loss = outputs.loss

        return {
            "loss": loss
        }

    def generate(self, mels, mels_len, wavs=None, wavs_len=None, ref_mels=None, ref_mels_len=None, ref_tokens=None, ref_tokens_len=None):
        if ref_mels is not None:
            mels = torch.cat([ref_mels, mels], dim=1)
            mels_len = mels_len + ref_mels_len
            ref_token_embeds, _, _ = self.get_speech_token_label_embedding(ref_tokens, ref_tokens_len)

        speech_embeds, speech_masks = self.get_embedding_from_mel(mels, mels_len, wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(self.SPEECH_BOS, self.SPEECH_EOS,
                                                           speech_embeds, speech_masks, None)
        
        if ref_mels is not None:
            embeds = torch.cat([speech_embeds, ref_token_embeds], dim=1)
        else:
            embeds = speech_embeds

        # output_ids = self.llama.generate(
        #     inputs_embeds=embeds,
        #     max_new_tokens=2000,  # 4000
        #     do_sample=True,
        #     temperature=1.0,  # 1.0
        #     top_p=0.9,
        #     # top_k=10,
        #     repetition_penalty=1.1,
        #     eos_token_id=self.vocab_size - 1,
        # )
        output_ids = self.llama.generate(
            inputs_embeds=embeds,
            max_new_tokens=2000,  # 4000
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=self.vocab_size - 1,
        )

        return output_ids
    
    def get_embedding_from_mel(self, mels, mels_len, wavs=None, wavs_len=None):
        """
        return:
        wav_embedding: (b, l, v)
        wav_mask:  (b, l), positions with valid wav values are true
        """
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
        if self.projector_name == "original":
            speech_embeds = self.speech_llama_proj(encoder_out)
        else:
            speech_embeds = self.projector(encoder_out)
        if encoder_pad_mask.shape[1] < speech_embeds.shape[1]:
            speech_embeds = speech_embeds[:, :encoder_pad_mask.shape[1], :]
        encoder_mask = ~encoder_pad_mask
        return speech_embeds, encoder_mask

    def get_speech_token_label_embedding(self, speech_token_labels, speech_tokens_length):
        pad_mask = make_pad_mask(speech_tokens_length)
        speech_token_labels = speech_token_labels.masked_fill(pad_mask, 0)
        embeds = self.speech_token_embeds(speech_token_labels)
        target = speech_token_labels.masked_fill(pad_mask, self.IGNORE_ID)
        mask = ~pad_mask
        return embeds, target, mask

    def _add_bos_eos(self, bos, eos, inputs_embeds, attention_mask, target=None, ignore_target=True):
        B = inputs_embeds.size(0)
        if ignore_target:
            bos_target = torch.full((B, 1), self.IGNORE_ID).to(inputs_embeds.device) if bos is not None else None
            eos_target = torch.full((B, 1), self.IGNORE_ID).to(inputs_embeds.device) if eos is not None else None
        else:
            bos_target = torch.full((B, 1), bos).to(inputs_embeds.device) if bos is not None else None
            eos_target = torch.full((B, 1), eos).to(inputs_embeds.device) if eos is not None else None
        bos_eos_mask = torch.ones((B, 1), dtype=torch.bool).to(inputs_embeds.device)

        if bos is not None:
            bos_embed = self.speech_token_embeds(torch.full((B, 1), bos).to(inputs_embeds.device))
            inputs_embeds = torch.cat([bos_embed, inputs_embeds], dim=1)
            attention_mask = torch.cat([bos_eos_mask, attention_mask], dim=1)
            if target is not None:
                target = torch.cat([bos_target, target], dim=1)

        if eos is not None:
            eos_embed = self.speech_token_embeds(torch.full((B, 1), eos).to(inputs_embeds.device))
            inputs_embeds = torch.cat([inputs_embeds, eos_embed], dim=1)
            attention_mask = torch.cat([attention_mask, bos_eos_mask], dim=1)
            if target is not None:
                target = torch.cat([target, eos_target], dim=1)

        return inputs_embeds, attention_mask, target

    def _remove_weight_norm_from_wavlm(self):
        """Remove weight_norm from WavLM to avoid deep copy issues"""
        try:
            from torch.nn.utils import remove_weight_norm
            # Access internal WavLM model based on WavLMEncoder structure
            if hasattr(self.encoder, 'model') and hasattr(self.encoder.model, 'encoder'):
                encoder = self.encoder.model.encoder
                if hasattr(encoder, 'pos_conv') and isinstance(encoder.pos_conv, nn.Sequential):
                    # pos_conv is a Sequential, first element is Conv1d with weight_norm
                    conv_layer = encoder.pos_conv[0]
                    try:
                        remove_weight_norm(conv_layer)
                        print("Successfully removed weight_norm from WavLM pos_conv")
                    except ValueError as e:
                        print(f"No weight_norm found in WavLM pos_conv: {e}")
                elif hasattr(encoder, 'pos_conv'):
                    # Directly Conv1d layer
                    try:
                        remove_weight_norm(encoder.pos_conv)
                        print("Successfully removed weight_norm from WavLM pos_conv")
                    except ValueError as e:
                        print(f"No weight_norm found in WavLM pos_conv: {e}")
            else:
                print("WavLM encoder structure not as expected")
        except Exception as e:
            print(f"Warning: Could not remove weight_norm from WavLM: {e}")
            # If removal fails, try setting to non-trainable to reduce issues
            try:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                print("Set all WavLM parameters to requires_grad=False as fallback")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
    

if __name__ == "__main__":
    model = LLM_LLaMA(
        encoder_output_dim=1024,
        d_model=768,
        nhead=16,
        num_layers=12,
        vocab_size=4096
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # wavs = torch.randn(1, 16000*4)
    # wavs_len = torch.tensor([wavs.shape[1]])
    mels = torch.zeros(2, 40, 80)
    mels_len = torch.tensor([mels.shape[1], 30])
    # speech_tokens = torch.randint(0, model.vocab_size, (1, 200))
    speech_tokens = torch.zeros(2, 20).long()
    speech_tokens_length = torch.tensor([speech_tokens.shape[1], 15])

    # out = model.generate(None, None, mels, mels_len)
    out = model(mels, mels_len, speech_tokens, speech_tokens_length)
    print(out)
