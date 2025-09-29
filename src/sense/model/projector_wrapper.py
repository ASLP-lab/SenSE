import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_encoder_projector(encoder_projector, **kwargs):
    if encoder_projector == "linear":
        from sense.model.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(**kwargs)
    elif encoder_projector == "cov1d-linear":
        from sense.model.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(**kwargs)
    elif encoder_projector == "q-former":
        from sense.model.projector import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(**kwargs)
    else:
        return None
    # print_module_size(encoder_projector, model_config.encoder_projector, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return encoder_projector