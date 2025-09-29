# training script.

import os
from importlib.resources import files

import hydra
from omegaconf import OmegaConf
import shutil

from sense.model import Trainer_LLM
from sense.model.dataset import load_dataset
from sense.model.utils import get_tokenizer


os.chdir(str(files("sense").joinpath("../..")))  # change working directory to root of project (local editable)


@hydra.main(version_base="1.3", config_path=str(files("sense").joinpath("configs")), config_name=None)
def main(model_cfg):
    model_cls = hydra.utils.get_class(f"sense.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    finetune = model_cfg.model.finetune
    ckpt_path = model_cfg.model.pretrain_path

    # Handle multiple dataset names
    dataset_names = model_cfg.datasets.name
    if isinstance(dataset_names, (list, tuple)):
        # If it's a list, concatenate dataset names with plus sign
        dataset_name_str = "+".join(dataset_names)
    else:
        # If it's a single string, keep as is
        dataset_name_str = dataset_names
        dataset_names = [dataset_names]

    exp_name = f"{model_cfg.model.name}_{mel_spec_type}_{model_cfg.model.tokenizer}_{dataset_name_str}_ft"
    wandb_resume_id = None

    # Dynamically set save directory
    if model_cfg.ckpts.save_dir is None:
        save_dir = f"ckpts/{model_cfg.model.name}_{model_cfg.model.mel_spec.mel_spec_type}_{model_cfg.model.tokenizer}_{dataset_name_str}_ft"
    else:
        save_dir = model_cfg.ckpts.save_dir
    
    checkpoint_path=str(files("sense").joinpath(f"../../{save_dir}"))
    
    if finetune:
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):  # Change: Add 'pretrained_' prefix to copied model
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print("copy checkpoint for finetune")

    # set text tokenizer
    if "s3tokenizer_v1" in tokenizer:
        vocab_char_map, vocab_size = None, 4096
    elif "s3tokenizer_v2" in tokenizer:
        vocab_char_map, vocab_size = None, 6561   # 3**8
    elif tokenizer == "custom":
        tokenizer_path = model_cfg.model.tokenizer_path
    else:
        # For multiple datasets, use the path of the first dataset as tokenizer path
        tokenizer_path = dataset_names[0]
    
    if "s3tokenizer" not in tokenizer:
        vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    llm_model = model_cls(**model_arc)

    # init trainer
    trainer = Trainer_LLM(
        llm_model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=str(files("sense").joinpath(f"../../{model_cfg.ckpts.save_dir}")),
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        wandb_project="LLM-SenSE",
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = load_dataset(
        model_cfg.datasets.name,
        tokenizer,
        mel_spec_kwargs=model_cfg.model.mel_spec,
        pad=model_cfg.datasets.pad,
        add_eos_token=model_cfg.datasets.add_eos_token,
        eos_token=model_cfg.datasets.eos_token
    )

    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()