import json
import os
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.checkpoint import load_checkpoint
from src.utils.init_utils import set_random_seed, supports_native_bf16


def load_local_wandb_key():
    path = Path(__file__).parent / ".secrets/credentials.json"
    if "WANDB_API_KEY" not in os.environ and path.is_file():
        credential = json.loads(path.read_text()).get("wandb")
        if credential:
            os.environ["WANDB_API_KEY"] = credential


@hydra.main(version_base=None, config_path="src/configs", config_name="train_char100")
def main(config):
    set_random_seed(config.trainer.seed)
    torch.set_num_threads(config.trainer.cpu_threads)
    load_local_wandb_key()
    directory = Path(config.trainer.save_dir).expanduser().resolve()
    checkpoint = None
    resume_from = config.trainer.resume_from
    if resume_from == "auto":
        resume_from = (
            str(directory / "last.pt") if (directory / "last.pt").is_file() else None
        )
    if resume_from:
        checkpoint = load_checkpoint(resume_from)
    elif (directory / "last.pt").exists():
        raise FileExistsError(
            "existing run: use trainer.resume_from=auto or a new save_dir"
        )
    directory.mkdir(parents=True, exist_ok=True)
    device = config.trainer.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if config.trainer.require_cuda and not str(device).startswith("cuda"):
        raise RuntimeError(
            "full training requires a CUDA GPU; use smoke config for local checks"
        )
    if (
        str(device).startswith("cuda")
        and config.trainer.amp == "bf16"
        and not supports_native_bf16()
    ):
        raise RuntimeError("GPU lacks bf16; select trainer.amp=fp16 (e.g. T4)")
    text_encoder = instantiate(config.text_encoder)
    dataloaders, _ = get_dataloaders(config, text_encoder, device)
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    if checkpoint is None and config.trainer.init_from:
        initial = load_checkpoint(config.trainer.init_from)
        if initial["text_encoder"] != text_encoder.state_dict():
            raise ValueError("fine-tuning tokenizer mismatch")
        model.load_state_dict(initial["model"], strict=True)
    criterion = instantiate(config.loss_function).to(device)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    writer = instantiate(
        config.writer,
        directory=directory,
        project_config=OmegaConf.to_container(config, resolve=True),
        run_id=checkpoint["run_id"] if checkpoint else None,
        _recursive_=False,
        _convert_="all",
    )
    OmegaConf.save(
        config, directory / ("config-resume.yaml" if checkpoint else "config.yaml")
    )
    print(
        f"device={device}, parameters={sum(p.numel() for p in model.parameters()):,}, tokens={len(text_encoder)}",
        flush=True,
    )
    try:
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            text_encoder=text_encoder,
            config=config,
            device=device,
            dataloaders=dataloaders,
            writer=writer,
            resume_checkpoint=checkpoint,
        )
        trainer.train()
    finally:
        writer.finish()


if __name__ == "__main__":
    main()
