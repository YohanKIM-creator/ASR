import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from conftest import create
from src.datasets.data_utils import get_dataloaders
from src.logger import WandBWriter
from src.trainer import Trainer
from src.utils.checkpoint import CheckpointManager, atomic_torch_save, load_checkpoint
from src.utils.init_utils import set_random_seed

ROOT = Path(__file__).resolve().parents[1]


def configuration(data, saved):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "src/configs")):
        config = compose(config_name="smoke")
    config.datasets.train.data_dir = str(data)
    config.datasets.dev_other.data_dir = str(data)
    config.trainer.save_dir = str(saved)
    config.trainer.max_steps = 4
    config.trainer.validate_every = 4
    config.trainer.log_examples_every = 10000
    return config


def build_trainer(config, resume=None):
    set_random_seed(config.trainer.seed)
    encoder = instantiate(config.text_encoder)
    loaders, _ = get_dataloaders(config, encoder, "cpu")
    model = instantiate(config.model, n_tokens=len(encoder))
    optimizer = instantiate(config.optimizer, params=model.parameters())
    scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
    writer = WandBWriter(config.trainer.save_dir, {}, mode="disabled")
    return Trainer(
        model=model,
        criterion=instantiate(config.loss_function),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        text_encoder=encoder,
        config=config,
        device="cpu",
        dataloaders=loaders,
        writer=writer,
        resume_checkpoint=resume,
    )


def test_atomic_save_preserves_old_snapshot_on_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_torch_save({"value": 7}, path)
    original_save = torch.save

    def fail(*args, **kwargs):
        raise OSError("simulated disk interruption")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_torch_save({"value": 99}, path)
    monkeypatch.setattr(torch, "save", original_save)
    assert load_checkpoint(path)["value"] == 7
    assert not list(tmp_path.glob("*.tmp"))


def test_best_not_replaced_by_worse_and_last_is_current(tmp_path):
    manager = CheckpointManager(tmp_path)
    assert manager.save({"global_step": 1}, validation_wer=0.2)
    assert not manager.save({"global_step": 2}, validation_wer=0.5)
    assert load_checkpoint(tmp_path / "best.pt")["global_step"] == 1
    assert load_checkpoint(tmp_path / "last.pt")["global_step"] == 2
    assert load_checkpoint(tmp_path / "last.pt")["best_wer"] == 0.2


@pytest.mark.parametrize("accumulation", [1, 3])
def test_resume_exact_parameters_optimizer_schedule_and_cursor(tmp_path, accumulation):
    data = tmp_path / "dataset"
    create(data)
    full_config = configuration(data, tmp_path / "full")
    full_config.trainer.accumulation_steps = accumulation
    full_config.dataloader.max_items = 1
    full = build_trainer(full_config)
    full.train()
    interrupted_config = configuration(data, tmp_path / "resumed")
    interrupted_config.trainer.accumulation_steps = accumulation
    interrupted_config.dataloader.max_items = 1
    interrupted_config.trainer.session_max_steps = 1
    interrupted = build_trainer(interrupted_config)
    interrupted.train()
    checkpoint = load_checkpoint(tmp_path / "resumed/last.pt")
    assert checkpoint["global_step"] == 1
    interrupted_config.trainer.session_max_steps = None
    resumed = build_trainer(interrupted_config, resume=checkpoint)
    resumed.train()
    assert resumed.global_step == full.global_step == 4
    assert resumed.lr_scheduler.state_dict() == full.lr_scheduler.state_dict()
    for key, expected in full.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[key]), key
    final_full, final_resumed = (
        full.optimizer.state_dict(),
        resumed.optimizer.state_dict(),
    )
    for key, expected in final_full["state"].items():
        for field, value in expected.items():
            assert torch.equal(value, final_resumed["state"][key][field])
    saved = load_checkpoint(tmp_path / "resumed/last.pt")
    assert {
        "scaler",
        "rng",
        "epoch",
        "batch_in_epoch",
        "run_id",
        "text_encoder",
        "config",
    } <= set(saved)


def test_resume_rejects_data_or_scheduler_change(tmp_path):
    data = tmp_path / "dataset"
    create(data)
    config = configuration(data, tmp_path / "run")
    config.trainer.session_max_steps = 1
    trainer = build_trainer(config)
    trainer.train()
    saved = load_checkpoint(tmp_path / "run/last.pt")
    config.lr_scheduler.warmup_steps = 6
    with pytest.raises(ValueError, match="resume contract"):
        build_trainer(config, saved)


def test_inference_cli_and_metric_cli(tmp_path):
    data, saved, output = (
        tmp_path / "dataset",
        tmp_path / "run",
        tmp_path / "predictions",
    )
    create(data)
    config = configuration(data, saved)
    config.trainer.session_max_steps = 1
    build_trainer(config).train()
    result = subprocess.run(
        [
            sys.executable,
            "inference.py",
            f"inferencer.from_pretrained={saved / 'best.pt'}",
            f"datasets.custom.data_dir={data}",
            f"inferencer.save_path={output}",
            "inferencer.device=cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert {p.stem for p in (output / "custom").glob("*.txt")} == {
        p.stem for p in (data / "audio").glob("*.wav")
    }
    metric_run = subprocess.run(
        [
            sys.executable,
            "scripts/calc_metrics.py",
            "--reference-dir",
            str(data / "transcriptions"),
            "--prediction-dir",
            str(output / "custom"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert metric_run.returncode == 0, metric_run.stdout + metric_run.stderr
    assert json.loads(metric_run.stdout)["utterances"] == 4

    unlabeled = tmp_path / "unlabeled"
    (unlabeled / "audio").mkdir(parents=True)
    for audio in (data / "audio").glob("*.wav"):
        shutil.copyfile(audio, unlabeled / "audio" / audio.name)
    unscored_output = tmp_path / "unscored-predictions"
    unlabeled_run = subprocess.run(
        [
            sys.executable,
            "inference.py",
            f"inferencer.from_pretrained={saved / 'best.pt'}",
            f"datasets.custom.data_dir={unlabeled}",
            f"inferencer.save_path={unscored_output}",
            "inferencer.device=cpu",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert unlabeled_run.returncode == 0, unlabeled_run.stdout + unlabeled_run.stderr
    unscored = json.loads((unscored_output / "custom-metrics.json").read_text())
    assert unscored["predictions"] == 4
    assert unscored["utterances"] == 0
    assert unscored["reference_words"] == 0
    assert unscored["wer"] is None and unscored["cer"] is None
    for prediction in (output / "custom").glob("*.txt"):
        assert (
            unscored_output / "custom" / prediction.name
        ).read_text() == prediction.read_text()


def test_wandb_offline_logging_has_scalars_and_media(tmp_path):
    data = tmp_path / "dataset"
    create(data)
    config = configuration(data, tmp_path / "run")
    encoder = instantiate(config.text_encoder)
    loaders, _ = get_dataloaders(config, encoder, "cpu")
    batch = next(iter(loaders["train"]))
    writer = WandBWriter(
        tmp_path / "offline",
        {"test": True},
        mode="offline",
        run_name="preflight-logging",
        log_checkpoints=False,
    )
    try:
        writer.log({"train/loss": 1.2, "train/lr": 0.001, "train/grad_norm": 3.4}, 0)
        writer.log_examples(
            batch, ["a"] * len(batch["text"]), encoder, 0, "train", count=1
        )
        writer.log_examples(
            batch, ["a"] * len(batch["text"]), encoder, 0, "dev_other", count=1
        )
    finally:
        writer.finish()
    scalars = json.loads(
        (tmp_path / "offline/metrics.jsonl").read_text().splitlines()[0]
    )
    assert scalars["train/grad_norm"] == 3.4
    assert list((tmp_path / "offline/examples").glob("*.wav"))
    assert list((tmp_path / "offline/examples").glob("*.png"))
    assert list((tmp_path / "offline/examples").glob("*predictions.json"))
    assert list((tmp_path / "offline/wandb").glob("offline-run-*"))


def test_finished_checkpoint_resume_does_not_step(tmp_path):
    data = tmp_path / "dataset"
    create(data)
    config = configuration(data, tmp_path / "run")
    config.trainer.max_steps = 1
    original = build_trainer(config)
    original.train()
    saved = load_checkpoint(tmp_path / "run/last.pt")
    resumed = build_trainer(config, saved)
    resumed.train()
    assert resumed.global_step == 1
    for name, expected in original.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name])
