import json
import zipfile

import pytest
import torch

from src.logger import CometWriter
from src.text_encoder import CTCTextEncoder


def test_comet_offline_logs_media_scalars_and_preserves_legacy_identity(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    encoder = CTCTextEncoder()
    secret = "should-never-be-logged-private-value"
    writer = CometWriter(
        tmp_path,
        {"api_key": secret, "recipe": {"width": 288}},
        mode="offline",
        workspace=None,
        run_id="bs1uranz",
    )
    key = writer.comet_key
    assert writer.run_id == "bs1uranz" and len(key) == 32
    waveform = torch.linspace(-0.1, 0.1, 1600).unsqueeze(0)
    batch = {
        "text": ["hello"],
        "utterance_id": ["1-2-3"],
        "has_transcript": [True],
        "original_audio": [waveform],
        "audio": [waveform * 0.5],
        "spectrogram": torch.randn(1, 40, 20),
        "spectrogram_length": torch.tensor([20]),
    }
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": {"weight": torch.ones(2)}, "global_step": 1}, checkpoint)
    try:
        writer.log({"train/loss": None, "train/lr": 0.0005, "train/grad_norm": 0.0}, 0)
        writer.log({"train/loss": 1.2, "train/grad_norm": 3.4}, 1)
        writer.log_examples(batch, ["helo"], encoder, 0, "train", count=1)
        writer.log_examples(batch, ["hello"], encoder, 0, "dev_other", count=1)
        writer.publish_best(checkpoint, {"step": 1})
    finally:
        writer.finish()
    metrics = [
        json.loads(row) for row in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[0]["step"] == 0 and metrics[0]["train/loss"] is None
    assert metrics[1]["train/grad_norm"] == 3.4
    predictions = json.loads(
        (tmp_path / "examples/train-0-predictions.json").read_text()
    )
    assert (
        predictions[0]["target"] == "hello" and predictions[0]["prediction"] == "helo"
    )
    assert predictions[0]["wer"] == 1 and predictions[0]["cer"] == 0.2
    assert len(list((tmp_path / "examples").glob("*.wav"))) == 4
    assert len(list((tmp_path / "examples").glob("*.png"))) == 2
    archives = list((tmp_path / "comet-offline").rglob("*.zip"))
    assert archives
    archive_text = ""
    for path in archives:
        with zipfile.ZipFile(path) as archive:
            archive_text += "\n".join(archive.namelist())
            for name in archive.namelist():
                if name.endswith((".json", ".jsonl", ".txt")):
                    archive_text += archive.read(name).decode(errors="replace")
    for required in (
        "train/grad_norm",
        "predictions.csv",
        "original.wav",
        "augmented.wav",
        "spectrogram.png",
        "best.pt",
    ):
        assert required in archive_text, required
    assert secret not in archive_text
    resumed = CometWriter(
        tmp_path, {}, mode="offline", workspace=None, run_id="bs1uranz"
    )
    try:
        assert resumed.comet_key == key and resumed.run_id == "bs1uranz"
        resumed.log({"train/loss": 1.1}, 2)
    finally:
        resumed.finish()
    with pytest.raises(ValueError, match="checkpoint run identity"):
        CometWriter(tmp_path, {}, mode="offline", workspace=None, run_id="different")


def test_comet_identity_survives_actual_training_checkpoint_resume(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from conftest import create
    from src.datasets.data_utils import get_dataloaders
    from src.trainer import Trainer
    from src.utils.checkpoint import load_checkpoint
    from src.utils.init_utils import set_random_seed
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data, saved = tmp_path / "data", tmp_path / "run"
    create(data)
    with initialize_config_dir(version_base=None, config_dir=str(root / "src/configs")):
        config = compose(config_name="smoke")
    config.datasets.train.data_dir = config.datasets.dev_other.data_dir = str(data)
    config.trainer.save_dir = str(saved)
    config.trainer.max_steps = 4
    config.trainer.session_max_steps = 1
    config.trainer.log_examples_every = 10000

    def build(checkpoint=None):
        set_random_seed(config.trainer.seed)
        encoder = instantiate(config.text_encoder)
        loaders, _ = get_dataloaders(config, encoder, "cpu")
        model = instantiate(config.model, n_tokens=len(encoder))
        optimizer = instantiate(config.optimizer, params=model.parameters())
        scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)
        writer = CometWriter(
            saved,
            {},
            mode="disabled",
            run_id=checkpoint["run_id"] if checkpoint else "bs1uranz",
        )
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
            resume_checkpoint=checkpoint,
        )

    first = build()
    first.train()
    checkpoint = load_checkpoint(saved / "last.pt")
    registry = json.loads((saved / "comet-run.json").read_text())
    assert checkpoint["global_step"] == 1 and checkpoint["run_id"] == "bs1uranz"
    assert registry["logical_run_id"] == checkpoint["run_id"]
    assert checkpoint["optimizer"]["state"]
    second = build(checkpoint)
    assert (
        second.global_step == 1
        and second.batch_in_epoch == checkpoint["batch_in_epoch"]
    )
    for name, value in checkpoint["model"].items():
        assert torch.equal(second.model.state_dict()[name], value)
    second.train()
    resumed = load_checkpoint(saved / "last.pt")
    assert resumed["global_step"] == 2 and resumed["run_id"] == checkpoint["run_id"]
    assert (
        json.loads((saved / "comet-run.json").read_text())["experiment_key"]
        == registry["experiment_key"]
    )
