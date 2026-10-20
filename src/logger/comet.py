import json
import os
import re
import uuid
from pathlib import Path

from src.logger.wandb import WandBWriter


def redact_config(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if str(key).lower()
            in {
                "api_key",
                "comet_api_key",
                "wandb_api_key",
                "github_token",
                "access_token",
                "password",
                "secret",
                "credentials",
            }
            else redact_config(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    return value


class CometWriter(WandBWriter):
    def __init__(
        self,
        directory,
        project_config,
        project_name="hse-asr-conformer",
        workspace=None,
        run_id=None,
        run_name="conformer-char",
        mode="online",
        logger=None,
        log_checkpoints=True,
        credential_path=None,
        **kwargs,
    ):
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("Comet mode must be online/offline/disabled")
        super().__init__(
            directory,
            project_config,
            run_id=run_id,
            mode="disabled",
            logger=logger,
            log_checkpoints=log_checkpoints,
        )
        self.mode = mode
        self.experiment = None
        registry = self.directory / "comet-run.json"
        saved = json.loads(registry.read_text()) if registry.is_file() else None
        if saved:
            if run_id is not None and run_id != saved["logical_run_id"]:
                raise ValueError(
                    "Comet registry does not match checkpoint run identity"
                )
            if project_name != saved["project_name"] or (
                workspace and workspace != saved["workspace"]
            ):
                raise ValueError("Comet project/workspace changed for an existing run")
            self.run_id, self.comet_key = (
                saved["logical_run_id"],
                saved["experiment_key"],
            )
            workspace = saved["workspace"]
        else:
            # Old checkpoints may carry a W&B run ID; Comet needs a separate ID.
            self.comet_key = (
                run_id
                if run_id and re.fullmatch(r"[0-9a-f]{32}", run_id)
                else uuid.uuid4().hex
            )
            self.run_id = run_id or self.comet_key
        if not re.fullmatch(r"[0-9a-f]{32}", self.comet_key):
            raise ValueError("Invalid saved Comet experiment key")
        self.url = None
        if mode != "disabled":
            if mode == "online" and not os.environ.get("COMET_API_KEY"):
                private = (
                    Path(credential_path)
                    if credential_path
                    else Path(__file__).resolve().parents[2]
                    / ".secrets/credentials.json"
                )
                if private.is_file():
                    credential = json.loads(private.read_text()).get("comet")
                    if credential:
                        os.environ["COMET_API_KEY"] = credential
                if not os.environ.get("COMET_API_KEY"):
                    raise RuntimeError(
                        "Set COMET_API_KEY or ignored credentials before online logging"
                    )
            os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
            import comet_ml

            settings = comet_ml.ExperimentConfig(
                log_code=False,
                log_graph=False,
                parse_args=False,
                log_git_metadata=False,
                log_git_patch=False,
                log_env_details=False,
                log_env_gpu=False,
                log_env_host=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
                auto_output_logging=False,
                auto_param_logging=False,
                auto_metric_logging=False,
                auto_log_co2=False,
                display_summary_level=0,
                offline_directory=str(self.directory / "comet-offline"),
                name=run_name,
            )
            self.experiment = comet_ml.start(
                workspace=workspace,
                project_name=project_name,
                experiment_key=self.comet_key,
                mode="get_or_create",
                online=mode == "online",
                experiment_config=settings,
            )
            if (
                self.experiment is None
                or self.experiment.disabled
                or not self.experiment.alive
            ):
                raise RuntimeError("Comet experiment did not initialize successfully")
            if self.experiment.get_key() != self.comet_key:
                raise RuntimeError("Comet returned a different experiment identity")
            self.url = self.experiment.url
            self.experiment.log_parameters(redact_config(project_config))
            self.experiment.log_other("logical_run_id", self.run_id)
            self.experiment.log_other("tracker_backend", "comet")
        state = {
            "tracker": "comet",
            "experiment_key": self.comet_key,
            "logical_run_id": self.run_id,
            "workspace": workspace,
            "project_name": project_name,
            "url": self.url,
        }
        temporary = registry.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, registry)

    def log(self, metrics, step):
        super().log(metrics, step)
        if self.experiment is not None:
            values = {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
            for key, value in metrics.items():
                if value is not None and not isinstance(value, (int, float)):
                    self.experiment.log_other(f"{key}/step-{step}", str(value))
            self.experiment.log_metrics(
                {"trainer/global_step": int(step), **values}, step=int(step)
            )

    def log_examples(
        self,
        batch,
        predictions,
        text_encoder,
        step,
        partition,
        sample_rate=16000,
        count=3,
    ):
        super().log_examples(
            batch, predictions, text_encoder, step, partition, sample_rate, count
        )
        if self.experiment is None:
            return
        prefix = f"{partition}-{step}-"
        table = self.examples_dir / f"{partition}-{step}-predictions.json"
        rows = json.loads(table.read_text())
        columns = list(rows[0]) if rows else []
        self.experiment.log_table(
            filename=f"{partition}-{step}-predictions.csv",
            tabular_data=[[row[key] for key in columns] for row in rows],
            headers=columns,
            step=int(step),
        )
        self.experiment.log_asset(
            str(table),
            file_name=table.name,
            overwrite=True,
            metadata={"step": int(step), "partition": partition},
        )
        for path in sorted(self.examples_dir.glob(prefix + "*.wav")):
            self.experiment.log_audio(
                str(path),
                file_name=path.name,
                overwrite=True,
                step=int(step),
                metadata={"partition": partition},
            )
        for path in sorted(self.examples_dir.glob(prefix + "*.png")):
            self.experiment.log_image(
                str(path), name=path.name, overwrite=True, step=int(step)
            )

    def publish_best(self, path, metadata):
        if self.experiment is not None and self.log_checkpoints:
            self.experiment.log_model(
                "ctc-conformer-" + self.run_id,
                str(path),
                file_name="best.pt",
                overwrite=True,
                metadata=metadata,
            )

    def finish(self):
        if self.experiment is not None:
            self.experiment.end()
