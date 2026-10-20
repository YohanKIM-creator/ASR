from src.logger.wandb import WandBWriter
from src.logger.comet import CometWriter

__all__ = ["WandBWriter", "CometWriter"]


def create_writer(backend="comet", *args, **kwargs):
    writers = {"comet": CometWriter, "wandb": WandBWriter}
    if backend not in writers:
        raise ValueError("unknown tracker backend")
    return writers[backend](*args, **kwargs)
