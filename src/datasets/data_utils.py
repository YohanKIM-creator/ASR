import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from src.datasets.batch_sampler import DurationBatchSampler
from src.datasets.collate import collate_fn


def get_dataloaders(config, text_encoder, device):
    loaders = {}
    for partition, dataset_config in config.datasets.items():
        training = partition == "train"
        if training and str(dataset_config.get("part", "")).startswith(("dev", "test")):
            raise ValueError("held-out LibriSpeech split cannot be training data")
        features = instantiate(config.features) if config.get("features") else None
        wave_augment = instantiate(config.augmentation) if training and config.get("augmentation") else None
        dataset = instantiate(
            dataset_config, text_encoder=text_encoder, features=features, wave_augment=wave_augment
        )
        dataset.set_epoch(0, config.trainer.seed if "trainer" in config else 42)
        sampler = DurationBatchSampler(
            dataset.durations,
            max_seconds=config.dataloader.max_seconds,
            max_items=config.dataloader.max_items,
            seed=config.trainer.seed if "trainer" in config else 42,
            shuffle=training,
        )
        generator = torch.Generator().manual_seed(42)  # loader creation must not consume model RNG
        loaders[partition] = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=config.dataloader.num_workers,
            pin_memory=str(device).startswith("cuda"),
            generator=generator,
            persistent_workers=False,
        )
    return loaders, {}
