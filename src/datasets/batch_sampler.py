import random

from torch.utils.data import Sampler


class DurationBatchSampler(Sampler):
    """Deterministic length buckets with a padded audio-second memory budget."""

    def __init__(self, durations, max_seconds=100, max_items=24, seed=42, shuffle=True, bucket_size=500):
        self.durations = list(durations)
        self.max_seconds, self.max_items, self.seed = max_seconds, max_items, seed
        self.shuffle, self.bucket_size, self.epoch = shuffle, bucket_size, 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def batches(self):
        ids = list(range(len(self.durations)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(ids)
        ordered = []
        for start in range(0, len(ids), self.bucket_size):
            ordered.extend(sorted(ids[start : start + self.bucket_size], key=self.durations.__getitem__))
        batches, current, longest = [], [], 0
        for index in ordered:
            next_longest = max(longest, self.durations[index])
            if current and (
                next_longest * (len(current) + 1) > self.max_seconds or len(current) >= self.max_items
            ):
                batches.append(current)
                current, longest = [], 0
            current.append(index)
            longest = max(longest, self.durations[index])
        if current:
            batches.append(current)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self.batches())

    def __len__(self):
        return len(self.batches())
