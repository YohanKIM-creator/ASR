import math

import numpy as np


def logadd(*values):
    largest = max(values)
    if largest == -math.inf:
        return largest
    return largest + math.log(sum(math.exp(v - largest) for v in values))


def prefix_beam_search(log_probs, beam_size=32, blank_id=0, token_topk=None):
    """Return [(token_tuple, log_mass), ...], summing all retained CTC paths.

    Separate blank/nonblank masses are essential for repeated labels. token_topk
    is optional acoustic pruning; None gives the exact unpruned token expansion.
    Beam pruning still approximates unless beam_size can hold every prefix.
    """
    if hasattr(log_probs, "detach"):
        log_probs = log_probs.detach().float().cpu().numpy()
    log_probs = np.asarray(log_probs, dtype=np.float64)
    if log_probs.ndim != 2 or beam_size < 1:
        raise ValueError("expected T,V log probabilities and beam_size >= 1")
    if not 0 <= blank_id < log_probs.shape[1]:
        raise ValueError("blank_id outside vocabulary")
    beam = {(): (0.0, -math.inf)}
    for frame in log_probs:
        if token_topk is None or token_topk >= len(frame):
            candidates = range(len(frame))
        else:
            if token_topk < 1:
                raise ValueError("token_topk must be positive")
            candidates = set(np.argsort(frame)[-token_topk:].tolist()) | {blank_id}
        updated = {}

        def add(prefix, blank=-math.inf, nonblank=-math.inf):
            old_blank, old_nonblank = updated.get(prefix, (-math.inf, -math.inf))
            updated[prefix] = (logadd(old_blank, blank), logadd(old_nonblank, nonblank))

        for prefix, (p_blank, p_nonblank) in beam.items():
            total = logadd(p_blank, p_nonblank)
            for token in candidates:
                prob = float(frame[token])
                if token == blank_id:
                    add(prefix, blank=total + prob)
                elif prefix and token == prefix[-1]:
                    add(prefix, nonblank=p_nonblank + prob)
                    add(prefix + (int(token),), nonblank=p_blank + prob)
                else:
                    add(prefix + (int(token),), nonblank=total + prob)
        ranked = sorted(updated.items(), key=lambda kv: logadd(*kv[1]), reverse=True)
        beam = dict(ranked[:beam_size])
    return sorted(
        [(prefix, logadd(*mass)) for prefix, mass in beam.items()],
        key=lambda entry: entry[1],
        reverse=True,
    )
