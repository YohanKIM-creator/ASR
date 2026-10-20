"""Widen a Conformer checkpoint."""

import argparse
import copy
import sys
from pathlib import Path


def repeat_rows(tensor, row_factor, column_factor=1, groups=1):
    shape = tensor.shape
    grouped = tensor.reshape(groups, shape[0] // groups, *shape[1:])
    repetitions = [1, row_factor] + [column_factor] + [1] * (tensor.ndim - 2)
    return grouped.repeat(*repetitions).reshape(
        shape[0] * row_factor, shape[1] * column_factor, *shape[2:]
    )


def repeat_bias(tensor, factor, groups=1):
    return tensor.reshape(groups, -1).repeat(1, factor).flatten()


def _separate_convolution_copies(model):
    """Separate copied channels while preserving each outgoing pair sum."""
    import torch

    def copied_rows(value, groups=1):
        if value is None:
            return True
        if value.shape[0] % (2 * groups):
            return False
        grouped = value.reshape(groups, value.shape[0] // groups, *value.shape[1:])
        half = grouped.shape[1] // 2
        return torch.equal(grouped[:, :half], grouped[:, half:])

    generator = torch.Generator().manual_seed(20261013)
    changed = []
    with torch.no_grad():
        for index, block in enumerate(model.blocks):
            module = block.convolution
            weight = module.pointwise_out.weight
            if weight.shape[1] % 2:
                continue
            half = weight.shape[1] // 2
            left, right = weight[:, :half], weight[:, half:]
            copied = (
                copied_rows(module.pointwise_in.weight, 2)
                and copied_rows(module.pointwise_in.bias, 2)
                and copied_rows(module.depthwise.weight)
                and copied_rows(module.depthwise.bias)
                and copied_rows(module.post_norm.weight)
                and copied_rows(module.post_norm.bias)
                and torch.equal(left, right)
            )
            if not copied:
                continue
            pair_sum = left + right
            magnitude = left.square().mean().sqrt()
            noise = torch.randn(left.shape, generator=generator, dtype=weight.dtype).to(
                weight.device
            )
            left.add_(noise * magnitude * 0.05)
            right.copy_(pair_sum - left)
            changed.append(index)
    model._widening_conv_split = {
        "kind": "sum_preserving_outgoing_column_redistribution",
        "seed": 20261013,
        "relative_noise_rms": 0.05,
        "blocks": changed,
    }


def widen_model(old, new):
    from torch import nn

    if len(old.blocks) != len(new.blocks):
        raise ValueError("Widening keeps block count unchanged")
    for before, after in zip(old.blocks, new.blocks):
        if before.attention.head_dim != after.attention.head_dim:
            raise ValueError(
                "Keep RoPE head dimension unchanged by scaling heads with width"
            )
    old_modules, new_modules = dict(old.named_modules()), dict(new.named_modules())
    old_state, new_state = old.state_dict(), new.state_dict()
    if set(old_state) != set(new_state):
        raise ValueError("Unexpected Conformer state layout")
    mapped = set()
    for name, module in old_modules.items():
        target = new_modules[name]
        for parameter, value in module.named_parameters(recurse=False):
            key = name + "." + parameter if name else parameter
            wanted = new_state[key]
            if value.shape == wanted.shape:
                result = value.detach().clone()
            elif isinstance(module, nn.LayerNorm):
                if value.ndim != 1 or wanted.numel() % value.numel():
                    raise ValueError("Unexpected LayerNorm geometry: " + key)
                result = value.detach().repeat(wanted.numel() // value.numel())
            elif isinstance(module, nn.Linear):
                rows = target.out_features // module.out_features
                columns = target.in_features // module.in_features
                if (
                    target.out_features != rows * module.out_features
                    or target.in_features != columns * module.in_features
                ):
                    raise ValueError("Noninteger linear widening: " + key)
                groups = 3 if name.endswith(".qkv") else 1
                result = (
                    repeat_rows(value.detach(), rows, columns, groups) / columns
                    if parameter == "weight"
                    else repeat_bias(value.detach(), rows, groups)
                )
            elif isinstance(module, nn.Conv1d):
                rows = target.out_channels // module.out_channels
                columns = target.in_channels // module.in_channels
                if module.groups == module.in_channels == module.out_channels:
                    if (
                        target.groups != target.in_channels
                        or target.out_channels != target.in_channels
                    ):
                        raise ValueError("Depthwise groups must widen with channels")
                    result = (
                        value.detach().repeat(rows, 1, 1)
                        if parameter == "weight"
                        else value.detach().repeat(rows)
                    )
                elif module.groups == target.groups == 1:
                    groups = 2 if name.endswith(".pointwise_in") else 1
                    result = (
                        repeat_rows(value.detach(), rows, columns, groups) / columns
                        if parameter == "weight"
                        else repeat_bias(value.detach(), rows, groups)
                    )
                else:
                    raise ValueError("Unsupported grouped convolution: " + key)
            else:
                raise ValueError("Unsupported parameter widening: " + key)
            if result.shape != wanted.shape:
                raise ValueError("Mapped shape mismatch: " + key)
            new_state[key] = result.to(
                device=wanted.device, dtype=wanted.dtype
            ).contiguous()
            mapped.add(key)
    for key, value in old_state.items():
        if key not in mapped:
            if value.shape != new_state[key].shape:
                raise ValueError("Unsupported buffer widening: " + key)
            new_state[key] = value.detach().clone()
    new.load_state_dict(new_state, strict=True)
    _separate_convolution_copies(new)
    return new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.factor < 2:
        parser.error("factor must be at least two")
    if args.output.exists():
        raise RuntimeError(
            "Output already exists; inspect the frozen initialization before retry"
        )
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import torch
    from hydra.utils import instantiate
    from src.text_encoder import CTCTextEncoder
    from src.utils.checkpoint import atomic_torch_save

    torch.set_num_threads(2)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --device cuda")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = CTCTextEncoder.from_state_dict(state["text_encoder"])
    configuration = copy.deepcopy(state["config"])
    old_config = copy.deepcopy(configuration["model"])
    configuration["model"]["dim"] *= args.factor
    configuration["model"]["heads"] *= args.factor
    configuration["model"][
        "gradient_checkpointing"
    ] = not args.no_gradient_checkpointing
    old = instantiate(old_config, n_tokens=len(encoder)).to(device).eval()
    old.load_state_dict(state["model"], strict=True)
    new = instantiate(configuration["model"], n_tokens=len(encoder)).to(device).eval()
    widen_model(old, new)
    generator = torch.Generator().manual_seed(20261013)
    features = torch.randn(3, old_config["n_feats"], 129, generator=generator)
    features = features.to(device)
    lengths = torch.tensor([129, 97, 65], device=device)
    with torch.inference_mode():
        before = old(features, lengths)
        after = new(features, lengths)
    assert torch.equal(before["log_probs_length"], after["log_probs_length"])
    difference = float((before["log_probs"] - after["log_probs"]).abs().max())
    if not torch.allclose(
        before["log_probs"], after["log_probs"], atol=2e-5, rtol=2e-5
    ):
        raise RuntimeError("FP32 function-preserving check failed: " + str(difference))
    provenance = {
        "kind": "channel_and_head_widening",
        "teacher_step": state["global_step"],
        "teacher_git_sha": state["git_sha"],
        "teacher_run_id": state.get("run_id"),
        "old_parameters": old.num_parameters,
        "new_parameters": new.num_parameters,
        "old_dim": old_config["dim"],
        "new_dim": configuration["model"]["dim"],
        "old_heads": old_config["heads"],
        "new_heads": configuration["model"]["heads"],
        "preserved_head_dim": old.blocks[0].attention.head_dim,
        "max_fp32_logprob_difference": difference,
        "convolution_copy_split": new._widening_conv_split,
        "initialization_only": True,
        "quality_verified": False,
        "test_data_used": False,
    }
    output = {
        "model": new.state_dict(),
        "text_encoder": state["text_encoder"],
        "config": configuration,
        "global_step": 0,
        "best_wer": state["best_wer"],
        "git_sha": state["git_sha"],
        "initialization_provenance": provenance,
    }
    atomic_torch_save(output, args.output)
    print(
        f"Saved {args.output}: {old_config['dim']} -> {configuration['model']['dim']} channels, "
        f"{old.num_parameters:,} -> {new.num_parameters:,} parameters",
        flush=True,
    )


if __name__ == "__main__":
    main()
