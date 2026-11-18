"""Precompute batched JDC F0 contours as Quick Convert tensor resources."""
from __future__ import annotations

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from quick_convert.data.types import AudioBatch

from . import train as _train  # noqa: F401  # registers QC's LibriSpeech Hydra config
from .f0_resource import OnlineF0


def precompute(dataset, extractor: OnlineF0, output_root: Path, *, batch_size: int = 8,
               num_workers: int = 4, overwrite: bool = False, device: str = "cpu") -> tuple[int, int]:
    """Write one [T,1] Hz tensor per sample; return (written, skipped)."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    extractor = extractor.to(device).eval()
    written = skipped = 0
    loader = dataset.make_dataloader(batch_size=batch_size, num_workers=num_workers)
    for batch in loader:
        if not isinstance(batch, AudioBatch):
            raise TypeError("Precomputation requires an AudioBatch")
        paths = []
        for split, utt_id in zip(batch.splits, batch.utt_ids):
            if not split or not utt_id or Path(utt_id).name != utt_id:
                raise ValueError(f"Invalid cache key: {split!r}/{utt_id!r}")
            paths.append(output_root / split / f"{utt_id}.pt")
        if len(set(paths)) != len(paths):
            raise ValueError("Duplicate cache keys within batch")
        pending = [i for i, path in enumerate(paths) if overwrite or not path.exists()]
        skipped += len(paths) - len(pending)
        if not pending:
            continue
        # Avoid recomputing already-cached samples, including the JDC forward.
        samples = [batch[i] for i in pending]
        sub_batch = AudioBatch.from_samples(samples)
        sub_batch.waveforms = sub_batch.waveforms.to(device)
        sub_batch.lengths = sub_batch.lengths.to(device)
        sub_batch.sample_rates = sub_batch.sample_rates.to(device)
        with torch.inference_mode():
            result = extractor(sub_batch)
        for index, contour, length in zip(pending, result.values, result.lengths):
            path = paths[index]
            path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic rename prevents partially written sidecars on interruption.
            temporary = path.with_name(path.name + ".tmp")
            torch.save(contour[:int(length)].detach().cpu().float().contiguous(), temporary)
            temporary.replace(path)
            written += 1
    return written, skipped


@hydra.main(version_base=None, config_path="configs", config_name="precompute_f0")
def main(cfg):
    output_root = Path(cfg.f0_cache_root).expanduser().resolve()
    extractor = OnlineF0(backend="jdc", checkpoint_path=cfg.get("checkpoint_path"))
    total_written = total_skipped = 0
    for split in cfg.splits:
        dataset = instantiate(cfg.dataset, splits=[split], load=["audio"],
                              resource_providers=[], target_sr=16000)
        written, skipped = precompute(
            dataset, extractor, output_root,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            overwrite=cfg.overwrite, device=cfg.device,
        )
        print(f"{split}: wrote {written}, skipped {skipped}")
        total_written += written
        total_skipped += skipped
    print(f"Total: wrote {total_written}, skipped {total_skipped}; cache: {output_root}")


if __name__ == "__main__":
    main()
