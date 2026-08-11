"""Online RGB clips for vae_only DualFlow stage-2 training."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from dualflow import logger
from dualflow.ltx.video_io import crop_video, read_video, resize_video


class OnlineVideoFolderDataset(Dataset):
    """Loads videos from a directory tree; yields `pixel_values` [F,C,H,W] in [0,1]."""

    def __init__(
        self,
        video_root: str | Path,
        *,
        total_source_frames: int,
        video_width: int,
        video_height: int,
        extensions: tuple[str, ...] = (".mp4", ".webm", ".avi", ".mkv"),
        explicit_paths: list[Path | str] | None = None,
    ) -> None:
        root = Path(video_root)
        if not root.is_dir():
            raise FileNotFoundError(f"video_root is not a directory: {root}")
        self.total_source_frames = total_source_frames
        self.video_width = video_width
        self.video_height = video_height
        if explicit_paths is not None:
            self.paths = sorted({Path(p).expanduser().resolve() for p in explicit_paths})
            for p in self.paths:
                if not p.is_file():
                    raise FileNotFoundError(f"explicit_paths entry is not a file: {p}")
            if not self.paths:
                raise ValueError("explicit_paths is empty")
            logger.info("OnlineVideoFolderDataset: %s explicit files", len(self.paths))
        else:
            paths: list[Path] = []
            for ext in extensions:
                paths.extend(root.rglob(f"*{ext}"))
            self.paths = sorted({p.resolve() for p in paths if p.is_file()})
            if not self.paths:
                raise ValueError(f"No videos with extensions {extensions} under {root}")
            logger.info(f"OnlineVideoFolderDataset: {len(self.paths)} files under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | float | str]:
        path = self.paths[idx]
        frames, fps = read_video(path, target_frames=self.total_source_frames)
        frames = resize_video(frames, self.video_width, self.video_height)
        frames = crop_video(frames, self.video_width, self.video_height)
        return {
            "pixel_values": frames,
            "fps": float(fps),
            "path": str(path),
        }
