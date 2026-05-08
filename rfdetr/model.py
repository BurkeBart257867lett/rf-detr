# Copyright 2024 Roboflow. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Core RF-DETR model definition and inference interface."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np


@dataclass
class DetectionResult:
    """Container for object detection results from a single image."""

    boxes: np.ndarray  # shape (N, 4), format xyxy, absolute pixel coords
    scores: np.ndarray  # shape (N,)
    labels: np.ndarray  # shape (N,), integer class indices
    class_names: Optional[List[str]] = field(default=None)

    def __len__(self) -> int:
        return len(self.scores)

    def __repr__(self) -> str:
        return (
            f"DetectionResult(num_detections={len(self)}, "
            f"classes={self.class_names})"
        )

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Return a new DetectionResult keeping only detections above *threshold*."""
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=self.labels[mask],
            class_names=self.class_names,
        )

    def to_dict(self) -> dict:
        """Serialize results to a plain dict — handy for JSON logging."""
        return {
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "labels": self.labels.tolist(),
            "class_names": self.class_names,
        }

    def top_k(self, k: int) -> "DetectionResult":
        """Return a new DetectionResult with only the top-k highest scoring detections."""
        if k >= len(self):
            return self
        indices = np.argsort(self.scores)[::-1][:k]
        return DetectionResult(
            boxes=self.boxes[indices],
            scores=self.scores[indices],
            labels=self.labels[indices],
            class_names=self.class_names,
        )


class RFDETRBase:
    """Base class shared by all RF-DETR model variants.

    Sub-classes must implement :py:meth:`_load_model` and
    :py:meth:`_run_inference`.
    """

    #: Mapping from variant name to pretrained checkpoint URL.
    MODEL_URLS: dict = {}

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_classes: int = 80,
        # Lowered default threshold from 0.5 to 0.4 to catch more detections
        # during my experiments — easy to tighten per-call via filter_by_confidence.
        confidence_threshold: float = 0.4,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_path: Path to a local checkpoint file or a URL.  When
                ``None`` the default pretrained weights are used.
            num_classes: Number of object categories the model was trained on.
            confidence_threshold: Detections below this score are discarded
                before results are returned.  Defaults to ``0.4``.
            device: PyTorch device string such as ``"cpu"`` or ``"cuda:0"``.
                Defaults to CUDA when available, otherwise CPU.
        """
        import torch  # local import so the module is importable without torch

        self.num_classes = num_classes
 