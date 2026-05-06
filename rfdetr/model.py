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
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_path: Path to a local checkpoint file or a URL.  When
                ``None`` the default pretrained weights are used.
            num_classes: Number of object categories the model was trained on.
            confidence_threshold: Detections below this score are discarded
                before results are returned.
            device: PyTorch device string such as ``"cpu"`` or ``"cuda:0"``.
                Defaults to CUDA when available, otherwise CPU.
        """
        import torch  # local import so the module is importable without torch

        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(model_path) if model_path else None
        self._model = None  # populated by _load_model()
        self._load_model()

    # ------------------------------------------------------------------
    # Sub-class interface
    # ------------------------------------------------------------------

    def _load_model(self) -> None:  # pragma: no cover
        """Load model weights into ``self._model``.  Must be overridden."""
        raise NotImplementedError

    def _run_inference(
        self, images: List[np.ndarray]
    ) -> List[DetectionResult]:  # pragma: no cover
        """Run a forward pass.  Must be overridden by sub-classes."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Run inference on one or more images.

        Args:
            images: A single image as an ``(H, W, 3)`` NumPy array *or* a list
                of such arrays.  Images are expected in **BGR** channel order
                (OpenCV convention).
            confidence_threshold: Override the instance-level threshold for
                this call only.

        Returns:
            A list of :class:`DetectionResult` objects, one per input image.
        """
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]

        if not images:
            warnings.warn("predict() received an empty list of images.", stacklevel=2)
            return []

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        results = self._run_inference(images)
        return [r.filter_by_confidence(threshold) for r in results]
