# Copyright 2024 Roboflow. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RF-DETR: Real-time object detection using transformer-based architecture.

This package provides a high-performance object detection model built on
the DETR (Detection Transformer) architecture, optimized for real-time
inference and fine-tuning on custom datasets.

Example:
    Basic inference usage::

        from rfdetr import RFDETRBase

        model = RFDETRBase()
        detections = model.predict("image.jpg")

    Fine-tuning on a custom dataset::

        from rfdetr import RFDETRBase

        model = RFDETRBase()
        model.train(
            dataset_dir="/path/to/dataset",
            epochs=50,
            batch_size=8,
        )
"""

from rfdetr.main import RFDETRBase, RFDETRLarge

__version__ = "1.0.0"
__author__ = "Roboflow"
__email__ = "support@roboflow.com"
__license__ = "Apache-2.0"

__all__ = [
    "RFDETRBase",
    "RFDETRLarge",
    "__version__",
]
