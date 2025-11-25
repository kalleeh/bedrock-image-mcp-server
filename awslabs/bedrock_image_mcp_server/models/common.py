# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Common models and enums shared across all Bedrock image generation services."""

import base64
import os
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, List, Optional


class OutputFormat(str, Enum):
    """Supported output formats across all models.

    Attributes:
        JPEG: JPEG image format.
        PNG: PNG image format.
        WEBP: WebP image format.
    """
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class BedrockModelId(str, Enum):
    """Bedrock model identifiers for all supported image generation models.

    Attributes:
        NOVA_CANVAS: Amazon Nova Canvas model.
        SD35_LARGE: Stable Diffusion 3.5 Large model.
        STABLE_UPSCALE_CREATIVE: Stability AI creative upscale model.
        STABLE_UPSCALE_CONSERVATIVE: Stability AI conservative upscale model.
        STABLE_UPSCALE_FAST: Stability AI fast upscale model.
        STABLE_INPAINT: Stability AI inpainting model.
        STABLE_OUTPAINT: Stability AI outpainting model.
        STABLE_SEARCH_REPLACE: Stability AI search and replace model.
        STABLE_SEARCH_RECOLOR: Stability AI search and recolor model.
        STABLE_ERASE_OBJECT: Stability AI object removal model.
        STABLE_REMOVE_BACKGROUND: Stability AI background removal model.
        STABLE_CONTROL_SKETCH: Stability AI sketch-to-image control model.
        STABLE_CONTROL_STRUCTURE: Stability AI structure control model.
        STABLE_STYLE_GUIDE: Stability AI style guide model.
        STABLE_STYLE_TRANSFER: Stability AI style transfer model.
    """
    NOVA_CANVAS = "amazon.nova-canvas-v1:0"
    SD35_LARGE = "stability.sd3-5-large-v1:0"
    STABLE_UPSCALE_CREATIVE = "us.stability.stable-creative-upscale-v1:0"
    STABLE_UPSCALE_CONSERVATIVE = "us.stability.stable-conservative-upscale-v1:0"
    STABLE_UPSCALE_FAST = "us.stability.stable-fast-upscale-v1:0"
    STABLE_INPAINT = "us.stability.stable-image-inpaint-v1:0"
    STABLE_OUTPAINT = "us.stability.stable-outpaint-v1:0"
    STABLE_SEARCH_REPLACE = "us.stability.stable-image-search-replace-v1:0"
    STABLE_SEARCH_RECOLOR = "us.stability.stable-image-search-recolor-v1:0"
    STABLE_ERASE_OBJECT = "us.stability.stable-image-erase-object-v1:0"
    STABLE_REMOVE_BACKGROUND = "us.stability.stable-image-remove-background-v1:0"
    STABLE_CONTROL_SKETCH = "us.stability.stable-image-control-sketch-v1:0"
    STABLE_CONTROL_STRUCTURE = "us.stability.stable-image-control-structure-v1:0"
    STABLE_STYLE_GUIDE = "us.stability.stable-image-style-guide-v1:0"
    STABLE_STYLE_TRANSFER = "us.stability.stable-style-transfer-v1:0"


class ImageGenerationResponse(BaseModel):
    """Unified response model for all image generation services.

    This model represents the response from any Bedrock image generation service,
    providing a consistent interface across different models.

    Attributes:
        status: Status of the image generation request ('success' or 'error').
        message: Message describing the result or error.
        paths: List of absolute file paths to the generated image files.
        model_id: The Bedrock model ID used for generation.
        prompt: The text prompt used to generate the images, if applicable.
        seed: The seed value used for generation, if applicable.
        metadata: Additional metadata about the generation (e.g., finish_reasons, parameters).
    """
    status: str
    message: str
    paths: List[str]
    model_id: str
    prompt: Optional[str] = None
    seed: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseImageInput(BaseModel):
    """Base model for image inputs supporting both base64 and file paths.

    This model validates image inputs and provides a consistent interface
    for handling images from different sources.

    Attributes:
        image: Either a base64-encoded image string or a file path to an image.
    """
    image: str

    @field_validator('image')
    @classmethod
    def validate_image_input(cls, v: str) -> str:
        """Validate that the image is either valid base64 or an existing file path.

        Args:
            v: The image string to validate (base64 or file path).

        Returns:
            The validated image string.

        Raises:
            ValueError: If the image is neither valid base64 nor an existing file.
        """
        # Check if it's a file path
        if os.path.exists(v):
            return v

        # Check if it's valid base64
        try:
            # Try to decode as base64
            base64.b64decode(v, validate=True)
            return v
        except Exception:
            raise ValueError(
                "Image must be either a valid base64-encoded string or an existing file path"
            )
