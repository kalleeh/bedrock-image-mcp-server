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
"""Pydantic models for Stable Diffusion 3.5 Large parameters.

This module defines the data models and validation logic for SD3.5 Large
text-to-image and image-to-image generation.
"""

import base64
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from enum import Enum
from io import BytesIO
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class AspectRatio(str, Enum):
    """SD3.5 aspect ratio options.

    Attributes:
        RATIO_16_9: 16:9 widescreen landscape format.
        RATIO_1_1: 1:1 square format.
        RATIO_21_9: 21:9 ultra-wide format.
        RATIO_2_3: 2:3 portrait format.
        RATIO_3_2: 3:2 landscape format.
        RATIO_4_5: 4:5 portrait format.
        RATIO_5_4: 5:4 landscape format.
        RATIO_9_16: 9:16 vertical/mobile format.
        RATIO_9_21: 9:21 ultra-tall format.
    """
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_21_9 = "21:9"
    RATIO_2_3 = "2:3"
    RATIO_3_2 = "3:2"
    RATIO_4_5 = "4:5"
    RATIO_5_4 = "5:4"
    RATIO_9_16 = "9:16"
    RATIO_9_21 = "9:21"


class GenerationMode(str, Enum):
    """SD3.5 generation modes.

    Attributes:
        TEXT_TO_IMAGE: Generate image from text prompt only.
        IMAGE_TO_IMAGE: Transform existing image with text guidance.
    """
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


class SD35TextToImageParams(BaseModel):
    """Parameters for SD3.5 text-to-image generation.

    This model validates all parameters for generating images from text prompts
    using Stable Diffusion 3.5 Large.

    Attributes:
        prompt: Text description of the image to generate (1-10,000 characters).
        aspect_ratio: Desired aspect ratio for the generated image.
        seed: Random seed for reproducible generation (0-4,294,967,294).
        negative_prompt: Text describing what to exclude from the image.
        output_format: Output image format (jpeg, png, or webp).
    """
    prompt: str = Field(..., min_length=1, max_length=10000)
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    seed: int = Field(default=0, ge=0, le=4294967294)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    output_format: OutputFormat = OutputFormat.PNG


class SD35ImageToImageParams(BaseModel):
    """Parameters for SD3.5 image-to-image transformation.

    This model validates all parameters for transforming existing images
    using Stable Diffusion 3.5 Large with text guidance.

    Attributes:
        prompt: Text description guiding the transformation (1-10,000 characters).
        image: Base64-encoded input image.
        strength: Transformation intensity (0.0=preserve input, 1.0=ignore input).
        seed: Random seed for reproducible generation (0-4,294,967,294).
        negative_prompt: Text describing what to exclude from the image.
        output_format: Output image format (jpeg, png, or webp).
    """
    prompt: str = Field(..., min_length=1, max_length=10000)
    image: str  # base64 encoded
    strength: float = Field(..., ge=0.0, le=1.0)
    seed: int = Field(default=0, ge=0, le=4294967294)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    output_format: OutputFormat = OutputFormat.PNG

    @field_validator('image')
    @classmethod
    def validate_image_dimensions(cls, v: str) -> str:
        """Validate that the image meets minimum dimension requirements.

        SD3.5 requires images to be at least 64px on each side.
        Accepts either file paths or base64-encoded image strings.

        Args:
            v: File path or base64-encoded image string.

        Returns:
            The validated image string (unchanged).

        Raises:
            ValueError: If image dimensions are below minimum (64px per side).
        """
        import os

        try:
            # Check if it's a file path
            if os.path.exists(v):
                # Load from file
                image = Image.open(v)
            else:
                # Assume it's base64
                image_data = base64.b64decode(v)
                image = Image.open(BytesIO(image_data))

            # Get dimensions
            width, height = image.size

            # Validate minimum dimensions
            if width < 64 or height < 64:
                raise ValueError(
                    f"Image dimensions ({width}x{height}) are below minimum requirement "
                    f"of 64px per side for SD3.5"
                )

            return v

        except Exception as e:
            if isinstance(e, ValueError) and "dimensions" in str(e):
                raise
            raise ValueError(f"Invalid image data: {str(e)}")
