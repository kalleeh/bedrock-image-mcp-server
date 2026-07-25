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
"""Pydantic models for Stable Image Ultra and Stable Image Core parameters.

Both models are text-to-image only. Verified against the Bedrock API: they reject
`mode`, `image`, `strength`, `style_preset`, `width` and `height` with a
ValidationException, and accept only lowercase `png` or `jpeg` as the output format.
"""

from awslabs.bedrock_image_mcp_server.consts import MAX_PROMPT_LENGTH_SD35, SD35_MAX_SEED
from awslabs.bedrock_image_mcp_server.models.sd35_models import AspectRatio
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class StableImageOutputFormat(str, Enum):
    """Output formats accepted by Stable Image Ultra and Core.

    These models reject webp, unlike the other Stability AI services.

    Attributes:
        JPEG: JPEG image format.
        PNG: PNG image format.
    """

    JPEG = 'jpeg'
    PNG = 'png'


class StableImageParams(BaseModel):
    """Parameters shared by Stable Image Ultra and Stable Image Core.

    Attributes:
        prompt: Text description of the image to generate (1-10,000 characters).
        aspect_ratio: Desired aspect ratio for the generated image.
        seed: Random seed for reproducible generation (0-4,294,967,294). 0 means random.
        negative_prompt: Text describing what to exclude from the image.
        output_format: Output image format (png or jpeg).
    """

    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH_SD35)
    aspect_ratio: AspectRatio = AspectRatio.RATIO_1_1
    seed: int = Field(default=0, ge=0, le=SD35_MAX_SEED)
    negative_prompt: Optional[str] = Field(default=None, max_length=MAX_PROMPT_LENGTH_SD35)
    output_format: StableImageOutputFormat = StableImageOutputFormat.PNG
