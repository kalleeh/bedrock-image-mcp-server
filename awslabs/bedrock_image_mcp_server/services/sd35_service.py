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
"""Service implementation for Stable Diffusion 3.5 Large.

This module provides functions for text-to-image and image-to-image generation
using Stable Diffusion 3.5 Large through AWS Bedrock.
"""

import base64
import os
from awslabs.bedrock_image_mcp_server.consts import SD35_LARGE_MODEL_ID
from awslabs.bedrock_image_mcp_server.models.common import ImageGenerationResponse
from awslabs.bedrock_image_mcp_server.models.sd35_models import (
    GenerationMode,
    SD35ImageToImageParams,
    SD35TextToImageParams,
)
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    invoke_bedrock_model,
    save_images,
)
from loguru import logger
from typing import TYPE_CHECKING, Any, Dict, Optional, Union


if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
else:
    BedrockRuntimeClient = object


def build_sd35_request(
    params: Union[SD35TextToImageParams, SD35ImageToImageParams]
) -> Dict[str, Any]:
    """Build API request body for SD3.5.

    Constructs the request payload for Stable Diffusion 3.5 Large API calls,
    handling both text-to-image and image-to-image modes.

    Args:
        params: Either SD35TextToImageParams or SD35ImageToImageParams.

    Returns:
        Dictionary containing the formatted API request body.
    """
    # Base request structure
    request_body: Dict[str, Any] = {
        "prompt": params.prompt,
        "seed": params.seed,
        "output_format": params.output_format.value,
    }

    # Add negative prompt if provided
    if params.negative_prompt:
        request_body["negative_prompt"] = params.negative_prompt

    # Handle text-to-image vs image-to-image
    if isinstance(params, SD35TextToImageParams):
        # Text-to-image mode
        request_body["mode"] = GenerationMode.TEXT_TO_IMAGE.value
        request_body["aspect_ratio"] = params.aspect_ratio.value
    else:
        # Image-to-image mode
        request_body["mode"] = GenerationMode.IMAGE_TO_IMAGE.value
        request_body["image"] = params.image
        request_body["strength"] = params.strength

    logger.debug(f"Built SD3.5 request with mode: {request_body.get('mode')}")
    return request_body


async def generate_text_to_image(
    params: SD35TextToImageParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Generate image from text using SD3.5.

    Workflow:
    1. Validate parameters (handled by Pydantic model)
    2. Build API request body
    3. Invoke Bedrock model (with automatic retries via boto3)
    4. Save generated images
    5. Return response with file paths

    Args:
        params: Validated SD3.5 text-to-image parameters.
        bedrock_client: BedrockRuntimeClient object with retry configuration.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, paths, and metadata.

    Raises:
        BedrockAPIError: On API failures with detailed error classification.
        ContentFilterError: On content filtering.
    """
    logger.info(
        f"Generating SD3.5 text-to-image with aspect ratio: {params.aspect_ratio.value}",
        extra={
            'model': 'sd3.5-large',
            'aspect_ratio': params.aspect_ratio.value,
            'seed': params.seed,
            'prompt_length': len(params.prompt),
            'has_negative_prompt': params.negative_prompt is not None
        }
    )

    try:
        # Build request body
        request_body = build_sd35_request(params)

        # Invoke Bedrock model
        result = await invoke_bedrock_model(
            model_id=SD35_LARGE_MODEL_ID,
            request_body=request_body,
            bedrock_client=bedrock_client
        )

        # Extract images from response
        base64_images = result.get('images', [])
        if not base64_images:
            raise ValueError("No images returned from Bedrock API")

        # Determine filename prefix
        if filename:
            filename_prefix = filename
        else:
            filename_prefix = 'sd35'

        # Save images
        saved_paths = save_images(
            base64_images=base64_images,
            workspace_dir=workspace_dir,
            filename_prefix=filename_prefix,
            output_format=params.output_format
        )

        # Build response
        response = ImageGenerationResponse(
            status='success',
            message=f'Successfully generated {len(saved_paths)} image(s) using SD3.5',
            paths=saved_paths,
            model_id=SD35_LARGE_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed,
            metadata={
                'aspect_ratio': params.aspect_ratio.value,
                'mode': GenerationMode.TEXT_TO_IMAGE.value,
                'finish_reasons': result.get('finish_reasons', [])
            }
        )

        logger.info(
            f"SD3.5 text-to-image generation successful: {len(saved_paths)} image(s)",
            extra={
                'model': 'sd3.5-large',
                'images_count': len(saved_paths),
                'aspect_ratio': params.aspect_ratio.value,
                'output_dir': workspace_dir or 'current_directory'
            }
        )
        return response

    except Exception as e:
        logger.error(
            f"SD3.5 text-to-image generation failed: {str(e)}",
            extra={
                'model': 'sd3.5-large',
                'error_type': type(e).__name__,
                'aspect_ratio': params.aspect_ratio.value
            }
        )
        raise


async def generate_image_to_image(
    params: SD35ImageToImageParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Transform image using SD3.5.

    Workflow:
    1. Validate parameters (handled by Pydantic model)
    2. Handle image input (file path or base64)
    3. Validate image dimensions
    4. Build API request body with mode="image-to-image"
    5. Invoke Bedrock model (with automatic retries via boto3)
    6. Save generated images
    7. Return response with file paths

    Args:
        params: Validated SD3.5 image-to-image parameters.
        bedrock_client: BedrockRuntimeClient object with retry configuration.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, paths, and metadata.

    Raises:
        BedrockAPIError: On API failures with detailed error classification.
        ContentFilterError: On content filtering.
    """
    logger.info(
        f"Generating SD3.5 image-to-image with strength: {params.strength}",
        extra={
            'model': 'sd3.5-large',
            'mode': 'image-to-image',
            'strength': params.strength,
            'seed': params.seed,
            'prompt_length': len(params.prompt)
        }
    )

    try:
        # Handle file path input - convert to base64 if needed
        image_data = params.image
        if os.path.exists(params.image):
            logger.debug(f"Loading image from file: {params.image}")
            with open(params.image, 'rb') as f:
                image_bytes = f.read()
                image_data = base64.b64encode(image_bytes).decode('utf-8')

            # Update params with base64 data for validation
            params.image = image_data

        # Build request body
        request_body = build_sd35_request(params)

        # Invoke Bedrock model
        result = await invoke_bedrock_model(
            model_id=SD35_LARGE_MODEL_ID,
            request_body=request_body,
            bedrock_client=bedrock_client
        )

        # Extract images from response
        base64_images = result.get('images', [])
        if not base64_images:
            raise ValueError("No images returned from Bedrock API")

        # Determine filename prefix
        if filename:
            filename_prefix = filename
        else:
            filename_prefix = 'sd35_transform'

        # Save images
        saved_paths = save_images(
            base64_images=base64_images,
            workspace_dir=workspace_dir,
            filename_prefix=filename_prefix,
            output_format=params.output_format
        )

        # Build response
        response = ImageGenerationResponse(
            status='success',
            message='Successfully transformed image using SD3.5',
            paths=saved_paths,
            model_id=SD35_LARGE_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed,
            metadata={
                'strength': params.strength,
                'mode': GenerationMode.IMAGE_TO_IMAGE.value,
                'finish_reasons': result.get('finish_reasons', [])
            }
        )

        logger.info(
            f"SD3.5 image-to-image transformation successful: {len(saved_paths)} image(s)",
            extra={
                'model': 'sd3.5-large',
                'mode': 'image-to-image',
                'images_count': len(saved_paths),
                'strength': params.strength
            }
        )
        return response

    except Exception as e:
        logger.error(
            f"SD3.5 image-to-image transformation failed: {str(e)}",
            extra={
                'model': 'sd3.5-large',
                'mode': 'image-to-image',
                'error_type': type(e).__name__,
                'strength': params.strength
            }
        )
        raise
