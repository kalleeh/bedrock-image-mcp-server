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
"""Stability AI upscaling services for AWS Bedrock.

This module provides creative, conservative, and fast upscaling services
that can upscale images to 4K resolution with various enhancement options.
"""

import os
from awslabs.bedrock_image_mcp_server.consts import (
    MAX_CONSERVATIVE_UPSCALE_INPUT_PIXELS,
    MAX_CREATIVE_UPSCALE_INPUT_PIXELS,
    MAX_FAST_UPSCALE_INPUT_PIXELS,
    MIN_FAST_UPSCALE_INPUT_PIXELS,
    MIN_IMAGE_DIMENSION,
    STABLE_UPSCALE_CONSERVATIVE_MODEL_ID,
    STABLE_UPSCALE_CREATIVE_MODEL_ID,
    STABLE_UPSCALE_FAST_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import ImageGenerationResponse
from awslabs.bedrock_image_mcp_server.models.stability_models import (
    ConservativeUpscaleParams,
    CreativeUpscaleParams,
    FastUpscaleParams,
)
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    invoke_bedrock_model,
    save_images,
)
from awslabs.bedrock_image_mcp_server.utils.image_utils import (
    decode_base64_image,
    encode_image_file,
    validate_image_dimensions,
)
from loguru import logger
from typing import TYPE_CHECKING, Any, Dict, Optional


if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
else:
    BedrockRuntimeClient = object


async def upscale_creative(
    params: CreativeUpscaleParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Upscale image to 4K with creative enhancement.

    Creative upscaling uses AI to enhance and add details while upscaling
    images to 4K resolution (20-40x upscale). Best for low-resolution or
    degraded images that need improvement.

    Workflow:
    1. Validate input image (64x64 to 1MP)
    2. Warn if image too large for creative upscaling
    3. Build request with creativity parameter
    4. Invoke creative upscale model
    5. Save upscaled image

    Args:
        params: CreativeUpscaleParams with image, prompt, and creativity settings.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting creative upscale')

    # Handle image input (file path or base64)
    if os.path.exists(params.image):
        logger.debug(f'Encoding image from file: {params.image}')
        image_base64 = encode_image_file(params.image)
    else:
        logger.debug('Using provided base64 image')
        image_base64 = params.image

    # Validate image dimensions
    image_bytes = decode_base64_image(image_base64)
    width, height = validate_image_dimensions(
        image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION,
        max_pixels=MAX_CREATIVE_UPSCALE_INPUT_PIXELS
    )

    total_pixels = width * height
    logger.info(f'Input image dimensions: {width}x{height} ({total_pixels} pixels)')

    # Warn if image is too large for optimal creative upscaling
    if total_pixels > MAX_CREATIVE_UPSCALE_INPUT_PIXELS:
        logger.warning(
            f'Input image has {total_pixels} pixels, exceeding recommended '
            f'{MAX_CREATIVE_UPSCALE_INPUT_PIXELS} pixels for creative upscaling. '
            'Consider using conservative upscale for larger images.'
        )

    # Build request body
    request_body: Dict[str, Any] = {
        'image': image_base64,
        'prompt': params.prompt,
        'creativity': params.creativity,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    # Add optional parameters
    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt
    if params.style_preset:
        request_body['style_preset'] = params.style_preset.value

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    # Invoke Bedrock model
    result = await invoke_bedrock_model(
        model_id=STABLE_UPSCALE_CREATIVE_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client
    )

    # Extract images from response
    images = result.get('images', [])
    if not images:
        logger.error('No images returned from creative upscale')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=STABLE_UPSCALE_CREATIVE_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed
        )

    # Save images
    filename_prefix = filename or 'upscale_creative'
    saved_paths = save_images(
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename_prefix,
        output_format=params.output_format
    )

    logger.info(f'Creative upscale completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message='Successfully upscaled image with creative enhancement',
        paths=saved_paths,
        model_id=STABLE_UPSCALE_CREATIVE_MODEL_ID,
        prompt=params.prompt,
        seed=params.seed,
        metadata={
            'creativity': params.creativity,
            'input_dimensions': f'{width}x{height}',
            'style_preset': params.style_preset.value if params.style_preset else None
        }
    )


async def upscale_conservative(
    params: ConservativeUpscaleParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Upscale image to 4K preserving details.

    Conservative upscaling increases resolution to 4K while preserving
    the original image characteristics with minimal alterations. Best for
    images that already have good quality but need higher resolution.

    Workflow:
    1. Validate input image (64x64 to 9.4MP)
    2. Build request without creativity parameter
    3. Invoke conservative upscale model
    4. Save upscaled image

    Args:
        params: ConservativeUpscaleParams with image and prompt.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting conservative upscale')

    # Handle image input (file path or base64)
    if os.path.exists(params.image):
        logger.debug(f'Encoding image from file: {params.image}')
        image_base64 = encode_image_file(params.image)
    else:
        logger.debug('Using provided base64 image')
        image_base64 = params.image

    # Validate image dimensions
    image_bytes = decode_base64_image(image_base64)
    width, height = validate_image_dimensions(
        image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION,
        max_pixels=MAX_CONSERVATIVE_UPSCALE_INPUT_PIXELS
    )

    total_pixels = width * height
    logger.info(f'Input image dimensions: {width}x{height} ({total_pixels} pixels)')

    # Build request body
    request_body: Dict[str, Any] = {
        'image': image_base64,
        'prompt': params.prompt,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    # Add optional parameters
    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    # Invoke Bedrock model
    result = await invoke_bedrock_model(
        model_id=STABLE_UPSCALE_CONSERVATIVE_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client
    )

    # Extract images from response
    images = result.get('images', [])
    if not images:
        logger.error('No images returned from conservative upscale')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=STABLE_UPSCALE_CONSERVATIVE_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed
        )

    # Save images
    filename_prefix = filename or 'upscale_conservative'
    saved_paths = save_images(
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename_prefix,
        output_format=params.output_format
    )

    logger.info(f'Conservative upscale completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message='Successfully upscaled image with detail preservation',
        paths=saved_paths,
        model_id=STABLE_UPSCALE_CONSERVATIVE_MODEL_ID,
        prompt=params.prompt,
        seed=params.seed,
        metadata={
            'input_dimensions': f'{width}x{height}'
        }
    )


async def upscale_fast(
    params: FastUpscaleParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Fast 4x upscaling without creative enhancement.

    Fast upscaling provides quick 4x resolution increase without AI enhancement.
    Best for images that need simple resolution increase without style changes.

    Workflow:
    1. Validate input image (minimum 1024 pixels, max 1MP)
    2. Build simple request (no prompt needed)
    3. Invoke fast upscale model
    4. Save upscaled image

    Args:
        params: FastUpscaleParams with image.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting fast upscale')

    # Handle image input (file path or base64)
    if os.path.exists(params.image):
        logger.debug(f'Encoding image from file: {params.image}')
        image_base64 = encode_image_file(params.image)
    else:
        logger.debug('Using provided base64 image')
        image_base64 = params.image

    # Validate image dimensions
    image_bytes = decode_base64_image(image_base64)
    width, height = validate_image_dimensions(
        image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION,
        max_pixels=MAX_FAST_UPSCALE_INPUT_PIXELS
    )

    total_pixels = width * height
    logger.info(f'Input image dimensions: {width}x{height} ({total_pixels} pixels)')

    # Validate minimum pixels for fast upscale
    if total_pixels < MIN_FAST_UPSCALE_INPUT_PIXELS:
        raise ValueError(
            f'Image has {total_pixels} pixels, below minimum {MIN_FAST_UPSCALE_INPUT_PIXELS} '
            'pixels for fast upscale'
        )

    # Build request body (simple, no prompt needed)
    request_body: Dict[str, Any] = {
        'image': image_base64,
        'output_format': params.output_format.value,
    }

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    # Invoke Bedrock model
    result = await invoke_bedrock_model(
        model_id=STABLE_UPSCALE_FAST_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client
    )

    # Extract images from response
    images = result.get('images', [])
    if not images:
        logger.error('No images returned from fast upscale')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=STABLE_UPSCALE_FAST_MODEL_ID
        )

    # Save images
    filename_prefix = filename or 'upscale_fast'
    saved_paths = save_images(
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename_prefix,
        output_format=params.output_format
    )

    logger.info(f'Fast upscale completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message='Successfully upscaled image 4x',
        paths=saved_paths,
        model_id=STABLE_UPSCALE_FAST_MODEL_ID,
        metadata={
            'input_dimensions': f'{width}x{height}',
            'upscale_factor': '4x'
        }
    )
