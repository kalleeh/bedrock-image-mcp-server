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
"""Stability AI edit services for AWS Bedrock.

This module provides image editing services including inpainting, outpainting,
search and replace, search and recolor, object removal, and background removal.
"""

from awslabs.bedrock_image_mcp_server.consts import (
    STABLE_ERASE_OBJECT_MODEL_ID,
    STABLE_INPAINT_MODEL_ID,
    STABLE_OUTPAINT_MODEL_ID,
    STABLE_REMOVE_BACKGROUND_MODEL_ID,
    STABLE_SEARCH_RECOLOR_MODEL_ID,
    STABLE_SEARCH_REPLACE_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import ImageGenerationResponse, OutputFormat
from awslabs.bedrock_image_mcp_server.models.stability_models import (
    BackgroundRemovalParams,
    InpaintParams,
    OutpaintParams,
    RemoveObjectParams,
    SearchRecolorParams,
    SearchReplaceParams,
)
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    finalize_image_response,
    invoke_bedrock_model,
    prepare_image,
    save_images,
)
from loguru import logger
from typing import TYPE_CHECKING, Any, Dict, Optional


if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
else:
    BedrockRuntimeClient = object


async def inpaint(
    params: InpaintParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Fill masked regions with AI-generated content.

    Inpainting fills masked regions of images with AI-generated content that
    blends naturally with the surrounding image. The mask defines which areas
    to fill (white) and which to preserve (black).

    Workflow:
    1. Validate image and mask compatibility
    2. Validate mask format (grayscale)
    3. Build request with grow_mask parameter
    4. Invoke inpainting model
    5. Save result

    Args:
        params: InpaintParams with image, mask, and prompt.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If the image or mask is invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting inpaint')

    image_base64, width, height = await prepare_image(params.image)
    # Bedrock resizes a mask that differs from the image and converts it to greyscale, so
    # the mask only has to be a decodable image.
    mask_base64, _, _ = await prepare_image(params.mask, label='mask', min_dimension=1)

    logger.info(f'Input image dimensions: {width}x{height}')

    request_body: Dict[str, Any] = {
        'image': image_base64,
        'mask': mask_base64,
        'prompt': params.prompt,
        'grow_mask': params.grow_mask,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    result = await invoke_bedrock_model(
        model_id=STABLE_INPAINT_MODEL_ID, request_body=request_body, bedrock_client=bedrock_client
    )

    return await finalize_image_response(
        result=result,
        model_id=STABLE_INPAINT_MODEL_ID,
        operation='inpaint',
        saver=save_images,
        default_prefix='inpaint',
        filename=filename,
        workspace_dir=workspace_dir,
        output_format=params.output_format,
        success_message='Successfully inpainted image',
        prompt=params.prompt,
        seed=params.seed,
        metadata={'grow_mask': params.grow_mask, 'dimensions': f'{width}x{height}'},
    )


async def outpaint(
    params: OutpaintParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Extend images beyond their original boundaries.

    Outpainting extends images beyond their original boundaries in specified
    directions, generating new content that matches the original image style.

    Workflow:
    1. Validate input image
    2. Build request with direction parameters
    3. Invoke outpainting model
    4. Save result

    Args:
        params: OutpaintParams with image, prompt, and direction settings.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting outpaint')

    image_base64, width, height = await prepare_image(params.image)

    logger.info(f'Input image dimensions: {width}x{height}')
    logger.info(
        f'Outpaint directions: left={params.left}, right={params.right}, '
        f'up={params.up}, down={params.down}'
    )

    request_body: Dict[str, Any] = {
        'image': image_base64,
        'prompt': params.prompt,
        'left': params.left,
        'right': params.right,
        'up': params.up,
        'down': params.down,
        'creativity': params.creativity,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    result = await invoke_bedrock_model(
        model_id=STABLE_OUTPAINT_MODEL_ID, request_body=request_body, bedrock_client=bedrock_client
    )

    return await finalize_image_response(
        result=result,
        model_id=STABLE_OUTPAINT_MODEL_ID,
        operation='outpaint',
        saver=save_images,
        default_prefix='outpaint',
        filename=filename,
        workspace_dir=workspace_dir,
        output_format=params.output_format,
        success_message='Successfully outpainted image',
        prompt=params.prompt,
        seed=params.seed,
        metadata={
            'creativity': params.creativity,
            'input_dimensions': f'{width}x{height}',
            'expansion': f'left={params.left}, right={params.right}, up={params.up}, down={params.down}',
        },
    )


async def search_and_replace(
    params: SearchReplaceParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Find and replace objects using text prompts.

    Search and replace finds objects in images using text prompts and replaces
    them with AI-generated content, automatically handling masking.

    Workflow:
    1. Validate image
    2. Build request with search_prompt and prompt
    3. Invoke search-replace model
    4. Save result

    Args:
        params: SearchReplaceParams with image, search_prompt, and prompt.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting search and replace')

    image_base64, width, height = await prepare_image(params.image)

    logger.info(f'Input image dimensions: {width}x{height}')
    logger.info(f'Search prompt: "{params.search_prompt}"')
    logger.info(f'Replace prompt: "{params.prompt}"')

    request_body: Dict[str, Any] = {
        'image': image_base64,
        'search_prompt': params.search_prompt,
        'prompt': params.prompt,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    result = await invoke_bedrock_model(
        model_id=STABLE_SEARCH_REPLACE_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client,
    )

    return await finalize_image_response(
        result=result,
        model_id=STABLE_SEARCH_REPLACE_MODEL_ID,
        operation='search and replace',
        saver=save_images,
        default_prefix='search_replace',
        filename=filename,
        workspace_dir=workspace_dir,
        output_format=params.output_format,
        success_message='Successfully replaced objects in image',
        prompt=params.prompt,
        seed=params.seed,
        metadata={'search_prompt': params.search_prompt, 'dimensions': f'{width}x{height}'},
    )


async def search_and_recolor(
    params: SearchRecolorParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Recolor objects using text prompts.

    Search and recolor finds objects in images using text prompts and changes
    their colors while preserving structure and detail.

    Note: This service uses 'select_prompt' (not 'search_prompt') to identify objects.

    Workflow:
    1. Validate image
    2. Build request with select_prompt and prompt
    3. Invoke search-recolor model
    4. Save result

    Args:
        params: SearchRecolorParams with image, select_prompt, and prompt.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting search and recolor')

    image_base64, width, height = await prepare_image(params.image)

    logger.info(f'Input image dimensions: {width}x{height}')
    logger.info(f'Select prompt: "{params.select_prompt}"')
    logger.info(f'Recolor prompt: "{params.prompt}"')

    # Note: this model uses select_prompt, not search_prompt
    request_body: Dict[str, Any] = {
        'image': image_base64,
        'select_prompt': params.select_prompt,
        'prompt': params.prompt,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    result = await invoke_bedrock_model(
        model_id=STABLE_SEARCH_RECOLOR_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client,
    )

    return await finalize_image_response(
        result=result,
        model_id=STABLE_SEARCH_RECOLOR_MODEL_ID,
        operation='search and recolor',
        saver=save_images,
        default_prefix='search_recolor',
        filename=filename,
        workspace_dir=workspace_dir,
        output_format=params.output_format,
        success_message='Successfully recolored objects in image',
        prompt=params.prompt,
        seed=params.seed,
        metadata={'select_prompt': params.select_prompt, 'dimensions': f'{width}x{height}'},
    )


async def remove_object(
    params: RemoveObjectParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Remove unwanted objects from images.

    Remove object intelligently removes unwanted objects from images and fills
    the area with content that blends naturally with surroundings.

    Workflow:
    1. Validate image and mask compatibility
    2. Validate mask format (grayscale)
    3. Build request with grow_mask parameter
    4. Invoke object removal model
    5. Save result

    Args:
        params: RemoveObjectParams with image and mask.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If the image or mask is invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting remove object')

    image_base64, width, height = await prepare_image(params.image)
    # Bedrock resizes a mask that differs from the image and converts it to greyscale, so
    # the mask only has to be a decodable image.
    mask_base64, _, _ = await prepare_image(params.mask, label='mask', min_dimension=1)

    logger.info(f'Input image dimensions: {width}x{height}')

    request_body: Dict[str, Any] = {
        'image': image_base64,
        'mask': mask_base64,
        'grow_mask': params.grow_mask,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    result = await invoke_bedrock_model(
        model_id=STABLE_ERASE_OBJECT_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client,
    )

    return await finalize_image_response(
        result=result,
        model_id=STABLE_ERASE_OBJECT_MODEL_ID,
        operation='remove object',
        saver=save_images,
        default_prefix='remove_object',
        filename=filename,
        workspace_dir=workspace_dir,
        output_format=params.output_format,
        success_message='Successfully removed object from image',
        seed=params.seed,
        metadata={'grow_mask': params.grow_mask, 'dimensions': f'{width}x{height}'},
    )


async def remove_background(
    params: BackgroundRemovalParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Remove background from image.

    Background removal automatically removes backgrounds from images, isolating
    the main subject with clean edges. Always outputs PNG with transparency.

    Workflow:
    1. Validate image
    2. Build simple request (no prompt needed)
    3. Invoke background removal model
    4. Save as PNG with transparency

    Args:
        params: BackgroundRemovalParams with image.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting background removal')

    image_base64, width, height = await prepare_image(params.image)

    logger.info(f'Input image dimensions: {width}x{height}')

    request_body: Dict[str, Any] = {
        'image': image_base64,
        'output_format': 'png',  # Always PNG for transparency
    }

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    result = await invoke_bedrock_model(
        model_id=STABLE_REMOVE_BACKGROUND_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client,
    )

    return await finalize_image_response(
        result=result,
        model_id=STABLE_REMOVE_BACKGROUND_MODEL_ID,
        operation='background removal',
        saver=save_images,
        default_prefix='remove_background',
        filename=filename,
        workspace_dir=workspace_dir,
        output_format=OutputFormat.PNG,
        success_message='Successfully removed background from image',
        metadata={'input_dimensions': f'{width}x{height}', 'output_format': 'png'},
    )
