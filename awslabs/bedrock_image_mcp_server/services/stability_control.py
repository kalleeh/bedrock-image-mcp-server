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
"""Stability AI control services for AWS Bedrock.

This module provides control-based image generation services including
sketch-to-image, structure control, style guide, and style transfer.
"""

import os
from awslabs.bedrock_image_mcp_server.consts import (
    MIN_IMAGE_DIMENSION,
    STABLE_CONTROL_SKETCH_MODEL_ID,
    STABLE_CONTROL_STRUCTURE_MODEL_ID,
    STABLE_STYLE_GUIDE_MODEL_ID,
    STABLE_STYLE_TRANSFER_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import ImageGenerationResponse
from awslabs.bedrock_image_mcp_server.models.stability_models import (
    SketchToImageParams,
    StructureControlParams,
    StyleGuideParams,
    StyleTransferParams,
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


async def sketch_to_image(
    params: SketchToImageParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Convert sketch to detailed image.

    Sketch-to-image converts sketches or line art into detailed images while
    preserving the structure and composition of the original sketch.

    Workflow:
    1. Validate control image (sketch/line art)
    2. Build request with control_strength
    3. Invoke sketch control model
    4. Save result

    Args:
        params: SketchToImageParams with control_image and prompt.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If control image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting sketch-to-image')

    # Handle control image input (file path or base64)
    if os.path.exists(params.control_image):
        logger.debug(f'Encoding control image from file: {params.control_image}')
        control_image_base64 = encode_image_file(params.control_image)
    else:
        logger.debug('Using provided base64 control image')
        control_image_base64 = params.control_image

    # Validate control image dimensions
    control_image_bytes = decode_base64_image(control_image_base64)
    width, height = validate_image_dimensions(
        control_image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION
    )

    logger.info(f'Control image dimensions: {width}x{height}')
    logger.info(f'Control strength: {params.control_strength}')

    # Build request body
    # Note: AWS API expects 'image' not 'control_image' per AWS documentation
    request_body: Dict[str, Any] = {
        'image': control_image_base64,
        'prompt': params.prompt,
        'control_strength': params.control_strength,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    # Add optional parameters
    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    # Invoke Bedrock model
    result = await invoke_bedrock_model(
        model_id=STABLE_CONTROL_SKETCH_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client
    )

    # Extract images from response
    images = result.get('images', [])
    if not images:
        logger.error('No images returned from sketch-to-image')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=STABLE_CONTROL_SKETCH_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed
        )

    # Save images
    filename_prefix = filename or 'sketch_to_image'
    saved_paths = save_images(
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename_prefix,
        output_format=params.output_format
    )

    logger.info(f'Sketch-to-image completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message='Successfully converted sketch to image',
        paths=saved_paths,
        model_id=STABLE_CONTROL_SKETCH_MODEL_ID,
        prompt=params.prompt,
        seed=params.seed,
        metadata={
            'control_strength': params.control_strength,
            'control_dimensions': f'{width}x{height}'
        }
    )


async def structure_control(
    params: StructureControlParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Generate image following structural guide.

    Structure control generates images that follow structural guides like edge maps
    or depth maps, maintaining specific compositions while adding detail and style.

    Workflow:
    1. Validate structure/edge map
    2. Build request with control parameters
    3. Invoke structure control model
    4. Save result

    Args:
        params: StructureControlParams with control_image and prompt.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If control image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting structure control')

    # Handle control image input (file path or base64)
    if os.path.exists(params.control_image):
        logger.debug(f'Encoding control image from file: {params.control_image}')
        control_image_base64 = encode_image_file(params.control_image)
    else:
        logger.debug('Using provided base64 control image')
        control_image_base64 = params.control_image

    # Validate control image dimensions
    control_image_bytes = decode_base64_image(control_image_base64)
    width, height = validate_image_dimensions(
        control_image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION
    )

    logger.info(f'Control image dimensions: {width}x{height}')
    logger.info(f'Control strength: {params.control_strength}')

    # Build request body
    # Note: AWS API expects 'image' not 'control_image' per AWS documentation
    request_body: Dict[str, Any] = {
        'image': control_image_base64,
        'prompt': params.prompt,
        'control_strength': params.control_strength,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    # Add optional parameters
    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    # Invoke Bedrock model
    result = await invoke_bedrock_model(
        model_id=STABLE_CONTROL_STRUCTURE_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client
    )

    # Extract images from response
    images = result.get('images', [])
    if not images:
        logger.error('No images returned from structure control')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=STABLE_CONTROL_STRUCTURE_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed
        )

    # Save images
    filename_prefix = filename or 'structure_control'
    saved_paths = save_images(
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename_prefix,
        output_format=params.output_format
    )

    logger.info(f'Structure control completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message='Successfully generated image with structure control',
        paths=saved_paths,
        model_id=STABLE_CONTROL_STRUCTURE_MODEL_ID,
        prompt=params.prompt,
        seed=params.seed,
        metadata={
            'control_strength': params.control_strength,
            'control_dimensions': f'{width}x{height}'
        }
    )


async def style_guide(
    params: StyleGuideParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Generate image matching reference style.

    Style guide generates images matching a reference style while following
    the content description in the prompt. The fidelity parameter controls
    how closely the output matches the reference style.

    Workflow:
    1. Validate reference image
    2. Build request with fidelity parameter
    3. Invoke style guide model
    4. Save result

    Args:
        params: StyleGuideParams with reference_image, prompt, and fidelity.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If reference image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting style guide')

    # Handle reference image input (file path or base64)
    if os.path.exists(params.reference_image):
        logger.debug(f'Encoding reference image from file: {params.reference_image}')
        reference_image_base64 = encode_image_file(params.reference_image)
    else:
        logger.debug('Using provided base64 reference image')
        reference_image_base64 = params.reference_image

    # Validate reference image dimensions
    reference_image_bytes = decode_base64_image(reference_image_base64)
    width, height = validate_image_dimensions(
        reference_image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION
    )

    logger.info(f'Reference image dimensions: {width}x{height}')
    logger.info(f'Style fidelity: {params.fidelity}')

    # Build request body
    # Note: AWS API expects 'image' not 'reference_image' per AWS documentation
    request_body: Dict[str, Any] = {
        'image': reference_image_base64,
        'prompt': params.prompt,
        'fidelity': params.fidelity,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    # Add optional parameters
    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    # Invoke Bedrock model
    result = await invoke_bedrock_model(
        model_id=STABLE_STYLE_GUIDE_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client
    )

    # Extract images from response
    images = result.get('images', [])
    if not images:
        logger.error('No images returned from style guide')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=STABLE_STYLE_GUIDE_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed
        )

    # Save images
    filename_prefix = filename or 'style_guide'
    saved_paths = save_images(
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename_prefix,
        output_format=params.output_format
    )

    logger.info(f'Style guide completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message='Successfully generated image with style guide',
        paths=saved_paths,
        model_id=STABLE_STYLE_GUIDE_MODEL_ID,
        prompt=params.prompt,
        seed=params.seed,
        metadata={
            'fidelity': params.fidelity,
            'reference_dimensions': f'{width}x{height}'
        }
    )


async def style_transfer(
    params: StyleTransferParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None
) -> ImageGenerationResponse:
    """Apply style from one image to content of another.

    Style transfer applies the style from one image to the content of another,
    with fine-grained control over composition, style strength, and changes.

    Workflow:
    1. Validate init_image and style_image
    2. Build request with all control parameters
    3. Invoke style transfer model
    4. Save result

    Args:
        params: StyleTransferParams with init_image, style_image, and control parameters.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        ValueError: If image dimensions are invalid.
        BedrockAPIError: If API call fails.
    """
    logger.info('Starting style transfer')

    # Handle init image input (file path or base64)
    if os.path.exists(params.init_image):
        logger.debug(f'Encoding init image from file: {params.init_image}')
        init_image_base64 = encode_image_file(params.init_image)
    else:
        logger.debug('Using provided base64 init image')
        init_image_base64 = params.init_image

    # Handle style image input (file path or base64)
    if os.path.exists(params.style_image):
        logger.debug(f'Encoding style image from file: {params.style_image}')
        style_image_base64 = encode_image_file(params.style_image)
    else:
        logger.debug('Using provided base64 style image')
        style_image_base64 = params.style_image

    # Validate init image dimensions
    init_image_bytes = decode_base64_image(init_image_base64)
    init_width, init_height = validate_image_dimensions(
        init_image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION
    )

    # Validate style image dimensions
    style_image_bytes = decode_base64_image(style_image_base64)
    style_width, style_height = validate_image_dimensions(
        style_image_bytes,
        min_width=MIN_IMAGE_DIMENSION,
        min_height=MIN_IMAGE_DIMENSION
    )

    logger.info(f'Init image dimensions: {init_width}x{init_height}')
    logger.info(f'Style image dimensions: {style_width}x{style_height}')
    logger.info(f'Composition fidelity: {params.composition_fidelity}, '
                f'Style strength: {params.style_strength}, '
                f'Change strength: {params.change_strength}')

    # Build request body
    request_body: Dict[str, Any] = {
        'init_image': init_image_base64,
        'style_image': style_image_base64,
        'prompt': params.prompt,
        'composition_fidelity': params.composition_fidelity,
        'style_strength': params.style_strength,
        'change_strength': params.change_strength,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    # Add optional parameters
    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    logger.debug(f'Request body keys: {list(request_body.keys())}')

    # Invoke Bedrock model
    result = await invoke_bedrock_model(
        model_id=STABLE_STYLE_TRANSFER_MODEL_ID,
        request_body=request_body,
        bedrock_client=bedrock_client
    )

    # Extract images from response
    images = result.get('images', [])
    if not images:
        logger.error('No images returned from style transfer')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=STABLE_STYLE_TRANSFER_MODEL_ID,
            prompt=params.prompt,
            seed=params.seed
        )

    # Save images
    filename_prefix = filename or 'style_transfer'
    saved_paths = save_images(
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename_prefix,
        output_format=params.output_format
    )

    logger.info(f'Style transfer completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message='Successfully transferred style',
        paths=saved_paths,
        model_id=STABLE_STYLE_TRANSFER_MODEL_ID,
        prompt=params.prompt,
        seed=params.seed,
        metadata={
            'composition_fidelity': params.composition_fidelity,
            'style_strength': params.style_strength,
            'change_strength': params.change_strength,
            'init_dimensions': f'{init_width}x{init_height}',
            'style_dimensions': f'{style_width}x{style_height}'
        }
    )
