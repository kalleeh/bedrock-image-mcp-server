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
"""Amazon Nova Canvas service implementation.

This module provides functions for generating images using Amazon Nova Canvas
through the AWS Bedrock service. It uses the common bedrock utilities for
API invocation and image saving.
"""

import random
from awslabs.bedrock_image_mcp_server.consts import (
    DEFAULT_CFG_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_NUMBER_OF_IMAGES,
    DEFAULT_QUALITY,
    DEFAULT_WIDTH,
    NOVA_CANVAS_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.nova_models import (
    ColorGuidedGenerationParams,
    ColorGuidedRequest,
    ImageGenerationConfig,
    ImageGenerationResponse,
    Quality,
    TextImageRequest,
    TextToImageParams,
)
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    invoke_bedrock_model,
    save_images,
)
from loguru import logger
from typing import TYPE_CHECKING, List, Optional


if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
else:
    BedrockRuntimeClient = object


async def generate_image_with_text(
    prompt: str,
    bedrock_runtime_client: BedrockRuntimeClient,
    negative_prompt: Optional[str] = None,
    filename: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    quality: str = DEFAULT_QUALITY,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    seed: Optional[int] = None,
    number_of_images: int = DEFAULT_NUMBER_OF_IMAGES,
    workspace_dir: Optional[str] = None,
) -> ImageGenerationResponse:
    """Generate an image using Amazon Nova Canvas with text prompt.

    This function uses Amazon Nova Canvas to generate images based on a text prompt.
    The generated image will be saved to a file and the path will be returned.

    Args:
        prompt: The text description of the image to generate (1-1024 characters).
        bedrock_runtime_client: BedrockRuntimeClient object.
        negative_prompt: Text to define what not to include in the image (1-1024 characters).
        filename: The name of the file to save the image to (without extension).
            If not provided, a random name will be generated.
        width: The width of the generated image (320-4096, divisible by 16).
        height: The height of the generated image (320-4096, divisible by 16).
        quality: The quality of the generated image ("standard" or "premium").
        cfg_scale: How strongly the image adheres to the prompt (1.1-10.0).
        seed: Seed for generation (0-858,993,459). Random if not provided.
        number_of_images: The number of images to generate (1-5).
        workspace_dir: Directory where the images should be saved. If None, uses current directory.

    Returns:
        ImageGenerationResponse: An object containing the paths to the generated images
        and status information.
    """
    logger.debug(
        f"Generating text-to-image with prompt: '{prompt[:30]}...'",
        extra={
            'model': 'nova-canvas',
            'dimensions': f'{width}x{height}',
            'quality': quality,
            'cfg_scale': cfg_scale,
            'num_images': number_of_images,
            'prompt_length': len(prompt),
            'has_negative_prompt': negative_prompt is not None
        }
    )

    try:
        # Validate input parameters using Pydantic
        try:
            logger.debug('Validating parameters and creating request model')

            # Create image generation config
            config = ImageGenerationConfig(
                width=width,
                height=height,
                quality=Quality.STANDARD if quality == DEFAULT_QUALITY else Quality.PREMIUM,
                cfgScale=cfg_scale,
                seed=seed if seed is not None else random.randint(0, 858993459),
                numberOfImages=number_of_images,
            )

            # Create text-to-image params
            if negative_prompt is not None:
                text_params = TextToImageParams(text=prompt, negativeText=negative_prompt)
            else:
                text_params = TextToImageParams(text=prompt)

            # Create the full request
            request_model = TextImageRequest(
                textToImageParams=text_params, imageGenerationConfig=config
            )

            # Convert model to dictionary
            request_model_dict = request_model.to_api_dict()
            logger.info('Request validation successful')

        except Exception as e:
            logger.error(f'Parameter validation failed: {str(e)}')
            return ImageGenerationResponse(
                status='error',
                message=f'Validation error: {str(e)}',
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
            )

        try:
            # Invoke the Nova Canvas API using common function
            # boto3 will automatically retry with exponential backoff if configured
            logger.debug('Sending request to Nova Canvas API')
            model_response = await invoke_bedrock_model(
                model_id=NOVA_CANVAS_MODEL_ID,
                request_body=request_model_dict,
                bedrock_client=bedrock_runtime_client
            )

            # Extract the image data
            base64_images = model_response['images']
            logger.info(
                f'Received {len(base64_images)} images from Nova Canvas API',
                extra={'images_count': len(base64_images), 'model': 'nova-canvas'}
            )

            # Save the generated images using common function
            saved_paths = save_images(
                base64_images=base64_images,
                workspace_dir=workspace_dir,
                filename_prefix='nova_canvas',
                output_format=OutputFormat.PNG
            )

            logger.info(
                f'Successfully generated {len(saved_paths)} image(s)',
                extra={
                    'images_count': len(saved_paths),
                    'model': 'nova-canvas',
                    'output_dir': workspace_dir or 'current_directory'
                }
            )
            return ImageGenerationResponse(
                status='success',
                message=f'Generated {len(saved_paths)} image(s)',
                paths=saved_paths,
                prompt=prompt,
                negative_prompt=negative_prompt,
            )
        except Exception as e:
            logger.error(
                f'Image generation failed: {str(e)}',
                extra={
                    'model': 'nova-canvas',
                    'error_type': type(e).__name__,
                    'prompt_length': len(prompt)
                }
            )
            return ImageGenerationResponse(
                status='error',
                message=str(e),
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
            )

    except Exception as e:
        logger.error(f'Unexpected error in generate_image_with_text: {str(e)}')
        return ImageGenerationResponse(
            status='error',
            message=str(e),
            paths=[],
            prompt=prompt,
            negative_prompt=negative_prompt,
        )


async def generate_image_with_colors(
    prompt: str,
    colors: List[str],
    bedrock_runtime_client: BedrockRuntimeClient,
    negative_prompt: Optional[str] = None,
    filename: Optional[str] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    quality: str = DEFAULT_QUALITY,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    seed: Optional[int] = None,
    number_of_images: int = DEFAULT_NUMBER_OF_IMAGES,
    workspace_dir: Optional[str] = None,
) -> ImageGenerationResponse:
    """Generate an image using Amazon Nova Canvas with color guidance.

    This function uses Amazon Nova Canvas to generate images based on a text prompt and color palette.
    The generated image will be saved to a file and the path will be returned.

    Args:
        prompt: The text description of the image to generate (1-1024 characters).
        colors: List of up to 10 hexadecimal color values (e.g., "#FF9800").
        bedrock_runtime_client: BedrockRuntimeClient object.
        negative_prompt: Text to define what not to include in the image (1-1024 characters).
        filename: The name of the file to save the image to (without extension).
            If not provided, a random name will be generated.
        width: The width of the generated image (320-4096, divisible by 16).
        height: The height of the generated image (320-4096, divisible by 16).
        quality: The quality of the generated image ("standard" or "premium").
        cfg_scale: How strongly the image adheres to the prompt (1.1-10.0).
        seed: Seed for generation (0-858,993,459). Random if not provided.
        number_of_images: The number of images to generate (1-5).
        workspace_dir: Directory where the images should be saved. If None, uses current directory.

    Returns:
        ImageGenerationResponse: An object containing the paths to the generated images
        and status information.
    """
    logger.debug(
        f"Generating color-guided image with prompt: '{prompt[:30]}...' and {len(colors)} colors"
    )

    try:
        # Validate input parameters using Pydantic
        try:
            logger.debug('Validating parameters and creating color-guided request model')

            # Create image generation config
            config = ImageGenerationConfig(
                width=width,
                height=height,
                quality=Quality.STANDARD if quality == DEFAULT_QUALITY else Quality.PREMIUM,
                cfgScale=cfg_scale,
                seed=seed if seed is not None else random.randint(0, 858993459),
                numberOfImages=number_of_images,
            )

            # Create color-guided params
            if negative_prompt is not None:
                color_params = ColorGuidedGenerationParams(
                    colors=colors,
                    text=prompt,
                    negativeText=negative_prompt,
                )
            else:
                color_params = ColorGuidedGenerationParams(
                    colors=colors,
                    text=prompt,
                )

            # Create the full request
            request_model = ColorGuidedRequest(
                colorGuidedGenerationParams=color_params, imageGenerationConfig=config
            )

            # Convert model to dictionary
            request_model_dict = request_model.to_api_dict()
            logger.info('Color-guided request validation successful')

        except Exception as e:
            logger.error(f'Color-guided parameter validation failed: {str(e)}')
            return ImageGenerationResponse(
                status='error',
                message=f'Validation error: {str(e)}',
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
                colors=colors,
            )

        try:
            # Invoke the Nova Canvas API using common function
            logger.debug('Sending color-guided request to Nova Canvas API')
            model_response = await invoke_bedrock_model(
                model_id=NOVA_CANVAS_MODEL_ID,
                request_body=request_model_dict,
                bedrock_client=bedrock_runtime_client
            )

            # Extract the image data
            base64_images = model_response['images']
            logger.info(f'Received {len(base64_images)} images from Nova Canvas API')

            # Save the generated images using common function
            saved_paths = save_images(
                base64_images=base64_images,
                workspace_dir=workspace_dir,
                filename_prefix='nova_canvas_color',
                output_format=OutputFormat.PNG
            )

            logger.info(f'Successfully generated {len(saved_paths)} color-guided image(s)')
            return ImageGenerationResponse(
                status='success',
                message=f'Generated {len(saved_paths)} image(s)',
                paths=saved_paths,
                prompt=prompt,
                negative_prompt=negative_prompt,
                colors=colors,
            )
        except Exception as e:
            logger.error(f'Color-guided image generation failed: {str(e)}')
            return ImageGenerationResponse(
                status='error',
                message=str(e),
                paths=[],
                prompt=prompt,
                negative_prompt=negative_prompt,
                colors=colors,
            )

    except Exception as e:
        logger.error(f'Unexpected error in generate_image_with_colors: {str(e)}')
        return ImageGenerationResponse(
            status='error',
            message=str(e),
            paths=[],
            prompt=prompt,
            negative_prompt=negative_prompt,
            colors=colors,
        )
