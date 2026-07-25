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
"""Stable Image Ultra and Stable Image Core service implementation.

Both are text-to-image only and share one request schema, so they differ solely by
model ID and the resolution they return.
"""

from awslabs.bedrock_image_mcp_server.consts import (
    STABLE_IMAGE_CORE_MODEL_ID,
    STABLE_IMAGE_ULTRA_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import ImageGenerationResponse, OutputFormat
from awslabs.bedrock_image_mcp_server.models.stable_image_models import StableImageParams
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    finalize_image_response,
    invoke_bedrock_model,
    save_images,
)
from loguru import logger
from typing import TYPE_CHECKING, Any, Dict, Optional


if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
else:
    BedrockRuntimeClient = object


def build_stable_image_request(params: StableImageParams) -> Dict[str, Any]:
    """Build the API request body for Stable Image Ultra or Core.

    Args:
        params: Validated generation parameters.

    Returns:
        Dictionary containing the formatted API request body.
    """
    request_body: Dict[str, Any] = {
        'prompt': params.prompt,
        'aspect_ratio': params.aspect_ratio.value,
        'seed': params.seed,
        'output_format': params.output_format.value,
    }

    if params.negative_prompt:
        request_body['negative_prompt'] = params.negative_prompt

    return request_body


async def _generate(
    params: StableImageParams,
    bedrock_client: BedrockRuntimeClient,
    model_id: str,
    operation: str,
    default_prefix: str,
    workspace_dir: Optional[str],
    filename: Optional[str],
) -> ImageGenerationResponse:
    """Generate an image with one of the Stable Image models.

    Args:
        params: Validated generation parameters.
        bedrock_client: BedrockRuntimeClient object.
        model_id: The Bedrock model ID to invoke.
        operation: Lower-case operation name used in log and response messages.
        default_prefix: Filename prefix to use when the caller supplied none.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        BedrockAPIError: If the API call fails.
        ContentFilterError: On content filtering.
    """
    logger.bind(
        model_id=model_id,
        aspect_ratio=params.aspect_ratio.value,
        seed=params.seed,
        prompt_length=len(params.prompt),
    ).info(f'Generating image with {operation}')

    request_body = build_stable_image_request(params)

    result = await invoke_bedrock_model(
        model_id=model_id,
        request_body=request_body,
        bedrock_client=bedrock_client,
    )

    return await finalize_image_response(
        result=result,
        model_id=model_id,
        operation=operation,
        saver=save_images,
        default_prefix=default_prefix,
        filename=filename,
        workspace_dir=workspace_dir,
        output_format=OutputFormat(params.output_format.value),
        success_message=f'Successfully generated image with {operation}',
        prompt=params.prompt,
        seed=params.seed,
        metadata={'aspect_ratio': params.aspect_ratio.value},
    )


async def generate_image_ultra(
    params: StableImageParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Generate an image with Stable Image Ultra.

    Args:
        params: Validated generation parameters.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        BedrockAPIError: If the API call fails.
        ContentFilterError: On content filtering.
    """
    return await _generate(
        params=params,
        bedrock_client=bedrock_client,
        model_id=STABLE_IMAGE_ULTRA_MODEL_ID,
        operation='stable image ultra',
        default_prefix='stable_image_ultra',
        workspace_dir=workspace_dir,
        filename=filename,
    )


async def generate_image_core(
    params: StableImageParams,
    bedrock_client: BedrockRuntimeClient,
    workspace_dir: Optional[str] = None,
    filename: Optional[str] = None,
) -> ImageGenerationResponse:
    """Generate an image with Stable Image Core.

    Args:
        params: Validated generation parameters.
        bedrock_client: BedrockRuntimeClient object.
        workspace_dir: Directory where images should be saved.
        filename: Optional custom filename prefix.

    Returns:
        ImageGenerationResponse with status, message, and file paths.

    Raises:
        BedrockAPIError: If the API call fails.
        ContentFilterError: On content filtering.
    """
    return await _generate(
        params=params,
        bedrock_client=bedrock_client,
        model_id=STABLE_IMAGE_CORE_MODEL_ID,
        operation='stable image core',
        default_prefix='stable_image_core',
        workspace_dir=workspace_dir,
        filename=filename,
    )
