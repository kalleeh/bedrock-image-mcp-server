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
"""Common utilities for AWS Bedrock image generation services.

This module provides shared functionality for all Bedrock image generation models,
including API invocation, image saving, and error handling.
"""

import asyncio
import base64
import json
import os
import random
from awslabs.bedrock_image_mcp_server.consts import (
    DEFAULT_OUTPUT_DIR,
    MAX_FILENAME_LENGTH,
    MIN_IMAGE_DIMENSION,
)
from awslabs.bedrock_image_mcp_server.models.common import ImageGenerationResponse, OutputFormat
from awslabs.bedrock_image_mcp_server.utils.image_utils import (
    decode_base64_image,
    encode_image_file,
    validate_image_dimensions,
)
from botocore.exceptions import ClientError
from loguru import logger
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple


if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
else:
    BedrockRuntimeClient = object


class BedrockAPIError(Exception):
    """Base exception for Bedrock API errors.

    Attributes:
        error_code: AWS error code (e.g., 'ValidationException', 'ThrottlingException').
        message: Human-readable error message.
        retryable: Whether this error is retryable.
    """

    def __init__(self, message: str, error_code: str = 'Unknown', retryable: bool = False):
        """Initialize BedrockAPIError.

        Args:
            message: Human-readable error message.
            error_code: AWS error code.
            retryable: Whether this error should be retried.
        """
        self.error_code = error_code
        self.message = message
        self.retryable = retryable
        super().__init__(message)


class ContentFilterError(BedrockAPIError):
    """Raised when content is filtered by Bedrock.

    Attributes:
        reason: The reason for content filtering.
    """

    def __init__(self, reason: str):
        """Initialize ContentFilterError with reason.

        Args:
            reason: The reason for content filtering.
        """
        self.reason = reason
        super().__init__(
            message=f'Content filtered: {reason}', error_code='ContentFiltered', retryable=False
        )


def sanitize_filename(name: str, fallback: str) -> str:
    """Reduce a caller-supplied filename to a safe basename.

    Args:
        name: Untrusted filename or prefix supplied by the caller.
        fallback: Value to use when nothing safe remains.

    Returns:
        A basename with no path separators, parent-directory references, drive or stream
        qualifiers, control characters, or length that would exceed filesystem limits.
    """
    candidate = os.path.basename(name.strip().replace('\\', '/').rstrip('/'))
    # ':' would form a Windows drive-relative path or NTFS alternate data stream.
    candidate = candidate.split(':')[0]
    candidate = ''.join(c for c in candidate if c.isprintable() and c not in '/\0')
    candidate = candidate.strip().lstrip('.')
    # Leave room for the random suffix and extension appended by save_images.
    candidate = candidate[:MAX_FILENAME_LENGTH]
    if not candidate:
        return fallback
    return candidate


def resolve_output_path(output_dir: str, filename: str) -> str:
    """Join a filename into the output directory, refusing to escape it.

    Args:
        output_dir: Directory the file must be written inside.
        filename: Already-sanitized basename.

    Returns:
        Absolute path to the file inside output_dir.

    Raises:
        ValueError: If the resolved path would fall outside output_dir.
    """
    root = os.path.realpath(output_dir)
    target = os.path.realpath(os.path.join(root, filename))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f'Refusing to write outside the output directory: {filename}')
    return target


def resolve_image_input(value: str, label: str = 'image') -> str:
    """Return base64 image data for a value that is either a file path or base64 data.

    Args:
        value: Either a path to an image file on disk or base64-encoded image data.
        label: Human-readable name of the input, used in log messages.

    Returns:
        Base64-encoded image data.

    Raises:
        IOError: If an existing file cannot be read.
        ValueError: If an existing file cannot be encoded.
    """
    if os.path.exists(value):
        logger.debug(f'Encoding {label} from file: {value}')
        return encode_image_file(value)
    logger.debug(f'Using provided base64 {label}')
    return value


def measure_image(
    image_base64: str,
    *,
    min_dimension: int = MIN_IMAGE_DIMENSION,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    """Decode base64 image data and validate its dimensions.

    Args:
        image_base64: Base64-encoded image data.
        min_dimension: Minimum allowed width and height in pixels.
        max_pixels: Maximum allowed total pixels, or None for no limit.

    Returns:
        Tuple of (width, height) in pixels.

    Raises:
        ValueError: If the data is not decodable or the dimensions are out of range.
    """
    image_bytes = decode_base64_image(image_base64)
    return validate_image_dimensions(
        image_bytes,
        min_width=min_dimension,
        min_height=min_dimension,
        max_pixels=max_pixels,
    )


async def prepare_image(
    value: str,
    *,
    label: str = 'image',
    min_dimension: int = MIN_IMAGE_DIMENSION,
    max_pixels: Optional[int] = None,
) -> Tuple[str, int, int]:
    """Resolve an image input and validate its dimensions.

    Reading and decoding a large image takes long enough to stall the event loop, so the
    work runs on a worker thread.

    Args:
        value: Either a path to an image file on disk or base64-encoded image data.
        label: Human-readable name of the input, used in log messages.
        min_dimension: Minimum allowed width and height in pixels.
        max_pixels: Maximum allowed total pixels, or None for no limit.

    Returns:
        Tuple of (base64 image data, width, height).

    Raises:
        ValueError: If the data is not decodable or the dimensions are out of range.
    """

    def _prepare() -> Tuple[str, int, int]:
        image_base64 = resolve_image_input(value, label)
        width, height = measure_image(
            image_base64, min_dimension=min_dimension, max_pixels=max_pixels
        )
        return image_base64, width, height

    return await asyncio.to_thread(_prepare)


async def finalize_image_response(
    *,
    result: Dict[str, Any],
    model_id: str,
    operation: str,
    saver: Callable[..., List[str]],
    default_prefix: str,
    filename: Optional[str],
    workspace_dir: Optional[str],
    output_format: OutputFormat,
    success_message: str,
    prompt: Optional[str] = None,
    seed: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ImageGenerationResponse:
    """Save the images from a Bedrock result and build the service response.

    Args:
        result: Decoded Bedrock response, expected to carry an 'images' list.
        model_id: The Bedrock model ID that produced the result.
        operation: Lower-case operation name used in log messages (e.g. 'fast upscale').
        saver: Callable used to persist the images, with the save_images signature.
        default_prefix: Filename prefix to use when the caller supplied none.
        filename: Caller-supplied filename prefix, if any.
        workspace_dir: Directory where images should be saved.
        output_format: Format the images should be saved in.
        success_message: Message for the successful response.
        prompt: Prompt to echo back in the response, if applicable.
        seed: Seed to echo back in the response, if applicable.
        metadata: Per-operation metadata for the successful response.

    Returns:
        An error ImageGenerationResponse when no images were returned, otherwise a
        success response with the saved file paths.

    Raises:
        IOError: If saving an image fails.
    """
    images = result.get('images', [])
    if not images:
        logger.error(f'No images returned from {operation}')
        return ImageGenerationResponse(
            status='error',
            message='No images generated',
            paths=[],
            model_id=model_id,
            prompt=prompt,
            seed=seed,
        )

    # Decoding and writing several multi-megabyte images blocks for over a second, so it
    # runs on a worker thread to keep the event loop responsive.
    saved_paths = await asyncio.to_thread(
        saver,
        base64_images=images,
        workspace_dir=workspace_dir,
        filename_prefix=filename or default_prefix,
        output_format=output_format,
    )

    logger.info(f'{operation.capitalize()} completed: {len(saved_paths)} image(s) saved')

    return ImageGenerationResponse(
        status='success',
        message=success_message,
        paths=saved_paths,
        model_id=model_id,
        prompt=prompt,
        seed=seed,
        metadata=metadata or {},
    )


async def invoke_bedrock_model(
    model_id: str, request_body: Dict[str, Any], bedrock_client: BedrockRuntimeClient
) -> Dict[str, Any]:
    """Invoke any Bedrock model with comprehensive error handling.

    This function provides a unified interface for invoking Bedrock image generation
    models. It handles API calls, error parsing, content filtering detection, and
    provides detailed error classification following AWS best practices.

    The boto3 client should be configured with retry logic:
        Config(retries={'max_attempts': 3, 'mode': 'adaptive'})

    Args:
        model_id: The Bedrock model ID to invoke.
        request_body: Dictionary containing the request parameters.
        bedrock_client: BedrockRuntimeClient object with retry configuration.

    Returns:
        Dictionary containing the API response with 'images', 'seeds', and 'finish_reasons'.

    Raises:
        BedrockAPIError: On API failures with detailed error classification.
        ContentFilterError: On content filtering.
    """
    # Log request with structured data for debugging
    logger.bind(
        model_id=model_id,
        request_keys=list(request_body.keys()),
    ).debug(f'Invoking Bedrock model: {model_id}')

    try:
        # Convert the request payload to JSON
        request = json.dumps(request_body)

        # Invoke the model (boto3 handles retries automatically if configured).
        # boto3 is synchronous and a generation can take 30-90s, so it runs on a worker
        # thread to keep the event loop free for pings, progress and cancellation.
        logger.info(f'Sending request to Bedrock model: {model_id}')

        def _invoke() -> bytes:
            response = bedrock_client.invoke_model(modelId=model_id, body=request)
            return response['body'].read()

        raw_body = await asyncio.to_thread(_invoke)

        # Decode the response body
        result = json.loads(raw_body.decode('utf-8'))
        logger.bind(
            model_id=model_id,
            images_count=len(result.get('images', [])),
        ).info(f'Bedrock API call successful for model: {model_id}')

        # Check for content filtering
        if 'finish_reasons' in result:
            finish_reasons = result['finish_reasons']
            for reason in finish_reasons:
                # null means success, any other value means filtered or error
                if reason is not None:
                    logger.bind(
                        model_id=model_id,
                        filter_reason=reason,
                    ).warning(f'Content filtered: {reason}')
                    raise ContentFilterError(reason)

        return result

    except ContentFilterError:
        # Re-raise content filter errors as-is (not retryable)
        raise

    except ClientError as e:
        # Parse AWS ClientError for detailed error classification
        error_details = e.response.get('Error', {})
        error_code = error_details.get('Code', 'Unknown')
        error_message = error_details.get('Message', str(e))

        logger.bind(
            model_id=model_id,
            error_code=error_code,
            error_message=error_message,
        ).error(f'Bedrock API error: {error_code}')

        # Classify errors following AWS best practices
        if error_code == 'ValidationException':
            raise BedrockAPIError(
                message=f'Invalid parameters: {error_message}',
                error_code=error_code,
                retryable=False,
            )
        elif error_code == 'AccessDeniedException':
            raise BedrockAPIError(
                message=f'Access denied. Check IAM permissions for model {model_id}. '
                f"Ensure you have 'bedrock:InvokeModel' permission and model access is enabled.",
                error_code=error_code,
                retryable=False,
            )
        elif error_code == 'ThrottlingException':
            raise BedrockAPIError(
                message='Rate limit exceeded. AWS SDK will automatically retry with exponential backoff. '
                'If this persists, consider requesting a quota increase.',
                error_code=error_code,
                retryable=True,
            )
        elif error_code == 'ModelNotReadyException':
            raise BedrockAPIError(
                message=f'Model {model_id} is not ready. Please try again in a few moments.',
                error_code=error_code,
                retryable=True,
            )
        elif error_code == 'ServiceUnavailableException':
            raise BedrockAPIError(
                message='Bedrock service is temporarily unavailable. AWS SDK will automatically retry.',
                error_code=error_code,
                retryable=True,
            )
        elif error_code == 'InternalServerException':
            raise BedrockAPIError(
                message='Internal server error. AWS SDK will automatically retry.',
                error_code=error_code,
                retryable=True,
            )
        else:
            # Unknown error - log for investigation
            logger.bind(
                model_id=model_id,
                error_code=error_code,
            ).exception(f'Unexpected AWS error: {error_code}')
            raise BedrockAPIError(
                message=f'API call failed: {error_message}', error_code=error_code, retryable=False
            )

    except Exception as e:
        # Catch-all for unexpected errors
        logger.bind(
            model_id=model_id,
        ).exception(f'Unexpected error invoking Bedrock model: {model_id}')
        raise BedrockAPIError(
            message=f'Unexpected error: {str(e)}', error_code='UnexpectedError', retryable=False
        )


def save_images(
    base64_images: List[str],
    workspace_dir: Optional[str],
    filename_prefix: str,
    output_format: OutputFormat = OutputFormat.PNG,
) -> List[str]:
    """Save base64-encoded images to workspace.

    This function decodes base64 images and saves them to the workspace output directory.
    It handles directory creation, filename generation, and returns absolute file paths.

    Args:
        base64_images: List of base64-encoded image data.
        workspace_dir: Directory where images should be saved. If None, uses current directory.
        filename_prefix: Prefix for generated filenames (e.g., 'sd35', 'upscale').
        output_format: Output image format (default: PNG).

    Returns:
        List of absolute file paths to the saved images.

    Raises:
        IOError: If directory creation or file writing fails.
    """
    safe_prefix = sanitize_filename(filename_prefix, 'image')
    logger.debug(f'Saving {len(base64_images)} images with prefix: {safe_prefix}')

    # Determine the output directory
    if workspace_dir:
        output_dir = os.path.join(workspace_dir, DEFAULT_OUTPUT_DIR)
    else:
        output_dir = DEFAULT_OUTPUT_DIR

    # Create output directory if it doesn't exist
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.debug(f'Created output directory: {output_dir}')
    except Exception as e:
        raise IOError(f'Failed to create output directory {output_dir}: {str(e)}')

    # Determine file extension
    extension_map = {OutputFormat.JPEG: 'jpg', OutputFormat.PNG: 'png', OutputFormat.WEBP: 'webp'}
    extension = extension_map.get(output_format, 'png')

    # Save the generated images
    saved_paths: List[str] = []
    for i, base64_image_data in enumerate(base64_images):
        try:
            # Generate filename
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            if len(base64_images) > 1:
                image_filename = f'{safe_prefix}_{random_id}_{i + 1}.{extension}'
            else:
                image_filename = f'{safe_prefix}_{random_id}.{extension}'

            # Decode the base64 image data, tolerating the line wrapping encoders emit
            normalized = base64_image_data.strip().replace('\r', '').replace('\n', '')
            image_data = base64.b64decode(normalized, validate=True)

            # Save the image
            image_path = resolve_output_path(output_dir, image_filename)
            with open(image_path, 'wb') as file:
                file.write(image_data)

            saved_paths.append(image_path)
            logger.debug(f'Saved image to: {image_path}')

        except Exception as e:
            logger.error(f'Failed to save image {i + 1}: {str(e)}')
            raise IOError(f'Failed to save image {i + 1}: {str(e)}')

    logger.info(f'Successfully saved {len(saved_paths)} image(s)')
    return saved_paths
