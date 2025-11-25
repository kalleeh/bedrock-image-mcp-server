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

import base64
import json
import os
import random
from awslabs.bedrock_image_mcp_server.consts import DEFAULT_OUTPUT_DIR
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from botocore.exceptions import ClientError
from loguru import logger
from typing import TYPE_CHECKING, Any, Dict, List, Optional


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
            message=f"Content filtered: {reason}",
            error_code='ContentFiltered',
            retryable=False
        )


async def invoke_bedrock_model(
    model_id: str,
    request_body: Dict[str, Any],
    bedrock_client: BedrockRuntimeClient
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
    logger.debug(
        f'Invoking Bedrock model: {model_id}',
        extra={
            'model_id': model_id,
            'request_keys': list(request_body.keys())
        }
    )

    try:
        # Convert the request payload to JSON
        request = json.dumps(request_body)

        # Invoke the model (boto3 handles retries automatically if configured)
        logger.info(f'Sending request to Bedrock model: {model_id}')
        response = bedrock_client.invoke_model(modelId=model_id, body=request)

        # Decode the response body
        result = json.loads(response['body'].read().decode('utf-8'))
        logger.info(
            f'Bedrock API call successful for model: {model_id}',
            extra={'model_id': model_id, 'images_count': len(result.get('images', []))}
        )

        # Check for content filtering
        if 'finish_reasons' in result:
            finish_reasons = result['finish_reasons']
            for reason in finish_reasons:
                # null means success, any other value means filtered or error
                if reason is not None:
                    logger.warning(
                        f'Content filtered: {reason}',
                        extra={'model_id': model_id, 'filter_reason': reason}
                    )
                    raise ContentFilterError(reason)

        return result

    except ContentFilterError:
        # Re-raise content filter errors as-is (not retryable)
        raise

    except ClientError as e:
        # Parse AWS ClientError for detailed error classification
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']

        logger.error(
            f'Bedrock API error: {error_code}',
            extra={
                'model_id': model_id,
                'error_code': error_code,
                'error_message': error_message
            }
        )

        # Classify errors following AWS best practices
        if error_code == 'ValidationException':
            raise BedrockAPIError(
                message=f"Invalid parameters: {error_message}",
                error_code=error_code,
                retryable=False
            )
        elif error_code == 'AccessDeniedException':
            raise BedrockAPIError(
                message=f"Access denied. Check IAM permissions for model {model_id}. "
                        f"Ensure you have 'bedrock:InvokeModel' permission and model access is enabled.",
                error_code=error_code,
                retryable=False
            )
        elif error_code == 'ThrottlingException':
            raise BedrockAPIError(
                message="Rate limit exceeded. AWS SDK will automatically retry with exponential backoff. "
                        "If this persists, consider requesting a quota increase.",
                error_code=error_code,
                retryable=True
            )
        elif error_code == 'ModelNotReadyException':
            raise BedrockAPIError(
                message=f"Model {model_id} is not ready. Please try again in a few moments.",
                error_code=error_code,
                retryable=True
            )
        elif error_code == 'ServiceUnavailableException':
            raise BedrockAPIError(
                message="Bedrock service is temporarily unavailable. AWS SDK will automatically retry.",
                error_code=error_code,
                retryable=True
            )
        elif error_code == 'InternalServerException':
            raise BedrockAPIError(
                message="Internal server error. AWS SDK will automatically retry.",
                error_code=error_code,
                retryable=True
            )
        else:
            # Unknown error - log for investigation
            logger.exception(
                f'Unexpected AWS error: {error_code}',
                extra={'model_id': model_id, 'error_code': error_code}
            )
            raise BedrockAPIError(
                message=f"API call failed: {error_message}",
                error_code=error_code,
                retryable=False
            )

    except Exception as e:
        # Catch-all for unexpected errors
        logger.exception(
            f'Unexpected error invoking Bedrock model: {model_id}',
            extra={'model_id': model_id}
        )
        raise BedrockAPIError(
            message=f"Unexpected error: {str(e)}",
            error_code='UnexpectedError',
            retryable=False
        )


def save_images(
    base64_images: List[str],
    workspace_dir: Optional[str],
    filename_prefix: str,
    output_format: OutputFormat = OutputFormat.PNG
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
    logger.debug(f'Saving {len(base64_images)} images with prefix: {filename_prefix}')

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
        raise IOError(f"Failed to create output directory {output_dir}: {str(e)}")

    # Determine file extension
    extension_map = {
        OutputFormat.JPEG: 'jpg',
        OutputFormat.PNG: 'png',
        OutputFormat.WEBP: 'webp'
    }
    extension = extension_map.get(output_format, 'png')

    # Save the generated images
    saved_paths: List[str] = []
    for i, base64_image_data in enumerate(base64_images):
        try:
            # Generate filename
            random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
            if len(base64_images) > 1:
                image_filename = f'{filename_prefix}_{random_id}_{i + 1}.{extension}'
            else:
                image_filename = f'{filename_prefix}_{random_id}.{extension}'

            # Decode the base64 image data
            image_data = base64.b64decode(base64_image_data)

            # Save the image
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, 'wb') as file:
                file.write(image_data)

            # Convert to absolute path
            abs_image_path = os.path.abspath(image_path)
            saved_paths.append(abs_image_path)
            logger.debug(f'Saved image to: {abs_image_path}')

        except Exception as e:
            logger.error(f'Failed to save image {i + 1}: {str(e)}')
            raise IOError(f"Failed to save image {i + 1}: {str(e)}")

    logger.info(f'Successfully saved {len(saved_paths)} image(s)')
    return saved_paths
