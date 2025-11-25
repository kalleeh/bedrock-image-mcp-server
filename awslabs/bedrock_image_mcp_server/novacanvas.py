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
"""Amazon Nova Canvas API interaction module.

DEPRECATED: This module is maintained for backward compatibility.
New code should import from awslabs.bedrock_image_mcp_server.services.nova_canvas instead.

This module provides functions for generating images using Amazon Nova Canvas
through the AWS Bedrock service. It handles the API requests, response processing,
and image saving functionality.
"""

# Import all functions from services.nova_canvas for backward compatibility
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    invoke_bedrock_model,
    save_images,
)
from awslabs.bedrock_image_mcp_server.services.nova_canvas import (
    generate_image_with_colors,
    generate_image_with_text,
)
from typing import Dict, List, Optional


def save_generated_images(
    base64_images: List[str],
    filename: Optional[str] = None,
    number_of_images: int = 1,
    prefix: str = 'nova_canvas',
    workspace_dir: Optional[str] = None,
) -> Dict[str, List]:
    """Save base64-encoded images to files.

    DEPRECATED: This function is maintained for backward compatibility.
    Use save_images from services.bedrock_common instead.

    Args:
        base64_images: List of base64-encoded image data.
        filename: Base filename to use (without extension). If None, a random name is generated.
        number_of_images: Number of images being saved (ignored, uses len(base64_images)).
        prefix: Prefix to use for randomly generated filenames.
        workspace_dir: Directory where the images should be saved. If None, uses current directory.

    Returns:
        Dictionary with lists of paths to the saved image files.
    """
    # Call the new save_images function
    paths = save_images(
        base64_images=base64_images,
        workspace_dir=workspace_dir,
        filename_prefix=prefix,
        output_format=OutputFormat.PNG
    )
    return {'paths': paths}


# For backward compatibility, create an alias for invoke_nova_canvas
async def invoke_nova_canvas(request_model_dict, bedrock_runtime_client):
    """Invoke the Nova Canvas API with the given request.

    DEPRECATED: Use invoke_bedrock_model from services.bedrock_common instead.

    Args:
        request_model_dict: Dictionary representation of the request model.
        bedrock_runtime_client: BedrockRuntimeClient object.

    Returns:
        Dictionary containing the API response.

    Raises:
        Exception: If the API call fails.
    """
    from awslabs.bedrock_image_mcp_server.consts import NOVA_CANVAS_MODEL_ID
    return await invoke_bedrock_model(
        model_id=NOVA_CANVAS_MODEL_ID,
        request_body=request_model_dict,
        bedrock_client=bedrock_runtime_client
    )


__all__ = [
    'generate_image_with_text',
    'generate_image_with_colors',
    'save_generated_images',
    'invoke_nova_canvas',
]
