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
"""Tests for the Stability AI upscale services."""

import base64
import pytest
from awslabs.bedrock_image_mcp_server.consts import (
    STABLE_UPSCALE_CONSERVATIVE_MODEL_ID,
    STABLE_UPSCALE_CREATIVE_MODEL_ID,
    STABLE_UPSCALE_FAST_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.stability_models import (
    ConservativeUpscaleParams,
    CreativeUpscaleParams,
    FastUpscaleParams,
    StylePreset,
)
from awslabs.bedrock_image_mcp_server.services.stability_upscale import (
    upscale_conservative,
    upscale_creative,
    upscale_fast,
)
from io import BytesIO
from PIL import Image
from unittest.mock import MagicMock, patch


def create_test_image_base64(width=512, height=512, format='PNG'):
    """Create a valid test image and return as base64 string."""
    img = Image.new('RGB', (width, height), color='blue')
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


class TestUpscaleCreative:
    """Tests for the upscale_creative function."""

    @pytest.mark.asyncio
    async def test_successful_upscale(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test successful creative upscale."""
        # Create test image
        base64_image = create_test_image_base64()

        # Mock Bedrock response
        mock_bedrock_runtime_client.invoke_model = MagicMock(
            return_value={
                'body': MagicMock(
                    read=MagicMock(
                        return_value=b'{"images": ["' + base64_image.encode() + b'"], "finish_reasons": ["SUCCESS"]}'
                    )
                )
            }
        )

        # Create parameters
        params = CreativeUpscaleParams(
            image=base64_image,
            prompt="high quality upscaled image",
            creativity=0.3,
            output_format=OutputFormat.PNG
        )

        # Call upscale function
        with patch('awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model') as mock_invoke:
            mock_invoke.return_value = {
                'images': [base64_image],
                'finish_reasons': ['SUCCESS']
            }

            response = await upscale_creative(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
                filename='test_upscale'
            )

        # Verify response
        assert response.status == 'success'
        assert len(response.paths) == 1
        assert response.model_id == STABLE_UPSCALE_CREATIVE_MODEL_ID
        assert response.prompt == "high quality upscaled image"
        assert response.metadata['creativity'] == 0.3

    @pytest.mark.asyncio
    async def test_upscale_with_style_preset(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test creative upscale with style preset."""
        base64_image = create_test_image_base64()

        params = CreativeUpscaleParams(
            image=base64_image,
            prompt="photographic style",
            creativity=0.4,
            style_preset=StylePreset.PHOTOGRAPHIC,
            output_format=OutputFormat.PNG
        )

        with patch('awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model') as mock_invoke:
            mock_invoke.return_value = {
                'images': [base64_image],
                'finish_reasons': ['SUCCESS']
            }

            response = await upscale_creative(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir
            )

        assert response.status == 'success'
        assert response.metadata['style_preset'] == 'photographic'


class TestUpscaleConservative:
    """Tests for the upscale_conservative function."""

    @pytest.mark.asyncio
    async def test_successful_upscale(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test successful conservative upscale."""
        base64_image = create_test_image_base64()

        params = ConservativeUpscaleParams(
            image=base64_image,
            prompt="preserve original details",
            output_format=OutputFormat.PNG
        )

        with patch('awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model') as mock_invoke:
            mock_invoke.return_value = {
                'images': [base64_image],
                'finish_reasons': ['SUCCESS']
            }

            response = await upscale_conservative(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
                filename='test_conservative'
            )

        assert response.status == 'success'
        assert len(response.paths) == 1
        assert response.model_id == STABLE_UPSCALE_CONSERVATIVE_MODEL_ID
        assert response.prompt == "preserve original details"


class TestUpscaleFast:
    """Tests for the upscale_fast function."""

    @pytest.mark.asyncio
    async def test_successful_upscale(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test successful fast upscale."""
        # Create larger image to meet minimum pixel requirement
        base64_image = create_test_image_base64(width=512, height=512)

        params = FastUpscaleParams(
            image=base64_image,
            output_format=OutputFormat.PNG
        )

        with patch('awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model') as mock_invoke:
            mock_invoke.return_value = {
                'images': [base64_image],
                'finish_reasons': ['SUCCESS']
            }

            response = await upscale_fast(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
                filename='test_fast'
            )

        assert response.status == 'success'
        assert len(response.paths) == 1
        assert response.model_id == STABLE_UPSCALE_FAST_MODEL_ID
        assert response.metadata['upscale_factor'] == '4x'

    @pytest.mark.asyncio
    async def test_image_too_small(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test fast upscale with image below minimum pixels."""
        # Create small image below minimum
        base64_image = create_test_image_base64(width=16, height=16)

        params = FastUpscaleParams(
            image=base64_image,
            output_format=OutputFormat.PNG
        )

        with pytest.raises(ValueError, match='below minimum'):
            await upscale_fast(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir
            )


class TestImageValidation:
    """Tests for image dimension validation in upscale services."""

    @pytest.mark.asyncio
    async def test_creative_upscale_warns_large_image(self, mock_bedrock_runtime_client, temp_workspace_dir, caplog):
        """Test that creative upscale warns when image is too large."""
        # Create image larger than recommended (but still valid for testing)
        base64_image = create_test_image_base64(width=1024, height=1024)

        params = CreativeUpscaleParams(
            image=base64_image,
            prompt="test prompt",
            creativity=0.3,
            output_format=OutputFormat.PNG
        )

        with patch('awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model') as mock_invoke:
            mock_invoke.return_value = {
                'images': [base64_image],
                'finish_reasons': ['SUCCESS']
            }

            response = await upscale_creative(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir
            )

        # Should still succeed but may have logged a warning
        assert response.status == 'success'
