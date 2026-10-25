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


UPSCALE_MODULE = 'awslabs.bedrock_image_mcp_server.services.stability_upscale'


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
                        return_value=b'{"images": ["'
                        + base64_image.encode()
                        + b'"], "finish_reasons": ["SUCCESS"]}'
                    )
                )
            }
        )

        # Create parameters
        params = CreativeUpscaleParams(
            image=base64_image,
            prompt='high quality upscaled image',
            creativity=0.3,
            output_format=OutputFormat.PNG,
        )

        # Call upscale function
        with patch(
            'awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model'
        ) as mock_invoke:
            mock_invoke.return_value = {'images': [base64_image], 'finish_reasons': ['SUCCESS']}

            response = await upscale_creative(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
                filename='test_upscale',
            )

        # Verify response
        assert response.status == 'success'
        assert len(response.paths) == 1
        assert response.model_id == STABLE_UPSCALE_CREATIVE_MODEL_ID
        assert response.prompt == 'high quality upscaled image'
        assert response.metadata['creativity'] == 0.3

        mock_invoke.assert_called_once()
        call = mock_invoke.call_args[1]
        assert call['model_id'] == STABLE_UPSCALE_CREATIVE_MODEL_ID
        assert call['request_body'] == {
            'image': base64_image,
            'prompt': 'high quality upscaled image',
            'creativity': 0.3,
            'seed': 0,
            'output_format': 'png',
        }

    @pytest.mark.asyncio
    async def test_upscale_with_style_preset(
        self, mock_bedrock_runtime_client, temp_workspace_dir
    ):
        """Test creative upscale with style preset."""
        base64_image = create_test_image_base64()

        params = CreativeUpscaleParams(
            image=base64_image,
            prompt='photographic style',
            creativity=0.4,
            style_preset=StylePreset.PHOTOGRAPHIC,
            output_format=OutputFormat.PNG,
        )

        with patch(
            'awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model'
        ) as mock_invoke:
            mock_invoke.return_value = {'images': [base64_image], 'finish_reasons': ['SUCCESS']}

            response = await upscale_creative(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
            )

        assert response.status == 'success'
        assert response.metadata['style_preset'] == 'photographic'
        request_body = mock_invoke.call_args[1]['request_body']
        assert request_body['style_preset'] == 'photographic'
        assert request_body['creativity'] == 0.4


class TestUpscaleConservative:
    """Tests for the upscale_conservative function."""

    @pytest.mark.asyncio
    async def test_successful_upscale(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test successful conservative upscale."""
        base64_image = create_test_image_base64()

        params = ConservativeUpscaleParams(
            image=base64_image,
            prompt='preserve original details',
            negative_prompt='blur',
            seed=7,
            output_format=OutputFormat.WEBP,
        )

        with patch(
            'awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model'
        ) as mock_invoke:
            mock_invoke.return_value = {'images': [base64_image], 'finish_reasons': ['SUCCESS']}

            response = await upscale_conservative(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
                filename='test_conservative',
            )

        assert response.status == 'success'
        assert len(response.paths) == 1
        assert response.model_id == STABLE_UPSCALE_CONSERVATIVE_MODEL_ID
        assert response.prompt == 'preserve original details'

        call = mock_invoke.call_args[1]
        assert call['model_id'] == STABLE_UPSCALE_CONSERVATIVE_MODEL_ID
        assert call['request_body'] == {
            'image': base64_image,
            'prompt': 'preserve original details',
            'negative_prompt': 'blur',
            'seed': 7,
            'output_format': 'webp',
        }


class TestUpscaleFast:
    """Tests for the upscale_fast function."""

    @pytest.mark.asyncio
    async def test_successful_upscale(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test successful fast upscale."""
        # Create larger image to meet minimum pixel requirement
        base64_image = create_test_image_base64(width=512, height=512)

        params = FastUpscaleParams(image=base64_image, output_format=OutputFormat.PNG)

        with patch(
            'awslabs.bedrock_image_mcp_server.services.stability_upscale.invoke_bedrock_model'
        ) as mock_invoke:
            mock_invoke.return_value = {'images': [base64_image], 'finish_reasons': ['SUCCESS']}

            response = await upscale_fast(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
                filename='test_fast',
            )

        assert response.status == 'success'
        assert len(response.paths) == 1
        assert response.model_id == STABLE_UPSCALE_FAST_MODEL_ID
        assert response.metadata['upscale_factor'] == '4x'

        call = mock_invoke.call_args[1]
        assert call['model_id'] == STABLE_UPSCALE_FAST_MODEL_ID
        # Fast upscale takes no prompt, seed or creativity.
        assert call['request_body'] == {'image': base64_image, 'output_format': 'png'}

    @pytest.mark.asyncio
    async def test_image_too_small(self, mock_bedrock_runtime_client, temp_workspace_dir):
        """Test fast upscale with image below minimum pixels."""
        # Create small image below minimum
        base64_image = create_test_image_base64(width=16, height=16)

        params = FastUpscaleParams(image=base64_image, output_format=OutputFormat.PNG)

        with pytest.raises(ValueError, match='below minimum'):
            await upscale_fast(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
                workspace_dir=temp_workspace_dir,
            )


class TestImageValidation:
    """Tests for the input pixel caps in upscale services."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'upscale, params_cls, size',
        [
            (upscale_creative, CreativeUpscaleParams, (1024, 1025)),
            (upscale_fast, FastUpscaleParams, (1024, 1025)),
            (upscale_conservative, ConservativeUpscaleParams, (3072, 3073)),
        ],
        ids=['creative-1MP', 'fast-1MP', 'conservative-9.4MP'],
    )
    async def test_input_over_pixel_cap_is_rejected(
        self, upscale, params_cls, size, mock_bedrock_runtime_client
    ):
        """Test one row over each documented pixel cap is rejected before calling Bedrock."""
        image = create_test_image_base64(*size)
        params = (
            params_cls(image=image)
            if params_cls is FastUpscaleParams
            else params_cls(image=image, prompt='p')
        )

        with patch(f'{UPSCALE_MODULE}.invoke_bedrock_model') as mock_invoke:
            with pytest.raises(ValueError, match='exceeding maximum'):
                await upscale(params=params, bedrock_client=mock_bedrock_runtime_client)

        mock_invoke.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'upscale, params_cls, size',
        [
            (upscale_creative, CreativeUpscaleParams, (1024, 1024)),
            (upscale_conservative, ConservativeUpscaleParams, (3072, 3072)),
        ],
        ids=['creative-1MP', 'conservative-9.4MP'],
    )
    async def test_input_at_pixel_cap_is_accepted(
        self, upscale, params_cls, size, mock_bedrock_runtime_client
    ):
        """Test an image exactly at the cap still goes through."""
        params = params_cls(image=create_test_image_base64(*size), prompt='p')

        with (
            patch(f'{UPSCALE_MODULE}.invoke_bedrock_model') as mock_invoke,
            patch(f'{UPSCALE_MODULE}.save_images', return_value=['/tmp/out.png']),
        ):
            mock_invoke.return_value = {'images': ['upscaled']}
            response = await upscale(params=params, bedrock_client=mock_bedrock_runtime_client)

        assert response.status == 'success'


class TestDocumentedUpscaleLimits:
    """Tests for the input limits AWS documents for the upscale models."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('size', [(32, 32), (32, 1536), (1536, 32)])
    async def test_fast_upscale_accepts_documented_range(self, size, mock_bedrock_runtime_client):
        """Test fast upscale accepts 32-1,536px per side, which AWS documents as valid."""
        params = FastUpscaleParams(image=create_test_image_base64(*size))

        with (
            patch(f'{UPSCALE_MODULE}.invoke_bedrock_model') as mock_invoke,
            patch(f'{UPSCALE_MODULE}.save_images', return_value=['/tmp/out.png']),
        ):
            mock_invoke.return_value = {'images': ['upscaled']}
            response = await upscale_fast(
                params=params, bedrock_client=mock_bedrock_runtime_client
            )

        assert response.status == 'success'
        mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'size, match',
        [
            ((31, 64), 'below minimum 32px'),
            ((64, 31), 'below minimum 32px'),
            ((1537, 64), '1536px per-side maximum'),
            ((64, 1537), '1536px per-side maximum'),
        ],
    )
    async def test_fast_upscale_rejects_outside_range(
        self, size, match, mock_bedrock_runtime_client
    ):
        """Test fast upscale rejects sides outside 32-1,536px before calling Bedrock."""
        params = FastUpscaleParams(image=create_test_image_base64(*size))

        with patch(f'{UPSCALE_MODULE}.invoke_bedrock_model') as mock_invoke:
            with pytest.raises(ValueError, match=match):
                await upscale_fast(params=params, bedrock_client=mock_bedrock_runtime_client)

        mock_invoke.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('size', [(64, 160), (160, 64)])
    async def test_conservative_upscale_accepts_2_5_to_1(self, size, mock_bedrock_runtime_client):
        """Test the 1:2.5 and 2.5:1 extremes are accepted."""
        params = ConservativeUpscaleParams(image=create_test_image_base64(*size), prompt='p')

        with (
            patch(f'{UPSCALE_MODULE}.invoke_bedrock_model') as mock_invoke,
            patch(f'{UPSCALE_MODULE}.save_images', return_value=['/tmp/out.png']),
        ):
            mock_invoke.return_value = {'images': ['upscaled']}
            response = await upscale_conservative(
                params=params, bedrock_client=mock_bedrock_runtime_client
            )

        assert response.status == 'success'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('size', [(64, 161), (161, 64)])
    async def test_conservative_upscale_rejects_beyond_2_5_to_1(
        self, size, mock_bedrock_runtime_client
    ):
        """Test an aspect ratio past 2.5:1 is rejected before calling Bedrock."""
        params = ConservativeUpscaleParams(image=create_test_image_base64(*size), prompt='p')

        with patch(f'{UPSCALE_MODULE}.invoke_bedrock_model') as mock_invoke:
            with pytest.raises(ValueError, match='1:2.5 to 2.5:1'):
                await upscale_conservative(
                    params=params, bedrock_client=mock_bedrock_runtime_client
                )

        mock_invoke.assert_not_called()
