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
"""Tests for the stability_edit module of the bedrock-image-mcp-server."""

import base64
import io
import pytest
from awslabs.bedrock_image_mcp_server.consts import (
    STABLE_ERASE_OBJECT_MODEL_ID,
    STABLE_INPAINT_MODEL_ID,
    STABLE_OUTPAINT_MODEL_ID,
    STABLE_REMOVE_BACKGROUND_MODEL_ID,
    STABLE_SEARCH_RECOLOR_MODEL_ID,
    STABLE_SEARCH_REPLACE_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.stability_models import (
    BackgroundRemovalParams,
    InpaintParams,
    OutpaintParams,
    RemoveObjectParams,
    SearchRecolorParams,
    SearchReplaceParams,
)
from awslabs.bedrock_image_mcp_server.services.stability_edit import (
    inpaint,
    outpaint,
    remove_background,
    remove_object,
    search_and_recolor,
    search_and_replace,
)
from PIL import Image
from pydantic import ValidationError
from unittest.mock import patch


def _encode_png(img: Image.Image) -> str:
    """Encode a PIL image as base64 PNG data."""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@pytest.fixture
def sample_image_base64() -> str:
    """Create a sample base64-encoded image for testing."""
    # Create a simple 256x256 RGB image
    img = Image.new('RGB', (256, 256), color='red')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


@pytest.fixture
def sample_mask_base64() -> str:
    """Create a sample base64-encoded grayscale mask for testing."""
    # Create a simple 256x256 grayscale mask
    img = Image.new('L', (256, 256), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


@pytest.fixture
def sample_mismatched_mask_base64() -> str:
    """Create a mask with different dimensions for testing validation."""
    # Create a 128x128 mask (different from 256x256 image)
    img = Image.new('L', (128, 128), color=128)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


class TestInpaint:
    """Tests for the inpaint function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_successful_inpaint(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        sample_mask_base64,
        temp_workspace_dir,
    ):
        """Test successful inpainting with mask."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {'images': ['base64_inpainted_image']}
        mock_save_images.return_value = ['/path/to/inpainted.png']

        # Create params
        params = InpaintParams(
            image=sample_image_base64,
            mask=sample_mask_base64,
            prompt='fill with flowers',
            grow_mask=5,
            seed=12345,
        )

        # Call the function
        result = await inpaint(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_inpaint',
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.message == 'Successfully inpainted image'
        assert result.paths == ['/path/to/inpainted.png']
        assert result.model_id == STABLE_INPAINT_MODEL_ID
        assert result.prompt == 'fill with flowers'
        assert result.seed == 12345

        # Check that invoke_bedrock_model was called with correct parameters
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_INPAINT_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['prompt'] == 'fill with flowers'
        assert request_body['grow_mask'] == 5
        assert request_body['seed'] == 12345
        assert 'image' in request_body
        assert 'mask' in request_body

        # Check that save_images was called
        mock_save_images.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mask_mode', ['L', '1', 'P', 'RGB'])
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_inpaint_sends_any_decodable_mask(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mask_mode,
        mock_bedrock_runtime_client,
        sample_image_base64,
    ):
        """Test a smaller mask in any PIL mode reaches Bedrock, which resizes and converts it.

        Bedrock documents that a mask of a different size than the image is resized
        automatically, and it places no restriction on the mask's colour mode.
        """
        mock_invoke_bedrock.return_value = {'images': ['base64_inpainted_image']}
        mock_save_images.return_value = ['/path/to/inpainted.png']
        mask = _encode_png(Image.new(mask_mode, (128, 128)))

        params = InpaintParams(image=sample_image_base64, mask=mask, prompt='fill with flowers')
        result = await inpaint(params=params, bedrock_client=mock_bedrock_runtime_client)

        assert result.status == 'success'
        assert mock_invoke_bedrock.call_args[1]['request_body']['mask'] == mask

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    async def test_inpaint_rejects_undecodable_mask(
        self, mock_invoke_bedrock, mock_bedrock_runtime_client, sample_image_base64
    ):
        """Test a mask that is not image data fails before any Bedrock call."""
        params = InpaintParams(
            image=sample_image_base64,
            mask=base64.b64encode(b'not an image').decode('utf-8'),
            prompt='fill with flowers',
        )

        with pytest.raises((ValueError, IOError)):
            await inpaint(params=params, bedrock_client=mock_bedrock_runtime_client)

        mock_invoke_bedrock.assert_not_called()

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_inpaint_with_negative_prompt(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        sample_mask_base64,
    ):
        """Test inpaint with negative prompt."""
        mock_invoke_bedrock.return_value = {'images': ['base64_image']}
        mock_save_images.return_value = ['/path/to/image.png']

        params = InpaintParams(
            image=sample_image_base64,
            mask=sample_mask_base64,
            prompt='beautiful garden',
            negative_prompt='weeds, dead plants',
        )

        result = await inpaint(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        assert result.status == 'success'

        # Check negative prompt was included
        call_args = mock_invoke_bedrock.call_args[1]
        request_body = call_args['request_body']
        assert request_body['negative_prompt'] == 'weeds, dead plants'


class TestOutpaint:
    """Tests for the outpaint function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_successful_outpaint(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        temp_workspace_dir,
    ):
        """Test successful outpainting with direction parameters."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {'images': ['base64_outpainted_image']}
        mock_save_images.return_value = ['/path/to/outpainted.png']

        # Create params with directional expansion
        params = OutpaintParams(
            image=sample_image_base64,
            prompt='extend the landscape',
            left=100,
            right=100,
            up=50,
            down=50,
            creativity=0.7,
            seed=12345,
        )

        # Call the function
        result = await outpaint(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_outpaint',
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.message == 'Successfully outpainted image'
        assert result.paths == ['/path/to/outpainted.png']
        assert result.model_id == STABLE_OUTPAINT_MODEL_ID
        assert result.prompt == 'extend the landscape'
        assert result.seed == 12345

        # Check that invoke_bedrock_model was called with correct parameters
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_OUTPAINT_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['prompt'] == 'extend the landscape'
        assert request_body['left'] == 100
        assert request_body['right'] == 100
        assert request_body['up'] == 50
        assert request_body['down'] == 50
        assert request_body['creativity'] == 0.7
        assert request_body['seed'] == 12345

    def test_outpaint_requires_a_direction(self, sample_image_base64):
        """Test the all-zero default is rejected locally, since Bedrock requires a direction."""
        with pytest.raises(ValidationError, match='At least one of left, right, up or down'):
            OutpaintParams(image=sample_image_base64, prompt='extend the scene')

    def test_outpaint_accepts_a_single_direction(self, sample_image_base64):
        """Test one non-zero direction is enough."""
        params = OutpaintParams(image=sample_image_base64, prompt='extend the scene', down=1)
        assert (params.left, params.right, params.up, params.down) == (0, 0, 0, 1)


class TestSearchAndReplace:
    """Tests for the search_and_replace function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_successful_search_replace(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        temp_workspace_dir,
    ):
        """Test successful search and replace."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {'images': ['base64_replaced_image']}
        mock_save_images.return_value = ['/path/to/replaced.png']

        # Create params
        params = SearchReplaceParams(
            image=sample_image_base64,
            search_prompt='old car',
            prompt='modern electric car',
            seed=12345,
        )

        # Call the function
        result = await search_and_replace(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_replace',
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.message == 'Successfully replaced objects in image'
        assert result.paths == ['/path/to/replaced.png']
        assert result.model_id == STABLE_SEARCH_REPLACE_MODEL_ID
        assert result.prompt == 'modern electric car'
        assert result.seed == 12345

        # Check that invoke_bedrock_model was called with correct parameters
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_SEARCH_REPLACE_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['search_prompt'] == 'old car'
        assert request_body['prompt'] == 'modern electric car'
        assert request_body['seed'] == 12345


class TestSearchAndRecolor:
    """Tests for the search_and_recolor function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_successful_search_recolor(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        temp_workspace_dir,
    ):
        """Test successful search and recolor with select_prompt."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {'images': ['base64_recolored_image']}
        mock_save_images.return_value = ['/path/to/recolored.png']

        # Create params
        params = SearchRecolorParams(
            image=sample_image_base64,
            select_prompt='the car',
            prompt='bright red color',
            seed=12345,
        )

        # Call the function
        result = await search_and_recolor(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_recolor',
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.message == 'Successfully recolored objects in image'
        assert result.paths == ['/path/to/recolored.png']
        assert result.model_id == STABLE_SEARCH_RECOLOR_MODEL_ID
        assert result.prompt == 'bright red color'
        assert result.seed == 12345

        # Check that invoke_bedrock_model was called with correct parameters
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_SEARCH_RECOLOR_MODEL_ID
        request_body = call_args['request_body']
        # IMPORTANT: Verify select_prompt is used (not search_prompt)
        assert request_body['select_prompt'] == 'the car'
        assert request_body['prompt'] == 'bright red color'
        assert request_body['seed'] == 12345
        # Ensure search_prompt is NOT in the request
        assert 'search_prompt' not in request_body


class TestRemoveObject:
    """Tests for the remove_object function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_successful_remove_object(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        sample_mask_base64,
        temp_workspace_dir,
    ):
        """Test successful object removal."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {'images': ['base64_removed_image']}
        mock_save_images.return_value = ['/path/to/removed.png']

        # Create params
        params = RemoveObjectParams(
            image=sample_image_base64,
            mask=sample_mask_base64,
            grow_mask=10,
            seed=12345,
        )

        # Call the function
        result = await remove_object(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_remove',
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.message == 'Successfully removed object from image'
        assert result.paths == ['/path/to/removed.png']
        assert result.model_id == STABLE_ERASE_OBJECT_MODEL_ID
        assert result.seed == 12345

        # Check that invoke_bedrock_model was called with correct parameters
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_ERASE_OBJECT_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['grow_mask'] == 10
        assert request_body['seed'] == 12345
        assert 'image' in request_body
        assert 'mask' in request_body

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_remove_object_sends_mismatched_mask(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        sample_mismatched_mask_base64,
    ):
        """Test a mask smaller than the image reaches Bedrock, which resizes it."""
        mock_invoke_bedrock.return_value = {'images': ['base64_image']}
        mock_save_images.return_value = ['/path/to/image.png']

        params = RemoveObjectParams(image=sample_image_base64, mask=sample_mismatched_mask_base64)
        result = await remove_object(params=params, bedrock_client=mock_bedrock_runtime_client)

        assert result.status == 'success'
        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['mask'] == sample_mismatched_mask_base64


class TestRemoveBackground:
    """Tests for the remove_background function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_successful_remove_background(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
        temp_workspace_dir,
    ):
        """Test successful background removal."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {'images': ['base64_nobg_image']}
        mock_save_images.return_value = ['/path/to/nobg.png']

        # Create params
        params = BackgroundRemovalParams(
            image=sample_image_base64,
        )

        # Call the function
        result = await remove_background(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_nobg',
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.message == 'Successfully removed background from image'
        assert result.paths == ['/path/to/nobg.png']
        assert result.model_id == STABLE_REMOVE_BACKGROUND_MODEL_ID

        # Check that invoke_bedrock_model was called with correct parameters
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_REMOVE_BACKGROUND_MODEL_ID
        request_body = call_args['request_body']
        assert 'image' in request_body
        # Background removal should always output PNG
        assert request_body['output_format'] == 'png'
        # No prompt needed for background removal
        assert 'prompt' not in request_body

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.stability_edit.save_images')
    async def test_remove_background_png_output(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_image_base64,
    ):
        """Test that background removal always outputs PNG format."""
        mock_invoke_bedrock.return_value = {'images': ['base64_image']}
        mock_save_images.return_value = ['/path/to/image.png']

        params = BackgroundRemovalParams(
            image=sample_image_base64,
        )

        result = await remove_background(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        assert result.status == 'success'

        # Verify save_images was called with PNG format
        mock_save_images.assert_called_once()
        call_args = mock_save_images.call_args[1]
        assert call_args['output_format'] == OutputFormat.PNG
