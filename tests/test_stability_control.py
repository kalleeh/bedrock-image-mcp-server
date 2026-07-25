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
"""Tests for the stability_control module of the bedrock-image-mcp-server."""

import base64
import io
import os
import pytest
from awslabs.bedrock_image_mcp_server.consts import (
    STABLE_CONTROL_SKETCH_MODEL_ID,
    STABLE_CONTROL_STRUCTURE_MODEL_ID,
    STABLE_STYLE_GUIDE_MODEL_ID,
    STABLE_STYLE_TRANSFER_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.stability_models import (
    SketchToImageParams,
    StructureControlParams,
    StyleGuideParams,
    StyleTransferParams,
)
from awslabs.bedrock_image_mcp_server.services.stability_control import (
    sketch_to_image,
    structure_control,
    style_guide,
    style_transfer,
)
from PIL import Image
from unittest.mock import patch


CONTROL_MODULE = 'awslabs.bedrock_image_mcp_server.services.stability_control'


def _png_base64(width: int = 256, height: int = 256, color: str = 'blue') -> str:
    """Build a base64-encoded PNG image of the requested size.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: Fill color for the image.

    Returns:
        The base64-encoded PNG bytes as an ASCII string.
    """
    img = Image.new('RGB', (width, height), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@pytest.fixture
def sample_control_image() -> str:
    """Return a valid 256x256 base64 PNG usable as a control/reference image."""
    return _png_base64()


@pytest.fixture
def sample_style_image() -> str:
    """Return a second valid base64 PNG, distinct from the control image."""
    return _png_base64(320, 320, 'green')


@pytest.fixture
def tiny_image() -> str:
    """Return a base64 PNG below the minimum accepted dimension."""
    return _png_base64(32, 32)


@pytest.fixture
def image_file_on_disk(temp_workspace_dir: str) -> str:
    """Write a real PNG file to a temp directory and return its path."""
    path = os.path.join(temp_workspace_dir, 'control_input.png')
    Image.new('RGB', (128, 128), color='white').save(path, format='PNG')
    return path


class TestSketchToImage:
    """Tests for the sketch_to_image function."""

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_successful_sketch_to_image(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
        temp_workspace_dir,
    ):
        """Test sketch-to-image sends control_strength and the sketch as 'image'."""
        mock_invoke_bedrock.return_value = {'images': ['base64_sketch_result']}
        mock_save_images.return_value = ['/path/to/sketch.png']

        params = SketchToImageParams(
            control_image=sample_control_image,
            prompt='a detailed watercolor house',
            control_strength=0.85,
            seed=4242,
            output_format=OutputFormat.WEBP,
        )

        result = await sketch_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_sketch',
        )

        assert result.status == 'success'
        assert result.message == 'Successfully converted sketch to image'
        assert result.paths == ['/path/to/sketch.png']
        assert result.model_id == STABLE_CONTROL_SKETCH_MODEL_ID
        assert result.prompt == 'a detailed watercolor house'
        assert result.seed == 4242
        assert result.metadata['control_strength'] == 0.85
        assert result.metadata['control_dimensions'] == '256x256'

        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_CONTROL_SKETCH_MODEL_ID
        request_body = call_args['request_body']
        # AWS expects the sketch under 'image', never 'control_image'.
        assert request_body['image'] == sample_control_image
        assert 'control_image' not in request_body
        assert request_body['prompt'] == 'a detailed watercolor house'
        assert request_body['control_strength'] == 0.85
        assert request_body['seed'] == 4242
        assert request_body['output_format'] == 'webp'
        # Sketch control has no fidelity/style knobs.
        assert 'fidelity' not in request_body
        assert 'style_strength' not in request_body
        assert 'negative_prompt' not in request_body

        save_kwargs = mock_save_images.call_args[1]
        assert save_kwargs['base64_images'] == ['base64_sketch_result']
        assert save_kwargs['workspace_dir'] == temp_workspace_dir
        assert save_kwargs['filename_prefix'] == 'test_sketch'
        assert save_kwargs['output_format'] == OutputFormat.WEBP

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_sketch_defaults_and_negative_prompt(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
    ):
        """Test default control_strength is forwarded and negative_prompt is included."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/sketch.png']

        params = SketchToImageParams(
            control_image=sample_control_image,
            prompt='a cottage',
            negative_prompt='blurry, text',
        )

        await sketch_to_image(params=params, bedrock_client=mock_bedrock_runtime_client)

        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['control_strength'] == 0.7
        assert request_body['seed'] == 0
        assert request_body['output_format'] == 'png'
        assert request_body['negative_prompt'] == 'blurry, text'

        # Default filename prefix when caller passes none.
        assert mock_save_images.call_args[1]['filename_prefix'] == 'sketch_to_image'
        assert mock_save_images.call_args[1]['workspace_dir'] is None

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_sketch_reads_control_image_from_file_path(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        image_file_on_disk,
    ):
        """Test a control image given as an existing file path is read and encoded."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/sketch.png']

        params = SketchToImageParams(
            control_image=image_file_on_disk,
            prompt='from a file',
        )

        result = await sketch_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        assert result.status == 'success'
        # The file path must not be forwarded verbatim; it must be base64 of the file.
        with open(image_file_on_disk, 'rb') as handle:
            expected = base64.b64encode(handle.read()).decode('utf-8')
        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['image'] == expected
        assert result.metadata['control_dimensions'] == '128x128'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_sketch_no_images_returned(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
    ):
        """Test an empty images list yields an error response and skips saving."""
        mock_invoke_bedrock.return_value = {'images': []}

        params = SketchToImageParams(
            control_image=sample_control_image,
            prompt='a cottage',
            seed=7,
        )

        result = await sketch_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        assert result.status == 'error'
        assert result.message == 'No images generated'
        assert result.paths == []
        assert result.model_id == STABLE_CONTROL_SKETCH_MODEL_ID
        assert result.seed == 7
        mock_save_images.assert_not_called()

    async def test_sketch_rejects_undersized_control_image(
        self,
        mock_bedrock_runtime_client,
        tiny_image,
    ):
        """Test a control image below the minimum dimension is rejected."""
        params = SketchToImageParams(control_image=tiny_image, prompt='too small')

        with pytest.raises(ValueError, match='below minimum'):
            await sketch_to_image(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
            )


class TestStructureControl:
    """Tests for the structure_control function."""

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_successful_structure_control(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
        temp_workspace_dir,
    ):
        """Test structure control uses the structure model and control_strength."""
        mock_invoke_bedrock.return_value = {'images': ['base64_structure_result']}
        mock_save_images.return_value = ['/path/to/structure.png']

        params = StructureControlParams(
            control_image=sample_control_image,
            prompt='futuristic city following the edge map',
            control_strength=0.4,
            seed=999,
        )

        result = await structure_control(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_structure',
        )

        assert result.status == 'success'
        assert result.message == 'Successfully generated image with structure control'
        assert result.model_id == STABLE_CONTROL_STRUCTURE_MODEL_ID
        assert result.metadata['control_strength'] == 0.4

        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_CONTROL_STRUCTURE_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['image'] == sample_control_image
        assert 'control_image' not in request_body
        assert request_body['control_strength'] == 0.4
        # Structure control must not send the style-guide knob.
        assert 'fidelity' not in request_body
        assert request_body['seed'] == 999
        assert request_body['output_format'] == 'png'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_structure_control_negative_prompt_omitted_when_none(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
    ):
        """Test negative_prompt is absent from the request when it is not set."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/structure.png']

        params = StructureControlParams(
            control_image=sample_control_image,
            prompt='a bridge',
        )

        await structure_control(params=params, bedrock_client=mock_bedrock_runtime_client)

        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert 'negative_prompt' not in request_body
        assert mock_save_images.call_args[1]['filename_prefix'] == 'structure_control'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_structure_control_from_file_path(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        image_file_on_disk,
    ):
        """Test structure control encodes a control image supplied as a file path."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/structure.png']

        params = StructureControlParams(
            control_image=image_file_on_disk,
            prompt='from a file',
            negative_prompt='noise',
        )

        result = await structure_control(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['image'] != image_file_on_disk
        assert base64.b64decode(request_body['image'], validate=True)[:8] == b'\x89PNG\r\n\x1a\n'
        assert request_body['negative_prompt'] == 'noise'
        assert result.metadata['control_dimensions'] == '128x128'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_structure_control_no_images_returned(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
    ):
        """Test a response without an images key yields an error response."""
        mock_invoke_bedrock.return_value = {'seeds': [1]}

        params = StructureControlParams(
            control_image=sample_control_image,
            prompt='a bridge',
        )

        result = await structure_control(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        assert result.status == 'error'
        assert result.paths == []
        assert result.model_id == STABLE_CONTROL_STRUCTURE_MODEL_ID
        mock_save_images.assert_not_called()


class TestStyleGuide:
    """Tests for the style_guide function."""

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_successful_style_guide(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
        temp_workspace_dir,
    ):
        """Test style guide sends 'fidelity' and the reference image as 'image'."""
        mock_invoke_bedrock.return_value = {'images': ['base64_style_guide_result']}
        mock_save_images.return_value = ['/path/to/style_guide.png']

        params = StyleGuideParams(
            reference_image=sample_control_image,
            prompt='a lighthouse in this style',
            fidelity=0.25,
            seed=31337,
            output_format=OutputFormat.JPEG,
        )

        result = await style_guide(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_style_guide',
        )

        assert result.status == 'success'
        assert result.message == 'Successfully generated image with style guide'
        assert result.model_id == STABLE_STYLE_GUIDE_MODEL_ID
        assert result.seed == 31337
        assert result.metadata['fidelity'] == 0.25
        assert result.metadata['reference_dimensions'] == '256x256'

        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_STYLE_GUIDE_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['image'] == sample_control_image
        assert 'reference_image' not in request_body
        # Style guide uses 'fidelity', not 'control_strength'.
        assert request_body['fidelity'] == 0.25
        assert 'control_strength' not in request_body
        assert 'style_strength' not in request_body
        assert request_body['output_format'] == 'jpeg'
        assert mock_save_images.call_args[1]['output_format'] == OutputFormat.JPEG

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_style_guide_default_fidelity_and_negative_prompt(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
    ):
        """Test the default fidelity value and negative_prompt pass-through."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/style_guide.png']

        params = StyleGuideParams(
            reference_image=sample_control_image,
            prompt='a lighthouse',
            negative_prompt='cartoon',
        )

        await style_guide(params=params, bedrock_client=mock_bedrock_runtime_client)

        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['fidelity'] == 0.5
        assert request_body['negative_prompt'] == 'cartoon'
        assert mock_save_images.call_args[1]['filename_prefix'] == 'style_guide'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_style_guide_from_file_path(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        image_file_on_disk,
    ):
        """Test style guide encodes a reference image supplied as a file path."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/style_guide.png']

        params = StyleGuideParams(
            reference_image=image_file_on_disk,
            prompt='from a file',
        )

        result = await style_guide(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['image'] != image_file_on_disk
        assert base64.b64decode(request_body['image'], validate=True)[:8] == b'\x89PNG\r\n\x1a\n'
        assert result.metadata['reference_dimensions'] == '128x128'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_style_guide_no_images_returned(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
    ):
        """Test style guide returns an error response when no images come back."""
        mock_invoke_bedrock.return_value = {'images': []}

        params = StyleGuideParams(
            reference_image=sample_control_image,
            prompt='a lighthouse',
        )

        result = await style_guide(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        assert result.status == 'error'
        assert result.model_id == STABLE_STYLE_GUIDE_MODEL_ID
        mock_save_images.assert_not_called()

    async def test_style_guide_rejects_undersized_reference_image(
        self,
        mock_bedrock_runtime_client,
        tiny_image,
    ):
        """Test a reference image below the minimum dimension is rejected."""
        params = StyleGuideParams(reference_image=tiny_image, prompt='too small')

        with pytest.raises(ValueError, match='below minimum'):
            await style_guide(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
            )


class TestStyleTransfer:
    """Tests for the style_transfer function."""

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_successful_style_transfer(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
        sample_style_image,
        temp_workspace_dir,
    ):
        """Test style transfer sends init_image, style_image and all three strengths."""
        mock_invoke_bedrock.return_value = {'images': ['base64_style_transfer_result']}
        mock_save_images.return_value = ['/path/to/style_transfer.png']

        params = StyleTransferParams(
            init_image=sample_control_image,
            style_image=sample_style_image,
            prompt='van gogh treatment',
            composition_fidelity=0.6,
            style_strength=0.8,
            change_strength=0.3,
            seed=555,
        )

        result = await style_transfer(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename='test_style_transfer',
        )

        assert result.status == 'success'
        assert result.message == 'Successfully transferred style'
        assert result.model_id == STABLE_STYLE_TRANSFER_MODEL_ID
        assert result.seed == 555
        assert result.metadata['composition_fidelity'] == 0.6
        assert result.metadata['style_strength'] == 0.8
        assert result.metadata['change_strength'] == 0.3
        assert result.metadata['init_dimensions'] == '256x256'
        assert result.metadata['style_dimensions'] == '320x320'

        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == STABLE_STYLE_TRANSFER_MODEL_ID
        request_body = call_args['request_body']
        # Style transfer keeps both images under their own keys, not 'image'.
        assert request_body['init_image'] == sample_control_image
        assert request_body['style_image'] == sample_style_image
        assert 'image' not in request_body
        assert request_body['composition_fidelity'] == 0.6
        assert request_body['style_strength'] == 0.8
        assert request_body['change_strength'] == 0.3
        # Style transfer must not send the single-image control knobs.
        assert 'control_strength' not in request_body
        assert 'fidelity' not in request_body
        assert request_body['seed'] == 555
        assert request_body['output_format'] == 'png'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_style_transfer_defaults_and_negative_prompt(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
        sample_style_image,
    ):
        """Test the default strength values and negative_prompt inclusion."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/style_transfer.png']

        params = StyleTransferParams(
            init_image=sample_control_image,
            style_image=sample_style_image,
            prompt='impressionist',
            negative_prompt='photorealistic',
        )

        await style_transfer(params=params, bedrock_client=mock_bedrock_runtime_client)

        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['composition_fidelity'] == 0.9
        assert request_body['style_strength'] == 1.0
        assert request_body['change_strength'] == 0.9
        assert request_body['negative_prompt'] == 'photorealistic'
        assert mock_save_images.call_args[1]['filename_prefix'] == 'style_transfer'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_style_transfer_mixes_file_path_and_base64(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        image_file_on_disk,
        sample_style_image,
    ):
        """Test init image from disk and style image from base64 are handled separately."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/style_transfer.png']

        params = StyleTransferParams(
            init_image=image_file_on_disk,
            style_image=sample_style_image,
            prompt='mixed inputs',
        )

        result = await style_transfer(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        with open(image_file_on_disk, 'rb') as handle:
            expected_init = base64.b64encode(handle.read()).decode('utf-8')
        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['init_image'] == expected_init
        assert request_body['style_image'] == sample_style_image
        assert result.metadata['init_dimensions'] == '128x128'
        assert result.metadata['style_dimensions'] == '320x320'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_style_transfer_style_image_from_file_path(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
        image_file_on_disk,
    ):
        """Test a style image given as a file path is encoded into style_image."""
        mock_invoke_bedrock.return_value = {'images': ['img']}
        mock_save_images.return_value = ['/path/to/style_transfer.png']

        params = StyleTransferParams(
            init_image=sample_control_image,
            style_image=image_file_on_disk,
            prompt='style from disk',
        )

        result = await style_transfer(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        with open(image_file_on_disk, 'rb') as handle:
            expected_style = base64.b64encode(handle.read()).decode('utf-8')
        request_body = mock_invoke_bedrock.call_args[1]['request_body']
        assert request_body['style_image'] == expected_style
        assert request_body['init_image'] == sample_control_image
        assert result.metadata['init_dimensions'] == '256x256'
        assert result.metadata['style_dimensions'] == '128x128'

    @patch(f'{CONTROL_MODULE}.invoke_bedrock_model')
    @patch(f'{CONTROL_MODULE}.save_images')
    async def test_style_transfer_no_images_returned(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        sample_control_image,
        sample_style_image,
    ):
        """Test style transfer returns an error response when no images come back."""
        mock_invoke_bedrock.return_value = {'images': []}

        params = StyleTransferParams(
            init_image=sample_control_image,
            style_image=sample_style_image,
            prompt='impressionist',
            seed=11,
        )

        result = await style_transfer(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
        )

        assert result.status == 'error'
        assert result.message == 'No images generated'
        assert result.model_id == STABLE_STYLE_TRANSFER_MODEL_ID
        assert result.seed == 11
        mock_save_images.assert_not_called()

    async def test_style_transfer_rejects_undersized_style_image(
        self,
        mock_bedrock_runtime_client,
        sample_control_image,
        tiny_image,
    ):
        """Test an undersized style image is rejected even when init image is valid."""
        params = StyleTransferParams(
            init_image=sample_control_image,
            style_image=tiny_image,
            prompt='too small',
        )

        with pytest.raises(ValueError, match='below minimum'):
            await style_transfer(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
            )

    async def test_style_transfer_rejects_invalid_base64_init_image(
        self,
        mock_bedrock_runtime_client,
        sample_style_image,
    ):
        """Test a non-base64, non-existent init image value is rejected."""
        params = StyleTransferParams(
            init_image='not base64 and not a path',
            style_image=sample_style_image,
            prompt='bad input',
        )

        with pytest.raises(ValueError, match='Failed to decode base64 image'):
            await style_transfer(
                params=params,
                bedrock_client=mock_bedrock_runtime_client,
            )
