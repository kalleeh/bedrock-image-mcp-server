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
"""Tests for the SD3.5 service module."""

import base64
import pytest
from awslabs.bedrock_image_mcp_server.consts import SD35_LARGE_MODEL_ID
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.sd35_models import (
    AspectRatio,
    GenerationMode,
    SD35ImageToImageParams,
    SD35TextToImageParams,
)
from awslabs.bedrock_image_mcp_server.services.sd35_service import (
    build_sd35_request,
    generate_image_to_image,
    generate_text_to_image,
)
from io import BytesIO
from PIL import Image
from unittest.mock import MagicMock, patch


def create_test_image_base64(width=128, height=128, format='PNG'):
    """Create a valid test image and return as base64 string."""
    img = Image.new('RGB', (width, height), color='red')
    buffer = BytesIO()
    img.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


class TestBuildSD35Request:
    """Tests for the build_sd35_request function."""

    def test_text_to_image_request(self):
        """Test building request for text-to-image mode."""
        params = SD35TextToImageParams(
            prompt="A beautiful mountain landscape",
            aspect_ratio=AspectRatio.RATIO_16_9,
            seed=12345,
            negative_prompt="low quality",
            output_format=OutputFormat.PNG
        )

        request = build_sd35_request(params)

        assert request["prompt"] == "A beautiful mountain landscape"
        assert request["mode"] == GenerationMode.TEXT_TO_IMAGE.value
        assert request["aspect_ratio"] == "16:9"
        assert request["seed"] == 12345
        assert request["negative_prompt"] == "low quality"
        assert request["output_format"] == "png"

    def test_text_to_image_without_negative_prompt(self):
        """Test building request without negative prompt."""
        params = SD35TextToImageParams(
            prompt="A serene lake",
            aspect_ratio=AspectRatio.RATIO_1_1,
            seed=0
        )

        request = build_sd35_request(params)

        assert request["prompt"] == "A serene lake"
        assert request["mode"] == GenerationMode.TEXT_TO_IMAGE.value
        assert request["aspect_ratio"] == "1:1"
        assert request["seed"] == 0
        assert "negative_prompt" not in request

    def test_image_to_image_request(self):
        """Test building request for image-to-image mode."""
        base64_image = create_test_image_base64()
        params = SD35ImageToImageParams(
            prompt="Transform to watercolor",
            image=base64_image,
            strength=0.7,
            seed=54321,
            negative_prompt="photorealistic",
            output_format=OutputFormat.JPEG
        )

        request = build_sd35_request(params)

        assert request["prompt"] == "Transform to watercolor"
        assert request["mode"] == GenerationMode.IMAGE_TO_IMAGE.value
        assert request["image"] == base64_image
        assert request["strength"] == 0.7
        assert request["seed"] == 54321
        assert request["negative_prompt"] == "photorealistic"
        assert request["output_format"] == "jpeg"
        assert "aspect_ratio" not in request

    def test_image_to_image_without_negative_prompt(self):
        """Test building image-to-image request without negative prompt."""
        base64_image = create_test_image_base64()
        params = SD35ImageToImageParams(
            prompt="Add sunset colors",
            image=base64_image,
            strength=0.5,
            seed=0
        )

        request = build_sd35_request(params)

        assert request["prompt"] == "Add sunset colors"
        assert request["mode"] == GenerationMode.IMAGE_TO_IMAGE.value
        assert request["image"] == base64_image
        assert request["strength"] == 0.5
        assert "negative_prompt" not in request


class TestGenerateTextToImage:
    """Tests for the generate_text_to_image function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.save_images')
    async def test_successful_generation(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        temp_workspace_dir
    ):
        """Test successful text-to-image generation."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {
            'images': ['base64_image_1'],
            'finish_reasons': ['SUCCESS']
        }
        mock_save_images.return_value = ['/path/to/sd35_image.png']

        # Create parameters
        params = SD35TextToImageParams(
            prompt="A futuristic cityscape at night",
            aspect_ratio=AspectRatio.RATIO_16_9,
            seed=12345,
            negative_prompt="daytime, rural",
            output_format=OutputFormat.PNG
        )

        # Call the function
        result = await generate_text_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename="test_sd35"
        )

        # Verify result
        assert result.status == 'success'
        assert 'Successfully generated 1 image(s)' in result.message
        assert result.paths == ['/path/to/sd35_image.png']
        assert result.model_id == SD35_LARGE_MODEL_ID
        assert result.prompt == "A futuristic cityscape at night"
        assert result.seed == 12345
        assert result.metadata['aspect_ratio'] == "16:9"
        assert result.metadata['mode'] == GenerationMode.TEXT_TO_IMAGE.value
        assert result.metadata['finish_reasons'] == ['SUCCESS']

        # Verify invoke_bedrock_model was called correctly
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == SD35_LARGE_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['prompt'] == "A futuristic cityscape at night"
        assert request_body['mode'] == GenerationMode.TEXT_TO_IMAGE.value
        assert request_body['aspect_ratio'] == "16:9"
        assert request_body['seed'] == 12345
        assert request_body['negative_prompt'] == "daytime, rural"

        # Verify save_images was called correctly
        mock_save_images.assert_called_once_with(
            base64_images=['base64_image_1'],
            workspace_dir=temp_workspace_dir,
            filename_prefix='test_sd35',
            output_format=OutputFormat.PNG
        )

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.save_images')
    async def test_generation_with_default_filename(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client
    ):
        """Test generation with default filename prefix."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {
            'images': ['base64_image_1'],
            'finish_reasons': ['SUCCESS']
        }
        mock_save_images.return_value = ['/path/to/sd35_image.png']

        # Create parameters
        params = SD35TextToImageParams(
            prompt="A peaceful forest",
            aspect_ratio=AspectRatio.RATIO_1_1,
            seed=0
        )

        # Call the function without filename
        result = await generate_text_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client
        )

        # Verify result
        assert result.status == 'success'
        assert result.paths == ['/path/to/sd35_image.png']

        # Verify save_images was called with default prefix
        mock_save_images.assert_called_once()
        call_args = mock_save_images.call_args[1]
        assert call_args['filename_prefix'] == 'sd35'

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    async def test_api_error(
        self,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client
    ):
        """Test handling of API errors."""
        # Set up mock to raise exception
        mock_invoke_bedrock.side_effect = Exception("Bedrock API error")

        # Create parameters
        params = SD35TextToImageParams(
            prompt="Test prompt",
            aspect_ratio=AspectRatio.RATIO_1_1,
            seed=0
        )

        # Verify exception is raised
        with pytest.raises(Exception, match="Bedrock API error"):
            await generate_text_to_image(
                params=params,
                bedrock_client=mock_bedrock_runtime_client
            )

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    async def test_no_images_returned(
        self,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client
    ):
        """Test handling when no images are returned."""
        # Set up mock to return empty images list
        mock_invoke_bedrock.return_value = {'images': []}

        # Create parameters
        params = SD35TextToImageParams(
            prompt="Test prompt",
            aspect_ratio=AspectRatio.RATIO_1_1,
            seed=0
        )

        # Verify exception is raised
        with pytest.raises(ValueError, match="No images returned from Bedrock API"):
            await generate_text_to_image(
                params=params,
                bedrock_client=mock_bedrock_runtime_client
            )


class TestGenerateImageToImage:
    """Tests for the generate_image_to_image function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.save_images')
    async def test_successful_generation_with_base64(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        temp_workspace_dir
    ):
        """Test successful image-to-image generation with base64 input."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {
            'images': ['base64_transformed_image'],
            'finish_reasons': ['SUCCESS']
        }
        mock_save_images.return_value = ['/path/to/sd35_transform_image.png']

        # Create base64 image
        base64_image = create_test_image_base64()

        # Create parameters
        params = SD35ImageToImageParams(
            prompt="Transform to oil painting style",
            image=base64_image,
            strength=0.7,
            seed=54321,
            negative_prompt="photorealistic",
            output_format=OutputFormat.PNG
        )

        # Call the function
        result = await generate_image_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir,
            filename="test_transform"
        )

        # Verify result
        assert result.status == 'success'
        assert 'Successfully transformed image' in result.message
        assert result.paths == ['/path/to/sd35_transform_image.png']
        assert result.model_id == SD35_LARGE_MODEL_ID
        assert result.prompt == "Transform to oil painting style"
        assert result.seed == 54321
        assert result.metadata['strength'] == 0.7
        assert result.metadata['mode'] == GenerationMode.IMAGE_TO_IMAGE.value
        assert result.metadata['finish_reasons'] == ['SUCCESS']

        # Verify invoke_bedrock_model was called correctly
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        assert call_args['model_id'] == SD35_LARGE_MODEL_ID
        request_body = call_args['request_body']
        assert request_body['prompt'] == "Transform to oil painting style"
        assert request_body['mode'] == GenerationMode.IMAGE_TO_IMAGE.value
        assert request_body['image'] == base64_image
        assert request_body['strength'] == 0.7
        assert request_body['seed'] == 54321
        assert request_body['negative_prompt'] == "photorealistic"

        # Verify save_images was called correctly
        mock_save_images.assert_called_once_with(
            base64_images=['base64_transformed_image'],
            workspace_dir=temp_workspace_dir,
            filename_prefix='test_transform',
            output_format=OutputFormat.PNG
        )

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.save_images')
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.os.path.exists')
    @patch('builtins.open', create=True)
    async def test_successful_generation_with_file_path(
        self,
        mock_open,
        mock_exists,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client,
        temp_workspace_dir
    ):
        """Test successful image-to-image generation with file path input."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {
            'images': ['base64_transformed_image'],
            'finish_reasons': ['SUCCESS']
        }
        mock_save_images.return_value = ['/path/to/sd35_transform_image.png']

        # Mock file existence check
        mock_exists.return_value = True

        # Create a real image in memory for the file read
        img = Image.new('RGB', (128, 128), color='blue')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        image_bytes = buffer.read()

        # Mock file open to return our image bytes
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = image_bytes
        mock_open.return_value = mock_file

        # Create parameters with file path (will be converted to base64 in service)
        # Note: We need to pass base64 to params since validator runs first
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        params = SD35ImageToImageParams(
            prompt="Add vibrant colors",
            image=base64_image,
            strength=0.5,
            seed=0,
            output_format=OutputFormat.WEBP
        )

        # Override the image with file path after validation
        # This simulates what would happen if we passed a file path to the service
        params.image = '/path/to/input_image.png'

        # Call the function
        result = await generate_image_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client,
            workspace_dir=temp_workspace_dir
        )

        # Verify result
        assert result.status == 'success'
        assert result.paths == ['/path/to/sd35_transform_image.png']
        assert result.metadata['strength'] == 0.5

        # Verify invoke_bedrock_model was called with base64-encoded image
        mock_invoke_bedrock.assert_called_once()
        call_args = mock_invoke_bedrock.call_args[1]
        request_body = call_args['request_body']
        # The image should have been converted to base64
        assert isinstance(request_body['image'], str)
        assert len(request_body['image']) > 0

        # Verify save_images was called with default prefix
        mock_save_images.assert_called_once()
        call_args = mock_save_images.call_args[1]
        assert call_args['filename_prefix'] == 'sd35_transform'
        assert call_args['output_format'] == OutputFormat.WEBP

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.save_images')
    async def test_generation_with_default_filename(
        self,
        mock_save_images,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client
    ):
        """Test generation with default filename prefix."""
        # Set up mocks
        mock_invoke_bedrock.return_value = {
            'images': ['base64_image_1'],
            'finish_reasons': ['SUCCESS']
        }
        mock_save_images.return_value = ['/path/to/sd35_transform_image.png']

        # Create base64 image
        base64_image = create_test_image_base64()

        # Create parameters
        params = SD35ImageToImageParams(
            prompt="Enhance details",
            image=base64_image,
            strength=0.3,
            seed=0
        )

        # Call the function without filename
        result = await generate_image_to_image(
            params=params,
            bedrock_client=mock_bedrock_runtime_client
        )

        # Verify result
        assert result.status == 'success'
        assert result.paths == ['/path/to/sd35_transform_image.png']

        # Verify save_images was called with default prefix
        mock_save_images.assert_called_once()
        call_args = mock_save_images.call_args[1]
        assert call_args['filename_prefix'] == 'sd35_transform'

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    async def test_api_error(
        self,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client
    ):
        """Test handling of API errors."""
        # Set up mock to raise exception
        mock_invoke_bedrock.side_effect = Exception("Bedrock API error")

        # Create base64 image
        base64_image = create_test_image_base64()

        # Create parameters
        params = SD35ImageToImageParams(
            prompt="Test prompt",
            image=base64_image,
            strength=0.5,
            seed=0
        )

        # Verify exception is raised
        with pytest.raises(Exception, match="Bedrock API error"):
            await generate_image_to_image(
                params=params,
                bedrock_client=mock_bedrock_runtime_client
            )

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.services.sd35_service.invoke_bedrock_model')
    async def test_no_images_returned(
        self,
        mock_invoke_bedrock,
        mock_bedrock_runtime_client
    ):
        """Test handling when no images are returned."""
        # Set up mock to return empty images list
        mock_invoke_bedrock.return_value = {'images': []}

        # Create base64 image
        base64_image = create_test_image_base64()

        # Create parameters
        params = SD35ImageToImageParams(
            prompt="Test prompt",
            image=base64_image,
            strength=0.5,
            seed=0
        )

        # Verify exception is raised
        with pytest.raises(ValueError, match="No images returned from Bedrock API"):
            await generate_image_to_image(
                params=params,
                bedrock_client=mock_bedrock_runtime_client
            )
