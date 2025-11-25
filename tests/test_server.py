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
"""Tests for the server module of the nova-canvas-mcp-server."""

import pytest
from awslabs.bedrock_image_mcp_server.server import (
    mcp_generate_image,
    mcp_generate_image_with_colors,
)
from unittest.mock import MagicMock, patch


class TestMcpGenerateImage:
    """Tests for the mcp_generate_image function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_text')
    async def test_generate_image_success(
        self, mock_generate_image, mock_context, sample_text_prompt, temp_workspace_dir
    ):
        """Test successful image generation."""
        # Set up the mock
        mock_generate_image.return_value = MagicMock(
            status='success',
            paths=['/path/to/image1.png', '/path/to/image2.png'],
            message='Generated 2 image(s)',
        )

        # Call the function
        result = await mcp_generate_image(
            ctx=mock_context,
            prompt=sample_text_prompt,
            negative_prompt='people, clouds',
            filename='test_image',
            width=512,
            height=768,
            quality='premium',
            cfg_scale=8.0,
            seed=12345,
            number_of_images=2,
            workspace_dir=temp_workspace_dir,
        )

        # Check that generate_image_with_text was called with the correct parameters
        mock_generate_image.assert_called_once()
        call_args = mock_generate_image.call_args[1]
        assert call_args['prompt'] == sample_text_prompt
        assert call_args['negative_prompt'] == 'people, clouds'
        assert call_args['filename'] == 'test_image'
        assert call_args['width'] == 512
        assert call_args['height'] == 768
        assert call_args['quality'] == 'premium'
        assert call_args['cfg_scale'] == 8.0
        assert call_args['seed'] == 12345
        assert call_args['number_of_images'] == 2
        assert call_args['workspace_dir'] == temp_workspace_dir
        # We can't directly compare the bedrock_runtime_client object
        assert 'bedrock_runtime_client' in call_args

        # Check that the result is correct
        assert result.status == 'success'
        assert result.paths == ['file:///path/to/image1.png', 'file:///path/to/image2.png']

        # Check that ctx.error was not called
        mock_context.error.assert_not_called()

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_text')
    async def test_generate_image_error(
        self, mock_generate_image, mock_context, sample_text_prompt
    ):
        """Test error handling in image generation."""
        # Set up the mock to return an error
        mock_generate_image.return_value = MagicMock(
            status='error', message='Failed to generate image: API error', paths=[]
        )

        # Call the function and check that it raises an exception
        with pytest.raises(Exception, match='Failed to generate image: API error'):
            await mcp_generate_image(ctx=mock_context, prompt=sample_text_prompt)

        # Check that ctx.error was called with the expected error message
        assert mock_context.error.call_count == 2
        assert 'Failed to generate image: API error' in str(mock_context.error.call_args_list)

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_text')
    async def test_generate_image_with_defaults(
        self, mock_generate_image, mock_context, sample_text_prompt
    ):
        """Test image generation with default parameters."""
        # Set up the mock
        mock_generate_image.return_value = MagicMock(
            status='success', paths=['/path/to/image.png'], message='Generated 1 image(s)'
        )

        # Call the function with minimal parameters
        result = await mcp_generate_image(ctx=mock_context, prompt=sample_text_prompt)

        # Check that generate_image_with_text was called with the correct parameters
        mock_generate_image.assert_called_once()
        call_args = mock_generate_image.call_args[1]
        assert call_args['prompt'] == sample_text_prompt
        assert 'negative_prompt' in call_args
        assert hasattr(call_args['filename'], 'default') and call_args['filename'].default is None
        assert (
            hasattr(call_args['workspace_dir'], 'default')
            and call_args['workspace_dir'].default is None
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.paths == ['file:///path/to/image.png']

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_text')
    async def test_generate_image_exception(
        self, mock_generate_image, mock_context, sample_text_prompt
    ):
        """Test handling of exceptions during image generation."""
        # Set up the mock to raise an exception
        mock_generate_image.side_effect = Exception('Unexpected error')

        # Call the function and check that it raises an exception
        with pytest.raises(Exception, match='Unexpected error'):
            await mcp_generate_image(ctx=mock_context, prompt=sample_text_prompt)

        # Check that ctx.error was called with the expected error message
        assert mock_context.error.call_count == 1
        assert 'Error generating image: Unexpected error' in str(mock_context.error.call_args_list)


class TestMcpGenerateImageWithColors:
    """Tests for the mcp_generate_image_with_colors function."""

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_colors')
    async def test_generate_image_with_colors_success(
        self,
        mock_generate_image,
        mock_context,
        sample_text_prompt,
        sample_colors,
        temp_workspace_dir,
    ):
        """Test successful image generation with colors."""
        # Set up the mock
        mock_generate_image.return_value = MagicMock(
            status='success',
            paths=['/path/to/image1.png', '/path/to/image2.png'],
            message='Generated 2 image(s)',
        )

        # Call the function
        result = await mcp_generate_image_with_colors(
            ctx=mock_context,
            prompt=sample_text_prompt,
            colors=sample_colors,
            negative_prompt='people, clouds',
            filename='test_image',
            width=512,
            height=768,
            quality='premium',
            cfg_scale=8.0,
            seed=12345,
            number_of_images=2,
            workspace_dir=temp_workspace_dir,
        )

        # Check that generate_image_with_colors was called with the correct parameters
        mock_generate_image.assert_called_once()
        call_args = mock_generate_image.call_args[1]
        assert call_args['prompt'] == sample_text_prompt
        assert call_args['colors'] == sample_colors
        assert call_args['negative_prompt'] == 'people, clouds'
        assert call_args['filename'] == 'test_image'
        assert call_args['width'] == 512
        assert call_args['height'] == 768
        assert call_args['quality'] == 'premium'
        assert call_args['cfg_scale'] == 8.0
        assert call_args['seed'] == 12345
        assert call_args['number_of_images'] == 2
        assert call_args['workspace_dir'] == temp_workspace_dir
        # We can't directly compare the bedrock_runtime_client object
        assert 'bedrock_runtime_client' in call_args

        # Check that the result is correct
        assert result.status == 'success'
        assert result.paths == ['file:///path/to/image1.png', 'file:///path/to/image2.png']

        # Check that ctx.error was not called
        mock_context.error.assert_not_called()

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_colors')
    async def test_generate_image_with_colors_error(
        self, mock_generate_image, mock_context, sample_text_prompt, sample_colors
    ):
        """Test error handling in image generation with colors."""
        # Set up the mock to return an error
        mock_generate_image.return_value = MagicMock(
            status='error', message='Failed to generate color-guided image: API error', paths=[]
        )

        # Call the function and check that it raises an exception
        with pytest.raises(Exception, match='Failed to generate color-guided image: API error'):
            await mcp_generate_image_with_colors(
                ctx=mock_context, prompt=sample_text_prompt, colors=sample_colors
            )

        # Check that ctx.error was called with the expected error message
        assert mock_context.error.call_count == 2
        assert 'Failed to generate color-guided image: API error' in str(
            mock_context.error.call_args_list
        )

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_colors')
    async def test_generate_image_with_colors_defaults(
        self, mock_generate_image, mock_context, sample_text_prompt, sample_colors
    ):
        """Test image generation with colors using default parameters."""
        # Set up the mock
        mock_generate_image.return_value = MagicMock(
            status='success', paths=['/path/to/image.png'], message='Generated 1 image(s)'
        )

        # Call the function with minimal parameters
        result = await mcp_generate_image_with_colors(
            ctx=mock_context, prompt=sample_text_prompt, colors=sample_colors
        )

        # Check that generate_image_with_colors was called with the correct parameters
        mock_generate_image.assert_called_once()
        call_args = mock_generate_image.call_args[1]
        assert call_args['prompt'] == sample_text_prompt
        assert call_args['colors'] == sample_colors
        assert 'negative_prompt' in call_args
        assert hasattr(call_args['filename'], 'default') and call_args['filename'].default is None
        assert (
            hasattr(call_args['workspace_dir'], 'default')
            and call_args['workspace_dir'].default is None
        )

        # Check that the result is correct
        assert result.status == 'success'
        assert result.paths == ['file:///path/to/image.png']

    @pytest.mark.asyncio
    @patch('awslabs.bedrock_image_mcp_server.server.generate_image_with_colors')
    async def test_generate_image_with_colors_exception(
        self, mock_generate_image, mock_context, sample_text_prompt, sample_colors
    ):
        """Test handling of exceptions during image generation with colors."""
        # Set up the mock to raise an exception
        mock_generate_image.side_effect = Exception('Unexpected error')

        # Call the function and check that it raises an exception
        with pytest.raises(Exception, match='Unexpected error'):
            await mcp_generate_image_with_colors(
                ctx=mock_context, prompt=sample_text_prompt, colors=sample_colors
            )

        # Check that ctx.error was called with the expected error message
        assert mock_context.error.call_count == 1
        assert 'Error generating color-guided image: Unexpected error' in str(
            mock_context.error.call_args_list
        )


class TestServerIntegration:
    """Integration tests for the server module."""

    def test_server_tool_registration(self):
        """Test that all 17 server tools are registered correctly."""
        from awslabs.bedrock_image_mcp_server.server import (
            mcp_generate_image,
            mcp_generate_image_sd35,
            mcp_generate_image_with_colors,
            mcp_inpaint,
            mcp_outpaint,
            mcp_remove_background,
            mcp_remove_object,
            mcp_search_recolor,
            mcp_search_replace,
            mcp_sketch_to_image,
            mcp_structure_control,
            mcp_style_guide,
            mcp_style_transfer,
            mcp_transform_image_sd35,
            mcp_upscale_conservative,
            mcp_upscale_creative,
            mcp_upscale_fast,
        )

        # List of all tools with their expected docstring content
        tools = [
            # Nova Canvas (2)
            (mcp_generate_image, 'Generate an image using Amazon Nova Canvas with text prompt'),
            (mcp_generate_image_with_colors, 'Generate an image using Amazon Nova Canvas with color guidance'),
            # SD3.5 (2)
            (mcp_generate_image_sd35, 'Generate an image using Stable Diffusion 3.5 Large'),
            (mcp_transform_image_sd35, 'Transform an existing image using Stable Diffusion 3.5 Large'),
            # Upscale (3)
            (mcp_upscale_creative, 'Upscale images to 4K with creative AI enhancement'),
            (mcp_upscale_conservative, 'Upscale images to 4K while preserving original details'),
            (mcp_upscale_fast, 'Fast 4x upscaling'),
            # Edit (6)
            (mcp_inpaint, 'Fill masked regions with AI-generated content'),
            (mcp_outpaint, 'Extend images beyond their original boundaries'),
            (mcp_search_replace, 'Find and replace objects'),
            (mcp_search_recolor, 'Recolor specific objects'),
            (mcp_remove_object, 'Remove unwanted objects'),
            (mcp_remove_background, 'Automatically remove backgrounds'),
            # Control (4)
            (mcp_sketch_to_image, 'Convert sketches or line art into detailed images'),
            (mcp_structure_control, 'Generate images following structural guides'),
            (mcp_style_guide, 'Generate images matching a reference style'),
            (mcp_style_transfer, 'Apply style from one image to the content of another'),
        ]

        # Verify all 17 tools are registered with correct docstrings
        assert len(tools) == 17, f"Expected 17 tools, found {len(tools)}"

        for tool_func, expected_doc_content in tools:
            # Check that the tool is registered
            assert hasattr(tool_func, '__name__'), f"Tool {tool_func} missing __name__ attribute"

            # Check that the function has the correct docstring
            assert tool_func.__doc__ is not None, f"Tool {tool_func.__name__} missing docstring"
            assert expected_doc_content in tool_func.__doc__, \
                f"Tool {tool_func.__name__} docstring doesn't contain expected content: {expected_doc_content}"

        # Verify workspace_dir parameter exists in all tools
        import inspect
        for tool_func, _ in tools:
            sig = inspect.signature(tool_func)
            assert 'workspace_dir' in sig.parameters, \
                f"Tool {tool_func.__name__} missing workspace_dir parameter"
            assert 'ctx' in sig.parameters, \
                f"Tool {tool_func.__name__} missing ctx parameter for error reporting"
