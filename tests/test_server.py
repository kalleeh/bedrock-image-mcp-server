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
"""Tests for the server module of the bedrock-image-mcp-server."""

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

        # Check that ctx.error reported the failure exactly once
        assert mock_context.error.call_count == 1
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
        assert 'Unexpected error' in str(mock_context.error.call_args_list)


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

        # Check that ctx.error reported the failure exactly once
        assert mock_context.error.call_count == 1
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
        assert 'Unexpected error' in str(mock_context.error.call_args_list)


class TestServerIntegration:
    """Integration tests for the server module."""

    async def test_expected_tools_are_registered(self):
        """Test that every expected tool name is registered with the MCP server."""
        from awslabs.bedrock_image_mcp_server.server import mcp

        expected = {
            'generate_image',
            'generate_image_with_colors',
            'generate_image_sd35',
            'transform_image_sd35',
            'upscale_creative',
            'upscale_conservative',
            'upscale_fast',
            'inpaint_image',
            'outpaint_image',
            'search_and_replace',
            'search_and_recolor',
            'remove_object',
            'remove_background',
            'sketch_to_image',
            'structure_control',
            'style_guide',
            'style_transfer',
            'create_rectangular_mask',
            'create_ellipse_mask',
            'create_full_mask',
        }

        registered = {tool.name for tool in await mcp.list_tools()}

        assert expected == registered

    async def test_registered_tools_expose_required_parameters(self):
        """Test that every registered tool accepts workspace_dir and reports errors via ctx."""
        import inspect
        from awslabs.bedrock_image_mcp_server.server import mcp

        for tool in await mcp.list_tools():
            registered = mcp._tool_manager.get_tool(tool.name)
            assert registered is not None, f'Tool {tool.name} not resolvable from the registry'
            sig = inspect.signature(registered.fn)
            assert 'workspace_dir' in sig.parameters, (
                f'Tool {tool.name} missing workspace_dir parameter'
            )
            assert 'ctx' in sig.parameters, (
                f'Tool {tool.name} missing ctx parameter for error reporting'
            )
            assert tool.description, f'Tool {tool.name} missing description'


class TestValidationHelpers:
    """Tests for the shared parameter-validation helpers."""

    def test_output_format_is_case_insensitive(self):
        """Test that uppercase format names are accepted."""
        from awslabs.bedrock_image_mcp_server.server import parse_output_format

        assert parse_output_format('PNG').value == 'png'
        assert parse_output_format('webp').value == 'webp'

    def test_invalid_output_format_lists_valid_options(self):
        """Test that an unsupported format raises and names the supported ones."""
        from awslabs.bedrock_image_mcp_server.server import parse_output_format

        with pytest.raises(ValueError, match='jpeg, png, webp'):
            parse_output_format('tiff')

    def test_invalid_aspect_ratio_lists_valid_options(self):
        """Test that an unsupported aspect ratio raises and names the supported ones."""
        from awslabs.bedrock_image_mcp_server.server import parse_aspect_ratio

        with pytest.raises(ValueError, match='16:9'):
            parse_aspect_ratio('BOGUS')


class TestErrorsAreReportedOnce:
    """Tests that a failing tool reports to the MCP context exactly once."""

    async def test_invalid_aspect_ratio_reports_once(self, mock_context):
        """Test the SD3.5 aspect-ratio failure is not double-reported to the client."""
        from awslabs.bedrock_image_mcp_server.server import mcp_generate_image_sd35

        with pytest.raises(ValueError, match='Invalid aspect ratio'):
            await mcp_generate_image_sd35(
                ctx=mock_context,
                prompt='a cat',
                aspect_ratio='BOGUS',
                negative_prompt=None,
                seed=0,
                output_format='png',
                workspace_dir=None,
                filename=None,
            )

        assert mock_context.error.call_count == 1

    async def test_invalid_output_format_reports_once(self, mock_context):
        """Test a control tool's format failure is not double-reported to the client."""
        from awslabs.bedrock_image_mcp_server.server import mcp_sketch_to_image

        with pytest.raises(ValueError, match='Invalid output format'):
            await mcp_sketch_to_image(
                ctx=mock_context,
                sketch='x' * 100,
                prompt='a cat',
                control_strength=0.7,
                negative_prompt=None,
                seed=0,
                output_format='tiff',
                workspace_dir=None,
                filename=None,
            )

        assert mock_context.error.call_count == 1
