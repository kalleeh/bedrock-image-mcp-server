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

import importlib
import os
import pytest
from unittest.mock import patch


class TestDefaultRegion:
    """Tests for the AWS region the server falls back to."""

    def test_default_region_serves_every_bedrock_backed_tool(self):
        """An unset AWS_REGION must land in us-west-2, the only fully-capable region.

        us-west-2 is the sole home of SD3.5, Ultra and Core, and it also serves the 13
        Stability edit/upscale/control tools. Defaulting anywhere else (us-east-1 and
        us-east-2 are identical strict subsets) makes the four recommended text-to-image
        tools fail with an invalid model identifier for anyone who never sets the variable.

        The region is resolved at import time, so the module is reloaded under a cleared
        environment rather than inspected as source.
        """
        import awslabs.bedrock_image_mcp_server.server as server_module

        with patch.dict(os.environ, {}, clear=True):
            reloaded = importlib.reload(server_module)
            try:
                assert reloaded.aws_region == 'us-west-2'
            finally:
                # Restore the module under the real environment so later tests and other
                # modules keep the client this reload replaced.
                importlib.reload(server_module)


class TestServerIntegration:
    """Integration tests for the server module."""

    async def test_expected_tools_are_registered(self):
        """Test that every expected tool name is registered with the MCP server."""
        from awslabs.bedrock_image_mcp_server.server import mcp

        expected = {
            'generate_image_sd35',
            'generate_image_ultra',
            'generate_image_core',
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
        """Test that every registered tool accepts workspace_dir and an injected context."""
        import inspect
        from awslabs.bedrock_image_mcp_server.server import mcp

        for tool in await mcp.list_tools():
            registered = mcp._tool_manager.get_tool(tool.name)
            assert registered is not None, f'Tool {tool.name} not resolvable from the registry'
            sig = inspect.signature(registered.fn)
            assert 'workspace_dir' in sig.parameters, (
                f'Tool {tool.name} missing workspace_dir parameter'
            )
            assert 'ctx' in sig.parameters, f'Tool {tool.name} missing ctx parameter'
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
    """Tests that a failing tool logs the failure exactly once."""

    async def test_invalid_aspect_ratio_reports_once(self, mock_context, logged_errors):
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

        assert len(logged_errors) == 1

    async def test_invalid_output_format_reports_once(self, mock_context, logged_errors):
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

        assert len(logged_errors) == 1
