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
"""Tests for the MCP tool wrappers in server.py that are not covered by test_server.py.

Every tool exposed by the server is exercised here for three things:

1. The success path returns ``status='success'`` with ``file://`` URIs, and the parameter
   model handed to the underlying service carries the values the caller supplied under the
   field names that service expects.
2. The failure path (service reports ``status='error'``) raises and reports the failure to
   ``ctx.error`` exactly once -- a double report is a regression.
3. Caller-supplied ``output_format`` strings are normalized case-insensitively and rejected
   when unsupported, again reporting to ``ctx.error`` exactly once.

The tools are FastMCP-decorated coroutines whose defaults are ``pydantic.Field`` objects, so
calling them directly requires passing every parameter explicitly.
"""

import base64
import os
import pytest
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.sd35_models import AspectRatio
from awslabs.bedrock_image_mcp_server.models.stability_models import StylePreset
from awslabs.bedrock_image_mcp_server.server import (
    mcp_create_ellipse_mask,
    mcp_create_full_mask,
    mcp_create_rectangular_mask,
    mcp_generate_image_core,
    mcp_generate_image_sd35,
    mcp_generate_image_ultra,
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
from io import BytesIO
from PIL import Image
from unittest.mock import MagicMock, patch


SERVER_MODULE = 'awslabs.bedrock_image_mcp_server.server'


def create_test_image_base64(width=128, height=128, image_format='PNG'):
    """Create a real in-memory image and return it base64-encoded.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        image_format: PIL format name to encode with.

    Returns:
        Base64-encoded image data as an ASCII string.
    """
    img = Image.new('RGB', (width, height), color='red')
    buffer = BytesIO()
    img.save(buffer, format=image_format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


TEST_IMAGE = create_test_image_base64()
TEST_MASK = create_test_image_base64()


def success_response(paths=None):
    """Build a stand-in successful service response.

    Args:
        paths: Absolute output paths the service claims to have written.

    Returns:
        An object shaped like ImageGenerationResponse with status 'success'.
    """
    return MagicMock(
        status='success',
        paths=paths if paths is not None else ['/tmp/out.png'],
        message='Generated 1 image(s)',
    )


def error_response(message='Bedrock rejected the request'):
    """Build a stand-in failed service response.

    Args:
        message: Failure message the service reports.

    Returns:
        An object shaped like ImageGenerationResponse with status 'error'.
    """
    return MagicMock(status='error', paths=[], message=message)


# Complete kwargs for every tool under test, keyed by the registered MCP tool name.
# 'service' is the name the server module imported the underlying service function under;
# it is None for the mask tools, which touch no Bedrock service.
TOOL_KWARGS = {
    'generate_image_sd35': (
        mcp_generate_image_sd35,
        'generate_text_to_image',
        {
            'prompt': 'a lighthouse in a storm',
            'aspect_ratio': '16:9',
            'negative_prompt': 'blurry',
            'seed': 42,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'generate_image_ultra': (
        mcp_generate_image_ultra,
        'generate_image_ultra',
        {
            'prompt': 'a brass compass on a nautical chart',
            'aspect_ratio': '3:2',
            'negative_prompt': 'blurry',
            'seed': 11,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'generate_image_core': (
        mcp_generate_image_core,
        'generate_image_core',
        {
            'prompt': 'three logo concepts, flat vector',
            'aspect_ratio': '1:1',
            'negative_prompt': None,
            'seed': 12,
            'output_format': 'jpeg',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'transform_image_sd35': (
        mcp_transform_image_sd35,
        'generate_image_to_image',
        {
            'prompt': 'make it a watercolor',
            'image': TEST_IMAGE,
            'strength': 0.55,
            'negative_prompt': 'photorealistic',
            'seed': 7,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'upscale_creative': (
        mcp_upscale_creative,
        'upscale_creative',
        {
            'image': TEST_IMAGE,
            'prompt': 'restored vintage photograph',
            'creativity': 0.4,
            'negative_prompt': 'noise',
            'style_preset': 'photographic',
            'seed': 11,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'upscale_conservative': (
        mcp_upscale_conservative,
        'upscale_conservative',
        {
            'image': TEST_IMAGE,
            'prompt': 'product photograph',
            'negative_prompt': 'artifacts',
            'seed': 12,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'upscale_fast': (
        mcp_upscale_fast,
        'upscale_fast',
        {
            'image': TEST_IMAGE,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'inpaint_image': (
        mcp_inpaint,
        'inpaint',
        {
            'image': TEST_IMAGE,
            'mask': TEST_MASK,
            'prompt': 'a red sports car',
            'negative_prompt': 'trucks',
            'grow_mask': 8,
            'seed': 13,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'outpaint_image': (
        mcp_outpaint,
        'outpaint',
        {
            'image': TEST_IMAGE,
            'prompt': 'mountain peaks and sky',
            'left': 10,
            'right': 20,
            'up': 30,
            'down': 40,
            'creativity': 0.35,
            'negative_prompt': 'buildings',
            'seed': 14,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'search_and_replace': (
        mcp_search_replace,
        'search_and_replace',
        {
            'image': TEST_IMAGE,
            'search_prompt': 'wooden chair',
            'prompt': 'leather armchair',
            'negative_prompt': 'stools',
            'seed': 15,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'search_and_recolor': (
        mcp_search_recolor,
        'search_and_recolor',
        {
            'image': TEST_IMAGE,
            'select_prompt': 'car body',
            'prompt': 'bright red metallic paint',
            'negative_prompt': 'rust',
            'seed': 16,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'remove_object': (
        mcp_remove_object,
        'remove_object',
        {
            'image': TEST_IMAGE,
            'mask': TEST_MASK,
            'grow_mask': 3,
            'seed': 17,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'remove_background': (
        mcp_remove_background,
        'remove_background',
        {
            'image': TEST_IMAGE,
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'sketch_to_image': (
        mcp_sketch_to_image,
        'sketch_to_image',
        {
            'sketch': TEST_IMAGE,
            'prompt': 'detailed fantasy character',
            'control_strength': 0.65,
            'negative_prompt': 'text',
            'seed': 18,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'structure_control': (
        mcp_structure_control,
        'structure_control',
        {
            'structure_image': TEST_IMAGE,
            'prompt': 'modern architecture at sunset',
            'control_strength': 0.85,
            'negative_prompt': 'people',
            'seed': 19,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'style_guide': (
        mcp_style_guide,
        'style_guide',
        {
            'reference_image': TEST_IMAGE,
            'prompt': 'mountain landscape at sunset',
            'fidelity': 0.75,
            'negative_prompt': 'text',
            'seed': 20,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'style_transfer': (
        mcp_style_transfer,
        'style_transfer',
        {
            'init_image': TEST_IMAGE,
            'style_image': create_test_image_base64(64, 64),
            'prompt': 'impressionist painting style',
            'composition_fidelity': 0.8,
            'style_strength': 0.6,
            'change_strength': 0.45,
            'negative_prompt': 'sharp edges',
            'seed': 21,
            'output_format': 'png',
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'create_rectangular_mask': (
        mcp_create_rectangular_mask,
        None,
        {
            'width': 128,
            'height': 96,
            'x': 10,
            'y': 20,
            'mask_width': 40,
            'mask_height': 30,
            'feather': 0,
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'create_ellipse_mask': (
        mcp_create_ellipse_mask,
        None,
        {
            'width': 128,
            'height': 96,
            'center_x': 64,
            'center_y': 48,
            'radius_x': 20,
            'radius_y': 15,
            'feather': 0,
            'filename': None,
            'workspace_dir': None,
        },
    ),
    'create_full_mask': (
        mcp_create_full_mask,
        None,
        {
            'width': 64,
            'height': 32,
            'filename': None,
            'workspace_dir': None,
        },
    ),
}

SERVICE_TOOLS = [name for name, (_, service, _) in TOOL_KWARGS.items() if service]
MASK_TOOLS = [name for name, (_, service, _) in TOOL_KWARGS.items() if service is None]
FORMAT_TOOLS = [
    name
    for name, (_, service, kwargs) in TOOL_KWARGS.items()
    if service and 'output_format' in kwargs
]


def call_kwargs(tool_name, **overrides):
    """Build a complete kwargs mapping for a tool, applying any overrides.

    Args:
        tool_name: Registered MCP tool name from TOOL_KWARGS.
        **overrides: Parameter values to replace in the baseline mapping.

    Returns:
        A fresh dict of every parameter the raw tool function requires, minus ctx.
    """
    kwargs = dict(TOOL_KWARGS[tool_name][2])
    kwargs.update(overrides)
    return kwargs


async def invoke_success(tool_name, ctx, **overrides):
    """Invoke a service-backed tool with its service patched to succeed.

    Args:
        tool_name: Registered MCP tool name from TOOL_KWARGS.
        ctx: Mock MCP context.
        **overrides: Parameter values to replace in the baseline mapping.

    Returns:
        Tuple of (tool result, params model forwarded to the service, service mock).
    """
    tool, service, _ = TOOL_KWARGS[tool_name]
    with patch(f'{SERVER_MODULE}.{service}') as mock_service:
        mock_service.return_value = success_response()
        result = await tool(ctx=ctx, **call_kwargs(tool_name, **overrides))
    mock_service.assert_called_once()
    return result, mock_service.call_args[1]['params'], mock_service


class TestSharedToolContract:
    """Behaviour every tool must share, checked across the whole tool surface."""

    @pytest.mark.parametrize('tool_name', SERVICE_TOOLS)
    async def test_paths_are_returned_as_file_uris(self, tool_name, mock_context):
        """Test that service output paths are converted to file:// URIs."""
        tool, service, _ = TOOL_KWARGS[tool_name]
        with patch(f'{SERVER_MODULE}.{service}') as mock_service:
            mock_service.return_value = success_response(['/tmp/a.png', '/tmp/b.png'])
            result = await tool(ctx=mock_context, **call_kwargs(tool_name))

        assert result.status == 'success'
        assert result.paths == ['file:///tmp/a.png', 'file:///tmp/b.png']
        mock_context.error.assert_not_called()

    @pytest.mark.parametrize('tool_name', SERVICE_TOOLS)
    async def test_service_error_reports_exactly_once(self, tool_name, mock_context):
        """Test that a failed service raises and reports the failure to ctx.error once."""
        tool, service, _ = TOOL_KWARGS[tool_name]
        with patch(f'{SERVER_MODULE}.{service}') as mock_service:
            mock_service.return_value = error_response('service exploded')
            with pytest.raises(Exception, match='service exploded'):
                await tool(ctx=mock_context, **call_kwargs(tool_name))

        assert mock_context.error.call_count == 1

    @pytest.mark.parametrize('tool_name', FORMAT_TOOLS)
    async def test_uppercase_output_format_is_normalized(self, tool_name, mock_context):
        """Test that an uppercase output_format is accepted and lowercased to the enum."""
        _, params, _ = await invoke_success(tool_name, mock_context, output_format='PNG')

        # Ultra and Core use a narrower format enum, so compare the value not the member.
        assert params.output_format.value == OutputFormat.PNG.value
        mock_context.error.assert_not_called()

    @pytest.mark.parametrize('tool_name', FORMAT_TOOLS)
    async def test_invalid_output_format_reports_exactly_once(self, tool_name, mock_context):
        """Test that an unsupported output_format raises before any service call."""
        tool, service, _ = TOOL_KWARGS[tool_name]
        with patch(f'{SERVER_MODULE}.{service}') as mock_service:
            with pytest.raises(ValueError, match='Invalid output format: tiff'):
                await tool(ctx=mock_context, **call_kwargs(tool_name, output_format='tiff'))

        mock_service.assert_not_called()
        assert mock_context.error.call_count == 1

    @pytest.mark.parametrize('tool_name', SERVICE_TOOLS)
    async def test_workspace_dir_and_filename_are_forwarded(
        self, tool_name, mock_context, temp_workspace_dir
    ):
        """Test that the save location is passed through to the service, not swallowed."""
        _, _, mock_service = await invoke_success(
            tool_name,
            mock_context,
            workspace_dir=temp_workspace_dir,
            filename='chosen_name',
        )

        assert mock_service.call_args[1]['workspace_dir'] == temp_workspace_dir
        assert mock_service.call_args[1]['filename'] == 'chosen_name'


class TestSD35Tools:
    """Tests for the Stable Diffusion 3.5 tools."""

    async def test_generate_image_sd35_forwards_params(self, mock_context):
        """Test that generate_image_sd35 builds SD35TextToImageParams from its arguments."""
        _, params, _ = await invoke_success('generate_image_sd35', mock_context)

        assert params.prompt == 'a lighthouse in a storm'
        assert params.aspect_ratio is AspectRatio.RATIO_16_9
        assert params.negative_prompt == 'blurry'
        assert params.seed == 42

    async def test_generate_image_sd35_rejects_invalid_aspect_ratio(self, mock_context):
        """Test that an unsupported aspect ratio raises and reports to ctx.error once."""
        with patch(f'{SERVER_MODULE}.generate_text_to_image') as mock_service:
            with pytest.raises(ValueError, match='Invalid aspect ratio: 3:1'):
                await mcp_generate_image_sd35(
                    ctx=mock_context, **call_kwargs('generate_image_sd35', aspect_ratio='3:1')
                )

        mock_service.assert_not_called()
        assert mock_context.error.call_count == 1

    async def test_transform_image_sd35_forwards_image_and_strength(self, mock_context):
        """Test that transform_image_sd35 forwards the input image and strength knob."""
        _, params, _ = await invoke_success('transform_image_sd35', mock_context)

        assert params.image == TEST_IMAGE
        assert params.strength == 0.55
        assert params.prompt == 'make it a watercolor'
        assert params.seed == 7

    async def test_transform_image_sd35_rejects_undersized_image(self, mock_context):
        """Test that an image below the 64px minimum is rejected before the service call."""
        tiny_image = create_test_image_base64(32, 32)
        with patch(f'{SERVER_MODULE}.generate_image_to_image') as mock_service:
            with pytest.raises(Exception, match='32x32'):
                await mcp_transform_image_sd35(
                    ctx=mock_context, **call_kwargs('transform_image_sd35', image=tiny_image)
                )

        mock_service.assert_not_called()
        assert mock_context.error.call_count == 1


class TestUpscaleTools:
    """Tests for the Stability AI upscale tools."""

    async def test_upscale_creative_forwards_creativity_and_style_preset(self, mock_context):
        """Test that upscale_creative forwards creativity and converts the style preset."""
        _, params, _ = await invoke_success('upscale_creative', mock_context)

        assert params.image == TEST_IMAGE
        assert params.creativity == 0.4
        assert params.style_preset is StylePreset.PHOTOGRAPHIC
        assert params.negative_prompt == 'noise'
        assert params.seed == 11

    async def test_upscale_creative_omits_style_preset_when_not_given(self, mock_context):
        """Test that no style preset is invented when the caller passes None."""
        _, params, _ = await invoke_success('upscale_creative', mock_context, style_preset=None)

        assert params.style_preset is None

    async def test_upscale_creative_rejects_unknown_style_preset(self, mock_context):
        """Test that an unknown style preset raises and reports to ctx.error once."""
        with patch(f'{SERVER_MODULE}.upscale_creative') as mock_service:
            with pytest.raises(ValueError, match='Invalid style preset: watercolour'):
                await mcp_upscale_creative(
                    ctx=mock_context,
                    **call_kwargs('upscale_creative', style_preset='watercolour'),
                )

        mock_service.assert_not_called()
        assert mock_context.error.call_count == 1

    async def test_upscale_conservative_forwards_params(self, mock_context):
        """Test that upscale_conservative builds ConservativeUpscaleParams as given."""
        _, params, _ = await invoke_success('upscale_conservative', mock_context)

        assert params.image == TEST_IMAGE
        assert params.prompt == 'product photograph'
        assert params.negative_prompt == 'artifacts'
        assert params.seed == 12
        assert not hasattr(params, 'creativity')

    async def test_upscale_fast_forwards_only_image_and_format(self, mock_context):
        """Test that upscale_fast forwards the image and requested format."""
        _, params, _ = await invoke_success('upscale_fast', mock_context, output_format='webp')

        assert params.image == TEST_IMAGE
        assert params.output_format is OutputFormat.WEBP


class TestEditTools:
    """Tests for the Stability AI edit tools."""

    async def test_inpaint_forwards_mask_and_grow_mask(self, mock_context):
        """Test that inpaint_image forwards image, mask, prompt and grow_mask."""
        _, params, _ = await invoke_success('inpaint_image', mock_context)

        assert params.image == TEST_IMAGE
        assert params.mask == TEST_MASK
        assert params.prompt == 'a red sports car'
        assert params.negative_prompt == 'trucks'
        assert params.grow_mask == 8
        assert params.seed == 13

    async def test_outpaint_forwards_each_direction_separately(self, mock_context):
        """Test that the four outpaint directions are not transposed."""
        _, params, _ = await invoke_success('outpaint_image', mock_context)

        assert (params.left, params.right, params.up, params.down) == (10, 20, 30, 40)
        assert params.creativity == 0.35
        assert params.prompt == 'mountain peaks and sky'

    async def test_search_replace_forwards_search_prompt(self, mock_context):
        """Test that search_and_replace keeps search_prompt and prompt distinct."""
        _, params, _ = await invoke_success('search_and_replace', mock_context)

        assert params.search_prompt == 'wooden chair'
        assert params.prompt == 'leather armchair'
        assert params.negative_prompt == 'stools'
        assert params.seed == 15

    async def test_search_recolor_forwards_select_prompt(self, mock_context):
        """Test that search_and_recolor uses select_prompt, not search_prompt."""
        _, params, _ = await invoke_success('search_and_recolor', mock_context)

        assert params.select_prompt == 'car body'
        assert params.prompt == 'bright red metallic paint'
        assert not hasattr(params, 'search_prompt')

    async def test_remove_object_forwards_mask_without_prompt(self, mock_context):
        """Test that remove_object forwards the mask and grow_mask."""
        _, params, _ = await invoke_success('remove_object', mock_context)

        assert params.image == TEST_IMAGE
        assert params.mask == TEST_MASK
        assert params.grow_mask == 3
        assert params.seed == 17

    async def test_remove_background_forwards_only_image(self, mock_context):
        """Test that remove_background forwards just the image."""
        _, params, _ = await invoke_success('remove_background', mock_context)

        assert params.image == TEST_IMAGE
        assert not hasattr(params, 'output_format')


class TestControlTools:
    """Tests for the Stability AI control tools, whose image fields are easily confused."""

    async def test_sketch_to_image_maps_sketch_to_control_image(self, mock_context):
        """Test that the sketch argument lands on SketchToImageParams.control_image."""
        _, params, _ = await invoke_success('sketch_to_image', mock_context)

        assert params.control_image == TEST_IMAGE
        assert params.control_strength == 0.65
        assert params.prompt == 'detailed fantasy character'
        assert params.negative_prompt == 'text'
        assert params.seed == 18

    async def test_structure_control_maps_structure_image_to_control_image(self, mock_context):
        """Test that structure_image lands on StructureControlParams.control_image."""
        _, params, _ = await invoke_success('structure_control', mock_context)

        assert params.control_image == TEST_IMAGE
        assert params.control_strength == 0.85
        assert params.prompt == 'modern architecture at sunset'
        assert params.seed == 19

    async def test_style_guide_maps_reference_image_and_fidelity(self, mock_context):
        """Test that style_guide uses reference_image and fidelity, not control_* names."""
        _, params, _ = await invoke_success('style_guide', mock_context)

        assert params.reference_image == TEST_IMAGE
        assert params.fidelity == 0.75
        assert not hasattr(params, 'control_image')
        assert not hasattr(params, 'control_strength')

    async def test_style_transfer_maps_both_images_and_all_strengths(self, mock_context):
        """Test that style_transfer keeps init_image, style_image and its three knobs apart."""
        style_image = TOOL_KWARGS['style_transfer'][2]['style_image']
        _, params, _ = await invoke_success('style_transfer', mock_context)

        assert params.init_image == TEST_IMAGE
        assert params.style_image == style_image
        assert params.init_image != params.style_image
        assert params.composition_fidelity == 0.8
        assert params.style_strength == 0.6
        assert params.change_strength == 0.45
        assert params.seed == 21


class TestMaskTools:
    """Tests for the mask creation tools, which write real PNG files to disk."""

    @staticmethod
    def local_path(result):
        """Strip the file:// prefix from a single-path tool result.

        Args:
            result: McpImageGenerationResponse returned by a mask tool.

        Returns:
            The filesystem path the mask was written to.
        """
        assert result.status == 'success'
        assert len(result.paths) == 1
        assert result.paths[0].startswith('file://')
        return result.paths[0][len('file://') :]

    async def test_rectangular_mask_writes_expected_png(self, mock_context, temp_workspace_dir):
        """Test that create_rectangular_mask writes a valid PNG with a white rectangle."""
        result = await mcp_create_rectangular_mask(
            ctx=mock_context,
            **call_kwargs('create_rectangular_mask', workspace_dir=temp_workspace_dir),
        )

        path = self.local_path(result)
        assert os.path.exists(path)
        with Image.open(path) as mask:
            assert mask.format == 'PNG'
            assert mask.size == (128, 96)
            assert mask.mode == 'L'
            assert mask.getpixel((30, 30)) == 255
            assert mask.getpixel((0, 0)) == 0
        mock_context.error.assert_not_called()

    async def test_ellipse_mask_writes_expected_png(self, mock_context, temp_workspace_dir):
        """Test that create_ellipse_mask writes a valid PNG white at the ellipse center."""
        result = await mcp_create_ellipse_mask(
            ctx=mock_context,
            **call_kwargs('create_ellipse_mask', workspace_dir=temp_workspace_dir),
        )

        path = self.local_path(result)
        with Image.open(path) as mask:
            assert mask.format == 'PNG'
            assert mask.size == (128, 96)
            assert mask.getpixel((64, 48)) == 255
            assert mask.getpixel((0, 0)) == 0

    async def test_full_mask_writes_all_white_png(self, mock_context, temp_workspace_dir):
        """Test that create_full_mask writes a fully white PNG of the requested size."""
        result = await mcp_create_full_mask(
            ctx=mock_context, **call_kwargs('create_full_mask', workspace_dir=temp_workspace_dir)
        )

        path = self.local_path(result)
        with Image.open(path) as mask:
            assert mask.format == 'PNG'
            assert mask.size == (64, 32)
            assert mask.getextrema() == (255, 255)

    @pytest.mark.parametrize('tool_name', MASK_TOOLS)
    async def test_mask_is_written_inside_workspace_output_dir(
        self, tool_name, mock_context, temp_workspace_dir
    ):
        """Test that masks land in the workspace output directory."""
        tool = TOOL_KWARGS[tool_name][0]
        result = await tool(
            ctx=mock_context, **call_kwargs(tool_name, workspace_dir=temp_workspace_dir)
        )

        path = self.local_path(result)
        expected_dir = os.path.realpath(os.path.join(temp_workspace_dir, 'output'))
        assert os.path.dirname(path) == expected_dir
        assert path.endswith('.png')

    @pytest.mark.parametrize('tool_name', MASK_TOOLS)
    async def test_hostile_filename_stays_inside_output_dir(
        self, tool_name, mock_context, temp_workspace_dir
    ):
        """Test that a traversal attempt in filename cannot escape the output directory."""
        tool = TOOL_KWARGS[tool_name][0]
        result = await tool(
            ctx=mock_context,
            **call_kwargs(tool_name, workspace_dir=temp_workspace_dir, filename='../escape'),
        )

        path = self.local_path(result)
        expected_dir = os.path.realpath(os.path.join(temp_workspace_dir, 'output'))
        assert os.path.dirname(path) == expected_dir
        assert os.path.basename(path) == 'escape.png'
        assert os.path.exists(path)
        assert not os.path.exists(os.path.join(temp_workspace_dir, 'escape.png'))

    @pytest.mark.parametrize(
        'tool_name,overrides,expected',
        [
            ('create_rectangular_mask', {'width': 0}, 'dimensions must be positive'),
            ('create_rectangular_mask', {'mask_width': 500}, 'exceeds image bounds'),
            ('create_ellipse_mask', {'radius_x': 0}, 'radii must be positive'),
            ('create_ellipse_mask', {'center_x': 120}, 'exceeds image bounds'),
            ('create_full_mask', {'height': -1}, 'dimensions must be positive'),
        ],
    )
    async def test_invalid_mask_geometry_reports_exactly_once(
        self, tool_name, overrides, expected, mock_context, temp_workspace_dir
    ):
        """Test that invalid mask geometry raises and reports to ctx.error once."""
        tool = TOOL_KWARGS[tool_name][0]
        with pytest.raises(ValueError, match=expected):
            await tool(
                ctx=mock_context,
                **call_kwargs(tool_name, workspace_dir=temp_workspace_dir, **overrides),
            )

        assert mock_context.error.call_count == 1
        output_dir = os.path.join(temp_workspace_dir, 'output')
        assert not os.path.exists(output_dir) or os.listdir(output_dir) == []
