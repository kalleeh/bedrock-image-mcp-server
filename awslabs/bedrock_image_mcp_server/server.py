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
"""Amazon Bedrock image generation MCP Server implementation."""

import boto3
import os
import sys
import uuid
from awslabs.bedrock_image_mcp_server.consts import (
    BEDROCK_CONNECT_TIMEOUT,
    BEDROCK_MAX_POOL_CONNECTIONS,
    BEDROCK_MAX_RETRY_ATTEMPTS,
    BEDROCK_READ_TIMEOUT,
    BEDROCK_RETRY_MODE,
    DEFAULT_CFG_SCALE,
    DEFAULT_CHANGE_STRENGTH,
    DEFAULT_COMPOSITION_FIDELITY,
    DEFAULT_CONTROL_STRENGTH,
    DEFAULT_CREATIVE_UPSCALE_CREATIVITY,
    DEFAULT_GROW_MASK,
    DEFAULT_HEIGHT,
    DEFAULT_NUMBER_OF_IMAGES,
    DEFAULT_OUTPAINT_CREATIVITY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_QUALITY,
    DEFAULT_SD35_ASPECT_RATIO,
    DEFAULT_SD35_SEED,
    DEFAULT_STYLE_FIDELITY,
    DEFAULT_STYLE_STRENGTH,
    DEFAULT_WIDTH,
    PROMPT_INSTRUCTIONS,
    SD35_PROMPT_INSTRUCTIONS,
    STABILITY_SERVICES_INSTRUCTIONS,
)
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.nova_models import McpImageGenerationResponse
from awslabs.bedrock_image_mcp_server.models.sd35_models import (
    AspectRatio,
    SD35ImageToImageParams,
    SD35TextToImageParams,
)
from awslabs.bedrock_image_mcp_server.models.stability_models import (
    BackgroundRemovalParams,
    ConservativeUpscaleParams,
    CreativeUpscaleParams,
    FastUpscaleParams,
    InpaintParams,
    OutpaintParams,
    RemoveObjectParams,
    SearchRecolorParams,
    SearchReplaceParams,
    SketchToImageParams,
    StructureControlParams,
    StyleGuideParams,
    StylePreset,
    StyleTransferParams,
)
from awslabs.bedrock_image_mcp_server.models.stable_image_models import (
    StableImageOutputFormat,
    StableImageParams,
)
from awslabs.bedrock_image_mcp_server.services.bedrock_common import (
    resolve_output_path,
    sanitize_filename,
)
from awslabs.bedrock_image_mcp_server.services.nova_canvas import (
    generate_image_with_colors,
    generate_image_with_text,
)
from awslabs.bedrock_image_mcp_server.services.sd35_service import (
    generate_image_to_image,
    generate_text_to_image,
)
from awslabs.bedrock_image_mcp_server.services.stability_control import (
    sketch_to_image,
    structure_control,
    style_guide,
    style_transfer,
)
from awslabs.bedrock_image_mcp_server.services.stability_edit import (
    inpaint,
    outpaint,
    remove_background,
    remove_object,
    search_and_recolor,
    search_and_replace,
)
from awslabs.bedrock_image_mcp_server.services.stability_upscale import (
    upscale_conservative,
    upscale_creative,
    upscale_fast,
)
from awslabs.bedrock_image_mcp_server.services.stable_image_service import (
    generate_image_core,
    generate_image_ultra,
)
from awslabs.bedrock_image_mcp_server.utils.image_utils import (
    create_ellipse_mask,
    create_full_mask,
    create_rectangular_mask,
)
from botocore.config import Config
from loguru import logger
from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field
from typing import TYPE_CHECKING, List, Optional


# Logging
logger.remove()
logger.add(sys.stderr, level=os.getenv('FASTMCP_LOG_LEVEL', 'WARNING'))

# Bedrock Runtime Client typing
if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
else:
    BedrockRuntimeClient = object


# Bedrock Runtime Client with AWS best practice configuration
bedrock_runtime_client: BedrockRuntimeClient
aws_region: str = os.environ.get('AWS_REGION', 'us-east-1')

# Configure retry logic following AWS best practices
# Reference: https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html
# - Adaptive mode: AWS SDK automatically adjusts retry strategy based on throttling responses
# - Max attempts: Configurable retries (default: 3 retries = 4 total attempts)
# - Exponential backoff with jitter is handled automatically by boto3
# - Timeouts: Sufficient for image generation (30-60s) and upscaling (up to 90s)
retry_config = Config(
    retries={
        'max_attempts': BEDROCK_MAX_RETRY_ATTEMPTS,
        'mode': BEDROCK_RETRY_MODE,  # 'adaptive' mode handles exponential backoff with jitter
    },
    connect_timeout=BEDROCK_CONNECT_TIMEOUT,
    read_timeout=BEDROCK_READ_TIMEOUT,
    max_pool_connections=BEDROCK_MAX_POOL_CONNECTIONS,
)

try:
    if aws_profile := os.environ.get('AWS_PROFILE'):
        bedrock_runtime_client = boto3.Session(
            profile_name=aws_profile, region_name=aws_region
        ).client('bedrock-runtime', config=retry_config)
        logger.bind(
            region=aws_region,
            profile=aws_profile,
        ).info(f'Bedrock client initialized with AWS profile: {aws_profile}')
    else:
        bedrock_runtime_client = boto3.Session(region_name=aws_region).client(
            'bedrock-runtime', config=retry_config
        )
        logger.bind(region=aws_region).info('Bedrock client initialized with default credentials')
except Exception as e:
    logger.bind(region=aws_region).error(f'Error creating bedrock runtime client: {str(e)}')
    raise


# Create the MCP server with detailed instructions
mcp = FastMCP(
    'bedrock-image-mcp-server',
    instructions=f"""
# Amazon Bedrock Image Generation

This MCP server provides tools for generating images using Amazon Nova Canvas, Stable Diffusion 3.5 Large, and Stability AI Image Services through Amazon Bedrock.

## Choosing a text-to-image tool

Three Stability AI models cover different jobs. Pick by what the image is for, not by
whichever is newest:

- **generate_image_ultra** — highest fidelity: typography, intricate composition, dynamic
  lighting, photorealism. Use for **commercial and advertising creative, marketing assets,
  hero and product imagery, print** — anything client-facing where the image is the
  deliverable. Costs more per image.
- **generate_image_sd35** — strong all-rounder with the widest style range (3D, photography,
  painting, line art) and excellent text rendering. Use for **concept art, visual effects,
  product renders, billboards and print ads**, or when a specific non-photographic style
  matters. Good default.
- **generate_image_core** — fastest and cheapest (enhanced SDXL). Use for **drafts, exploring
  several concepts, thumbnails and bulk work**. Not for client-facing deliverables.

If the user asks for professional, commercial or marketing imagery, reach for
generate_image_ultra — or generate_image_sd35 when they also need a particular style or a lot
of legible text. All three accept prompts up to 10,000 characters and beat Nova Canvas on
prompt adherence.
Use the Nova Canvas tools only for something they cannot do: explicit pixel width/height, a
color palette, Nova style presets, or several images in one request.

Note that AWS has marked Nova Canvas as a Legacy model with end-of-life on 2026-09-30, and
accounts can lose access to it after 15 days of inactivity. Treat generate_image_sd35 as the
migration path.

Region availability differs per model family, and no region has all of them. SD3.5, Ultra
and Core are us-west-2 only; the Stability AI tools are in us-east-1, us-east-2 and us-west-2; Nova
Canvas is in us-east-1, eu-west-1 and ap-northeast-1. If a tool reports an invalid model
identifier, the model is not available in the configured AWS_REGION.

## Available Tools

### Stability AI Text-to-Image Tools (preferred)
- **generate_image_ultra**: Highest quality text-to-image (Stable Image Ultra). Best photorealism and text rendering.
- **generate_image_sd35**: Balanced text-to-image (Stable Diffusion 3.5 Large). Good general default.
- **generate_image_core**: Fastest and cheapest text-to-image (Stable Image Core). Best for drafts.
- **transform_image_sd35**: Transform an existing image using SD3.5 with text guidance. The only image-to-image option of the four.

### Amazon Nova Canvas Tools (DEPRECATED — retires 2026-09-30)
- **generate_image**: Generate an image from a text prompt using Amazon Nova Canvas. Deprecated; prefer generate_image_sd35 unless you need explicit pixel dimensions, quality/cfg_scale tuning, Nova style presets, or several images per request.
- **generate_image_with_colors**: Generate an image from a text prompt and color palette using Amazon Nova Canvas. Deprecated; no direct replacement, so describe the colours in a generate_image_sd35 prompt instead.

### Stability AI Upscale Tools
- **upscale_creative**: Upscale images to 4K with creative AI enhancement (20-40x upscale).
- **upscale_conservative**: Upscale images to 4K while preserving original details.
- **upscale_fast**: Fast 4x upscaling without creative enhancement.

### Stability AI Edit Tools
- **inpaint_image**: Fill masked regions with AI-generated content (generative fill).
- **outpaint_image**: Extend images beyond their original boundaries.
- **search_and_replace**: Find and replace objects using text prompts.
- **search_and_recolor**: Recolor specific objects using text prompts.
- **remove_object**: Remove unwanted objects from images.
- **remove_background**: Automatically remove backgrounds from images.

### Stability AI Control Tools
- **sketch_to_image**: Convert sketches/line art into detailed images.
- **structure_control**: Generate images following structural guides/edge maps.
- **style_guide**: Generate images matching a reference style.
- **style_transfer**: Transfer style from one image to another with fine-grained control.

### Mask Creation Tools
- **create_rectangular_mask**: Create a rectangular mask for inpainting or object removal.
- **create_ellipse_mask**: Create an elliptical mask for inpainting or object removal.
- **create_full_mask**: Create a full white mask covering the entire image.

## Nova Canvas Prompt Best Practices

{PROMPT_INSTRUCTIONS}

## Stable Diffusion 3.5 Large Prompt Best Practices

{SD35_PROMPT_INSTRUCTIONS}

## Stability AI Services Best Practices

{STABILITY_SERVICES_INSTRUCTIONS}
""",
    dependencies=[
        'pydantic',
        'boto3',
    ],
)


class ImageServiceError(Exception):
    """Raised when a Bedrock image service reports a failed operation."""


def parse_output_format(output_format: str) -> OutputFormat:
    """Convert a caller-supplied format string to an OutputFormat.

    Args:
        output_format: Format name, case-insensitive (jpeg, png, or webp).

    Returns:
        The matching OutputFormat member.

    Raises:
        ValueError: If the format is not one of jpeg, png, or webp.
    """
    try:
        return OutputFormat(output_format.lower())
    except ValueError:
        raise ValueError(
            f'Invalid output format: {output_format}. Must be one of: jpeg, png, webp'
        )


def parse_stable_image_format(output_format: str) -> StableImageOutputFormat:
    """Convert a caller-supplied format string to a StableImageOutputFormat.

    Stable Image Ultra and Core reject webp, so they accept a narrower set than the
    other Stability AI tools.

    Args:
        output_format: Format name, case-insensitive (png or jpeg).

    Returns:
        The matching StableImageOutputFormat member.

    Raises:
        ValueError: If the format is not png or jpeg.
    """
    try:
        return StableImageOutputFormat(output_format.lower())
    except ValueError:
        raise ValueError(
            f'Invalid output format: {output_format}. Must be one of: png, jpeg. '
            f'Stable Image Ultra and Core do not support webp.'
        )


def parse_aspect_ratio(aspect_ratio: str) -> AspectRatio:
    """Convert a caller-supplied aspect ratio string to an AspectRatio.

    Args:
        aspect_ratio: Ratio name such as '16:9'.

    Returns:
        The matching AspectRatio member.

    Raises:
        ValueError: If the ratio is not one of the supported values.
    """
    try:
        return AspectRatio(aspect_ratio)
    except ValueError:
        supported = ', '.join(ratio.value for ratio in AspectRatio)
        raise ValueError(f'Invalid aspect ratio: {aspect_ratio}. Must be one of: {supported}')


def write_mask(mask_bytes: bytes, filename: Optional[str], workspace_dir: Optional[str]) -> str:
    """Write mask bytes to the workspace output directory.

    Args:
        mask_bytes: PNG-encoded mask data.
        filename: Optional caller-supplied name (without extension).
        workspace_dir: Directory to write into. If None, uses the current directory.

    Returns:
        Absolute path to the written mask file.
    """
    output_dir = (
        os.path.join(workspace_dir, DEFAULT_OUTPUT_DIR) if workspace_dir else DEFAULT_OUTPUT_DIR
    )
    os.makedirs(output_dir, exist_ok=True)
    safe_name = sanitize_filename(filename or f'mask_{uuid.uuid4().hex[:8]}', 'mask')
    mask_path = resolve_output_path(output_dir, f'{safe_name}.png')
    with open(mask_path, 'wb') as f:
        f.write(mask_bytes)
    return mask_path


@mcp.tool(name='generate_image')
async def mcp_generate_image(
    ctx: Context,
    prompt: str = Field(
        description='The text description of the image to generate (1-1024 characters)'
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Text to define what not to include in the image (1-1024 characters)',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    width: int = Field(
        default=DEFAULT_WIDTH,
        description='The width of the generated image (320-4096, divisible by 16)',
    ),
    height: int = Field(
        default=DEFAULT_HEIGHT,
        description='The height of the generated image (320-4096, divisible by 16)',
    ),
    quality: str = Field(
        default=DEFAULT_QUALITY,
        description='The quality of the generated image ("standard" or "premium")',
    ),
    cfg_scale: float = Field(
        default=DEFAULT_CFG_SCALE,
        description='How strongly the image adheres to the prompt (1.1-10.0)',
    ),
    seed: Optional[int] = Field(default=None, description='Seed for generation (0-858,993,459)'),
    number_of_images: int = Field(
        default=DEFAULT_NUMBER_OF_IMAGES,
        description='The number of images to generate (1-5)',
    ),
    style: Optional[str] = Field(
        default=None,
        description='Visual style preset: "3D_ANIMATED_FAMILY_FILM", "DESIGN_SKETCH", "FLAT_VECTOR_ILLUSTRATION", "GRAPHIC_NOVEL_ILLUSTRATION", "MAXIMALISM", "MIDCENTURY_RETRO", "PHOTOREALISM", "SOFT_DIGITAL_PAINTING"',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """DEPRECATED. Generate an image using Amazon Nova Canvas with text prompt.

    This tool uses Amazon Nova Canvas to generate images based on a text prompt.
    The generated image will be saved to a file and the path will be returned.

    DEPRECATED: AWS marked Nova Canvas as a Legacy model and retires it on 2026-09-30, after
    which this tool will stop working. AWS also revokes access after 15 days of inactivity.
    Use generate_image_sd35 (Stable Diffusion 3.5 Large, in us-west-2) instead, which also
    gives noticeably better prompt adherence and image quality. Only use this tool when you
    specifically need a Nova feature that SD3.5 lacks: explicit width/height in pixels,
    quality/cfg_scale tuning, Nova style presets, or several images in one request.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!
    The workspace_dir parameter should be set to the directory where the user is currently working
    so that images are saved to a location accessible to the user.

    ## Prompt Best Practices

    An effective prompt often includes short descriptions of:
    1. The subject
    2. The environment
    3. (optional) The position or pose of the subject
    4. (optional) Lighting description
    5. (optional) Camera position/framing
    6. (optional) The visual style or medium ("photo", "illustration", "painting", etc.)

    Do not use negation words like "no", "not", "without" in your prompt. Instead, use the
    negative_prompt parameter to specify what you don't want in the image.

    You should always include "people, anatomy, hands, low quality, low resolution, low detail" in your negative_prompt

    ## Example Prompts

    - "realistic editorial photo of female teacher standing at a blackboard with a warm smile"
    - "whimsical and ethereal soft-shaded story illustration: A woman in a large hat stands at the ship's railing looking out across the ocean"
    - "drone view of a dark river winding through a stark Iceland landscape, cinematic quality"

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug(
        f"MCP tool generate_image called with prompt: '{prompt[:30]}...', dims: {width}x{height}"
    )

    try:
        logger.info(
            f'Generating image with text prompt, quality: {quality}, cfg_scale: {cfg_scale}'
        )
        response = await generate_image_with_text(
            prompt=prompt,
            bedrock_runtime_client=bedrock_runtime_client,
            negative_prompt=negative_prompt,
            filename=filename,
            width=width,
            height=height,
            quality=quality,
            cfg_scale=cfg_scale,
            seed=seed,
            number_of_images=number_of_images,
            workspace_dir=workspace_dir,
            style=style,
        )

        if response.status == 'success':
            # return response.paths
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to generate image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_generate_image: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='generate_image_with_colors')
async def mcp_generate_image_with_colors(
    ctx: Context,
    prompt: str = Field(
        description='The text description of the image to generate (1-1024 characters)'
    ),
    colors: List[str] = Field(
        description='List of up to 10 hexadecimal color values (e.g., "#FF9800")'
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Text to define what not to include in the image (1-1024 characters)',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    width: int = Field(
        default=1024,
        description='The width of the generated image (320-4096, divisible by 16)',
    ),
    height: int = Field(
        default=1024,
        description='The height of the generated image (320-4096, divisible by 16)',
    ),
    quality: str = Field(
        default='standard',
        description='The quality of the generated image ("standard" or "premium")',
    ),
    cfg_scale: float = Field(
        default=6.5,
        description='How strongly the image adheres to the prompt (1.1-10.0)',
    ),
    seed: Optional[int] = Field(default=None, description='Seed for generation (0-858,993,459)'),
    number_of_images: int = Field(default=1, description='The number of images to generate (1-5)'),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="The current workspace directory where the image should be saved. CRITICAL: Assistant must always provide this parameter to save images to the user's current project.",
    ),
) -> McpImageGenerationResponse:
    """DEPRECATED. Generate an image using Amazon Nova Canvas with color guidance.

    This tool uses Amazon Nova Canvas to generate images based on a text prompt and color palette.
    The generated image will be saved to a file and the path will be returned.

    DEPRECATED: AWS marked Nova Canvas as a Legacy model and retires it on 2026-09-30, after
    which this tool will stop working. There is no direct replacement for colour-palette
    guidance; describe the desired colours in the prompt to generate_image_sd35 instead.

    IMPORTANT FOR Assistant: Always send the current workspace directory when calling this tool!
    The workspace_dir parameter should be set to the directory where the user is currently working
    so that images are saved to a location accessible to the user.

    ## Prompt Best Practices

    An effective prompt often includes short descriptions of:
    1. The subject
    2. The environment
    3. (optional) The position or pose of the subject
    4. (optional) Lighting description
    5. (optional) Camera position/framing
    6. (optional) The visual style or medium ("photo", "illustration", "painting", etc.)

    Do not use negation words like "no", "not", "without" in your prompt. Instead, use the
    negative_prompt parameter to specify what you don't want in the image.

    ## Example Colors

    - ["#FF5733", "#33FF57", "#3357FF"] - A vibrant color scheme with red, green, and blue
    - ["#000000", "#FFFFFF"] - A high contrast black and white scheme
    - ["#FFD700", "#B87333"] - A gold and bronze color scheme

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug(
        f"MCP tool generate_image_with_colors called with prompt: '{prompt[:30]}...', {len(colors)} colors"
    )

    try:
        color_hex_list = ', '.join(colors[:3]) + (', ...' if len(colors) > 3 else '')
        logger.info(
            f'Generating color-guided image with colors: [{color_hex_list}], quality: {quality}'
        )

        response = await generate_image_with_colors(
            prompt=prompt,
            colors=colors,
            bedrock_runtime_client=bedrock_runtime_client,
            negative_prompt=negative_prompt,
            filename=filename,
            width=width,
            height=height,
            quality=quality,
            cfg_scale=cfg_scale,
            seed=seed,
            number_of_images=number_of_images,
            workspace_dir=workspace_dir,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to generate color-guided image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_generate_image_with_colors: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='generate_image_sd35')
async def mcp_generate_image_sd35(
    ctx: Context,
    prompt: str = Field(
        description='The text description of the image to generate (1-10,000 characters)'
    ),
    aspect_ratio: str = Field(
        default=DEFAULT_SD35_ASPECT_RATIO,
        description='Aspect ratio for the image: "16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Text to define what not to include in the image (1-10,000 characters)',
    ),
    seed: int = Field(
        default=DEFAULT_SD35_SEED,
        description='Seed for reproducible generation (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default='png',
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Generate an image from a text prompt. BALANCED quality and cost.

    This tool uses Stable Diffusion 3.5 Large, a strong all-rounder and a good general default.
    AWS positions it for concept art, visual effects and detailed product imagery across media,
    gaming, advertising and retail. It renders a wide range of styles (3D, photography, painting,
    line art), handles long complex prompts up to 10,000 characters, and has notably good text
    quality: fewer errors in spelling, kerning, letter forming and spacing.

    Best for: concept art, product renders, billboards and print ads, and any work needing a
    specific non-photographic style or a lot of legible text.

    ## Choosing between the text-to-image tools

    - generate_image_ultra: highest fidelity. AWS positions it for typography, intricate
      compositions, dynamic lighting and photorealism. Use for commercial and advertising
      assets, hero images, print and anything where the image is the deliverable.
    - generate_image_sd35: strong all-rounder with the widest style range (3D, photography,
      painting, line art) and excellent text rendering. AWS positions it for concept art,
      visual effects, product imagery, billboards and print ads. Good default.
    - generate_image_core: fastest and cheapest, built on an enhanced SDXL. Use for drafts,
      exploring several concepts, and bulk or throwaway work.

    If the user asks for professional, commercial, marketing or client-facing imagery, use
    generate_image_ultra, or generate_image_sd35 when they also need a specific non-photographic
    style or a lot of legible text. Use generate_image_core only when speed or cost matters more
    than fidelity.

    Reach for generate_image (Nova Canvas) only when you need explicit pixel dimensions,
    a color palette, Nova style presets, or several images in one request.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!
    The workspace_dir parameter should be set to the directory where the user is currently working
    so that images are saved to a location accessible to the user.

    ## Key Features

    - Supports prompts up to 10,000 characters (vs 1,024 for Nova Canvas)
    - Better prompt adherence and detail preservation
    - Aspect ratio selection instead of explicit dimensions
    - Reproducible results with seed values

    ## Prompt Best Practices

    - Be specific and descriptive with your prompts
    - Use aspect ratios appropriate for your use case (16:9 for landscapes, 9:16 for portraits, etc.)
    - Leverage negative prompts to exclude unwanted elements
    - Use seeds for reproducible results
    - SD3.5 can handle longer, more detailed prompts effectively

    ## Example Prompts

    - "A serene Japanese garden at dawn, with a wooden bridge over a koi pond, cherry blossoms in full bloom, soft morning mist, traditional stone lanterns, and Mount Fuji visible in the distance, photorealistic style"
    - "Portrait of a wise elderly wizard with a long silver beard, wearing deep blue robes embroidered with golden stars, holding an ancient wooden staff, warm candlelight illuminating his face, fantasy art style"

    ## Aspect Ratios

    - 16:9 - Widescreen landscape
    - 1:1 - Square
    - 21:9 - Ultra-wide
    - 2:3 - Portrait
    - 3:2 - Landscape
    - 4:5 - Portrait
    - 5:4 - Landscape
    - 9:16 - Vertical/mobile
    - 9:21 - Ultra-tall

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug(
        f"MCP tool generate_image_sd35 called with prompt: '{prompt[:30]}...', aspect_ratio: {aspect_ratio}"
    )

    try:
        aspect_ratio_enum = parse_aspect_ratio(aspect_ratio)
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = SD35TextToImageParams(
            prompt=prompt,
            aspect_ratio=aspect_ratio_enum,
            seed=seed,
            negative_prompt=negative_prompt,
            output_format=output_format_enum,
        )

        logger.info(f'Generating SD3.5 image with aspect ratio: {aspect_ratio}, seed: {seed}')

        response = await generate_text_to_image(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to generate SD3.5 image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_generate_image_sd35: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='generate_image_ultra')
async def mcp_generate_image_ultra(
    ctx: Context,
    prompt: str = Field(
        description='The text description of the image to generate (1-10,000 characters)'
    ),
    aspect_ratio: str = Field(
        default=DEFAULT_SD35_ASPECT_RATIO,
        description='Aspect ratio: 16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, or 9:21',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the generated image (optional, up to 10,000 characters)',
    ),
    seed: int = Field(
        default=DEFAULT_SD35_SEED,
        description='Random seed for reproducibility (0-4,294,967,294). Use 0 for random.',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: png or jpeg. This model does not support webp.',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
) -> McpImageGenerationResponse:
    """Generate an image from a text prompt. HIGHEST QUALITY option.

    This tool uses Stable Image Ultra, Stability AI's flagship text-to-image model. It draws on
    their top models including SD3.5, and AWS describes it as excelling at typography, intricate
    compositions, dynamic lighting, vibrant colours and artistic cohesion, producing photorealism
    with exceptional detail. It costs more per image and is slightly slower than the others.

    Best for: commercial and advertising creative, marketing campaign assets, hero and product
    imagery, print, and any final deliverable where image quality is the point.

    ## Choosing between the text-to-image tools

    - generate_image_ultra: highest fidelity. AWS positions it for typography, intricate
      compositions, dynamic lighting and photorealism. Use for commercial and advertising
      assets, hero images, print and anything where the image is the deliverable.
    - generate_image_sd35: strong all-rounder with the widest style range (3D, photography,
      painting, line art) and excellent text rendering. AWS positions it for concept art,
      visual effects, product imagery, billboards and print ads. Good default.
    - generate_image_core: fastest and cheapest, built on an enhanced SDXL. Use for drafts,
      exploring several concepts, and bulk or throwaway work.

    If the user asks for professional, commercial, marketing or client-facing imagery, use
    generate_image_ultra, or generate_image_sd35 when they also need a specific non-photographic
    style or a lot of legible text. Use generate_image_core only when speed or cost matters more
    than fidelity.

    ## Requirements and limits

    - Available in us-west-2 only
    - Text-to-image only; it cannot transform an existing image. Use transform_image_sd35
      for image-to-image work.
    - output_format must be png or jpeg. webp is rejected by the model.
    - Returns a single image per call. There is no width/height, cfg_scale or style preset;
      use aspect_ratio to control the shape.

    ## Example Usage

    - Prompt: "a weathered brass compass on a nautical chart, macro photograph, soft window light"
      Aspect Ratio: 3:2
      (Produces a high-detail final asset)

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug(f"MCP tool generate_image_ultra called with prompt: '{prompt[:30]}...'")

    try:
        params = StableImageParams(
            prompt=prompt,
            aspect_ratio=parse_aspect_ratio(aspect_ratio),
            seed=seed,
            negative_prompt=negative_prompt,
            output_format=parse_stable_image_format(output_format),
        )

        logger.info(f'Generating Stable Image Ultra image with aspect ratio: {aspect_ratio}')

        response = await generate_image_ultra(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to generate Ultra image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_generate_image_ultra: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='generate_image_core')
async def mcp_generate_image_core(
    ctx: Context,
    prompt: str = Field(
        description='The text description of the image to generate (1-10,000 characters)'
    ),
    aspect_ratio: str = Field(
        default=DEFAULT_SD35_ASPECT_RATIO,
        description='Aspect ratio: 16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, or 9:21',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the generated image (optional, up to 10,000 characters)',
    ),
    seed: int = Field(
        default=DEFAULT_SD35_SEED,
        description='Random seed for reproducibility (0-4,294,967,294). Use 0 for random.',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: png or jpeg. This model does not support webp.',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
) -> McpImageGenerationResponse:
    """Generate an image from a text prompt. FASTEST and cheapest option.

    This tool uses Stable Image Core, Stability AI's fast tier, built on an enhanced SDXL. AWS
    describes it as delivering exceptional speed and efficiency at consistent quality. It is the
    quickest and least expensive option here, at lower fidelity than SD3.5 or Ultra.

    Best for: drafts, exploring several concepts quickly, thumbnails, and bulk or throwaway
    work. Do not use it for client-facing or commercial deliverables; use generate_image_ultra
    for those.

    ## Choosing between the text-to-image tools

    - generate_image_ultra: highest fidelity. AWS positions it for typography, intricate
      compositions, dynamic lighting and photorealism. Use for commercial and advertising
      assets, hero images, print and anything where the image is the deliverable.
    - generate_image_sd35: strong all-rounder with the widest style range (3D, photography,
      painting, line art) and excellent text rendering. AWS positions it for concept art,
      visual effects, product imagery, billboards and print ads. Good default.
    - generate_image_core: fastest and cheapest, built on an enhanced SDXL. Use for drafts,
      exploring several concepts, and bulk or throwaway work.

    If the user asks for professional, commercial, marketing or client-facing imagery, use
    generate_image_ultra, or generate_image_sd35 when they also need a specific non-photographic
    style or a lot of legible text. Use generate_image_core only when speed or cost matters more
    than fidelity.

    ## Requirements and limits

    - Available in us-west-2 only
    - Text-to-image only; it cannot transform an existing image. Use transform_image_sd35
      for image-to-image work.
    - output_format must be png or jpeg. webp is rejected by the model.
    - Returns a single image per call. There is no width/height, cfg_scale or style preset;
      use aspect_ratio to control the shape.

    ## Example Usage

    - Prompt: "three thumbnail concepts for a coffee shop logo, flat vector"
      Aspect Ratio: 1:1
      (Produces a quick draft to iterate on)

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug(f"MCP tool generate_image_core called with prompt: '{prompt[:30]}...'")

    try:
        params = StableImageParams(
            prompt=prompt,
            aspect_ratio=parse_aspect_ratio(aspect_ratio),
            seed=seed,
            negative_prompt=negative_prompt,
            output_format=parse_stable_image_format(output_format),
        )

        logger.info(f'Generating Stable Image Core image with aspect ratio: {aspect_ratio}')

        response = await generate_image_core(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to generate Core image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_generate_image_core: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='transform_image_sd35')
async def mcp_transform_image_sd35(
    ctx: Context,
    prompt: str = Field(
        description='The text description guiding the image transformation (1-10,000 characters)'
    ),
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    strength: float = Field(
        default=0.7,
        description='Transformation intensity: 0.0 (preserve input) to 1.0 (ignore input). Start with 0.7 for balanced results.',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Text to define what not to include in the transformed image (1-10,000 characters)',
    ),
    seed: int = Field(
        default=DEFAULT_SD35_SEED,
        description='Seed for reproducible generation (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default='png',
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the transformed image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Transform an existing image using Stable Diffusion 3.5 Large with text guidance.

    This tool uses SD3.5 to transform existing images based on text prompts. The strength
    parameter controls how much the output differs from the input image.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!
    The workspace_dir parameter should be set to the directory where the user is currently working
    so that images are saved to a location accessible to the user.

    ## CRITICAL: When to Use This Tool vs Others

    **USE THIS TOOL FOR:**
    - Changing overall style (e.g., "make it look like a watercolor painting")
    - Applying artistic effects to the entire image
    - Changing lighting, mood, or atmosphere across the whole image
    - Converting to different art styles (photorealistic to cartoon, etc.)

    **DO NOT USE THIS TOOL FOR:**
    - Changing ONLY the background → Use search_and_replace instead
    - Replacing specific objects → Use search_and_replace instead
    - Changing colors of specific objects → Use search_and_recolor instead
    - Removing objects → Use remove_object instead
    - Extending image boundaries → Use outpaint_image instead

    This tool transforms the ENTIRE image based on your prompt. It cannot selectively
    edit only parts of the image (like just the background). For selective edits,
    use the Stability AI editing tools (search_and_replace, search_and_recolor, etc.).

    ## Key Features

    - Transform existing images with text guidance
    - Control transformation intensity with strength parameter
    - Supports file paths or base64-encoded images
    - Minimum input image size: 64px per side

    ## Strength Parameter Guide

    - 0.0-0.3: Very subtle changes, mostly preserves input
    - 0.3-0.5: Moderate changes, clear influence from input
    - 0.5-0.7: Balanced transformation (recommended starting point)
    - 0.7-0.9: Strong transformation, input provides structure
    - 0.9-1.0: Dramatic reimagining, minimal input influence

    ## Prompt Best Practices

    - Describe the desired transformation clearly
    - Use negative prompts to exclude unwanted elements
    - Start with strength 0.7 and adjust based on results
    - Use consistent seeds to iterate on results

    ## Example Transformations

    - Prompt: "Transform into a watercolor painting style, soft pastel colors, artistic brush strokes"
      Strength: 0.6
      Negative: "photorealistic, sharp edges, digital art"

    - Prompt: "Convert to a dramatic black and white photograph with high contrast"
      Strength: 0.5
      Negative: "color, washed out, low contrast"

    Returns:
        McpImageGenerationResponse: A response containing the transformed image paths.
    """
    logger.debug(
        f"MCP tool transform_image_sd35 called with prompt: '{prompt[:30]}...', strength: {strength}"
    )

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = SD35ImageToImageParams(
            prompt=prompt,
            image=image,
            strength=strength,
            seed=seed,
            negative_prompt=negative_prompt,
            output_format=output_format_enum,
        )

        logger.info(f'Transforming image with SD3.5, strength: {strength}, seed: {seed}')

        response = await generate_image_to_image(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to transform image with SD3.5: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_transform_image_sd35: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='upscale_creative')
async def mcp_upscale_creative(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    prompt: str = Field(
        description='Descriptive prompt to guide upscaling style (1-10,000 characters)'
    ),
    creativity: float = Field(
        default=DEFAULT_CREATIVE_UPSCALE_CREATIVITY,
        description='Enhancement level: 0.1 (subtle) to 0.5 (very creative). Higher values add more AI-generated details.',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the upscaled image (1-10,000 characters)',
    ),
    style_preset: Optional[str] = Field(
        default=None,
        description='Optional style preset: "3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance", "fantasy-art", "photographic", etc.',
    ),
    seed: int = Field(
        default=0,
        description='Seed for reproducible generation (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the upscaled image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Upscale images to 4K with creative AI enhancement.

    This tool uses Stability AI's creative upscaling to enhance and upscale images
    to 4K resolution (20-40x upscale). Best for low-resolution or degraded images
    that need improvement and enhancement.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!
    The workspace_dir parameter should be set to the directory where the user is currently working
    so that images are saved to a location accessible to the user.

    ## Key Features

    - Upscales images to 4K resolution (20-40x)
    - AI enhancement adds details and improves quality
    - Creativity parameter controls enhancement level
    - Optional style presets for specific aesthetics
    - Input: 64x64 to 1 megapixel (1024x1024)

    IMPORTANT: use output_format="jpeg" for this tool. The result is around 3150x3150, which
    exceeds the Bedrock response size limit as a PNG and fails regardless of input size.

    ## Creativity Parameter Guide

    - 0.1-0.2: Subtle enhancement, mostly preserves original
    - 0.3: Balanced enhancement (default, recommended)
    - 0.4-0.5: Strong enhancement, adds significant details

    ## When to Use

    - Low-resolution images that need quality improvement
    - Degraded or compressed images
    - Images that benefit from AI enhancement
    - When you want to add artistic details during upscaling

    ## Style Presets

    Available presets: 3d-model, analog-film, anime, cinematic, comic-book, digital-art,
    enhance, fantasy-art, isometric, line-art, low-poly, neon-punk, origami, photographic,
    pixel-art, tile-texture

    ## Example Usage

    - Image: "old_photo.jpg"
      Prompt: "vintage photograph, restored quality, clear details"
      Creativity: 0.3
      Style: "photographic"

    Returns:
        McpImageGenerationResponse: A response containing the upscaled image paths.
    """
    logger.debug(
        f'MCP tool upscale_creative called with creativity: {creativity}, style: {style_preset}'
    )

    try:
        output_format_enum = parse_output_format(output_format)

        # Validate and convert style preset if provided
        style_preset_enum = None
        if style_preset:
            try:
                style_preset_enum = StylePreset(style_preset)
            except ValueError:
                raise ValueError(f'Invalid style preset: {style_preset}')

        # Create parameters model
        params = CreativeUpscaleParams(
            image=image,
            prompt=prompt,
            creativity=creativity,
            negative_prompt=negative_prompt,
            seed=seed,
            style_preset=style_preset_enum,
            output_format=output_format_enum,
        )

        logger.info(f'Creative upscaling image with creativity: {creativity}, seed: {seed}')

        response = await upscale_creative(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to upscale image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_upscale_creative: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='upscale_conservative')
async def mcp_upscale_conservative(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    prompt: str = Field(description='Descriptive prompt for context (1-10,000 characters)'),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the upscaled image (1-10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Seed for reproducible generation (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the upscaled image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Upscale images to 4K while preserving original details.

    This tool uses Stability AI's conservative upscaling to increase resolution
    to 4K while preserving the original image characteristics with minimal
    alterations. Best for images that already have good quality but need
    higher resolution.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!
    The workspace_dir parameter should be set to the directory where the user is currently working
    so that images are saved to a location accessible to the user.

    ## Key Features

    - Upscales images to 4K resolution
    - Preserves original image characteristics
    - Minimal AI alterations
    - Input: 64x64 to 9.4 megapixels

    ## When to Use

    - High-quality images that just need more resolution
    - When you want to preserve the original look
    - Professional photos that need print resolution
    - When creative enhancement is not desired

    ## Example Usage

    - Image: "product_photo.jpg"
      Prompt: "professional product photograph, high quality"

    - Image: "portrait.png"
      Prompt: "portrait photograph, natural lighting"

    Returns:
        McpImageGenerationResponse: A response containing the upscaled image paths.
    """
    logger.debug('MCP tool upscale_conservative called')

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = ConservativeUpscaleParams(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=output_format_enum,
        )

        logger.info(f'Conservative upscaling image, seed: {seed}')

        response = await upscale_conservative(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to upscale image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_upscale_conservative: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='upscale_fast')
async def mcp_upscale_fast(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the upscaled image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Fast 4x upscaling without creative enhancement.

    This tool provides quick 4x resolution increase without AI enhancement.
    Best for images that need simple resolution increase without style changes.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!
    The workspace_dir parameter should be set to the directory where the user is currently working
    so that images are saved to a location accessible to the user.

    ## Key Features

    - Fast 4x upscaling
    - No AI enhancement or style changes
    - Simple and straightforward
    - Input: 1024 to 1 megapixel (1024x1024)

    ## When to Use

    - When you need quick upscaling without enhancement
    - When preserving exact original appearance is critical
    - For technical images, diagrams, or screenshots
    - When speed is more important than quality enhancement

    ## Example Usage

    - Image: "screenshot.png"
      (No prompt needed for fast upscale)

    - Image: "diagram.jpg"
      (Simple 4x resolution increase)

    Returns:
        McpImageGenerationResponse: A response containing the upscaled image paths.
    """
    logger.debug('MCP tool upscale_fast called')

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = FastUpscaleParams(
            image=image,
            output_format=output_format_enum,
        )

        logger.info('Fast upscaling image 4x')

        response = await upscale_fast(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to upscale image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_upscale_fast: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='inpaint_image')
async def mcp_inpaint(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    mask: str = Field(
        description='Path to the mask image file or base64-encoded mask data. White areas are filled, black areas are preserved.'
    ),
    prompt: str = Field(description='Description of desired fill content (1-10,000 characters)'),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the generated content (1-10,000 characters)',
    ),
    grow_mask: int = Field(
        default=DEFAULT_GROW_MASK,
        description='Pixels to expand mask edges (0-20). Helps blend edges.',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the inpainted image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Fill masked regions with AI-generated content (generative fill).

    This tool fills masked regions of images with AI-generated content that
    blends naturally with the surrounding image. The mask defines which areas
    to fill (white) and which to preserve (black).

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Fill masked areas with AI-generated content
    - Natural blending with surrounding image
    - Mask-based control (white=fill, black=preserve)
    - grow_mask parameter expands mask edges for better blending

    ## When to Use

    - Adding objects to images
    - Replacing specific regions
    - Filling in missing or damaged areas
    - Creative image composition

    ## Example Usage

    - Image: "photo.jpg"
      Mask: "mask.png"
      Prompt: "a red sports car"
      (Fills masked area with a red sports car)

    Returns:
        McpImageGenerationResponse: A response containing the inpainted image paths.
    """
    logger.debug(f"MCP tool inpaint_image called with prompt: '{prompt[:30]}...'")

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = InpaintParams(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            grow_mask=grow_mask,
            seed=seed,
            output_format=output_format_enum,
        )

        logger.info(f'Inpainting image with prompt: {prompt[:50]}...')

        response = await inpaint(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to inpaint image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_inpaint: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='outpaint_image')
async def mcp_outpaint(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    prompt: str = Field(
        description='Description of desired extended content (1-10,000 characters)'
    ),
    left: int = Field(
        default=0,
        description='Pixels to extend left (0-2000)',
    ),
    right: int = Field(
        default=0,
        description='Pixels to extend right (0-2000)',
    ),
    up: int = Field(
        default=0,
        description='Pixels to extend up (0-2000)',
    ),
    down: int = Field(
        default=0,
        description='Pixels to extend down (0-2000)',
    ),
    creativity: float = Field(
        default=DEFAULT_OUTPAINT_CREATIVITY,
        description='Controls extension creativity (0.0-1.0). Higher = more creative.',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the extended content (1-10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the outpainted image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Extend images beyond their original boundaries.

    This tool extends images beyond their original boundaries in specified
    directions, generating new content that matches the original image style.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Extend images in any direction (left, right, up, down)
    - Generate content matching original style
    - Control creativity level
    - Useful for changing aspect ratios or expanding compositions

    ## When to Use

    - Expanding image canvas
    - Changing aspect ratios
    - Adding more context to scenes
    - Creating panoramic views

    ## Example Usage

    - Image: "landscape.jpg"
      Prompt: "mountain peaks and sky"
      Up: 500
      (Extends image upward with mountains and sky)

    Returns:
        McpImageGenerationResponse: A response containing the outpainted image paths.
    """
    logger.debug(f"MCP tool outpaint_image called with prompt: '{prompt[:30]}...'")

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = OutpaintParams(
            image=image,
            prompt=prompt,
            left=left,
            right=right,
            up=up,
            down=down,
            creativity=creativity,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=output_format_enum,
        )

        logger.info(f'Outpainting image: left={left}, right={right}, up={up}, down={down}')

        response = await outpaint(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to outpaint image: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_outpaint: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='search_and_replace')
async def mcp_search_replace(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    search_prompt: str = Field(
        description='Description of object to find and replace (1-10,000 characters)'
    ),
    prompt: str = Field(description='Description of replacement content (1-10,000 characters)'),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the replacement (1-10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Find and replace objects using text prompts.

    This tool automatically finds specific objects or areas in images using text prompts
    and replaces them with AI-generated content. No manual masking required.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## CRITICAL: When to Use This Tool

    **USE THIS TOOL FOR:**
    - Replacing specific objects (e.g., "replace the chair with a sofa")
    - Changing ONLY the background (e.g., search_prompt="background", prompt="beach scene")
    - Swapping elements while keeping the rest unchanged
    - Product photography variations (e.g., change product color/style)
    - Targeted edits to specific parts of an image

    **DO NOT USE THIS TOOL FOR:**
    - Changing overall image style → Use transform_image_sd35 instead
    - Just recoloring an object → Use search_and_recolor instead (faster, preserves structure)
    - Removing objects without replacement → Use remove_object instead

    ## How It Works

    1. search_prompt: Describes what to find (e.g., "background", "chair", "person's shirt")
    2. prompt: Describes what to replace it with (e.g., "beach with palm trees", "red leather sofa")
    3. The tool automatically detects the object/area and replaces it seamlessly

    ## Key Features

    - Automatic object detection via text prompts
    - No manual masking required
    - Natural replacement blending
    - Preserves the rest of the image unchanged

    ## Example Usage

    - Image: "room.jpg"
      Search Prompt: "wooden chair"
      Prompt: "modern leather armchair"
      (Finds wooden chair and replaces with leather armchair)

    - Image: "portrait.jpg"
      Search Prompt: "background"
      Prompt: "beautiful beach scene with sand and ocean"
      (Replaces only the background, keeps subject unchanged)

    Returns:
        McpImageGenerationResponse: A response containing the edited image paths.
    """
    logger.debug('MCP tool search_and_replace called')

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = SearchReplaceParams(
            image=image,
            search_prompt=search_prompt,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=output_format_enum,
        )

        logger.info(f'Search and replace: "{search_prompt}" -> "{prompt}"')

        response = await search_and_replace(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to search and replace: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_search_replace: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='search_and_recolor')
async def mcp_search_recolor(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    select_prompt: str = Field(
        description='Description of object to recolor (1-10,000 characters)'
    ),
    prompt: str = Field(description='Description of desired color/style (1-10,000 characters)'),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the recoloring (1-10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Recolor specific objects using text prompts.

    This tool automatically finds objects in images and changes their colors while
    preserving the object's structure, texture, and details. This is a specialized
    version of inpainting that focuses only on color changes.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## CRITICAL: When to Use This Tool

    **USE THIS TOOL FOR:**
    - Changing ONLY the color of specific objects (e.g., "make the car red")
    - Product photography color variations (e.g., show shirt in different colors)
    - Recoloring elements while keeping structure intact
    - Quick color experiments without changing shape/texture

    **DO NOT USE THIS TOOL FOR:**
    - Replacing objects entirely → Use search_and_replace instead
    - Changing object shape or structure → Use search_and_replace instead
    - Changing overall image style → Use transform_image_sd35 instead

    ## How It Works

    1. select_prompt: Describes what object to recolor (e.g., "car body", "shirt", "wall")
    2. prompt: Describes the desired color/style (e.g., "bright red", "navy blue", "golden yellow")
    3. The tool automatically segments the object and recolors it while preserving all details

    ## Key Features

    - Automatic object detection via text prompts
    - Color changes while preserving structure and texture
    - No manual masking required
    - Maintains image quality and detail
    - Faster than search_and_replace for color-only changes

    ## Example Usage

    - Image: "car.jpg"
      Select Prompt: "car body"
      Prompt: "bright red metallic paint"
      (Changes car body color to red while keeping all details)

    - Image: "room.jpg"
      Select Prompt: "wall"
      Prompt: "soft beige color"
      (Recolors the wall without changing texture or structure)

    Returns:
        McpImageGenerationResponse: A response containing the recolored image paths.
    """
    logger.debug('MCP tool search_and_recolor called')

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = SearchRecolorParams(
            image=image,
            select_prompt=select_prompt,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=output_format_enum,
        )

        logger.info(f'Search and recolor: "{select_prompt}" -> "{prompt}"')

        response = await search_and_recolor(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to search and recolor: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_search_recolor: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='remove_object')
async def mcp_remove_object(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    mask: str = Field(
        description='Path to the mask image file or base64-encoded mask data. White areas are removed, black areas are preserved.'
    ),
    grow_mask: int = Field(
        default=DEFAULT_GROW_MASK,
        description='Pixels to expand mask edges (0-20). Helps blend edges.',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294)',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: "jpeg", "png", or "webp"',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Remove unwanted objects from images.

    This tool intelligently removes unwanted objects from images and fills
    the area with content that blends naturally with surroundings.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Intelligent object removal
    - Natural background filling
    - Mask-based control (white=remove, black=preserve)
    - grow_mask parameter for better edge blending

    ## When to Use

    - Removing unwanted objects
    - Cleaning up photos
    - Removing distractions
    - Photo restoration

    ## Example Usage

    - Image: "photo.jpg"
      Mask: "object_mask.png"
      (Removes object defined by mask)

    Returns:
        McpImageGenerationResponse: A response containing the edited image paths.
    """
    logger.debug('MCP tool remove_object called')

    try:
        output_format_enum = parse_output_format(output_format)

        # Create parameters model
        params = RemoveObjectParams(
            image=image,
            mask=mask,
            grow_mask=grow_mask,
            seed=seed,
            output_format=output_format_enum,
        )

        logger.info('Removing object from image')

        response = await remove_object(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to remove object: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_remove_object: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='remove_background')
async def mcp_remove_background(
    ctx: Context,
    image: str = Field(description='Path to the input image file or base64-encoded image data'),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Automatically remove backgrounds from images.

    This tool automatically removes backgrounds from images, isolating the
    main subject with clean edges. Always outputs PNG with transparency.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Automatic background removal
    - Clean edge detection
    - Handles complex subjects (hair, transparent objects)
    - Always outputs PNG with transparency

    ## When to Use

    - Product photography
    - Portrait isolation
    - Creating transparent assets
    - Compositing preparation

    ## Example Usage

    - Image: "product.jpg"
      (Automatically removes background, outputs PNG with transparency)

    Returns:
        McpImageGenerationResponse: A response containing the image paths with transparent background.
    """
    logger.debug('MCP tool remove_background called')

    try:
        # Create parameters model
        params = BackgroundRemovalParams(
            image=image,
        )

        logger.info('Removing background from image')

        response = await remove_background(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to remove background: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_remove_background: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='sketch_to_image')
async def mcp_sketch_to_image(
    ctx: Context,
    sketch: str = Field(
        description='Path to the sketch/line art image file or base64-encoded image data'
    ),
    prompt: str = Field(
        description='Description of desired output style and content (1-10,000 characters)'
    ),
    control_strength: float = Field(
        default=DEFAULT_CONTROL_STRENGTH,
        description='How closely to follow the sketch structure (0.0-1.0). Higher values mean stricter adherence to the sketch.',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the generated image (optional, up to 10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294). Use 0 for random.',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: jpeg, png, or webp',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Convert sketches or line art into detailed images.

    This tool converts sketches or line art into detailed images while
    preserving the structure and composition of the original sketch.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Preserves sketch structure and composition
    - Adds detail and style based on prompt
    - Control strength parameter for fine-tuning
    - Supports various artistic styles

    ## Parameters

    - **sketch**: Path to sketch/line art image or base64 data
    - **prompt**: Description of desired output (style, colors, details)
    - **control_strength**: 0.0-1.0 (default 0.7)
      - 0.0 = Ignore sketch, generate freely
      - 0.7 = Balanced adherence (recommended)
      - 1.0 = Strict adherence to sketch
    - **negative_prompt**: Elements to exclude (optional)
    - **seed**: For reproducible results (0 = random)
    - **output_format**: jpeg, png, or webp

    ## When to Use

    - Converting concept sketches to detailed art
    - Rapid visualization of ideas
    - Maintaining specific compositions
    - Iterating on sketch-based designs

    ## Example Usage

    - Sketch: "character_sketch.png"
      Prompt: "detailed fantasy character, vibrant colors, digital art style"
      Control Strength: 0.7
      (Converts sketch to detailed character art)

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug('MCP tool sketch_to_image called')

    try:
        # Create parameters model
        params = SketchToImageParams(
            control_image=sketch,
            prompt=prompt,
            control_strength=control_strength,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=parse_output_format(output_format),
        )

        logger.info(f'Converting sketch to image with control_strength={control_strength}')

        response = await sketch_to_image(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to convert sketch: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_sketch_to_image: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='structure_control')
async def mcp_structure_control(
    ctx: Context,
    structure_image: str = Field(
        description='Path to the structure/edge map image file or base64-encoded image data'
    ),
    prompt: str = Field(
        description='Description of desired output style and content (1-10,000 characters)'
    ),
    control_strength: float = Field(
        default=DEFAULT_CONTROL_STRENGTH,
        description='How closely to follow the structure (0.0-1.0). Higher values mean stricter adherence.',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the generated image (optional, up to 10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294). Use 0 for random.',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: jpeg, png, or webp',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Generate images following structural guides like edge maps or depth maps.

    This tool generates images that follow structural guides while adding
    detail and style based on the prompt. Maintains specific compositions.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Follows structural guidance (edges, depth, etc.)
    - Maintains composition while adding detail
    - Control strength for fine-tuning adherence
    - Supports various structural inputs

    ## Parameters

    - **structure_image**: Path to structure/edge map or base64 data
    - **prompt**: Description of desired output (style, colors, details)
    - **control_strength**: 0.0-1.0 (default 0.7)
      - 0.0 = Ignore structure, generate freely
      - 0.7 = Balanced adherence (recommended)
      - 1.0 = Strict adherence to structure
    - **negative_prompt**: Elements to exclude (optional)
    - **seed**: For reproducible results (0 = random)
    - **output_format**: jpeg, png, or webp

    ## When to Use

    - Maintaining specific compositions
    - Architectural visualization
    - Scene layout control
    - Depth-guided generation

    ## Example Usage

    - Structure Image: "edge_map.png"
      Prompt: "modern architecture, glass and steel, sunset lighting"
      Control Strength: 0.8
      (Generates detailed architecture following edge map)

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug('MCP tool structure_control called')

    try:
        # Create parameters model
        params = StructureControlParams(
            control_image=structure_image,
            prompt=prompt,
            control_strength=control_strength,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=parse_output_format(output_format),
        )

        logger.info(
            f'Generating image with structure control, control_strength={control_strength}'
        )

        response = await structure_control(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(
                f'Failed to generate with structure control: {response.message}'
            )
    except Exception as e:
        logger.error(f'Error in mcp_structure_control: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='style_guide')
async def mcp_style_guide(
    ctx: Context,
    reference_image: str = Field(
        description='Path to the reference image for style guidance or base64-encoded image data'
    ),
    prompt: str = Field(description='Description of desired content (1-10,000 characters)'),
    fidelity: float = Field(
        default=DEFAULT_STYLE_FIDELITY,
        description='How closely to match the reference style (0.0-1.0). Higher values mean closer style match.',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the generated image (optional, up to 10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294). Use 0 for random.',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: jpeg, png, or webp',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Generate images matching a reference style.

    This tool generates images matching a reference style while following
    the content description in the prompt. The fidelity parameter controls
    how closely the output matches the reference style.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Matches reference image style
    - Applies style to new content
    - Fidelity control for style strength
    - Maintains consistent visual aesthetics

    ## Parameters

    - **reference_image**: Path to style reference image or base64 data
    - **prompt**: Description of desired content
    - **fidelity**: 0.0-1.0 (default 0.5)
      - 0.0 = Loose style interpretation
      - 0.5 = Balanced style matching (recommended)
      - 1.0 = Very close style match
    - **negative_prompt**: Elements to exclude (optional)
    - **seed**: For reproducible results (0 = random)
    - **output_format**: jpeg, png, or webp

    ## When to Use

    - Maintaining consistent visual style across images
    - Applying artistic styles to new content
    - Brand consistency in generated images
    - Style exploration and iteration

    ## Example Usage

    - Reference Image: "art_style.jpg"
      Prompt: "mountain landscape at sunset"
      Fidelity: 0.7
      (Generates landscape in the reference art style)

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug('MCP tool style_guide called')

    try:
        # Create parameters model
        params = StyleGuideParams(
            reference_image=reference_image,
            prompt=prompt,
            fidelity=fidelity,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=parse_output_format(output_format),
        )

        logger.info(f'Generating image with style guide, fidelity={fidelity}')

        response = await style_guide(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to generate with style guide: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_style_guide: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='style_transfer')
async def mcp_style_transfer(
    ctx: Context,
    init_image: str = Field(
        description='Path to the content/initialization image file or base64-encoded image data'
    ),
    style_image: str = Field(
        description='Path to the style reference image file or base64-encoded image data'
    ),
    prompt: str = Field(description='Description to guide the transfer (1-10,000 characters)'),
    composition_fidelity: float = Field(
        default=DEFAULT_COMPOSITION_FIDELITY,
        description='How closely to preserve init_image composition (0.0-1.0). Higher values preserve more of the original composition.',
    ),
    style_strength: float = Field(
        default=DEFAULT_STYLE_STRENGTH,
        description='Strength of style application (0.0-1.0). Higher values apply stronger style.',
    ),
    change_strength: float = Field(
        default=DEFAULT_CHANGE_STRENGTH,
        description='Amount of change allowed (0.0-1.0). Higher values allow more changes.',
    ),
    negative_prompt: Optional[str] = Field(
        default=None,
        description='Elements to exclude from the result (optional, up to 10,000 characters)',
    ),
    seed: int = Field(
        default=0,
        description='Random seed for reproducibility (0-4,294,967,294). Use 0 for random.',
    ),
    output_format: str = Field(
        default=DEFAULT_OUTPUT_FORMAT,
        description='Output image format: jpeg, png, or webp',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the image to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description="""The current workspace directory where the image should be saved.
        CRITICAL: Assistant must always provide the current IDE workspace directory parameter to save images to the user's current project.""",
    ),
) -> McpImageGenerationResponse:
    """Apply style from one image to the content of another.

    This tool applies the style from one image to the content of another,
    with fine-grained control over composition, style strength, and changes.

    IMPORTANT FOR ASSISTANT: Always send the current workspace directory when calling this tool!

    ## Key Features

    - Transfers style between images
    - Fine-grained control over composition and style
    - Preserves content while applying style
    - Multiple control parameters for precise results

    ## Parameters

    - **init_image**: Path to content/initialization image or base64 data
    - **style_image**: Path to style reference image or base64 data
    - **prompt**: Description to guide the transfer
    - **composition_fidelity**: 0.0-1.0 (default 0.9)
      - Controls how much of the init_image composition is preserved
      - Higher = more preservation of original composition
    - **style_strength**: 0.0-1.0 (default 1.0)
      - Controls strength of style application
      - Higher = stronger style application
    - **change_strength**: 0.0-1.0 (default 0.9)
      - Controls amount of change allowed
      - Higher = more changes permitted
    - **negative_prompt**: Elements to exclude (optional)
    - **seed**: For reproducible results (0 = random)
    - **output_format**: jpeg, png, or webp

    ## When to Use

    - Artistic style transfer
    - Photo stylization
    - Creating variations with different styles
    - Combining content and style from different sources

    ## Example Usage

    - Init Image: "photo.jpg"
      Style Image: "painting.jpg"
      Prompt: "transfer impressionist painting style to photograph"
      Composition Fidelity: 0.9
      Style Strength: 1.0
      (Applies painting style to photo while preserving composition)

    Returns:
        McpImageGenerationResponse: A response containing the generated image paths.
    """
    logger.debug('MCP tool style_transfer called')

    try:
        # Create parameters model
        params = StyleTransferParams(
            init_image=init_image,
            style_image=style_image,
            prompt=prompt,
            composition_fidelity=composition_fidelity,
            style_strength=style_strength,
            change_strength=change_strength,
            negative_prompt=negative_prompt,
            seed=seed,
            output_format=parse_output_format(output_format),
        )

        logger.info(
            f'Transferring style with composition_fidelity={composition_fidelity}, '
            f'style_strength={style_strength}, change_strength={change_strength}'
        )

        response = await style_transfer(
            params=params,
            bedrock_client=bedrock_runtime_client,
            workspace_dir=workspace_dir,
            filename=filename,
        )

        if response.status == 'success':
            return McpImageGenerationResponse(
                status='success',
                paths=[f'file://{path}' for path in response.paths],
            )
        else:
            raise ImageServiceError(f'Failed to transfer style: {response.message}')
    except Exception as e:
        logger.error(f'Error in mcp_style_transfer: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='create_rectangular_mask')
async def mcp_create_rectangular_mask(
    ctx: Context,
    width: int = Field(description='Total width of the mask image in pixels'),
    height: int = Field(description='Total height of the mask image in pixels'),
    x: int = Field(description='X coordinate of the top-left corner of the white rectangle'),
    y: int = Field(description='Y coordinate of the top-left corner of the white rectangle'),
    mask_width: int = Field(description='Width of the white rectangle in pixels'),
    mask_height: int = Field(description='Height of the white rectangle in pixels'),
    feather: int = Field(
        default=0,
        description='Optional feathering/blur radius in pixels for soft edges (0-50)',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the mask to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description='The current workspace directory where the mask should be saved.',
    ),
) -> McpImageGenerationResponse:
    """Create a rectangular mask for inpainting or object removal.

    This tool creates a grayscale mask image where:
    - White pixels (255) = areas to fill/remove
    - Black pixels (0) = areas to preserve

    The mask is a rectangle positioned at (x, y) with the specified dimensions.
    Use this mask with inpaint_image or remove_object tools.

    ## When to Use

    - Creating masks for inpainting specific rectangular regions
    - Removing rectangular objects or areas
    - Testing inpaint/remove_object functionality
    - Quick mask creation without external tools

    ## Parameters Guide

    - **width, height**: Match your source image dimensions
    - **x, y**: Top-left corner of the area to mask
    - **mask_width, mask_height**: Size of the masked area
    - **feather**: Add soft edges (5-10 for subtle, 20+ for very soft)

    ## Example Usage

    - Image: 1024x768 photo
      Mask: x=200, y=150, mask_width=300, mask_height=200
      (Creates mask for 300x200 area starting at position 200,150)

    Returns:
        McpImageGenerationResponse: A response containing the mask file path.
    """
    logger.debug('MCP tool create_rectangular_mask called')

    try:
        mask_bytes = create_rectangular_mask(
            width=width,
            height=height,
            x=x,
            y=y,
            mask_width=mask_width,
            mask_height=mask_height,
            feather=feather,
        )

        mask_path = write_mask(mask_bytes, filename, workspace_dir)

        logger.bind(
            path=mask_path,
            feather=feather,
        ).info(f'Created rectangular mask: {mask_width}x{mask_height} at ({x},{y})')

        return McpImageGenerationResponse(
            status='success',
            paths=[f'file://{mask_path}'],
        )
    except Exception as e:
        logger.error(f'Error in mcp_create_rectangular_mask: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='create_ellipse_mask')
async def mcp_create_ellipse_mask(
    ctx: Context,
    width: int = Field(description='Total width of the mask image in pixels'),
    height: int = Field(description='Total height of the mask image in pixels'),
    center_x: int = Field(description='X coordinate of the ellipse center'),
    center_y: int = Field(description='Y coordinate of the ellipse center'),
    radius_x: int = Field(description='Horizontal radius of the ellipse in pixels'),
    radius_y: int = Field(description='Vertical radius of the ellipse in pixels'),
    feather: int = Field(
        default=0,
        description='Optional feathering/blur radius in pixels for soft edges (0-50)',
    ),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the mask to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description='The current workspace directory where the mask should be saved.',
    ),
) -> McpImageGenerationResponse:
    """Create an elliptical mask for inpainting or object removal.

    This tool creates a grayscale mask image where:
    - White pixels (255) = areas to fill/remove
    - Black pixels (0) = areas to preserve

    The mask is an ellipse centered at (center_x, center_y) with the specified radii.
    Use this mask with inpaint_image or remove_object tools.

    ## When to Use

    - Creating masks for circular or oval objects
    - Removing round objects (faces, balls, wheels, etc.)
    - More natural-looking masks than rectangles
    - Masking organic shapes

    ## Parameters Guide

    - **width, height**: Match your source image dimensions
    - **center_x, center_y**: Center point of the ellipse
    - **radius_x, radius_y**: Half-width and half-height of the ellipse
    - **feather**: Add soft edges (5-10 for subtle, 20+ for very soft)

    ## Example Usage

    - Image: 1024x768 photo with a face at center
      Mask: center_x=512, center_y=384, radius_x=100, radius_y=120
      (Creates oval mask around the face)

    Returns:
        McpImageGenerationResponse: A response containing the mask file path.
    """
    logger.debug('MCP tool create_ellipse_mask called')

    try:
        mask_bytes = create_ellipse_mask(
            width=width,
            height=height,
            center_x=center_x,
            center_y=center_y,
            radius_x=radius_x,
            radius_y=radius_y,
            feather=feather,
        )

        mask_path = write_mask(mask_bytes, filename, workspace_dir)

        logger.bind(
            path=mask_path,
            feather=feather,
        ).info(f'Created ellipse mask: radii {radius_x}x{radius_y} at ({center_x},{center_y})')

        return McpImageGenerationResponse(
            status='success',
            paths=[f'file://{mask_path}'],
        )
    except Exception as e:
        logger.error(f'Error in mcp_create_ellipse_mask: {str(e)}')
        await ctx.error(str(e))
        raise


@mcp.tool(name='create_full_mask')
async def mcp_create_full_mask(
    ctx: Context,
    width: int = Field(description='Width of the mask image in pixels'),
    height: int = Field(description='Height of the mask image in pixels'),
    filename: Optional[str] = Field(
        default=None,
        description='The name of the file to save the mask to (without extension)',
    ),
    workspace_dir: Optional[str] = Field(
        default=None,
        description='The current workspace directory where the mask should be saved.',
    ),
) -> McpImageGenerationResponse:
    """Create a full white mask covering the entire image.

    This tool creates a completely white mask useful for:
    - Complete image replacement with inpainting
    - Testing inpaint functionality
    - Full-image generative fill

    ## When to Use

    - Testing inpaint_image tool
    - Replacing entire image content while maintaining dimensions
    - Quick mask creation for full-image operations

    ## Example Usage

    - Image: 1024x768 photo
      Mask: width=1024, height=768
      (Creates full white mask for complete replacement)

    Returns:
        McpImageGenerationResponse: A response containing the mask file path.
    """
    logger.debug('MCP tool create_full_mask called')

    try:
        mask_bytes = create_full_mask(width=width, height=height)

        mask_path = write_mask(mask_bytes, filename, workspace_dir)

        logger.bind(path=mask_path).info(f'Created full mask: {width}x{height}')

        return McpImageGenerationResponse(
            status='success',
            paths=[f'file://{mask_path}'],
        )
    except Exception as e:
        logger.error(f'Error in mcp_create_full_mask: {str(e)}')
        await ctx.error(str(e))
        raise


def main():
    """Run the MCP server with CLI argument support."""
    logger.info('Starting bedrock-image-mcp-server MCP server')
    mcp.run()


if __name__ == '__main__':
    main()
