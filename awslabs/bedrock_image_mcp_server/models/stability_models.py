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
"""Pydantic models for Stability AI Image Services."""

from .common import OutputFormat
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class StylePreset(str, Enum):
    """Style presets for upscaling and other Stability AI services.

    Attributes:
        MODEL_3D: 3D model style.
        ANALOG_FILM: Analog film photography style.
        ANIME: Anime/manga style.
        CINEMATIC: Cinematic/movie style.
        COMIC_BOOK: Comic book illustration style.
        DIGITAL_ART: Digital art style.
        ENHANCE: General enhancement style.
        FANTASY_ART: Fantasy art style.
        ISOMETRIC: Isometric perspective style.
        LINE_ART: Line art/sketch style.
        LOW_POLY: Low polygon 3D style.
        MODELING_COMPOUND: Modeling compound/clay style.
        NEON_PUNK: Neon punk/cyberpunk style.
        ORIGAMI: Origami paper craft style.
        PHOTOGRAPHIC: Photographic/realistic style.
        PIXEL_ART: Pixel art style.
        TILE_TEXTURE: Seamless tile texture style.
    """
    MODEL_3D = '3d-model'
    ANALOG_FILM = 'analog-film'
    ANIME = 'anime'
    CINEMATIC = 'cinematic'
    COMIC_BOOK = 'comic-book'
    DIGITAL_ART = 'digital-art'
    ENHANCE = 'enhance'
    FANTASY_ART = 'fantasy-art'
    ISOMETRIC = 'isometric'
    LINE_ART = 'line-art'
    LOW_POLY = 'low-poly'
    MODELING_COMPOUND = 'modeling-compound'
    NEON_PUNK = 'neon-punk'
    ORIGAMI = 'origami'
    PHOTOGRAPHIC = 'photographic'
    PIXEL_ART = 'pixel-art'
    TILE_TEXTURE = 'tile-texture'


class CreativeUpscaleParams(BaseModel):
    """Parameters for creative upscaling to 4K with enhancement.

    Creative upscaling uses AI to enhance and add details while upscaling
    images to 4K resolution (20-40x upscale). Best for low-resolution or
    degraded images that need improvement.

    Attributes:
        image: Base64-encoded input image or file path.
        prompt: Descriptive prompt to guide upscaling style (1-10,000 chars).
        creativity: Controls enhancement level (0.1-0.5). Higher = more creative.
        negative_prompt: Elements to exclude from the upscaled image.
        seed: Random seed for reproducibility (0-4,294,967,294).
        style_preset: Optional style preset to apply during upscaling.
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    prompt: str = Field(..., min_length=1, max_length=10000)
    creativity: float = Field(default=0.3, ge=0.1, le=0.5)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    style_preset: Optional[StylePreset] = None
    output_format: OutputFormat = OutputFormat.PNG


class ConservativeUpscaleParams(BaseModel):
    """Parameters for conservative upscaling to 4K preserving details.

    Conservative upscaling increases resolution to 4K while preserving
    the original image characteristics with minimal alterations. Best for
    images that already have good quality but need higher resolution.

    Attributes:
        image: Base64-encoded input image or file path.
        prompt: Descriptive prompt for context (1-10,000 chars).
        negative_prompt: Elements to exclude from the upscaled image.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG


class FastUpscaleParams(BaseModel):
    """Parameters for fast 4x upscaling without creative enhancement.

    Fast upscaling provides quick 4x resolution increase without AI enhancement.
    Best for images that need simple resolution increase without style changes.

    Attributes:
        image: Base64-encoded input image or file path.
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    output_format: OutputFormat = OutputFormat.PNG


class InpaintParams(BaseModel):
    """Parameters for inpainting (generative fill) service.

    Inpainting fills masked regions of images with AI-generated content that
    blends naturally with the surrounding image. The mask defines which areas
    to fill (white) and which to preserve (black).

    Attributes:
        image: Base64-encoded input image or file path.
        mask: Base64-encoded grayscale mask image. White areas are filled, black preserved.
        prompt: Description of desired fill content (1-10,000 chars).
        negative_prompt: Elements to exclude from the generated content.
        grow_mask: Pixels to expand mask edges (0-20). Helps blend edges.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    mask: str
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    grow_mask: int = Field(default=5, ge=0, le=20)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG


class OutpaintParams(BaseModel):
    """Parameters for outpainting (image expansion) service.

    Outpainting extends images beyond their original boundaries in specified
    directions, generating new content that matches the original image style.

    Attributes:
        image: Base64-encoded input image or file path.
        prompt: Description of desired extended content (1-10,000 chars).
        left: Pixels to extend left (0-2000).
        right: Pixels to extend right (0-2000).
        up: Pixels to extend up (0-2000).
        down: Pixels to extend down (0-2000).
        creativity: Controls extension creativity (0.0-1.0). Higher = more creative.
        negative_prompt: Elements to exclude from the extended content.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    prompt: str = Field(..., min_length=1, max_length=10000)
    left: int = Field(default=0, ge=0, le=2000)
    right: int = Field(default=0, ge=0, le=2000)
    up: int = Field(default=0, ge=0, le=2000)
    down: int = Field(default=0, ge=0, le=2000)
    creativity: float = Field(default=0.5, ge=0.0, le=1.0)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG


class SearchReplaceParams(BaseModel):
    """Parameters for search and replace service.

    Search and replace finds objects in images using text prompts and replaces
    them with AI-generated content, automatically handling masking.

    Attributes:
        image: Base64-encoded input image or file path.
        search_prompt: Description of object to find and replace (1-10,000 chars).
        prompt: Description of replacement content (1-10,000 chars).
        negative_prompt: Elements to exclude from the replacement.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    search_prompt: str = Field(..., min_length=1, max_length=10000)
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG


class SearchRecolorParams(BaseModel):
    """Parameters for search and recolor service.

    Search and recolor finds objects in images using text prompts and changes
    their colors while preserving structure and detail.

    Note: This service uses 'select_prompt' (not 'search_prompt') to identify objects.

    Attributes:
        image: Base64-encoded input image or file path.
        select_prompt: Description of object to recolor (1-10,000 chars).
        prompt: Description of desired color/style (1-10,000 chars).
        negative_prompt: Elements to exclude from the recoloring.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    select_prompt: str = Field(..., min_length=1, max_length=10000)
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG


class RemoveObjectParams(BaseModel):
    """Parameters for object removal service.

    Remove object intelligently removes unwanted objects from images and fills
    the area with content that blends naturally with surroundings.

    Attributes:
        image: Base64-encoded input image or file path.
        mask: Base64-encoded grayscale mask image. White areas are removed, black preserved.
        grow_mask: Pixels to expand mask edges (0-20). Helps blend edges.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    image: str
    mask: str
    grow_mask: int = Field(default=5, ge=0, le=20)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG


class BackgroundRemovalParams(BaseModel):
    """Parameters for background removal service.

    Background removal automatically removes backgrounds from images, isolating
    the main subject with clean edges. Always outputs PNG with transparency.

    Attributes:
        image: Base64-encoded input image or file path.
    """
    image: str


class SketchToImageParams(BaseModel):
    """Parameters for sketch-to-image control service.

    Sketch-to-image converts sketches or line art into detailed images while
    preserving the structure and composition of the original sketch.

    Note: AWS API expects 'image' parameter, not 'control_image'.

    Attributes:
        control_image: Base64-encoded sketch/line art image or file path (aliased to 'image' for API).
        prompt: Description of desired output style and content (1-10,000 chars).
        control_strength: How closely to follow the sketch (0.0-1.0). Higher = stricter adherence.
        negative_prompt: Elements to exclude from the generated image.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    control_image: str = Field(..., alias='image')
    prompt: str = Field(..., min_length=1, max_length=10000)
    control_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG

    class Config:
        populate_by_name = True  # Allow both 'control_image' and 'image'


class StructureControlParams(BaseModel):
    """Parameters for structure control service.

    Structure control generates images that follow structural guides like edge maps
    or depth maps, maintaining specific compositions while adding detail and style.

    Note: AWS API expects 'image' parameter, not 'control_image'.

    Attributes:
        control_image: Base64-encoded structure/edge map image or file path (aliased to 'image' for API).
        prompt: Description of desired output style and content (1-10,000 chars).
        control_strength: How closely to follow the structure (0.0-1.0). Higher = stricter adherence.
        negative_prompt: Elements to exclude from the generated image.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output format (jpeg, png, webp).
    """
    control_image: str = Field(..., alias='image')
    prompt: str = Field(..., min_length=1, max_length=10000)
    control_strength: float = Field(default=0.7, ge=0.0, le=1.0)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG

    class Config:
        populate_by_name = True  # Allow both 'control_image' and 'image'


class StyleGuideParams(BaseModel):
    """Parameters for style guide service.

    Style guide generates images matching a reference style while following
    the content description in the prompt. The fidelity parameter controls
    how closely the output matches the reference style.

    Note: AWS API expects 'image' parameter, not 'reference_image'.

    Attributes:
        reference_image: Base64-encoded reference image for style guidance or file path (aliased to 'image' for API).
        prompt: Description of desired content (1-10,000 chars).
        fidelity: How closely to match reference style (0.0-1.0). Higher = closer match.
        negative_prompt: Elements to exclude from the generated image.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    reference_image: str = Field(..., alias='image')
    prompt: str = Field(..., min_length=1, max_length=10000)
    fidelity: float = Field(default=0.5, ge=0.0, le=1.0)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG

    class Config:
        populate_by_name = True  # Allow both 'reference_image' and 'image'


class StyleTransferParams(BaseModel):
    """Parameters for style transfer service.

    Style transfer applies the style from one image to the content of another,
    with fine-grained control over composition, style strength, and changes.

    Attributes:
        init_image: Base64-encoded content/initialization image or file path.
        style_image: Base64-encoded style reference image or file path.
        prompt: Description to guide the transfer (1-10,000 chars).
        composition_fidelity: How closely to preserve init_image composition (0.0-1.0).
        style_strength: Strength of style application (0.0-1.0). Higher = stronger style.
        change_strength: Amount of change allowed (0.0-1.0). Higher = more changes.
        negative_prompt: Elements to exclude from the result.
        seed: Random seed for reproducibility (0-4,294,967,294).
        output_format: Output image format (jpeg, png, webp).
    """
    init_image: str
    style_image: str
    prompt: str = Field(..., min_length=1, max_length=10000)
    composition_fidelity: float = Field(default=0.9, ge=0.0, le=1.0)
    style_strength: float = Field(default=1.0, ge=0.0, le=1.0)
    change_strength: float = Field(default=0.9, ge=0.0, le=1.0)
    negative_prompt: Optional[str] = Field(None, max_length=10000)
    seed: int = Field(default=0, ge=0, le=4294967294)
    output_format: OutputFormat = OutputFormat.PNG
