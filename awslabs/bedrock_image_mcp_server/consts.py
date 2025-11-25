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
# Constants
NOVA_CANVAS_MODEL_ID = 'amazon.nova-canvas-v1:0'
SD35_LARGE_MODEL_ID = 'stability.sd3-5-large-v1:0'

# Stability AI Upscale Model IDs
STABLE_UPSCALE_CREATIVE_MODEL_ID = 'us.stability.stable-creative-upscale-v1:0'
STABLE_UPSCALE_CONSERVATIVE_MODEL_ID = 'us.stability.stable-conservative-upscale-v1:0'
STABLE_UPSCALE_FAST_MODEL_ID = 'us.stability.stable-fast-upscale-v1:0'

# Stability AI Edit Service Model IDs
STABLE_INPAINT_MODEL_ID = 'us.stability.stable-image-inpaint-v1:0'
STABLE_OUTPAINT_MODEL_ID = 'us.stability.stable-outpaint-v1:0'
STABLE_SEARCH_REPLACE_MODEL_ID = 'us.stability.stable-image-search-replace-v1:0'
STABLE_SEARCH_RECOLOR_MODEL_ID = 'us.stability.stable-image-search-recolor-v1:0'
STABLE_ERASE_OBJECT_MODEL_ID = 'us.stability.stable-image-erase-object-v1:0'
STABLE_REMOVE_BACKGROUND_MODEL_ID = 'us.stability.stable-image-remove-background-v1:0'

# Stability AI Control Service Model IDs
STABLE_CONTROL_SKETCH_MODEL_ID = 'us.stability.stable-image-control-sketch-v1:0'
STABLE_CONTROL_STRUCTURE_MODEL_ID = 'us.stability.stable-image-control-structure-v1:0'
STABLE_STYLE_GUIDE_MODEL_ID = 'us.stability.stable-image-style-guide-v1:0'
STABLE_STYLE_TRANSFER_MODEL_ID = 'us.stability.stable-style-transfer-v1:0'

# Nova Canvas defaults
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_QUALITY = 'standard'
DEFAULT_CFG_SCALE = 6.5
DEFAULT_NUMBER_OF_IMAGES = 1
DEFAULT_OUTPUT_DIR = 'output'  # Default directory inside workspace_dir

# SD3.5 defaults
DEFAULT_SD35_ASPECT_RATIO = '1:1'
DEFAULT_SD35_SEED = 0
MAX_PROMPT_LENGTH_SD35 = 10000
MAX_PROMPT_LENGTH_NOVA = 1024
SD35_MAX_SEED = 4294967294
NOVA_MAX_SEED = 858993459

# Upscale defaults
DEFAULT_CREATIVE_UPSCALE_CREATIVITY = 0.3
DEFAULT_CONSERVATIVE_UPSCALE_CREATIVITY = 0.35
DEFAULT_OUTPUT_FORMAT = 'png'

# Edit service defaults
DEFAULT_GROW_MASK = 5
DEFAULT_OUTPAINT_CREATIVITY = 0.5
MAX_OUTPAINT_DIRECTION_PIXELS = 2000

# Control service defaults
DEFAULT_CONTROL_STRENGTH = 0.7
DEFAULT_STYLE_FIDELITY = 0.5
DEFAULT_COMPOSITION_FIDELITY = 0.9
DEFAULT_STYLE_STRENGTH = 1.0
DEFAULT_CHANGE_STRENGTH = 0.9

# Pixel constraint constants
MIN_IMAGE_DIMENSION = 64
MAX_CREATIVE_UPSCALE_INPUT_PIXELS = 1048576  # 1 megapixel
MAX_CONSERVATIVE_UPSCALE_INPUT_PIXELS = 9437184  # ~9.4 megapixels
MAX_FAST_UPSCALE_INPUT_PIXELS = 1048576  # 1 megapixel
MIN_FAST_UPSCALE_INPUT_PIXELS = 1024  # Minimum pixels for fast upscale

# AWS SDK Retry Configuration (following AWS best practices)
# See: https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html
BEDROCK_MAX_RETRY_ATTEMPTS = 3  # Total of 4 attempts (1 initial + 3 retries)
BEDROCK_RETRY_MODE = 'adaptive'  # AWS SDK handles exponential backoff with jitter
BEDROCK_CONNECT_TIMEOUT = 10  # Seconds to wait for connection
BEDROCK_READ_TIMEOUT = 120  # Seconds to wait for response (image generation can take 30-90s)
BEDROCK_MAX_POOL_CONNECTIONS = 50  # Connection pool size for concurrent requests


# Nova Canvas Prompt Best Practices
PROMPT_INSTRUCTIONS = """
# Amazon Nova Canvas Prompting Best Practices

## General Guidelines

- Prompts must be no longer than 1024 characters. For very long prompts, place the least important details near the end.
- Do not use negation words like "no", "not", "without" in your prompt. The model doesn't understand negation and will result in the opposite of what you intend.
- Use negative prompts (via the `negative_prompt` parameter) to specify objects or characteristics to exclude from the image.
- Omit negation words from your negative prompts as well.

## Effective Prompt Structure

An effective prompt often includes short descriptions of:

1. The subject
2. The environment
3. (optional) The position or pose of the subject
4. (optional) Lighting description
5. (optional) Camera position/framing
6. (optional) The visual style or medium ("photo", "illustration", "painting", etc.)

## Refining Results

When the output is close to what you want but not perfect:

1. Use a consistent `seed` value and make small changes to your prompt or negative prompt.
2. Once the prompt is refined, generate more variations using the same prompt but different `seed` values.

## Examples

### Example 1: Stock Photo
**Prompt:** "realistic editorial photo of female teacher standing at a blackboard with a warm smile"
**Negative Prompt:** "crossed arms"

### Example 2: Story Illustration
**Prompt:** "whimsical and ethereal soft-shaded story illustration: A woman in a large hat stands at the ship's railing looking out across the ocean"
**Negative Prompt:** "clouds, waves"

### Example 3: Pre-visualization for TV/Film
**Prompt:** "drone view of a dark river winding through a stark Iceland landscape, cinematic quality"

### Example 4: Fashion/Editorial Content
**Prompt:** "A cool looking stylish man in an orange jacket, dark skin, wearing reflective glasses. Shot from slightly low angle, face and chest in view, aqua blue sleek building shapes in background."

## Using Negative Prompts

Negative prompts can be surprisingly useful. Use them to exclude objects or style characteristics that might otherwise naturally occur as a result of your main prompt.

For example, adding "waves, clouds" as a negative prompt to a ship scene will result in a cleaner, more minimal composition.
"""


# Stable Diffusion 3.5 Large Prompt Best Practices
SD35_PROMPT_INSTRUCTIONS = """
# Stable Diffusion 3.5 Large Prompting Best Practices

## Key Differences from Nova Canvas

- Supports up to 10,000 characters (vs 1,024 for Nova)
- Better prompt adherence and detail preservation
- Supports aspect ratios instead of explicit dimensions
- No quality or cfg_scale parameters

## Effective Prompting

- Be specific and descriptive with your prompts
- Use aspect ratios appropriate for your use case (16:9 for landscapes, 9:16 for portraits, etc.)
- Leverage negative prompts to exclude unwanted elements
- Use seeds for reproducible results
- Unlike Nova Canvas, SD3.5 can handle longer, more detailed prompts effectively

## Image-to-Image Tips

- Strength parameter controls transformation intensity (0.0-1.0)
  - 0.0 = preserve input image completely
  - 1.0 = ignore input image completely
- Start with 0.7 for balanced transformation
- Lower strength (0.3-0.5) for subtle changes
- Higher strength (0.8-0.95) for dramatic reimagining
- Use consistent seeds to iterate on results

## Aspect Ratios

Available aspect ratios:
- 16:9 - Widescreen landscape
- 1:1 - Square
- 21:9 - Ultra-wide
- 2:3 - Portrait
- 3:2 - Landscape
- 4:5 - Portrait
- 5:4 - Landscape
- 9:16 - Vertical/mobile
- 9:21 - Ultra-tall

## Examples

### Example 1: Detailed Scene
**Prompt:** "A serene Japanese garden at dawn, with a wooden bridge over a koi pond, cherry blossoms in full bloom, soft morning mist, traditional stone lanterns, and Mount Fuji visible in the distance, photorealistic style"
**Aspect Ratio:** 16:9
**Negative Prompt:** "people, modern buildings, cars, power lines"

### Example 2: Character Portrait
**Prompt:** "Portrait of a wise elderly wizard with a long silver beard, wearing deep blue robes embroidered with golden stars, holding an ancient wooden staff, warm candlelight illuminating his face, fantasy art style"
**Aspect Ratio:** 2:3
**Negative Prompt:** "modern clothing, glasses, hat"

### Example 3: Image-to-Image Transformation
**Prompt:** "Transform into a watercolor painting style, soft pastel colors, artistic brush strokes"
**Strength:** 0.6
**Negative Prompt:** "photorealistic, sharp edges, digital art"
"""


# Stability AI Services Best Practices
STABILITY_SERVICES_INSTRUCTIONS = """
# Stability AI Image Services Best Practices

## Upscale Services

### Creative Upscale
- Upscales images to 4K resolution (20-40x)
- Best for: Low-resolution images that need enhancement
- Input: 64x64 to 1 megapixel
- Creativity parameter (0.1-0.5): Controls how much AI enhancement to apply
  - 0.1 = Minimal enhancement, closer to original
  - 0.5 = Maximum creative enhancement
- Use descriptive prompts to guide the upscaling style
- Supports style presets for specific aesthetics

### Conservative Upscale
- Upscales to 4K while preserving original details
- Best for: Images that need resolution increase without alteration
- Input: 64x64 to 9.4 megapixels
- No creativity parameter - focuses on detail preservation
- Still accepts prompts for context

### Fast Upscale
- Quick 4x upscaling without creative enhancement
- Best for: Quick resolution increases
- Input: 32x32 to 1 megapixel
- Fastest option, minimal processing time

## Edit Services

### Inpainting (Generative Fill)
- Fill masked regions with AI-generated content
- Requires: Input image + grayscale mask (white=fill, black=preserve)
- grow_mask parameter (0-20): Expands mask edges for better blending
- Use descriptive prompts for the content to generate
- Blends naturally with surrounding image

### Outpainting
- Extend images beyond original boundaries
- Directional expansion: left, right, up, down (0-2000 pixels each)
- Creativity parameter controls extension style
- Use prompts to describe desired extended content
- Maintains consistency with original image

### Search and Replace
- Find and replace objects using text prompts
- search_prompt: Describes object to replace
- prompt: Describes replacement content
- Automatic object detection and masking
- No manual mask required

### Search and Recolor
- Recolor specific objects using text prompts
- select_prompt: Describes object to recolor
- prompt: Describes desired color/style
- Preserves object structure while changing colors
- Maintains image quality and detail

### Remove Object
- Remove unwanted objects from images
- Requires: Input image + mask defining object
- Intelligently fills removed area with surrounding context
- grow_mask parameter for better blending
- Seamless results

### Remove Background
- Automatically remove backgrounds from images
- No prompt required - fully automatic
- Returns PNG with transparency
- Handles complex subjects (hair, transparent objects)
- Clean edge detection

## Control Services

### Sketch to Image
- Convert sketches/line art into detailed images
- control_strength (0.0-1.0): How closely to follow sketch
  - 0.7 = Balanced (recommended)
  - Higher = Stricter adherence to sketch
  - Lower = More creative freedom
- Use descriptive prompts for style and details

### Structure Control
- Generate images following structural guides/edge maps
- control_strength parameter for adherence level
- Maintains composition while adding details
- Use prompts to describe desired style

### Style Guide
- Generate images matching a reference style
- fidelity parameter (0.0-1.0): Style matching strength
  - 0.5 = Balanced (recommended)
  - Higher = Closer to reference style
  - Lower = More creative interpretation
- Accepts prompts for content description

### Style Transfer
- Transfer style from one image to another
- Requires: init_image (content) + style_image (style reference)
- composition_fidelity (0.0-1.0): Preserve content structure
- style_strength (0.0-1.0): How much style to apply
- change_strength (0.0-1.0): Overall transformation intensity
- Fine-grained control over style application

## General Tips

- All services support seeds for reproducible results
- Use negative prompts to exclude unwanted elements
- Start with default parameters and adjust based on results
- Validate image dimensions before processing
- PNG format recommended for transparency support
- JPEG for smaller file sizes when transparency not needed
"""
