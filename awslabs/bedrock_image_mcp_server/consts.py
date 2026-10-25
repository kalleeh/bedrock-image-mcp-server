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
SD35_LARGE_MODEL_ID = 'stability.sd3-5-large-v1:0'
STABLE_IMAGE_ULTRA_MODEL_ID = 'stability.stable-image-ultra-v1:1'
STABLE_IMAGE_CORE_MODEL_ID = 'stability.stable-image-core-v1:1'

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

# Output file defaults
DEFAULT_OUTPUT_DIR = 'output'  # Default directory inside workspace_dir
MAX_FILENAME_LENGTH = 100  # Leaves room for the random suffix and extension within NAME_MAX

# SD3.5 defaults
DEFAULT_SD35_ASPECT_RATIO = '1:1'
DEFAULT_SD35_SEED = 0
MAX_PROMPT_LENGTH_SD35 = 10000
SD35_MAX_SEED = 4294967294

# Upscale defaults
DEFAULT_CREATIVE_UPSCALE_CREATIVITY = 0.3
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
MAX_DECODED_IMAGE_PIXELS = 67108864  # 64 megapixels; guards against decompression bombs
MAX_MASK_FEATHER = 50  # Gaussian blur radius cap for generated masks
# InvokeModel caps the request body at 25,000,000 bytes and base64 inflates by 4/3, so a larger
# file cannot be sent. Checked before reading so a huge file is never loaded into memory.
MAX_INPUT_IMAGE_BYTES = 18_000_000
MAX_CREATIVE_UPSCALE_INPUT_PIXELS = 1048576  # 1 megapixel
MAX_CONSERVATIVE_UPSCALE_INPUT_PIXELS = 9437184  # ~9.4 megapixels
MAX_FAST_UPSCALE_INPUT_PIXELS = 1048576  # 1 megapixel
MIN_FAST_UPSCALE_DIMENSION = 32  # Fast upscale accepts 32-1,536px per side
MAX_FAST_UPSCALE_DIMENSION = 1536
MAX_CONSERVATIVE_UPSCALE_ASPECT_RATIO = 2.5  # Conservative accepts 1:2.5 to 2.5:1

# AWS SDK Retry Configuration (following AWS best practices)
# See: https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/retry-backoff.html
BEDROCK_MAX_RETRY_ATTEMPTS = 3  # Total of 4 attempts (1 initial + 3 retries)
BEDROCK_RETRY_MODE = 'adaptive'  # AWS SDK handles exponential backoff with jitter
BEDROCK_CONNECT_TIMEOUT = 10  # Seconds to wait for connection
BEDROCK_READ_TIMEOUT = 120  # Seconds to wait for response (image generation can take 30-90s)
BEDROCK_MAX_POOL_CONNECTIONS = 50  # Connection pool size for concurrent requests

# Seconds to keep serving after stdin reaches EOF while a tool call is still in flight, so the
# response is not discarded. Sized just past BEDROCK_READ_TIMEOUT (120s) plus retries, which is
# the longest a tool can legitimately run. The bound matters: EOF-triggered shutdown is
# deliberate upstream (python-sdk#2231) because it stops the server outliving a dead client, so
# a hung call must not wedge the process forever.
STDIN_EOF_GRACE_SECONDS = 150


# Stable Diffusion 3.5 Large Prompt Best Practices
SD35_PROMPT_INSTRUCTIONS = """
# Stable Diffusion 3.5 Large Prompting Best Practices

## Key Features

- Supports up to 10,000 characters
- Strong prompt adherence and detail preservation
- Supports aspect ratios instead of explicit dimensions

## Effective Prompting

- Be specific and descriptive with your prompts
- Use aspect ratios appropriate for your use case (16:9 for landscapes, 9:16 for portraits, etc.)
- Leverage negative prompts to exclude unwanted elements
- Use seeds for reproducible results
- SD3.5 handles long, detailed prompts effectively

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

**On output_format for upscaling.** All three formats (png, jpeg, webp) are supported here as
everywhere else; png is the default except for upscale_creative, which defaults to jpeg. The constraint is output *size*, not format: Bedrock
returns the image base64-encoded in the JSON response and caps that response at 16MB, which
works out to roughly a 12MB image. Ordinary generation is ~2MB and never comes close, but
upscaling returns 3K-4K images, where a PNG is typically 20-35MB and the call fails with
"Response payload size ... exceeds the maximum allowed size".

So for a full-size upscale, pass output_format="jpeg" or "webp" (a few MB at the same
resolution). PNG is fine when the result is small — a 256x256 input upscaled 4x came back as
a 1.4MB PNG. Creative upscale is the exception: its output is a fixed ~3150x3150 no matter
the input, so PNG has no working input size there.

### Creative Upscale
- Upscales images to 4K resolution (20-40x)
- Best for: Low-resolution images that need enhancement
- Input: 64x64 to 1 megapixel
- Creativity parameter (0.1-0.5): Controls how much AI enhancement to apply
  - 0.1 = Minimal enhancement, closer to original
  - 0.5 = Maximum creative enhancement
- Use descriptive prompts to guide the upscaling style
- Supports style presets for specific aesthetics
- Use output_format="jpeg" or "webp": the ~3150x3150 result is fixed, so it never fits the
  16MB response limit as a PNG at any input size

### Conservative Upscale
- Upscales to 4K while preserving original details
- Best for: Images that need resolution increase without alteration
- Input: 64px per side minimum, up to 9.4 megapixels, aspect ratio 1:2.5 to 2.5:1
- No creativity parameter - focuses on detail preservation
- Still accepts prompts for context
- Use output_format="jpeg" or "webp" for 1MP inputs: the ~3112x3112 result exceeds the 16MB
  response limit as a PNG. PNG works from small inputs (256x256 in gave a 9.9MB PNG)

### Fast Upscale
- Quick 4x upscaling without creative enhancement
- Best for: Quick resolution increases
- Input: 32-1,536px per side, up to 1 megapixel
- Fastest option, minimal processing time
- Output scales 4x with the input, so use output_format="jpeg" or "webp" above ~256x256:
  a 1MP input yields 4096x4096, which exceeds the 16MB response limit as a PNG

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
