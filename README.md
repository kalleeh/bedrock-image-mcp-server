# Amazon Bedrock Image Generation MCP Server

[![PyPI version](https://badge.fury.io/py/bedrock-image-mcp-server.svg)](https://badge.fury.io/py/bedrock-image-mcp-server)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/kalleeh/bedrock-image-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/kalleeh/bedrock-image-mcp-server/actions/workflows/ci.yml)
[![GitHub Actions](https://github.com/kalleeh/bedrock-image-mcp-server/workflows/Publish%20to%20PyPI/badge.svg)](https://github.com/kalleeh/bedrock-image-mcp-server/actions)

> **Note:** This is a community-maintained fork of [awslabs/mcp/bedrock-image-mcp-server](https://github.com/awslabs/mcp) with additional features and improvements. Original work by Amazon Web Services under Apache 2.0 license.

MCP server for generating and editing images using Amazon Nova Canvas, Stable Diffusion 3.5 Large, and Stability AI Image Services through Amazon Bedrock.

## Which model should I use?

Three Stability AI text-to-image models form a quality ladder, all in us-west-2:

| Tool | Model | Stability AI: "Ideal For" | Credits |
|---|---|---|---|
| `generate_image_ultra` | Stable Image Ultra — *"Photorealistic, Large-Scale Output"* | "Ultra-realistic imagery for luxury brands and high-end campaigns"; "professional print media and large format applications". Their example: a luxury brand producing magazine spreads | 8 |
| `generate_image_sd35` | Stable Diffusion 3.5 Large — *"High-Quality, High-Quantity Creative Assets"* | "High-volume outputs like marketing campaigns and digital assets"; "professional use cases at 1 megapixel resolution". Their example: a game team producing environment textures and character concepts | 6.5 |
| `generate_image_core` | Stable Image Core — *"Fast and Affordable"* | "Rapid content generation at scale"; "rapidly iterating on concepts during ideation". Their example: a retailer generating product images for new arrivals | 3 |

**In short:** `generate_image_ultra` for a few premium, large-format or print pieces;
`generate_image_sd35` when you need many good assets; `generate_image_core` when speed and cost
dominate. Stability credits only Ultra with **typography**, so prefer it when the image contains
text.

All quotes are Stability AI's own words, from their
[Bedrock launch post](https://stability.ai/news-updates/stability-ais-top-3-text-to-image-models-now-available-in-amazon-bedrock)
and their API specification (`api.stability.ai/v2alpha/openapi`). Credits are Stability's billing
unit — on Bedrock you are billed per image by AWS, so treat them as a cost *ratio*, not a price.

For image-to-image, `transform_image_sd35` is the only option of the four; Ultra and Core are
text-to-image only.

**On seeds:** a fixed non-zero `seed` reproduces the same image reliably within a short window
(verified 6/6 identical), but is best-effort rather than guaranteed — repeats separated by
longer intervals occasionally differ, which appears to be Bedrock serving the request from a
different backend. Use `seed=0` for explicitly random output.

## Features

### Stability AI Text-to-Image (4 tools) — recommended

#### Highest quality
- Generate images with `generate_image_ultra` (Stable Image Ultra)
- Stability AI's flagship model: best photorealism, lighting and legible text
- Same aspect ratios and prompt length as SD3.5; higher cost per image
- Text-to-image only, `png` or `jpeg` output (no webp)

#### Fastest and cheapest
- Generate images with `generate_image_core` (Stable Image Core)
- Lowest cost and latency; ideal for drafts and iterating on concepts
- Text-to-image only, `png` or `jpeg` output (no webp)

#### Balanced text-to-image generation
- Generate images from text prompts with `generate_image_sd35`
- Supports prompts up to 10,000 characters (vs 1,024 for Nova Canvas)
- 9 aspect ratio options: 16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21
- Superior prompt adherence and detail preservation
- Seed support for reproducible results (0-4,294,967,294)

#### Image-to-image transformation
- Transform existing images with `transform_image_sd35`
- Strength parameter (0.0-1.0) controls transformation intensity
- Supports file paths and base64 image inputs
- All text-to-image parameters available

### Amazon Nova Canvas (2 tools)

#### Text-based image generation
- Create images from text prompts with `generate_image`
- Customizable dimensions (320-4096px), quality options, and negative prompting
- Supports multiple image generation (1-5) in single request
- Adjustable parameters like cfg_scale (1.1-10.0) and seeded generation

#### Color-guided image generation
- Generate images with specific color palettes using `generate_image_with_colors`
- Define up to 10 hex color values to influence the image style and mood
- Same customization options as text-based generation

### Stability AI Upscale Services (3 tools)

#### Creative upscaling
- Upscale images to 4K with AI enhancement using `upscale_creative`
- 20-40x upscale from low-resolution inputs (64x64 to 1MP)
- Creativity parameter (0.1-0.5) controls enhancement level
- Style preset support for specific aesthetics

#### Conservative upscaling
- Upscale to 4K while preserving details with `upscale_conservative`
- Supports inputs up to 9.4 megapixels
- Minimal alterations to original image

#### Fast upscaling
- Quick 4x upscaling with `upscale_fast`
- Fast processing for quick resolution increases
- Supports inputs from 32x32 to 1MP

### Stability AI Edit Services (6 tools)

#### Inpainting (Generative Fill)
- Fill masked regions with AI content using `inpaint_image`
- Grayscale mask support (white=fill, black=preserve)
- grow_mask parameter (0-20) for edge blending

#### Outpainting
- Extend images beyond boundaries with `outpaint_image`
- Directional expansion: left, right, up, down (0-2000 pixels each)
- Creativity parameter for extension style

#### Search and Replace
- Find and replace objects with `search_and_replace`
- Automatic object detection and masking
- No manual mask required

#### Search and Recolor
- Recolor specific objects with `search_and_recolor`
- Preserves structure while changing colors
- Maintains image quality

#### Remove Object
- Remove unwanted objects with `remove_object`
- Context-aware filling of removed areas
- Seamless blending with surroundings

#### Remove Background
- Automatic background removal with `remove_background`
- Returns PNG with transparency
- Handles complex subjects (hair, transparent objects)

### Stability AI Control Services (4 tools)

#### Sketch to Image
- Convert sketches to detailed images with `sketch_to_image`
- control_strength parameter (0.0-1.0)
- Preserves sketch structure while adding detail

#### Structure Control
- Generate images from structural guides with `structure_control`
- Follows edge maps and structural guidance
- control_strength for adherence level

#### Style Guide
- Match reference image style with `style_guide`
- fidelity parameter (0.0-1.0) for style matching
- Accepts prompts for content description

#### Style Transfer
- Transfer style between images with `style_transfer`
- Fine-grained control: composition_fidelity, style_strength, change_strength
- Requires init_image (content) and style_image (style reference)

### Mask Creation Utilities (3 tools)

Create masks programmatically for use with `inpaint_image` and `remove_object` tools. Masks are grayscale images where white pixels indicate areas to fill/remove and black pixels indicate areas to preserve.

#### Rectangular Mask
- Create rectangular masks with `create_rectangular_mask`
- Configurable position (x, y) and size (width, height)
- Optional feathering (0-50 pixels) for soft edges
- Perfect for signs, windows, rectangular objects

#### Ellipse Mask
- Create elliptical/circular masks with `create_ellipse_mask`
- Configurable center point and radii
- Optional feathering for soft edges
- Ideal for faces, balls, wheels, organic shapes

#### Full Mask
- Create full white masks with `create_full_mask`
- Covers entire image
- Useful for testing and full-image replacement

### Workspace Integration

- All images saved to user-specified workspace directories with automatic folder creation
- Support for multiple output formats (PNG, JPEG, WebP)
- Unique filename generation or custom naming

### AWS Authentication

- Uses AWS profiles for secure access to Amazon Bedrock services
- Supports all AWS regions where Bedrock models are available

## Prerequisites

1. Install `uv` from [Astral](https://docs.astral.sh/uv/getting-started/installation/) or the [GitHub README](https://github.com/astral-sh/uv#installation)
2. Install Python using `uv python install 3.10`
3. Set up AWS credentials with access to Amazon Bedrock
   - You need an AWS account with Amazon Bedrock enabled
   - Configure AWS credentials with `aws configure` or environment variables
   - Ensure your IAM role/user has the required permissions (see [AWS IAM Permissions](#aws-iam-permissions) below)

## Installation

| Cursor | VS Code |
|:------:|:-------:|
| [![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/en/install-mcp?name=bedrock-image-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGJlZHJvY2staW1hZ2UtbWNwLXNlcnZlckBsYXRlc3QiLCJlbnYiOnsiQVdTX1BST0ZJTEUiOiJ5b3VyLWF3cy1wcm9maWxlIiwiQVdTX1JFR0lPTiI6InVzLXdlc3QtMiIsIkZBU1RNQ1BfTE9HX0xFVkVMIjoiRVJST1IifSwiZGlzYWJsZWQiOmZhbHNlLCJhdXRvQXBwcm92ZSI6W119) | [![Install on VS Code](https://img.shields.io/badge/Install_on-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=Bedrock%20Image%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22bedrock-image-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-west-2%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

Configure the MCP server in your MCP client configuration (e.g., for Amazon Q Developer CLI, edit `~/.aws/amazonq/mcp.json`):

> **Pick your region deliberately.** The examples below use `us-west-2`, which is also the
> default when `AWS_REGION` is unset. It is the only region carrying the recommended
> text-to-image models (Ultra, Core and SD3.5) and it also serves all 13 Stability
> edit/upscale/control tools, so every tool except Nova Canvas works there. Nova Canvas is
> *not* in us-west-2 — use `us-east-1`, `eu-west-1` or `ap-northeast-1` for that, and note it
> retires 2026-09-30. See [Supported AWS Regions](#supported-aws-regions).

```json
{
  "mcpServers": {
    "bedrock-image-mcp-server": {
      "command": "uvx",
      "args": ["bedrock-image-mcp-server@latest"],
      "env": {
        "AWS_PROFILE": "your-aws-profile",
        "AWS_REGION": "us-west-2",
        "FASTMCP_LOG_LEVEL": "ERROR"
      },
      "disabled": false,
      "autoApprove": []
    }
  }
}
```
### Windows Installation

For Windows users, the MCP server configuration format is slightly different:

```json
{
  "mcpServers": {
    "bedrock-image-mcp-server": {
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "uv",
      "args": [
        "tool",
        "run",
        "--from",
        "bedrock-image-mcp-server@latest",
        "bedrock-image-mcp-server.exe"
      ],
      "env": {
        "FASTMCP_LOG_LEVEL": "ERROR",
        "AWS_PROFILE": "your-aws-profile",
        "AWS_REGION": "us-west-2"
      }
    }
  }
}
```


or docker after a successful `docker build -t bedrock-image-mcp-server .`:

```file
# fictitious `.env` file with AWS temporary credentials
AWS_ACCESS_KEY_ID=ASIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_SESSION_TOKEN=AQoEXAMPLEH4aoAH0gNCAPy...truncated...zrkuWJOgQs8IZZaIv2BXIa2R4Olgk
```

```json
  {
    "mcpServers": {
      "bedrock-image-mcp-server": {
        "command": "docker",
        "args": [
          "run",
          "--rm",
          "--interactive",
          "--env",
          "AWS_REGION=us-west-2",
          "--env",
          "FASTMCP_LOG_LEVEL=ERROR",
          "--env-file",
          "/full/path/to/file/above/.env",
          "bedrock-image-mcp-server:latest"
        ],
        "env": {},
        "disabled": false,
        "autoApprove": []
      }
    }
  }
```

NOTE: Your credentials will need to be kept refreshed from your host

### Installing via Smithery

To install Amazon Bedrock Image MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/bedrock-image-mcp-server):

```bash
npx -y @smithery/cli install bedrock-image-mcp-server --client claude
```

### AWS Authentication

The MCP server uses the AWS profile specified in the `AWS_PROFILE` environment variable. If not provided, it defaults to the "default" profile in your AWS configuration file.

```json
"env": {
  "AWS_PROFILE": "your-aws-profile",
  "AWS_REGION": "us-west-2"
}
```

Make sure the AWS profile has permissions to access Amazon Bedrock and the image generation models. The MCP server creates a boto3 session using the specified profile to authenticate with AWS services. Your AWS IAM credentials remain on your local machine and are strictly used for using the Amazon Bedrock model APIs.

## Usage Examples

### Stability AI Text-to-Image (start here)

#### Highest quality
```python
# Stable Image Ultra: final assets, best text rendering
generate_image_ultra(
    prompt="A weathered brass compass on an antique nautical chart, macro photo",
    aspect_ratio="3:2",
    output_format="png"   # png or jpeg only; webp is not supported
)
```

#### Fastest draft
```python
# Stable Image Core: quick concepts to iterate on
generate_image_core(
    prompt="Three flat vector logo concepts for a coffee shop",
    aspect_ratio="1:1"
)
```

#### Balanced default
```python
generate_image_sd35(
    prompt="A serene mountain landscape at sunset",
    aspect_ratio="1:1"
)
```

#### Text-to-Image with Long Prompt
```python
# SD3.5 supports up to 10,000 character prompts
generate_image_sd35(
    prompt="A detailed cyberpunk cityscape at night with neon signs, flying cars, holographic advertisements, rain-slicked streets reflecting colorful lights, towering skyscrapers with intricate architectural details, bustling crowds of people with futuristic fashion, street vendors with glowing food stalls, and a massive digital billboard displaying animated content",
    aspect_ratio="16:9",
    negative_prompt="blurry, low quality, distorted",
    seed=42
)
```

#### Image-to-Image Transformation
```python
# Transform an existing image
transform_image_sd35(
    prompt="Transform into a watercolor painting style",
    image="/path/to/image.jpg",
    strength=0.7,
    aspect_ratio="1:1"
)
```

### Amazon Nova Canvas

Use these when you need exact pixel dimensions, a color palette, or multiple images per request.

#### Text-to-Image with Explicit Dimensions
```python
generate_image(
    prompt="A serene mountain landscape at sunset",
    width=1024,
    height=1024
)
```

#### Color-Guided Generation
```python
# Generate with specific color palette
generate_image_with_colors(
    prompt="A modern living room interior",
    colors=["#2C3E50", "#ECF0F1", "#E74C3C"],
    width=1280,
    height=720
)
```

### Stability AI Upscale Services

#### Creative Upscaling
```python
# Upscale with AI enhancement
upscale_creative(
    image="/path/to/low_res_image.jpg",
    prompt="A professional portrait photograph",
    creativity=0.3,
    style_preset="photographic"
)
```

#### Conservative Upscaling
```python
# Upscale preserving original details
upscale_conservative(
    image="/path/to/image.jpg",
    prompt="Product photography"
)
```

#### Fast Upscaling
```python
# Quick 4x upscale
upscale_fast(
    image="/path/to/image.jpg"
)
```

### Stability AI Edit Services

#### Inpainting
```python
# Fill masked region
inpaint_image(
    image="/path/to/image.jpg",
    mask="/path/to/mask.png",
    prompt="A red sports car",
    grow_mask=5
)
```

#### Outpainting
```python
# Extend image boundaries
outpaint_image(
    image="/path/to/image.jpg",
    prompt="Continue the landscape",
    left=500,
    right=500,
    creativity=0.5
)
```

#### Search and Replace
```python
# Replace objects without manual masking
search_and_replace(
    image="/path/to/image.jpg",
    search_prompt="old wooden chair",
    prompt="modern leather armchair"
)
```

#### Search and Recolor
```python
# Recolor specific objects
search_and_recolor(
    image="/path/to/image.jpg",
    select_prompt="the car",
    prompt="bright red color"
)
```

#### Remove Object
```python
# Remove unwanted objects
remove_object(
    image="/path/to/image.jpg",
    mask="/path/to/object_mask.png"
)
```

#### Remove Background
```python
# Automatic background removal
remove_background(
    image="/path/to/image.jpg"
)
```

### Stability AI Control Services

#### Sketch to Image
```python
# Convert sketch to detailed image
sketch_to_image(
    sketch="/path/to/sketch.jpg",
    prompt="A realistic portrait of a person",
    control_strength=0.7
)
```

#### Structure Control
```python
# Generate from structural guide
structure_control(
    control_image="/path/to/edge_map.jpg",
    prompt="A modern building facade",
    control_strength=0.8
)
```

#### Style Guide
```python
# Match reference style
style_guide(
    reference_image="/path/to/style_ref.jpg",
    prompt="A mountain landscape",
    fidelity=0.5
)
```

#### Style Transfer
```python
# Transfer style with fine control
style_transfer(
    init_image="/path/to/content.jpg",
    style_image="/path/to/style.jpg",
    prompt="Apply artistic style",
    composition_fidelity=0.9,
    style_strength=1.0,
    change_strength=0.9
)
```

## AWS IAM Permissions

Your AWS IAM user or role needs the following permissions to use this MCP server:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel"
            ],
            "Resource": [
                "arn:aws:bedrock:*::foundation-model/amazon.nova-canvas-v1:0",
                "arn:aws:bedrock:*::foundation-model/stability.sd3-5-large-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-creative-upscale-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-conservative-upscale-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-fast-upscale-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-inpaint-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-outpaint-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-search-replace-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-search-recolor-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-erase-object-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-remove-background-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-control-sketch-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-control-structure-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-image-style-guide-v1:0",
                "arn:aws:bedrock:*::foundation-model/us.stability.stable-style-transfer-v1:0"
            ]
        }
    ]
}
```

## Supported AWS Regions

Region coverage differs sharply between the model families, and **no single region runs all of
them**. Pick your `AWS_REGION` based on which tools you need.

| Tools | Regions | Lifecycle |
|---|---|---|
| `generate_image_ultra`, `generate_image_core`, `generate_image_sd35`, `transform_image_sd35` | **us-west-2 only** | Active |
| The 13 Stability AI upscale / edit / control tools | us-east-1, us-east-2, us-west-2 | Active |
| `create_rectangular_mask`, `create_ellipse_mask`, `create_full_mask` | Any — these run locally and never call Bedrock | Active |
| `generate_image`, `generate_image_with_colors` (Nova Canvas) | us-east-1, eu-west-1, ap-northeast-1 | **Legacy — EOL 2026-09-30** |

Practical consequences:

- **us-west-2** is the default, and the only region where all 20 non-Nova tools work: the four
  SD3.5/Ultra/Core tools, all 13 Stability edit/upscale/control tools, and the 3 local mask
  helpers. Nova Canvas is the only thing missing.
- **us-east-1 and us-east-2** are strict subsets — identical to each other in image-model
  coverage, and missing all four text-to-image tools. us-east-1 adds only Nova Canvas, which
  retires 2026-09-30; after that it offers nothing us-west-2 does not.
- If you need both SD3.5 and Nova Canvas before the EOL date, run two server instances with
  different `AWS_REGION` values.

The Stability AI tools are invoked through US Geo cross-region inference profiles (their model
IDs carry a `us.` prefix), so a request sent to any of the three regions may be served from
another one. The underlying in-region model IDs are not enabled for direct on-demand use.

### Nova Canvas is retiring

AWS moved Nova Canvas to **Legacy on 2026-03-30, with end-of-life on 2026-09-30**. After that
date the two Nova tools will stop working. AWS also restricts Legacy models in ways that bite
before then:

- New customers cannot start using a Legacy model at all
- Existing customers **may lose access after 15 days of inactivity**, which surfaces as
  `ResourceNotFoundException` (see [Troubleshooting](#this-model-is-marked-by-provider-as-legacy-nova-canvas))

If you rely on Nova Canvas today, plan to move to `generate_image_sd35` in us-west-2.

**Note**: verified against both the Bedrock API (`GetFoundationModel` lifecycle status) and the
AWS model cards. Availability changes, so check your own region with:

```bash
aws bedrock list-foundation-models --region us-west-2 \
  --query "modelSummaries[?contains(modelId,'stability') || contains(modelId,'nova-canvas')].[modelId,modelLifecycle.status]"
```

See the AWS [regional availability by model](https://docs.aws.amazon.com/bedrock/latest/userguide/models-region-compatibility.html)
and [model lifecycle](https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html)
pages for the authoritative lists.

## Troubleshooting

### Common Issues

#### "Model not found" or "Access denied" errors

**Problem**: You receive errors indicating the model is not available or you don't have access.

**Solutions**:
1. Verify your AWS region supports the model you're trying to use (see [Supported AWS Regions](#supported-aws-regions)).
   `The provided model identifier is invalid` almost always means the model is not in your
   region — most often SD3.5, which is us-west-2 only.
2. Ensure you've requested model access in the AWS Bedrock console:
   - Go to AWS Bedrock console → Model access
   - Request access for the models you want to use
   - Wait for approval (usually instant for most models)
3. Verify your IAM permissions include `bedrock:InvokeModel` for the specific model ARN

#### "This Model is marked by provider as Legacy" (Nova Canvas)

**Problem**: `generate_image` or `generate_image_with_colors` fails with
`ResourceNotFoundException: Access denied. This Model is marked by provider as Legacy and you
have not been actively using the model in the last 30 days.`

**Cause**: AWS moved Nova Canvas to Legacy on 2026-03-30, with **end-of-life on 2026-09-30**.
Per the AWS model lifecycle policy, existing customers may lose access to a Legacy model after
**15 days of inactivity**, and new customers cannot use it at all. Previously granted model
access does not exempt you.

**Solutions**:
- Prefer `generate_image_sd35` in us-west-2. This is the recommended text-to-image tool and is
  Active, so it is the migration path rather than a workaround.
- To keep using Nova Canvas before EOL, re-request access in the Bedrock console and invoke it
  at least once every 15 days.
- Note that after 2026-09-30 the two Nova tools will stop working regardless.

#### "Response payload size exceeds limit" (all three upscale tools)

**Problem**: an upscale tool fails with
`{"detail":"Response payload size NNNNNNNN bytes exceeds the maximum allowed size of 16777216 bytes"}`.

**Cause**: this is a limit on output *size*, not on format support. Bedrock returns the image
base64-encoded inside the JSON response and caps that response at 16MB, so the practical
ceiling is roughly a 12MB image. Ordinary generation is ~2MB and never comes close; upscaling
returns 3K-4K images, where a PNG is 20-35MB. It is an API limit, not a bug in this server.

PNG is supported and remains the default for every tool. The catch is that it is the default
for the case most likely to exceed the cap — a full-size upscale.

**Solution**: request `output_format="jpeg"` or `"webp"` for full-size upscales. Measured on a
1MP input:

| Tool | Output | PNG | JPEG | WebP |
| --- | --- | --- | --- | --- |
| `upscale_fast` | 4096x4096 | 34.8MB — fails | 4.1MB | 2.7MB |
| `upscale_creative` | ~3152x3152 | fails at every input size | 1.9MB | 0.9MB |
| `upscale_conservative` | ~3112x3112 | 20.1MB — fails | 2.4MB | 1.6MB |

PNG does work when the output is small enough, because for `upscale_fast` and
`upscale_conservative` the output scales with the input — a 256x256 input gave a 1.4MB PNG from
fast and a 9.9MB PNG from conservative, and 512x512 gave a 5.9MB PNG from fast. `upscale_creative`
is the exception: its output is a fixed ~3150x3150 whatever you feed it, so no input size makes
PNG work there.

#### "Invalid image dimensions" errors

**Problem**: Image generation fails with dimension validation errors.

**Solutions**:
- **Nova Canvas**: Ensure dimensions are between 320-4096 pixels and divisible by 16
- **SD3.5**: Use one of the supported aspect ratios (16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21)
- **Upscale services**: Check input image size constraints:
  - Creative/Fast: 64x64 to 1MP
  - Conservative: 64x64 to 9.4MP

#### "Content filtered" responses

**Problem**: Your generated image is blocked by content filtering.

**Solutions**:
1. Review your prompt for potentially sensitive content
2. Use negative prompts to exclude problematic elements
3. Adjust your prompt to be more specific and less ambiguous
4. Try different seed values

#### Mask validation errors (Inpainting/Remove Object)

**Problem**: Mask image is rejected during inpainting or object removal.

**Solutions**:
1. Ensure mask is grayscale (not RGB or RGBA)
2. Verify mask dimensions exactly match the input image
3. Use white (255) for areas to fill/remove, black (0) for areas to preserve
4. Save mask as PNG or JPEG format

#### "Image too large" warnings (Upscaling)

**Problem**: Warning about input image being too large for creative upscaling.

**Solutions**:
1. Use `upscale_conservative` instead for larger images (up to 9.4MP) — it is the only one of
   the three that accepts inputs above 1MP
2. Resize your input image to under 1MP before creative upscaling
3. Note that `upscale_fast` has the same 1MP input cap as creative upscaling, so it is not a
   workaround for an oversized input

#### AWS credentials not found

**Problem**: Server fails to start with AWS credential errors.

**Solutions**:
1. Run `aws configure` to set up your credentials
2. Set `AWS_PROFILE` environment variable to your profile name
3. Verify credentials file exists at `~/.aws/credentials`
4. For temporary credentials, ensure `AWS_SESSION_TOKEN` is also set

#### Slow image generation

**Problem**: Image generation takes longer than expected.

**Solutions**:
1. This is normal - AI image generation can take 10-60 seconds depending on:
   - Model complexity (SD3.5 and upscaling are slower)
   - Image resolution
   - AWS region latency
2. Use `upscale_fast` instead of creative upscaling for faster results
3. Consider using a closer AWS region
4. For Nova Canvas, reduce `number_of_images` parameter

#### File path issues

**Problem**: Images not found or saved to unexpected locations.

**Solutions**:
1. Use absolute file paths for input images
2. Specify `workspace_dir` parameter to control output location
3. Check that output directory has write permissions
4. Verify input image files exist and are readable

### Getting Help

If you encounter issues not covered here:

1. Check the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/)
2. Review the [Model Context Protocol specification](https://modelcontextprotocol.io/)
3. Open an issue on the [GitHub repository](https://github.com/awslabs/mcp)
4. Check AWS service health dashboard for outages

## Development

### Running Tests

```bash
# Install dependencies
uv sync --dev

# Run all tests
pytest

# Run with coverage
pytest --cov=awslabs --cov-report=html

# Run specific test file
pytest tests/test_server.py
```

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type check
pyright
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING](CONTRIBUTING.md) for guidelines.
