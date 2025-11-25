# Amazon Bedrock Image Generation MCP Server

[![PyPI version](https://badge.fury.io/py/bedrock-image-mcp-server.svg)](https://badge.fury.io/py/bedrock-image-mcp-server)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Actions](https://github.com/kalleeh/bedrock-image-mcp-server/workflows/Publish%20to%20PyPI/badge.svg)](https://github.com/kalleeh/bedrock-image-mcp-server/actions)

> **Note:** This is a community-maintained fork of [awslabs/mcp/bedrock-image-mcp-server](https://github.com/awslabs/mcp) with additional features and improvements. Original work by Amazon Web Services under Apache 2.0 license.

MCP server for generating and editing images using Amazon Nova Canvas, Stable Diffusion 3.5 Large, and Stability AI Image Services through Amazon Bedrock.

## Features

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

### Stable Diffusion 3.5 Large (2 tools)

#### Text-to-image generation
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
| [![Install MCP Server](https://cursor.com/deeplink/mcp-install-light.svg)](https://cursor.com/en/install-mcp?name=bedrock-image-mcp-server&config=eyJjb21tYW5kIjoidXZ4IGJlZHJvY2staW1hZ2UtbWNwLXNlcnZlckBsYXRlc3QiLCJlbnYiOnsiQVdTX1BST0ZJTEUiOiJ5b3VyLWF3cy1wcm9maWxlIiwiQVdTX1JFR0lPTiI6InVzLWVhc3QtMSIsIkZBU1RNQ1BfTE9HX0xFVkVMIjoiRVJST1IifSwiZGlzYWJsZWQiOmZhbHNlLCJhdXRvQXBwcm92ZSI6W119) | [![Install on VS Code](https://img.shields.io/badge/Install_on-VS_Code-FF9900?style=flat-square&logo=visualstudiocode&logoColor=white)](https://insiders.vscode.dev/redirect/mcp/install?name=Bedrock%20Image%20MCP%20Server&config=%7B%22command%22%3A%22uvx%22%2C%22args%22%3A%5B%22bedrock-image-mcp-server%40latest%22%5D%2C%22env%22%3A%7B%22AWS_PROFILE%22%3A%22your-aws-profile%22%2C%22AWS_REGION%22%3A%22us-east-1%22%2C%22FASTMCP_LOG_LEVEL%22%3A%22ERROR%22%7D%2C%22disabled%22%3Afalse%2C%22autoApprove%22%3A%5B%5D%7D) |

Configure the MCP server in your MCP client configuration (e.g., for Amazon Q Developer CLI, edit `~/.aws/amazonq/mcp.json`):

```json
{
  "mcpServers": {
    "bedrock-image-mcp-server": {
      "command": "uvx",
      "args": ["bedrock-image-mcp-server@latest"],
      "env": {
        "AWS_PROFILE": "your-aws-profile",
        "AWS_REGION": "us-east-1",
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
        "AWS_REGION": "us-east-1"
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
          "AWS_REGION=us-east-1",
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
  "AWS_REGION": "us-east-1"
}
```

Make sure the AWS profile has permissions to access Amazon Bedrock and the image generation models. The MCP server creates a boto3 session using the specified profile to authenticate with AWS services. Your AWS IAM credentials remain on your local machine and are strictly used for using the Amazon Bedrock model APIs.

## Usage Examples

### Amazon Nova Canvas

#### Basic Text-to-Image
```python
# Generate a simple image
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

### Stable Diffusion 3.5 Large

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

The following AWS regions support Amazon Bedrock with the image generation models used by this server:

### Amazon Nova Canvas
- us-east-1 (US East - N. Virginia)
- us-west-2 (US West - Oregon)
- eu-west-1 (Europe - Ireland)
- ap-southeast-1 (Asia Pacific - Singapore)
- ap-northeast-1 (Asia Pacific - Tokyo)

### Stable Diffusion 3.5 Large
- us-east-1 (US East - N. Virginia)
- us-west-2 (US West - Oregon)
- eu-west-1 (Europe - Ireland)
- eu-central-1 (Europe - Frankfurt)
- ap-southeast-1 (Asia Pacific - Singapore)
- ap-northeast-1 (Asia Pacific - Tokyo)

### Stability AI Image Services
- us-east-1 (US East - N. Virginia)
- us-west-2 (US West - Oregon)
- eu-west-1 (Europe - Ireland)
- eu-central-1 (Europe - Frankfurt)
- ap-southeast-1 (Asia Pacific - Singapore)
- ap-northeast-1 (Asia Pacific - Tokyo)

**Note**: Model availability may change. Check the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html) for the most current information.

## Troubleshooting

### Common Issues

#### "Model not found" or "Access denied" errors

**Problem**: You receive errors indicating the model is not available or you don't have access.

**Solutions**:
1. Verify your AWS region supports the model you're trying to use (see [Supported AWS Regions](#supported-aws-regions))
2. Ensure you've requested model access in the AWS Bedrock console:
   - Go to AWS Bedrock console → Model access
   - Request access for the models you want to use
   - Wait for approval (usually instant for most models)
3. Verify your IAM permissions include `bedrock:InvokeModel` for the specific model ARN

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
1. Use `upscale_conservative` instead for larger images (up to 9.4MP)
2. Resize your input image to under 1MP before creative upscaling
3. Use `upscale_fast` for quick 4x upscaling without size restrictions

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
