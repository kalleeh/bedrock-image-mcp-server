# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-25

### Changed
- Forked from [awslabs/mcp/bedrock-image-mcp-server](https://github.com/awslabs/mcp) v1.0.8
- Reset version to 0.1.0 to indicate community fork
- Updated licensing documentation to properly acknowledge original work
- Fixed test dependencies being in main dependencies (moved to dev group)

### Note
This is a community-maintained fork with additional features and improvements.
Original work by Amazon Web Services under Apache 2.0 license.

---

## Original awslabs/mcp Changelog

## [2.0.0] - 2025-11-24

### Added

#### Stable Diffusion 3.5 Large Support
- **generate_image_sd35**: Text-to-image generation with SD3.5 Large
  - Supports prompts up to 10,000 characters
  - 9 aspect ratio options (16:9, 1:1, 21:9, 2:3, 3:2, 4:5, 5:4, 9:16, 9:21)
  - Seed support for reproducible results (0-4,294,967,294)
  - Multiple output formats (JPEG, PNG, WebP)
- **transform_image_sd35**: Image-to-image transformation with SD3.5
  - Strength parameter (0.0-1.0) for transformation control
  - Supports file paths and base64 image inputs
  - All text-to-image parameters available

#### Stability AI Upscale Services (3 tools)
- **upscale_creative**: Creative 4K upscaling with AI enhancement
  - 20-40x upscale from 64x64 to 1MP inputs
  - Creativity parameter (0.1-0.5) for enhancement control
  - Style preset support for specific aesthetics
- **upscale_conservative**: Detail-preserving 4K upscaling
  - Supports inputs up to 9.4 megapixels
  - Minimal alterations to original image
- **upscale_fast**: Quick 4x upscaling
  - Fast processing for quick resolution increases
  - Supports inputs from 32x32 to 1MP

#### Stability AI Edit Services (6 tools)
- **inpaint_image**: Generative fill for masked regions
  - Grayscale mask support (white=fill, black=preserve)
  - grow_mask parameter (0-20) for edge blending
- **outpaint_image**: Extend images beyond boundaries
  - Directional expansion (left, right, up, down)
  - Up to 2000 pixels per direction
  - Creativity parameter for extension style
- **search_and_replace**: Text-based object replacement
  - Automatic object detection and masking
  - No manual mask required
- **search_and_recolor**: Text-based object recoloring
  - Preserves structure while changing colors
  - Maintains image quality
- **remove_object**: Intelligent object removal
  - Context-aware filling of removed areas
  - Seamless blending with surroundings
- **remove_background**: Automatic background removal
  - Returns PNG with transparency
  - Handles complex subjects (hair, transparent objects)
  - No prompt required

#### Stability AI Control Services (4 tools)
- **sketch_to_image**: Convert sketches to detailed images
  - control_strength parameter (0.0-1.0)
  - Preserves sketch structure while adding detail
- **structure_control**: Generate images from structural guides
  - Follows edge maps and structural guidance
  - control_strength for adherence level
- **style_guide**: Match reference image style
  - fidelity parameter (0.0-1.0) for style matching
  - Accepts prompts for content description
- **style_transfer**: Transfer style between images
  - Fine-grained control with composition_fidelity, style_strength, change_strength
  - Requires init_image (content) and style_image (style reference)

#### Infrastructure Improvements
- Refactored codebase into modular structure (models/, services/, utils/)
- Unified Bedrock invocation with `bedrock_common.py`
- Comprehensive Pydantic validation for all parameters
- Enhanced error handling and user-friendly messages
- Content filtering detection and reporting
- Image dimension validation utilities
- Support for multiple output formats across all services

### Changed
- Reorganized project structure for better maintainability
- Moved existing Nova Canvas models to `models/nova_models.py`
- Generalized image saving and Bedrock invocation functions
- Enhanced server instructions with comprehensive best practices
- Updated documentation with all new capabilities

### Backward Compatibility
- **Zero breaking changes**: All existing Nova Canvas functionality preserved
- Existing test suite passes without modification
- Existing MCP client integrations continue to work
- Configuration format unchanged

## [1.0.0] - 2025-05-26

### Removed

- **BREAKING CHANGE:** Server Sent Events (SSE) support has been removed in accordance with the Model Context Protocol specification's [backwards compatibility guidelines](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#backwards-compatibility)
- This change prepares for future support of [Streamable HTTP](https://modelcontextprotocol.io/specification/draft/basic/transports#streamable-http) transport

## v0.1.5 (2025-03-30)

### Fix

- **version**

## v0.1.4 (2025-03-30)

### Fix

- **version**

## v0.1.3 (2025-03-30)

### Fix

- pyproject.toml

## v0.1.2 (2025-03-30)

### Fix

- uv package
- release

## v0.1.1 (2025-03-30)

### Fix

- release

## v0.1.0 (2025-03-30)

### Feat

- MCP server for generating images with Amazon Nova Canvas
- **doc**: material mkdocs (#5)
- **doc**: initial documentation (#4)
- **security**: add CODEOWNERS (#2)
- **cicd**: add github workflows (#1)

### Fix

- pyright errors on  overrides
- optional fields
