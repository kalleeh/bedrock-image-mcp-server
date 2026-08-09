# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-08-09

Reported and correctly diagnosed by [@aleck31](https://github.com/aleck31) in
[#5](https://github.com/kalleeh/bedrock-image-mcp-server/issues/5), root cause and workaround
included.

### Fixed
- **Every fresh install was broken.** `mcp 2.0.0` (published 2026-07-28, three days after
  0.3.2) removed `mcp.server.fastmcp` — `FastMCP` is now `MCPServer` under
  `mcp.server.mcpserver`. Because the dependency was declared `mcp[cli]>=1.11.0` with no
  upper bound, `uvx bedrock-image-mcp-server@latest` resolved `mcp` to 2.x and crashed at
  import with `ModuleNotFoundError: No module named 'mcp.server.fastmcp'` before the server
  could speak protocol — surfacing in MCP clients as an opaque transport error (`-32000`).
  Existing users with a warm `uv` cache or a lockfile were unaffected, which made it look
  intermittent. The requirement is now `mcp[cli]>=1.11.0,<2`; a fresh install resolves
  `mcp` 1.29.0.

### Added
- A `fresh-install` CI job that resolves dependencies **unlocked** from the built wheel and
  boots the server over stdio. The existing `test` job installs with `uv sync --frozen`,
  i.e. from `uv.lock`, so it stayed green throughout this outage and structurally could not
  have caught it — real users install unlocked.
- Regression tests (`tests/test_dependency_bounds.py`) asserting the `<2` ceiling is present
  and that it agrees with the API `server.py` actually imports, so the ceiling cannot be
  dropped without porting to `mcp.server.mcpserver.MCPServer`.

## [0.3.2] - 2026-07-25

### Fixed
- Every configuration example specified `AWS_REGION: us-east-1`, where none of the recommended
  text-to-image tools work. SD3.5, Ultra and Core are us-west-2 only, so `generate_image_sd35`,
  `generate_image_ultra`, `generate_image_core` and `transform_image_sd35` all failed with an
  invalid model identifier for anyone who copied the docs. The examples now use us-west-2, which
  also serves all 13 Stability edit, upscale and control tools.
- Nova Canvas is not available in us-west-2. Added a note pointing at the region table for anyone
  who needs it, along with the reminder that it retires 2026-09-30.

## [0.3.1] - 2026-07-25

### Changed
- Tool descriptions and documentation now carry Stability AI's own "ideal for" guidance for each
  text-to-image model, rather than a quality ranking:
  - `generate_image_ultra` — "Photorealistic, Large-Scale Output". Ideal for "ultra-realistic
    imagery for luxury brands and high-end campaigns" and "professional print media and large
    format applications". The only one of the three Stability credits with typography.
  - `generate_image_sd35` — "High-Quality, High-Quantity Creative Assets". Ideal for
    "high-volume outputs like marketing campaigns and digital assets".
  - `generate_image_core` — "Fast and Affordable". Ideal for "rapid content generation at scale"
    and "rapidly iterating on concepts during ideation".
- Documents Stability's credit cost per image (Core 3, SD3.5 Large 6.5, Ultra 8) as a cost ratio
  rather than a Bedrock price, since AWS does not publish per-image rates for these models.
- Quotes are taken from Stability's Bedrock launch post and their API specification at
  `api.stability.ai/v2alpha/openapi`, so the guidance is attributable rather than inferred.

### Security
- Added a decode limit for untrusted images and a size cap on generated masks, so oversized input
  can no longer exhaust memory.
- Base64 image data is now validated rather than silently truncated on corrupt input.
- `ResourceNotFoundException` now explains the likely cause instead of surfacing the raw AWS
  message: either the Nova Canvas Legacy retirement, naming the replacement tool, or which
  regions carry the requested model family.

### Changed
- Blocking Bedrock calls, image decoding and image writing now run on worker threads, keeping the
  event loop responsive during the 30-90s a generation takes.
- Deduplicated the Stability service layer (-465 lines) behind shared helpers; the request bodies
  sent to Bedrock are unchanged.
- Documentation and tool descriptions now recommend `generate_image_sd35` for general
  text-to-image work, with Nova Canvas positioned for its specific features.
- Added a CI workflow running lint, format, type checks and tests on pull requests.

### Removed
- `BaseImageInput` and the `CommonImageGenerationResponse` alias from
  `awslabs.bedrock_image_mcp_server.models` (both unused).
- The deprecated `models.py` and `novacanvas.py` compatibility shims, which re-exported module
  paths that never existed in this fork.
- `DEFAULT_CONSERVATIVE_UPSCALE_CREATIVITY` (the API accepts no such parameter) and the unused
  `image` field alias on the control parameter models.

### Note
All 20 MCP tools and every tool parameter are unchanged in this release; the removals above affect
only Python-level imports, not the MCP interface.

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
