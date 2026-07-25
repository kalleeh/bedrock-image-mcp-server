# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-07-25

### Changed
- Tool descriptions and documentation now say what each text-to-image model is *for*, not just
  that one is "highest quality", quoting Stability AI's own positioning:
  - `generate_image_ultra` — "professional print media and large format applications" and
    "luxury brands and high-end campaigns". The only one of the three Stability credits with
    typography. Use for a single high-value asset.
  - `generate_image_sd35` — "professional use cases at 1 megapixel resolution" and
    "high-volume, high-quality digital assets like websites, newsletters, and marketing
    materials". Use when producing many assets.
  - `generate_image_core` — "rapidly iterating on concepts during ideation". Drafts and bulk
    work, not client-facing deliverables.
- Documents the relative cost per image as Stability's credit rates (Core 3, SD3.5 Large 6.5,
  Ultra 8), noting they are a ratio rather than a Bedrock price.
- Quotes are verified against Stability's own API specification at
  `api.stability.ai/v2alpha/openapi` rather than paraphrased, since their documentation site is
  JavaScript-rendered and not directly readable.
- Corrects the split between the two professional tiers. Both Stability and AWS call SD3.5 Large
  "ideal for professional use cases", which reads as competing with Ultra, but Stability's own
  wording separates them by job: Ultra for print and large format, SD3.5 Large for high-volume
  asset production. Only Ultra carries their typography claim, so the earlier note crediting
  SD3.5 with text quality is now attributed to AWS specifically.

## [0.3.0] - 2026-07-25

### Added
- **`generate_image_ultra`** (Stable Image Ultra) and **`generate_image_core`** (Stable Image
  Core), completing a text-to-image quality ladder: Core for drafts, SD3.5 as the balanced
  default, Ultra for final assets and legible text. Both were verified against live Bedrock.
  - Available in us-west-2 only, alongside SD3.5.
  - Text-to-image only. Neither accepts `mode`, `image` or `strength`; use
    `transform_image_sd35` for image-to-image work.
  - `output_format` accepts `png` or `jpeg` only. These two models reject `webp`, unlike every
    other tool here, so they use a narrower format enum and give a specific error message.
  - Neither supports `width`/`height`, `cfg_scale`, `number_of_images` or `style_preset`; the
    Bedrock API rejects those fields. Use `aspect_ratio` to control the shape.

### Changed
- Documentation and tool descriptions now present the three-model ladder rather than naming a
  single preferred text-to-image tool.

## [0.2.0] - 2026-07-25

### Fixed
- **Path traversal**: a caller-supplied `filename` could escape the workspace output directory
  (`../x` wrote to the parent, an absolute path ignored `workspace_dir` entirely). Filenames are
  now reduced to a safe basename and every write is confined to the output directory.
- **Crash on prompts containing braces**: any prompt with `{...}` raised `KeyError` before the
  request was sent, because structured log calls made loguru treat the message as a format string.
- **Silent premium billing**: an unrecognised `quality` value (including `"Standard"`) resolved to
  `premium` instead of being rejected.
- Output format validation was skipped by `sketch_to_image`, `structure_control`, `style_guide`
  and `style_transfer`, so `output_format="PNG"` raised there while working on every other tool.
- Failures were reported to the MCP client twice, and service errors were flattened into a bare
  `Exception`, so callers could not distinguish a retryable throttle from a permanent failure.
- Nova Canvas accepted a `filename` argument and silently ignored it.
- Docker healthcheck looked for a process name from before the fork rename, so every container
  reported `unhealthy` forever.
- `__init__.py` was left at 0.1.0 when the project bumped to 0.1.1.

### Deprecated
- **`generate_image` and `generate_image_with_colors` (Amazon Nova Canvas).** AWS marked Nova
  Canvas as a Legacy model on 2026-03-30 and retires it on **2026-09-30**, after which both tools
  will stop working. AWS also revokes Legacy model access after 15 days of inactivity and blocks
  new customers entirely.
  - Use **`generate_image_sd35`** (Stable Diffusion 3.5 Large, us-west-2) instead. It has better
    prompt adherence and is an Active model.
  - `generate_image_with_colors` has no direct replacement; describe the desired colours in the
    prompt to `generate_image_sd35`.
  - Nova-only parameters with no SD3.5 equivalent: `width`, `height`, `quality`, `cfg_scale`,
    `number_of_images`, `style`. SD3.5 uses `aspect_ratio` and `output_format` instead.
  - Both tools still work unchanged in this release. Nothing has been renamed or repointed.

### Compatibility policy
`generate_image` will **not** be silently repointed at a different model. Six of its eleven
parameters have no SD3.5 equivalent, so swapping the model behind the existing name would accept
calls and then quietly ignore the dimensions, quality and image count the caller asked for. The
Nova tools instead keep their current behaviour until the AWS end-of-life date and will then be
removed in a major release. New capabilities arrive as new tool names.

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
