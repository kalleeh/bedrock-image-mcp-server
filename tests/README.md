# Bedrock Image MCP Server Tests

This directory contains tests for the Bedrock Image MCP Server, which provides tools for
generating and editing images using Amazon Nova Canvas, Stable Diffusion 3.5 Large, and
Stability AI Image Services through Amazon Bedrock.

## Test Structure

- `conftest.py`: pytest fixtures shared across the suite
- `test_models.py`: Nova Canvas request/response models
- `test_sd35_models.py`: SD3.5 parameter models and validators
- `test_bedrock_common.py`: shared Bedrock invocation, error classification, and image saving
- `test_novacanvas.py`: Nova Canvas service functions
- `test_sd35_service.py`: SD3.5 service functions
- `test_stability_upscale.py`: upscale service functions
- `test_stability_edit.py`: inpaint, outpaint, search/replace, recolor, and removal services
- `test_stability_control.py`: sketch, structure, style guide, and style transfer services
- `test_mask_utils.py`: mask builders and base64 decoding
- `test_server.py`: server wiring, tool registration, and validation helpers
- `test_server_tools.py`: invocation tests for every registered MCP tool

## Running Tests

```bash
uv run pytest
```

With a coverage report:

```bash
uv run pytest --cov=awslabs --cov-report=term-missing
```

The same checks CI runs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## Adding New Tests

1. Use the test file matching the module under test
2. Mock only at the seams (`invoke_bedrock_model`, `save_images`) and assert on the real
   request bodies and responses the code builds
3. Cover both success and error paths, including validation boundaries
4. Assert that a failing tool logs the failure exactly once (via the `logged_errors` fixture)
5. Use fixtures from `conftest.py`, and write real temp files rather than patching
   `os.path.exists` — patching it mutates the shared `os` module and leaks into other tests
6. Avoid assertions that only re-check a mock the test itself configured

`test_server_tools.py` drives every tool from a single `TOOL_KWARGS` table, so a new tool
inherits the shared contract tests by adding one entry.
