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
"""Tests for the Stable Image Ultra and Core service of the bedrock-image-mcp-server."""

import pytest
from awslabs.bedrock_image_mcp_server.consts import (
    STABLE_IMAGE_CORE_MODEL_ID,
    STABLE_IMAGE_ULTRA_MODEL_ID,
)
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.sd35_models import AspectRatio
from awslabs.bedrock_image_mcp_server.models.stable_image_models import (
    StableImageOutputFormat,
    StableImageParams,
)
from awslabs.bedrock_image_mcp_server.services import stable_image_service
from awslabs.bedrock_image_mcp_server.services.stable_image_service import (
    build_stable_image_request,
    generate_image_core,
    generate_image_ultra,
)
from pydantic import ValidationError
from unittest.mock import MagicMock, patch


class TestBuildStableImageRequest:
    """Tests for the Stable Image request body builder."""

    def test_includes_every_supported_field(self):
        """Test the request carries prompt, aspect ratio, seed, format and negative prompt."""
        params = StableImageParams(
            prompt='a brass compass',
            aspect_ratio=AspectRatio.RATIO_3_2,
            seed=99,
            negative_prompt='blurry',
            output_format=StableImageOutputFormat.JPEG,
        )

        request = build_stable_image_request(params)

        assert request == {
            'prompt': 'a brass compass',
            'aspect_ratio': '3:2',
            'seed': 99,
            'output_format': 'jpeg',
            'negative_prompt': 'blurry',
        }

    def test_omits_negative_prompt_when_absent(self):
        """Test negative_prompt is left out rather than sent as null."""
        request = build_stable_image_request(StableImageParams(prompt='a compass'))

        assert 'negative_prompt' not in request

    def test_never_sends_fields_the_models_reject(self):
        """Test the body excludes fields Bedrock rejects for these models."""
        request = build_stable_image_request(StableImageParams(prompt='a compass'))

        # Verified against the live API: these all raise ValidationException.
        for field in ('mode', 'image', 'strength', 'style_preset', 'width', 'height'):
            assert field not in request


class TestStableImageParams:
    """Tests for validation of the shared Stable Image parameters."""

    def test_defaults(self):
        """Test the documented defaults are applied."""
        params = StableImageParams(prompt='a compass')

        assert params.aspect_ratio is AspectRatio.RATIO_1_1
        assert params.seed == 0
        assert params.negative_prompt is None
        assert params.output_format is StableImageOutputFormat.PNG

    def test_webp_is_rejected(self):
        """Test webp is refused, since these models do not accept it."""
        with pytest.raises(ValidationError):
            StableImageParams.model_validate({'prompt': 'a compass', 'output_format': 'webp'})

    @pytest.mark.parametrize('fmt', ['png', 'jpeg'])
    def test_supported_formats_are_accepted(self, fmt):
        """Test both supported output formats validate."""
        assert StableImageParams(prompt='a compass', output_format=fmt).output_format.value == fmt

    def test_empty_prompt_is_rejected(self):
        """Test an empty prompt fails validation."""
        with pytest.raises(ValidationError):
            StableImageParams(prompt='')

    def test_seed_upper_bound_is_enforced(self):
        """Test a seed above the documented maximum fails validation."""
        with pytest.raises(ValidationError):
            StableImageParams(prompt='a compass', seed=4294967295)


class TestGenerateImageUltraAndCore:
    """Tests for the Ultra and Core service functions."""

    @pytest.mark.parametrize(
        ('func', 'model_id', 'prefix'),
        [
            (generate_image_ultra, STABLE_IMAGE_ULTRA_MODEL_ID, 'stable_image_ultra'),
            (generate_image_core, STABLE_IMAGE_CORE_MODEL_ID, 'stable_image_core'),
        ],
    )
    async def test_invokes_the_right_model_and_saves(self, func, model_id, prefix, tmp_path):
        """Test each function targets its own model ID and default filename prefix."""
        captured = {}

        async def fake_invoke(model_id, request_body, bedrock_client):
            captured['model_id'] = model_id
            captured['request_body'] = request_body
            return {'images': ['aW1n'], 'seeds': [7], 'finish_reasons': [None]}

        def fake_save(**kwargs):
            captured['save'] = kwargs
            return ['/out/image.png']

        with (
            patch.object(stable_image_service, 'invoke_bedrock_model', fake_invoke),
            patch.object(stable_image_service, 'save_images', fake_save),
        ):
            response = await func(
                params=StableImageParams(prompt='a compass', aspect_ratio=AspectRatio.RATIO_16_9),
                bedrock_client=MagicMock(),
                workspace_dir=str(tmp_path),
            )

        assert response.status == 'success'
        assert response.paths == ['/out/image.png']
        assert response.model_id == model_id
        assert response.metadata['aspect_ratio'] == '16:9'
        assert captured['model_id'] == model_id
        assert captured['request_body']['aspect_ratio'] == '16:9'
        assert captured['save']['filename_prefix'] == prefix
        assert captured['save']['output_format'] is OutputFormat.PNG

    async def test_custom_filename_overrides_the_prefix(self, tmp_path):
        """Test a caller-supplied filename is used instead of the default prefix."""
        captured = {}

        async def fake_invoke(model_id, request_body, bedrock_client):
            return {'images': ['aW1n'], 'finish_reasons': [None]}

        def fake_save(**kwargs):
            captured.update(kwargs)
            return ['/out/mine.png']

        with (
            patch.object(stable_image_service, 'invoke_bedrock_model', fake_invoke),
            patch.object(stable_image_service, 'save_images', fake_save),
        ):
            await generate_image_ultra(
                params=StableImageParams(prompt='a compass'),
                bedrock_client=MagicMock(),
                workspace_dir=str(tmp_path),
                filename='mine',
            )

        assert captured['filename_prefix'] == 'mine'

    async def test_jpeg_reaches_the_saver_as_jpeg(self, tmp_path):
        """Test the narrower format enum is translated for the shared saver."""
        captured = {}

        async def fake_invoke(model_id, request_body, bedrock_client):
            captured['request_body'] = request_body
            return {'images': ['aW1n'], 'finish_reasons': [None]}

        def fake_save(**kwargs):
            captured['save'] = kwargs
            return ['/out/image.jpg']

        with (
            patch.object(stable_image_service, 'invoke_bedrock_model', fake_invoke),
            patch.object(stable_image_service, 'save_images', fake_save),
        ):
            await generate_image_core(
                params=StableImageParams(
                    prompt='a compass', output_format=StableImageOutputFormat.JPEG
                ),
                bedrock_client=MagicMock(),
                workspace_dir=str(tmp_path),
            )

        assert captured['request_body']['output_format'] == 'jpeg'
        assert captured['save']['output_format'] is OutputFormat.JPEG

    async def test_empty_image_list_returns_an_error_response(self, tmp_path):
        """Test a response with no images produces an error rather than a false success."""

        async def fake_invoke(model_id, request_body, bedrock_client):
            return {'images': []}

        with patch.object(stable_image_service, 'invoke_bedrock_model', fake_invoke):
            response = await generate_image_ultra(
                params=StableImageParams(prompt='a compass'),
                bedrock_client=MagicMock(),
                workspace_dir=str(tmp_path),
            )

        assert response.status == 'error'
        assert response.paths == []
