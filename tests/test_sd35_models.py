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
"""Tests for the SD3.5 models module of the nova-canvas-mcp-server."""

import base64
import pytest
from awslabs.bedrock_image_mcp_server.models.common import OutputFormat
from awslabs.bedrock_image_mcp_server.models.sd35_models import (
    AspectRatio,
    GenerationMode,
    SD35ImageToImageParams,
    SD35TextToImageParams,
)
from io import BytesIO
from PIL import Image
from pydantic import ValidationError


class TestEnums:
    """Tests for the enum classes."""

    def test_aspect_ratio_enum(self):
        """Test that AspectRatio enum has the expected values."""
        assert AspectRatio.RATIO_16_9 == "16:9"
        assert AspectRatio.RATIO_1_1 == "1:1"
        assert AspectRatio.RATIO_21_9 == "21:9"
        assert AspectRatio.RATIO_2_3 == "2:3"
        assert AspectRatio.RATIO_3_2 == "3:2"
        assert AspectRatio.RATIO_4_5 == "4:5"
        assert AspectRatio.RATIO_5_4 == "5:4"
        assert AspectRatio.RATIO_9_16 == "9:16"
        assert AspectRatio.RATIO_9_21 == "9:21"

    def test_generation_mode_enum(self):
        """Test that GenerationMode enum has the expected values."""
        assert GenerationMode.TEXT_TO_IMAGE == "text-to-image"
        assert GenerationMode.IMAGE_TO_IMAGE == "image-to-image"


class TestSD35TextToImageParams:
    """Tests for the SD35TextToImageParams model."""

    def test_valid_params(self):
        """Test that valid parameters are accepted."""
        params = SD35TextToImageParams(prompt="A beautiful mountain landscape")
        assert params.prompt == "A beautiful mountain landscape"
        assert params.aspect_ratio == AspectRatio.RATIO_1_1
        assert params.seed == 0
        assert params.negative_prompt is None
        assert params.output_format == OutputFormat.PNG

    def test_custom_values(self):
        """Test that custom values are accepted."""
        params = SD35TextToImageParams(
            prompt="A beautiful mountain landscape",
            aspect_ratio=AspectRatio.RATIO_16_9,
            seed=12345,
            negative_prompt="people, clouds",
            output_format=OutputFormat.JPEG,
        )
        assert params.prompt == "A beautiful mountain landscape"
        assert params.aspect_ratio == AspectRatio.RATIO_16_9
        assert params.seed == 12345
        assert params.negative_prompt == "people, clouds"
        assert params.output_format == OutputFormat.JPEG

    def test_prompt_length_validation_empty(self):
        """Test that empty prompts are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SD35TextToImageParams(prompt="")

        # Verify the error is about string length
        errors = exc_info.value.errors()
        assert any("at least 1 character" in str(error).lower() for error in errors)

    def test_prompt_length_validation_max(self):
        """Test that prompts up to 10,000 characters are accepted."""
        # Valid: exactly 10,000 characters
        long_prompt = "a" * 10000
        params = SD35TextToImageParams(prompt=long_prompt)
        assert len(params.prompt) == 10000

        # Invalid: 10,001 characters
        with pytest.raises(ValidationError) as exc_info:
            SD35TextToImageParams(prompt="a" * 10001)

        # Verify the error is about string length
        errors = exc_info.value.errors()
        assert any("at most 10000 character" in str(error).lower() for error in errors)

    def test_negative_prompt_length_validation(self):
        """Test that negative prompt length is validated."""
        # Valid: exactly 10,000 characters
        long_negative = "b" * 10000
        params = SD35TextToImageParams(
            prompt="test",
            negative_prompt=long_negative
        )
        assert len(params.negative_prompt) == 10000

        # Invalid: 10,001 characters
        with pytest.raises(ValidationError) as exc_info:
            SD35TextToImageParams(
                prompt="test",
                negative_prompt="b" * 10001
            )

        # Verify the error is about string length
        errors = exc_info.value.errors()
        assert any("at most 10000 character" in str(error).lower() for error in errors)

    def test_seed_bounds_minimum(self):
        """Test that seed must be at least 0."""
        # Valid: 0
        params = SD35TextToImageParams(prompt="test", seed=0)
        assert params.seed == 0

        # Invalid: -1
        with pytest.raises(ValidationError) as exc_info:
            SD35TextToImageParams(prompt="test", seed=-1)

        # Verify the error is about the minimum value
        errors = exc_info.value.errors()
        assert any("greater than or equal to 0" in str(error).lower() for error in errors)

    def test_seed_bounds_maximum(self):
        """Test that seed must be at most 4,294,967,294."""
        # Valid: exactly 4,294,967,294
        params = SD35TextToImageParams(prompt="test", seed=4294967294)
        assert params.seed == 4294967294

        # Invalid: 4,294,967,295
        with pytest.raises(ValidationError) as exc_info:
            SD35TextToImageParams(prompt="test", seed=4294967295)

        # Verify the error is about the maximum value
        errors = exc_info.value.errors()
        assert any("less than or equal to 4294967294" in str(error).lower() for error in errors)


class TestSD35ImageToImageParams:
    """Tests for the SD35ImageToImageParams model."""

    @pytest.fixture
    def valid_base64_image(self):
        """Create a valid base64-encoded image for testing."""
        # Create a simple 100x100 white image
        img = Image.new('RGB', (100, 100), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    @pytest.fixture
    def small_base64_image(self):
        """Create a base64-encoded image that's too small (below 64px)."""
        # Create a 50x50 image (below minimum)
        img = Image.new('RGB', (50, 50), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        return base64.b64encode(image_bytes).decode('utf-8')

    def test_valid_params(self, valid_base64_image):
        """Test that valid parameters are accepted."""
        params = SD35ImageToImageParams(
            prompt="Transform this image",
            image=valid_base64_image,
            strength=0.7
        )
        assert params.prompt == "Transform this image"
        assert params.image == valid_base64_image
        assert params.strength == 0.7
        assert params.seed == 0
        assert params.negative_prompt is None
        assert params.output_format == OutputFormat.PNG

    def test_custom_values(self, valid_base64_image):
        """Test that custom values are accepted."""
        params = SD35ImageToImageParams(
            prompt="Transform this image",
            image=valid_base64_image,
            strength=0.5,
            seed=54321,
            negative_prompt="artifacts, noise",
            output_format=OutputFormat.WEBP,
        )
        assert params.prompt == "Transform this image"
        assert params.strength == 0.5
        assert params.seed == 54321
        assert params.negative_prompt == "artifacts, noise"
        assert params.output_format == OutputFormat.WEBP

    def test_strength_bounds_minimum(self, valid_base64_image):
        """Test that strength must be at least 0.0."""
        # Valid: 0.0
        params = SD35ImageToImageParams(
            prompt="test",
            image=valid_base64_image,
            strength=0.0
        )
        assert params.strength == 0.0

        # Invalid: -0.1
        with pytest.raises(ValidationError) as exc_info:
            SD35ImageToImageParams(
                prompt="test",
                image=valid_base64_image,
                strength=-0.1
            )

        # Verify the error is about the minimum value
        errors = exc_info.value.errors()
        assert any("greater than or equal to 0" in str(error).lower() for error in errors)

    def test_strength_bounds_maximum(self, valid_base64_image):
        """Test that strength must be at most 1.0."""
        # Valid: 1.0
        params = SD35ImageToImageParams(
            prompt="test",
            image=valid_base64_image,
            strength=1.0
        )
        assert params.strength == 1.0

        # Invalid: 1.1
        with pytest.raises(ValidationError) as exc_info:
            SD35ImageToImageParams(
                prompt="test",
                image=valid_base64_image,
                strength=1.1
            )

        # Verify the error is about the maximum value
        errors = exc_info.value.errors()
        assert any("less than or equal to 1" in str(error).lower() for error in errors)

    def test_image_dimension_validation_minimum(self, small_base64_image):
        """Test that images below 64px per side are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SD35ImageToImageParams(
                prompt="test",
                image=small_base64_image,
                strength=0.7
            )

        # Verify the error is about image dimensions
        errors = exc_info.value.errors()
        assert any("dimensions" in str(error).lower() and "64" in str(error) for error in errors)

    def test_image_dimension_validation_valid(self, valid_base64_image):
        """Test that images at or above 64px per side are accepted."""
        # Create a 64x64 image (minimum valid size)
        img = Image.new('RGB', (64, 64), color='white')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        min_size_image = base64.b64encode(image_bytes).decode('utf-8')

        params = SD35ImageToImageParams(
            prompt="test",
            image=min_size_image,
            strength=0.7
        )
        assert params.image == min_size_image

    def test_invalid_base64_image(self):
        """Test that invalid base64 data is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SD35ImageToImageParams(
                prompt="test",
                image="not-valid-base64!!!",
                strength=0.7
            )

        # Verify the error is about invalid image data
        errors = exc_info.value.errors()
        assert any("invalid image data" in str(error).lower() for error in errors)

    def test_seed_bounds(self, valid_base64_image):
        """Test that seed validation works for image-to-image."""
        # Valid: 0
        params = SD35ImageToImageParams(
            prompt="test",
            image=valid_base64_image,
            strength=0.7,
            seed=0
        )
        assert params.seed == 0

        # Valid: max
        params = SD35ImageToImageParams(
            prompt="test",
            image=valid_base64_image,
            strength=0.7,
            seed=4294967294
        )
        assert params.seed == 4294967294

        # Invalid: below minimum
        with pytest.raises(ValidationError):
            SD35ImageToImageParams(
                prompt="test",
                image=valid_base64_image,
                strength=0.7,
                seed=-1
            )

        # Invalid: above maximum
        with pytest.raises(ValidationError):
            SD35ImageToImageParams(
                prompt="test",
                image=valid_base64_image,
                strength=0.7,
                seed=4294967295
            )
