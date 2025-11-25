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
"""Tests for mask creation utilities."""

import pytest
from PIL import Image
from io import BytesIO
from awslabs.bedrock_image_mcp_server.utils.image_utils import (
    create_rectangular_mask,
    create_ellipse_mask,
    create_full_mask,
)


class TestRectangularMask:
    """Tests for rectangular mask creation."""

    def test_create_basic_rectangular_mask(self):
        """Test creating a basic rectangular mask."""
        mask_bytes = create_rectangular_mask(
            width=100,
            height=100,
            x=25,
            y=25,
            mask_width=50,
            mask_height=50,
        )

        # Verify it's valid PNG
        mask = Image.open(BytesIO(mask_bytes))
        assert mask.size == (100, 100)
        assert mask.mode == 'L'  # Grayscale

        # Check that center of rectangle is white
        assert mask.getpixel((50, 50)) == 255

        # Check that corner is black
        assert mask.getpixel((0, 0)) == 0

    def test_rectangular_mask_with_feather(self):
        """Test rectangular mask with feathering."""
        mask_bytes = create_rectangular_mask(
            width=100,
            height=100,
            x=25,
            y=25,
            mask_width=50,
            mask_height=50,
            feather=5,
        )

        mask = Image.open(BytesIO(mask_bytes))
        assert mask.size == (100, 100)

    def test_rectangular_mask_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match='must be positive'):
            create_rectangular_mask(
                width=0,
                height=100,
                x=0,
                y=0,
                mask_width=50,
                mask_height=50,
            )

    def test_rectangular_mask_out_of_bounds(self):
        """Test that out of bounds rectangle raises ValueError."""
        with pytest.raises(ValueError, match='exceeds image bounds'):
            create_rectangular_mask(
                width=100,
                height=100,
                x=80,
                y=80,
                mask_width=50,
                mask_height=50,
            )


class TestEllipseMask:
    """Tests for ellipse mask creation."""

    def test_create_basic_ellipse_mask(self):
        """Test creating a basic ellipse mask."""
        mask_bytes = create_ellipse_mask(
            width=100,
            height=100,
            center_x=50,
            center_y=50,
            radius_x=25,
            radius_y=25,
        )

        # Verify it's valid PNG
        mask = Image.open(BytesIO(mask_bytes))
        assert mask.size == (100, 100)
        assert mask.mode == 'L'

        # Check that center is white
        assert mask.getpixel((50, 50)) == 255

        # Check that corner is black
        assert mask.getpixel((0, 0)) == 0

    def test_ellipse_mask_with_feather(self):
        """Test ellipse mask with feathering."""
        mask_bytes = create_ellipse_mask(
            width=100,
            height=100,
            center_x=50,
            center_y=50,
            radius_x=25,
            radius_y=25,
            feather=5,
        )

        mask = Image.open(BytesIO(mask_bytes))
        assert mask.size == (100, 100)

    def test_ellipse_mask_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match='must be positive'):
            create_ellipse_mask(
                width=100,
                height=0,
                center_x=50,
                center_y=50,
                radius_x=25,
                radius_y=25,
            )


class TestFullMask:
    """Tests for full mask creation."""

    def test_create_full_mask(self):
        """Test creating a full white mask."""
        mask_bytes = create_full_mask(width=100, height=100)

        # Verify it's valid PNG
        mask = Image.open(BytesIO(mask_bytes))
        assert mask.size == (100, 100)
        assert mask.mode == 'L'

        # Check that all pixels are white
        assert mask.getpixel((0, 0)) == 255
        assert mask.getpixel((50, 50)) == 255
        assert mask.getpixel((99, 99)) == 255

    def test_full_mask_invalid_dimensions(self):
        """Test that invalid dimensions raise ValueError."""
        with pytest.raises(ValueError, match='must be positive'):
            create_full_mask(width=-1, height=100)
