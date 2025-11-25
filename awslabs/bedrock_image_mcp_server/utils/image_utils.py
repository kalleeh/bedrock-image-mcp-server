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
"""Image processing utilities for encoding, decoding, and validation."""

import base64
import os
from io import BytesIO
from PIL import Image
from typing import Tuple


def encode_image_file(file_path: str) -> str:
    """Encode an image file to base64 string.

    This function reads an image file from disk and encodes it as a base64 string
    suitable for transmission to AWS Bedrock APIs.

    Args:
        file_path: Path to the image file to encode.

    Returns:
        Base64-encoded string representation of the image.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read.
        ValueError: If the file is not a valid image.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file not found: {file_path}")

    try:
        with open(file_path, 'rb') as image_file:
            image_data = image_file.read()
            base64_encoded = base64.b64encode(image_data).decode('utf-8')
            return base64_encoded
    except IOError as e:
        raise IOError(f"Failed to read image file {file_path}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to encode image file {file_path}: {str(e)}")


def decode_base64_image(base64_str: str) -> bytes:
    """Decode a base64 string to image bytes.

    Args:
        base64_str: Base64-encoded image string.

    Returns:
        Raw image bytes.

    Raises:
        ValueError: If the base64 string is invalid.
    """
    try:
        return base64.b64decode(base64_str)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def validate_image_dimensions(
    image_data: bytes,
    min_width: int = 64,
    min_height: int = 64,
    max_pixels: int = None
) -> Tuple[int, int]:
    """Validate image dimensions against constraints.

    This function opens an image from bytes and validates its dimensions
    against minimum and maximum constraints. It's used to ensure images
    meet the requirements of various Bedrock models before API calls.

    Args:
        image_data: Raw image bytes to validate.
        min_width: Minimum allowed width in pixels (default: 64).
        min_height: Minimum allowed height in pixels (default: 64).
        max_pixels: Maximum allowed total pixels (width * height), or None for no limit.

    Returns:
        Tuple of (width, height) in pixels.

    Raises:
        ValueError: If the image dimensions don't meet the constraints.
        IOError: If the image data cannot be opened.
    """
    try:
        # Open image from bytes
        image = Image.open(BytesIO(image_data))
        width, height = image.size

        # Validate minimum dimensions
        if width < min_width:
            raise ValueError(
                f"Image width {width}px is below minimum {min_width}px"
            )
        if height < min_height:
            raise ValueError(
                f"Image height {height}px is below minimum {min_height}px"
            )

        # Validate maximum pixels if specified
        if max_pixels is not None:
            total_pixels = width * height
            if total_pixels > max_pixels:
                raise ValueError(
                    f"Image has {total_pixels} pixels, exceeding maximum {max_pixels} pixels"
                )

        return width, height

    except IOError as e:
        raise IOError(f"Failed to open image data: {str(e)}")
    except ValueError:
        # Re-raise ValueError as-is (from our validation)
        raise
    except Exception as e:
        raise ValueError(f"Failed to validate image dimensions: {str(e)}")


def create_rectangular_mask(
    width: int,
    height: int,
    x: int,
    y: int,
    mask_width: int,
    mask_height: int,
    feather: int = 0
) -> bytes:
    """Create a rectangular mask for inpainting or object removal.

    Creates a grayscale mask image where white (255) indicates the area to fill/remove
    and black (0) indicates areas to preserve. The mask is a rectangle positioned at
    (x, y) with the specified dimensions.

    Args:
        width: Total width of the mask image in pixels.
        height: Total height of the mask image in pixels.
        x: X coordinate of the top-left corner of the white rectangle.
        y: Y coordinate of the top-left corner of the white rectangle.
        mask_width: Width of the white rectangle in pixels.
        mask_height: Height of the white rectangle in pixels.
        feather: Optional feathering/blur radius in pixels for soft edges (default: 0).

    Returns:
        PNG-encoded mask image as bytes.

    Raises:
        ValueError: If dimensions are invalid or rectangle is out of bounds.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive: {width}x{height}")
    if mask_width <= 0 or mask_height <= 0:
        raise ValueError(f"Mask dimensions must be positive: {mask_width}x{mask_height}")
    if x < 0 or y < 0:
        raise ValueError(f"Mask position must be non-negative: ({x}, {y})")
    if x + mask_width > width or y + mask_height > height:
        raise ValueError(
            f"Mask rectangle ({x}, {y}, {mask_width}, {mask_height}) "
            f"exceeds image bounds ({width}x{height})"
        )

    # Create black background
    mask = Image.new('L', (width, height), 0)

    # Draw white rectangle
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x, y, x + mask_width - 1, y + mask_height - 1], fill=255)

    # Apply feathering if requested
    if feather > 0:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # Convert to PNG bytes
    buffer = BytesIO()
    mask.save(buffer, format='PNG')
    return buffer.getvalue()


def create_ellipse_mask(
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    radius_x: int,
    radius_y: int,
    feather: int = 0
) -> bytes:
    """Create an elliptical mask for inpainting or object removal.

    Creates a grayscale mask image where white (255) indicates the area to fill/remove
    and black (0) indicates areas to preserve. The mask is an ellipse centered at
    (center_x, center_y) with the specified radii.

    Args:
        width: Total width of the mask image in pixels.
        height: Total height of the mask image in pixels.
        center_x: X coordinate of the ellipse center.
        center_y: Y coordinate of the ellipse center.
        radius_x: Horizontal radius of the ellipse in pixels.
        radius_y: Vertical radius of the ellipse in pixels.
        feather: Optional feathering/blur radius in pixels for soft edges (default: 0).

    Returns:
        PNG-encoded mask image as bytes.

    Raises:
        ValueError: If dimensions are invalid or ellipse is out of bounds.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive: {width}x{height}")
    if radius_x <= 0 or radius_y <= 0:
        raise ValueError(f"Ellipse radii must be positive: {radius_x}x{radius_y}")
    if center_x < 0 or center_y < 0:
        raise ValueError(f"Ellipse center must be non-negative: ({center_x}, {center_y})")

    # Create black background
    mask = Image.new('L', (width, height), 0)

    # Draw white ellipse
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    bbox = [
        center_x - radius_x,
        center_y - radius_y,
        center_x + radius_x,
        center_y + radius_y
    ]
    draw.ellipse(bbox, fill=255)

    # Apply feathering if requested
    if feather > 0:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # Convert to PNG bytes
    buffer = BytesIO()
    mask.save(buffer, format='PNG')
    return buffer.getvalue()


def create_full_mask(width: int, height: int) -> bytes:
    """Create a full white mask covering the entire image.

    Useful for complete image replacement or testing inpainting tools.

    Args:
        width: Width of the mask image in pixels.
        height: Height of the mask image in pixels.

    Returns:
        PNG-encoded mask image as bytes.

    Raises:
        ValueError: If dimensions are invalid.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive: {width}x{height}")

    # Create white image
    mask = Image.new('L', (width, height), 255)

    # Convert to PNG bytes
    buffer = BytesIO()
    mask.save(buffer, format='PNG')
    return buffer.getvalue()
