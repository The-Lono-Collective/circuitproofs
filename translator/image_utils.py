"""
Image utilities for vision model testing and preprocessing.

This module provides helper functions for creating and validating images
for use with HuggingFace vision models (ViT, Swin, CLIP, etc.).
"""

from typing import Tuple, Union, Optional
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def create_dummy_image(
    size: Tuple[int, int] = (224, 224),
    mode: str = "RGB",
    seed: Optional[int] = None
) -> "Image.Image":
    """
    Create a dummy PIL image suitable for vision model testing.

    Args:
        size: Image dimensions as (width, height). Defaults to (224, 224).
        mode: PIL image mode ('RGB', 'L', 'RGBA'). Defaults to 'RGB'.
        seed: Optional random seed for reproducibility.

    Returns:
        A PIL Image with random pixel values in the correct format.

    Raises:
        ImportError: If PIL/Pillow is not installed.
        ValueError: If mode is not supported.

    Example:
        >>> img = create_dummy_image(size=(224, 224), mode='RGB')
        >>> img.size
        (224, 224)
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL/Pillow is required. Install with: pip install pillow")

    supported_modes = {"RGB", "L", "RGBA"}
    if mode not in supported_modes:
        raise ValueError(f"Mode must be one of {supported_modes}, got '{mode}'")

    if seed is not None:
        np.random.seed(seed)

    # Determine number of channels based on mode
    channels = {"RGB": 3, "L": 1, "RGBA": 4}[mode]

    # Create random uint8 array with values in [0, 255]
    if channels == 1:
        array = np.random.randint(0, 256, (size[1], size[0]), dtype=np.uint8)
    else:
        array = np.random.randint(0, 256, (size[1], size[0], channels), dtype=np.uint8)

    return Image.fromarray(array, mode=mode)


def validate_image_for_model(
    image: Union["Image.Image", np.ndarray],
    expected_size: Optional[Tuple[int, int]] = None,
    expected_mode: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Validate that an image is suitable for vision model processing.

    Args:
        image: PIL Image or numpy array to validate.
        expected_size: Optional expected (width, height).
        expected_mode: Optional expected mode ('RGB', 'L', etc.).

    Returns:
        Tuple of (is_valid, message) where message explains any issues.

    Example:
        >>> img = create_dummy_image()
        >>> valid, msg = validate_image_for_model(img, expected_size=(224, 224))
        >>> valid
        True
    """
    if not PIL_AVAILABLE:
        return False, "PIL/Pillow is not installed"

    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        # Check array values are in valid range
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.min() < 0 or image.max() > 1:
                return False, (
                    f"Float array values must be in [0, 1], "
                    f"got [{image.min():.2f}, {image.max():.2f}]"
                )
        elif image.dtype == np.uint8:
            pass  # Valid range is [0, 255], which uint8 guarantees
        else:
            return False, f"Unsupported array dtype: {image.dtype}"

        try:
            image = Image.fromarray(image)
        except Exception as e:
            return False, f"Failed to convert array to PIL Image: {e}"

    if not isinstance(image, Image.Image):
        return False, f"Expected PIL Image, got {type(image).__name__}"

    # Validate size
    if expected_size is not None and image.size != expected_size:
        return False, f"Expected size {expected_size}, got {image.size}"

    # Validate mode
    if expected_mode is not None and image.mode != expected_mode:
        return False, f"Expected mode '{expected_mode}', got '{image.mode}'"

    return True, "Image is valid"


def normalize_image_array(
    array: np.ndarray,
    target_range: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Normalize a numpy array to a target range.

    Args:
        array: Input numpy array.
        target_range: Tuple of (min, max) for output range.

    Returns:
        Normalized numpy array as float32.

    Example:
        >>> arr = np.array([0, 128, 255], dtype=np.uint8)
        >>> normalized = normalize_image_array(arr)
        >>> normalized.min(), normalized.max()
        (0.0, 1.0)
    """
    array = array.astype(np.float32)

    # Handle edge case of constant array
    arr_min, arr_max = array.min(), array.max()
    if arr_min == arr_max:
        return np.full_like(array, target_range[0])

    # Normalize to [0, 1] first
    normalized = (array - arr_min) / (arr_max - arr_min)

    # Scale to target range
    target_min, target_max = target_range
    return normalized * (target_max - target_min) + target_min
