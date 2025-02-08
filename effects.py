import cv2
import numpy as np


def pixel_sort_effect(frame, glitch_chance=0.1):
    """
    Applies a pixel-sorting glitch effect to random rows in the frame.

    Args:
        frame (np.ndarray): The input image (H x W x 3).
        glitch_chance (float): Probability that any given row will be sorted.

    Returns:
        np.ndarray: The glitched frame.
    """
    h, w, _ = frame.shape
    for i in range(h):
        if np.random.rand() < glitch_chance:
            # Sort the row based on the brightness (average of B, G, R)
            row = frame[i]
            brightness = row.mean(axis=1)
            sorted_indices = np.argsort(brightness)
            frame[i] = row[sorted_indices]
    return frame


def apply_block_shift_glitch(frame, glitch_chance=0.07, max_offset=20, block_size=5):
    """
    Randomly shifts horizontal blocks of the frame to create a glitch effect.

    Args:
        frame (np.ndarray): The input image (H x W x 3).
        glitch_chance (float): Chance of applying a glitch to each block.
        max_offset (int): Maximum pixel offset for the horizontal shift.
        block_size (int): Number of consecutive rows in each block.
    Returns:
        np.ndarray: The glitched frame.
    """
    h, w, _ = frame.shape
    for y in range(0, h, block_size):
        if np.random.rand() < glitch_chance:
            # Determine a random horizontal offset (could be left or right)
            offset = np.random.randint(-max_offset, max_offset)
            # np.roll shifts the block along the horizontal axis (axis=1)
            frame[y : y + block_size] = np.roll(
                frame[y : y + block_size], offset, axis=1
            )
    return frame


def apply_color_shift_glitch(frame, shift_range=5):
    """
    Shifts the red channel of the frame by a random offset to create a color glitch effect.

    Args:
        frame (np.ndarray): The input image (H x W x 3).
        shift_range (int): Maximum pixel shift for the red channel.
    Returns:
        np.ndarray: The glitched frame.
    """
    b, g, r = cv2.split(frame)
    # Choose a random horizontal shift for the red channel
    shift = np.random.randint(-shift_range, shift_range)
    r_shifted = np.roll(r, shift, axis=1)
    # Merge channels back together
    return cv2.merge([b, g, r_shifted])


def add_scanlines(frame, line_spacing=3, intensity=0.6):
    """
    Overlays dark horizontal scan lines over the frame.

    Args:
        frame (np.ndarray): The input image.
        line_spacing (int): Space between each scan line.
        intensity (float): Multiplier for darkening (0 to 1, where 0 is completely black).

    Returns:
        np.ndarray: The frame with scanlines.
    """
    overlay = frame.copy().astype(np.float32)
    h, w, _ = frame.shape
    for i in range(0, h, line_spacing):
        overlay[i, :] *= intensity
    return np.clip(overlay, 0, 255).astype(np.uint8)


def wave_distortion(frame, amplitude=5, frequency=0.05):
    h, w, _ = frame.shape
    # Create coordinate maps
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))

    # Convert to float32 and ensure the arrays are contiguous
    map_x = np.ascontiguousarray(map_x.astype(np.float32))
    map_y = np.ascontiguousarray(map_y.astype(np.float32))

    # Apply a sine-based distortion to the x coordinates
    map_x += amplitude * np.sin(2 * np.pi * map_y * frequency)

    # Now both map_x and map_y are CV_32FC1
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def neon_glow(frame, threshold=200, blur_ksize=(15, 15), glow_intensity=0.5):
    """
    Applies a neon glow effect to bright regions in the frame.

    Args:
        frame (np.ndarray): The input image.
        threshold (int): Intensity threshold to detect bright regions.
        blur_ksize (tuple): Kernel size for the Gaussian blur.
        glow_intensity (float): How much of the blurred image to add back.

    Returns:
        np.ndarray: The frame with a neon glow effect.
    """
    # Convert to grayscale to find bright areas
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Create a mask for bright areas
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, blur_ksize, 0)
    # Convert mask to 3 channels
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    # Blur the original image to create the glow
    blurred = cv2.GaussianBlur(frame, blur_ksize, 0).astype(np.float32)
    # Combine the original image with the glow
    glowed = cv2.addWeighted(frame.astype(np.float32), 1.0, blurred, glow_intensity, 0)
    # Use the mask to blend the glow selectively
    frame = glowed * mask + frame.astype(np.float32) * (1 - mask)
    return np.clip(frame, 0, 255).astype(np.uint8)


def color_jitter(frame, intensity=0.05):
    """
    Randomly jitters the color channels to produce a neon-like effect.

    Args:
        frame (np.ndarray): The input image.
        intensity (float): The standard deviation factor for noise.

    Returns:
        np.ndarray: The color-jittered frame.
    """
    noise = np.random.randn(*frame.shape) * 255 * intensity
    frame = frame.astype(np.float32) + noise
    return np.clip(frame, 0, 255).astype(np.uint8)


def add_noise(frame, noise_level=0.02):
    """
    Adds random noise to the frame.

    Args:
        frame (np.ndarray): The input image.
        noise_level (float): The proportion of noise relative to the maximum intensity.

    Returns:
        np.ndarray: The noisy frame.
    """
    noise = np.random.randn(*frame.shape) * 255 * noise_level
    frame = frame.astype(np.float32) + noise
    return np.clip(frame, 0, 255).astype(np.uint8)
