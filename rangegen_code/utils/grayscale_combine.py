#!/usr/bin/env python3
"""Combine two grayscale images into a color image by mapping per-pixel values.

Overview
--------
- Both input images are expected to be single-channel (grayscale).
- If an image has multiple channels, PIL will implicitly convert it to
  grayscale when loading.
- Pixel values are normalized to the range [0, 1] before any color mapping.
- The resulting color image is saved as an 8-bit RGB image.


Notes:
- If sizes differ, imgB is resized to match imgA unless --no-resize is specified.

"""

import argparse
import sys
import numpy as np
from PIL import Image


def load_grayscale(path):
    """Load an image and return a float32 array in [0,1], single channel.

    Notes
    -----
    - Normalization is performed using the maximum pixel value in the image.
    - No explicit channel selection is done if the image is multi-channel.

    """
    img = Image.open(path)
    # this is fine with range and intensity, because both of them will include maximum
    # value in invalid pixels
    arr = np.asarray(img, dtype=np.float32) / np.max(img)
    return arr


def save_rgb(arr_rgb, path):
    """Save a float array in [0,1] shaped (H, W, 3) as an 8-bit image."""
    arr_uint8 = np.clip(arr_rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr_uint8, mode="RGB").save(path)


def to_rgb_mode(a, b, blue_strategy="average"):
    """to_rgb_mode
    Map A->R, C->G, and compute B channel by strategy:
    - "average": B = (A + B)/2
    - "invdiff": B = 1 - |A - B|
    - "zero":    B = 0.
    """
    if blue_strategy == "average":
        c = 0.5 * (a + b)
    elif blue_strategy == "invdiff":
        c = 1.0 - np.abs(a - b)
    elif blue_strategy == "zero":
        c = np.zeros_like(a)
    else:
        raise ValueError(f"Unknown rgb-blue strategy: {blue_strategy}")
    rgb = np.stack([a, c, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def to_hsv_mode(a, b, v_strategy="average"):
    """to_hsv_mode.

    Interpret A as Hue and B as Saturation. Value (V) is:
      - "average": V = (A + B)/2 (often balanced)
      - "max":     V = max(A, B) (brighter wherever either is high)
      - "one":     V = 1.0       (max brightness; pure chroma from A,B)
    Hue range in colorsys is [0,1), Saturation/Value are [0,1].
    """
    if v_strategy == "average":
        v = 0.5 * (a + b)
    elif v_strategy == "max":
        v = np.maximum(a, b)
    elif v_strategy == "one":
        v = np.ones_like(a)
    else:
        raise ValueError(f"Unknown hsv-v strategy: {v_strategy}")

    # Vectorized HSV->RGB
    H = a
    S = np.clip(b, 0.0, 1.0)
    V = np.clip(v, 0.0, 1.0)

    # colorsys is scalar; implement a fast vectorized conversion
    # Reference: standard HSV->RGB algorithm
    h = (H % 1.0) * 6.0
    i = np.floor(h).astype(int)
    f = h - i
    p = V * (1.0 - S)
    q = V * (1.0 - S * f)
    t = V * (1.0 - S * (1.0 - f))

    r = np.choose(i % 6, [V, q, p, p, t, V])
    g = np.choose(i % 6, [t, V, V, q, p, p])
    b = np.choose(i % 6, [p, p, t, V, V, q])

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Combine two grayscale images into a color image."
    )
    parser.add_argument(
        "--output", help="Path to output color images (e.g., out)", default="output"
    )

    parser.add_argument(
        "--mode",
        choices=["rgb", "hsv"],
        default="rgb",
        help="Color mapping mode (default: rgb)",
    )
    parser.add_argument(
        "--rgb-blue",
        choices=["average", "invdiff", "zero"],
        default="average",
        help="Blue channel strategy for --mode rgb (default: average)",
    )
    parser.add_argument(
        "--hsv-v",
        choices=["average", "max", "one"],
        default="average",
        help="Value channel strategy for --mode hsv (default: average)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Do not resize B to match A; will error if sizes differ",
    )
    args = parser.parse_args()

    import os

    frames = os.listdir(path="intensity")

    modes = ["hsv", "rgb"]
    rgb_blue = ["average", "invdiff", "zero"]
    hsv_v = ["average", "max", "one"]
    for mode in modes:
        if mode == "rgb":
            strategies = rgb_blue
        elif mode == "hsv":
            strategies = hsv_v

        for strategy in strategies:
            for intensity_frame in frames:
                # Load inputs as float arrays in [0,1]
                b = 1 - load_grayscale(os.path.join("intensity", intensity_frame))
                a = load_grayscale(os.path.join("range", intensity_frame))

                # Match sizes
                if a.shape != b.shape:
                    if args.no_resize:
                        print(
                            f"Error: image sizes differ: A={a.shape}, B={b.shape}.\
                              Use matching sizes or remove --no-resize.",
                            file=sys.stderr,
                        )
                        sys.exit(1)
                    else:
                        # Resize B to match A using PIL (bilinear)
                        b_img = Image.fromarray((b * 255.0).astype(np.uint8), mode="L")
                        b_img = b_img.resize(
                            (a.shape[1], a.shape[0]), resample=Image.BILINEAR
                        )
                        b = np.asarray(b_img, dtype=np.float32) / 255.0

                # Compute color mapping
                if mode == "rgb":
                    rgb = to_rgb_mode(a, b, blue_strategy=strategy)
                elif mode == "hsv":
                    rgb = to_hsv_mode(a, b, v_strategy=strategy)
                else:
                    raise ValueError("Unknown mode")

                output_path = os.path.join(args.output, mode, strategy)
                os.makedirs(output_path, exist_ok=True)
                save_rgb(rgb, os.path.join(output_path, intensity_frame))
                print(
                    f"Saved color image to: {os.path.join(output_path, intensity_frame)}"
                )


if __name__ == "__main__":
    main()
