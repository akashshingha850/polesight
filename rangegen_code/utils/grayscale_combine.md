# Grayscale Image Combiner

`grayscale_combine.py` is a small Python utility that combines **two grayscale images** into a **single color image** by mapping per‑pixel values from each image into color channels. It is intended primarily for **visualization and exploratory analysis** of paired single‑channel images.

---

## Overview

- Both input images are expected to be **single‑channel (grayscale)**.
- If an image has multiple channels, Pillow will implicitly convert it to grayscale when loading.
- Pixel values are normalized to the range **[0, 1]** before color mapping.
- The resulting image is saved as an **8‑bit RGB image**.
- If input image sizes differ, the second image can be resized to match the first (unless resizing is explicitly disabled).

---

## Features

- Combines **two grayscale images** into one color image
- Automatic per‑image normalization
- Multiple **channel mapping strategies**
- Supports **RGB‑style** and **HSV‑style** interpretations
- Minimal dependencies (`numpy`, `Pillow`)

---

## Installation

### Requirements

- Python **3.7+**
- `numpy`
- `Pillow`

Install dependencies with:

```bash
pip install numpy pillow
```

---

## Usage

Basic usage pattern:

```bash
python grayscale_combine.py imageA.png imageB.png --output output.png
```

The output image is written as a standard RGB image with values scaled to 8‑bit (0–255).

> Note: The exact set of command‑line options depends on how `main()` is extended or finalized.

---

## RGB Mode Mapping

In RGB mode, the two input images are mapped as follows:

- Image **A** → Red channel
- Image **B** → Blue channel
- Green channel is computed using a selectable strategy

### Green / Blue Channel Strategies

- **`average`**
  C = (A + B) / 2
  Balanced visualization of both images.

- **`invdiff`**
  C = 1 − |A − B|
  Highlights regions where the two images are similar.

- **`zero`**
  C = 0
  Produces a red–blue composite without a third contribution.

All values are clipped to the range **[0, 1]** before saving.

---

## HSV Mode Mapping

The script can also interpret the grayscale images in HSV color space:

- Image **A** → Hue
- Image **B** → Saturation
- Value (brightness) is computed separately

### Value (V) Strategies

- **`average`** – Balanced brightness:
  V = (A + B) / 2
    - Currently used. Best looking
- **`max`** – Bright wherever either image is strong:
  V = max(A, B)
    - Should have most clarity but constract is strange
- **`one`** – Constant maximum brightness:
    - Images look washed out and hurts ayes to look
  V = 1

Hue, saturation, and value are all defined in the normalized range **[0, 1]**.

---

## Implementation Notes

- Normalization uses the **maximum pixel value** of each image independently.
    - This is fine with range and intensity because both contain maximum value (in invalid pixels)
- No explicit channel selection is performed if an input image is multi‑channel.
    - Images are transformed into grayscale
- The tool is designed for **qualitative visualization**, not photometrically accurate color reconstruction.

---
