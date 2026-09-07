# RangeGen

RangeGen is the range-image generation framework described in Section III of the
paper: it projects dense mobile laser scanning point clouds into 2D range and
intensity frames from perturbed virtual sensor poses, transfers verified 2D
annotations back onto the source 3D points, refines those labels through
neighbourhood analysis and distance-based clustering, and reprojects the labelled
cloud to yield further annotated frames without repeating manual labelling.

## Status

**Not included in this repository yet.** RangeGen is maintained separately from
the dataset and benchmark code, and is being prepared for release.

<!-- TODO before submitting the camera-ready:
     replace this section with either
       (a) the source, vendored into this directory, or
       (b) a URL to its own repository plus the commit or tag used to build
           the released frames.
     If it will not be released, say so explicitly here and in the paper, so
     the omission is a stated scope decision rather than a missing artifact. -->

## What is reproducible without it

RangeGen produces the frames in `data/`. Because those frames are released
directly, everything downstream of generation reproduces from this repository
alone:

- the dataset statistics and every figure in Section IV-A
  (`scripts/dataset_analysis.py`)
- the fifteen instance-segmentation baselines and the baseline table
  (`scripts/train.py`, `scripts/make_table.py`)

What cannot be reproduced without RangeGen is the generation step itself:
producing new frames from a different point cloud, or from different virtual
sensor poses.

## Source data

The mobile laser scanning point clouds that RangeGen consumes are not
redistributed here. See `LICENSE-DATA` for the provenance note.
