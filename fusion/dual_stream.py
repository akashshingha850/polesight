"""Mid-level fusion: a two-stream YOLO11-seg (fusion_sota.md, Tier 3, fusion option 1).

Two backbones, one per modality, exchange features at P3, P4 and P5 (after layers 4, 6
and 10 of the YOLO11 backbone). A single neck and segmentation head sit on the fused
features. Ultralytics' YAML parser has a single input, so the second backbone is added
by subclassing ``SegmentationModel``:

    main stream  the stock backbone + neck + head, fed channel ``chans[0]`` of the image
    aux stream   a copy of backbone layers 0-10, fed channel ``chans[1]``
    fusion       x_k <- x_k + Conv1x1([x_k, aux_k]) at each tap k, zero-initialised

The zero init makes the model *identical to the main-stream model at step 0*, so each
stream can be warm-started from its own single-modality checkpoint (the main stream with
its neck and head, the aux stream with its backbone) and the fusion convolutions are the
only new parameters. That removes most of the modality-competition risk at initialisation
(fusion_sota.md 1.5 and 3.4).

Images are the packed 3-channel layouts of ``make_fused.py``; a stream takes one channel
and replicates it to three so the pretrained stems apply unchanged.

This module must stay importable under the name ``dual_stream``: Ultralytics pickles the
model class by module path into every checkpoint.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils import RANK

TAPS = (4, 6, 10)  # P3, P4, P5 outputs of the YOLO11 backbone
N_BACKBONE = 11  # layers 0-10 are the backbone in yolo11-seg.yaml


class DualStreamSegModel(SegmentationModel):
    def __init__(self, cfg="yolo11m-seg.yaml", ch=3, nc=None, verbose=True, chans=(1, 0)):
        super().__init__(cfg, ch=ch, nc=nc, verbose=verbose)
        self.chans = tuple(chans)
        self.aux = nn.ModuleList([copy.deepcopy(self.model[i]) for i in range(N_BACKBONE)])
        # Channel widths at the taps, read off a dummy forward rather than hard-coded.
        widths, x = {}, torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            self.aux.eval()
            for m in self.aux:
                x = m(x)
                if m.i in TAPS:
                    widths[m.i] = x.shape[1]
        self.fusers = nn.ModuleDict({str(k): nn.Conv2d(2 * c, c, 1) for k, c in widths.items()})
        for f in self.fusers.values():
            nn.init.zeros_(f.weight)
            nn.init.zeros_(f.bias)

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        if "aux" not in self._modules:  # the parent constructor probes the stride before the aux stream exists
            return super()._predict_once(x, profile, visualize, embed)
        ca, cb = self.chans
        xa = x[:, ca : ca + 1].expand(-1, 3, -1, -1)
        b = x[:, cb : cb + 1].expand(-1, 3, -1, -1)
        taps = {}
        for m in self.aux:
            b = m(b)
            if m.i in TAPS:
                taps[m.i] = b
        y, x = [], xa
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            if m.i in taps:
                x = x + self.fusers[str(m.i)](torch.cat([x, taps[m.i]], 1))
            y.append(x if m.i in self.save else None)
        return x

    def load_streams(self, main_ckpt: str, aux_ckpt: str) -> None:
        """Warm-start: main stream from ``main_ckpt`` (whole model), aux stream from ``aux_ckpt`` (backbone)."""
        main, _ = load_checkpoint(main_ckpt, fuse=False)
        self.load(main)  # shape-checked transfer of every matching tensor
        aux, _ = load_checkpoint(aux_ckpt, fuse=False)
        for i in range(N_BACKBONE):
            self.aux[i].load_state_dict(aux.model[i].float().state_dict())


class DualStreamTrainer(SegmentationTrainer):
    """SegmentationTrainer that builds the two-stream model and warm-starts it."""

    main_ckpt = None
    aux_ckpt = None
    chans = (1, 0)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DualStreamSegModel(cfg, ch=3, nc=self.data["nc"], verbose=verbose and RANK == -1, chans=self.chans)
        model.load_streams(self.main_ckpt, self.aux_ckpt)
        return model
