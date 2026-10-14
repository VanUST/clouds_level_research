# PURPOSE: Scratch probe of the RoMa / LightGlue / DeDoDe python APIs on one real
#          cloud frame, so the production backends are written against APIs that
#          were actually executed rather than remembered. Not part of the result;
#          it prints shapes and match counts and writes nothing.
# INPUTS: one prepared 640x640 cloud crop pair from 2025-05-23.
# OUTPUTS: stdout only.
# KEYWORDS: probe, api, roma, lightglue, dedode, scratch
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import harness  # noqa: E402


def _grab():
    cfg = harness.load_cfg(None)
    pairs = harness.select_frames(cfg, date="2025-05-23", t0="12:00:00", t1="12:20:00", step_s=600)
    g1, g2, _, _ = harness.prepare_crop(cfg, pairs[0][0], pairs[0][1])
    return g1, g2


def probe_lightglue(g1, g2, dev):
    import torch
    import kornia.feature as KF
    t = torch.device(dev)
    x1 = torch.from_numpy(g1)[None, None].float().to(t) / 255.0
    x2 = torch.from_numpy(g2)[None, None].float().to(t) / 255.0
    x1 = x1.repeat(1, 3, 1, 1)
    x2 = x2.repeat(1, 3, 1, 1)
    for name, extractor in (
        ("disk", KF.DISK.from_pretrained("depth", device=t)),
        ("aliked", KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=2048, device=t)),
    ):
        ex = extractor.eval()
        lg = KF.LightGlueMatcher(name).to(t).eval()
        with torch.no_grad():
            f0 = ex(x1)[0]
            f1 = ex(x2)[0]
            print(name, "feat type", type(f0).__name__,
                  "kp", tuple(f0.keypoints.shape), "desc", tuple(f0.descriptors.shape))
            l0 = KF.laf_from_center_scale_ori(f0.keypoints[None])
            l1 = KF.laf_from_center_scale_ori(f1.keypoints[None])
            sc, m = lg(f0.descriptors, f1.descriptors, l0, l1, (640, 640), (640, 640))
            print(name, "matches", tuple(m.shape), "scores", tuple(sc.shape))


def probe_dedode(g1, g2, dev):
    import torch
    import kornia.feature as KF
    t = torch.device(dev)
    x1 = torch.from_numpy(g1)[None, None].float().to(t) / 255.0
    x2 = torch.from_numpy(g2)[None, None].float().to(t) / 255.0
    x1 = x1.repeat(1, 3, 1, 1)
    x2 = x2.repeat(1, 3, 1, 1)
    dd = KF.DeDoDe.from_pretrained(detector_weights="L-C4-v2", descriptor_weights="G-upright")
    dd = dd.eval().to(t)
    with torch.no_grad():
        k1, s1, d1 = dd(x1)
        k2, s2, d2 = dd(x2)
        print("dedode kp", tuple(k1.shape), "desc", tuple(d1.shape), "scores", tuple(s1.shape))


def _pad14(gray):
    h, w = gray.shape
    ph = (-h) % 14
    pw = (-w) % 14
    out = cv2.copyMakeBorder(gray, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return out, ph, pw


def _to_roma_tensor(gray, t):
    import torch
    p, _, _ = _pad14(gray)
    rgb = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1)[None].float().to(t) / 255.0


def probe_roma(g1, g2, dev):
    import torch
    from PIL import Image
    from romatch import roma_outdoor
    t = torch.device(dev)
    x1 = Image.fromarray(cv2.cvtColor(g1, cv2.COLOR_GRAY2RGB))
    x2 = Image.fromarray(cv2.cvtColor(g2, cv2.COLOR_GRAY2RGB))
    m = roma_outdoor(device=t, amp_dtype=torch.float16, use_custom_corr=False)
    tt = time.time()
    with torch.no_grad():
        out = m.match(x1, x2)
    print("roma time %.2fs" % (time.time() - tt))
    if isinstance(out, tuple):
        for i, o in enumerate(out):
            print("roma out[%d] %s %s" % (i, type(o), tuple(o.shape)))
    else:
        print("roma type", type(out))


if __name__ == "__main__":
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g1, g2 = _grab()
    print("crops", g1.shape, g2.shape, g1.dtype)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "lightglue"):
        probe_lightglue(g1, g2, dev)
    if which in ("all", "dedode"):
        probe_dedode(g1, g2, dev)
    if which in ("all", "roma"):
        probe_roma(g1, g2, dev)
