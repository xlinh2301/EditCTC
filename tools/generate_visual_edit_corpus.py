#!/usr/bin/env python3
"""Generate a rendered, image-conditioned edit-pretraining corpus.

Each row contains a clean rendered image (the visual target), a clean GT
transcript, and a corrupted seed transcript.  The seed is deliberately kept
out of the rendered image: this makes the edit decoder answer the intended
question, namely whether the pixels support changing the proposed seed.

The generator is deterministic and records every corruption/render parameter.
It is a pretraining corpus, not an evaluation set.  Natural CTC seeds must be
generated later with a frozen recognizer and used for the transfer gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


CONFUSIONS = {
    "0": ["6", "8", "9"],
    "1": ["7"],
    "2": ["3", "7"],
    "3": ["2", "5", "8"],
    "4": ["1", "9"],
    "5": ["3", "6", "8"],
    "6": ["0", "5", "8"],
    "7": ["1", "2"],
    "8": ["0", "3", "6", "9"],
    "9": ["0", "4", "8"],
}


def read_texts(path: Path):
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = line.split("\t", 1)[-1].strip()
        if text.isdigit() and 1 <= len(text) <= 25:
            texts.append(text)
    if not texts:
        raise ValueError(f"no numeric transcripts found in {path}")
    return texts


def discover_fonts(font_file: str | None):
    if font_file:
        paths = [Path(font_file)]
    else:
        paths = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf"),
        ]
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise ValueError("no usable font found; pass --font")
    return paths


def choose_gt(rng: random.Random, texts, lengths):
    # Keep the real length/leading-zero distribution, then add random strings
    # so a few hundred thousand rows do not merely duplicate the label file.
    if rng.random() < 0.75:
        return rng.choice(texts)
    n = rng.choices(list(lengths), weights=[lengths[x] for x in lengths])[0]
    first = rng.choice("0123456789")
    return first + "".join(rng.choice("0123456789") for _ in range(n - 1))


def corrupt_once(gt: str, rng: random.Random):
    """Return (seed, operation records, corruption kind).

    Corruptions are confusion-aware and include multi-event cases.  The clean
    rows are retained as hard KEEP examples.  Rule corruption is only a
    bootstrap distribution; transfer is gated on frozen-CTC natural seeds.
    """
    if rng.random() < 0.25:
        return gt, [], "clean"
    seed = list(gt)
    operations = []
    count = 1 if rng.random() < 0.70 else 2
    for _ in range(count):
        if not seed:
            break
        typ = rng.choices(["replace", "delete", "insert"], weights=[0.50, 0.25, 0.25])[0]
        if typ == "replace":
            pos = rng.randrange(len(seed))
            old = seed[pos]
            new = rng.choice(CONFUSIONS.get(old, [d for d in "0123456789" if d != old]))
            seed[pos] = new
            operations.append({"op": "REPLACE", "position": pos, "from": old, "to": new})
        elif typ == "delete" and len(seed) > 1:
            pos = rng.randrange(len(seed))
            old = seed.pop(pos)
            operations.append({"op": "DELETE", "position": pos, "from": old})
        else:
            pos = rng.randrange(len(seed) + 1)
            anchor = seed[pos - 1] if pos else seed[0]
            new = rng.choice(CONFUSIONS.get(anchor, list("0123456789")))
            seed.insert(pos, new)
            operations.append({"op": "INSERT", "position": pos, "to": new})
    if not operations or not seed:
        return gt, [], "clean"
    kind = "+".join(x["op"].lower() for x in operations)
    return "".join(seed[:25]), operations, kind


def _fit_font(font_path: Path, text: str, rng: random.Random, width: int, height: int):
    size = rng.randint(max(18, height - 18), height - 3)
    while size >= 12:
        font = ImageFont.truetype(str(font_path), size=size)
        box = font.getbbox(text)
        if box[2] - box[0] <= width - 12:
            return font
        size -= 1
    return ImageFont.truetype(str(font_path), size=12)


def render(gt: str, rng: random.Random, fonts, width=320, height=48):
    # Three display families approximate the major visual factors without
    # pretending to reproduce one specific meter model.
    style = rng.choice(["dark_digits", "light_digits", "lcd"])
    if style == "dark_digits":
        bg = np.full((height, width, 3), rng.randint(205, 250), dtype=np.uint8)
        fg = (rng.randint(10, 55),) * 3
    elif style == "light_digits":
        bg = np.full((height, width, 3), rng.randint(20, 65), dtype=np.uint8)
        fg = (rng.randint(190, 245),) * 3
    else:
        base = rng.randint(125, 205)
        bg = np.full((height, width, 3), base, dtype=np.uint8)
        fg = (rng.randint(20, 90), rng.randint(35, 120), rng.randint(10, 70))
    # Low-frequency illumination and sensor noise.
    illum = np.linspace(rng.uniform(0.82, 1.12), rng.uniform(0.82, 1.12), width)
    bg = np.clip(bg.astype(np.float32) * illum[None, :, None], 0, 255).astype(np.uint8)
    image = Image.fromarray(bg, mode="RGB")
    draw = ImageDraw.Draw(image)
    font_path = rng.choice(fonts)
    font = _fit_font(font_path, gt, rng, width, height)
    box = draw.textbbox((0, 0), gt, font=font, stroke_width=0)
    tw, th = box[2] - box[0], box[3] - box[1]
    x = max(3, (width - tw) // 2 + rng.randint(-8, 8))
    y = max(0, (height - th) // 2 - box[1] + rng.randint(-3, 3))
    stroke = rng.choice([0, 0, 1])
    draw.text((x, y), gt, font=font, fill=fg, stroke_width=stroke, stroke_fill=fg)

    # Small geometric/photometric defects, applied after the clean text is
    # rendered so the visual encoder sees the target digit itself.
    if rng.random() < 0.45:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 1.05)))
    if rng.random() < 0.35:
        image = ImageOps.autocontrast(image, cutoff=rng.randint(0, 4))
    if rng.random() < 0.25:
        draw = ImageDraw.Draw(image, "RGBA")
        gx = rng.randint(0, width - 1)
        draw.ellipse((gx - 35, -15, gx + 35, height + 15), fill=(255, 255, 255, rng.randint(15, 65)))
    if rng.random() < 0.18:
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(rng.randint(1, 3)):
            xx = rng.randint(0, width - 1)
            draw.rectangle((xx, rng.randint(0, height - 5), xx + rng.randint(1, 5), height), fill=(0, 0, 0, rng.randint(20, 90)))
    arr = np.asarray(image).astype(np.int16)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(0.5, 5.0), arr.shape[:2] + (1,))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB"), {"style": style, "font": font_path.name, "font_size": font.size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--num-samples", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=4035)
    ap.add_argument("--font", default=None)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=48)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()
    texts = read_texts(args.labels)
    lengths = Counter(map(len, texts))
    fonts = discover_fonts(args.font)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "manifest.jsonl"
    stats = Counter()
    rng = random.Random(args.seed)
    with manifest.open("w", encoding="utf-8") as out:
        for idx in range(args.num_samples):
            gt = choose_gt(rng, texts, lengths)
            seed_text, ops, kind = corrupt_once(gt, rng)
            image, render_meta = render(gt, rng, fonts, args.width, args.height)
            name = f"synth_{idx:07d}.jpg"
            image.save(image_dir / name, format="JPEG", quality=args.jpeg_quality, optimize=True)
            split = "val" if idx % 50 == 0 else "train"
            row = {
                "image": f"images/{name}",
                "gt": gt,
                "seed": seed_text,
                "ops": ops,
                "kind": kind,
                "split": split,
                "render": render_meta,
                "source": "rule_rendered_gt",
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats[kind] += 1
    summary = {
        "version": "visual_edit_v1",
        "num_samples": args.num_samples,
        "seed": args.seed,
        "labels": str(args.labels),
        "fonts": [str(x) for x in fonts],
        "image_shape": [3, args.height, args.width],
        "stats": dict(stats),
        "note": "Rule-corrupted seeds are bootstrap only; generate frozen-CTC seeds before transfer evaluation.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
