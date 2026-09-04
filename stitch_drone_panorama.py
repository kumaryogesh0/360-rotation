#!/usr/bin/env python3
"""
Build one equirectangular panorama from a set of drone photos by placing
each one on the sphere using its embedded gimbal yaw/pitch (from DJI XMP
metadata), rather than blind feature-matching. Correct approach for a
multi-row "sphere panorama" shoot (nadir + rings at different tilts),
where extreme angle differences between shots make feature matching
unreliable.

lon/lat convention matches the panorama viewer in index.html:
  lon = yaw   (0-360, wraps)
  lat = pitch (+90 = straight up, -90 = straight down)

Usage:
  python3 stitch_drone_panorama.py <folder_of_photos> [output.jpg] [--width 4096]
"""
import sys
import re
import glob
import os
import argparse
import numpy as np
import cv2


def read_gimbal_xmp(path):
    data = open(path, "rb").read()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        return None
    xmp = data[start:end + 12].decode("utf-8", errors="ignore")

    def g(key):
        m = re.search(key + r'="([^"]+)"', xmp)
        return float(m.group(1)) if m else None

    yaw = g("GimbalYawDegree")
    pitch = g("GimbalPitchDegree")
    roll = g("GimbalRollDegree")
    if yaw is None or pitch is None:
        return None
    return {"yaw": yaw, "pitch": pitch, "roll": roll or 0.0}


def read_focal35(path):
    data = open(path, "rb").read()
    # EXIF FocalLengthIn35mmFilm tag (0xA405) — scan raw bytes is fragile,
    # so fall back to a fixed, sane default (this drone: 24mm equiv) if unknown.
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(path)
        exif = img._getexif() or {}
        for tag_id, val in exif.items():
            if TAGS.get(tag_id, tag_id) == "FocalLengthIn35mmFilm":
                return float(val)
    except Exception:
        pass
    return 24.0


def spherical_to_cartesian(lon_deg, lat_deg):
    phi = np.radians(90.0 - lat_deg)
    theta = np.radians(lon_deg)
    x = np.sin(phi) * np.cos(theta)
    y = np.cos(phi)
    z = np.sin(phi) * np.sin(theta)
    return np.array([x, y, z])


def camera_basis(yaw_deg, pitch_deg):
    forward = spherical_to_cartesian(yaw_deg, pitch_deg)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(forward, world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("output", nargs="?", default="drone_panorama.jpg")
    ap.add_argument("--width", type=int, default=4096)
    args = ap.parse_args()

    out_w = args.width
    out_h = out_w // 2

    paths = sorted(glob.glob(os.path.join(args.folder, "*.JPG"))) + \
            sorted(glob.glob(os.path.join(args.folder, "*.jpg")))
    paths = sorted(set(paths))

    shots = []
    skipped = []
    for p in paths:
        meta = read_gimbal_xmp(p)
        if meta is None:
            skipped.append(p)
            continue
        shots.append((p, meta))

    if skipped:
        print("Skipping (no gimbal metadata found):")
        for p in skipped:
            print("  ", os.path.basename(p))

    if len(shots) < 2:
        raise SystemExit("Not enough images with usable gimbal metadata.")

    print(f"Placing {len(shots)} images by their gimbal yaw/pitch...")

    # Output grid: lon/lat and world direction for every canvas pixel, once.
    xs = np.arange(out_w)
    ys = np.arange(out_h)
    lon = (xs + 0.5) / out_w * 360.0            # 0..360
    lat = 90.0 - (ys + 0.5) / out_h * 180.0      # +90 (top) .. -90 (bottom)
    lon_grid, lat_grid = np.meshgrid(lon, lat)   # (out_h, out_w)
    phi = np.radians(90.0 - lat_grid)
    theta = np.radians(lon_grid)
    Dx = np.sin(phi) * np.cos(theta)
    Dy = np.cos(phi)
    Dz = np.sin(phi) * np.sin(theta)

    accum = np.zeros((out_h, out_w, 3), dtype=np.float64)
    weight_accum = np.zeros((out_h, out_w), dtype=np.float64)

    for idx, (path, meta) in enumerate(shots):
        img = cv2.imread(path)
        if img is None:
            print("  could not read", path); continue
        h, w = img.shape[:2]
        focal35 = read_focal35(path)
        focal_px = w * focal35 / 36.0

        right, up, forward = camera_basis(meta["yaw"], meta["pitch"])

        x_cam = Dx * right[0] + Dy * right[1] + Dz * right[2]
        y_cam = Dx * up[0] + Dy * up[1] + Dz * up[2]
        z_cam = Dx * forward[0] + Dy * forward[1] + Dz * forward[2]

        valid = z_cam > 1e-3
        z_safe = np.where(valid, z_cam, 1.0)
        px = (w / 2.0 + (x_cam / z_safe) * focal_px).astype(np.float32)
        py = (h / 2.0 - (y_cam / z_safe) * focal_px).astype(np.float32)

        in_bounds = valid & (px >= 0) & (px < w) & (py >= 0) & (py < h)

        sampled = cv2.remap(img, px, py, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Feather weight: 1 at image center, fading to 0 near its edges,
        # so overlapping images blend instead of showing hard seams.
        nx = np.clip((px - w / 2.0) / (w / 2.0), -1, 1)
        ny = np.clip((py - h / 2.0) / (h / 2.0), -1, 1)
        edge_dist = np.maximum(np.abs(nx), np.abs(ny))  # 0 center .. 1 edge
        weight = np.clip(1.0 - edge_dist, 0.0, 1.0) ** 1.5
        weight = np.where(in_bounds, weight, 0.0)

        accum += sampled.astype(np.float64) * weight[..., None]
        weight_accum += weight

        print(f"  [{idx+1}/{len(shots)}] {os.path.basename(path)}  "
              f"yaw={meta['yaw']:.1f} pitch={meta['pitch']:.1f}  "
              f"coverage={100*np.count_nonzero(in_bounds)/in_bounds.size:.1f}%")

    covered = weight_accum > 1e-6
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[covered] = (accum[covered] / weight_accum[covered, None]).clip(0, 255).astype(np.uint8)

    total_coverage = 100 * np.count_nonzero(covered) / covered.size
    print(f"Total sphere coverage: {total_coverage:.1f}% (uncovered areas are black)")

    cv2.imwrite(args.output, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved: {args.output}  ({out_w}x{out_h})")


if __name__ == "__main__":
    main()
