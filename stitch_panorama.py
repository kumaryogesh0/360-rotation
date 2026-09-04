#!/usr/bin/env python3
"""
Stitch a sequence of overlapping rotation photos into one wide panorama,
then pad it to a 2:1 canvas so it can be used as the `data-image` in
index.html (the panorama viewer expects roughly equirectangular, 2:1 images).

How to shoot the source photos:
  - Stand in one spot on a tripod (or steady hands) and only rotate — don't
    move position between shots.
  - Keep the camera height/tilt level and the same for every shot.
  - Overlap each shot ~30-40% with the previous one.
  - Lock exposure/focus (or use the same auto settings) so brightness
    doesn't jump between frames — visible seams are usually exposure jumps.
  - Name the files in capture order, e.g. 01.jpg, 02.jpg, ... 30.jpg, so
    sorting them alphabetically reproduces the rotation order.

Usage:
  python3 stitch_panorama.py <folder_of_photos> [output.jpg]

Example:
  python3 stitch_panorama.py ./shoot1 panorama.jpg
"""
import sys
import glob
import os
import cv2
import numpy as np

def load_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(folder, e)))
    paths = sorted(set(paths))
    if len(paths) < 2:
        raise SystemExit(f"Found only {len(paths)} image(s) in {folder} — need at least 2.")
    print(f"Found {len(paths)} images, in this order:")
    for p in paths:
        print("  ", os.path.basename(p))
    imgs = [cv2.imread(p) for p in paths]
    bad = [p for p, im in zip(paths, imgs) if im is None]
    if bad:
        raise SystemExit(f"Could not read: {bad}")
    return imgs

def crop_to_content(img):
    """Trim the ragged black border stitching typically leaves behind."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)

    # Shrink a tight axis-aligned box inward until it's fully inside the
    # stitched content (handles the slightly wavy panorama border).
    x, y, w, h = cv2.boundingRect(largest)
    box = thresh[y:y + h, x:x + w]
    while box.shape[0] > 10 and box.shape[1] > 10:
        top_ok = np.all(box[0, :] > 0)
        bottom_ok = np.all(box[-1, :] > 0)
        left_ok = np.all(box[:, 0] > 0)
        right_ok = np.all(box[:, -1] > 0)
        if top_ok and bottom_ok and left_ok and right_ok:
            break
        if not top_ok:
            box = box[1:, :]; y += 1; h -= 1
        if not bottom_ok:
            box = box[:-1, :]; h -= 1
        if not left_ok:
            box = box[:, 1:]; x += 1; w -= 1
        if not right_ok:
            box = box[:, :-1]; w -= 1
    return img[y:y + h, x:x + w]

def pad_to_2x1(img):
    """Letterbox the stitched band onto a 2:1 canvas (black top/bottom bars),
    since a single-row rotation shoot only covers a horizontal band, not
    the full sphere up to the poles."""
    h, w = img.shape[:2]
    target_h = w // 2
    if h >= target_h:
        # Wider band than 2:1 expects — just resize down to fit height.
        scale = target_h / h
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, target_h))
        h, w = img.shape[:2]
        canvas = np.zeros((target_h, target_h * 2, 3), dtype=np.uint8)
        x_off = (canvas.shape[1] - w) // 2
        canvas[:, x_off:x_off + w] = img
        return canvas
    canvas = np.zeros((target_h, w, 3), dtype=np.uint8)
    y_off = (target_h - h) // 2
    canvas[y_off:y_off + h, :] = img
    return canvas

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    folder = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "panorama.jpg"

    imgs = load_images(folder)

    print("Stitching... this can take a minute for 30 full-resolution photos.")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)

    if status != cv2.Stitcher_OK:
        codes = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough overlap between images to find matches — reshoot with more overlap (30-40%) between consecutive photos.",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Couldn't reliably align the images — check they're all from the same spot, same height, and in the right order.",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter estimation failed — try more consistent overlap/exposure between shots.",
        }
        raise SystemExit("Stitching failed: " + codes.get(status, f"error code {status}"))

    print(f"Stitched raw panorama: {pano.shape[1]}x{pano.shape[0]}")
    pano = crop_to_content(pano)
    print(f"Cropped to content: {pano.shape[1]}x{pano.shape[0]}")
    pano = pad_to_2x1(pano)
    print(f"Padded to 2:1: {pano.shape[1]}x{pano.shape[0]}")

    cv2.imwrite(out_path, pano, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Saved: {out_path}")
    print("Copy/move it into the 360 project folder and point a tab's")
    print(f"data-image attribute in index.html at \"{os.path.basename(out_path)}\".")

if __name__ == "__main__":
    main()
