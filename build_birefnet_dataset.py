"""
BiRefNet Thin Structure Dataset Builder & Fractal Quantifier
============================================================
Creates a curated dataset of thin structure items using exclusively
segmented images from the BiRefNet segmentation model.

Organizes data into:
- dataset/low_thin_structure/
- dataset/mid_thin_structure/
- dataset/high_complex_thin_structure/

Includes root segmented images (image (10).png, image (5).png, image (11).png)
and extracted standardized segmented patches, then quantifies fractal dimensions
(DBC, Blanket, Variogram, Morphological) and generates a metadata catalog.
"""

import os
import shutil
import json
import csv
from typing import Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from fractal_analyzer import FractalAnalyzer, create_highlight_overlay


def analyze_thin_structure(img_rgba: np.ndarray) -> Dict[str, float]:
    """Computes physical thin structure metrics: thickness, skeleton length, active ratio."""
    if len(img_rgba.shape) > 2 and img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3]
    else:
        gray = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY) if len(img_rgba.shape) == 3 else img_rgba
        alpha = (gray > 15).astype(np.uint8) * 255

    bin_mask = (alpha > 30).astype(np.uint8)
    total_pixels = float(bin_mask.size)
    active_pixels = int(np.count_nonzero(bin_mask))
    active_ratio = (active_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    if active_pixels > 0:
        dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
        mean_thickness = float(dist[bin_mask > 0].mean() * 2.0)
        max_thickness = float(dist.max() * 2.0)
    else:
        mean_thickness, max_thickness = 0.0, 0.0

    scale = 1.0
    if bin_mask.shape[0] > 1024 or bin_mask.shape[1] > 1024:
        scale = 1024.0 / max(bin_mask.shape)
        sm_w = int(bin_mask.shape[1] * scale)
        sm_h = int(bin_mask.shape[0] * scale)
        eval_mask = cv2.resize(bin_mask, (sm_w, sm_h), interpolation=cv2.INTER_NEAREST)
    else:
        eval_mask = bin_mask

    skel = np.zeros_like(eval_mask)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_mask = eval_mask.copy()
    while np.count_nonzero(temp_mask) > 0:
        eroded = cv2.erode(temp_mask, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_mask, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_mask = eroded.copy()

    skeleton_len = int(np.count_nonzero(skel) * (1.0 / scale))
    edges = cv2.Canny(bin_mask * 255, 50, 150)
    edge_density = float(np.mean(edges > 0))

    return {
        'active_pixels': active_pixels,
        'active_ratio_pct': active_ratio,
        'mean_thickness_px': mean_thickness,
        'max_thickness_px': max_thickness,
        'skeleton_length_px': skeleton_len,
        'edge_density': edge_density,
    }


def build_dataset(root_dir=".", dataset_dir="./dataset", patch_size=512) -> List[Dict]:
    """Builds the 3-tier BiRefNet segmented thin structure dataset."""
    os.makedirs(dataset_dir, exist_ok=True)
    tiers = {
        'low': os.path.join(dataset_dir, "low_thin_structure"),
        'mid': os.path.join(dataset_dir, "mid_thin_structure"),
        'high': os.path.join(dataset_dir, "high_complex_thin_structure")
    }
    for t_path in tiers.values():
        os.makedirs(t_path, exist_ok=True)

    records = []
    root_mappings = [
        ("image (10).png", "low", "root_birefnet_image_10.png"),
        ("image (5).png", "mid", "root_birefnet_image_5.png"),
        ("image (11).png", "high", "root_birefnet_image_11.png"),
    ]

    print("[1/4] Processing root BiRefNet segmented images...")
    for orig_name, tier, dest_name in root_mappings:
        orig_path = os.path.join(root_dir, orig_name)
        if not os.path.exists(orig_path):
            continue

        dest_path = os.path.join(tiers[tier], dest_name)
        shutil.copyfile(orig_path, dest_path)
        img = cv2.imread(dest_path, cv2.IMREAD_UNCHANGED)
        h, w = img.shape[:2]
        struct_metrics = analyze_thin_structure(img)

        records.append({
            'filename': dest_name,
            'rel_path': os.path.relpath(dest_path, dataset_dir),
            'abs_path': os.path.abspath(dest_path),
            'tier': f"{tier}_thin_structure",
            'source_type': "root_birefnet_image",
            'parent_image': orig_name,
            'height': h,
            'width': w,
            **struct_metrics
        })

    print("[2/4] Extracting curated segmented patches...")
    img_10 = cv2.imread(os.path.join(root_dir, "image (10).png"), cv2.IMREAD_UNCHANGED)
    if img_10 is not None:
        h10, w10 = img_10.shape[:2]
        for idx, p in enumerate([img_10[20:20+patch_size, 20:20+patch_size], img_10[h10-patch_size-10:h10-10, 50:50+patch_size]], 1):
            if p.shape[0] == patch_size and p.shape[1] == patch_size:
                p_name = f"birefnet_patch_low_sparse_{idx}.png"
                p_path = os.path.join(tiers['low'], p_name)
                cv2.imwrite(p_path, p)
                records.append({
                    'filename': p_name,
                    'rel_path': os.path.relpath(p_path, dataset_dir),
                    'abs_path': os.path.abspath(p_path),
                    'tier': "low_thin_structure",
                    'source_type': "birefnet_patch",
                    'parent_image': "image (10).png",
                    'height': patch_size,
                    'width': patch_size,
                    **analyze_thin_structure(p)
                })

    img_5 = cv2.imread(os.path.join(root_dir, "image (5).png"), cv2.IMREAD_UNCHANGED)
    if img_5 is not None:
        h5, w5 = img_5.shape[:2]
        for idx, p in enumerate([img_5[40:40+patch_size, 40:40+patch_size], img_5[h5-patch_size-20:h5-20, 80:80+patch_size]], 1):
            if p.shape[0] == patch_size and p.shape[1] == patch_size:
                p_name = f"birefnet_patch_mid_branching_{idx}.png"
                p_path = os.path.join(tiers['mid'], p_name)
                cv2.imwrite(p_path, p)
                records.append({
                    'filename': p_name,
                    'rel_path': os.path.relpath(p_path, dataset_dir),
                    'abs_path': os.path.abspath(p_path),
                    'tier': "mid_thin_structure",
                    'source_type': "birefnet_patch",
                    'parent_image': "image (5).png",
                    'height': patch_size,
                    'width': patch_size,
                    **analyze_thin_structure(p)
                })

    print("[3/4] Quantifying Fractal Dimensions...")
    analyzer = FractalAnalyzer(window_size=21)
    for rec in records:
        fpath = rec['abs_path']
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) > 2 and img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32) / 255.0
            gray_base = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray = (gray_base * alpha).astype(np.uint8)
        elif len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if gray.shape[0] > 800 or gray.shape[1] > 800:
            scale_f = 600.0 / max(gray.shape)
            gray = cv2.resize(gray, (int(gray.shape[1]*scale_f), int(gray.shape[0]*scale_f)))

        analyzer.load_image(gray)
        dbc_mat = analyzer.compute_map('dbc')
        blanket_mat = analyzer.compute_map('blanket')
        vario_mat = analyzer.compute_map('variogram')
        morph_mat = analyzer.compute_map('morphological')
        _, dbc_th, dbc_stats = analyzer.threshold_map('dbc', 'percentile', 85.0)

        rec['dbc_fd_mean'] = round(float(dbc_stats['mean']), 4)
        rec['dbc_fd_max'] = round(float(dbc_stats['max']), 4)
        rec['dbc_high_cutoff'] = round(float(dbc_th), 4)
        rec['blanket_fd_mean'] = round(float(np.mean(blanket_mat)), 4)
        rec['variogram_fd_mean'] = round(float(np.mean(vario_mat)), 4)
        rec['morphological_fd_mean'] = round(float(np.mean(morph_mat)), 4)

    print("[4/4] Exporting Dataset Catalog...")
    csv_path = os.path.join(dataset_dir, "dataset_catalog.csv")
    json_path = os.path.join(dataset_dir, "dataset_catalog.json")
    if records:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

    print("[SUCCESS] Dataset built successfully!")
    return records


if __name__ == "__main__":
    build_dataset()
