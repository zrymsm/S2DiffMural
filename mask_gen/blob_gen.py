import os
import cv2
import math
import random
import shutil
import numpy as np
from tqdm import tqdm

def area_ratio(mask):
    return (mask > 0).sum() / mask.size

def random_blob_mask(h, w):
    mask = np.zeros((h, w), np.uint8)
    num_blobs = random.randint(3, 8)
    for _ in range(num_blobs):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        radius = random.randint(12, 48)
        pts = []
        n = random.randint(8, 16)
        for i in range(n):
            ang = 2 * math.pi * i / n
            r = radius * random.uniform(0.45, 1.2)
            x = int(cx + r * math.cos(ang))
            y = int(cy + r * math.sin(ang))
            pts.append([x, y])
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    return mask

def add_small_islands(mask):
    h, w = mask.shape
    num = random.randint(5, 24)
    for _ in range(num):
        cx = random.randint(0, w - 1)
        cy = random.randint(0, h - 1)
        r = random.randint(2, 7)
        if random.random() < 0.75:
            cv2.circle(mask, (cx, cy), r, 255, -1)
        else:
            rw = random.randint(4, 12)
            rh = random.randint(4, 12)
            angle = random.uniform(0, 180)
            box = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(np.int32)
            cv2.fillPoly(mask, [box], 255)
    return mask

def add_cracks(mask):
    h, w = mask.shape
    num = random.randint(1, 4)
    for _ in range(num):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        pts = [(x, y)]
        steps = random.randint(5, 14)
        angle = random.uniform(0, 2 * math.pi)
        for _ in range(steps):
            angle += random.uniform(-0.7, 0.7)
            length = random.randint(8, 20)
            x = int(np.clip(x + length * math.cos(angle), 0, w - 1))
            y = int(np.clip(y + length * math.sin(angle), 0, h - 1))
            pts.append((x, y))
        thickness = random.randint(2, 5)
        for i in range(len(pts) - 1):
            cv2.line(mask, pts[i], pts[i + 1], 255, thickness)
    return mask

def clean_irregular_boundary(mask):
    k1 = random.choice([5, 7, 9])
    k2 = random.choice([3, 5])
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
    if random.random() < 0.6:
        mask = cv2.dilate(mask, kernel1, iterations=1)
    if random.random() < 0.4:
        mask = cv2.erode(mask, kernel2, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    return (mask > 127).astype(np.uint8) * 255

def adjust_ratio(mask, min_ratio, max_ratio):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(120):
        ratio = area_ratio(mask)
        if min_ratio <= ratio <= max_ratio:
            return mask
        if ratio < min_ratio:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
        mask = (mask > 127).astype(np.uint8) * 255
    return mask

def generate_one_mask(h, w, min_ratio, max_ratio):
    for _ in range(300):
        mask = random_blob_mask(h, w)
        if random.random() < 0.85:
            mask = add_small_islands(mask)
        if random.random() < 0.35:
            mask = add_cracks(mask)
        mask = clean_irregular_boundary(mask)
        mask = adjust_ratio(mask, min_ratio, max_ratio)
        ratio = area_ratio(mask)
        if min_ratio <= ratio <= max_ratio:
            return mask
    return mask

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def generate_train_masks(root="./train_blob_mask", h=256, w=256):
    reset_dir(root)
    ratio_groups = [(0.01, 0.10), (0.10, 0.20), (0.20, 0.30)]
    masks = []
    for min_r, max_r in ratio_groups:
        for _ in tqdm(range(500), desc=f"train {int(min_r*100)}-{int(max_r*100)}"):
            mask = generate_one_mask(h, w, min_r, max_r)
            masks.append(mask)
    random.shuffle(masks)
    for i, mask in enumerate(tqdm(masks, desc="save train"), 1):
        cv2.imwrite(os.path.join(root, f"{i:04d}.png"), mask)

def generate_test_masks(root="./test_blob_mask", h=256, w=256):
    ratio_groups = [
        ("test_blob_mask_1_10", 0.01, 0.10),
        ("test_blob_mask_10_20", 0.10, 0.20),
        ("test_blob_mask_20_30", 0.20, 0.30)
    ]
    reset_dir(root)
    for folder, min_r, max_r in ratio_groups:
        save_dir = os.path.join(root, folder)
        os.makedirs(save_dir, exist_ok=True)
        for i in tqdm(range(1, 215), desc=folder):
            mask = generate_one_mask(h, w, min_r, max_r)
            cv2.imwrite(os.path.join(save_dir, f"{i:03d}.png"), mask)

def main():
    random.seed(42)
    np.random.seed(42)
    h, w = 256, 256
    generate_train_masks("./train_blob_mask", h, w)
    generate_test_masks("./test_blob_mask", h, w)
    print("Done.")
    print("Train: ./train_blob_mask, 1500 masks, 0001.png-1500.png")
    print("Test: ./test_blob_mask/test_blob_mask_1_10, 001.png-214.png")
    print("Test: ./test_blob_mask/test_blob_mask_10_20, 001.png-214.png")
    print("Test: ./test_blob_mask/test_blob_mask_20_30, 001.png-214.png")

if __name__ == "__main__":
    main()