import os
import numpy as np
import cv2
import random
from tqdm import tqdm
import shutil

def generate_mask_with_ratio_range(H, W, ratio_min, ratio_max, C=1, max_shapes=20, size_scale=0.01, max_retry=100):
    size = int((H + W) * size_scale)
    total_area = H * W * C

    for _ in range(max_retry):
        img = np.zeros((H, W, C), dtype=np.uint8)

        for _ in range(max_shapes):
            if random.random() > 0.5:
                center = (random.randint(0, W - 1), random.randint(0, H - 1))
                r = random.randint(size, size * 4)
                cv2.circle(img, center, r, (1,) * C, -1)
            elif random.random() > 0.5:
                center = (random.randint(0, W - 1), random.randint(0, H - 1))
                axes = (random.randint(size, size * 4), random.randint(size, size * 4))
                angle = random.randint(0, 180)
                cv2.ellipse(img, center, axes, angle, 0, 360, (1,) * C, -1)
            elif random.random() > 0.5:
                pt1 = (random.randint(0, W - 1), random.randint(0, H - 1))
                pt2 = (random.randint(0, W - 1), random.randint(0, H - 1))
                thickness = random.randint(size, size * 4)
                cv2.line(img, pt1, pt2, (1,) * C, thickness)
            else:
                center = (random.randint(0, W - 1), random.randint(0, H - 1))
                axes = (random.randint(1, W), random.randint(1, H))
                angle = random.randint(0, 180)
                start_angle = random.randint(size, 180)
                end_angle = random.randint(start_angle + 1, 360)
                thickness = random.randint(size, size * 4)
                cv2.ellipse(img, center, axes, angle, start_angle, end_angle, (1,) * C, thickness)

        ratio = np.sum(img) / total_area
        if ratio_min <= ratio <= ratio_max:
            return img * 255

    return None

def generate_masks_grouped(output_root="trainstrokemask", H=256, W=256):
    random.seed(0)
    np.random.seed(0)

    tmp_root = os.path.join(output_root, "tmp")
    final_root = os.path.join(output_root, "train_mask")

    os.makedirs(tmp_root, exist_ok=True)
    os.makedirs(final_root, exist_ok=True)

    ratio_groups = {
        "ratio10": (0.01, 0.10),
        "ratio20": (0.10, 0.20),
        "ratio30": (0.20, 0.30),
    }

    per_group = 500
    all_masks = []

    # Step 1: 每个比例生成 500 个
    for name, (low, high) in ratio_groups.items():
        out_dir = os.path.join(tmp_root, name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\nGenerating {per_group} masks for {name} [{low:.2f} ~ {high:.2f}]")
        count = 0
        with tqdm(total=per_group, desc=name) as pbar:
            while count < per_group:
                mask = generate_mask_with_ratio_range(H, W, low, high)
                if mask is not None:
                    filename = f"{count+1:04d}.png"
                    path = os.path.join(out_dir, filename)
                    cv2.imwrite(path, mask)
                    all_masks.append(path)
                    count += 1
                    pbar.update(1)

    # Step 2: 打乱顺序
    random.shuffle(all_masks)

    # Step 3: 统一重命名为 0001–1500
    print("\nShuffling and renaming masks to 0001–1500")
    for idx, src_path in enumerate(tqdm(all_masks)):
        dst_name = f"{idx+1:04d}.png"
        dst_path = os.path.join(final_root, dst_name)
        shutil.copy(src_path, dst_path)

    print(f"\nDone. Final training masks saved to: {final_root}")
    print("Total masks:", len(all_masks))

if __name__ == "__main__":
    generate_masks_grouped()
