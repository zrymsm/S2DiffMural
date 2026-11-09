import numpy as np
import torch
import glob
import os
import cv2

from PIL import Image
from lpips import LPIPS
from tqdm import tqdm
from skimage.metrics import structural_similarity

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = LPIPS(net='alex').to(device).eval()

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_l1(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return np.mean(np.abs(img1 - img2))

def open_img_as_np(path):
    image = np.array(Image.open(path).convert("RGB")).astype(np.uint8)
    return image

def read_ssim_im(fpath):
    img = np.fromfile(fpath, dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
    return img

def open_img_as_torch(path):
    image = np.array(Image.open(path).convert("RGB"))
    image = image.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image)
    image = image * 2.0 - 1.0
    return image

def metrics_calc(gt_img_path, rec_img_path):
    total_ssim_list = []
    total_lpips = 0.0
    total_psnr = 0.0
    total_l1 = 0.0

    input = sorted(list(glob.glob(os.path.join(gt_img_path, '*.*'))))
    rec = sorted(list(glob.glob(os.path.join(rec_img_path, '*.*'))))

    print("total rec img nums is", len(rec))
    print("total gt img nums is", len(input))
    img_nums = len(input)

    # SSIM
    for i in tqdm(range(img_nums)):
        input_t = read_ssim_im(input[i])
        rec_img = read_ssim_im(rec[i])
        assert input_t.shape == rec_img.shape, f"SSIM shape mismatch: {input[i]} vs {rec[i]}"
        ssim_val = structural_similarity(input_t, rec_img)
        total_ssim_list.append(ssim_val)
    total_ssim = np.mean(total_ssim_list)

    # PSNR + LPIPS + L1
    for i in range(img_nums):
        # PSNR & L1
        input_np = open_img_as_np(input[i])
        rec_np = open_img_as_np(rec[i])
        assert input_np.shape == rec_np.shape, f"Shape mismatch: {input[i]} vs {rec[i]}"

        total_psnr += calculate_psnr(input_np, rec_np)
        total_l1 += calculate_l1(input_np, rec_np)

        # LPIPS
        input_t = open_img_as_torch(input[i]).unsqueeze(0)
        rec_t = open_img_as_torch(rec[i]).unsqueeze(0)
        with torch.no_grad():
            temp_lpips = lpips_fn(input_t.to(device), rec_t.to(device))
        total_lpips += temp_lpips.item()

    total_psnr /= img_nums
    total_lpips /= img_nums
    total_l1 /= img_nums

    print("↑--ssim={:.4f},  ↓--lpips={:.4f},  ↑--psnr={:.4f},  ↓--l1={:.4f}"
          .format(total_ssim, total_lpips, total_psnr, total_l1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default="./10/gt-10")
    parser.add_argument('--rec', type=str, default="./10/result-10")
    args = parser.parse_args()

    metrics_calc(args.gt, args.rec)
