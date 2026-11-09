import sys
import torch
import os
import glob

from PIL import Image
from tqdm import tqdm
import numpy as np

from omegaconf import OmegaConf
from utils import instantiate_from_config
from calc_metrics import metrics_calc

from torchvision import transforms
import albumentations as A
import argparse
import random


def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 固定 kernel 路径
    torch.backends.cudnn.benchmark = False     # 禁止动态调优


def tensor_to_image():
    return transforms.ToPILImage()

def image_to_tensor():
    return transforms.ToTensor()

def make_batch(image, mask, device):
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask_L = np.array(Image.open(mask).convert("L")).astype(np.uint8)
    transform = A.Resize(height=256, width=256)
    mask_L = transform(image=mask_L)['image']
    mask_L = mask_L.astype(np.float32)/255.0
    mask_L = mask_L[None, None]
    mask_L[mask_L < 0.5] = 0
    mask_L[mask_L >= 0.5] = 1
    mask_L = torch.from_numpy(mask_L)

    mask_RGB = np.array(Image.open(mask).convert("RGB"))
    mask_RGB = mask_RGB.astype(np.float32)/255.0
    mask_RGB = mask_RGB[None].transpose(0, 3, 1, 2)
    mask_RGB = torch.from_numpy(mask_RGB)

    masked_image = (1-mask_L)*image

    batch = {"image": image, "mask": mask_L, "masked_image": masked_image,
             "mask_RGB": mask_RGB}

    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch

def test():

    set_seed(23)



    opts = {
        "ckpt_path": "./logs/2026-01-15T23-15-35/checkpoints/last.ckpt",
        "input_path": "/home/tomry/code/work2_mural_inpainting/20260530最新模型/数据/20260530新数据集-linux/原始壁画图像/test",
        "mask_path": "/home/tomry/code/work2_mural_inpainting/20260530最新模型/数据/20260530新数据集-linux/第一类stroke-mask/test_mask-3/test_mask30",
        "output_path": "./result-30",
        "masked_output_path": "./masked-30",
        "gt_output_path": "./gt-30"
    }

    config = OmegaConf.load(os.path.join(sys.path[0], "confs/denoise-network.yaml"))

    # argparse_dict = vars(opts)

    argparse_dict = opts
    merged_config = OmegaConf.merge(config, OmegaConf.create(argparse_dict))

    test_img_path = merged_config.input_path
    test_mask_path = merged_config.mask_path

    test_data = sorted(glob.glob(os.path.join(test_img_path, "*.*")))
    masks = sorted(glob.glob(os.path.join(test_mask_path, "*.*")))

    print(f"Finding {len(test_data)} images and {len(masks)} masks")

    model = instantiate_from_config(config.model)

    output_path = merged_config.output_path

    model.load_state_dict(torch.load(merged_config.ckpt_path)["state_dict"],
                          strict=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)

    os.makedirs(output_path, exist_ok=True)

    print("Strat sampling.........")

    with torch.no_grad():
        for image, mask in tqdm(zip(test_data, masks)):
            outpath = os.path.join(sys.path[0], output_path, os.path.split(image)[1])
            batch = make_batch(image, mask, device=device)

            # encode masked image and concat downsampled mask

            x0 = batch["masked_image"]
            shape = batch["masked_image"].shape[1:]
            cond = torch.cat((batch["masked_image"], batch["mask"]), dim=1)

            mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                               min=0.0, max=1.0)

            masked_image = torch.clamp((batch["masked_image"] + 1.0) / 2.0,
                                       min=0.0, max=1.0)


            predicted_image = model.p_sample_loop(
                            step=25,
                            cond=cond,
                            batch_size=cond.shape[0],
                            shape=shape,
                            mask=mask,
                            x0=x0,
                            )

            predicted_image = torch.clamp((predicted_image+1.0)/2.0, min=0.0, max=1.0)

            inpainted = (1-mask)*masked_image+mask*predicted_image

            masked_out_dir = merged_config.masked_output_path
            os.makedirs(masked_out_dir, exist_ok=True)
            masked_output_path = os.path.join(masked_out_dir, os.path.split(image)[1])

            masked_np = (masked_image[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            masked_img = Image.fromarray(masked_np)
            masked_img.save(masked_output_path)


            output = (inpainted[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            output = Image.fromarray(output)
            output.save(outpath)

            # ================= 保存 Ground Truth 原图 =================
            gt_out_dir = merged_config.gt_output_path
            os.makedirs(gt_out_dir, exist_ok=True)
            gt_output_path = os.path.join(gt_out_dir, os.path.split(image)[1])

            gt_image = Image.open(image).convert("RGB")
            gt_image.save(gt_output_path)
            # ==========================================================

    metrics_calc(os.path.join(sys.path[0], test_img_path), output_path)

if __name__ == "__main__":
    test()