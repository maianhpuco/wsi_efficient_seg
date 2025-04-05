import os
from glob import glob
import tifffile
import numpy as np
import imageio
import scipy.ndimage as ndi
from tqdm import tqdm
import yaml
import argparse
import json
import datetime

def get_next_log_number(log_dir):
    """Find the next available log number based on existing log files."""
    existing_logs = glob(os.path.join(log_dir, "log_*.json"))
    if not existing_logs:
        return 1
    numbers = [int(os.path.basename(f).replace("log_", "").replace(".json", "")) for f in existing_logs]
    return max(numbers) + 1

def extract_and_save_patches(wsi_dir, save_dir, log_dir, patch_size=2048, stride=1024):
    # Get all WSI image files
    image_paths = sorted(glob(os.path.join(wsi_dir, "**/*_wsi.tiff"), recursive=True))
    total_images = len(image_paths)
    
    # Initialize log data
    log_data = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "total_images": total_images,
        "processed_files": []
    }
    
    # Process each image and its corresponding mask
    for file_idx, img_path in enumerate(image_paths, 1):
        print(f"Processing file {file_idx}/{total_images}: {os.path.basename(img_path)}")
        
        # Determine corresponding mask path
        mask_path = img_path.replace("_wsi.tiff", "_mask.tiff")
        if not os.path.exists(mask_path):
            print(f"Warning: Mask file not found for {img_path}, skipping...")
            log_data["processed_files"].append({
                "tiff_name": os.path.basename(img_path),
                "status": "skipped",
                "expected_patches": 0,
                "actual_patches": 0
            })
            continue

        # Determine resolution level
        lv = 1 if 'NEP25' in img_path else 2
        key = 0  # Use first page

        # Load and downsample WSI image
        wsi_img = tifffile.imread(img_path, key=key)
        wsi_img = ndi.zoom(wsi_img, (1 / lv, 1 / lv, 1), order=1)  # RGB image

        # Load and downsample mask
        wsi_mask = tifffile.imread(mask_path, key=key)
        wsi_mask = ndi.zoom(wsi_mask, (1 / lv, 1 / lv), order=0)  # Binary mask, nearest neighbor

        # Save full 20X WSI as PNG
        out_img_path = img_path.replace(wsi_dir, save_dir).replace(".tiff", ".png")
        out_mask_path = mask_path.replace(wsi_dir, save_dir).replace(".tiff", ".png")
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)
        imageio.imwrite(out_img_path, wsi_img)
        imageio.imwrite(out_mask_path, wsi_mask)

        # Calculate expected patches
        h, w = wsi_img.shape[:2]
        x_slide = (h - patch_size) // stride + 1
        y_slide = (w - patch_size) // stride + 1
        expected_patches = x_slide * y_slide

        # Subfolder and WSI ID for saving patches
        subfolder = os.path.dirname(img_path).split("/")[-1]
        wsi_id = os.path.basename(img_path).replace("_wsi.tiff", "")
        save_path = os.path.join(save_dir, subfolder, wsi_id)
        os.makedirs(save_path, exist_ok=True)

        # Extract patches and count actual patches written
        actual_patches = 0
        with tqdm(total=expected_patches, desc=f"Extracting patches for {wsi_id}") as pbar:
            for xi in range(x_slide):
                for yi in range(y_slide):
                    now_x = xi * stride if xi != x_slide - 1 else h - patch_size
                    now_y = yi * stride if yi != y_slide - 1 else w - patch_size

                    # Extract image patch
                    img_patch = wsi_img[now_x:now_x + patch_size, now_y:now_y + patch_size, :]
                    assert img_patch.shape == (patch_size, patch_size, 3)

                    # Extract mask patch
                    mask_patch = wsi_mask[now_x:now_x + patch_size, now_y:now_y + patch_size]
                    assert mask_patch.shape == (patch_size, patch_size)

                    # Save patches
                    img_fname = f"{subfolder}_{wsi_id}_{xi}_{now_x}_{now_y}_img.png"
                    mask_fname = f"{subfolder}_{wsi_id}_{xi}_{now_x}_{now_y}_mask.png"
                    imageio.imwrite(os.path.join(save_path, img_fname), img_patch)
                    imageio.imwrite(os.path.join(save_path, mask_fname), mask_patch)

                    actual_patches += 1
                    pbar.update(1)

        # Log this file's results
        log_data["processed_files"].append({
            "tiff_name": os.path.basename(img_path),
            "status": "processed",
            "expected_patches": expected_patches,
            "actual_patches": actual_patches
        })

    # Save log file
    os.makedirs(log_dir, exist_ok=True)
    log_number = get_next_log_number(log_dir)
    log_file = os.path.join(log_dir, f"log_{log_number:03d}.json")
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"Log saved to {log_file}")

if __name__ == "__main__":
    # Argument parser for YAML config file
    parser = argparse.ArgumentParser(description="Extract patches from WSI TIFFs")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required config keys
    required_keys = ["data_dir", "train_wsi_dir", "train_wsi_processed_patch_save_dir"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

    data_dir = config["data_dir"]
    train_wsi_dir = config["train_wsi_dir"]
    train_wsi_processed_patch_save_dir = config["train_wsi_processed_patch_save_dir"]
    
    # Use log_dir from config if provided, otherwise default to data_dir/logs/kpis_patching
    log_dir = config.get("log_dir", os.path.join(data_dir, "logs", "kpis_patching"))
    
    # Create the save directory if it doesn't exist
    os.makedirs(train_wsi_processed_patch_save_dir, exist_ok=True)
    
    # Run the extraction with logging
    extract_and_save_patches(
        train_wsi_dir,
        train_wsi_processed_patch_save_dir,
        log_dir,
        patch_size=2048,
        stride=1024
    )