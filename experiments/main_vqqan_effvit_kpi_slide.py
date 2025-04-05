import os
from glob import glob
import yaml
import torch
import argparse
from tqdm import tqdm 
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
import io

import PIL
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
# from dall_e import map_pixels, unmap_pixels
# from dall_e.encoder import Encoder
# from dall_e.decoder import Decoder



# Device setup
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")


# Model loading functions
def load_model(path, device):
    if path.startswith('http'):
        from urllib.request import urlopen
        with urlopen(path) as f:
            buf = io.BytesIO(f.read())
    else:
        with open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
    return torch.load(buf, map_location=device, weights_only=False)

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

# Preprocessing functions
def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x

def preprocess(img, target_image_size=256, map_dalle=True):
    s = min(img.size)
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    if map_dalle:
        img = map_pixels(img)
    return img

# Dataset for WSI patches
class WSIPatchDataset(Dataset):
    def __init__(self, patch_dir, target_size=256):
        self.patch_dir = patch_dir
        self.target_size = target_size
        self.image_paths = sorted(glob(os.path.join(patch_dir, "**/*_img.png"), recursive=True))
        if not self.image_paths:
            raise ValueError(f"No image patches found in {patch_dir}")
        print(f"Loaded {len(self.image_paths)} patches from {patch_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_vqgan = preprocess(img, target_image_size=self.target_size, map_dalle=False)
        # img_dalle = preprocess(img, target_image_size=self.target_size, map_dalle=True)
        return img_vqgan, os.path.basename(img_path)

# Encoding functions
def encode_with_vqgan(x, model):
    z, _, [_, _, indices] = model.encode(x)
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    print("Indices shape: ", indices.shape, "Indices unique: ", torch.unique(indices).shape)
    return z, indices

def encode_with_dalle(x, encoder):
    z_logits = encoder(x)
    z = torch.argmax(z_logits, axis=1)
    print(f"DALL-E: latent shape: {z.shape}")
    return z

# Main encoding pipeline
def encode_patches(dataloader, model32x32, encoder_dalle=None, save_dir=None):
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, (img_vqgan, filenames) in enumerate(tqdm(dataloader, desc="Encoding patches")):
    # for batch_idx, (img_vqgan, img_dalle, filenames) in enumerate(dataloader):
        img_vqgan = img_vqgan.to(DEVICE)
        
        # Encode with VQ-GAN models
        z_32x32, indices_32x32 = encode_with_vqgan(preprocess_vqgan(img_vqgan), model32x32)
        
        # Save latent representations
        for i, filename in enumerate(filenames):
            base_name = filename.replace("_img.png", "")
            features = z_32x32[i]
            indices = indices_32x32[i] 
            print("result")
            print("feature_shape: ", features.shape) 
            print("indice shape:", indices.shape)
            print("indices[:3]: ", indices[:3])  
            
            torch.save(z_32x32[i], os.path.join(save_dir, f"{base_name}_vqgan_32x32.pt"))
            torch.save(indices_32x32[i], os.path.join(save_dir, f"{base_name}_vqgan_32x32_indices.pt"))

        print(f"Processed batch {batch_idx + 1}/{len(dataloader)}")
        break 
    
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # Argument parserte 
    parser = argparse.ArgumentParser(description="Encode WSI patches with VQ-GAN and DALL-E")
    parser.add_argument("--config", type=str, default="configs/main_vqqan_effvit_kpi_slide.yaml", help="Path to YAML config file")
    parser.add_argument("--data_config", type=str, default="configs/kpis.yaml", help="Path to YAML config file")
    parser.add_argument("--train_test_val", type=str, default="train", help="Specify train/test/val")
    
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.data_config, 'r') as f:
        config.update(yaml.safe_load(f))
        
        
    # Validate config keys
    required_keys = ["data_dir", f"{args.train_test_val}_wsi_processed_patch_save_dir"]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise KeyError(f"Missing required config keys: {missing_keys}")

    if args.train_test_val not in ["train", "test", "val"]:
        raise ValueError("train_test_val must be one of 'train', 'test', or 'val'")
    
    vqgan_logs_dir = config.get('vqgan_logs_dir')
    config32x32 = load_config(f"{vqgan_logs_dir}/vqgan_gumbel_f8/configs/model.yaml", display=False)
    model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to(DEVICE)


    # Dataset and DataLoader
    data_dir = config["data_dir"]
    if args.train_test_val not in ["train", "test", "val"]:
        raise ValueError("train_test_val must be one of 'train', 'test', or 'val'")

    patch_dir = config[f"{args.train_test_val}_wsi_processed_patch_save_dir"]
    save_dir = os.path.join(config[f"{args.train_test_val}_feature_dir"])
    
    dataset = WSIPatchDataset(patch_dir, target_size=256)  # Adjust size as needed
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Encode patches
    encode_patches(dataloader, model32x32, save_dir=save_dir)
    print("Encoding complete")