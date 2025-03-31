import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
# ---------- image reconstruction code : DECODER ONLY ------------ 

import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


from dall_e import map_pixels, unmap_pixels
from torch.serialization import add_safe_globals
from dall_e.encoder import Encoder
from dall_e.decoder import Decoder  # just in case decoder is also needed

add_safe_globals([Encoder, Decoder])  # allow custom classes

def load_model(path, device):
    if path.startswith('http'):
        from urllib.request import urlopen
        with urlopen(path) as f:
            buf = io.BytesIO(f.read())
    else:
        with open(path, 'rb') as f:
            buf = io.BytesIO(f.read())
    return torch.load(buf, map_location=device, weights_only=False)  # weights_only=True by default is now okay 
# from IPython.display import display, display_markdown
# ---------- image reconstruction code : DECODER ONLY ------------  

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

def preprocess_vqgan(x):
    x = 2.*x - 1.
    return x

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def reconstruct_with_vqgan(x, model):
    # could also use model(x) for reconstruction but use explicit encoding and decoding here
    z, _, [_, _, indices] = model.encode(x)
    
    print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
    print("indices shape: ", indices.shape, "indices unique: ", torch.unique(indices).shape) 
    xrec = model.decode(z)
    return xrec


font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)

def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))


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


def reconstruct_with_dalle(x, encoder, decoder, do_preprocess=False):
    # takes in tensor (or optionally, a PIL image) and returns a PIL image
    if do_preprocess:
        x = preprocess(x)
    z_logits = encoder(x)
    z = torch.argmax(z_logits, axis=1)
    
    print(f"DALL-E: latent shape: {z.shape}")
    z = F.one_hot(z, num_classes=encoder.vocab_size).permute(0, 3, 1, 2).float()

    x_stats = decoder(z).float()
    x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
    x_rec = T.ToPILImage(mode='RGB')(x_rec[0])

    return x_rec

def stack_reconstructions(input, x0, x1, x2, x3, titles=[]):
    assert input.size == x1.size == x2.size == x3.size
    w, h = input.size[0], input.size[1]
    img = Image.new("RGB", (5*w, h))
    img.paste(input, (0,0))
    img.paste(x0, (1*w,0))
    img.paste(x1, (2*w,0))
    img.paste(x2, (3*w,0))
    img.paste(x3, (4*w,0))
    
    # Load a larger font (adjust path and size as needed)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 28)
    except:
        font = ImageFont.load_default()
    
    # Add titles in black
    draw = ImageDraw.Draw(img)
    for i, title in enumerate(titles):
        draw.text((i*w + 10, 10), f'{title}', fill=(0, 0, 0), font=font)

    return img
 

# def stack_reconstructions(input, x0, x1, x2, x3, titles=[]):
#     assert input.size == x1.size == x2.size == x3.size
#     w, h = input.size[0], input.size[1]
#     img = Image.new("RGB", (5*w, h))
#     img.paste(input, (0,0))
#     img.paste(x0, (1*w,0))
#     img.paste(x1, (2*w,0))
#     img.paste(x2, (3*w,0))
#     img.paste(x3, (4*w,0))
#     for i, title in enumerate(titles):
#         ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255)) # coordinates, text, color, font
#     return img 

def reconstruction_pipeline(image, size=320, name=None): 
    x_dalle = preprocess(image, target_image_size=size, map_dalle=True)
    x_vqgan = preprocess(image, target_image_size=size, map_dalle=False)
    print("x_dalle, x_vqgan") 
    print(x_dalle.shape, x_vqgan.shape) 
    x_dalle = x_dalle.to(DEVICE)
    x_vqgan = x_vqgan.to(DEVICE)
    
    print(f"input is of size: {x_vqgan.shape}")
    x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model32x32)
    x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model16384)
    x2 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model1024)
    x3 = reconstruct_with_dalle(x_dalle, encoder_dalle, decoder_dalle)
    img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])), x3, 
                                custom_to_pil(x0[0]), custom_to_pil(x1[0]), 
                                custom_to_pil(x2[0]), titles=titles)
    
    tmp_dir = './tmp_save/'
    save_path = os.path.join(tmp_dir, f"reconstructions_{name}.jpg") 
    img.save(save_path) 
    print("done save the image") 
    print(f"Saved reconstruction to: {save_path}") 
    
def load_dalle_models():
    global encoder_dalle, decoder_dalle
    
    encoder_dalle = load_model("https://cdn.openai.com/dall-e/encoder.pkl", DEVICE)
    decoder_dalle = load_model("https://cdn.openai.com/dall-e/decoder.pkl", DEVICE)
 
if __name__=='__main__':
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print(f"DEVICE: {DEVICE}") 
    import sys
    sys.path.append(".")
    import torch
    torch.set_grad_enabled(False)
    load_dalle_models() 
    
    
    config1024 = load_config("logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    config16384 = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)

    model1024 = load_vqgan(config1024, ckpt_path="logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)
    model16384 = load_vqgan(config16384, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to(DEVICE)
    config32x32 = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
    model32x32 = load_vqgan(config32x32, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to(DEVICE)   
    
    print("---model1024 codebook shape:", model1024.quantize.embedding.weight.shape)
    
    titles = ["input", "DALL-E", "VQGAN 32x32", "VQGAN 1024", "VQGAN 16384"] 
    
    #--------
    import cv2  
    # img_path = "/project/hnguyen2/mvu9/datasets/kidney_pathology_image/train/Task1_patch_level/train/NEP25/08_373_01/mask/08_373_01_242_4096_14336_mask.jpg"
    
    # img = cv2.imread(img_path) 
    
    print('-----------------------------------------------------------------------------------------------')
    # img = Image.open(img_path) 
    
    url='https://assets.bwbx.io/images/users/iqjWHBFdfxIU/iKIWgaiJUtss/v2/1000x-1.jpg' 
    internet_image=download_image(url) 
    print(internet_image.size)
    
    from glob import glob
    image_paths = glob('/project/hnguyen2/mvu9/datasets/kidney_pathology_image/train/Task1_patch_level/train/NEP25/08_368_01/img/*.jpg')[:5]
        # img_path = '/project/hnguyen2/mvu9/datasets/kidney_pathology_image/train/Task1_patch_level/train/NEP25/08_368_01/img/08_368_01_54_8192_2048_img.jpg'
    for img_path in image_paths:  
        name = img_path.split('/')[-1].split('.')[0] 
        print("processing: ", name)
        img = Image.open(img_path)  
        reconstruction_pipeline(img, size=384, name=name) 
        
    print("------> done")  
    #conda environments:

