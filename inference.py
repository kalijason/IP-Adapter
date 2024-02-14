from types import MethodType
import json
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image
import os

from ip_adapter import IPAdapter


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "/home/kalijason/git/IP-Adapter/models/image_encoder"

org_ip_ckpt = "/home/kalijason/git/IP-Adapter/models/ip-adapter_sd15.bin"
tryon_ip_ckpt = "/home/kalijason/git/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip_adapter_tryons.bin"
tryoncnetv11_ip_ckpt = "/home/kalijason/git/ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus/models/ip_adapter_tryons_controlnet_v11_10000.bin"

controlnet_model_path = "lllyasviel/control_v11p_sd15_openpose"

device = "cuda"
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
# load SD pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)

# load ip-adapter
org_ip_model = IPAdapter(pipe, image_encoder_path, org_ip_ckpt, device)
tryon_ip_model = IPAdapter(pipe, image_encoder_path, tryon_ip_ckpt, device)
# tryoncnetv11_ip_model = IPAdapter(pipe, image_encoder_path, tryoncnetv11_ip_ckpt, device)

# Define the path to your JSON file
file_path = '/home/kalijason/git/IP-Adapter/tryons_images.json'
tryon_image_folder = '/home/kalijason/train_images/tryons'
# Open the file and read the JSON data
with open(file_path, 'r') as file:
    tryons = json.load(file)
    total = 0
    for tryon in tryons:
        cloth_image = Image.open(os.path.join(tryon_image_folder, tryon['cloth_image_file']))
        tryon_image = Image.open(os.path.join(tryon_image_folder, tryon['tryon_image_file']))
        conditioning_image = Image.open(os.path.join(tryon_image_folder, tryon['conditioning_image_file']))

        # generate image variations
        org_out_image = org_ip_model.generate(
            pil_image=cloth_image, image=conditioning_image, num_samples=1, num_inference_steps=20, seed=42)[0]
        tryon_out_image = tryon_ip_model.generate(
            pil_image=cloth_image, image=conditioning_image, num_samples=1, num_inference_steps=20, seed=42)[0]
        # tryoncnetv11_out_image = tryoncnetv11_ip_model.generate(pil_image=cloth_image, image=conditioning_image, num_samples=1, num_inference_steps=20, seed=42)[0]

        grid = image_grid([
            cloth_image.resize((256, 256)), tryon_image.resize((256, 256)), conditioning_image.resize((256, 256)),
            org_out_image.resize((256, 256)), tryon_out_image.resize((256, 256))], 1, 5)
        grid.show()
        total += 1
        if total > 10:
            break
