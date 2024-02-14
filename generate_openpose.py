from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
import os
from tqdm import tqdm
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')
processor.to(device)

# read all images from a folder and process them
folder = '/home/kalijason/train_images/tryons/'
for filename in tqdm(os.listdir(folder)):
    if filename.endswith(".jpg") and filename.startswith("tryon"):
        newfullfilename = os.path.join(folder, filename.replace('tryon', 'pose'))
        # if os.path.exists(newfullfilename):
        # continue
        image = load_image(os.path.join(folder, filename))

        # full pose
        # pose_image = processor(image,  hand_and_face=True)
        # simple pose
        pose_image = processor(image)

        pose_image.save(newfullfilename)
