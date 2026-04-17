import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os


def generate_images(model_path, prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.unet.load_state_dict(torch.load(model_path))
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    for i, prompt in enumerate(prompts):
        image = pipe(prompt).images[0]
        image.save(os.path.join(output_dir, f"gen_{i}.png"))


