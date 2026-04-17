import argparse
import logging
import os
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)


def _load_unet_state(model_path):
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict) and "unet" in state:
        return state["unet"]
    return state


def generate_images(model_path, prompts, output_dir, base_model="stable-diffusion-v1-5/stable-diffusion-v1-5", hf_token=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading base model %s", base_model)
    pipe_kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}
    if hf_token:
        pipe_kwargs["token"] = hf_token
    pipe = StableDiffusionPipeline.from_pretrained(base_model, **pipe_kwargs)

    logger.info("Loading checkpoint from %s", model_path)
    pipe.unet.load_state_dict(_load_unet_state(model_path), strict=False)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    for i, prompt in enumerate(prompts):
        logger.info("Generating sample %s for prompt: %s", i, prompt)
        image = pipe(prompt).images[0]
        image.save(output_dir / f"gen_{i}.png")


def build_parser():
    parser = argparse.ArgumentParser(description="Generate preview images from a trained checkpoint.")
    parser.add_argument("--model-path", required=True, help="Path to a checkpoint file or UNet state dict.")
    parser.add_argument("--output-dir", required=True, help="Directory to save generated images.")
    parser.add_argument("--prompt", action="append", default=[], help="Prompt to generate. Repeat for multiple.")
    parser.add_argument("--base-model", default="stable-diffusion-v1-5/stable-diffusion-v1-5", help="Diffusers base model id or local path.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token for gated models.")
    return parser


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if not args.prompt:
        raise ValueError("At least one --prompt is required")
    generate_images(
        args.model_path,
        args.prompt,
        args.output_dir,
        base_model=args.base_model,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
