import argparse
import logging
import os

from training.train import Trainer


def build_parser():
    parser=argparse.ArgumentParser(description="Train the layout-conditioned UI diffusion prototype.")
    parser.add_argument("--train-json",required=True,help="Path to the training split JSON file.")
    parser.add_argument("--val-json",default=None,help="Optional path to the validation split JSON file.")
    parser.add_argument("--tensor-dir",required=True,help="Directory containing layout tensors (*.npy).")
    parser.add_argument("--image-dir",required=True,help="Directory containing UI screenshots (*.jpg).")
    parser.add_argument("--output-dir",required=True,help="Directory to store checkpoints.")
    parser.add_argument("--epochs",type=int,default=1,help="Number of epochs to train for.")
    parser.add_argument("--batch-size",type=int,default=2,help="Training batch size.")
    parser.add_argument("--lr",type=float,default=1e-4,help="Learning rate.")
    parser.add_argument("--num-workers",type=int,default=2,help="DataLoader worker count.")
    parser.add_argument("--precision",choices=["fp32","fp16","bf16"],default="fp16",help="Weights precision to use.")
    parser.add_argument("--base-model",default="stable-diffusion-v1-5/stable-diffusion-v1-5",help="Diffusers base model id or local path.")
    parser.add_argument("--hf-token",default=None,help="Optional Hugging Face token for gated models.")
    parser.add_argument("--log-every",type=int,default=25,help="How often to log training step output.")
    parser.add_argument("--resume-from",default=None,help="Optional checkpoint path to resume training from.")
    return parser


def main():
    args=build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    config={
        "train_json":args.train_json,
        "val_json":args.val_json,
        "tensor_dir":args.tensor_dir,
        "image_dir":args.image_dir,
        "output_dir":args.output_dir,
        "batch_size":args.batch_size,
        "lr":args.lr,
        "num_workers":args.num_workers,
        "precision":args.precision,
        "base_model":args.base_model,
        "hf_token":args.hf_token,
        "log_every":args.log_every,
        "resume_from":args.resume_from,
    }
    logging.getLogger(__name__).info("Launching training with config: %s",config)
    trainer=Trainer(config)
    trainer.train(args.epochs)


if __name__=="__main__":
    main()
