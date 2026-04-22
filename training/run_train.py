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
    parser.add_argument("--epochs",type=int,default=12,help="Number of epochs to train for.")
    parser.add_argument("--batch-size",type=int,default=4,help="Per-GPU training batch size when using torchrun.")
    parser.add_argument("--lr",type=float,default=8e-5,help="Learning rate.")
    parser.add_argument("--weight-decay",type=float,default=0.01,help="AdamW weight decay.")
    parser.add_argument("--max-grad-norm",type=float,default=1.0,help="Gradient clipping norm; 0 disables clipping.")
    parser.add_argument("--num-workers",type=int,default=4,help="DataLoader worker count per process.")
    parser.add_argument("--grad-accum-steps",type=int,default=1,help="Number of batches to accumulate before an optimizer step.")
    parser.add_argument("--precision",choices=["fp32","fp16","bf16"],default="bf16",help="Weights precision to use.")
    parser.add_argument("--train-sample-limit",type=int,default=30000,help="Target number of training samples; defaults to 30000 and is capped at 30000.")
    parser.add_argument("--sample-seed",type=int,default=42,help="Random seed used when subsampling the training set.")
    parser.add_argument("--base-model",default="stable-diffusion-v1-5/stable-diffusion-v1-5",help="Diffusers base model id or local path.")
    parser.add_argument("--hf-token",default=None,help="Optional Hugging Face token for gated models.")
    parser.add_argument("--save-unet-in-checkpoint",action="store_true",help="Include frozen UNet weights in checkpoints (large files).")
    parser.add_argument("--max-checkpoints",type=int,default=2,help="How many checkpoint_epoch_*.pt files to keep (0 disables pruning).")
    parser.add_argument("--log-every",type=int,default=20,help="How often to log training step output.")
    parser.add_argument("--resume-from",default=None,help="Optional checkpoint path to resume training from.")
    parser.add_argument("--max-train-hours",type=float,default=0.0,help="Optional wall-clock time limit in hours; 0 disables the limit.")
    return parser


def main():
    args=build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "INFO")
    # Add a file handler so logs are persisted in batch runs.
    try:
        rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or "0"
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        log_file = os.environ.get("LOG_FILE")
        if not log_file:
            scratch = os.environ.get("SCRATCH_ROOT", "/mnt/scratch/ui-gen")
            log_file = os.path.join(scratch, "train.log")
        # allow templated filenames like '/path/train.{rank}.log'
        if "{rank}" in log_file:
            log_file = log_file.format(rank=rank)
        elif world_size > 1:
            base, ext = os.path.splitext(log_file)
            log_file = f"{base}.rank{rank}{ext}"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logging.getLogger().addHandler(fh)
        logging.getLogger(__name__).info("Logging to %s", log_file)
    except Exception:
        logging.getLogger(__name__).warning("Could not configure file logging; continuing with console only.")
    config={
        "train_json":args.train_json,
        "val_json":args.val_json,
        "tensor_dir":args.tensor_dir,
        "image_dir":args.image_dir,
        "output_dir":args.output_dir,
        "batch_size":args.batch_size,
        "lr":args.lr,
        "weight_decay":args.weight_decay,
        "max_grad_norm":args.max_grad_norm,
        "num_workers":args.num_workers,
        "grad_accum_steps":args.grad_accum_steps,
        "precision":args.precision,
        "train_sample_limit":args.train_sample_limit,
        "sample_seed":args.sample_seed,
        "base_model":args.base_model,
        "hf_token":args.hf_token,
        "save_unet_in_checkpoint":args.save_unet_in_checkpoint,
        "max_checkpoints":args.max_checkpoints,
        "log_every":args.log_every,
        "resume_from":args.resume_from,
        "max_train_hours":args.max_train_hours,
    }
    logging.getLogger(__name__).info("Launching training with config: %s",config)
    trainer=Trainer(config)
    trainer.train(args.epochs)


if __name__=="__main__":
    main()
