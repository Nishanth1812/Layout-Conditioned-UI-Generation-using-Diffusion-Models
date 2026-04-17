import argparse

from training.train import Trainer


def build_parser():
    parser=argparse.ArgumentParser(description="Train the layout-conditioned UI diffusion prototype.")
    parser.add_argument("--train-json",required=True,help="Path to the training split JSON file.")
    parser.add_argument("--tensor-dir",required=True,help="Directory containing layout tensors (*.npy).")
    parser.add_argument("--image-dir",required=True,help="Directory containing UI screenshots (*.jpg).")
    parser.add_argument("--output-dir",required=True,help="Directory to store checkpoints.")
    parser.add_argument("--epochs",type=int,default=1,help="Number of epochs to train for.")
    parser.add_argument("--batch-size",type=int,default=2,help="Training batch size.")
    parser.add_argument("--lr",type=float,default=1e-4,help="Learning rate.")
    parser.add_argument("--num-workers",type=int,default=2,help="DataLoader worker count.")
    parser.add_argument("--precision",choices=["fp32","fp16","bf16"],default="fp16",help="Weights precision to use.")
    parser.add_argument("--base-model",default="runwayml/stable-diffusion-v1-5",help="Diffusers base model id.")
    return parser


def main():
    args=build_parser().parse_args()
    config={
        "train_json":args.train_json,
        "tensor_dir":args.tensor_dir,
        "image_dir":args.image_dir,
        "output_dir":args.output_dir,
        "batch_size":args.batch_size,
        "lr":args.lr,
        "num_workers":args.num_workers,
        "precision":args.precision,
        "base_model":args.base_model,
    }
    trainer=Trainer(config)
    trainer.train(args.epochs)


if __name__=="__main__":
    main()
