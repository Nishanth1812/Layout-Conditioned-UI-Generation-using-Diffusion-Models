import logging
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import DDPMScheduler, StableDiffusionPipeline
from huggingface_hub import login
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import CLIPTokenizer

from models.controlnet_encoder import ControlNet
from models.fusion import ConditioningFusion
from models.layout_encoder import LayoutEncoder
from training.dataset import LayoutDataset, safe_collate

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
        self.is_distributed = self.world_size > 1

        if self.is_distributed and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")

        if torch.cuda.is_available():
            if self.is_distributed:
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.is_rank0 = self.rank == 0

        self.output_dir = config.get("output_dir", "checkpoints")
        os.makedirs(self.output_dir, exist_ok=True)

        precision = config.get("precision", "fp16")
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in precision_map:
            raise ValueError("Unsupported precision '%s'. Choose from fp32, fp16, bf16." % precision)

        self.weight_dtype = precision_map[precision]
        self.base_model = config.get("base_model", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN")
        self.grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))
        self.use_amp = self.device.type == "cuda" and self.weight_dtype in (torch.float16, torch.bfloat16)
        self.autocast_dtype = self.weight_dtype if self.use_amp else torch.float32
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda" and self.weight_dtype == torch.float16)

        if self.hf_token:
            login(token=self.hf_token, add_to_git_credential=False)
            logger.info("Authenticated with Hugging Face using a provided token")

        self.dataset = LayoutDataset(
            config["train_json"],
            config["tensor_dir"],
            config["image_dir"],
            skip_report=os.path.join(self.output_dir, "skipped_train_samples.json"),
        )
        logger.info("Loaded dataset with %s samples from %s", len(self.dataset), config["train_json"])

        self.train_sampler = None
        self.val_loader = None
        self.best_val_loss = float("inf")

        val_json = config.get("val_json")
        if val_json and self.is_rank0:
            val_dataset = LayoutDataset(
                val_json,
                config["tensor_dir"],
                config["image_dir"],
                skip_report=os.path.join(self.output_dir, "skipped_val_samples.json"),
            )
            val_loader_kwargs = dict(
                batch_size=config.get("val_batch_size", config["batch_size"]),
                shuffle=False,
                num_workers=config.get("num_workers", 0),
                pin_memory=torch.cuda.is_available(),
                collate_fn=safe_collate,
            )
            if config.get("num_workers", 0) > 0:
                val_loader_kwargs["persistent_workers"] = True
                val_loader_kwargs["prefetch_factor"] = 4
            self.val_loader = DataLoader(val_dataset, **val_loader_kwargs)
            logger.info("Loaded validation dataset with %s samples from %s", len(val_dataset), val_json)
        elif not val_json:
            logger.info("No validation split provided; validation will be skipped")

        train_loader_kwargs = dict(
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            collate_fn=safe_collate,
        )
        if self.is_distributed:
            self.train_sampler = DistributedSampler(self.dataset, shuffle=True, drop_last=False)
            train_loader_kwargs["shuffle"] = False
            train_loader_kwargs["sampler"] = self.train_sampler
        if config.get("num_workers", 0) > 0:
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = 4
        self.loader = DataLoader(self.dataset, **train_loader_kwargs)

        logger.info(
            "Initializing Stable Diffusion base model %s on %s with %s precision",
            self.base_model,
            self.device,
            precision,
        )
        pretrained_kwargs = {"torch_dtype": self.weight_dtype}
        if self.hf_token:
            pretrained_kwargs["token"] = self.hf_token
        self.pipe = StableDiffusionPipeline.from_pretrained(self.base_model, **pretrained_kwargs)
        self.pipe.to(self.device)

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        self.controlnet = ControlNet().to(self.device)
        self.layout_encoder = LayoutEncoder().to(self.device)
        self.fusion = ConditioningFusion().to(self.device)

        if self.is_distributed:
            self.unet = DDP(self.unet, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
            self.controlnet = DDP(self.controlnet, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
            self.layout_encoder = DDP(self.layout_encoder, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.log_every = max(1, int(config.get("log_every", 25)))
        self.global_step = 0
        self.start_epoch = 0

        self.optimizer = torch.optim.AdamW(
            list(self.unet.parameters())
            + list(self.controlnet.parameters())
            + list(self.layout_encoder.parameters())
            + list(self.fusion.parameters()),
            lr=config["lr"],
        )

        resume_from = config.get("resume_from") or self._latest_checkpoint()
        if resume_from:
            self.load_checkpoint(resume_from)
            logger.info("Resumed training from %s", resume_from)

        logger.info(
            "Trainer ready: batch_size=%s, lr=%s, output_dir=%s, steps_per_epoch=%s",
            config["batch_size"],
            config["lr"],
            self.output_dir,
            len(self.loader),
        )

    def _latest_checkpoint(self):
        checkpoints = sorted(Path(self.output_dir).glob("checkpoint_epoch_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None

    def _unwrap(self, module):
        return module.module if hasattr(module, "module") else module

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        return nullcontext()

    def _set_training_mode(self, training):
        self.unet.train(training)
        self.controlnet.train(training)
        self.layout_encoder.train(training)
        self.fusion.train(training)

    def encode_text(self, captions):
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(self.device, non_blocking=True)

        with torch.no_grad():
            return self.text_encoder(input_ids)[0]

    def encode_images(self, images):
        images = images.to(self.device, dtype=self.weight_dtype, non_blocking=True)
        images = (images * 2.0) - 1.0

        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        return latents

    def train_step(self, batch):
        layouts = batch["layout"].to(self.device, dtype=torch.float32, non_blocking=True)
        control = batch["control"].to(self.device, dtype=torch.float32, non_blocking=True)
        images = batch["image"].to(self.device, non_blocking=True)
        captions = batch["caption"]

        with self._autocast():
            latents = self.encode_images(images)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                self.scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=self.device,
            ).long()

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            text_emb = self.encode_text(captions)
            layout_emb = self.layout_encoder(layouts)
            cond = self.fusion(text_emb.float(), layout_emb.float()).to(dtype=self.weight_dtype)

            control_feats = self.controlnet(control.to(dtype=self.weight_dtype))
            for feat in control_feats:
                resized = torch.nn.functional.interpolate(
                    feat,
                    size=noisy_latents.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                noisy_latents = noisy_latents + resized.to(dtype=noisy_latents.dtype)

            pred_noise = self.unet(noisy_latents, timesteps, encoder_hidden_states=cond).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
        return loss

    def validate(self, epoch):
        if self.val_loader is None or not self.is_rank0:
            return None

        self._set_training_mode(False)
        total_loss = 0.0
        total_batches = 0
        logger.info("Running validation for epoch %s", epoch)

        try:
            with torch.inference_mode():
                for batch_idx, batch in enumerate(tqdm(self.val_loader, desc=f"Val {epoch}", leave=False), start=1):
                    if batch is None:
                        continue
                    loss = self.train_step(batch)
                    total_loss += float(loss.detach().item())
                    total_batches += 1
                    if batch_idx == 1 or batch_idx % self.log_every == 0:
                        logger.info("val_epoch=%s batch=%s loss=%.6f", epoch, batch_idx, loss.item())
        finally:
            self._set_training_mode(True)

        if total_batches == 0:
            return None
        return total_loss / total_batches

    def save_checkpoint(self, epoch, best=False, val_loss=None):
        if not self.is_rank0:
            return None

        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "config": self.config,
            "unet": self._unwrap(self.unet).state_dict(),
            "controlnet": self._unwrap(self.controlnet).state_dict(),
            "layout_encoder": self._unwrap(self.layout_encoder).state_dict(),
            "fusion": self._unwrap(self.fusion).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
        }
        torch.save(state, checkpoint_path)
        logger.info("Saved checkpoint: %s", checkpoint_path)
        if best:
            best_path = os.path.join(self.output_dir, "best.pt")
            torch.save(state, best_path)
            logger.info(
                "Updated best checkpoint: %s (val_loss=%.6f)",
                best_path,
                val_loss if val_loss is not None else float("nan"),
            )
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._unwrap(self.unet).load_state_dict(checkpoint["unet"])
        self._unwrap(self.controlnet).load_state_dict(checkpoint["controlnet"])
        self._unwrap(self.layout_encoder).load_state_dict(checkpoint["layout_encoder"])
        self._unwrap(self.fusion).load_state_dict(checkpoint["fusion"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scaler.is_enabled() and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        logger.info("Loaded checkpoint epoch=%s global_step=%s", self.start_epoch, self.global_step)

    def train(self, epochs):
        target_epoch = self.start_epoch + epochs
        logger.info(
            "Starting training for %s additional epoch(s) up to epoch %s with batch_size=%s grad_accum_steps=%s",
            epochs,
            target_epoch,
            self.config["batch_size"],
            self.grad_accum_steps,
        )
        self._set_training_mode(True)
        for epoch in range(self.start_epoch, target_epoch):
            logger.info("Epoch %s/%s started", epoch + 1, target_epoch)
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            loop = tqdm(self.loader, desc=f"Epoch {epoch + 1}/{target_epoch}", disable=not self.is_rank0)
            self.optimizer.zero_grad(set_to_none=True)
            accum_steps = 0

            for batch_idx, batch in enumerate(loop, start=1):
                if batch is None:
                    logger.warning("Skipping empty training batch at global_step=%s", self.global_step + 1)
                    continue

                loss = self.train_step(batch) / self.grad_accum_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_steps += 1
                if accum_steps < self.grad_accum_steps:
                    continue

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                self.global_step += 1

                loop.set_postfix(loss=loss.item())
                if self.global_step == 1 or self.global_step % self.log_every == 0:
                    logger.info(
                        "step=%s epoch=%s/%s batch=%s loss=%.6f",
                        self.global_step,
                        epoch + 1,
                        target_epoch,
                        batch_idx,
                        loss.item(),
                    )

            if accum_steps > 0:
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            if self.is_distributed and dist.is_initialized():
                dist.barrier()
            val_loss = self.validate(epoch + 1) if self.is_rank0 else None
            if self.is_distributed and dist.is_initialized():
                dist.barrier()

            if val_loss is not None:
                logger.info("Validation epoch=%s val_loss=%.6f", epoch + 1, val_loss)
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                is_best = False

            self.save_checkpoint(epoch + 1, best=is_best, val_loss=val_loss)
            logger.info("Epoch %s/%s complete", epoch + 1, target_epoch)

        logger.info("Training finished")
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
