import gc
import logging
import os
import errno
import shutil
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from diffusers import DDPMScheduler, StableDiffusionPipeline
from huggingface_hub import login
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
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
        self.save_unet_in_checkpoint = bool(config.get("save_unet_in_checkpoint", False))
        self.max_checkpoints = max(0, int(config.get("max_checkpoints", 2) or 0))

        precision = config.get("precision", "fp16")
        precision_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if precision not in precision_map:
            raise ValueError("Unsupported precision '%s'. Choose from fp32, fp16, bf16." % precision)

        if precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
            logger.warning("bf16 is not supported on this GPU; falling back to fp16 for stability")
            precision = "fp16"

        self.weight_dtype = precision_map[precision]
        self.trainable_dtype = torch.float32
        self.base_model = config.get("base_model", "stable-diffusion-v1-5/stable-diffusion-v1-5")
        self.hf_token = config.get("hf_token") or os.environ.get("HF_TOKEN")
        self.grad_accum_steps = max(1, int(config.get("grad_accum_steps", 1)))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        requested_sample_limit = int(config.get("train_sample_limit", 10000) or 10000)
        # Keep training fast by default while allowing up to 20k samples.
        self.train_sample_limit = max(1, min(20000, requested_sample_limit))
        self.sample_seed = int(config.get("sample_seed", 42))
        max_train_hours = float(config.get("max_train_hours", 0) or 0)
        self.max_train_seconds = max_train_hours * 3600.0 if max_train_hours > 0 else None
        self.metric_ema_decay = float(config.get("metric_ema_decay", 0.98))
        self.use_amp = self.device.type == "cuda" and self.weight_dtype in (torch.float16, torch.bfloat16)
        self.autocast_dtype = self.weight_dtype if self.use_amp else torch.float32
        if self.device.type == "cuda" and self.weight_dtype == torch.float16 and hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda" and self.weight_dtype == torch.float16)

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if self.hf_token:
            login(token=self.hf_token, add_to_git_credential=False)
            logger.info("Authenticated with Hugging Face using a provided token")

        self.dataset = LayoutDataset(
            config["train_json"],
            config["tensor_dir"],
            config["image_dir"],
            skip_report=os.path.join(self.output_dir, "skipped_train_samples.json"),
        )
        raw_train_size = len(self.dataset)
        if raw_train_size > self.train_sample_limit:
            generator = torch.Generator()
            generator.manual_seed(self.sample_seed)
            selected_indices = torch.randperm(raw_train_size, generator=generator)[: self.train_sample_limit].tolist()
            self.dataset = Subset(self.dataset, selected_indices)
            logger.info(
                "Capped training dataset from %s to %s samples (requested=%s, hard_cap=20000, seed=%s)",
                raw_train_size,
                len(self.dataset),
                requested_sample_limit,
                self.sample_seed,
            )
        else:
            logger.info(
                "Loaded dataset with %s samples from %s (limit=%s; no cap needed)",
                raw_train_size,
                config["train_json"],
                self.train_sample_limit,
            )

        self.train_sampler = None
        self.val_loader = None
        self.best_val_loss = float("inf")

        val_json = config.get("val_json")
        # Validation disabled: skip loading a separate validation dataset to avoid
        # running validation during distributed training (can cause collective timeouts).
        self.val_loader = None
        if val_json:
            logger.info("Validation split provided but validation step is disabled by configuration; skipping val dataset load")
        else:
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

        logger.info("Loading model A weights: base diffusion model %s on %s", self.base_model, self.device)
        pretrained_kwargs = {"torch_dtype": self.weight_dtype, "low_cpu_mem_usage": True}
        if self.hf_token:
            pretrained_kwargs["token"] = self.hf_token
        self.pipe = StableDiffusionPipeline.from_pretrained(self.base_model, **pretrained_kwargs)
        self.pipe.to(self.device)

        self.unet = self.pipe.unet
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.unet.to(dtype=self.weight_dtype)
        self.unet.requires_grad_(False)
        self.unet.eval()
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.vae.eval()
        self.text_encoder.eval()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        if hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()

        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.controlnet = ControlNet().to(self.device, dtype=self.trainable_dtype)
        self.layout_encoder = LayoutEncoder().to(self.device, dtype=self.trainable_dtype)
        self.fusion = ConditioningFusion().to(self.device, dtype=self.trainable_dtype)

        if self.is_distributed:
            # Enable find_unused_parameters to avoid deadlocks when some parameters
            # are not used on every forward (safer for dynamic control networks).
            self.controlnet = DDP(
                self.controlnet,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )
            self.layout_encoder = DDP(
                self.layout_encoder,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
            )

        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.log_every = max(1, int(config.get("log_every", 25)))
        self.global_step = 0
        self.start_epoch = 0
        self.resume_batch_idx = 0
        self.loss_ema = None
        self.step_time_ema = None

        trainable_params = (
            list(self.controlnet.parameters())
            + list(self.layout_encoder.parameters())
            + list(self.fusion.parameters())
        )
        optimizer_kwargs = {
            "lr": config["lr"],
            "weight_decay": float(config.get("weight_decay", 0.01)),
            "betas": (0.9, 0.99),
        }
        if self.device.type == "cuda":
            try:
                self.optimizer = torch.optim.AdamW(trainable_params, fused=True, **optimizer_kwargs)
            except TypeError:
                self.optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)
        else:
            self.optimizer = torch.optim.AdamW(trainable_params, **optimizer_kwargs)

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
        latest_path = Path(self.output_dir) / "latest.pt"
        if latest_path.is_file():
            return str(latest_path)
        checkpoints = sorted(Path(self.output_dir).glob("checkpoint_epoch_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None

    def _unwrap(self, module):
        return module.module if hasattr(module, "module") else module

    def _trainable_parameters(self):
        for module in (self.controlnet, self.layout_encoder, self.fusion):
            yield from self._unwrap(module).parameters()

    def _current_lr(self):
        return self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0

    def _grad_norm(self):
        total = 0.0
        for param in self._trainable_parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.detach().float().norm(2).item()
            total += grad_norm * grad_norm
        return total ** 0.5

    def _gpu_memory_gb(self):
        if not torch.cuda.is_available():
            return None
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 3)
        return allocated, reserved

    def _autocast(self):
        if self.use_amp:
            return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        return nullcontext()

    def _set_training_mode(self, training):
        self.unet.eval()
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
            cond = self.fusion(text_emb.float(), layout_emb.float())

            control_feats = self.controlnet(control)
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

    def _checkpoint_state(self, epoch, val_loss=None, batch_idx=0, include_optimizer=True):
        state = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "global_step": self.global_step,
            "config": self.config,
            "base_model": self.base_model,
            "save_unet_in_checkpoint": self.save_unet_in_checkpoint,
            "controlnet": self._unwrap(self.controlnet).state_dict(),
            "layout_encoder": self._unwrap(self.layout_encoder).state_dict(),
            "fusion": self._unwrap(self.fusion).state_dict(),
            "best_val_loss": self.best_val_loss,
            "val_loss": val_loss,
            "loss_ema": self.loss_ema,
            "step_time_ema": self.step_time_ema,
        }

        if self.save_unet_in_checkpoint:
            # UNet is frozen in this trainer; saving it is optional and expensive.
            state["unet"] = self._unwrap(self.unet).state_dict()

        if include_optimizer:
            state["optimizer"] = self.optimizer.state_dict()
            state["scaler"] = self.scaler.state_dict() if self.scaler.is_enabled() else None

        return state

    def _safe_torch_save(self, state, path, label):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(target.parent),
                prefix=f"{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                tmp_path = Path(handle.name)

            torch.save(state, str(tmp_path))
            os.replace(str(tmp_path), str(target))
        except OSError as exc:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            free_gb = shutil.disk_usage(self.output_dir).free / (1024 ** 3)
            logger.error("Failed saving %s to %s: %s (free_disk_gb=%.2f)", label, target, exc, free_gb)
            if exc.errno == errno.ENOSPC:
                raise RuntimeError("No space left on device while writing checkpoint") from exc
            raise
        except RuntimeError as exc:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if "unexpected pos" in str(exc):
                free_gb = shutil.disk_usage(self.output_dir).free / (1024 ** 3)
                raise RuntimeError(
                    f"Checkpoint write failed due to partial zip write (free_disk_gb={free_gb:.2f})"
                ) from exc
            raise

    def _prune_epoch_checkpoints(self):
        if self.max_checkpoints <= 0:
            return

        checkpoint_files = sorted(
            Path(self.output_dir).glob("checkpoint_epoch_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        stale = checkpoint_files[:-self.max_checkpoints]
        for stale_path in stale:
            try:
                stale_path.unlink()
                logger.info("Pruned old checkpoint: %s", stale_path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                logger.warning("Could not prune %s: %s", stale_path, exc)

    def save_checkpoint(self, epoch, best=False, val_loss=None, final=False, batch_idx=0):
        if not self.is_rank0:
            return None

        checkpoint_suffix = f"_batch_{batch_idx}" if batch_idx > 0 else ""
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}{checkpoint_suffix}.pt")
        latest_path = os.path.join(self.output_dir, "latest.pt")
        state = self._checkpoint_state(epoch, val_loss=val_loss, batch_idx=batch_idx, include_optimizer=True)
        save_targets = [
            (checkpoint_path, "epoch checkpoint"),
            (latest_path, "latest checkpoint"),
        ]
        if best:
            best_path = os.path.join(self.output_dir, "best.pt")
            save_targets.append((best_path, "best checkpoint"))
        if final:
            final_path = os.path.join(self.output_dir, "final.pt")
            save_targets.append((final_path, "final checkpoint"))

        saved_any = False
        try:
            for target_path, label in save_targets:
                self._safe_torch_save(state, target_path, label)
                saved_any = True
                logger.info("Updated %s: %s", label, target_path)
        except RuntimeError as exc:
            logger.warning(
                "Failed to save full checkpoint state (%s). Retrying with lightweight state (no optimizer/scaler).",
                exc,
            )
            lightweight_state = self._checkpoint_state(
                epoch,
                val_loss=val_loss,
                batch_idx=batch_idx,
                include_optimizer=False,
            )
            for target_path, label in save_targets:
                try:
                    self._safe_torch_save(lightweight_state, target_path, f"{label} lightweight")
                    saved_any = True
                    logger.info("Updated %s (lightweight): %s", label, target_path)
                except Exception as fallback_exc:
                    logger.error("Could not save %s after fallback: %s", target_path, fallback_exc)

        if not saved_any:
            logger.error("All checkpoint save attempts failed for epoch=%s", epoch)
            return None

        self._prune_epoch_checkpoints()
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        logger.info("Loading model B weights: checkpoint %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "unet" in checkpoint:
            self._unwrap(self.unet).load_state_dict(checkpoint["unet"])
        else:
            logger.info(
                "Checkpoint %s has no UNet weights; using base model UNet from %s",
                checkpoint_path,
                self.base_model,
            )
        self._unwrap(self.controlnet).load_state_dict(checkpoint["controlnet"])
        self._unwrap(self.layout_encoder).load_state_dict(checkpoint["layout_encoder"])
        self._unwrap(self.fusion).load_state_dict(checkpoint["fusion"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, RuntimeError, KeyError) as exc:
            logger.warning("Skipping optimizer state restore from %s: %s", checkpoint_path, exc)
        if self.scaler.is_enabled() and checkpoint.get("scaler"):
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.resume_batch_idx = int(checkpoint.get("batch_idx", 0))
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        logger.info("Loaded checkpoint epoch=%s batch_idx=%s global_step=%s", self.start_epoch, self.resume_batch_idx, self.global_step)

    def train(self, epochs):
        target_epoch = self.start_epoch + epochs
        training_start = time.monotonic()
        last_val_loss = None
        last_completed_epoch = self.start_epoch
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
            epoch_loss_total = 0.0
            epoch_batches = 0
            window_loss_total = 0.0
            window_batches = 0
            epoch_start = time.monotonic()
            stop_reason = None
            start_batch_idx = self.resume_batch_idx if epoch == self.start_epoch else 0
            if start_batch_idx and self.is_rank0:
                logger.info("Resuming epoch %s from batch %s", epoch + 1, start_batch_idx + 1)
            self.resume_batch_idx = 0

            for batch_idx, batch in enumerate(loop, start=1):
                if batch_idx <= start_batch_idx:
                    continue
                if batch is None:
                    logger.warning("Skipping empty training batch at global_step=%s", self.global_step + 1)
                    continue

                step_start = time.monotonic()
                raw_loss = self.train_step(batch)
                loss = raw_loss / self.grad_accum_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_steps += 1
                raw_loss_value = float(raw_loss.detach().item())
                epoch_loss_total += raw_loss_value
                epoch_batches += 1
                window_loss_total += raw_loss_value
                window_batches += 1
                if accum_steps < self.grad_accum_steps:
                    continue

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(list(self._trainable_parameters()), self.max_grad_norm)
                grad_norm = self._grad_norm()
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                self.global_step += 1

                batch_time = time.monotonic() - step_start
                if self.loss_ema is None:
                    self.loss_ema = raw_loss_value
                else:
                    self.loss_ema = self.metric_ema_decay * self.loss_ema + (1.0 - self.metric_ema_decay) * raw_loss_value
                if self.step_time_ema is None:
                    self.step_time_ema = batch_time
                else:
                    self.step_time_ema = self.metric_ema_decay * self.step_time_ema + (1.0 - self.metric_ema_decay) * batch_time

                current_lr = self._current_lr()
                memory_stats = self._gpu_memory_gb()
                gpu_alloc = memory_stats[0] if memory_stats is not None else 0.0
                gpu_reserved = memory_stats[1] if memory_stats is not None else 0.0
                window_loss = window_loss_total / max(1, window_batches)
                epoch_loss = epoch_loss_total / max(1, epoch_batches)
                samples_per_sec = (batch["image"].shape[0] * self.grad_accum_steps) / max(batch_time, 1e-8)

                loop.set_postfix(
                    loss=f"{raw_loss_value:.4f}",
                    ema=f"{self.loss_ema:.4f}",
                    gnorm=f"{grad_norm:.2f}",
                    lr=f"{current_lr:.2e}",
                )
                if self.global_step == 1 or self.global_step % self.log_every == 0:
                    if self.is_rank0:
                        logger.info(
                            "step=%s epoch=%s/%s batch=%s raw_loss=%.6f window_loss=%.6f epoch_loss=%.6f loss_ema=%.6f grad_norm=%.4f lr=%.2e step_time=%.2fs samples_per_sec=%.2f gpu_alloc=%.2fGB gpu_reserved=%.2fGB",
                            self.global_step,
                            epoch + 1,
                            target_epoch,
                            batch_idx,
                            raw_loss_value,
                            window_loss,
                            epoch_loss,
                            self.loss_ema,
                            grad_norm,
                            current_lr,
                            batch_time,
                            samples_per_sec,
                            gpu_alloc,
                            gpu_reserved,
                        )

                window_loss_total = 0.0
                window_batches = 0

                if self.max_train_seconds is not None and (time.monotonic() - training_start) >= self.max_train_seconds:
                    stop_reason = f"Reached max_train_hours={self.max_train_seconds / 3600.0:.2f}"
                    break

            if accum_steps > 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(list(self._trainable_parameters()), self.max_grad_norm)
                grad_norm = self._grad_norm()
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

            epoch_time = time.monotonic() - epoch_start
            epoch_avg_loss = epoch_loss_total / max(1, epoch_batches)
            if self.is_rank0:
                logger.info(
                    "Epoch %s summary: avg_loss=%.6f loss_ema=%.6f epoch_time=%.2fs batches=%s",
                    epoch + 1,
                    epoch_avg_loss,
                    self.loss_ema if self.loss_ema is not None else float("nan"),
                    epoch_time,
                    epoch_batches,
                )

            if stop_reason:
                if self.is_rank0:
                    logger.info("%s; saving final checkpoint and stopping training", stop_reason)
                self.save_checkpoint(epoch, best=False, val_loss=last_val_loss, final=True, batch_idx=batch_idx)
                break

            if self.is_distributed and dist.is_initialized():
                dist.barrier()
            # Validation has been disabled to prevent distributed collective hangs.
            val_loss = None
            if self.is_distributed and dist.is_initialized():
                dist.barrier()
            last_val_loss = val_loss

            if val_loss is not None:
                logger.info("Validation epoch=%s val_loss=%.6f", epoch + 1, val_loss)
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                is_best = False

            self.save_checkpoint(epoch + 1, best=is_best, val_loss=val_loss, final=False)
            last_completed_epoch = epoch + 1
            if self.is_rank0:
                logger.info("Epoch %s/%s complete", epoch + 1, target_epoch)

        if self.is_rank0:
            logger.info("Training finished")
            if last_completed_epoch > self.start_epoch:
                self.save_checkpoint(last_completed_epoch, best=False, val_loss=last_val_loss, final=True)
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
