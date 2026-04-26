import json
import os

import torch


class DDIMScheduler:
    """与 SD1.5 配置尽量对齐的 DDIM 采样调度器"""

    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        model_path=None,
    ):
        config = self._load_scheduler_config(model_path)

        self.num_train_timesteps = config.get("num_train_timesteps", num_train_timesteps)
        self.beta_start = config.get("beta_start", beta_start)
        self.beta_end = config.get("beta_end", beta_end)
        self.steps_offset = config.get("steps_offset", 0)
        self.set_alpha_to_one = config.get("set_alpha_to_one", True)
        self.clip_sample = config.get("clip_sample", False)

        self.betas = torch.linspace(
            self.beta_start ** 0.5,
            self.beta_end ** 0.5,
            self.num_train_timesteps,
            dtype=torch.float32,
        ) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        final_alpha = 1.0 if self.set_alpha_to_one else self.alphas_cumprod[0].item()
        self.final_alpha_cumprod = torch.tensor(final_alpha, dtype=torch.float32)

        self.timesteps = None
        self.num_inference_steps = None
        self.device = "cpu"

    def _load_scheduler_config(self, model_path):
        if not model_path:
            return {}

        config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
        if not os.path.exists(config_path):
            return {}

        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set_timesteps(self, num_inference_steps, device="cuda"):
        self.device = device
        self.num_inference_steps = num_inference_steps

        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(num_inference_steps, device=device) * step_ratio
        timesteps = timesteps.flip(0) + self.steps_offset
        self.timesteps = torch.clamp(timesteps, 0, self.num_train_timesteps - 1).long()

        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.final_alpha_cumprod = self.final_alpha_cumprod.to(device)

    def step(self, model_output, timestep, sample, eta=0.0):
        del eta  # 当前实现固定为确定性 DDIM

        t = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        prev_t = t - step_ratio

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1, 1)

        pred_sample_direction = (1 - alpha_prod_t_prev).sqrt() * model_output
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction
        return prev_sample

    def add_noise(self, original_samples, noise, timesteps):
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps].sqrt().flatten()
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]).sqrt().flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

    def scale_model_input(self, sample, timestep):
        del timestep
        return sample
