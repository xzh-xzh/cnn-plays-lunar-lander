from __future__ import annotations

import argparse
import importlib.util
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from gymnasium.wrappers import (
    AddRenderObservation,
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
)
from torch.distributions import Categorical

from configs.config import TrainingConfig
from models import build_model


STORAGE_DEVICE = torch.device("cpu")


def load_config(config_path: str | Path) -> TrainingConfig:
    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return TrainingConfig.from_dict(data)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    return device


def make_env(config: TrainingConfig, env_index: int):
    def thunk() -> gym.Env:
        env = gym.make(
            config.ENV_ID,
            render_mode="rgb_array",
            max_episode_steps=config.MAX_EPISODE_STEPS,
        )
        env = AddRenderObservation(env, render_only=True)
        env = ResizeObservation(env, (config.FRAME_SIZE, config.FRAME_SIZE))
        env = GrayscaleObservation(env, keep_dim=True)
        env = FrameStackObservation(env, config.STACK_SIZE)
        env.action_space.seed(config.SEED + env_index)
        return env

    return thunk


def maybe_compile(model: torch.nn.Module, config: TrainingConfig, device: torch.device):
    if not config.USE_TORCH_COMPILE:
        return model
    if device.type != "cuda":
        print("torch.compile disabled: it is only useful for this project on CUDA.")
        return model
    if importlib.util.find_spec("triton") is None:
        print("torch.compile disabled: Triton is not installed in this environment.")
        return model
    try:
        return torch.compile(model, mode="reduce-overhead")
    except Exception as exc:
        print(f"torch.compile disabled after setup failure: {exc}")
        return model


def strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("_orig_mod.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }


class PPOTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.DEVICE)
        self.num_steps = config.NUM_STEPS_PER_ROLLOUT
        self.num_envs = config.NUM_ENVS
        self.global_step = 0

        env_fns = [make_env(config, index) for index in range(config.NUM_ENVS)]
        if config.VECTOR_ENV == "async":
            self.env = gym.vector.AsyncVectorEnv(env_fns)
        elif config.VECTOR_ENV == "sync":
            self.env = gym.vector.SyncVectorEnv(env_fns)
        else:
            raise ValueError(f"Unsupported VECTOR_ENV: {config.VECTOR_ENV}")

        self.policy = build_model(
            model_name=config.model_name,
            num_actions=self.env.single_action_space.n,
            num_frames=config.STACK_SIZE,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.GRU_LAYERS,
            pretrained_backbone=config.PRETRAINED_BACKBONE,
        ).to(self.device)
        self._load_checkpoint_if_requested()
        self._configure_trainable_parameters()
        self.optimizer = self._build_optimizer()
        self.model = maybe_compile(self.policy, config, self.device)

        self.amp_enabled = config.USE_MIXED_PRECISION and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.amp_enabled and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16
        )

        pin_memory = config.PIN_MEMORY and self.device.type == "cuda"
        self.obs = torch.empty(
            self.num_steps,
            self.num_envs,
            1,
            config.STACK_SIZE,
            config.FRAME_SIZE,
            config.FRAME_SIZE,
            dtype=torch.uint8,
            device=STORAGE_DEVICE,
            pin_memory=pin_memory,
        )
        self.actions = torch.empty(
            self.num_steps,
            self.num_envs,
            dtype=torch.int64,
            device=STORAGE_DEVICE,
            pin_memory=pin_memory,
        )
        self.log_probs = torch.empty(
            self.num_steps,
            self.num_envs,
            dtype=torch.float32,
            device=STORAGE_DEVICE,
            pin_memory=pin_memory,
        )
        self.rewards = torch.empty_like(self.log_probs)
        self.dones = torch.empty_like(self.log_probs)
        self.values = torch.empty_like(self.log_probs)
        self.advantages = torch.empty_like(self.log_probs)
        self.returns = torch.empty_like(self.log_probs)

    def _load_checkpoint_if_requested(self) -> None:
        self.checkpoint_optimizer_state: dict[str, Any] | None = None
        if not self.config.CHECKPOINT_PATH:
            return

        checkpoint = torch.load(self.config.CHECKPOINT_PATH, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.policy.load_state_dict(
                strip_compile_prefix(checkpoint["model_state_dict"])
            )
            self.checkpoint_optimizer_state = checkpoint.get("optimizer_state_dict")
        else:
            self.policy.load_state_dict(strip_compile_prefix(checkpoint))

    def _configure_trainable_parameters(self) -> None:
        if self.config.model_name == "TemporalResNetGRU" and self.config.FREEZE_RESNET:
            for name, param in self.policy.named_parameters():
                if name.startswith("backbone."):
                    param.requires_grad = False

        trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.policy.parameters())
        frozen = total - trainable
        if frozen:
            print(
                f"Frozen parameters: {frozen:,}; "
                f"trainable parameters: {trainable:,}."
            )

    def _build_optimizer(self) -> optim.Optimizer:
        backbone_lr = self.config.BACKBONE_LEARNING_RATE
        if backbone_lr is None or not hasattr(self.policy, "backbone"):
            trainable_params = [
                param for param in self.policy.parameters() if param.requires_grad
            ]
            optimizer = optim.Adam(trainable_params, lr=self.config.LEARNING_RATE)
        else:
            backbone_params = []
            head_params = []
            for name, param in self.policy.named_parameters():
                if not param.requires_grad:
                    continue
                if name.startswith("backbone."):
                    backbone_params.append(param)
                else:
                    head_params.append(param)
            param_groups = []
            if backbone_params:
                param_groups.append({"params": backbone_params, "lr": backbone_lr})
            if head_params:
                param_groups.append(
                    {"params": head_params, "lr": self.config.LEARNING_RATE}
                )
            optimizer = optim.Adam(param_groups)

        if self.checkpoint_optimizer_state:
            try:
                optimizer.load_state_dict(self.checkpoint_optimizer_state)
            except Exception as exc:
                print(f"Optimizer checkpoint ignored: {exc}")
        return optimizer

    def close(self) -> None:
        if self.config.VECTOR_ENV == "async" and hasattr(self.env, "close_extras"):
            self.env.close_extras(terminate=True)
        else:
            self.env.close()

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.uint8).permute(0, 4, 1, 2, 3).contiguous()

    def _forward(self, obs: torch.Tensor):
        model_input = obs.to(self.device, dtype=torch.float32, non_blocking=True) / 255.0
        with torch.amp.autocast(
            self.device.type,
            enabled=self.amp_enabled,
            dtype=self.amp_dtype,
        ):
            return self.model(model_input)

    def _print_progress(
        self, round_num: int, phase: str, done: int, total: int
    ) -> None:
        total = max(total, 1)
        done = min(done, total)
        width = 30
        filled = int(width * done / total)
        bar = "#" * filled + "-" * (width - filled)
        percent = 100.0 * done / total
        print(
            f"\rround={round_num} {phase:<7} [{bar}] {done}/{total} {percent:5.1f}%",
            end="",
            flush=True,
        )

    @torch.no_grad()
    def collect_rollout(self, round_num: int) -> dict[str, float]:
        self.model.eval()
        observations, _ = self.env.reset(seed=self.config.SEED + round_num)

        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        current_rewards = np.zeros(self.num_envs, dtype=np.float32)
        current_lengths = np.zeros(self.num_envs, dtype=np.int32)
        progress_every = max(1, self.num_steps // 50)
        self._print_progress(round_num, "rollout", 0, self.num_steps)

        for step in range(self.num_steps):
            obs_batch = self._obs_to_tensor(observations)
            self.obs[step].copy_(obs_batch)

            logits, values = self._forward(self.obs[step])
            distribution = Categorical(logits=logits.float())
            actions = distribution.sample()

            self.actions[step].copy_(actions.to(STORAGE_DEVICE))
            self.log_probs[step].copy_(distribution.log_prob(actions).to(STORAGE_DEVICE))
            self.values[step].copy_(values.squeeze(-1).to(STORAGE_DEVICE))

            next_observations, rewards, terminated, truncated, _ = self.env.step(
                actions.cpu().numpy()
            )
            dones = np.logical_or(terminated, truncated)

            self.rewards[step].copy_(torch.as_tensor(rewards, dtype=torch.float32))
            self.dones[step].copy_(torch.as_tensor(dones, dtype=torch.float32))

            current_rewards += rewards
            current_lengths += 1
            self.global_step += self.num_envs

            for env_index, done in enumerate(dones):
                if done:
                    episode_rewards.append(float(current_rewards[env_index]))
                    episode_lengths.append(int(current_lengths[env_index]))
                    current_rewards[env_index] = 0.0
                    current_lengths[env_index] = 0

            observations = next_observations
            steps_done = step + 1
            if steps_done == self.num_steps or steps_done % progress_every == 0:
                self._print_progress(
                    round_num, "rollout", steps_done, self.num_steps
                )

        print()

        next_obs = self._obs_to_tensor(observations)
        _, next_values = self._forward(next_obs)
        next_values = next_values.squeeze(-1).to(STORAGE_DEVICE)
        next_values *= 1.0 - self.dones[-1]

        self._calculate_gae(next_values)
        self._normalize_advantages()

        return {
            "episode_reward_mean": float(np.mean(episode_rewards))
            if episode_rewards
            else 0.0,
            "episode_length_mean": float(np.mean(episode_lengths))
            if episode_lengths
            else 0.0,
            "rollout_reward_mean": float(self.rewards.mean().item()),
            "finished_episodes": float(len(episode_rewards)),
        }

    def _calculate_gae(self, next_values: torch.Tensor) -> None:
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32, device=STORAGE_DEVICE)
        values_with_next = torch.cat((self.values, next_values.unsqueeze(0)), dim=0)

        for step in reversed(range(self.num_steps)):
            not_done = 1.0 - self.dones[step]
            delta = (
                self.rewards[step]
                + self.config.GAMMA * values_with_next[step + 1] * not_done
                - values_with_next[step]
            )
            last_gae = (
                delta
                + self.config.GAMMA * self.config.GAE_LAMBDA * not_done * last_gae
            )
            self.advantages[step] = last_gae

        self.returns.copy_(self.advantages + self.values)

    def _normalize_advantages(self) -> None:
        mean = self.advantages.mean()
        std = self.advantages.std(unbiased=False)
        self.advantages.copy_((self.advantages - mean) / (std + 1e-8))

    def ppo_update(self, round_num: int) -> dict[str, float]:
        self.model.train()

        obs_flat = self.obs.reshape(
            -1,
            1,
            self.config.STACK_SIZE,
            self.config.FRAME_SIZE,
            self.config.FRAME_SIZE,
        )
        actions_flat = self.actions.reshape(-1)
        old_log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        values_flat = self.values.reshape(-1)

        explained_var = 1.0 - torch.var(
            returns_flat - values_flat, unbiased=False
        ) / (torch.var(returns_flat, unbiased=False) + 1e-8)

        totals = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "ratio_mean": 0.0,
            "clip_fraction": 0.0,
            "approx_kl": 0.0,
        }
        num_updates = 0
        indices = np.arange(self.num_steps * self.num_envs)
        batches_per_epoch = int(np.ceil(len(indices) / self.config.BATCH_SIZE))
        total_batches = self.config.PPO_EPOCHS * max(batches_per_epoch, 1)
        self._print_progress(round_num, "update", 0, total_batches)

        for _ in range(self.config.PPO_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.BATCH_SIZE):
                batch_indices = indices[start : start + self.config.BATCH_SIZE]

                states = (
                    obs_flat[batch_indices]
                    .to(self.device, dtype=torch.float32, non_blocking=True)
                    .div_(255.0)
                )
                actions = actions_flat[batch_indices].to(self.device, non_blocking=True)
                old_log_probs = old_log_probs_flat[batch_indices].to(
                    self.device, non_blocking=True
                )
                advantages = advantages_flat[batch_indices].to(
                    self.device, non_blocking=True
                )
                returns = returns_flat[batch_indices].to(
                    self.device, non_blocking=True
                )

                with torch.amp.autocast(
                    self.device.type,
                    enabled=self.amp_enabled,
                    dtype=self.amp_dtype,
                ):
                    logits, values = self.model(states)
                    distribution = Categorical(logits=logits.float())
                    log_probs = distribution.log_prob(actions)
                    entropy = distribution.entropy().mean()
                    log_ratio = log_probs - old_log_probs
                    ratio = log_ratio.exp()

                    surrogate_1 = ratio * advantages
                    surrogate_2 = torch.clamp(
                        ratio,
                        1.0 - self.config.CLIP_EPSILON,
                        1.0 + self.config.CLIP_EPSILON,
                    ) * advantages
                    actor_loss = -torch.min(surrogate_1, surrogate_2).mean()
                    critic_loss = F.mse_loss(values.squeeze(-1), returns)
                    loss = (
                        actor_loss
                        + self.config.VALUE_COEFF * critic_loss
                        - self.config.ENTROPY_COEFF * entropy
                    )

                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.MAX_GRAD_NORM
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.MAX_GRAD_NORM
                    )
                    self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.CLIP_EPSILON
                    ).float().mean()

                totals["actor_loss"] += float(actor_loss.item())
                totals["critic_loss"] += float(critic_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["ratio_mean"] += float(ratio.mean().item())
                totals["clip_fraction"] += float(clip_fraction.item())
                totals["approx_kl"] += float(approx_kl.item())
                num_updates += 1
                self._print_progress(round_num, "update", num_updates, total_batches)

                if (
                    self.config.TARGET_KL is not None
                    and approx_kl.item() > self.config.TARGET_KL
                ):
                    break

        print()

        metrics = {key: value / max(num_updates, 1) for key, value in totals.items()}
        metrics["explained_variance"] = float(explained_var.item())
        metrics["updates"] = float(num_updates)
        return metrics

    def save_checkpoint(self, path: Path, round_num: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "round": round_num,
                "config": self.config.to_dict(),
            },
            path,
        )


def create_run_dir(config: TrainingConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.OUTPUT_DIR) / f"{config.model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w", encoding="utf-8") as file:
        json.dump(config.to_dict(), file, indent=2)
    return run_dir


def train(config: TrainingConfig) -> Path:
    set_global_seed(config.SEED)
    run_dir = create_run_dir(config)
    metrics_path = run_dir / "metrics.jsonl"

    mlflow_run = None
    if config.USE_MLFLOW:
        import mlflow

        mlflow.set_experiment(config.model_name)
        mlflow_run = mlflow.start_run(run_name=run_dir.name)
        mlflow.log_params(config.to_dict())

    trainer = PPOTrainer(config)
    print(
        f"Training {config.model_name} on {trainer.device} "
        f"with {config.NUM_ENVS} {config.VECTOR_ENV} env(s)."
    )
    try:
        for round_num in range(config.NUM_ROUNDS):
            rollout_metrics = trainer.collect_rollout(round_num)
            update_metrics = trainer.ppo_update(round_num)
            metrics = {
                "round": round_num,
                "global_step": trainer.global_step,
                **rollout_metrics,
                **update_metrics,
            }

            with open(metrics_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(metrics) + "\n")

            if config.USE_MLFLOW:
                import mlflow

                mlflow.log_metrics(metrics, step=round_num)

            if round_num % config.LOG_INTERVAL == 0:
                print(
                    "round={round} step={global_step} "
                    "rollout_reward={rollout_reward_mean:.3f} "
                    "episode_reward={episode_reward_mean:.3f} "
                    "actor_loss={actor_loss:.4f} critic_loss={critic_loss:.4f} "
                    "entropy={entropy:.4f} kl={approx_kl:.6f}".format(**metrics)
                )

            should_checkpoint = (
                config.CHECKPOINT_FREQUENCY > 0
                and round_num % config.CHECKPOINT_FREQUENCY == 0
            )
            if should_checkpoint:
                checkpoint_path = (
                    Path(config.CHECKPOINT_DIR)
                    / run_dir.name
                    / f"{round_num:07d}.pt"
                )
                trainer.save_checkpoint(checkpoint_path, round_num)
    finally:
        trainer.close()
        if mlflow_run is not None:
            import mlflow

            mlflow.end_run()

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vision PPO for LunarLander-v3")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--rounds", type=int, help="Override NUM_ROUNDS")
    parser.add_argument("--device", help="Override DEVICE, e.g. cpu or cuda")
    parser.add_argument("--num-envs", type=int, help="Override NUM_ENVS")
    parser.add_argument("--rollout-steps", type=int, help="Override rollout length")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.rounds is not None:
        config.NUM_ROUNDS = args.rounds
    if args.device is not None:
        config.DEVICE = args.device
    if args.num_envs is not None:
        config.NUM_ENVS = args.num_envs
    if args.rollout_steps is not None:
        config.NUM_STEPS_PER_ROLLOUT = args.rollout_steps
    if args.mlflow:
        config.USE_MLFLOW = True
    if args.no_mlflow:
        config.USE_MLFLOW = False
    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    run_dir = train(config)
    print(f"Finished. Logs: {run_dir}")


if __name__ == "__main__":
    main()
