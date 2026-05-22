# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flow-OPD v2 actor worker — per-step Gaussian-KL reward on flow-VLA SDE.

Implements the Flow-OPD reward formula from Liu et al. (Flow-OPD,
github.com/CostaliyA/Flow-OPD, train_sd3.py:1093):

    kl_reward_t = ‖μ_s(x_t) − μ_T(x_t)‖² / (2 σ_t²)          (Eq. 10)
    advantage_t = kl_scale · kl_reward_t                       (kl_scale = -1)

where μ_s, μ_T are the SDE transition-kernel means produced by the student
(rollout-time snapshot) and the frozen teacher, respectively, evaluated at the
SAME state x_t — the student's own denoising chain. The reward is then summed
over denoise steps and averaged over action dimensions to get a scalar
advantage per (chunk, env). Standard PPO-clip surrogate is then used:

    L = -clip(ρ, 1-ε, 1+ε) · advantage,   ρ = exp(log π_new − log π_old)

This is exactly Flow-OPD applied to a flow-matching VLA. The KEY difference
from our previous (broken) OPD actor is:

  PREV:  r_t = log π_teacher(x_{t+1}|x_t) − log π_student_old(x_{t+1}|x_t)
         (LLM-style trajectory log-prob ratio — high-variance, saturates when
         student goes OOD for teacher; this is what the user identified as
         wrong against Flow-OPD).

  NEW:   r_t = -‖μ_s(x_t) − μ_T(x_t)‖² / (2 σ_t²)
         (closed-form per-step Gaussian-KL between same-σ SDE kernels — bounded
         from above by 0, no MC variance, never saturates).

The teacher init/load path is reused from the previous OPD actor (and so is the
optional MAR anchor against a frozen SFT-init ref, β · KL(s_new || s_ref) added
to the loss — currently inactive when kl_beta=0).
"""

import os

import torch

from rlinf.models import get_model
from rlinf.utils.metric_utils import compute_rollout_metrics
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _load_state_dict_from_path(path):
    """Load a state_dict from either a .pt file or an HF safetensors directory."""
    import os
    if os.path.isfile(path):
        sd = torch.load(path, map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        return sd
    if os.path.isdir(path):
        import json
        import safetensors.torch as st
        idx = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(idx):
            with open(idx) as f:
                index = json.load(f)
            sd = {}
            for shard in sorted(set(index["weight_map"].values())):
                sd.update(st.load_file(os.path.join(path, shard)))
            return sd
        single = os.path.join(path, "model.safetensors")
        if os.path.exists(single):
            return st.load_file(single)
    raise FileNotFoundError(f"Cannot load state_dict from {path!r}")


class EmbodiedOPDV2FSDPActor(EmbodiedFSDPActor):
    """Flow-OPD actor: per-step velocity-MSE KL reward, standard PPO-clip loss.

    Only init_worker() and compute_advantages_and_returns() are overridden;
    everything else — training loop, PPO inner update, sync_model_to_rollout,
    checkpointing — is inherited unchanged from the GRPO EmbodiedFSDPActor.

    Config knobs (all under cfg.algorithm):
      - teacher_checkpoint: str, required. Path to frozen teacher .pt or HF dir.
      - kl_scale: float, default -1.0. Multiplier on per-step Gaussian-KL when
        forming the advantage; Flow-OPD convention is -1 (= minimize KL).
      - kl_beta: float, default 0.0. MAR anchor coefficient (loss-side). When
        >0 we additionally load a frozen ref model (= the SFT-init student)
        and store its log-probs as ref_logprobs. Inherits the standard
        ref-KL machinery from EmbodiedFSDPActor for the loss.
    """

    def init_worker(self) -> None:
        super().init_worker()
        self._init_teacher_model()
        self._init_ref_model()

    # ---- frozen models -----------------------------------------------------
    def _init_teacher_model(self) -> None:
        teacher_ckpt = self.cfg.algorithm.get("teacher_checkpoint", None)
        if teacher_ckpt is None:
            raise ValueError(
                "Flow-OPD v2 requires algorithm.teacher_checkpoint to be set "
                "(path to frozen RL-oracle / specialist teacher .pt or HF dir)."
            )
        self.teacher_model = get_model(self.cfg.actor.model)
        if self.teacher_model is None:
            raise RuntimeError("get_model() returned None for teacher_model.")
        self.log_info(f"[OPD-V2] Loading teacher weights from {teacher_ckpt}")
        state_dict = _load_state_dict_from_path(teacher_ckpt)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("value_head")}
        missing, unexpected = self.teacher_model.load_state_dict(state_dict, strict=False)
        self.log_info(f"[OPD-V2] Teacher load: {len(missing)} missing, {len(unexpected)} unexpected")
        if unexpected:
            self.log_info(f"[OPD-V2] First unexpected: {unexpected[:3]}")
        device = f"{self.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        self.teacher_model = self.teacher_model.to(device)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()
        self.log_info("[OPD-V2] Teacher model initialised and frozen.")

    def _init_ref_model(self) -> None:
        # Optional MAR anchor against frozen SFT-init student.
        self._mar_kl_beta = float(self.cfg.algorithm.get("kl_beta", 0.0) or 0.0)
        if self._mar_kl_beta <= 0.0:
            self.ref_model = None
            self.log_info("[OPD-V2-MAR] kl_beta<=0; MAR anchor disabled.")
            return
        model_path = self.cfg.actor.model.model_path
        self.ref_model = get_model(self.cfg.actor.model)
        state_dict = _load_state_dict_from_path(model_path)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("value_head")}
        missing, unexpected = self.ref_model.load_state_dict(state_dict, strict=False)
        self.log_info(f"[OPD-V2-MAR] Ref(SFT-init) load: {len(missing)} missing, {len(unexpected)} unexpected")
        device = f"{self.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        self.ref_model = self.ref_model.to(device)
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        self.log_info(f"[OPD-V2-MAR] Ref frozen; MAR kl_beta={self._mar_kl_beta}")

    # ---- advantage = -KL(student_old || teacher) ---------------------------
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """Flow-OPD reward: advantage_t = kl_scale · ‖μ_s(x_t) − μ_T(x_t)‖²/(2σ²).

        Evaluates BOTH the current student model (= rollout-time policy, since
        we are pre-PPO-inner-update here) and the frozen teacher on the rollout
        chains, then forms the per-step Gaussian-KL and reduces to a scalar
        per (chunk, env). Result is written to rollout_batch["advantages"] in
        shape [n_chunks, B, 1] — identical to the GRPO advantage shape.
        """
        forward_inputs = self.rollout_batch.get("forward_inputs", None)
        if forward_inputs is None:
            raise RuntimeError(
                "Flow-OPD v2 requires forward_inputs in rollout_batch — the "
                "actor model has to be re-run on the rollout chains to get "
                "μ_s(x_t)."
            )
        n_chunks = self.rollout_batch["prev_logprobs"].shape[0]
        device = f"{self.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        self.teacher_model = self.teacher_model.to(device)

        kl_scale = float(self.cfg.algorithm.get("kl_scale", -1.0))

        kl_per_chunk = []
        teacher_lp_list = []
        ref_lp_list = []

        # Read raw KL knobs from rollout_batch for telemetry.
        for i in range(n_chunks):
            chunk_fwd = {
                k: v[i].to(device)
                for k, v in forward_inputs.items()
                if isinstance(v, torch.Tensor)
            }
            with torch.no_grad(), self.amp_context:
                # Student (rollout snapshot) per-step means and stds.
                # NB: actor_module is the FSDP-wrapped student; passing
                # return_step_means=True activates the Flow-OPD branch of
                # default_forward and the returned dict will carry step_means.
                student_out = self.actor_model(
                    forward_inputs=chunk_fwd,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=False,
                    use_cache=False,
                    return_step_means=True,
                )
                # Teacher per-step means at the SAME state x_t.
                teacher_out = self.teacher_model(
                    forward_inputs=chunk_fwd,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=False,
                    use_cache=False,
                    return_step_means=True,
                )

            s_mean = student_out["step_means"].float()  # [B, n_step(+1), chunk, action_dim]
            t_mean = teacher_out["step_means"].float()
            s_std = student_out["step_stds"].float().clamp_min(1e-6)

            # Drop the t=0 prior placeholder if joint_logprob added one
            # (constant N(0,1), would inflate KL with a fake term).
            # Heuristic: detect placeholder by  s_std[:, 0] all ones.
            if s_std.shape[1] > 1 and torch.allclose(
                s_std[:, 0], torch.ones_like(s_std[:, 0])
            ):
                s_mean = s_mean[:, 1:]
                t_mean = t_mean[:, 1:]
                s_std = s_std[:, 1:]

            # Closed-form Gaussian-KL between same-σ kernels:
            #   KL_step = ‖μ_s − μ_t‖² / (2 σ²)
            # Reduce over (chunk, action_dim), sum over denoise steps, mean over batch dim later.
            kl_step = ((s_mean - t_mean) ** 2) / (2.0 * s_std**2)
            # mean over action coords AND over denoise steps -> scalar per env in the chunk
            kl_chunk = kl_step.mean(dim=(1, 2, 3))  # [B]
            kl_per_chunk.append(kl_chunk)

            teacher_lp_list.append(teacher_out["logprobs"].detach().float())

            if getattr(self, "ref_model", None) is not None:
                self.ref_model = self.ref_model.to(device)
                with torch.no_grad(), self.amp_context:
                    ref_out = self.ref_model(
                        forward_inputs=chunk_fwd,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                    )
                ref_lp_list.append(ref_out["logprobs"].detach().float())

        # [n_chunks, B]
        kl_rewards = torch.stack(kl_per_chunk, dim=0)
        adv = (kl_scale * kl_rewards).unsqueeze(-1)  # [n_chunks, B, 1]

        # Optional normalize_advantages (GRPO-style) — turned on at the
        # config level via algorithm.normalize_advantages.
        if bool(self.cfg.algorithm.get("normalize_advantages", False)):
            adv_flat = adv.float()
            adv = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        self.rollout_batch["advantages"] = adv
        self.rollout_batch.pop("returns", None)

        # Teacher log-probs are no longer the advantage signal, but we keep
        # storing them for the optional logging hook used by the standard
        # KL-penalty path (some downstream loss functions read this name).
        teacher_lp = torch.stack(teacher_lp_list, dim=0)
        while teacher_lp.dim() > 2:
            teacher_lp = teacher_lp.sum(dim=-1)

        if ref_lp_list:
            ref_lp = torch.stack(ref_lp_list, dim=0)
            self.rollout_batch["ref_logprobs"] = ref_lp.to(
                self.rollout_batch["prev_logprobs"].device
            )
            self.log_info(
                f"[OPD-V2-MAR] ref_logprobs stored shape={tuple(ref_lp.shape)} "
                f"mean={ref_lp.float().mean().item():.3f}"
            )

        # --- telemetry ------------------------------------------------------
        try:
            kr = kl_rewards.float()
            q = torch.quantile(
                kr.flatten(),
                torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=kr.device),
            )
            self.log_info(
                f"[OPD-V2] kl_reward (=KL_step / chunk): "
                f"mean={kr.mean().item():.4f} std={kr.std().item():.4f} "
                f"p0={q[0].item():.4f} p25={q[1].item():.4f} p50={q[2].item():.4f} "
                f"p75={q[3].item():.4f} p100={q[4].item():.4f}  "
                f"advantage_mean={adv.float().mean().item():.4f}"
            )
        except Exception as _e:
            self.log_warning(f"[OPD-V2] kl_reward stat dump failed: {_e!r}")

        return compute_rollout_metrics(self.rollout_batch)
