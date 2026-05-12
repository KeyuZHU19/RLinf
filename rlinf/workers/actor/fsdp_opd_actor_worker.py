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

"""OPD (On-Policy Distillation) actor worker for flow-based VLA models.

Implements VLA-OPD (arXiv:2603.26666) on top of the Flow-Noise log-prob
machinery from pi-RL (arXiv:2510.25889).  The teacher is a frozen copy of
the RL-trained policy.  At each training step the KL reward

    r_t = log pi_teacher(a_t | s_t) - log pi_student_old(a_t | s_t)

is computed and injected as the advantage signal before calling the standard
EmbodiedFSDPActor training loop with loss_type: opd_flow.
"""

import os

import torch

from rlinf.models import get_model
from rlinf.utils.metric_utils import compute_rollout_metrics
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor




def _load_state_dict_from_path(path):
    """Load a model state_dict from either a .pt file or an HF safetensors directory.

    HF directories may have a single model.safetensors or a sharded layout (.index.json + shards).
    """
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

class EmbodiedOPDFSDPActor(EmbodiedFSDPActor):
    """EmbodiedFSDPActor that replaces the advantage computation with OPD KL rewards.

    Only compute_advantages_and_returns() and init_worker() are overridden.
    The rest of the training loop (run_training, sync_model_to_rollout, …) is
    inherited unchanged.
    """

    def init_worker(self) -> None:
        super().init_worker()
        self._init_teacher_model()

    def _init_teacher_model(self) -> None:
        teacher_ckpt = self.cfg.algorithm.get("teacher_checkpoint", None)
        model_path = self.cfg.actor.model.model_path

        self.teacher_model = get_model(self.cfg.actor.model)
        if self.teacher_model is None:
            raise RuntimeError(
                "get_model() returned None for model_type="
                f"{self.cfg.actor.model.model_type}. "
                "OPD requires a model that supports logprob computation."
            )

        ckpt_path = teacher_ckpt if teacher_ckpt is not None else model_path
        self.log_info(f"[OPD] Loading teacher weights from {ckpt_path}")

        if teacher_ckpt is not None:
            state_dict = _load_state_dict_from_path(teacher_ckpt)
            # Drop value_head.* keys — RLinf-released PPO checkpoints include
            # a critic head that does not exist in the teacher PI0Pytorch
            # built here (add_value_head=False for OPD). Without this,
            # load_state_dict(strict=True) raises on unexpected keys.
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("value_head")}
            missing, unexpected = self.teacher_model.load_state_dict(state_dict, strict=False)
            self.log_info(f"[OPD] Teacher load: {len(missing)} missing, {len(unexpected)} unexpected")
            if unexpected:
                self.log_info(f"[OPD] First unexpected: {unexpected[:3]}")

        device = f"{self.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        self.teacher_model = self.teacher_model.to(device)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        self.log_info("[OPD] Teacher model initialised and frozen.")

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """Compute and store teacher log-probs for on-policy KL reward (VLA-OPD Algorithm 1).

        Stores chunk-level teacher log-probs as rollout_batch["advantages"].
        r_t = log pi_teacher - log pi_student_CURRENT is computed fresh at every
        gradient step inside compute_opd_flow_loss, so the reward always reflects
        the current policy (not a stale rollout snapshot).

        Shape stored: [n_chunk_steps, B, 1]  — same as advantages in other algorithms.
        """
        prev_logprobs = self.rollout_batch["prev_logprobs"]  # [n_chunks, B, ...]
        forward_inputs = self.rollout_batch.get("forward_inputs", None)
        n_chunks = prev_logprobs.shape[0]

        # Use the local GPU device, not prev_logprobs.device: rollout-->actor
        # transfer goes through Ray channels which deserialize CUDA tensors to
        # CPU. Reading device from prev_logprobs would silently move teacher
        # (and the whole forward) onto CPU, hanging for hours.
        device = f"{self.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        self.teacher_model = self.teacher_model.to(device)

        teacher_logprobs_list = []
        for i in range(n_chunks):
            if forward_inputs is not None:
                chunk_fwd = {
                    k: v[i].to(device)
                    for k, v in forward_inputs.items()
                    if isinstance(v, torch.Tensor)
                }
            else:
                chunk_fwd = None

            with torch.no_grad(), self.amp_context:
                teacher_out = self.teacher_model(
                    forward_inputs=chunk_fwd,
                    compute_logprobs=True,
                    compute_entropy=False,
                    compute_values=False,
                    use_cache=False,
                )
            teacher_logprobs_list.append(teacher_out["logprobs"].detach().float())

        teacher_logprobs = torch.stack(teacher_logprobs_list, dim=0)  # [n_chunks, B, ...]

        # Reduce to chunk-level scalar (sum over action dims) to match chunk_level logprob_type.
        # This mirrors how student logprobs are processed in preprocess_loss_inputs.
        while teacher_logprobs.dim() > 2:
            teacher_logprobs = teacher_logprobs.sum(dim=-1)  # [n_chunks, B]

        # Store advantages per opd_form:
        #   - reinforce (default): advantages = teacher_logprobs; r_t computed fresh in loss
        #     using current student log_prob (VLA-OPD Algorithm 1 / Flow-OPD off the shelf).
        #   - ppo: advantages = teacher_logprobs - prev_logprobs (frozen r_t at rollout);
        #     the standard PPO-clip loss is then used with rho_t = exp(logprob_now - prev_logprobs).
        opd_form = self.cfg.algorithm.get("opd_form", "reinforce")
        if opd_form == "ppo":
            prev_logprobs_red = self.rollout_batch["prev_logprobs"]
            while prev_logprobs_red.dim() > 2:
                prev_logprobs_red = prev_logprobs_red.sum(dim=-1)
            adv = (teacher_logprobs - prev_logprobs_red.to(teacher_logprobs.device)).unsqueeze(-1)
        elif opd_form == "reinforce":
            adv = teacher_logprobs.unsqueeze(-1)
        else:
            raise ValueError(f"Unknown opd_form: {opd_form!r}")
        self.rollout_batch["advantages"] = adv  # [n_chunks, B, 1]
        self.rollout_batch.pop("returns", None)

        return compute_rollout_metrics(self.rollout_batch)
