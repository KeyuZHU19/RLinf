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
            state_dict = torch.load(teacher_ckpt, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]
            self.teacher_model.load_state_dict(state_dict, strict=True)

        device = f"{self.torch_device_type}:{int(os.environ['LOCAL_RANK'])}"
        self.teacher_model = self.teacher_model.to(device)
        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

        self.log_info("[OPD] Teacher model initialised and frozen.")

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """Replace standard advantage computation with OPD KL intrinsic rewards.

        Computes r_t = log pi_teacher(a_t|s_t) - log pi_student_old(a_t|s_t)
        for every chunk step and stores the result in rollout_batch["advantages"]
        with shape [n_chunk_steps, B, 1] so that the existing training pipeline
        consumes it unchanged via loss_type: opd_flow.
        """
        prev_logprobs = self.rollout_batch["prev_logprobs"]  # [n_chunks, B, ...]
        forward_inputs = self.rollout_batch.get("forward_inputs", None)
        n_chunks = prev_logprobs.shape[0]

        device = prev_logprobs.device
        self.teacher_model = self.teacher_model.to(device)

        teacher_logprobs_list = []
        for i in range(n_chunks):
            if forward_inputs is not None:
                chunk_fwd = {
                    k: v[i]
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

        # KL reward: sum over action-chunk dimension if present
        kl_reward = teacher_logprobs - prev_logprobs.float()
        while kl_reward.dim() > 2:
            kl_reward = kl_reward.sum(dim=-1)

        # Shape expected by preprocess_embodied_advantages_inputs: [n_chunks, B, 1]
        kl_reward = kl_reward.unsqueeze(-1).detach()

        self.rollout_batch["advantages"] = kl_reward
        self.rollout_batch.pop("returns", None)

        return compute_rollout_metrics(self.rollout_batch)
