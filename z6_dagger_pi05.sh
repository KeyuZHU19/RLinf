#!/bin/bash
#SBATCH --job-name=z6-dag
#SBATCH --partition=gpu
#SBATCH --account=cis250515-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/anvil/scratch/x-kzhu7/FlowOPD/logs/z6_dagger_%j.out
#SBATCH --error=/anvil/scratch/x-kzhu7/FlowOPD/logs/z6_dagger_%j.err

# ms-BC-OPAD Phase-1a: pre-built DAgger-flow (forward-KL flow-matching BC on
# expert-resampled actions + beta curriculum) = paper Eq.10+Eq.12 core (k=1).
# branch ms-bc-opad. compute = Anvil cis250515-gpu / gpu partition.
# MODE=probe : max_steps=2, max_epochs=1, no eval (validate load+forward).
# MODE=run   : max_steps=500, eval n=50 every 50.
set -eo pipefail
MODE="${1:-probe}"
SCRATCH=/anvil/scratch/x-kzhu7
RLINF_DIR=$SCRATCH/KeyuRLinf
VENV_DIR=$SCRATCH/FlowOPD/rlinf-openpi-venv
STU="${STU:?need STU env (student ckpt dir)}"
TEACHER="${TEACHER:?need TEACHER env (expert ckpt dir)}"

LOG_DIR=$SCRATCH/FlowOPD/results/z6_dagger_${MODE}_$(date +'%Y%m%d-%H%M%S')_${SLURM_JOB_ID}
mkdir -p "$LOG_DIR" "$SCRATCH/tmp"
echo "===== z6 DAgger-flow pi0.5 (MODE=$MODE) Job=$SLURM_JOB_ID Node=$(hostname) $(date) ====="
echo "  STUDENT: $STU"
echo "  EXPERT(teacher): $TEACHER"
for d in "$STU" "$TEACHER"; do
  [[ -f "$d/model.safetensors" || -f "$d/model.safetensors.index.json" ]] || { echo "MISSING ckpt: $d" >&2; exit 1; }
done

export RAY_TMPDIR="$SCRATCH/tmp/ray-$SLURM_JOB_ID"; mkdir -p "$RAY_TMPDIR"
export RAY_DISABLE_DASHBOARD=1
export RAY_DISABLE_IMPORT_WARNING=1
export PYTHONPATH=""
export UV_CACHE_DIR=$SCRATCH/.cache/uv
export HF_HOME=$SCRATCH/.cache/huggingface
export TMPDIR=$SCRATCH/tmp
export SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-bundle.crt
export HF_ENDPOINT=https://hf-mirror.com
export NVIDIA_DRIVER_CAPABILITIES=all
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export ROBOT_PLATFORM=LIBERO
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$SCRATCH/FlowOPD/libero_shared}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load modtree/gpu
module load cuda/12.8.0
module load cudnn/cuda-12.8_9.17
module load conda/2025.02
source "$VENV_DIR/bin/activate"

export EMBODIED_PATH="${RLINF_DIR}/examples/embodiment"
export REPO_PATH=$RLINF_DIR
export OPENPI_SRC_PATH=$SCRATCH/FlowOPD/openpi/src
export PYTHONPATH="${OPENPI_SRC_PATH}:${REPO_PATH}:${PYTHONPATH:-}"

CONFIG_NAME="libero_spatial_dagger_openpi_pi05"
SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

if [[ "$MODE" == "probe" ]]; then
    MAXSTEPS=2; MAXEPOCH=1; VALINT=-1; EVALENVS=4; EVALEPOCH=1
else
    # n=50 eval reached as 10 parallel envs x 5 eval rollout epochs,
    # matching the prior z3/z5 comparison runs exactly (NOT 50 parallel
    # envs, which both oversubscribed GPU 0 at model-load and deviated
    # from the comparable-run eval methodology).
    MAXSTEPS=500; MAXEPOCH=8000; VALINT=50; EVALENVS=10; EVALEPOCH=5
fi

OVERRIDES=(
    "actor.model.model_path=$STU"
    "rollout.model.model_path=$STU"
    "rollout.expert_model.model_path=$TEACHER"
    "runner.logger.log_path=$LOG_DIR"
    "runner.logger.experiment_name=z6_dagger_pi05"
    "runner.max_epochs=$MAXEPOCH"
    "runner.max_steps=$MAXSTEPS"
    "runner.save_interval=-1"
    "runner.val_check_interval=$VALINT"
    "env.train.total_num_envs=8"
    "env.train.max_steps_per_rollout_epoch=240"
    "env.eval.total_num_envs=$EVALENVS"
    "env.eval.max_steps_per_rollout_epoch=240"
    "env.train.video_cfg.save_video=False"
    "env.eval.video_cfg.save_video=False"
    "algorithm.eval_rollout_epoch=$EVALEPOCH"
    "actor.micro_batch_size=4"
    "actor.global_batch_size=16"
    "actor.enable_offload=True"
    "rollout.enable_offload=True"
)

echo "CONFIG=$CONFIG_NAME  ENTRY=$SRC_FILE"
python "$SRC_FILE" \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name "$CONFIG_NAME" \
    "${OVERRIDES[@]}" \
    2>&1 | tee -a "${LOG_DIR}/run.log"
echo "===== Done $(date) ====="
