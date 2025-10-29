#!/bin/bash
#SBATCH -N 4
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -J CV
#SBATCH -A <redacted>
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=128

PORT=6374

module load tensorflow/2.12.0

HEAD_NODE=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
HEAD_ADDRESS=$(ray start --head --port=$PORT 2>&1 | grep "ray start --address" | awk -F"'" '{print $2}')

echo "Ray head address is: $HEAD_ADDRESS"

sleep 10

WORKER_NODES=$(scontrol show hostnames "$SLURM_NODELIST" | tail -n +2)

echo "Worker nodes are: $WORKER_NODES"

srun --nodes=$((SLURM_NNODES-1)) --nodelist="$WORKER_NODES" ray start --address="$HEAD_ADDRESS" --block &
srun --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" python3 contrastive_vae_tuner.py --address="$HEAD_ADDRESS"

wait
