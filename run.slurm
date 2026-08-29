#!/bin/bash
#SBATCH --job-name=CL_test         # Job name
#SBATCH --output=log/pytorch_job_%j.out  # Standard output log (%j is the job ID)
#SBATCH --error=log/pytorch_job_%j.err   # Standard error log
#SBATCH --mem=50G                         # replace with amount suitable for your job
#SBATCH --time=1:00:00                   # replace with amount suitable for your job
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks (processes)
#SBATCH --cpus-per-task=24                  # Number of CPU cores per task
#SBATCH --gres=gpu:1                    # Number of GPUs per node
#SBATCH --partition=gpu                   # Partition name


conda run -n ttime_new python main.py \
  --config=./exps/ttime_officehome.json \
  --seed=1995 \
  --init_epoch=1 \
  --epochs=1 \
  --prefix="test" \