#!/bin/bash
#SBATCH --account=punim0811
#SBATCH --job-name=test
#SBATCH --partition=shortgpgpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=oxyoungmelbourne@gmail.com
#SBATCH --mail-type=FAIL,END
module load Python/3.5.2-intel-2017.u2-GCC-6.2.0-CUDA9
module load Tensorflow/1.10.0-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.5.2-GPU
module load Keras/2.1.5-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.5.2-GPU
module load PIL/1.1.7-intel-2016.u3
python job_dataset_config.py > job_dataset_config.txt
