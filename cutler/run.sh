#!/bin/bash
# Andrew H. Fagg
#
# Example with one experiment
#
# When you use this batch file:
#  Change the email address to yours! (I don't want email about your experiments!)
#  Change the chdir line to match the location of where your code is located
#
# Reasonable partitions: debug_5min, debug_30min, normal, debug_gpu, gpu
#
#
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
# memory in MB
#SBATCH --mem=2048
# The %j is translated into the job number
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --time=2-00:00:00
#SBATCH --job-name=train_det
#SBATCH --mail-user=steven.howell-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/d70howell/D70/CutLER/cutler
#SBATCH --gres=gpu:1
#
#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
# Change this line to start an instance of your experiment
echo "test"
nvidia-smi
conda info --envs
eval "$(conda shell.bash hook)"
conda activate exp
conda info --envs
echo "test"
nvidia-smi
CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib
echo $CUDNN_PATH
echo $CONDA_PREFIX
python train_net.py --num-gpus 2   --config-file model_zoo/configs/CODS/cascade_mask_rcnn_R_50_FPN_self_train.yaml
