#! /bin/bash

#SBATCH -J gd_instance  
#SBATCH -A kreshuk                             
#SBATCH -N 1				       # number of cluster nodes for the job
#SBATCH -n 8				       # number of cores per node for the job
#SBATCH --mem 128G			       # amount of memory per node
#SBATCH -t 24:00:00                              # runtime of the job, acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds". 
#SBATCH -o /g/kreshuk/buglakova/projects/fibsem_segm/unet_F107_A1_full/experiments/exp_01_generalized_dice/slurm-%j.out-%N			       # file to write the command line output to
#SBATCH -e /g/kreshuk/buglakova/projects/fibsem_segm/unet_F107_A1_full/experiments/exp_01_generalized_dice/slurm-%j.err-%N			       # file to write the error output to
#SBATCH --mail-type=BEGIN,END,FAIL		       # mail notifications types
#SBATCH --mail-user=elena.buglakova@embl.de    # mail address for mail notifications 
#SBATCH -p gpu-el8				       # queue to submit to; gpu-el8 - GPU queue, htc-el8 - default CPU queue
#SBATCH -C gpu=3090			       # specify the type of gpu you want
#SBATCH --gres=gpu:1			       # specify the number of gpus per node

source /home/buglakov/.bashrc
source activate torchem

python /g/kreshuk/buglakova/projects/fibsem_segm/unet_F107_A1_full/experiments/exp_01_generalized_dice/train.py