#!/bin/bash
#SBATCH --qos=es_normal
#SBATCH --partition=es1
#SBATCH --account=ac_mak
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --constraint=es1_v100
#SBATCH --gres=gpu:2 --cpus-per-task=4
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out


date

neat run --config yaml_ml_instructions/IMGVR_sample.yaml &> IMGVR_sample.out

date
 
