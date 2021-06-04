#!/bin/bash
GPU=$1
conda activate cs224u
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONUNBUFFERED=1
if [ -z "$GPU" ]
then
      echo "Starting training on CPU."
else
      echo "Starting training on GPU ${GPU}."
fi
python -u code/main.py --cfg code/cfg/cfg_file_output_disc.yml
echo "Done."
