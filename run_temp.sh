export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1300 run.py \
--run-type eval \
-c vlnce_baselines/config/CMA_AUG_DA_TUNE.yaml \
-e output_temp \
EVAL_CKPT_PATH_DIR ./model \
NUM_PROCESSES 10 \
use_ddppo True \
lamda 100.0