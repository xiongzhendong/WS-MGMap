export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node=1 --master_port=1668 run.py \
--run-type eval \
-c vlnce_baselines/config/CMA_AUG_DA_TUNE.yaml \
-e output \
EVAL_CKPT_PATH_DIR ./model \
NUM_PROCESSES 10 \
use_ddppo True \
lamda 0.9
# conda activate /share/home/wangzixu/miniconda3/envs/wsmgmap
# cd /share/home/wangzixu/xiongzhendong/WS-MGMap
# bash run.sh
# setfacl -R -m u:sunhaowei:rwx /share/home/wangzixu/xiongzhendong/WS-MGMap

# ssh sun_mu01
# ssh gpu21
# ssh kang_mu01
# ssh gpu28
# source /share/home/wangzixu/.bashrc