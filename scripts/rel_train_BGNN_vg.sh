#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=1
export CUDA_VISIBLE_DEVICES="7"

exp_name="BGNN-3-3-learnable_scaling"


python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relBGNN_vg.yaml" \
       EXPERIMENT_NAME "$exp_name" \
        SOLVER.IMS_PER_BATCH $[4*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 3000 \
        SOLVER.CHECKPOINT_PERIOD 3000 


