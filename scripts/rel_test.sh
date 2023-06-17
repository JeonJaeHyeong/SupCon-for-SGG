export CUDA_VISIBLE_DEVICES="4"
export gpu_num=1
export use_multi_gpu=false
export task='sgdet'

export test_list=('final') # checkpoint

export save_result=True
export output_dir="/home/users/jaehyeong/papers/PySGG/checkpoints/sgdet-BGNNPredictor/(2023-03-08_22)BGNN-3-3-learnable_scaling(resampling)" # Please input the checkpoint directory

if $use_multi_gpu;then
    for name in ${test_list[@]}
    do
        python -m torch.distributed.launch --master_port 10025 --nproc_per_node=${gpu_num} tools/relation_test_net.py --config-file "${output_dir}/config.yml" \
            TEST.IMS_PER_BATCH $[6*$gpu_num] \
            TEST.SAVE_PROPOSALS ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_${name}.pth"
    done
else
    for name in ${test_list[@]}
    do
        python tools/relation_test_net.py --config-file "${output_dir}/config.yml"  \
            TEST.IMS_PER_BATCH $[3*$gpu_num] \
            TEST.SAVE_PROPOSALS ${save_result} \
            OUTPUT_DIR ${output_dir} \
            MODEL.WEIGHT "${output_dir}/model_${name}.pth"
    done
fi