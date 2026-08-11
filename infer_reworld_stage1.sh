torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
 scripts/sample_dualflow_dit.py configs/dualflow/sample_reworld.yaml > sample_reworld_stage1.txt 2>&1