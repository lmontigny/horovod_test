# 1 machine with 4 GPUs
mpirun -np 4 \
 -bind-to none -map-by slot \
 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
 -mca pml ob1 -mca btl ^openib \
 python train.py
 
 # 4 machine with 4 GPU each
 mpirun -np 16 \
 -bind-to none -map-by slot \
 -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
 -mca pml ob1 -mca btl ^openib \
 python train.py
