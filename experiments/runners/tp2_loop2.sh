#!/bin/bash
set -u
cd /workspace/mirage
export NVSHMEM_INC_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/include
export NVSHMEM_LIB_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib
export MPI_INC_PATH=/usr/local/mpi/include MPI_LIB_PATH=/usr/local/mpi/lib
for i in 1 2 3; do
  CUDA_VISIBLE_DEVICES=1,2 mpirun -np 2 --allow-run-as-root \
    -x UCX_TLS=shm,self,tcp -x UCX_NET_DEVICES=eth0 \
    -x NVSHMEM_INC_PATH -x NVSHMEM_LIB_PATH -x MPI_INC_PATH -x MPI_LIB_PATH \
    -x LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:/usr/local/mpi/lib \
    -x CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 -x CUDA_ENABLE_LIGHTWEIGHT_COREDUMP=1 \
    -x CUDA_COREDUMP_FILE=/tmp/tp2mk_r${i}_%p.cudmp \
    python demo/qwen3/demo.py --use-mirage --model-path /workspace/qwen3-8b-mp2 \
    > /tmp/tp2mk_r${i}.log 2>&1
  echo "run $i exit=$?" >> /tmp/tp2_loop2.log
done
touch /tmp/tp2_loop2.done
