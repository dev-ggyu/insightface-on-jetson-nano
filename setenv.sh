#!/bin/bash

export CUDA_HOME=/usr/local/cuda-10.0:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.0/bin:$PATH
export LLVM_CONFIG=llvm-config-6.0
export PYTHONPATH=$(pwd)
