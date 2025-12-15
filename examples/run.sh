export ASCEND_RT_VISIBLE_DEVICES=7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

export CPLUS_INCLUDE_PATH=/usr/include/c++/12/:/usr/include/c++/12/aarch64-openEuler-linux/:$CPLUS_INCLUDE_PATH

# Arm
JEMALLOC_ARM_PATH="/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libjemalloc.so"
# X86
JEMALLOC_X86_PATH="/usr/local/Ascend/ascend-toolkit/latest/x86-linux/lib64/libjemalloc.so"

# 获取当前机器的平台
PLATFORM=$(uname -m)
if [ "$PLATFORM" == "aarch64" ]; then
    JEMALLOC_PATH=$JEMALLOC_ARM_PATH
else
    JEMALLOC_PATH=$JEMALLOC_X86_PATH
fi

if [ -f "$JEMALLOC_PATH" ]; then
    export LD_PRELOAD=$JEMALLOC_PATH:$LD_PRELOAD
else
    echo "Jemalloc is not installed"
fi

torchrun --nproc-per-node=1 ./projects/inference_seedvr2_3b.py \
    --video_path ./examples/test_videos \
    --output_dir ./examples/results \
    --seed 666 \
    --res_h 1280 \
    --res_w 720 \
    --sp_size 1