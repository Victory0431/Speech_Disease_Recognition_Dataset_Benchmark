#!/bin/bash
# 确保传入参数合法（可选）
if [ "$#" -ne 0 ]; then
    echo "Usage: $0 (no arguments required)"
    exit 1
fi

MAX_INDEX=27  # 数据集最大索引
BATCH_SIZE=4  # 每个任务处理的样本数

for i in {0..7}; do
    start=$((i * BATCH_SIZE))
    end=$((start + BATCH_SIZE - 1))  # 因为range是闭区间[start, end]
    
    # 安全边界检查
    if [ $end -gt $MAX_INDEX ]; then
        end=$MAX_INDEX
        echo "&#9888;️ Truncated end index from $((end+1)) to $MAX_INDEX for GPU $i"
    fi
    
    # 构造唯一日志名：包含GPU号、起始/结束索引
    log_file="cnn_gpu${i}_start_${start}_end_${end}.log"
    
    echo "Launching job on GPU $i (samples ${start}-${end})..."
    nohup python batch_run_mlp_cnn.py \
        --start $start --end $end --gpu $i --model cnn \
        > "$log_file" 2>&1 &
done

echo "All jobs submitted! Check logs with:"
ls cnn_gpu*.log | tr '\n' ' '
