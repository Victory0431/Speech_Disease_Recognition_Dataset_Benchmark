#!/bin/bash

# 配置路径
MAIN_DATA_DIR="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"
EXTRACT_SCRIPT="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/cnn_feature_extractor.py"
LOG_DIR="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/CNN/CNN_extraction_logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/cnn_extraction_${TIMESTAMP}.log"

# 日志函数
log() {
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $1" >> "$LOG_FILE"
}

# 初始化日志
log "===== 开始批量CNN特征提取任务 ====="
log "主数据集目录: $MAIN_DATA_DIR"
log "特征提取脚本: $EXTRACT_SCRIPT"
log "日志文件路径: $LOG_FILE"
log "=================================="
log ""

# 检查主数据目录是否存在
if [ ! -d "$MAIN_DATA_DIR" ]; then
    log "错误: 主数据集目录 $MAIN_DATA_DIR 不存在!"
    log "===== 任务失败 ====="
    exit 1
fi

# 检查提取脚本是否存在
if [ ! -f "$EXTRACT_SCRIPT" ]; then
    log "错误: 特征提取脚本 $EXTRACT_SCRIPT 不存在!"
    log "===== 任务失败 ====="
    exit 1
fi

# 获取所有子数据集目录
DATASETS=$(find "$MAIN_DATA_DIR" -maxdepth 1 -type d ! -path "$MAIN_DATA_DIR")

# 统计数据集数量
DATASET_COUNT=$(echo "$DATASETS" | wc -l | tr -d ' ')
log "发现 $DATASET_COUNT 个子数据集，开始处理..."
log ""

# 处理每个数据集
PROCESS_COUNT=0
SUCCESS_COUNT=0
FAIL_COUNT=0

for DATASET in $DATASETS; do
    PROCESS_COUNT=$((PROCESS_COUNT + 1))
    DATASET_NAME=$(basename "$DATASET")
    
    log "===== 处理第 $PROCESS_COUNT/$DATASET_COUNT 个数据集 ====="
    log "数据集名称: $DATASET_NAME"
    log "数据集路径: $DATASET"
    log "开始处理时间: $(date +"%Y-%m-%d %H:%M:%S")"
    
    # 执行特征提取
    python "$EXTRACT_SCRIPT" "$DATASET" >> "$LOG_FILE" 2>&1
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        log "数据集 $DATASET_NAME 处理成功"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        log "错误: 数据集 $DATASET_NAME 处理失败"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    log "结束处理时间: $(date +"%Y-%m-%d %H:%M:%S")"
    log "=================================================="
    log ""
done

# 总结结果
log "===== 批量处理完成 ====="
log "总处理数据集: $PROCESS_COUNT"
log "处理成功: $SUCCESS_COUNT"
log "处理失败: $FAIL_COUNT"
log "完成时间: $(date +"%Y-%m-%d %H:%M:%S")"
log "详细日志请查看: $LOG_FILE"
log "========================"

echo "批量处理完成! 日志文件: $LOG_FILE"
echo "处理结果: 成功 $SUCCESS_COUNT 个, 失败 $FAIL_COUNT 个, 总计 $PROCESS_COUNT 个"
    