#!/bin/bash

# 定义主数据集目录
MAIN_DATASET_DIR="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/fresh_datasets"

# 定义特征提取脚本路径
EXTRACTION_SCRIPT="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/MLP/MLP_feature_extraction.py"

# 定义日志目录和文件
LOG_DIR="/mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/Task2_69_classification/MLP/logs"
LOG_FILE="${LOG_DIR}/batch_extraction_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录（如果不存在）
mkdir -p "${LOG_DIR}"

# 记录开始时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "===== 批量特征提取开始于 ${START_TIME} =====" > "${LOG_FILE}"
echo "主数据集目录: ${MAIN_DATASET_DIR}" >> "${LOG_FILE}"
echo "特征提取脚本: ${EXTRACTION_SCRIPT}" >> "${LOG_FILE}"
echo "----------------------------------------" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# 检查主数据集目录是否存在
if [ ! -d "${MAIN_DATASET_DIR}" ]; then
    echo "错误: 主数据集目录 ${MAIN_DATASET_DIR} 不存在!" >> "${LOG_FILE}"
    echo "批量特征提取失败于 $(date +"%Y-%m-%d %H:%M:%S")" >> "${LOG_FILE}"
    exit 1
fi

# 检查特征提取脚本是否存在
if [ ! -f "${EXTRACTION_SCRIPT}" ]; then
    echo "错误: 特征提取脚本 ${EXTRACTION_SCRIPT} 不存在!" >> "${LOG_FILE}"
    echo "批量特征提取失败于 $(date +"%Y-%m-%d %H:%M:%S")" >> "${LOG_FILE}"
    exit 1
fi

# 获取主目录下的所有子文件夹（数据集）
DATASET_DIRS=$(find "${MAIN_DATASET_DIR}" -maxdepth 1 -type d ! -path "${MAIN_DATASET_DIR}")

# 统计数据集数量
DATASET_COUNT=$(echo "${DATASET_DIRS}" | wc -l)
echo "发现 ${DATASET_COUNT} 个数据集，开始处理..." >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# 遍历每个数据集目录并处理
COUNT=0
for DATASET_DIR in ${DATASET_DIRS}; do
    COUNT=$((COUNT + 1))
    DATASET_NAME=$(basename "${DATASET_DIR}")
    
    echo "===== 开始处理第 ${COUNT}/${DATASET_COUNT} 个数据集: ${DATASET_NAME} =====" >> "${LOG_FILE}"
    echo "处理时间: $(date +"%Y-%m-%d %H:%M:%S")" >> "${LOG_FILE}"
    echo "数据集路径: ${DATASET_DIR}" >> "${LOG_FILE}"
    
    # 执行特征提取脚本
    python "${EXTRACTION_SCRIPT}" "${DATASET_DIR}" >> "${LOG_FILE}" 2>&1
    
    # 记录处理结果
    if [ $? -eq 0 ]; then
        echo "数据集 ${DATASET_NAME} 处理成功" >> "${LOG_FILE}"
    else
        echo "错误: 数据集 ${DATASET_NAME} 处理失败" >> "${LOG_FILE}"
    fi
    
    echo "===== 第 ${COUNT}/${DATASET_COUNT} 个数据集处理结束 =====" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
done

# 记录结束时间
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "===== 批量特征提取结束于 ${END_TIME} =====" >> "${LOG_FILE}"
echo "共处理 ${COUNT} 个数据集" >> "${LOG_FILE}"

echo "批量处理完成! 日志文件: ${LOG_FILE}"
    