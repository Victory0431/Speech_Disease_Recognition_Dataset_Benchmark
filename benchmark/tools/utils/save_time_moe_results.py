import os
import numpy as np
import matplotlib.pyplot as plt
import csv  # 新增：导入CSV处理模块
from datetime import datetime  # 新增：替代pandas生成时间戳（可选，避免pandas依赖）
from sklearn.metrics import ConfusionMatrixDisplay

def save_time_moe_results(results, config, aggregation_strategy="mean"):
    """
    保存 Time-MoE 模型训练与评估结果（CSV格式，兼容二分类/多分类）
    核心功能：
    1. 绘制训练曲线（损失、准确率、F1、AUC）
    2. 绘制混淆矩阵（单独+综合图表）
    3. 保存详细指标到 CSV 文件（便于后续批量读取）
    """
    # 1. 初始化路径（确保输出目录存在）
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    num_classes = config.NUM_CLASSES
    class_names = config.CLASS_NAMES
    dataset_name = config.DATASET_NAME

    # 2. 提取核心指标（适配 Time-MoE 结果结构）
    train_losses = results["train_losses"]
    val_losses = results["val_losses"]
    test_losses = [epoch_metrics["loss"] for epoch_metrics in results["test_metrics_per_epoch"]]
    
    train_accuracies = results["train_accuracies"]
    val_accuracies = results["val_accuracies"]
    test_accuracies = results["test_accuracies"]
    
    test_f1_scores = [epoch_metrics["f1_score"] for epoch_metrics in results["test_metrics_per_epoch"]]
    test_auc_scores = [epoch_metrics["auc"] for epoch_metrics in results["test_metrics_per_epoch"]]
    final_test = results["final_test_metrics"]


    # -------------------------- 一、绘制综合评估图表（保持不变）--------------------------
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.figure(figsize=(16, 12))

    # 子图1：损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label=f"Train Loss", linewidth=2, marker='o', markersize=3)
    plt.plot(val_losses, label=f"Val Loss", linewidth=2, marker='s', markersize=3)
    plt.plot(test_losses, label=f"Test Loss", linewidth=2, marker='^', markersize=3)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title(f"{dataset_name} - Time-MoE Loss Curves", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 子图2：准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies, label=f"Train Accuracy", linewidth=2, marker='o', markersize=3)
    plt.plot(val_accuracies, label=f"Val Accuracy", linewidth=2, marker='s', markersize=3)
    plt.plot(test_accuracies, label=f"Test Accuracy", linewidth=2, marker='^', markersize=3)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.title(f"{dataset_name} - Time-MoE Accuracy Curves", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 子图3：混淆矩阵
    plt.subplot(2, 2, 3)
    cm = final_test["confusion_matrix"].astype(int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        cmap=plt.cm.Blues, 
        ax=plt.gca(), 
        xticks_rotation=45 if len(class_names) > 2 else 0,
        values_format='d'
    )
    plt.title(f"{dataset_name} - Final Test Confusion Matrix\n(Aggregation: {aggregation_strategy})", 
              fontsize=11, fontweight='bold')
    plt.tight_layout()

    # 子图4：F1 & AUC 曲线
    plt.subplot(2, 2, 4)
    plt.plot(test_f1_scores, label=f"Test F1 Score", linewidth=2, marker='o', markersize=3)
    auc_label = "Test AUC (One-vs-Rest)" if num_classes > 2 else "Test AUC"
    plt.plot(test_auc_scores, label=auc_label, linewidth=2, marker='s', markersize=3)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.title(f"{dataset_name} - Time-MoE F1 & AUC Curves", fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    # 保存综合图表
    picture_path = os.path.join(config.OUTPUT_DIR, f"{dataset_name}_time_moe_comprehensive.png")
    plt.tight_layout()
    plt.savefig(picture_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 综合评估图表已保存至: {picture_path}")


    # -------------------------- 二、单独保存混淆矩阵（保持不变）--------------------------
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(
        cmap=plt.cm.Blues,
        xticks_rotation=45 if len(class_names) > 2 else 0,
        values_format='d',
        include_values=True
    )
    plt.title(f"{dataset_name} - Time-MoE Final Test Confusion Matrix\n"
              f"Aggregation: {aggregation_strategy} | Total Samples: {final_test['total_samples']}",
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    confusion_matrix_path = os.path.join(config.OUTPUT_DIR, config.CONFUSION_MATRIX_FILENAME)
    plt.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 单独混淆矩阵已保存至: {confusion_matrix_path}")


    # -------------------------- 三、保存详细指标到 CSV 文件（核心修改）--------------------------
    metrics_file_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILENAME)
    # 打开CSV文件，使用csv.writer处理
    with open(metrics_file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)  # 初始化CSV写入器

        # 3.1 实验基础信息（用#开头标识注释，CSV读取时可跳过）
        writer.writerow([f"# ===== {dataset_name} Time-MoE Experiment Info ====="])
        writer.writerow([f"# Dataset Name: {dataset_name}"])
        writer.writerow([f"# Class Names: {', '.join(class_names)}"])
        writer.writerow([f"# Number of Classes: {num_classes}"])
        writer.writerow([f"# Window Aggregation Strategy: {aggregation_strategy}"])
        writer.writerow([f"# Window Size: {config.WINDOW_SIZE} (sampling points)"])
        writer.writerow([f"# Window Stride: {config.WINDOW_STRIDE} (sampling points)"])
        writer.writerow([f"# Batch Size: {config.BATCH_SIZE}"])
        writer.writerow([f"# Epochs: {config.NUM_EPOCHS}"])
        writer.writerow([f"# Learning Rate: {config.LR}"])
        writer.writerow([f"# Backbone Frozen: {config.FREEZE_BACKBONE}"])
        writer.writerow([f"# Random Seed: {config.SEED}"])
        # 用datetime替代pandas生成时间戳（减少依赖，可选保留pandas）
        writer.writerow([f"# Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([])  # 空行分隔注释与数据


        # 3.2 写入CSV表头（逗号分隔，适配二分类/多分类）
        if num_classes == 2:
            # 二分类表头：Epoch, Train Loss, Val Loss, ..., Predicted Patients
            headers = [
                "Epoch", "Train Loss", "Val Loss", "Test Loss",
                "Train Accuracy(%)", "Val Accuracy(%)", "Test Accuracy(%)",
                "Sensitivity", "Specificity", "F1 Score", "AUC",
                "TP", "TN", "FP", "FN",
                "Actual Healthy", "Actual Patients", "Predicted Healthy", "Predicted Patients"
            ]
        else:
            # 多分类表头：Epoch, Train Loss, ..., Recall_类别1, Recall_类别2, ...
            headers = [
                "Epoch", "Train Loss", "Val Loss", "Test Loss",
                "Train Accuracy(%)", "Val Accuracy(%)", "Test Accuracy(%)",
                "F1 Score (Macro)", "AUC (One-vs-Rest)", "Total Samples"
            ]
            # 追加每个类别的召回率表头
            headers.extend([f"Recall_{cls}" for cls in class_names])
        writer.writerow(headers)  # 写入表头行


        # 3.3 写入每个Epoch的指标数据（与表头一一对应）
        for epoch_idx in range(config.NUM_EPOCHS):
            epoch = epoch_idx + 1
            epoch_test = results["test_metrics_per_epoch"][epoch_idx]
            
            # 基础通用数据（所有场景共用）
            base_data = [
                epoch,
                round(train_losses[epoch_idx], 4),
                round(val_losses[epoch_idx], 4),
                round(epoch_test["loss"], 4),
                round(train_accuracies[epoch_idx], 2),
                round(val_accuracies[epoch_idx], 2),
                round(epoch_test["accuracy"], 2)
            ]

            # 二分类数据补充
            if num_classes == 2:
                binary_data = [
                    round(epoch_test["sensitivity"], 4),
                    round(epoch_test["specificity"], 4),
                    round(epoch_test["f1_score"], 4),
                    round(epoch_test["auc"], 4),
                    epoch_test["tp"],
                    epoch_test["tn"],
                    epoch_test["fp"],
                    epoch_test["fn"],
                    epoch_test["actual_healthy"],
                    epoch_test["actual_patients"],
                    epoch_test["predicted_healthy"],
                    epoch_test["predicted_patients"]
                ]
                full_data = base_data + binary_data  # 合并基础数据与二分类数据

            # 多分类数据补充
            else:
                multi_data = [
                    round(epoch_test["f1_score"], 4),
                    round(epoch_test["auc"], 4),
                    epoch_test["total_samples"]
                ]
                # 追加每个类别的召回率
                multi_data.extend([round(recall, 4) for recall in epoch_test["class_recall"]])
                full_data = base_data + multi_data  # 合并基础数据与多分类数据

            writer.writerow(full_data)  # 写入当前Epoch的完整数据行


        # 3.4 写入最终测试集指标（用#标识，作为补充注释）
        writer.writerow([])  # 空行分隔Epoch数据与最终指标
        writer.writerow([f"# ===== Final Test Set Metrics Summary ====="])
        writer.writerow([f"# Overall Accuracy: {round(final_test['accuracy'], 2)}%"])
        writer.writerow([f"# Average Loss: {round(final_test['loss'], 4)}"])
        writer.writerow([f"# F1 Score: {round(final_test['f1_score'], 4)}"])
        writer.writerow([f"# AUC: {round(final_test['auc'], 4)}"])
        writer.writerow([f"# Total Samples: {final_test['total_samples']}"])
        writer.writerow([])

        # 类别级最终指标（注释形式）
        writer.writerow([f"# Class-wise Final Metrics:"])
        if num_classes == 2:
            writer.writerow([f"# Sensitivity (Recall for {class_names[1]}): {round(final_test['sensitivity'], 4)}"])
            writer.writerow([f"# Specificity (Recall for {class_names[0]}): {round(final_test['specificity'], 4)}"])
            writer.writerow([f"# TP: {final_test['tp']}, TN: {final_test['tn']}, FP: {final_test['fp']}, FN: {final_test['fn']}"])
        else:
            for cls_idx, cls_name in enumerate(class_names):
                writer.writerow([f"# Recall for {cls_name}: {round(final_test['class_recall'][cls_idx], 4)}"])
            writer.writerow([f"# Mean Recall (Macro): {round(np.mean(final_test['class_recall']), 4)}"])

        # 混淆矩阵最终结果（注释形式，避免破坏CSV结构）
        writer.writerow([])
        writer.writerow([f"# Final Confusion Matrix (Actual → Predicted):"])
        writer.writerow([f"# Predicted Classes: {', '.join([f'Pred_{cls}' for cls in class_names])}"])
        for actual_idx, actual_cls in enumerate(class_names):
            cm_row = final_test["confusion_matrix"][actual_idx].astype(int)
            writer.writerow([f"# Actual_{actual_cls}: {', '.join(map(str, cm_row))}"])

    print(f"✅ 详细指标已保存至 CSV 文件: {metrics_file_path}")


# 主代码调用示例（无变化，确保Config中METRICS_FILENAME后缀为.csv）：
# from utils.save_time_moe_results import save_time_moe_results
# save_time_moe_results(
#     results=training_results,
#     config=config,
#     aggregation_strategy="mean"
# )