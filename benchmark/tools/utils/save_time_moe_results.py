# save as: tools/utils/save_time_moe_results.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_time_moe_results(results, config, aggregation_strategy="mean"):
    """
    保存 Time-MoE 模型训练与评估结果（兼容二分类/多分类）
    核心功能：
    1. 绘制训练曲线（损失、准确率、F1、AUC）
    2. 绘制混淆矩阵（单独+综合图表）
    3. 保存详细指标到文本文件（含 Time-MoE 特有参数）
    
    Args:
        results: train_and_evaluate_time_moe 返回的结果字典
        config: Config 类实例（含路径、类别名等配置）
        aggregation_strategy: 时序窗口聚合策略（需记录，用于复现）
    """
    # 1. 初始化路径（确保输出目录存在）
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    num_classes = config.NUM_CLASSES
    class_names = config.CLASS_NAMES
    dataset_name = config.DATASET_NAME

    # 2. 提取核心指标（适配 Time-MoE 结果结构）
    train_losses = results["train_losses"]
    val_losses = results["val_losses"]
    # 从每个 epoch 的测试指标中提取测试损失（Time-MoE 特有）
    test_losses = [epoch_metrics["loss"] for epoch_metrics in results["test_metrics_per_epoch"]]
    
    train_accuracies = results["train_accuracies"]
    val_accuracies = results["val_accuracies"]
    test_accuracies = results["test_accuracies"]
    
    # 提取 F1 和 AUC 曲线数据
    test_f1_scores = [epoch_metrics["f1_score"] for epoch_metrics in results["test_metrics_per_epoch"]]
    test_auc_scores = [epoch_metrics["auc"] for epoch_metrics in results["test_metrics_per_epoch"]]
    
    # 最终测试集指标
    final_test = results["final_test_metrics"]


    # -------------------------- 一、绘制综合评估图表 --------------------------
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 避免中文乱码
    plt.figure(figsize=(16, 12))  # 扩大画布，适配 4 个子图

    # 子图1：损失曲线（新增测试损失，Time-MoE 特有）
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

    # 子图3：混淆矩阵（动态适配类别数）
    plt.subplot(2, 2, 3)
    cm = final_test["confusion_matrix"]
    # 确保混淆矩阵数值为整数（避免小数显示）
    cm = cm.astype(int)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )
    disp.plot(
        cmap=plt.cm.Blues, 
        ax=plt.gca(), 
        xticks_rotation=45 if len(class_names) > 2 else 0,  # 多分类时旋转标签
        values_format='d'  # 显示整数
    )
    plt.title(f"{dataset_name} - Final Test Confusion Matrix\n(Aggregation: {aggregation_strategy})", 
              fontsize=11, fontweight='bold')
    plt.tight_layout()

    # 子图4：F1 & AUC 曲线（多分类时标注 AUC 计算方式）
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
    plt.savefig(picture_path, dpi=300, bbox_inches="tight")  # 高分辨率保存
    plt.close()
    print(f"✅ 综合评估图表已保存至: {picture_path}")


    # -------------------------- 二、单独保存混淆矩阵（便于论文使用） --------------------------
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm.astype(int),
        display_labels=class_names
    )
    disp.plot(
        cmap=plt.cm.Blues,
        xticks_rotation=45 if len(class_names) > 2 else 0,
        values_format='d',
        include_values=True  # 显示混淆矩阵数值
    )
    plt.title(f"{dataset_name} - Time-MoE Final Test Confusion Matrix\n"
              f"Aggregation: {aggregation_strategy} | Total Samples: {final_test['total_samples']}",
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    混淆矩阵路径 = os.path.join(config.OUTPUT_DIR, config.CONFUSION_MATRIX_FILENAME)
    plt.savefig(混淆矩阵路径, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 单独混淆矩阵已保存至: {混淆矩阵路径}")


    # -------------------------- 三、保存详细指标到文本文件 --------------------------
    指标文件路径 = os.path.join(config.OUTPUT_DIR, config.METRICS_FILENAME)
    with open(指标文件路径, "w", encoding="utf-8") as f:
        # 3.1 写入实验基础信息（Time-MoE 特有参数，便于复现）
        f.write(f"===== {dataset_name} Time-MoE Experiment Info =====\n")
        f.write(f"Dataset Name: {dataset_name}\n")
        f.write(f"Class Names: {', '.join(class_names)}\n")
        f.write(f"Number of Classes: {num_classes}\n")
        f.write(f"Window Aggregation Strategy: {aggregation_strategy}\n")
        f.write(f"Window Size: {config.WINDOW_SIZE} (sampling points)\n")
        f.write(f"Window Stride: {config.WINDOW_STRIDE} (sampling points)\n")
        f.write(f"Batch Size: {config.BATCH_SIZE}\n")
        f.write(f"Epochs: {config.NUM_EPOCHS}\n")
        f.write(f"Learning Rate: {config.LR}\n")
        f.write(f"Backbone Frozen: {config.FREEZE_BACKBONE}\n")
        f.write(f"Random Seed: {config.SEED}\n")
        f.write(f"Experiment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  # 需导入pandas
        f.write("\n")

        # 3.2 写入表头（动态适配二分类/多分类，新增测试损失列）
        f.write("===== Per-Epoch Metrics =====\n")
        表头 = (
            "Epoch\t"
            "Train Loss\tVal Loss\tTest Loss\t"  # 新增 Test Loss
            "Train Accuracy(%)\tVal Accuracy(%)\tTest Accuracy(%)\t"
        )
        if num_classes == 2:
            # 二分类特有指标
            表头 += (
                "Sensitivity\tSpecificity\t"
                "F1 Score\tAUC\t"
                "TP\tTN\tFP\tFN\t"
                "Actual Healthy\tActual Patients\t"
                "Predicted Healthy\tPredicted Patients\n"
            )
        else:
            # 多分类特有指标（宏观F1 + OVR-AUC + 每个类别的召回率）
            表头 += (
                "F1 Score (Macro)\tAUC (One-vs-Rest)\t"
                "Total Samples\t"
                f"{'\t'.join([f'Recall_{cls}' for cls in class_names])}\n"
            )
        f.write(表头)

        # 3.3 写入每个 Epoch 的详细指标
        for epoch_idx in range(config.NUM_EPOCHS):
            epoch = epoch_idx + 1
            epoch_test = results["test_metrics_per_epoch"][epoch_idx]
            
            # 基础指标（所有场景通用）
            行数据 = (
                f"{epoch}\t"
                f"{train_losses[epoch_idx]:.4f}\t{val_losses[epoch_idx]:.4f}\t{epoch_test['loss']:.4f}\t"  # 测试损失
                f"{train_accuracies[epoch_idx]:.2f}\t{val_accuracies[epoch_idx]:.2f}\t{epoch_test['accuracy']:.2f}\t"
            )

            # 二分类特有指标补充
            if num_classes == 2:
                行数据 += (
                    f"{epoch_test['sensitivity']:.4f}\t{epoch_test['specificity']:.4f}\t"
                    f"{epoch_test['f1_score']:.4f}\t{epoch_test['auc']:.4f}\t"
                    f"{epoch_test['tp']}\t{epoch_test['tn']}\t{epoch_test['fp']}\t{epoch_test['fn']}\t"
                    f"{epoch_test['actual_healthy']}\t{epoch_test['actual_patients']}\t"
                    f"{epoch_test['predicted_healthy']}\t{epoch_test['predicted_patients']}\n"
                )
            # 多分类特有指标补充
            else:
                行数据 += (
                    f"{epoch_test['f1_score']:.4f}\t{epoch_test['auc']:.4f}\t"
                    f"{epoch_test['total_samples']}\t"
                    f"{'\t'.join([f'{recall:.4f}' for recall in epoch_test['class_recall']])}\n"
                )
            f.write(行数据)

        # 3.4 写入最终测试集指标汇总（突出显示）
        f.write("\n===== Final Test Set Metrics Summary =====\n")
        f.write(f"Overall Accuracy: {final_test['accuracy']:.2f}%\n")
        f.write(f"Average Loss: {final_test['loss']:.4f}\n")
        f.write(f"F1 Score: {final_test['f1_score']:.4f}\n")
        f.write(f"AUC: {final_test['auc']:.4f}\n")
        f.write(f"Total Samples: {final_test['total_samples']}\n\n")

        # 类别级指标（二分类 vs 多分类）
        f.write("Class-wise Metrics:\n")
        if num_classes == 2:
            f.write(f"Sensitivity (Recall for {class_names[1]}): {final_test['sensitivity']:.4f}\n")
            f.write(f"Specificity (Recall for {class_names[0]}): {final_test['specificity']:.4f}\n")
            f.write(f"True Positive (TP): {final_test['tp']}\n")
            f.write(f"True Negative (TN): {final_test['tn']}\n")
            f.write(f"False Positive (FP): {final_test['fp']}\n")
            f.write(f"False Negative (FN): {final_test['fn']}\n")
        else:
            for cls_idx, cls_name in enumerate(class_names):
                f.write(f"Recall for {cls_name}: {final_test['class_recall'][cls_idx]:.4f}\n")
            f.write(f"Mean Recall (Macro): {np.mean(final_test['class_recall']):.4f}\n")

        # 混淆矩阵详情（格式化显示，便于阅读）
        f.write("\nConfusion Matrix (Actual → Predicted):\n")
        # 写入混淆矩阵表头（预测类别）
        f.write(f"\t{' '.join([f'Pred_{cls:<8}' for cls in class_names])}\n")
        # 写入混淆矩阵每行（实际类别 + 数值）
        for actual_idx, actual_cls in enumerate(class_names):
            行 = [f"Actual_{actual_cls}"] + [f"{final_test['confusion_matrix'][actual_idx][pred_idx]:<8d}" 
                                           for pred_idx in range(num_classes)]
            f.write(f"\t{' '.join(行)}\n")

    print(f"✅ 详细指标已保存至: {指标文件路径}")


# 注意：需在主代码中导入 pandas（用于时间戳），若未安装可执行：pip install pandas
# 主代码调用示例（在 train_and_evaluate_time_moe 后）：
# from utils.save_time_moe_results import save_time_moe_results
# save_time_moe_results(
#     results=training_results,
#     config=config,
#     aggregation_strategy="mean"  # 与评估时使用的策略一致
# )