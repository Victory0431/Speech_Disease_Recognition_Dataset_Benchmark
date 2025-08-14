import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_results(metrics, config):
    """
    保存模型训练结果，支持二分类和多分类场景
    - 绘制训练曲线、混淆矩阵等图表
    - 保存详细指标到文件
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    num_classes = len(metrics["class_names"])
    class_names = metrics["class_names"]

    # -------------------------- 绘制图表 --------------------------
    plt.figure(figsize=(14, 10))
    
    # 1. 损失曲线（通用）
    plt.subplot(2, 2, 1)
    plt.plot(metrics["train_losses"], label="Training Loss")
    plt.plot(metrics["val_losses"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # 2. 准确率曲线（通用）
    plt.subplot(2, 2, 2)
    plt.plot(metrics["train_accuracies"], label="Training Accuracy")
    plt.plot(metrics["val_accuracies"], label="Validation Accuracy")
    plt.plot(metrics["test_accuracies"], label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curves")
    plt.legend()

    # 3. 混淆矩阵（动态适应类别数）
    plt.subplot(2, 2, 3)
    cm = metrics["final_test_metrics"]["confusion_matrix"]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
    plt.title('Final Test Set Confusion Matrix')
    plt.tight_layout()

    # 4. F1和AUC曲线（通用，多分类AUC为one-vs-rest）
    plt.subplot(2, 2, 4)
    f1_scores = [m["f1_score"] for m in metrics["test_metrics_per_epoch"]]
    auc_scores = [m["auc"] for m in metrics["test_metrics_per_epoch"]]
    plt.plot(f1_scores, label="Test F1 Score")
    plt.plot(auc_scores, label="Test AUC")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("F1 and AUC Curves")
    plt.legend()

    # 保存综合图表
    plot_path = os.path.join(config.OUTPUT_DIR, config.PLOT_FILENAME)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"训练曲线和评估图表已保存至: {plot_path}")

    # 单独保存混淆矩阵
    cm_path = os.path.join(config.OUTPUT_DIR, config.CONFUSION_MATRIX_FILENAME)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Final Test Set Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"混淆矩阵已保存至: {cm_path}")

    # -------------------------- 保存详细指标到文件 --------------------------
    metrics_path = os.path.join(config.OUTPUT_DIR, config.METRICS_FILENAME)
    with open(metrics_path, "w") as f:
        # 1. 写入表头（根据类别数动态生成）
        header = "Epoch\tTrain Loss\tVal Loss\tTrain Accuracy(%)\tVal Accuracy(%)\tTest Accuracy(%)\t"
        if num_classes == 2:
            # 二分类特有表头
            header += "Sensitivity\tSpecificity\tF1 Score\tAUC\tTP\tTN\tFP\tFN\tTotal Samples\t"
            header += "Actual Healthy\tActual Patients\tPredicted Healthy\tPredicted Patients\n"
        else:
            # 多分类表头（通用指标+每个类别的召回率）
            header += "F1 Score (macro)\tAUC (ovr)\tTotal Samples\t"
            for i in range(num_classes):
                header += f"Recall_{class_names[i]}\t"
            header += "\n"
        f.write(header)

        # 2. 写入每个epoch的指标
        for i in range(len(metrics["train_losses"])):
            test = metrics["test_metrics_per_epoch"][i]
            line = f"{i+1}\t"
            line += f"{metrics['train_losses'][i]:.4f}\t{metrics['val_losses'][i]:.4f}\t"
            line += f"{metrics['train_accuracies'][i]:.2f}\t{metrics['val_accuracies'][i]:.2f}\t{test['accuracy']:.2f}\t"

            if num_classes == 2:
                # 二分类特有指标
                line += f"{test['sensitivity']:.4f}\t{test['specificity']:.4f}\t{test['f1_score']:.4f}\t{test['auc']:.4f}\t"
                line += f"{test['tp']}\t{test['tn']}\t{test['fp']}\t{test['fn']}\t{test['total_samples']}\t"
                line += f"{test['actual_healthy']}\t{test['actual_patients']}\t{test['predicted_healthy']}\t{test['predicted_patients']}\n"
            else:
                # 多分类指标（F1+AUC+每个类别的召回率）
                line += f"{test['f1_score']:.4f}\t{test['auc']:.4f}\t{test['total_samples']}\t"
                for recall in test['class_recall']:
                    line += f"{recall:.4f}\t"
                line += "\n"
            f.write(line)

        # 3. 写入最终测试集指标汇总
        f.write("\n===== Final Test Set Metrics Summary =====\n")
        final = metrics["final_test_metrics"]
        f.write(f"Overall Accuracy: {final['accuracy']:.2f}%\n")
        f.write(f"F1 Score: {final['f1_score']:.4f}\n")
        f.write(f"AUC: {final['auc']:.4f}\n\n")

        # 类别级指标（二分类vs多分类）
        if num_classes == 2:
            f.write(f"Sensitivity (Recall for {class_names[1]}): {final['sensitivity']:.4f}\n")
            f.write(f"Specificity (Recall for {class_names[0]}): {final['specificity']:.4f}\n\n")
        else:
            f.write("Class-wise Recall:\n")
            for i in range(num_classes):
                f.write(f"  {class_names[i]}: {final['class_recall'][i]:.4f}\n")
            f.write(f"Mean Recall (macro): {np.mean(final['class_recall']):.4f}\n\n")

        # 混淆矩阵详情（动态适应类别数）
        f.write("Confusion Matrix:\n")
        # 写入表头（预测类别）
        f.write("\t" + "\t".join([f"Predicted {name}" for name in class_names]) + "\n")
        # 写入每行（实际类别）
        for i in range(num_classes):
            row = [f"Actual {class_names[i]}"] + [str(final['confusion_matrix'][i][j]) for j in range(num_classes)]
            f.write("\t".join(row) + "\n")

    print(f"详细指标已保存至: {metrics_path}")
    