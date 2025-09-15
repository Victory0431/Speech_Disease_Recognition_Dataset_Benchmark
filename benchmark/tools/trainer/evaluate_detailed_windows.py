import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)
from pathlib import Path


def evaluate_model_detailed_windows(model, loader, aggregate="mean", log_dir="./logs"):
    """
    逐文件评估（增强版）：
    - 增加混淆矩阵可视化
    - 打印详细日志（每个文件的预测结果）
    - 统计类别级别的预测情况
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    model.eval()
    file_preds, file_labels = {}, {}
    detailed_log = []  # 存储详细日志

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) != 3:
                raise ValueError("Dataset must return (segments, label, file_id) in eval mode")

            segments, label, file_id = batch
            segments = segments.squeeze(0).to(model.device)   # [num_windows, T]
            true_label = label.item()
            file_id = file_id if isinstance(file_id, str) else file_id[0]
            file_type = "covid" if true_label == 1 else "non-covid"

            # 窗口级预测
            logits_list = []
            for seg_idx, seg in enumerate(segments):
                seg = seg.unsqueeze(0)  # [1, T]
                logits, _ = model(seg)  # [1, num_classes]
                logits_list.append(logits)

            # 聚合窗口结果（修复max聚合的bug）
            logits = torch.cat(logits_list, dim=0)  # [num_windows, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            
            if aggregate == "mean":
                file_prob = probs.mean(axis=0)  # [2]
            elif aggregate == "max":
                # 修复：之前返回的是 (max_val, index) 元组，这里只取最大值
                file_prob = probs.max(axis=0)[0]  # [2]
            else:
                raise ValueError(f"Unknown aggregate {aggregate}")

            # 文件级预测结果
            pred_prob = file_prob[1]  # 类别1的概率
            pred_label = 1 if pred_prob >= 0.5 else 0
            pred_type = "covid" if pred_label == 1 else "non-covid"

            # 记录结果
            file_preds[file_id] = file_prob
            file_labels[file_id] = true_label

            # 详细日志（记录每个文件的预测情况）
            log_line = (
                f"文件ID: {file_id}\n"
                f"  真实标签: {true_label} ({file_type})\n"
                f"  预测概率 (covid): {pred_prob:.4f}\n"
                f"  预测标签: {pred_label} ({pred_type})\n"
                f"  预测结果: {'正确' if pred_label == true_label else '错误'}\n"
                "-----------------------------------------"
            )
            detailed_log.append(log_line)
            # print(log_line)  # 终端打印


    # ---- 汇总全局结果 ----
    y_true = [file_labels[f] for f in file_labels.keys()]
    y_prob = [file_preds[f][1] for f in file_preds.keys()]  # 类别1的概率
    y_pred = [int(p >= 0.5) for p in y_prob]

    # 计算指标
    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except:
        auc = float("nan")

    # 打印类别级统计信息
    print("\n===== 类别预测统计 =====")
    total_covid = sum(1 for label in y_true if label == 1)
    total_non_covid = len(y_true) - total_covid
    pred_covid = sum(1 for pred in y_pred if pred == 1)
    pred_non_covid = len(y_pred) - pred_covid
    correct_covid = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    correct_non_covid = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    print(f"真实类别分布: covid={total_covid}, non-covid={total_non_covid}")
    print(f"预测类别分布: covid={pred_covid}, non-covid={pred_non_covid}")
    print(f"covid正确预测数: {correct_covid}/{total_covid}")
    print(f"non-covid正确预测数: {correct_non_covid}/{total_non_covid}")
    print(f"召回率 (covid识别率): {recall:.2f}%")  # 关键指标：covid被正确识别的比例


    # ---- 绘制混淆矩阵 ----
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["non-covid (0)", "covid (1)"],
        yticklabels=["non-covid (0)", "covid (1)"]
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵")
    cm_path = Path(log_dir) / "confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\n混淆矩阵已保存至: {cm_path}")
    plt.close()


    # 保存详细日志到文件
    log_path = Path(log_dir) / "evaluation_log.txt"
    with open(log_path, "w") as f:
        f.write("\n".join(detailed_log))
    print(f"详细评估日志已保存至: {log_path}")


    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist()
    }
