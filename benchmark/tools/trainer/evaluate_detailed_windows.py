# save as: trainer/evaluate_detailed_windows.py
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)


def evaluate_model_detailed_windows(model, loader, aggregate="mean"):
    """
    逐文件评估：
    - loader 必须 batch_size=1（每次一个文件）
    - Dataset 返回 (segments, label, file_id)
    - 每个文件可能有多个窗口 (segments)
    - 聚合方式：mean / max
    """
    model.eval()
    file_preds, file_labels = {}, {}

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 3:
                raise ValueError("Dataset must return (segments, label, file_id) in eval mode")

            segments, label, file_id = batch
            segments = segments.squeeze(0).to(model.device)   # [num_windows, T]
            label = label.item()
            file_id = file_id if isinstance(file_id, str) else file_id[0]

            logits_list = []
            for seg in segments:  # 遍历每个窗口
                seg = seg.unsqueeze(0)  # [1, T]
                logits, _ = model(seg)  # [1, num_classes]
                logits_list.append(logits)

            logits = torch.cat(logits_list, dim=0)  # [num_windows, num_classes]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            # ---- 文件级聚合 ----
            if aggregate == "mean":
                file_prob = probs.mean(axis=0)
            elif aggregate == "max":
                file_prob = probs.max(axis=0)
            else:
                raise ValueError(f"Unknown aggregate {aggregate}")

            file_preds[file_id] = file_prob
            file_labels[file_id] = label

    # ---- 汇总全局结果 ----
    y_true = [file_labels[f] for f in file_labels.keys()]
    y_prob = [file_preds[f][1] for f in file_preds.keys()]  # 取类别=1 的概率
    y_pred = [int(p >= 0.5) for p in y_prob]

    acc = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, zero_division=0) * 100
    recall = recall_score(y_true, y_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_true, y_prob) * 100
    except:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }
