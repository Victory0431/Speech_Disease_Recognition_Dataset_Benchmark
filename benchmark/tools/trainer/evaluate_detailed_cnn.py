# /mnt/data/test1/Speech_Disease_Recognition_Dataset_Benchmark/benchmark/tools/trainer/evaluate_detailed.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

def evaluate_model_detailed(model, data_loader, num_classes=2, class_names=None, verbose=False, device=None):
    """
    评估模型性能的详细函数，支持二分类和多分类任务
    
    参数:
        model: 训练好的模型
        data_loader: 数据加载器
        num_classes: 类别数量（默认2，支持多分类）
        class_names: 类别名称列表（可选，用于verbose输出）
        verbose: 是否打印详细评估结果
        device: 计算设备（CPU/GPU）
    """
    model.eval()
    y_pred = []
    y_true = []
    y_scores = []  # 用于计算AUC的概率值
    
    # 如果未指定设备，则使用模型所在设备
    if device is None:
        device = next(model.parameters()).device

    with torch.no_grad():
        for inputs, targets in data_loader:
            # 将输入移至正确设备
            inputs = inputs.to(device)
            outputs = model(inputs.float())  # 确保输入是float类型
            _, predicted = torch.max(outputs, 1)
            
            # 将结果移回CPU并转换为列表
            y_pred.extend(predicted.cpu().tolist())
            y_true.extend(targets.tolist())
            
            # 提取所有类别的概率作为分数
            y_scores.extend(torch.softmax(outputs, dim=1).cpu().tolist())

    # 计算整体指标
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # 混淆矩阵处理
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape[0] < num_classes or cm.shape[1] < num_classes:
        cm = np.pad(
            cm, 
            (
                (0, max(0, num_classes - cm.shape[0])), 
                (0, max(0, num_classes - cm.shape[1]))
            ), 
            mode='constant'
        )

    # 计算每个类别的召回率
    class_recall = []
    for i in range(num_classes):
        true_i = cm[i, :].sum()
        correct_i = cm[i, i]
        recall_i = correct_i / true_i if true_i > 0 else 0.0
        class_recall.append(recall_i)

    # 计算综合指标
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = class_recall[1]
        specificity = class_recall[0]
        f1_score_val = f1_score(y_true, y_pred)
    else:
        tn, fp, fn, tp = None, None, None, None
        sensitivity = None
        specificity = None
        f1_score_val = f1_score(y_true, y_pred, average='macro')

    # AUC计算
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, [score[1] for score in y_scores])
        else:
            auc = roc_auc_score(y_true, y_scores, multi_class='ovr')
    except:
        auc = 0.0

    # 计算每个类别的样本量
    actual_counts = [np.sum(np.array(y_true) == i) for i in range(num_classes)]
    predicted_counts = [np.sum(np.array(y_pred) == i) for i in range(num_classes)]
    total_samples = len(y_true)

    # 准备类别名称
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 二分类特有指标
    actual_healthy = actual_counts[0] if num_classes == 2 else None
    actual_patients = actual_counts[1] if num_classes == 2 else None
    predicted_healthy = predicted_counts[0] if num_classes == 2 else None
    predicted_patients = predicted_counts[1] if num_classes == 2 else None

    if verbose:
        print(f"准确率: {accuracy:.2f}%")
        if num_classes == 2:
            print(f"Sensitivity (Recall for {class_names[1]}): {sensitivity:.4f}")
            print(f"Specificity (Recall for {class_names[0]}): {specificity:.4f}")
        else:
            print(f"各类别召回率:")
            for i in range(num_classes):
                print(f"  {class_names[i]}: {class_recall[i]:.4f}")
            print(f"平均召回率 (macro): {np.mean(class_recall):.4f}")
        print(f"F1分数: {f1_score_val:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"混淆矩阵: \n{cm}")
        print("实际样本数:")
        for i in range(num_classes):
            print(f"  {class_names[i]}: {actual_counts[i]}")
        print("预测样本数:")
        for i in range(num_classes):
            print(f"  {class_names[i]}: {predicted_counts[i]}")

    return {
        "accuracy": accuracy,
        "f1_score": f1_score_val,
        "auc": auc,
        "confusion_matrix": cm,
        "class_recall": class_recall,
        "actual_counts": actual_counts,
        "predicted_counts": predicted_counts,
        "total_samples": total_samples,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "class_names": class_names,
        "actual_healthy": actual_healthy,
        "actual_patients": actual_patients,
        "predicted_healthy": predicted_healthy,
        "predicted_patients": predicted_patients
    }