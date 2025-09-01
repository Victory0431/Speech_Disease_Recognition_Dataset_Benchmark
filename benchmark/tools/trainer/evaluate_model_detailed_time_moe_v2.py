import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm

def aggregate_window_predictions(window_logits, aggregation_strategy="mean"):
    """
    聚合一个音频文件的所有窗口预测结果
    
    参数:
        window_logits: 所有窗口的logits，shape: [num_windows, num_classes]
        aggregation_strategy: 聚合策略 ("mean" 或 "majority_vote")
    返回:
        最终预测类别和概率
    """
    # 转换为概率
    window_probs = torch.softmax(torch.tensor(window_logits), dim=1).numpy()
    
    if aggregation_strategy == "mean":
        # 平均概率策略
        avg_probs = np.mean(window_probs, axis=0)
        predicted_class = np.argmax(avg_probs)
        return predicted_class, avg_probs
    elif aggregation_strategy == "majority_vote":
        # 多数投票策略
        window_preds = np.argmax(window_probs, axis=1)
        predicted_class = np.bincount(window_preds).argmax()
        # 计算多数类的概率
        majority_probs = np.mean(window_probs[window_preds == predicted_class], axis=0)
        return predicted_class, majority_probs
    else:
        raise ValueError(f"不支持的聚合策略: {aggregation_strategy}")


def evaluate_model_detailed_time_moe(model, data_loader, num_classes=2, class_names=None, 
                                    verbose=False, aggregation_strategy="mean"):
    """
    评估Time-MoE模型性能的详细函数，支持二分类和多分类，处理时序窗口聚合
    
    参数:
        model: 训练好的Time-MoE模型
        data_loader: 数据加载器（验证集或测试集）
        num_classes: 类别数量
        class_names: 类别名称列表
        verbose: 是否打印详细结果
        aggregation_strategy: 窗口聚合策略 ("mean" 或 "majority_vote")
    """
    model.eval()
    
    # 存储每个音频文件的最终预测结果
    all_y_true = []
    all_y_pred = []
    all_y_probs = []  # 用于计算AUC的概率
    total_loss = 0.0
    
    # 用于跟踪同一文件的窗口结果
    current_file_id = None
    current_window_logits = []
    current_label = None

    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc="评估中")
        for batch in eval_pbar:
            # Time-MoE的数据加载器返回 (windows, label, file_path)
            windows, label, file_path = batch
            file_id = file_path[0]  # 每个batch是单个文件的所有窗口
            
            # 将数据移至设备
            windows = windows.squeeze(0).to(model.device)  # [1, num_windows, window_size] → [num_windows, window_size]
            label = label.to(model.device)
            
            # 前向传播
            logits, _ = model(windows)  # [num_windows, num_classes]
            loss = torch.nn.functional.cross_entropy(logits, label.repeat(logits.shape[0]))
            total_loss += loss.item() * logits.shape[0]  # 累计损失
            
            # 处理当前文件的所有窗口
            if file_id != current_file_id and current_file_id is not None:
                # 聚合上一个文件的所有窗口结果
                pred_class, pred_probs = aggregate_window_predictions(
                    current_window_logits, 
                    aggregation_strategy
                )
                
                # 保存结果
                all_y_true.append(current_label)
                all_y_pred.append(pred_class)
                all_y_probs.append(pred_probs)
                
                # 重置跟踪变量
                current_window_logits = []
            
            # 累加当前文件的窗口结果
            current_file_id = file_id
            current_window_logits.extend(logits.cpu().numpy())
            current_label = label.item()
        
        # 处理最后一个文件
        if current_file_id is not None:
            pred_class, pred_probs = aggregate_window_predictions(
                current_window_logits, 
                aggregation_strategy
            )
            all_y_true.append(current_label)
            all_y_pred.append(pred_class)
            all_y_probs.append(pred_probs)
    
    # 计算整体指标
    accuracy = accuracy_score(all_y_true, all_y_pred) * 100
    avg_loss = total_loss / len(all_y_true)  # 按文件数平均损失
    
    # 混淆矩阵处理
    cm = confusion_matrix(all_y_true, all_y_pred)
    # 确保混淆矩阵维度正确
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

    # 计算F1分数（区分二分类和多分类）
    if num_classes == 2:
        f1_score_val = f1_score(all_y_true, all_y_pred)
    else:
        f1_score_val = f1_score(all_y_true, all_y_pred, average='macro')

    # 计算AUC（区分二分类和多分类）
    try:
        if num_classes == 2:
            # 二分类使用正类概率
            auc = roc_auc_score(all_y_true, [prob[1] for prob in all_y_probs])
        else:
            # 多分类使用one-vs-rest策略
            auc = roc_auc_score(all_y_true, all_y_probs, multi_class='ovr')
    except:
        auc = 0.0  # 处理只有一类或其他异常情况

    # 计算每个类别的样本量
    actual_counts = [np.sum(np.array(all_y_true) == i) for i in range(num_classes)]
    predicted_counts = [np.sum(np.array(all_y_pred) == i) for i in range(num_classes)]
    total_samples = len(all_y_true)

    # 准备类别名称
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 二分类特有指标
    tn, fp, fn, tp = None, None, None, None
    sensitivity, specificity = None, None
    actual_healthy, actual_patients = None, None
    predicted_healthy, predicted_patients = None, None

    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = class_recall[1]  # 正类召回率
        specificity = class_recall[0]  # 负类召回率（特异性）
        actual_healthy = actual_counts[0]
        actual_patients = actual_counts[1]
        predicted_healthy = predicted_counts[0]
        predicted_patients = predicted_counts[1]

    # 打印详细信息
    if verbose:
        print(f"评估样本数: {total_samples} 个音频文件")
        print(f"损失: {avg_loss:.4f}")
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
        
        print("\n实际样本数:")
        for i in range(num_classes):
            print(f"  {class_names[i]}: {actual_counts[i]}")
            
        print("预测样本数:")
        for i in range(num_classes):
            print(f"  {class_names[i]}: {predicted_counts[i]}")

    # 返回所有指标
    return {
        # 通用指标
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1_score_val,
        "auc": auc,
        "confusion_matrix": cm,
        "class_recall": class_recall,
        "actual_counts": actual_counts,
        "predicted_counts": predicted_counts,
        "total_samples": total_samples,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_probs": all_y_probs,
        
        # 二分类特有指标
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
