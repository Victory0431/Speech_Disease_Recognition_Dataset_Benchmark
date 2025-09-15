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
    # 转换为概率（此处用torch.tensor后立即处理，避免累积大张量）
    window_probs = torch.softmax(torch.tensor(window_logits), dim=1).numpy()
    
    if aggregation_strategy == "mean":
        avg_probs = np.mean(window_probs, axis=0)
        predicted_class = np.argmax(avg_probs)
        # 清理聚合过程中的临时大数组
        del window_probs
        return predicted_class, avg_probs
    elif aggregation_strategy == "majority_vote":
        window_preds = np.argmax(window_probs, axis=1)
        predicted_class = np.bincount(window_preds).argmax()
        majority_probs = np.mean(window_probs[window_preds == predicted_class], axis=0)
        # 清理聚合过程中的临时大数组
        del window_probs, window_preds
        return predicted_class, majority_probs
    else:
        raise ValueError(f"不支持的聚合策略: {aggregation_strategy}")


def evaluate_model_detailed_time_moe(model, data_loader, num_classes=2, class_names=None, 
                                    verbose=False, aggregation_strategy="mean"):
    """
    评估Time-MoE模型性能的详细函数（已添加内存清理）
    支持二分类和多分类，处理时序窗口聚合，解决评估时内存累积问题
    """
    model.eval()
    
    # 存储最终结果（仅保留小变量，避免大张量累积）
    all_y_true = []
    all_y_pred = []
    all_y_probs = []  
    total_loss = 0.0
    
    # 跟踪同一文件的窗口结果（避免列表无限累积）
    current_file_id = None
    current_window_logits = []  # 存储当前文件的窗口logits（numpy数组）
    current_label = None

    # 适配DataParallel模型：获取实际设备（主卡）
    device = model.device if hasattr(model, 'device') else model.module.device

    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc="评估中")
        for batch in eval_pbar:
            # 1. 加载当前batch（单个文件的所有滑窗、标签、路径）
            windows, label, file_path = batch
            file_id = file_path[0]  # 每个batch对应1个音频文件
            
            # 2. 数据预处理：移至GPU，删除冗余维度（避免额外内存占用）
            # non_blocking=True：异步传输，提升效率且减少内存阻塞
            windows = windows.squeeze(0).to(device, non_blocking=True)  # [num_windows, window_size]
            label = label.to(device, non_blocking=True)
            
            # 3. 模型前向传播（仅保留必要结果，立即清理中间变量）
            logits, _ = model(windows)  # [num_windows, num_classes]
            # 计算损失（每个窗口对应同一标签，repeat后立即用，不保留）
            loss = torch.nn.functional.cross_entropy(
                logits, 
                label.repeat(logits.shape[0])  # 扩展标签匹配窗口数
            )
            total_loss += loss.item() * logits.shape[0]  # 累计损失（用标量，不占GPU内存）
            
            # 4. 保存当前窗口结果（转numpy后删除GPU上的logits）
            current_window_logits.extend(logits.cpu().numpy())  # 存CPU numpy数组，释放GPU
            current_file_id = file_id
            current_label = label.item()  # 存标量标签，释放GPU标签张量
            
            # -------------------------- 内存清理1：删除当前batch的GPU大张量 --------------------------
            del windows, logits, loss, label  # 删除GPU上的大张量
            torch.cuda.empty_cache()  # 强制清空GPU缓存（释放无用内存）
            # -----------------------------------------------------------------------------------
            
            # 5. 处理上一个文件（当文件ID变化时，聚合结果并清理）
            if file_id != current_file_id and current_file_id is not None:
                # 5.1 聚合上一个文件的所有窗口结果
                pred_class, pred_probs = aggregate_window_predictions(
                    current_window_logits, 
                    aggregation_strategy
                )
                
                # 5.2 保存最终结果（仅小变量）
                all_y_true.append(current_label)
                all_y_pred.append(pred_class)
                all_y_probs.append(pred_probs)
                
                # -------------------------- 内存清理2：删除上一个文件的窗口数据 --------------------------
                del current_window_logits, pred_probs  # 删除累积的numpy数组和聚合结果
                current_window_logits = []  # 重置列表，避免内存泄漏
                torch.cuda.empty_cache()
                # -----------------------------------------------------------------------------------
            
        # 6. 处理最后一个文件（循环结束后）
        if current_file_id is not None:
            pred_class, pred_probs = aggregate_window_predictions(
                current_window_logits, 
                aggregation_strategy
            )
            all_y_true.append(current_label)
            all_y_pred.append(pred_class)
            all_y_probs.append(pred_probs)
            
            # -------------------------- 内存清理3：删除最后一个文件的窗口数据 --------------------------
            del current_window_logits, pred_probs
            torch.cuda.empty_cache()
            # -----------------------------------------------------------------------------------
    
    # 7. 计算整体指标（仅用小列表，不占GPU内存）
    accuracy = accuracy_score(all_y_true, all_y_pred) * 100
    avg_loss = total_loss / len(all_y_true)  # 按文件数平均损失
    
    # 混淆矩阵处理（numpy数组，按需pad，不占GPU）
    cm = confusion_matrix(all_y_true, all_y_pred)
    if cm.shape[0] < num_classes or cm.shape[1] < num_classes:
        cm = np.pad(
            cm, 
            (
                (0, max(0, num_classes - cm.shape[0])), 
                (0, max(0, num_classes - cm.shape[1]))
            ), 
            mode='constant'
        )

    # 类别召回率（小列表计算）
    class_recall = []
    for i in range(num_classes):
        true_i = cm[i, :].sum()
        correct_i = cm[i, i]
        recall_i = correct_i / true_i if true_i > 0 else 0.0
        class_recall.append(recall_i)

    # F1分数（二分类/多分类适配）
    f1_score_val = f1_score(all_y_true, all_y_pred) if num_classes == 2 else \
                   f1_score(all_y_true, all_y_pred, average='macro')

    # AUC计算（二分类/多分类适配）
    auc = 0.0
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_y_true, [prob[1] for prob in all_y_probs])
        else:
            auc = roc_auc_score(all_y_true, all_y_probs, multi_class='ovr')
    except Exception as e:
        print(f"⚠️  AUC计算失败: {str(e)}，默认设为0.0")

    # 样本量统计（小列表计算）
    actual_counts = [np.sum(np.array(all_y_true) == i) for i in range(num_classes)]
    predicted_counts = [np.sum(np.array(all_y_pred) == i) for i in range(num_classes)]
    total_samples = len(all_y_true)

    # 类别名称默认值
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # 二分类特有指标（仅小变量）
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

    # 打印详细信息（可选）
    if verbose:
        print(f"\n===== 评估结果汇总 =====")
        print(f"评估样本数: {total_samples} 个音频文件")
        print(f"平均损失: {avg_loss:.4f}")
        print(f"准确率: {accuracy:.2f}%")
        
        if num_classes == 2:
            print(f"Sensitivity ({class_names[1]}): {sensitivity:.4f}")
            print(f"Specificity ({class_names[0]}): {specificity:.4f}")
            print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        else:
            print(f"各类别召回率:")
            for i in range(num_classes):
                print(f"  {class_names[i]}: {class_recall[i]:.4f}")
            print(f"平均召回率 (macro): {np.mean(class_recall):.4f}")
            
        print(f"F1分数: {f1_score_val:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"混淆矩阵: \n{cm}")

    # -------------------------- 内存清理4：删除指标计算中的临时大数组 --------------------------
    del cm, actual_counts, predicted_counts  # 删除numpy大数组
    torch.cuda.empty_cache()
    # -----------------------------------------------------------------------------------

    # 返回所有指标（仅小变量，无大张量）
    return {
        # 通用指标
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_score": f1_score_val,
        "auc": auc,
        "confusion_matrix": cm.copy() if 'cm' in locals() else None,  # 按需返回，避免大数组
        "class_recall": class_recall,
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