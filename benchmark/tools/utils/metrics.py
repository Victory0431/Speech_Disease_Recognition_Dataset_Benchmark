import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score

def calculate_class_distribution(labels, class_names):
    """
    计算并返回数据集的类别分布统计
    
    参数:
        labels: 标签列表（整数形式，如[0, 1, 0, 2, ...]）
        class_names: 类别名称列表（与标签索引对应，如["class0", "class1", "class2"]）
    
    返回:
        字典，键为类别名称，值为元组(count, ratio)，其中：
        - count: 该类别的样本数量
        - ratio: 该类别占总样本的比例（百分比）
    """
    # 输入验证
    if len(labels) == 0:
        raise ValueError("标签列表不能为空")
    
    if len(class_names) == 0:
        raise ValueError("类别名称列表不能为空")
    
    # 转换为numpy数组便于统计操作
    labels_np = np.array(labels)
    total_samples = len(labels_np)
    
    # 计算每个类别的数量和比例
    distribution = {}
    for class_idx, class_name in enumerate(class_names):
        # 统计该类别的样本数量
        class_count = np.sum(labels_np == class_idx)
        # 计算占比（百分比）
        class_ratio = (class_count / total_samples) * 100 if total_samples > 0 else 0.0
        # 保存结果
        distribution[class_name] = (class_count, class_ratio)
    
    return distribution

def calculate_basic_metrics(y_true, y_pred, y_probs=None, num_classes=2):
    """
    计算基本分类指标（准确率、F1分数）
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率（用于计算AUC）
        num_classes: 类别数量
    
    返回:
        包含基本指标的字典
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred) * 100
    }
    
    # 计算F1分数
    if num_classes == 2:
        metrics["f1_score"] = f1_score(y_true, y_pred)
    else:
        metrics["f1_score"] = f1_score(y_true, y_pred, average='macro')
    
    # 计算AUC（如果提供了概率）
    if y_probs is not None:
        try:
            if num_classes == 2:
                metrics["auc"] = roc_auc_score(y_true, [prob[1] for prob in y_probs])
            else:
                metrics["auc"] = roc_auc_score(y_true, y_probs, multi_class='ovr')
        except:
            metrics["auc"] = 0.0  # 处理异常情况
    
    return metrics

def calculate_confusion_matrix(y_true, y_pred, num_classes):
    """
    计算混淆矩阵并确保维度正确
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        num_classes: 类别数量
    
    返回:
        规范化的混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 确保混淆矩阵维度与类别数量一致
    if cm.shape[0] < num_classes or cm.shape[1] < num_classes:
        cm = np.pad(
            cm, 
            (
                (0, max(0, num_classes - cm.shape[0])), 
                (0, max(0, num_classes - cm.shape[1]))
            ), 
            mode='constant'
        )
    
    return cm
    