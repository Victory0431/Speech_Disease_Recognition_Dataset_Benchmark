import torch
import numpy as np
from tqdm import tqdm
from trainer.evaluate_model_detailed_time_moe_v2 import evaluate_model_detailed_time_moe

def train_and_evaluate_time_moe(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    """
    Time-MoE模型的训练和评估主函数，支持二分类和多分类
    
    参数:
        model: TimeMoEClassifier模型实例
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
        test_loader: 测试集DataLoader
        criterion: 损失函数
        optimizer: 优化器
        config: 配置类实例
    """
    # 初始化指标存储列表
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    test_metrics_per_epoch = []  # 存储每个epoch的测试集详细指标

    # 最佳验证指标跟踪（用于早停机制）
    best_val_f1 = 0.0
    best_model_weights = None

    for epoch in range(config.NUM_EPOCHS):
        # ====================== 训练阶段 ======================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [训练]")
        for inputs, targets in train_pbar:
            # 移动数据到设备
            inputs = inputs.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            
            # 前向传播与优化
            optimizer.zero_grad()
            logits, _ = model(inputs)  # Time-MoE返回(logits, hidden)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            # 累计损失和计算准确率
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # 更新进度条
            train_pbar.set_postfix({"batch_loss": loss.item()})

        # 计算训练集指标
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)


        # ====================== 验证阶段 ======================
        val_metrics = evaluate_model_detailed_time_moe(
            model, 
            val_loader,
            num_classes=config.NUM_CLASSES, 
            class_names=config.CLASS_NAMES, 
            verbose=False
        )
        
        val_loss = val_metrics['loss']
        val_acc = val_metrics['accuracy']
        val_f1 = val_metrics['f1_score']
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)


        # ====================== 测试阶段（每个epoch评估） ======================
        test_metrics = evaluate_model_detailed_time_moe(
            model, 
            test_loader,
            num_classes=config.NUM_CLASSES, 
            class_names=config.CLASS_NAMES, 
            verbose=False
        )
        test_metrics_per_epoch.append(test_metrics)
        test_accuracies.append(test_metrics['accuracy'])


        # ====================== 打印epoch信息 ======================
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
        print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%, F1={val_f1:.4f}")
        print(f"测试: 准确率={test_metrics['accuracy']:.2f}%, F1={test_metrics['f1_score']:.4f}, AUC={test_metrics['auc']:.4f}")
        print("-" * 80)

        # 保存最佳模型权重（基于验证集F1）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_weights = model.state_dict()
            print(f"📌 最佳模型更新 (验证F1: {best_val_f1:.4f})")


    # ====================== 最终评估（使用最佳模型） ======================
    print("\n🏁 训练完成，使用最佳模型进行最终测试评估...")
    model.load_state_dict(best_model_weights)
    
    print("\n最终验证集评估:")
    final_val_metrics = evaluate_model_detailed_time_moe(
        model, 
        val_loader,
        num_classes=config.NUM_CLASSES, 
        class_names=config.CLASS_NAMES, 
        verbose=True
    )
    
    print("\n最终测试集评估:")
    final_test_metrics = evaluate_model_detailed_time_moe(
        model, 
        test_loader,
        num_classes=config.NUM_CLASSES, 
        class_names=config.CLASS_NAMES, 
        verbose=True
    )

    # 返回所有指标
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracies": test_accuracies,
        "test_metrics_per_epoch": test_metrics_per_epoch,
        "final_val_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "class_names": config.CLASS_NAMES,
        "best_val_f1": best_val_f1
    }
