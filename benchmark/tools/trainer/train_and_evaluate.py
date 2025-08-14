import torch
from trainer.evaluate_detailed import evaluate_model_detailed

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    test_metrics_per_epoch = []  # 存储每个epoch的测试集详细指标

    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # 计算训练指标
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 测试集指标（每个epoch都评估）
        test_metrics = evaluate_model_detailed(
            model, 
            test_loader,
            num_classes=len(config.CLASS_NAMES), 
            class_names=config.CLASS_NAMES, 
            verbose=False
        )
        test_metrics_per_epoch.append(test_metrics)
        test_accuracies.append(test_metrics['accuracy'])

        # 打印 epoch 信息
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"训练: 损失={train_loss:.4f}, 准确率={train_acc:.2f}%")
        print(f"验证: 损失={val_loss:.4f}, 准确率={val_acc:.2f}%")
        print(f"测试: 准确率={test_metrics['accuracy']:.2f}%, F1分数={test_metrics['f1_score']:.4f}, AUC={test_metrics['auc']:.4f}")
        print("----------------------------------------")

    # 最终测试评估（详细输出）
    print("\n最终测试集评估:")
    final_test_metrics = evaluate_model_detailed(
        model, 
        test_loader,
        num_classes=len(config.CLASS_NAMES), 
        class_names=config.CLASS_NAMES, 
        verbose=True
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "test_accuracies": test_accuracies,
        "test_metrics_per_epoch": test_metrics_per_epoch,
        "final_test_metrics": final_test_metrics,
        "class_names": config.CLASS_NAMES
    }