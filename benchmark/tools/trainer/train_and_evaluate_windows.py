# save as: trainer/train_and_evaluate_windows.py
import torch
from trainer.evaluate_detailed_windows import evaluate_model_detailed_windows


def train_and_evaluate_windows(model, train_loader, val_loader, test_loader, criterion, optimizer, config):
    for epoch in range(config.NUM_EPOCHS):
        # ---------- TRAIN ----------
        model.train()
        running_loss = 0.0
        for x, y in train_loader:  # train 返回的是 [wav, label]
            x, y = x.to(model.device), y.to(model.device).long()
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ---------- EVAL / TEST (文件级) ----------
        val_metrics = evaluate_model_detailed_windows(
            model, val_loader,
            aggregate="mean"  # 可选: "mean" | "max"
        )
        test_metrics = evaluate_model_detailed_windows(
            model, test_loader,
            aggregate="mean"
        )

        print(f"[Epoch {epoch+1}] "
              f"loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.2f}% "
              f"val_f1={val_metrics['f1']:.2f} "
              f"test_acc={test_metrics['accuracy']:.2f}% "
              f"test_f1={test_metrics['f1']:.2f}")
