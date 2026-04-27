"""
步骤 3：训练宠物品种识别模型（EfficientNet-B4，迁移学习）
Apple M 芯片使用 MPS 加速

支持断点续训：
  正常启动（从头）：  python 03_train.py
  从阶段二恢复：      python 03_train.py --resume-phase2
  从指定 epoch 恢复：  python 03_train.py --resume-phase2 --start-epoch 8

完成后运行：python 04_export_onnx.py
"""

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# ── 路径 ──────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "dataset"
MODEL_DIR     = BASE_DIR / "models"
CKPT_BEST     = MODEL_DIR / "pet_classifier_best.pth"
CKPT_RESUME   = MODEL_DIR / "pet_classifier_resume.pth"   # 每轮完整快照
META_FILE     = MODEL_DIR / "class_meta.json"

# ── 超参数 ────────────────────────────────────────────
MODEL_NAME   = "efficientnet_b4"
IMG_SIZE     = 380
BATCH_SIZE_P1 = 16    # 阶段一：只训分类头，内存压力小
BATCH_SIZE_P2 = 8     # 阶段二：全网微调，380×380 显存压力大，降到 8
PHASE1_LR    = 1e-3
PHASE2_LR    = 5e-5
PHASE1_EPOCH = 5
TOTAL_EPOCH  = 25
LABEL_SMOOTH = 0.1
NUM_WORKERS  = 0

# ── 命令行参数解析 ────────────────────────────────────
RESUME_PHASE2 = "--resume-phase2" in sys.argv
START_EPOCH   = PHASE1_EPOCH + 1   # 默认从阶段二第一轮开始
for i, arg in enumerate(sys.argv):
    if arg == "--start-epoch" and i + 1 < len(sys.argv):
        START_EPOCH = int(sys.argv[i + 1])

# ── 设备 ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ 使用 Apple MPS 加速")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("✓ 使用 CUDA 加速")
else:
    DEVICE = torch.device("cpu")
    print("! 使用 CPU（较慢）")

# ── 数据增强 ──────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_loaders(batch_size: int):
    train_ds = datasets.ImageFolder(str(DATA_DIR / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(DATA_DIR / "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(str(DATA_DIR / "test"),  transform=val_tf)
    kw = dict(num_workers=NUM_WORKERS, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        len(train_ds.classes),
        train_ds.classes,
    )


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        bar = tqdm(loader, desc="train" if training else "eval ", leave=False, ncols=88)
        for imgs, labels in bar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


def save_resume_ckpt(epoch, model, optimizer, scheduler, best_val_acc):
    """保存完整快照（用于断点恢复）"""
    torch.save({
        "epoch":        epoch,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "best_val_acc": best_val_acc,
    }, CKPT_RESUME)


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    # ────────────────────────────────────────────────
    # 模式：从阶段二断点恢复
    # ────────────────────────────────────────────────
    if RESUME_PHASE2:
        print(f"\n{'='*55}")
        print(f"断点续训：从 Epoch {START_EPOCH} 恢复阶段二")
        print(f"{'='*55}")

        # 用阶段二的 batch size 重建 loader
        train_ld, val_ld, test_ld, num_classes, _ = build_loaders(BATCH_SIZE_P2)
        print(f"类别数：{num_classes} | Batch Size（阶段二）：{BATCH_SIZE_P2}")

        # 优先从完整快照恢复，否则从最佳模型加载权重
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

        # 解冻所有参数（阶段二）
        for p in model.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)
        remaining = TOTAL_EPOCH - PHASE1_EPOCH
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining, eta_min=PHASE2_LR * 0.01
        )

        if CKPT_RESUME.exists():
            ckpt = torch.load(CKPT_RESUME, map_location=DEVICE)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_val_acc = ckpt["best_val_acc"]
            print(f"  ✓ 从完整快照恢复（Epoch {ckpt['epoch']}，best_val={best_val_acc:.4f}）")
        elif CKPT_BEST.exists():
            model.load_state_dict(torch.load(CKPT_BEST, map_location=DEVICE))
            best_val_acc = 0.9201   # Epoch 5 记录的值，手动填入
            print(f"  ✓ 从最佳模型权重加载（val_acc={best_val_acc:.4f}）")
            print(f"  ! 优化器/调度器状态未恢复，lr 从 {PHASE2_LR} 重新开始")
        else:
            print("[错误] 未找到任何可用的检查点，请先完成阶段一训练")
            return

        # 推进调度器到正确位置（跳过已完成的 epoch）
        already_done = START_EPOCH - (PHASE1_EPOCH + 1)
        for _ in range(already_done):
            scheduler.step()

        print(f"\n开始训练 Epoch {START_EPOCH} → {TOTAL_EPOCH}")

        for epoch in range(START_EPOCH, TOTAL_EPOCH + 1):
            t0 = time.time()
            tr_loss, tr_acc   = run_epoch(model, train_ld, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, val_ld,   criterion)
            scheduler.step()
            elapsed = time.time() - t0

            print(f"Epoch {epoch:02d}/{TOTAL_EPOCH} | "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"{elapsed:.0f}s")

            # 每轮保存完整快照（下次可精确恢复）
            save_resume_ckpt(epoch, model, optimizer, scheduler, best_val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), CKPT_BEST)
                print(f"  ✓ 保存最佳模型 val_acc={val_acc:.4f}")

        # 最终测试
        print(f"\n{'='*55}\n最终评估（测试集）\n{'='*55}")
        model.load_state_dict(torch.load(CKPT_BEST, map_location=DEVICE))
        _, test_acc = run_epoch(model, test_ld, criterion)
        print(f"\n最终 Test Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
        print(f"最佳 Val  Accuracy: {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
        print(f"\n模型已保存至：{CKPT_BEST}")
        print("下一步：运行 python 04_export_onnx.py")
        return

    # ────────────────────────────────────────────────
    # 正常模式：从头训练
    # ────────────────────────────────────────────────
    train_ld, val_ld, test_ld, num_classes, class_names = build_loaders(BATCH_SIZE_P1)
    print(f"类别数：{num_classes} | Batch Size（阶段一）：{BATCH_SIZE_P1}")

    model     = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # ── 阶段一：只训分类头 ────────────────────────────
    print(f"\n{'='*55}")
    print(f"阶段一（Epoch 1-{PHASE1_EPOCH}）：预热分类头")
    print(f"{'='*55}")
    for name, p in model.named_parameters():
        p.requires_grad = ("classifier" in name)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=PHASE1_LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PHASE1_EPOCH, eta_min=PHASE1_LR * 0.1
    )
    best_val_acc = 0.0

    for epoch in range(1, PHASE1_EPOCH + 1):
        t0 = time.time()
        tr_loss, tr_acc   = run_epoch(model, train_ld, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_ld,   criterion)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{TOTAL_EPOCH} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.0f}s")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_BEST)
            print(f"  ✓ 保存最佳模型 val_acc={val_acc:.4f}")

    # ── 阶段二：全网微调，切换到更小 batch ───────────
    print(f"\n{'='*55}")
    print(f"阶段二（Epoch {PHASE1_EPOCH+1}-{TOTAL_EPOCH}）：全网络微调")
    print(f"Batch Size 从 {BATCH_SIZE_P1} → {BATCH_SIZE_P2}（降低 MPS 内存压力）")
    print(f"{'='*55}")

    # 重建 DataLoader，使用更小的 batch
    train_ld, val_ld, test_ld, _, _ = build_loaders(BATCH_SIZE_P2)

    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=PHASE2_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TOTAL_EPOCH - PHASE1_EPOCH, eta_min=PHASE2_LR * 0.01
    )

    for epoch in range(PHASE1_EPOCH + 1, TOTAL_EPOCH + 1):
        t0 = time.time()
        tr_loss, tr_acc   = run_epoch(model, train_ld, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_ld,   criterion)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{TOTAL_EPOCH} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"{elapsed:.0f}s")

        # 每轮保存完整快照
        save_resume_ckpt(epoch, model, optimizer, scheduler, best_val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CKPT_BEST)
            print(f"  ✓ 保存最佳模型 val_acc={val_acc:.4f}")

    # ── 最终评估 ─────────────────────────────────────
    print(f"\n{'='*55}\n最终评估（测试集）\n{'='*55}")
    model.load_state_dict(torch.load(CKPT_BEST, map_location=DEVICE))
    _, test_acc = run_epoch(model, test_ld, criterion)
    print(f"\n最终 Test Accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)")
    print(f"最佳 Val  Accuracy: {best_val_acc:.4f}  ({best_val_acc*100:.1f}%)")
    print(f"\n模型已保存至：{CKPT_BEST}")
    print("下一步：运行 python 04_export_onnx.py")


if __name__ == "__main__":
    main()
