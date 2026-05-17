import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import lenet


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == targets).float().mean().item() * 100.0


def build_loaders(data_root: str, batch_size: int, num_workers: int):
    # Must match dataset.py:get_mnist transforms so inference uses the same input statistics.
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        acc = accuracy_top1(logits, labels)
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))

    return loss_meter.avg, acc_meter.avg


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, optimizer, criterion: nn.Module):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy_top1(logits.detach(), labels)
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(acc, labels.size(0))

    return loss_meter.avg, acc_meter.avg


def parse_args():
    p = argparse.ArgumentParser(description="Train LeNet on MNIST and save checkpoint for inference.py")
    p.add_argument("--data_root", default="./datasets/mnist/mnist-data",
                   help="Matches dataset.py:get_mnist path layout so inference can reuse the cache")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda", help="cuda|cpu")
    p.add_argument("--output", default="./models/lenet_mnist.pth",
                   help="quantize.py loads from <model_path><model>_<dataset>.pth")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available. Use --device cpu.")
    device = torch.device("cuda" if args.device == "cuda" else "cpu")

    train_loader, test_loader = build_loaders(args.data_root, args.batch_size, args.num_workers)

    model = lenet.lenet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    best_acc = -1.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, optimizer, criterion)
        te_loss, te_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}/{args.epochs} | lr {lr:.5f} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.2f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.2f}")

        if te_acc > best_acc:
            best_acc = te_acc
            cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, args.output)
            print(f"saved best checkpoint: {args.output} (acc={best_acc:.2f})")

    print(f"done in {time.time() - start:.1f}s, best_acc={best_acc:.2f}, output={args.output}")


if __name__ == "__main__":
    main()
