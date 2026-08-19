import argparse
import yaml
import torch
import torch.nn as nn
import torchvision.models as tvm
from data import get_dataloaders
from utils import setup_logging, set_seed, save_checkpoint, count_params
log = setup_logging(__name__)


def build_teacher(num_classes: int) -> nn.Module:
    """ResNet50 adapted for CIFAR's 32x32 input: replace the first 7x7/stride-2
    conv + maxpool (designed for 224x224 ImageNet) with a 3x3/stride-1 conv,
    otherwise CIFAR images get downsampled to near-nothing before the first residual block even runs"""
    model = tvm.resnet50(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train_teacher(cfg: dict):
    set_seed(cfg["search"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Training teacher on device: %s", device)
    train_loader, test_loader = get_dataloaders(cfg)
    model = build_teacher(cfg["data"]["num_classes"]).to(device)
    log.info("Teacher params: %.2fM", count_params(model) / 1e6)
    tcfg = cfg["teacher"]
    optimizer = torch.optim.SGD(model.parameters(),lr=float(tcfg["lr"]), momentum=float(tcfg["momentum"]),weight_decay=float(tcfg["weight_decay"]),nesterov=True,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"])
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(tcfg["epochs"]):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        acc = evaluate(model, test_loader, device)
        log.info("Epoch %d/%d | loss=%.4f | test_acc=%.4f",epoch + 1, tcfg["epochs"], total_loss / len(train_loader), acc)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, tcfg["checkpoint"], extra={"acc": acc, "epoch": epoch})

    log.info("Best teacher test accuracy: %.4f", best_acc)
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_teacher(cfg)