import argparse
import json
import yaml
import torch
import torch.nn.functional as F
from data import get_dataloaders
from teacher_model import build_teacher, evaluate
from student_model import build_student
from search_space import ArchConfig
from utils import setup_logging, set_seed, save_checkpoint, load_checkpoint, count_params, append_json_log

log = setup_logging(__name__)


def distillation_loss(student_logits, teacher_logits, labels, temperature: float, alpha: float):
    """
    alpha weights the distillation (soft) term; (1 - alpha) weights the
    standard hard-label cross-entropy term. The T^2 scaling on the KD term
    is the correction Hinton et al. specify to keep gradient magnitudes
    comparable between the two loss terms as temperature changes.
    """
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_distillation(cfg: dict):
    set_seed(cfg["search"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_dataloaders(cfg)
    teacher = build_teacher(cfg["data"]["num_classes"]).to(device)
    load_checkpoint(teacher, cfg["teacher"]["checkpoint"], device=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    with open(cfg["search"]["best_arch_path"]) as f:
        arch = ArchConfig.from_dict(json.load(f))
    student = build_student(arch, cfg["data"]["num_classes"]).to(device)
    log.info("Student params: %.2fM", count_params(student) / 1e6)
    dcfg = cfg["distillation"]
    optimizer = torch.optim.SGD(student.parameters(),lr=float(dcfg["lr"]),momentum=0.9,weight_decay=float(dcfg["weight_decay"]),nesterov=True,)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=dcfg["epochs"])
    best_acc = 0.0
    for epoch in range(dcfg["epochs"]):
        student.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)
            loss = distillation_loss(student_logits, teacher_logits, labels,dcfg["temperature"], dcfg["alpha"],)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        acc = evaluate(student, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        log.info("Epoch %d/%d | loss=%.4f | test_acc=%.4f", epoch + 1, dcfg["epochs"], avg_loss, acc)
        append_json_log("results/distillation_curve.json",{"epoch": epoch, "loss": avg_loss, "test_acc": acc})
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(student, dcfg["checkpoint"], extra={"acc": acc, "epoch": epoch})
    log.info("Best distilled student accuracy: %.4f", best_acc)
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train_distillation(cfg)

#Knowledge distillation training: student learns from a weighted combination of (a) the true hard-label cross-entropy loss and (b) a KL-divergence loss between temperature-softened teacher and student output distributions.