import random
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class StageConfig:
    num_blocks: int # depth of this stage
    out_channels: int  # base output channels (before width multiplier)
    expansion_ratio: int # inverted residual expansion factor
    kernel_size: int  # 3 or 5
    stride: int  # stride of the first block in the stage (downsampling)


@dataclass
class ArchConfig:
    stages: List[StageConfig]
    width_mult: float
    stem_channels: int

    def to_dict(self):
        return {"stages": [asdict(s) for s in self.stages],"width_mult": self.width_mult,"stem_channels": self.stem_channels,}

    @staticmethod
    def from_dict(d):
        stages = [StageConfig(**s) for s in d["stages"]]
        return ArchConfig(stages=stages, width_mult=d["width_mult"], stem_channels=d["stem_channels"])


NUM_STAGES = 4
DEPTH_CHOICES = [1, 2, 3]
CHANNEL_CHOICES = [16, 24, 32, 48, 64]
EXPANSION_CHOICES = [2, 3, 4, 6]
KERNEL_CHOICES = [3, 5]
WIDTH_MULT_CHOICES = [0.75, 1.0, 1.25]
STEM_CHANNEL_CHOICES = [16, 24, 32]
STAGE_STRIDES = [1, 2, 2, 2]  


def sample_random_arch(rng: random.Random) -> ArchConfig:
    stages = []
    for stage_idx in range(NUM_STAGES):
        stages.append(StageConfig(num_blocks=rng.choice(DEPTH_CHOICES),out_channels=rng.choice(CHANNEL_CHOICES),expansion_ratio=rng.choice(EXPANSION_CHOICES),kernel_size=rng.choice(KERNEL_CHOICES),stride=STAGE_STRIDES[stage_idx],))
    return ArchConfig(stages=stages,width_mult=rng.choice(WIDTH_MULT_CHOICES),stem_channels=rng.choice(STEM_CHANNEL_CHOICES),)


def mutate_arch(arch: ArchConfig, rng: random.Random) -> ArchConfig:
    import copy
    new_arch = copy.deepcopy(arch)
    mutation_type = rng.choice(["depth", "channels", "expansion", "kernel", "width_mult", "stem"])
    if mutation_type == "depth":
        stage = rng.choice(new_arch.stages)
        stage.num_blocks = rng.choice(DEPTH_CHOICES)
    elif mutation_type == "channels":
        stage = rng.choice(new_arch.stages)
        stage.out_channels = rng.choice(CHANNEL_CHOICES)
    elif mutation_type == "expansion":
        stage = rng.choice(new_arch.stages)
        stage.expansion_ratio = rng.choice(EXPANSION_CHOICES)
    elif mutation_type == "kernel":
        stage = rng.choice(new_arch.stages)
        stage.kernel_size = rng.choice(KERNEL_CHOICES)
    elif mutation_type == "width_mult":
        new_arch.width_mult = rng.choice(WIDTH_MULT_CHOICES)
    elif mutation_type == "stem":
        new_arch.stem_channels = rng.choice(STEM_CHANNEL_CHOICES)
    return new_arch