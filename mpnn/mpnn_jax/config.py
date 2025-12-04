from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class GraphType(Enum):
    BARABASI_ALBERT = "ba"
    ERDOS_RENYI = "er"
    SNDLIB = "snd"


@dataclass(frozen=True)
class NormConfig:
    mu_shift: float
    mu_scale: float
    W_shift: float
    W_scale: float


NORM_PRESETS: Dict[str, NormConfig] = {
    "ba": NormConfig(mu_shift=0.34,  mu_scale=0.27,  W_shift=55.3, W_scale=22.0),
    "er":  NormConfig(mu_shift=0.199, mu_scale=0.12,  W_shift=69.3, W_scale=15.95)
}


def get_norm(profile: str) -> NormConfig:
    return NORM_PRESETS[profile]


@dataclass
class DatasetConfig:
    output_dir: str = "dataset/er"
    seed: int = 420
    n_nodes: int = 40
    train_size: int = 20000
    val_size: int = 20
    test_size: int = 2000
    rl: float = 0.3
    rh: float = 0.9
    graph_type: GraphType = GraphType.ERDOS_RENYI
    snd_path: list[str] = field(default_factory=lambda: ["sndlib-networks-xml/germany50.xml"])


@dataclass
class ModelConfig:
    hidden_dim: int = 8
    num_passes: int = 4
    N_H: int = 14
    rn: int = 8


@dataclass
class TrainingConfig:
    train_dataset_path: str = "dataset/BarabasiAlbert/train.npz"
    val_dataset_path: str = "dataset/BarabasiAlbert/val.npz"
    output_path: str = "log"
    batch_size: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    steps: int = 20000
    log_interval: int = 200
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    norm_profile: str = "ba"
    

@dataclass
class TestConfig:
    test_dataset_path: str = "dataset/BarabasiAlbert/test.npz"
    checkpoint_path: str = "log"
    output_path: str = "results"
    norm_profile: str = "ba"
    
