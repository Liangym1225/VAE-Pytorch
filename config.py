from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

@dataclass
class VAEConfig:
    
    output_dir: Path = Path("outputs")
    data: Optional[str] = "MNIST"
    experiment_name: Optional[str] = None
    project_name: Optional[str] = "VAE"
    timestamp: str = "{timestamp}"
    epochs: int = 40
    batch_size: int = 128
    distribution: str = "mse"
    learning_rate: float = 1e-3
    latent_dimension: int = 2
    logging: str = "None"

    def set_timestamp(self):
        if self.timestamp=="{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    def get_base_dir(self):
        return Path(f"{self.output_dir}/{self.data}/{self.timestamp}")
        
    def save_config(self):
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_path = base_dir / "config.yml"
        print(f"Saving config to:{config_path}")
        config_path.write_text(yaml.dump(self),"utf8")