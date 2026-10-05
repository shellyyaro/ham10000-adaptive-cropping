from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resnet_vit_pretrained_comparison.scripts.evaluate_report_checkpoints import main


if __name__ == "__main__":
    main()
