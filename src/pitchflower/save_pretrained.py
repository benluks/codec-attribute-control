import json
import argparse
from pathlib import Path

import torch
from pitchflower.pitchflower import PitchFlower


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save PitchFlower model components from a checkpoint."
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the checkpoint."
    )
    parser.add_argument(
        "target",
        type=str,
        help="Path to the target directory."
    )

    args = parser.parse_args()
    ckpt_path = Path(args.checkpoint)
    target_path = Path(args.target)

    target_path.mkdir(parents=True, exist_ok=True)

    pitchflower = PitchFlower.load_from_checkpoint(ckpt_path, map_location='cpu')

    torch.save(pitchflower.autoencoder.state_dict(), target_path/'autoencoder.pt')
    torch.save(pitchflower.flow_matcher.state_dict(), target_path/'flow_matcher.pt')

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = ckpt.get("hyper_parameters", {})

    config_new = {
        "autoencoder": config["autoencoder"]["init_args"],
        "flow_matcher": config["flow_matcher"]["init_args"],
        "mel_transform": config["mel_transform"],
    }

    with open(target_path / 'config.json', 'w') as f:
        json.dump(config_new, f, indent=4)

    print(f"All files saved to {target_path}")


if __name__ == "__main__":
    main()
