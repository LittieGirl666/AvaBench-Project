"""Download the two model revisions used in the mechanism experiments."""

import argparse
from pathlib import Path
from common import ROOT, MODELS, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", *MODELS], default="both")
    parser.add_argument("--directory", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    settings = read_json(ROOT / "stage1/models.json")
    for model in MODELS if args.model == "both" else [args.model]:
        spec = settings[model]
        snapshot_download(
            repo_id=spec["model_id"],
            revision=spec["revision"],
            local_dir=args.directory / spec["model_id"].split("/")[1],
        )


if __name__ == "__main__":
    main()
