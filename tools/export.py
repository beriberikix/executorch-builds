"""Export a model described by models/<name>.yaml to an ExecuTorch program.

  python tools/export.py models/qwen3-0-6b-xnnpack.yaml [--out dist/models]

Writes <out>/<name>/ with model.pte, the tokenizer files the server needs,
and manifest.json: the source model and revision, tool versions, export
settings, runtime settings and file checksums. tools/validate.py reads the
manifest to build its quantized reference and its threshold.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "generation_config.json", "config.json",
]


def optimum_command(spec: dict, out: Path) -> list[str]:
    export = spec["export"]
    cmd = [
        "optimum-cli", "export", "executorch",
        "--model", spec["source"],
        "--task", "text-generation",
        "--recipe", export["recipe"],
        "--max_seq_len", str(export["max_seq_len"]),
        "--output_dir", str(out),
    ]
    if export.get("use_custom_sdpa"):
        cmd.append("--use_custom_sdpa")
    if export.get("use_custom_kv_cache"):
        cmd.append("--use_custom_kv_cache")
    for flag in ("qlinear", "qembedding"):
        if export.get(flag):
            cmd += [f"--{flag}", export[flag]]
    return cmd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, default=Path("dist/models"))
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    if spec["export"]["tool"] != "optimum-executorch":
        print(f"unsupported export tool: {spec['export']['tool']}", file=sys.stderr)
        return 1

    revision = HfApi().model_info(spec["source"]).sha
    out = args.out / spec["name"]
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        subprocess.run(optimum_command(spec, Path(tmp)), check=True)
        shutil.move(Path(tmp) / "model.pte", out / "model.pte")

    for name in TOKENIZER_FILES:
        shutil.copyfile(hf_hub_download(spec["source"], name, revision=revision), out / name)

    files = ["model.pte", *TOKENIZER_FILES]
    manifest = {
        "name": spec["name"],
        "source": spec["source"],
        "source_revision": revision,
        "backend": spec["backend"],
        "export": spec["export"],
        "runtime": spec.get("runtime", {}),
        "gate": spec.get("gate", {}),
        "tools": {pkg: version(pkg) for pkg in ("executorch", "optimum-executorch", "torch", "torchao", "transformers")},
        "files": {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size} for name in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {spec['source']}@{revision[:12]} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
