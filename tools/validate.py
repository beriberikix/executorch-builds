"""Numerics gate: compare the shipping runtime's logits with eager PyTorch.

Renders each chat prompt with the model's own template, runs it through
logits-check (the C++ runtime, fed `--chunk` tokens per pass like the worker)
and through the Hugging Face model in eager PyTorch, and compares the top-k
next-token predictions at every position.

  python tools/validate.py --model-dir <dir with model.pte + tokenizer> \
      --reference Qwen/Qwen3-0.6B --logits-check <path> [--chunk 1]

The gate compares against the eager model quantized the same way as the
export (read from the model's manifest.json, or --qlinear/--qembedding), so it
measures what the runtime adds, not quantization error. Agreement with the
fp32 model is reported for information. Exits non-zero when top-1 agreement
with the quantized reference falls under --min-top1.
"""

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--reference", help="Hugging Face model id or path (default: the manifest's source)")
    parser.add_argument("--logits-check", required=True, type=Path)
    parser.add_argument("--prompts", type=Path, default=Path(__file__).with_name("prompts.json"))
    parser.add_argument("--chunk", type=int, default=1, help="tokens per forward pass in the runtime")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--qlinear", help="linear quantization of the export (default: from manifest.json)")
    parser.add_argument("--qembedding", help="embedding quantization of the export (default: from manifest.json)")
    parser.add_argument("--min-top1", type=float, help="minimum top-1 agreement (default: manifest gate.min_top1, else 0.85)")
    parser.add_argument("--report", type=Path, help="write the metrics as JSON here")
    return parser.parse_args()


def render(tokenizer, conversations) -> list[list[int]]:
    prompts = []
    for messages in conversations:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(tokenizer(text, add_special_tokens=False)["input_ids"])
    return prompts


def runtime_top_k(args, prompts) -> dict[tuple[int, int], list[int]]:
    with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
        json.dump({"prompts": prompts}, f)
        f.flush()
        out = subprocess.run(
            [str(args.logits_check), str(args.model_dir / "model.pte"), f.name, str(args.chunk), str(args.top_k)],
            check=True, capture_output=True, text=True,
        ).stdout
    rows = (json.loads(line) for line in out.splitlines() if line.startswith("{"))
    return {(r["prompt"], r["pos"]): [t[0] for t in r["top"]] for r in rows}


def eager_top_k(model, prompts, k) -> dict[tuple[int, int], list[int]]:
    tops = {}
    with torch.no_grad():
        for p, ids in enumerate(prompts):
            logits = model(torch.tensor([ids])).logits[0]
            for pos, ranked in enumerate(logits.topk(k).indices.tolist()):
                tops[(p, pos)] = ranked
    return tops


def compare(prompts, runtime, reference, k) -> dict:
    per_prompt = []
    for p, ids in enumerate(prompts):
        keys = [(p, pos) for pos in range(len(ids))]
        per_prompt.append({
            "tokens": len(ids),
            "top1": sum(runtime[key][0] == reference[key][0] for key in keys) / len(keys),
            f"top{k}_overlap": sum(len(set(runtime[key]) & set(reference[key])) for key in keys) / (len(keys) * k),
            "last_top1": runtime[keys[-1]][0] == reference[keys[-1]][0],
        })
    total = sum(r["tokens"] for r in per_prompt)
    return {"top1": sum(r["top1"] * r["tokens"] for r in per_prompt) / total, "prompts": per_prompt}


def main() -> int:
    args = parse_args()
    manifest_path = args.model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    reference_id = args.reference or manifest.get("source")
    qlinear = args.qlinear or manifest.get("export", {}).get("qlinear")
    qembedding = args.qembedding or manifest.get("export", {}).get("qembedding")
    min_top1 = args.min_top1 if args.min_top1 is not None else manifest.get("gate", {}).get("min_top1", 0.85)
    if not reference_id:
        print("no --reference and no manifest.json source", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompts = render(tokenizer, json.loads(args.prompts.read_text()))
    runtime = runtime_top_k(args, prompts)

    fp32 = AutoModelForCausalLM.from_pretrained(reference_id, torch_dtype=torch.float32).eval()
    references = {"fp32": fp32}
    if qlinear or qembedding:
        from optimum.exporters.executorch.quantization import quantize_model_

        quantized = copy.deepcopy(fp32)
        quantize_model_(quantized, qlinear_config=qlinear, qembedding_config=qembedding)
        references["quantized"] = quantized
    gate_on = "quantized" if "quantized" in references else "fp32"

    results = {}
    for name, model in references.items():
        reference = eager_top_k(model, prompts, args.top_k)
        missing = set(reference) - set(runtime)
        if missing:
            print(f"runtime returned no logits for {len(missing)} positions", file=sys.stderr)
            return 1
        results[name] = compare(prompts, runtime, reference, args.top_k)

    for name, result in results.items():
        label = f"vs {name} eager" + (" (gate)" if name == gate_on else "")
        print(label)
        for i, r in enumerate(result["prompts"]):
            print(f"  prompt {i}: {r['tokens']:4d} tokens  top1 {r['top1']:.3f}  "
                  f"top{args.top_k} overlap {r[f'top{args.top_k}_overlap']:.3f}  "
                  f"last {'ok' if r['last_top1'] else 'DIFF'}")
        print(f"  top-1 agreement {result['top1']:.3f}")

    passed = results[gate_on]["top1"] >= min_top1
    print(f"chunk={args.chunk}: {gate_on} top-1 agreement {results[gate_on]['top1']:.3f} "
          f"(min {min_top1}) -> {'PASS' if passed else 'FAIL'}")
    if args.report:
        args.report.write_text(json.dumps(
            {"chunk": args.chunk, "gate_on": gate_on, "min_top1": min_top1, "passed": passed,
             "qlinear": qlinear, "qembedding": qembedding, "results": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
