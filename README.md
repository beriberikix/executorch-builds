# executorch-builds

Prebuilt [ExecuTorch](https://github.com/pytorch/executorch) runtimes and exported models for
[inference snaps](https://documentation.ubuntu.com/inference-snaps/), in the spirit of
[canonical/llama.cpp-builds](https://github.com/canonical/llama.cpp-builds). A model snap unpacks a runtime
tarball into a component with `plugin: dump` instead of compiling ExecuTorch itself.

Pinned to ExecuTorch **[`EXECUTORCH_VERSION`](EXECUTORCH_VERSION)** (1.5.1).

## What a runtime tarball contains

`executorch-<backend>-<arch>.tar.xz`, laid out like a runtime component:

| Path | Contents |
|---|---|
| `bin/<backend>-text-worker` | C++ worker, ExecuTorch and the backend linked statically; speaks `llm_server`'s JSONL protocol |
| `llm_server/` | ExecuTorch's [OpenAI control plane](https://github.com/pytorch/executorch/tree/v1.5.1/examples/llm_server), unmodified |
| `lib/python/` | its Python dependencies for python3.12 (core24) |
| `licenses/` | ExecuTorch and the third-party code linked into the worker |
| `BUILD_INFO.json` | ExecuTorch version and commit, this repo's commit |

Run it with `PYTHONPATH=<root>:<root>/lib/python python3.12 -m llm_server.server --worker-bin <root>/bin/... `.
The worker takes `--model_path` and `--tokenizer_path` from the server and the rest from the environment:
`EXECUTORCH_EXTRA_EOS_TOKENS`, `EXECUTORCH_MAX_PREFILL_CHUNK`, `EXECUTORCH_DATA_PATH`, `EXECUTORCH_BOS_ID`,
`EXECUTORCH_WARM_RESUME` (see [`xnnpack_text_worker.cpp`](backends/xnnpack/worker/xnnpack_text_worker.cpp)).

| Backend | Worker | Status |
|---|---|---|
| `xnnpack` | `xnnpack-text-worker`: single-method text decoders as optimum-executorch exports them for CPU | Builds on amd64; arm64 in CI |
| `qnn` | needs its own worker around ExecuTorch's `qnn_llama_runner` | Not started |
| `mediatek` | needs its own worker around ExecuTorch's MediaTek `llama_runner` | Not started |

NPU backends run LLMs through vendor-specific runners in ExecuTorch, so each gets its own worker. The server,
tarball layout and snap integration stay the same.

## Build a runtime

On Ubuntu 24.04 (for python3.12, matching core24):

```shell
sudo apt-get install -y git cmake ninja-build make g++ curl python3-venv xz-utils
./build.sh xnnpack          # -> dist/executorch-xnnpack-<arch>.tar.xz (about 20 min cold)
```

ExecuTorch's kernel codegen imports `torchgen`, so the build installs CPU PyTorch into a build-only venv.
[`scripts/fetch-executorch.sh`](scripts/fetch-executorch.sh) clones only the submodules the build needs and
falls back to GitLab's archive endpoint for Eigen, whose git server often refuses clones under load.

## Export and validate a model

Each model is a manifest in [`models/`](models/). Export it with the pinned tools:

```shell
uv venv --python 3.12 .venv
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -r tools/requirements-export.txt
.venv/bin/python tools/export.py models/qwen3-0-6b-xnnpack.yaml   # -> dist/models/qwen3-0-6b-xnnpack/
```

The output has `model.pte`, the tokenizer files and `manifest.json` (source revision, tool versions, export
and runtime settings, checksums).

[`tools/validate.py`](tools/validate.py) is the numerics gate. It runs fixed chat prompts through
`logits-check` (built with the worker, same kernels) and through eager PyTorch, and compares the top-5
predictions at every position:

```shell
.venv/bin/python tools/validate.py --model-dir dist/models/qwen3-0-6b-xnnpack \
  --logits-check work/worker-xnnpack/logits-check --chunk 4096
```

It gates on agreement with the eager model **quantized the same way as the export** (torchao, through
optimum-executorch's own quantization function), so it measures what the runtime adds rather than
quantization error. Agreement with fp32 is reported for information. For Qwen3-0.6B 8da4w on x86:

| Reference | Top-1 agreement, token-by-token prefill | Whole-prompt prefill |
|---|---|---|
| Eager, quantized like the export (gate) | 0.88 | 0.92 |
| Eager, fp32 | 0.70 | 0.73 |

The threshold lives in the model manifest (`gate.min_top1`, 0.85 here). The remaining gap to the quantized
reference is kernel-level numerics (XNNPACK's int8 kernels against torchao's reference ones); it is worth
an upstream question, not a blocker.

## CI and releases

[`.github/workflows/build.yaml`](.github/workflows/build.yaml) builds the runtime on amd64 and arm64, exports
each model, runs the numerics gate and ExecuTorch's OpenAI conformance suite against the runtime, and on a
`v*` tag publishes everything to a release (`v<executorch version>-r<n>`): runtime tarballs with checksums,
and `<model>.tar` with its manifest and validation report.
