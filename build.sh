#!/bin/bash
# Build an ExecuTorch runtime tarball for inference snaps.
#
# usage: ./build.sh <backend> [work-dir]
#
# Produces dist/executorch-<backend>-<arch>.tar.xz, laid out like a runtime
# component (the same shape as canonical/llama.cpp-builds tarballs):
#
#   bin/<backend>-text-worker   C++ worker, ExecuTorch linked in statically
#   llm_server/                 ExecuTorch's OpenAI control plane, unmodified
#   lib/python/                 its Python dependencies (pip --target)
#   licenses/                   ExecuTorch and statically linked third parties
#   BUILD_INFO.json             versions and commits this was built from
#
# Build on Ubuntu 24.04 so the Python dependencies match core24's python3.12.
# Needs: git cmake ninja-build make g++ curl python3-venv

set -euo pipefail

backend="${1:?usage: ./build.sh <backend> [work-dir]}"
root="$(cd "$(dirname "$0")" && pwd)"
work="$(realpath -m "${2:-$root/work}")"
version="$(cat "$root/EXECUTORCH_VERSION")"
arch="$(dpkg --print-architecture)"
jobs="${JOBS:-$(nproc)}"

case "$backend" in
  xnnpack) worker=xnnpack-text-worker ;;
  *) echo "unsupported backend: $backend (supported: xnnpack)" >&2; exit 1 ;;
esac

python=python3.12
command -v "$python" >/dev/null || { echo "$python is required (core24's interpreter)" >&2; exit 1; }

src="$work/src/executorch"
venv="$work/build-venv"
et_out="$work/executorch-out"
et_install="$work/executorch-install"
worker_out="$work/worker-$backend"
stage="$work/stage-$backend"
dist="$root/dist"
tarball="$dist/executorch-$backend-$arch.tar.xz"

"$root/scripts/fetch-executorch.sh" "$version" "$src"

# ExecuTorch's kernel registration codegen imports torchgen, which ships with
# torch. Only the build uses it; nothing from this venv reaches the tarball
# except the server's dependencies, installed with its pip.
torch_version="$(sed -n 's/^TORCH_VERSION = "\(.*\)"/\1/p' "$src/torch_pin.py")"
if [ ! -x "$venv/bin/python" ]; then
  "$python" -m venv "$venv"
  "$venv/bin/pip" install --quiet --index-url https://download.pytorch.org/whl/cpu "torch==$torch_version"
  "$venv/bin/pip" install --quiet pyyaml
fi

cmake -S "$src" -B "$et_out" -G Ninja --preset llm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$et_install" \
  -DPYTHON_EXECUTABLE="$venv/bin/python" \
  -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_ASR_RUNNER=OFF
cmake --build "$et_out" -j"$jobs" --target install

cmake -S "$root/backends/$backend/worker" -B "$worker_out" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$et_install" \
  -DEXECUTORCH_SOURCE_DIR="$src" \
  -DBUILD_LOGITS_CHECK=ON
cmake --build "$worker_out" -j"$jobs"

rm -rf "$stage" && mkdir -p "$stage/bin" "$stage/licenses"
install -m 755 "$worker_out/$worker" "$stage/bin/"

cp -a "$src/examples/llm_server/python" "$stage/llm_server"
rm -rf "$stage/llm_server/tests"

"$venv/bin/pip" install --quiet --no-compile --only-binary=:all: \
  --target "$stage/lib/python" -r "$root/backends/$backend/requirements.txt"
rm -rf "$stage/lib/python/bin"

install -Dm644 "$src/LICENSE" "$stage/licenses/executorch/LICENSE"
for dep in \
  backends/xnnpack/third-party/XNNPACK backends/xnnpack/third-party/cpuinfo \
  backends/xnnpack/third-party/pthreadpool backends/xnnpack/third-party/FP16 \
  backends/xnnpack/third-party/FXdiv extension/llm/tokenizers \
  extension/llm/tokenizers/third-party/sentencepiece extension/llm/tokenizers/third-party/re2 \
  extension/llm/tokenizers/third-party/pcre2 extension/llm/tokenizers/third-party/abseil-cpp \
  third-party/json third-party/flatbuffers third-party/ao kernels/optimized/third-party/eigen; do
  for file in "$src/$dep"/LICENSE* "$src/$dep"/COPYING*; do
    [ -f "$file" ] && install -Dm644 "$file" "$stage/licenses/$(basename "$dep")/$(basename "$file")"
  done
done

cat > "$stage/BUILD_INFO.json" <<EOF
{
  "backend": "$backend",
  "arch": "$arch",
  "executorch_version": "$version",
  "executorch_commit": "$(git -C "$src" rev-parse HEAD)",
  "builds_commit": "$(git -C "$root" rev-parse --verify --quiet HEAD || echo unknown)",
  "torch_version_for_codegen": "$torch_version",
  "python": "$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')",
  "worker": "bin/$worker"
}
EOF

mkdir -p "$dist"
tar -C "$stage" -cJf "$tarball" .
(cd "$dist" && sha256sum "$(basename "$tarball")" > "$(basename "$tarball").sha256")
echo "Built $tarball ($(du -h "$tarball" | cut -f1))"
