#!/bin/bash
# Shallow-clone ExecuTorch at a tag with only the submodules a CPU/LLM build
# needs. Eigen lives on GitLab, which often rejects clones under load, so it
# falls back to GitLab's archive endpoint for the pinned commit.
#
# usage: scripts/fetch-executorch.sh <version> <dest>   (dest must end in /executorch)

set -euo pipefail

version="$1"
dest="$2"

if [ "$(basename "$dest")" != "executorch" ]; then
  echo "destination must be named executorch (builds include it as <executorch/...>)" >&2
  exit 1
fi

submodules=(
  backends/xnnpack/third-party/FP16
  backends/xnnpack/third-party/FXdiv
  backends/xnnpack/third-party/XNNPACK
  backends/xnnpack/third-party/cpuinfo
  backends/xnnpack/third-party/pthreadpool
  extension/llm/tokenizers
  third-party/ao
  third-party/flatbuffers
  third-party/flatcc
  third-party/gflags
  third-party/json
  third-party/pocketfft
)

if [ ! -d "$dest/.git" ]; then
  git clone --quiet --depth 1 --branch "v$version" https://github.com/pytorch/executorch.git "$dest"
fi

git -C "$dest" submodule update --init --depth 1 --jobs 8 "${submodules[@]}"
git -C "$dest/extension/llm/tokenizers" submodule update --init --recursive --depth 1

eigen=kernels/optimized/third-party/eigen
if [ ! -f "$dest/$eigen/CMakeLists.txt" ]; then
  if ! git -C "$dest" submodule update --init --depth 1 "$eigen"; then
    sha="$(git -C "$dest" ls-tree HEAD "$eigen" | awk '{print $3}')"
    echo "Eigen clone failed, downloading archive for $sha"
    rm -rf "${dest:?}/$eigen" && mkdir -p "$dest/$eigen"
    curl --fail --silent --show-error --location --retry 5 --retry-delay 10 \
      "https://gitlab.com/libeigen/eigen/-/archive/$sha/eigen-$sha.tar.gz" |
      tar -xz -C "$dest/$eigen" --strip-components=1
  fi
fi

echo "ExecuTorch v$version ready at $dest ($(git -C "$dest" rev-parse --short HEAD))"
