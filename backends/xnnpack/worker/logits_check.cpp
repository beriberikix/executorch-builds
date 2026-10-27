/*
 * Dump the top-k next-token logits at every prompt position, as the shipping
 * runtime computes them, for comparison against eager PyTorch.
 *
 *   logits-check <model.pte> <prompts.json> [chunk] [k]
 *
 * prompts.json is {"prompts": [[token ids...], ...]}. Each prompt is fed in
 * passes of `chunk` tokens (default 1, like the worker) from position 0, and
 * one JSON line per position is printed:
 *   {"prompt": i, "pos": p, "top": [[token id, logit], ...]}
 */

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <nlohmann/json.hpp>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using ::executorch::aten::ScalarType;
using ::executorch::extension::from_blob;
using ::executorch::extension::Module;

namespace {

nlohmann::json top_k(const float* row, int64_t vocab, int k) {
  std::vector<int64_t> ids(vocab);
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int64_t a, int64_t b) {
    return row[a] > row[b];
  });
  nlohmann::json top = nlohmann::json::array();
  for (int i = 0; i < k; ++i) {
    top.push_back({ids[i], row[ids[i]]});
  }
  return top;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: logits-check <model.pte> <prompts.json> [chunk] [k]" << std::endl;
    return 1;
  }
  const int chunk = argc > 3 ? std::atoi(argv[3]) : 1;
  const int k = argc > 4 ? std::atoi(argv[4]) : 5;
  const auto prompts = nlohmann::json::parse(std::ifstream(argv[2])).at("prompts");

  Module module(argv[1], Module::LoadMode::MmapUseMlockIgnoreErrors);
  for (size_t p = 0; p < prompts.size(); ++p) {
    const auto ids = prompts[p].get<std::vector<int64_t>>();
    const int n = static_cast<int>(ids.size());
    for (int begin = 0; begin < n; begin += chunk) {
      const int len = std::min(chunk, n - begin);
      std::vector<int64_t> tokens(ids.begin() + begin, ids.begin() + begin + len);
      std::vector<int64_t> positions(len);
      std::iota(positions.begin(), positions.end(), begin);
      auto out = module.execute(
          "forward",
          {from_blob(tokens.data(), {1, len}, ScalarType::Long),
           from_blob(positions.data(), {len}, ScalarType::Long)});
      if (!out.ok()) {
        std::cerr << "forward failed at prompt " << p << " pos " << begin << std::endl;
        return 1;
      }
      const auto logits = out.get()[0].toTensor();
      const int64_t vocab = logits.size(logits.dim() - 1);
      // Programs that return logits for every input token cover the whole
      // pass; others return only the last position.
      const int rows = logits.dim() == 3 ? static_cast<int>(logits.size(1)) : 1;
      for (int r = 0; r < rows; ++r) {
        const float* row = logits.const_data_ptr<float>() + r * vocab;
        std::cout << nlohmann::json{{"prompt", p}, {"pos", begin + len - rows + r}, {"top", top_k(row, vocab, k)}}
                  << "\n";
      }
    }
  }
}
