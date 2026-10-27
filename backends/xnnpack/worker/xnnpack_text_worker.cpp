/*
 * Model worker for ExecuTorch's examples/llm_server control plane.
 *
 * The server launches it as:
 *   xnnpack-text-worker --model_path <model.pte> --tokenizer_path <tokenizer>
 * and inherits its environment, which carries the optional settings:
 *   EXECUTORCH_DATA_PATH         .ptd with externalized weights
 *   EXECUTORCH_EXTRA_EOS_TOKENS  comma-separated turn terminators, e.g.
 *                                "<|im_end|>"
 *   EXECUTORCH_MAX_PREFILL_CHUNK cap on tokens per prefill pass (default: none)
 *   EXECUTORCH_BOS_ID            token id prepended to every rendered prompt
 *   EXECUTORCH_WARM_RESUME       "0" disables warm resume for named sessions
 */

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include <executorch/examples/llm_server/cpp/worker_loop.h>
#include <executorch/runtime/platform/log.h>

#include "xnnpack_text_engine.h"

namespace {

std::string env_or(const char* name, const std::string& fallback = "") {
  const char* value = std::getenv(name);
  return value != nullptr ? value : fallback;
}

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> items;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      items.push_back(item);
    }
  }
  return items;
}

} // namespace

int main(int argc, char** argv) {
  infsnp::XnnpackTextConfig config;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--model_path") == 0) {
      config.model_path = argv[i + 1];
    } else if (std::strcmp(argv[i], "--tokenizer_path") == 0) {
      config.tokenizer_path = argv[i + 1];
    } else {
      ET_LOG(Error, "xnnpack-text-worker: unknown argument %s", argv[i]);
      return 1;
    }
  }
  if (config.model_path.empty() || config.tokenizer_path.empty()) {
    ET_LOG(
        Error,
        "usage: xnnpack-text-worker --model_path <pte> --tokenizer_path <tokenizer>");
    return 1;
  }
  config.data_path = env_or("EXECUTORCH_DATA_PATH");
  config.extra_eos_tokens = split_csv(env_or("EXECUTORCH_EXTRA_EOS_TOKENS"));
  config.max_prefill_chunk =
      std::stoll(env_or("EXECUTORCH_MAX_PREFILL_CHUNK", "0"));

  std::vector<uint64_t> prompt_prefix_ids;
  const std::string bos_id = env_or("EXECUTORCH_BOS_ID");
  if (!bos_id.empty()) {
    prompt_prefix_ids.push_back(std::stoull(bos_id));
  }
  const bool warm_resume = env_or("EXECUTORCH_WARM_RESUME", "1") != "0";

  auto engine = infsnp::XnnpackTextEngine::create(config);
  if (!engine.ok()) {
    ET_LOG(Error, "xnnpack-text-worker: failed to create engine");
    return 1;
  }

  return ::executorch::extension::llm::run_worker_stdio_loop(
      *engine.get(),
      *engine.get()->tokenizer(),
      engine.get()->metadata(),
      warm_resume,
      prompt_prefix_ids);
}
