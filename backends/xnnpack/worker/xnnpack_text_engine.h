/*
 * LLMEngine/LLMSession over a generic text-decoder .pte (export_llm or
 * optimum-executorch, XNNPACK or portable CPU ops), so ExecuTorch's
 * examples/llm_server control plane can serve it through worker_loop.h.
 *
 * The KV cache lives in the Module's mutable buffers, so one engine hosts
 * exactly one physical session.
 */

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/llm_session.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace infsnp {

namespace llm = ::executorch::extension::llm;

struct XnnpackTextConfig {
  std::string model_path;
  std::string tokenizer_path;
  // Optional .ptd with externalized weights.
  std::string data_path;
  // Token strings that also end a turn (e.g. "<|im_end|>"), on top of the
  // EOS ids found in the .pte metadata and the tokenizer.
  std::vector<std::string> extra_eos_tokens;
  // Upper bound on tokens per prefill forward pass; 0 uses the program's
  // max_seq_len.
  int64_t max_prefill_chunk = 0;
};

class XnnpackTextSession;

class XnnpackTextEngine : public llm::LLMEngine {
 public:
  static ::executorch::runtime::Result<std::unique_ptr<XnnpackTextEngine>>
  create(const XnnpackTextConfig& config);

  ::executorch::runtime::Result<std::unique_ptr<llm::LLMSession>>
  create_session() override;

  llm::LLMServingCapacity serving_capacity() const override {
    return {/*max_physical_sessions_without_weight_duplication=*/1,
            /*estimated_bytes_per_session=*/0};
  }

  const std::unordered_map<std::string, int64_t>& metadata() const override {
    return metadata_;
  }

  ::tokenizers::Tokenizer* tokenizer() const {
    return tokenizer_.get();
  }

 private:
  friend class XnnpackTextSession;
  XnnpackTextEngine() = default;

  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<::executorch::extension::Module> module_;
  std::unique_ptr<llm::IOManager> io_manager_;
  std::unique_ptr<llm::TextDecoderRunner> decoder_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unordered_set<uint64_t> eos_ids_;
  int32_t vocab_size_ = 0;
  int64_t max_context_len_ = 0;
  int64_t max_seq_len_ = 1;
  bool parallel_prefill_ = false;
  bool session_created_ = false;
};

class XnnpackTextSession : public llm::LLMSession {
 public:
  explicit XnnpackTextSession(XnnpackTextEngine& engine) : engine_(engine) {}

  ::executorch::runtime::Error prefill_tokens(
      const std::vector<uint64_t>& tokens,
      const llm::SamplingConfig* initial_sampling) override;

  ::executorch::runtime::Result<llm::DecodeResult> decode_one(
      const llm::SamplingConfig& sampling) override;

  int64_t position() const override {
    return pos_;
  }

  ::executorch::runtime::Error reset() override;

  void stop() override {
    stop_requested_.store(true);
  }

 private:
  // Run tokens through the model starting at pos_, then sample the next token
  // from the last logits into pending_token_.
  ::executorch::runtime::Error forward_and_sample(
      std::vector<uint64_t>& tokens);
  void configure_sampler(const llm::SamplingConfig& sampling);

  XnnpackTextEngine& engine_;
  std::unique_ptr<llm::Sampler> sampler_;
  llm::SamplingConfig sampler_config_;
  int64_t pos_ = 0;
  uint64_t last_token_ = 0;
  uint64_t pending_token_ = 0;
  bool has_pending_ = false;
  std::atomic<bool> stop_requested_{false};
};

} // namespace infsnp
