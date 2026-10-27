#include "xnnpack_text_engine.h"

#include <algorithm>
#include <random>

#include <executorch/extension/llm/runner/constants.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/sampler/util.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/platform/log.h>

namespace infsnp {

using ::executorch::extension::Module;
using ::executorch::runtime::Error;
using ::executorch::runtime::Result;

Result<std::unique_ptr<XnnpackTextEngine>> XnnpackTextEngine::create(
    const XnnpackTextConfig& config) {
  std::unique_ptr<XnnpackTextEngine> engine(new XnnpackTextEngine());

  engine->tokenizer_ = llm::load_tokenizer(config.tokenizer_path);
  if (!engine->tokenizer_ || !engine->tokenizer_->is_loaded()) {
    ET_LOG(Error, "Failed to load tokenizer %s", config.tokenizer_path.c_str());
    return Error::InvalidArgument;
  }

  const auto load_mode = Module::LoadMode::MmapUseMlockIgnoreErrors;
  if (config.data_path.empty()) {
    engine->module_ = std::make_unique<Module>(config.model_path, load_mode);
  } else {
    engine->module_ = std::make_unique<Module>(
        config.model_path,
        std::vector<std::string>{config.data_path},
        load_mode);
  }

  auto metadata =
      llm::get_llm_metadata(engine->tokenizer_.get(), engine->module_.get());
  if (!metadata.ok()) {
    ET_LOG(Error, "Failed to read LLM metadata from %s", config.model_path.c_str());
    return metadata.error();
  }
  engine->metadata_ = std::move(metadata.get());
  engine->vocab_size_ =
      static_cast<int32_t>(engine->metadata_.at(llm::kVocabSize));
  engine->max_context_len_ = engine->metadata_.at(llm::kMaxContextLen);
  engine->max_seq_len_ =
      std::max<int64_t>(1, engine->metadata_.at(llm::kMaxSeqLen));
  if (config.max_prefill_chunk > 0) {
    engine->max_seq_len_ =
        std::min(engine->max_seq_len_, config.max_prefill_chunk);
  }
  engine->parallel_prefill_ =
      engine->metadata_.at(llm::kEnableDynamicShape) != 0;

  engine->eos_ids_ =
      llm::get_eos_ids(engine->tokenizer_.get(), engine->module_.get());
  for (const auto& token : config.extra_eos_tokens) {
    auto ids = engine->tokenizer_->encode(token, /*bos=*/0, /*eos=*/0);
    if (!ids.ok() || ids->size() != 1) {
      ET_LOG(Error, "EOS token '%s' is not a single token", token.c_str());
      return Error::InvalidArgument;
    }
    engine->eos_ids_.insert(ids->front());
  }

  // Two-method (prefill/decode) programs are not handled here; the export
  // recipes in tools/ produce a single "forward" method.
  engine->io_manager_ = std::make_unique<llm::IOManager>(*engine->module_);
  engine->decoder_ = std::make_unique<llm::TextDecoderRunner>(
      engine->module_.get(), engine->io_manager_.get(), llm::kForwardMethod);
  ET_CHECK_OK_OR_RETURN_ERROR(engine->decoder_->load());
  ET_CHECK_OK_OR_RETURN_ERROR(engine->io_manager_->load());

  ET_LOG(
      Info,
      "Loaded %s: vocab=%d max_context=%" PRId64 " max_seq=%" PRId64
      " parallel_prefill=%d eos_ids=%zu",
      config.model_path.c_str(),
      engine->vocab_size_,
      engine->max_context_len_,
      engine->max_seq_len_,
      engine->parallel_prefill_,
      engine->eos_ids_.size());
  return engine;
}

Result<std::unique_ptr<llm::LLMSession>> XnnpackTextEngine::create_session() {
  if (session_created_) {
    return Error::InvalidState;
  }
  session_created_ = true;
  return std::unique_ptr<llm::LLMSession>(new XnnpackTextSession(*this));
}

void XnnpackTextSession::configure_sampler(
    const llm::SamplingConfig& sampling) {
  if (sampler_ && sampling.temperature == sampler_config_.temperature &&
      sampling.top_p == sampler_config_.top_p &&
      sampling.top_k == sampler_config_.top_k &&
      sampling.seed == sampler_config_.seed) {
    return;
  }
  const float temperature = std::max(0.0f, sampling.temperature);
  const unsigned long long seed =
      sampling.seed != 0 ? sampling.seed : std::random_device{}();
  sampler_ = std::make_unique<llm::Sampler>(
      engine_.vocab_size_, temperature, sampling.top_p, seed);
  sampler_->set_topk(sampling.top_k);
  sampler_config_ = sampling;
}

Error XnnpackTextSession::forward_and_sample(std::vector<uint64_t>& tokens) {
  const int64_t chunk_len =
      engine_.parallel_prefill_ ? engine_.max_seq_len_ : 1;
  const size_t n = tokens.size();
  for (size_t begin = 0; begin < n; begin += chunk_len) {
    const auto len = std::min<size_t>(chunk_len, n - begin);
    auto input = ::executorch::extension::from_blob(
        tokens.data() + begin,
        {1, static_cast<executorch::aten::SizesType>(len)},
        executorch::aten::ScalarType::Long);
    auto logits = engine_.decoder_->step(input, pos_);
    ET_CHECK_OK_OR_RETURN_ERROR(logits.error());
    pos_ += static_cast<int64_t>(len);
    if (begin + len == n) {
      pending_token_ = static_cast<uint64_t>(
          llm::sample_from_logits(logits.get(), *sampler_));
    }
  }
  last_token_ = tokens.back();
  has_pending_ = true;
  return Error::Ok;
}

Error XnnpackTextSession::prefill_tokens(
    const std::vector<uint64_t>& tokens,
    const llm::SamplingConfig* initial_sampling) {
  stop_requested_.store(false);
  if (tokens.empty() ||
      pos_ + static_cast<int64_t>(tokens.size()) > engine_.max_context_len_) {
    return Error::InvalidArgument;
  }
  configure_sampler(
      initial_sampling != nullptr ? *initial_sampling : llm::SamplingConfig{});
  std::vector<uint64_t> input = tokens;
  return forward_and_sample(input);
}

Result<llm::DecodeResult> XnnpackTextSession::decode_one(
    const llm::SamplingConfig& sampling) {
  llm::DecodeResult result;
  if (stop_requested_.exchange(false)) {
    result.is_terminal = true;
    return result;
  }
  if (!has_pending_) {
    return Error::InvalidState;
  }

  const uint64_t token = pending_token_;
  result.token_id = token;
  if (engine_.eos_ids_.count(token) > 0) {
    // Not forwarded: position stays put and the session can resume from here.
    has_pending_ = false;
    result.is_eos = true;
    result.is_terminal = true;
    return result;
  }
  if (pos_ >= engine_.max_context_len_) {
    return Error::OutOfResources;
  }

  auto piece = engine_.tokenizer_->decode(last_token_, token);
  if (!piece.ok()) {
    ET_LOG(Error, "Failed to decode token %" PRIu64, token);
    return Error::Internal;
  }
  result.text_piece = std::move(piece.get());

  configure_sampler(sampling);
  std::vector<uint64_t> input{token};
  ET_CHECK_OK_OR_RETURN_ERROR(forward_and_sample(input));
  return result;
}

Error XnnpackTextSession::reset() {
  // The static KV cache is overwritten from position 0 on the next prefill;
  // entries past pos_ are masked out, so there is nothing to clear.
  pos_ = 0;
  last_token_ = 0;
  has_pending_ = false;
  stop_requested_.store(false);
  return Error::Ok;
}

} // namespace infsnp
