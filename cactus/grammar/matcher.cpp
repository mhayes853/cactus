#include "grammar.h"
#include <cstdint>

namespace cactus {
namespace grammar {

GrammarMatcher::GrammarMatcher(const Grammar* grammar, const TokenizerInfo& tokenizer_info)
    : matcher(nullptr), tokenizer_info(nullptr) {
    auto handle = grammar->handle();
    if (!handle) {
        throw std::runtime_error("Grammar handle is null");
    }
    std::vector<int32_t> stop_token_ids_int32(tokenizer_info.stop_token_ids.begin(), tokenizer_info.stop_token_ids.end());
    xgrammar::TokenizerInfo xgrammar_tokenizer_info{
        tokenizer_info.encoded_vocab,
        xgrammar::VocabType::RAW,
        static_cast<int>(tokenizer_info.vocab_size),
        stop_token_ids_int32,
        tokenizer_info.add_prefix_space
    };
    auto compiler = xgrammar::GrammarCompiler(xgrammar_tokenizer_info);

    auto compiled_grammar = compiler.CompileGrammar(*handle);
    matcher = xgrammar::GrammarMatcher(compiled_grammar);
    this->tokenizer_info = xgrammar_tokenizer_info;
}

bool GrammarMatcher::accept(uint32_t token_id) {
    return matcher.AcceptToken(token_id);
}

bool GrammarMatcher::apply_bitmask(std::vector<__fp16>& logits) {
    return apply_bitmask(logits.data(), logits.size(), 16);
}

bool GrammarMatcher::apply_bitmask(std::vector<float>& logits) {
    return apply_bitmask(logits.data(), logits.size(), 32);
}

bool GrammarMatcher::apply_bitmask(void* logits, size_t num_logits, uint8_t bits) {
    const int32_t bitmask_size = xgrammar::GetBitmaskSize(tokenizer_info.GetVocabSize());
    std::vector<int32_t> token_bitmask(bitmask_size);

    int64_t logits_shape[1] = {static_cast<int64_t>(num_logits)};
    int64_t logits_strides[1] = {1};
    DLTensor logits_tensor;
    logits_tensor.data = logits;
    logits_tensor.device = DLDevice{kDLCPU, 0};
    logits_tensor.ndim = 1;
    logits_tensor.dtype = DLDataType{static_cast<uint8_t>(kDLFloat), bits, 1};
    logits_tensor.shape = logits_shape;
    logits_tensor.strides = logits_strides;
    logits_tensor.byte_offset = 0;

    int64_t bitmask_shape[1] = {bitmask_size};
    int64_t bitmask_strides[1] = {1};
    DLTensor bitmask_tensor;
    bitmask_tensor.data = token_bitmask.data();
    bitmask_tensor.device = DLDevice{kDLCPU, 0};
    bitmask_tensor.ndim = 1;
    bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();
    bitmask_tensor.shape = bitmask_shape;
    bitmask_tensor.strides = bitmask_strides;
    bitmask_tensor.byte_offset = 0;

    const bool should_apply = matcher.FillNextTokenBitmask(&bitmask_tensor, 0);
    if (!should_apply) {
        return false;
    }

    xgrammar::ApplyTokenBitmaskInplaceCPU(&logits_tensor, bitmask_tensor, tokenizer_info.GetVocabSize());
    return true;
}

}
}
