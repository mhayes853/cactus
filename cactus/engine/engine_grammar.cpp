#include "engine.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <variant>

namespace cactus {
namespace engine {

Grammar::Grammar() : grammar(nullptr) {}

Grammar::Grammar(xgrammar::Grammar raw_grammar)
    : grammar(std::make_shared<xgrammar::Grammar>(std::move(raw_grammar))) {}

Grammar Grammar::gbnf(const std::string& gbnf, const std::string& start_symbol) {
    return Grammar(xgrammar::Grammar::FromEBNF(gbnf, start_symbol));
}

Grammar Grammar::json() {
    return Grammar(xgrammar::Grammar::BuiltinJSONGrammar());
}

Grammar Grammar::json_schema(
    const std::string& json_schema,
    bool any_whitespace,
    int indent,
    std::pair<std::string, std::string> separators,
    bool strict_mode,
    int max_whitespace_count
) {
    return Grammar(xgrammar::Grammar::FromJSONSchema(
        json_schema,
        any_whitespace,
        indent,
        separators,
        strict_mode,
        max_whitespace_count >= 0 ? std::optional<int>(max_whitespace_count) : std::nullopt
    ));
}

Grammar Grammar::regex(const std::string& regex) {
    return Grammar(xgrammar::Grammar::FromRegex(regex));
}

Grammar Grammar::structural_tag(const std::string& structural_tag_json) {
    auto result = xgrammar::Grammar::FromStructuralTag(structural_tag_json);
    if (std::holds_alternative<xgrammar::Grammar>(result)) {
        return Grammar(std::get<xgrammar::Grammar>(std::move(result)));
    }
    throw std::get<xgrammar::StructuralTagError>(result);
}

Grammar Grammar::unite(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) {
            continue;
        }
        handles.push_back(*grammar.handle());
    }

    if (handles.empty()) {
        return Grammar();
    }
    return Grammar(xgrammar::Grammar::Union(handles));
}

Grammar Grammar::concatenate(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) {
            continue;
        }
        handles.push_back(*grammar.handle());
    }

    if (handles.empty()) {
        return Grammar();
    }
    return Grammar(xgrammar::Grammar::Concat(handles));
}

bool Grammar::is_empty() const {
    return grammar == nullptr;
}

std::shared_ptr<xgrammar::Grammar> Grammar::handle() const {
    return grammar;
}

namespace {

static xgrammar::VocabType to_xgrammar_vocab_type(VocabType vocab_type) {
    switch (vocab_type) {
        case VocabType::RAW:
            return xgrammar::VocabType::RAW;
        case VocabType::BYTE_LEVEL:
            return xgrammar::VocabType::BYTE_LEVEL;
    }
}

} // anonymous namespace

GrammarMatcher::GrammarMatcher(const Grammar* grammar, const TokenizerInfo& tokenizer_info)
    : matcher(nullptr), tokenizer_info(nullptr) {
    auto handle = grammar->handle();
    if (!handle) {
        throw std::runtime_error("Grammar handle is null");
    }
    std::vector<int32_t> stop_token_ids_int32(tokenizer_info.stop_token_ids.begin(), tokenizer_info.stop_token_ids.end());
    xgrammar::TokenizerInfo xgrammar_tokenizer_info{
        tokenizer_info.encoded_vocab,
        to_xgrammar_vocab_type(tokenizer_info.vocab_type),
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

} // namespace engine
} // namespace cactus
