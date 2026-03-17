#pragma once

#include "../../libs/xgrammar/include/dlpack/dlpack.h"
#include "../../libs/xgrammar/include/xgrammar/compiler.h"
#include "../../libs/xgrammar/include/xgrammar/config.h"
#include "../../libs/xgrammar/include/xgrammar/exception.h"
#include "../../libs/xgrammar/include/xgrammar/grammar.h"
#include "../../libs/xgrammar/include/xgrammar/matcher.h"
#include "../../libs/xgrammar/include/xgrammar/object.h"
#include "../../libs/xgrammar/include/xgrammar/tokenizer_info.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <arm_neon.h>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cactus {
namespace grammar {

enum class VocabType {
    RAW,
    BYTE_LEVEL,
};

struct TokenizerInfo {
    std::vector<std::string> encoded_vocab;
    VocabType vocab_type = VocabType::RAW;
    size_t vocab_size;
    std::vector<uint32_t> stop_token_ids;
    bool add_prefix_space = false;
};

class Grammar {
public:
    Grammar();
    ~Grammar() = default;

    static Grammar gbnf(const std::string& gbnf, const std::string& start_symbol = "root");
    static Grammar json();
    static Grammar json_schema(
        const std::string& json_schema,
        bool any_whitespace = true,
        int indent = 2,
        std::pair<std::string, std::string> separators = {",", ":"},
        bool strict_mode = true,
        int max_whitespace_count = -1
    );
    static Grammar regex(const std::string& regex);
    static Grammar structural_tag(const std::string& structural_tag_json);
    static Grammar unite(const std::vector<Grammar>& grammars);
    static Grammar concatenate(const std::vector<Grammar>& grammars);

    bool is_empty() const;

    std::shared_ptr<xgrammar::Grammar> handle() const;

private:
    explicit Grammar(xgrammar::Grammar raw_grammar);

    std::shared_ptr<xgrammar::Grammar> grammar;
};

class GrammarMatcher {
public:
    GrammarMatcher(const Grammar* grammar, const TokenizerInfo& tokenizer_info);
    ~GrammarMatcher() = default;

    bool accept(uint32_t token_id);

    bool apply_bitmask(std::vector<__fp16>& logits);
    bool apply_bitmask(std::vector<float>& logits);
private:
    bool apply_bitmask(void* logits, size_t num_logits, uint8_t bits);

    xgrammar::GrammarMatcher matcher;
    xgrammar::TokenizerInfo tokenizer_info;
};

}
}
