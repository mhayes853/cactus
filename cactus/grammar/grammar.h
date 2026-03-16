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
#include <cstdint>
#include <arm_neon.h>
#include <memory>
#include <vector>

namespace cactus {
namespace grammar {

struct TokenizerInfo {
    std::vector<std::string> encoded_vocab;
    size_t vocab_size;
    std::vector<uint32_t> stop_token_ids;
    bool add_prefix_space = false;
};

class Grammar {
public:
    Grammar();
    ~Grammar() = default;

    static Grammar json();

    std::shared_ptr<xgrammar::Grammar> handle() const;
private:
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
