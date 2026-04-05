#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>

using cactus::engine::GrammarVocabulary;

namespace {

struct TokenizerFixture {
    std::unique_ptr<cactus::engine::Tokenizer> tokenizer;
    GrammarVocabulary vocab;

    TokenizerFixture() : vocab() {
        const char* model_path = std::getenv("CACTUS_TEST_MODEL");
        if (!model_path) {
            throw std::runtime_error("CACTUS_TEST_MODEL is not set");
        }

        tokenizer = cactus::engine::create_tokenizer_from_model_dir(model_path);
        if (!tokenizer) {
            throw std::runtime_error("Failed to load tokenizer from test model files");
        }
        vocab = tokenizer->get_grammar_vocabulary();
    }
};

static bool test_sentencepiece_byte_fallback_decodes_to_text(const TokenizerFixture& fixture) {
    const auto& vocab = fixture.vocab.encoded_vocab;

    uint32_t quote_id = 0;
    uint32_t g_id = 0;
    bool found_quote = false;
    bool found_g = false;

    for (uint32_t i = 0; i < vocab.size(); ++i) {
        if (!found_quote && vocab[i] == "<0x22>") {
            quote_id = i;
            found_quote = true;
        }
        if (!found_g && vocab[i] == "<0x67>") {
            g_id = i;
            found_g = true;
        }
    }

    if (!found_quote || !found_g) {
        return true;
    }

    return fixture.tokenizer->decode({quote_id}) == "\""
        && fixture.tokenizer->decode({g_id}) == "g";
}

} // anonymous namespace

int main() {
    TestUtils::TestRunner runner("Tokenizer Tests");

    try {
        TokenizerFixture fixture;
        runner.run_test("sentencepiece_byte_fallback_decode", test_sentencepiece_byte_fallback_decodes_to_text(fixture));
    } catch (const std::exception& e) {
        std::cerr << "[✗] Tokenizer test setup failed: " << e.what() << "\n";
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
