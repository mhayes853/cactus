#include "test_utils.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

using cactus::engine::Grammar;
using cactus::engine::GrammarMatcher;
using cactus::engine::TokenizerInfo;

namespace {

struct GrammarFixture {
    std::unique_ptr<cactus::engine::Tokenizer> tokenizer;
    TokenizerInfo tokenizer_info;

    GrammarFixture()
        : tokenizer_info() {
        const char* model_path = std::getenv("CACTUS_TEST_MODEL");
        if (!model_path) {
            throw std::runtime_error("CACTUS_TEST_MODEL is not set");
        }

        tokenizer = cactus::engine::create_tokenizer_from_model_dir(model_path);
        if (!tokenizer) {
            throw std::runtime_error("Failed to load tokenizer from test model files");
        }
        tokenizer_info = tokenizer->get_tokenizer_info();
    }
};

static bool accept_text(GrammarMatcher& matcher, const GrammarFixture& fixture, const std::string& text) {
    const std::vector<uint32_t> tokens = fixture.tokenizer->encode(text);
    if (tokens.empty() && !text.empty()) {
        return false;
    }
    for (uint32_t token : tokens) {
        if (!matcher.accept(token)) {
            return false;
        }
    }
    return true;
}

static bool accepts_complete_text(const Grammar& grammar, const GrammarFixture& fixture, const std::string& text) {
    GrammarMatcher matcher(&grammar, fixture.tokenizer_info);
    if (!accept_text(matcher, fixture, text)) {
        return false;
    }
    return matcher.accept(fixture.tokenizer->get_eos_token());
}

static bool rejects_text(const Grammar& grammar, const GrammarFixture& fixture, const std::string& text) {
    GrammarMatcher matcher(&grammar, fixture.tokenizer_info);
    return !accept_text(matcher, fixture, text);
}

static bool rejects_eos_after_text(const Grammar& grammar, const GrammarFixture& fixture, const std::string& text) {
    GrammarMatcher matcher(&grammar, fixture.tokenizer_info);
    if (!accept_text(matcher, fixture, text)) {
        return true;
    }
    return !matcher.accept(fixture.tokenizer->get_eos_token());
}

static bool test_empty_grammar_properties() {
    Grammar empty;
    Grammar empty2;
    Grammar simple = Grammar::gbnf("root ::= \"hello\"");

    if (!empty.is_empty() || !empty2.is_empty() || simple.is_empty()) {
        return false;
    }

    Grammar empty_union = Grammar::unite({empty, empty2});
    Grammar empty_concat = Grammar::concatenate({empty, empty2});
    Grammar union_with_simple = Grammar::unite({empty, simple});
    Grammar concat_with_simple = Grammar::concatenate({empty, simple});

    return empty_union.is_empty()
        && empty_concat.is_empty()
        && !union_with_simple.is_empty()
        && !concat_with_simple.is_empty();
}

static bool test_concat_accepts_expected_language(const GrammarFixture& fixture) {
    Grammar left = Grammar::gbnf("root ::= \"hello\"");
    Grammar right = Grammar::gbnf("root ::= \" world\"");
    Grammar combined = Grammar::concatenate({left, right});

    return accepts_complete_text(combined, fixture, "hello world")
        && rejects_text(combined, fixture, "world hello")
        && rejects_eos_after_text(combined, fixture, "hello");
}

static bool test_union_accepts_expected_language(const GrammarFixture& fixture) {
    Grammar left = Grammar::gbnf("root ::= \"hello\"");
    Grammar right = Grammar::gbnf("root ::= \"goodbye\"");
    Grammar combined = Grammar::unite({left, right});

    return accepts_complete_text(combined, fixture, "hello")
        && accepts_complete_text(combined, fixture, "goodbye")
        && rejects_text(combined, fixture, "hello goodbye");
}

static bool test_three_way_concat(const GrammarFixture& fixture) {
    Grammar combined = Grammar::concatenate({
        Grammar::gbnf("root ::= \"alpha\""),
        Grammar::gbnf("root ::= \"-\""),
        Grammar::gbnf("root ::= \"beta\"")
    });

    return accepts_complete_text(combined, fixture, "alpha-beta")
        && rejects_text(combined, fixture, "alphabeta")
        && rejects_text(combined, fixture, "beta-alpha");
}

static bool test_three_way_union(const GrammarFixture& fixture) {
    Grammar combined = Grammar::unite({
        Grammar::gbnf("root ::= \"red\""),
        Grammar::gbnf("root ::= \"green\""),
        Grammar::gbnf("root ::= \"blue\"")
    });

    return accepts_complete_text(combined, fixture, "red")
        && accepts_complete_text(combined, fixture, "green")
        && accepts_complete_text(combined, fixture, "blue")
        && rejects_text(combined, fixture, "orange");
}

static bool test_unordered_choice(const GrammarFixture& fixture) {
    Grammar combined = Grammar::gbnf("root ::= \"a\" | \"ab\"");

    return accepts_complete_text(combined, fixture, "a")
        && accepts_complete_text(combined, fixture, "ab")
        && rejects_text(combined, fixture, "ac");
}

static bool test_regex_and_json_schema_construction() {
    Grammar regex = Grammar::regex("(cat|dog)");
    Grammar json_schema = Grammar::json_schema(R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})");
    return !regex.is_empty() && !json_schema.is_empty();
}

static bool test_regex_accepts_expected_text(const GrammarFixture& fixture) {
    Grammar regex = Grammar::regex("(cat|dog)");

    return accepts_complete_text(regex, fixture, "cat")
        && accepts_complete_text(regex, fixture, "dog")
        && rejects_text(regex, fixture, "cow");
}

static bool test_json_schema_accepts_expected_text(const GrammarFixture& fixture) {
    Grammar json_schema = Grammar::json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    );

    return accepts_complete_text(json_schema, fixture, R"({"name":"cactus"})")
        && rejects_text(json_schema, fixture, R"({"age":1})");
}

static bool test_model_decode_accepts_direct_output_when_reasoning_enabled(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return accepts_complete_text(combined, fixture, "hello");
}

static bool test_model_decode_accepts_thinking_then_output(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return accepts_complete_text(combined, fixture, "<think>\nreasoning\n</think>\n\nhello");
}

static bool test_model_decode_accepts_less_than_in_thinking_payload(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return accepts_complete_text(combined, fixture, "<think>\n1 < 2\n</think>\n\nhello");
}

static bool test_model_decode_rejects_thinking_only_eos(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return rejects_eos_after_text(combined, fixture, "<think>\nreasoning\n</think>\n\n");
}

static bool test_model_decode_rejects_thinking_when_reasoning_disabled(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false);

    return rejects_text(combined, fixture, "<think>\nreasoning\n</think>\n\nhello")
        && accepts_complete_text(combined, fixture, "hello");
}

static bool test_model_decode_accepts_thinking_then_json_schema(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    );
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return accepts_complete_text(combined, fixture, "<think>\nreasoning\n</think>\n\n{\"name\":\"cactus\"}");
}

static bool test_model_decode_rejects_invalid_json_schema_after_thinking(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    );
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return rejects_eos_after_text(combined, fixture, "<think>\nreasoning\n</think>\n\n{\"name\":1}")
        && rejects_text(combined, fixture, "<think>\nreasoning\n</think>\n\nthis is some random text");
}

static bool test_model_decode_rejects_nested_thinking_close_tag(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return rejects_text(combined, fixture, "<think>\nreasoning</think>\n</think>\n\nhello");
}

static bool test_model_decode_rejects_invalid_close_only_thinking_string(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true);

    return rejects_text(combined, fixture, "</think>\nreasoning\n</think>\nhello");
}

} // anonymous namespace

int main() {
    TestUtils::TestRunner runner("Grammar Tests");

    runner.run_test("empty_properties", test_empty_grammar_properties());
    runner.run_test("regex_json_schema_init", test_regex_and_json_schema_construction());

    try {
        GrammarFixture fixture;
        runner.run_test("concat_language", test_concat_accepts_expected_language(fixture));
        runner.run_test("union_language", test_union_accepts_expected_language(fixture));
        runner.run_test("three_way_concat", test_three_way_concat(fixture));
        runner.run_test("three_way_union", test_three_way_union(fixture));
        runner.run_test("unordred_choice", test_unordered_choice(fixture));
        runner.run_test("regex_language", test_regex_accepts_expected_text(fixture));
        runner.run_test("json_schema_language", test_json_schema_accepts_expected_text(fixture));
        runner.run_test("model_decode_direct_output_reasoning", test_model_decode_accepts_direct_output_when_reasoning_enabled(fixture));
        runner.run_test("model_decode_thinking_then_output", test_model_decode_accepts_thinking_then_output(fixture));
        runner.run_test("model_decode_thinking_payload_less_than", test_model_decode_accepts_less_than_in_thinking_payload(fixture));
        runner.run_test("model_decode_rejects_thinking_only_eos", test_model_decode_rejects_thinking_only_eos(fixture));
        runner.run_test("model_decode_no_reasoning_prefix", test_model_decode_rejects_thinking_when_reasoning_disabled(fixture));
        runner.run_test("model_decode_thinking_then_json_schema", test_model_decode_accepts_thinking_then_json_schema(fixture));
        runner.run_test("model_decode_rejects_invalid_json_schema_after_thinking", test_model_decode_rejects_invalid_json_schema_after_thinking(fixture));
        runner.run_test("model_decode_invalid_close_only", test_model_decode_rejects_invalid_close_only_thinking_string(fixture));
        runner.run_test("model_decode_rejects_nested_thinking_close_tag", test_model_decode_rejects_nested_thinking_close_tag(fixture));
    } catch (const std::exception& e) {
        std::cerr << "[✗] Grammar test setup failed: " << e.what() << "\n";
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
