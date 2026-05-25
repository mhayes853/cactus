#include "test_utils.h"

#include "../src/utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace cactus::engine;
using namespace cactus::ffi;

namespace {

using GrammarHandle = std::unique_ptr<void, decltype(&cactus_grammar_destroy)>;
using GrammarVocabularyHandle = std::unique_ptr<void, decltype(&cactus_grammar_vocabulary_destroy)>;
using GrammarEngineHandle = std::unique_ptr<void, decltype(&cactus_grammar_engine_destroy)>;
using GrammarMatcherHandle = std::unique_ptr<void, decltype(&cactus_grammar_matcher_destroy)>;

static GrammarHandle grammar_handle(cactus_grammar_t grammar) {
    if (!grammar) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return GrammarHandle(grammar, &cactus_grammar_destroy);
}

static GrammarVocabularyHandle vocab_handle(cactus_grammar_vocabulary_t vocabulary) {
    return GrammarVocabularyHandle(vocabulary, &cactus_grammar_vocabulary_destroy);
}

static GrammarEngineHandle engine_handle(cactus_grammar_engine_t engine) {
    return GrammarEngineHandle(engine, &cactus_grammar_engine_destroy);
}

static GrammarMatcherHandle matcher_handle(cactus_grammar_matcher_t matcher) {
    if (!matcher) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return GrammarMatcherHandle(matcher, &cactus_grammar_matcher_destroy);
}

static const std::string require_model_path() {
    const char* model_path = std::getenv("CACTUS_TEST_MODEL");
    if (!model_path) {
        throw std::runtime_error("CACTUS_TEST_MODEL is not set");
    }
    return model_path;
}

static std::unique_ptr<cactus::engine::Tokenizer> create_test_tokenizer() {
    auto tokenizer = Tokenizer::from_model_dir(require_model_path());
    if (!tokenizer) {
        throw std::runtime_error("Failed to load tokenizer from test model files");
    }
    return tokenizer;
}

static cactus_grammar_t json_schema_grammar(const std::string& json_schema) {
    return cactus_grammar_init_json_schema(
        json_schema.c_str(),
        cactus_grammar_json_schema_default_options()
    );
}

static cactus_grammar_matcher_t make_matcher(cactus_grammar_t grammar, cactus_grammar_engine_t engine) {
    auto matcher = cactus_grammar_engine_compile_matcher(engine, grammar);
    if (!matcher) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return matcher;
}

struct GrammarFixture {
    std::unique_ptr<cactus::engine::Tokenizer> tokenizer;
    GrammarVocabularyHandle vocab;
    GrammarEngineHandle engine;
    bool add_prefix_space = false;
    size_t vocab_size = 0;
    std::vector<uint32_t> stop_token_ids;

    GrammarFixture()
        : tokenizer(create_test_tokenizer()),
          vocab(vocab_handle(cactus_grammar_vocabulary_init(require_model_path().c_str()))),
          engine(engine_handle(cactus_grammar_engine_init(vocab.get()))) {
        if (!engine) {
            throw std::runtime_error(cactus_get_last_error());
        }
        add_prefix_space = cactus_grammar_vocabulary_get_add_prefix_space(vocab.get());
        vocab_size = cactus_grammar_vocabulary_get_size(vocab.get());
        if (vocab_size == 0) {
            throw std::runtime_error(cactus_get_last_error());
        }
        std::vector<uint32_t> actual_stop_token_ids(16);
        size_t out_token_count = 0;
        auto result = cactus_grammar_vocabulary_get_stop_token_ids(
            vocab.get(),
            actual_stop_token_ids.data(),
            actual_stop_token_ids.size(),
            &out_token_count
        );
        if (result != 0) throw std::runtime_error(cactus_get_last_error());
        actual_stop_token_ids.resize(out_token_count);
        stop_token_ids = std::move(actual_stop_token_ids);
    }
};

static bool accept_text(cactus_grammar_matcher_t matcher, const GrammarFixture& fixture, const std::string& text) {
    const std::vector<uint32_t> tokens = fixture.tokenizer->encode(text);
    if (tokens.empty() && !text.empty()) return false;
    for (uint32_t token : tokens) {
        if (!cactus_grammar_matcher_accept(matcher, token)) return false;
    }
    return true;
}

static bool accepts_complete_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = matcher_handle(make_matcher(grammar, fixture.engine.get()));
    if (!accept_text(matcher.get(), fixture, text)) return false;
    return cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool rejects_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = matcher_handle(make_matcher(grammar, fixture.engine.get()));
    return !accept_text(matcher.get(), fixture, text);
}

static bool rejects_eos_after_text(cactus_grammar_t grammar, const GrammarFixture& fixture, const std::string& text) {
    auto matcher = matcher_handle(make_matcher(grammar, fixture.engine.get()));
    if (!accept_text(matcher.get(), fixture, text)) return true;
    return !cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool bitmask_allows_token(const std::vector<int32_t>& bitmask, uint32_t token_id) {
    const size_t word_index = token_id / 32;
    const uint32_t bit_index = token_id % 32;
    if (word_index >= bitmask.size()) return false;
    return ((bitmask[word_index]) & (uint32_t{1} << bit_index)) != 0;
}

static size_t bitmask_size(size_t logits_buffer_size) {
    return (logits_buffer_size + 31) / 32;
}

static std::string grammar_ebnf(cactus_grammar_t grammar) {
    std::vector<char> buffer(4096, '\0');
    if (cactus_grammar_get_ebnf(grammar, buffer.data(), buffer.size()) != 0) {
        throw std::runtime_error(cactus_get_last_error());
    }
    return std::string(buffer.data());
}

static std::string tool_call_structural_tag_json() {
    return R"({
        "type": "structural_tag",
        "format": {
            "type": "tags_with_separator",
            "separator": "\n",
            "tags": [
                {
                    "begin": "<|tool_call>\ncall:function_name_1(",
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            },
                            "required": ["city"],
                            "additionalProperties": false
                        }
                    },
                    "end": ")\n<tool_call|>"
                },
                {
                    "begin": "<|tool_call>\ncall:function_name_2(",
                    "content": {
                        "type": "json_schema",
                        "json_schema": {
                            "type": "object",
                            "properties": {
                                "count": {"type": "integer"}
                            },
                            "required": ["count"],
                            "additionalProperties": false
                        }
                    },
                    "end": ")\n<tool_call|>"
                }
            ]
        }
    })";
}

static std::string test_model_tools_json() {
    return R"([
        {
            "type": "function",
            "function": {
                "name": "complex_tool",
                "description": "A tool with broad parameter coverage.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "count": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "mode": {
                            "type": "string",
                            "enum": ["dry_run", "execute"]
                        },
                        "ticket_id": {
                            "type": "string",
                            "pattern": "[A-Z]{3}-[0-9]{2}"
                        },
                        "priority": {
                            "anyOf": [
                                {"type": "integer"},
                                {
                                    "type": "string",
                                    "enum": ["low", "high"]
                                }
                            ]
                        },
                        "routing": {
                            "oneOf": [
                                {
                                    "type": "string",
                                    "enum": ["auto"]
                                },
                                {
                                    "type": "object",
                                    "properties": {
                                        "region": {"type": "string"}
                                    },
                                    "required": ["region"],
                                    "additionalProperties": false
                                }
                            ]
                        },
                        "labels": {
                            "type": "object",
                            "patternProperties": {
                                "[A-Z_]+": {"type": "integer"}
                            },
                            "additionalProperties": false
                        },
                        "window": {
                            "allOf": [
                                {"type": "integer", "minimum": 1, "maximum": 5}
                            ]
                        },
                        "tuple_args": {
                            "type": "array",
                            "prefixItems": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "boolean"}
                            ],
                            "items": false
                        },
                        "optional_note": {"type": ["string", "null"]},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": true
                        },
                        "config": {
                            "type": "object",
                            "properties": {
                                "threshold": {"type": "number"},
                                "flags": {
                                    "type": "array",
                                    "items": {"type": "boolean"}
                                }
                            },
                            "required": ["threshold", "flags"],
                            "additionalProperties": false
                        }
                    },
                    "required": ["title", "count", "enabled", "mode", "ticket_id", "priority", "routing", "labels", "window", "tuple_args", "optional_note", "tags", "config"],
                    "additionalProperties": false
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather lookup.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        }
    ])";
}

static bool test_vocab_accessors(const GrammarFixture& fixture) {
    const auto vocabulary = GrammarVocabulary::from_tokenizer(*fixture.tokenizer);
    return fixture.vocab_size == fixture.tokenizer->get_vocab_size()
        && fixture.add_prefix_space == vocabulary.add_prefix_space()
        && !fixture.stop_token_ids.empty()
        && fixture.stop_token_ids.front() == fixture.tokenizer->get_eos_token();
}

static bool test_empty_grammar_properties() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto empty2 = grammar_handle(cactus_grammar_init_empty());
    auto simple = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));

    if (!cactus_grammar_is_empty(empty.get())
        || !cactus_grammar_is_empty(empty2.get())
        || cactus_grammar_is_empty(simple.get())) {
        return false;
    }

    cactus_grammar_t empty_union_inputs[] = {empty.get(), empty2.get()};
    auto empty_union = grammar_handle(cactus_grammar_union(empty_union_inputs, 2));
    auto empty_concat = grammar_handle(cactus_grammar_concatenate(empty_union_inputs, 2));

    cactus_grammar_t union_with_simple_inputs[] = {empty.get(), simple.get()};
    auto union_with_simple = grammar_handle(cactus_grammar_union(union_with_simple_inputs, 2));
    auto concat_with_simple = grammar_handle(cactus_grammar_concatenate(union_with_simple_inputs, 2));

    return cactus_grammar_is_empty(empty_union.get())
        && cactus_grammar_is_empty(empty_concat.get())
        && !cactus_grammar_is_empty(union_with_simple.get())
        && !cactus_grammar_is_empty(concat_with_simple.get());
}

static bool test_epsilon_grammar_accepts_only_empty_string(const GrammarFixture& fixture) {
    auto epsilon = grammar_handle(cactus_grammar_init_epsilon());
    return accepts_complete_text(epsilon.get(), fixture, "")
        && rejects_text(epsilon.get(), fixture, "hello");
}

static bool test_optional_grammar_accepts_zero_or_one_occurrence(const GrammarFixture& fixture) {
    auto hello = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto optional = grammar_handle(cactus_grammar_optional(hello.get()));

    return accepts_complete_text(optional.get(), fixture, "")
        && accepts_complete_text(optional.get(), fixture, "hello")
        && rejects_text(optional.get(), fixture, "goodbye")
        && rejects_text(optional.get(), fixture, "hellohello");
}

static bool test_optional_empty_grammar_stays_empty() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto optional = grammar_handle(cactus_grammar_optional(empty.get()));
    return cactus_grammar_is_empty(optional.get());
}

static bool test_star_grammar_accepts_zero_or_more_occurrences(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto star = grammar_handle(cactus_grammar_star(ha.get()));

    return accepts_complete_text(star.get(), fixture, "")
        && accepts_complete_text(star.get(), fixture, "ha")
        && accepts_complete_text(star.get(), fixture, "hahaha")
        && rejects_text(star.get(), fixture, "hello");
}

static bool test_star_empty_grammar_stays_empty() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto star = grammar_handle(cactus_grammar_star(empty.get()));
    return cactus_grammar_is_empty(star.get());
}

static bool test_repeat_exact_language(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto repeated = grammar_handle(cactus_grammar_repeat(ha.get(), 3));

    return accepts_complete_text(repeated.get(), fixture, "hahaha")
        && rejects_eos_after_text(repeated.get(), fixture, "")
        && rejects_eos_after_text(repeated.get(), fixture, "ha")
        && rejects_text(repeated.get(), fixture, "hahahaha")
        && rejects_text(repeated.get(), fixture, "hello");
}

static bool test_repeat_range_language(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto repeated = grammar_handle(cactus_grammar_repeat_range(ha.get(), 2, 4));

    return accepts_complete_text(repeated.get(), fixture, "haha")
        && accepts_complete_text(repeated.get(), fixture, "hahaha")
        && accepts_complete_text(repeated.get(), fixture, "hahahaha")
        && rejects_eos_after_text(repeated.get(), fixture, "")
        && rejects_eos_after_text(repeated.get(), fixture, "ha")
        && rejects_text(repeated.get(), fixture, "hahahahaha");
}

static bool test_repeat_range_unbounded_language(const GrammarFixture& fixture) {
    auto ha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"ha\"", "root"));
    auto repeated = grammar_handle(cactus_grammar_repeat_range(ha.get(), 2, -1));

    return accepts_complete_text(repeated.get(), fixture, "haha")
        && accepts_complete_text(repeated.get(), fixture, "hahaha")
        && accepts_complete_text(repeated.get(), fixture, "hahahaha")
        && rejects_eos_after_text(repeated.get(), fixture, "")
        && rejects_eos_after_text(repeated.get(), fixture, "ha");
}

static bool test_repeat_empty_grammar_stays_empty() {
    auto empty = grammar_handle(cactus_grammar_init_empty());
    auto exact = grammar_handle(cactus_grammar_repeat(empty.get(), 3));
    auto range = grammar_handle(cactus_grammar_repeat_range(empty.get(), 1, 3));
    return cactus_grammar_is_empty(exact.get()) && cactus_grammar_is_empty(range.get());
}

static bool test_ebnf_string_export_matches_parenthesized_input_ebnf() {
    const std::string source = "root ::= ((\"hello\") | (\"hi\"))\n";
    auto grammar = grammar_handle(cactus_grammar_init_ebnf(source.c_str(), "root"));
    return grammar_ebnf(grammar.get()) == source;
}

static bool test_concat_accepts_expected_language(const GrammarFixture& fixture) {
    auto left = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto right = grammar_handle(cactus_grammar_init_ebnf("root ::= \" world\"", "root"));
    cactus_grammar_t handles[] = {left.get(), right.get()};
    auto combined = grammar_handle(cactus_grammar_concatenate(handles, 2));

    return accepts_complete_text(combined.get(), fixture, "hello world")
        && rejects_text(combined.get(), fixture, "world hello")
        && rejects_eos_after_text(combined.get(), fixture, "hello");
}

static bool test_union_accepts_expected_language(const GrammarFixture& fixture) {
    auto left = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto right = grammar_handle(cactus_grammar_init_ebnf("root ::= \"goodbye\"", "root"));
    cactus_grammar_t handles[] = {left.get(), right.get()};
    auto combined = grammar_handle(cactus_grammar_union(handles, 2));

    return accepts_complete_text(combined.get(), fixture, "hello")
        && accepts_complete_text(combined.get(), fixture, "goodbye")
        && rejects_text(combined.get(), fixture, "hello goodbye");
}

static bool test_three_way_concat(const GrammarFixture& fixture) {
    auto alpha = grammar_handle(cactus_grammar_init_ebnf("root ::= \"alpha\"", "root"));
    auto dash = grammar_handle(cactus_grammar_init_ebnf("root ::= \"-\"", "root"));
    auto beta = grammar_handle(cactus_grammar_init_ebnf("root ::= \"beta\"", "root"));
    cactus_grammar_t handles[] = {alpha.get(), dash.get(), beta.get()};
    auto combined = grammar_handle(cactus_grammar_concatenate(handles, 3));

    return accepts_complete_text(combined.get(), fixture, "alpha-beta")
        && rejects_text(combined.get(), fixture, "alphabeta")
        && rejects_text(combined.get(), fixture, "beta-alpha");
}

static bool test_three_way_union(const GrammarFixture& fixture) {
    auto red = grammar_handle(cactus_grammar_init_ebnf("root ::= \"red\"", "root"));
    auto green = grammar_handle(cactus_grammar_init_ebnf("root ::= \"green\"", "root"));
    auto blue = grammar_handle(cactus_grammar_init_ebnf("root ::= \"blue\"", "root"));
    cactus_grammar_t handles[] = {red.get(), green.get(), blue.get()};
    auto combined = grammar_handle(cactus_grammar_union(handles, 3));

    return accepts_complete_text(combined.get(), fixture, "red")
        && accepts_complete_text(combined.get(), fixture, "green")
        && accepts_complete_text(combined.get(), fixture, "blue")
        && rejects_text(combined.get(), fixture, "orange");
}

static bool test_unordered_choice(const GrammarFixture& fixture) {
    auto combined = grammar_handle(cactus_grammar_init_ebnf("root ::= \"a\" | \"ab\"", "root"));

    return accepts_complete_text(combined.get(), fixture, "a")
        && accepts_complete_text(combined.get(), fixture, "ab")
        && rejects_text(combined.get(), fixture, "ac");
}

static bool test_regex_and_json_schema_construction() {
    auto regex = grammar_handle(cactus_grammar_init_regex("(cat|dog)"));
    auto json_schema = grammar_handle(json_schema_grammar(R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"));
    return !cactus_grammar_is_empty(regex.get()) && !cactus_grammar_is_empty(json_schema.get());
}

static bool test_universal_grammar_accepts_anything(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_universal());
    return accepts_complete_text(grammar.get(), fixture, "")
        && accepts_complete_text(grammar.get(), fixture, "blob says hello from cactus")
        && accepts_complete_text(grammar.get(), fixture, "line one\nline two\nline three");
}

static bool test_structural_tag_accepts_and_rejects_expected_text(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(
        cactus_grammar_init_structural_tag(tool_call_structural_tag_json().c_str(), fixture.vocab.get())
    );
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    std::vector<int32_t> bitmask(bitmask_size(fixture.vocab_size));

    return !cactus_grammar_is_empty(grammar.get())
        && matcher != nullptr
        && cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) == 1;
}

static bool test_model_tools_gemma4_accepts_valid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("gemma4", test_model_tools_json().c_str()));

    const auto complex_tool_call =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1,BETA_LABEL:2},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto complex_tool_call_with_string_whitespace =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha beta<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>XYZ-34<|"|>,)"
        R"(priority:<|"|>high<|"|>,)"
        R"(routing:{region:<|"|>us west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:5,)"
        R"(tuple_args:[<|"|>alpha beta<|"|>,2,true],)"
        R"(optional_note:<|"|>contains spaces<|"|>,)"
        R"(tags:[<|"|>first value<|"|>,<|"|>second value<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto get_weather_call =
        R"(<|tool_call>call:get_weather{)"
        R"(location:<|"|>Seoul<|"|>)"
        R"(}<tool_call|>)";
    const auto multiple_tool_calls =
        R"(<|tool_call>call:get_weather{)"
        R"(location:<|"|>Seoul<|"|>)"
        R"(}<tool_call|>)"
        R"(<|tool_call>call:get_weather{)"
        R"(location:<|"|>Henry's Altar<|"|>)"
        R"(}<tool_call|>)";

    return accepts_complete_text(grammar.get(), fixture, complex_tool_call)
        && accepts_complete_text(grammar.get(), fixture, complex_tool_call_with_string_whitespace)
        && accepts_complete_text(grammar.get(), fixture, get_weather_call)
        && accepts_complete_text(grammar.get(), fixture, multiple_tool_calls);
}

static bool test_model_tools_functiongemma_accepts_valid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("functiongemma", test_model_tools_json().c_str()));

    const auto get_weather_call =
        R"(<start_function_call>call:get_weather{)"
        R"(location:<escape>Seoul<escape>)"
        R"(}<end_function_call>)";
    const auto multiple_tool_calls =
        R"(<start_function_call>call:get_weather{)"
        R"(location:<escape>Seoul<escape>)"
        R"(}<end_function_call>)"
        R"(<start_function_call>call:get_weather{)"
        R"(location:<escape>Henry's Altar<escape>)"
        R"(}<end_function_call>)";

    return accepts_complete_text(grammar.get(), fixture, get_weather_call)
        && accepts_complete_text(grammar.get(), fixture, multiple_tool_calls);
}

static bool test_model_tools_qwen_accepts_valid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("qwen3p5", test_model_tools_json().c_str()));

    const auto complex_tool_call =
        R"(<tool_call>
{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1,"BETA_LABEL":2},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]})"
        R"(}}
</tool_call>)";
    const auto get_weather_call =
        R"(<tool_call>
{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Seoul"})"
        R"(}
</tool_call>)";

    return accepts_complete_text(grammar.get(), fixture, complex_tool_call)
        && accepts_complete_text(grammar.get(), fixture, get_weather_call);
}

static bool test_model_tools_qwen_accepts_simple_call(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("qwen3p5", test_model_tools_json().c_str()));

    const auto get_weather_call =
        R"(<tool_call>
{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Seoul"})"
        R"(}
</tool_call>)";

    return accepts_complete_text(grammar.get(), fixture, get_weather_call);
}

static bool test_model_tools_qwen_accepts_multiple_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("qwen3p5", test_model_tools_json().c_str()));

    const auto multiple_tool_calls =
        R"(<tool_call>
{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Seoul"})"
        R"(}
</tool_call>
<tool_call>
{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Henry's Cave"})"
        R"(}
</tool_call>)";

    return accepts_complete_text(grammar.get(), fixture, multiple_tool_calls);
}

static bool test_model_tools_needle_accepts_valid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("needle", test_model_tools_json().c_str()));

    const auto complex_tool_call =
        R"(<tool_call>[{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1,"BETA_LABEL":2},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]})"
        R"(}}])";
    const auto get_weather_call =
        R"(<tool_call>[{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Seoul"})"
        R"(}])";
    const auto multiple_tool_calls =
        R"(<tool_call>[{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Seoul"})"
        R"(},{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Henry's Cave"})"
        R"(}])";

    return accepts_complete_text(grammar.get(), fixture, complex_tool_call)
        && accepts_complete_text(grammar.get(), fixture, get_weather_call)
        && accepts_complete_text(grammar.get(), fixture, multiple_tool_calls);
}

static bool test_model_tools_lfm2_accepts_valid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("lfm2", test_model_tools_json().c_str()));

    const auto complex_tool_call =
        R"(<|tool_call_start|>[)"
        R"(complex_tool()"
        R"(title="alpha",)"
        R"(count=3.5,)"
        R"(enabled=True,)"
        R"(mode="execute",)"
        R"(ticket_id="ABC-12",)"
        R"(priority=4,)"
        R"(routing={"region":"us-west"},)"
        R"(labels={"ALPHA":1,"BETA_LABEL":2},)"
        R"(window=3,)"
        R"(tuple_args=["alpha",2,True],)"
        R"(optional_note=None,)"
        R"(tags=["a","b"],)"
        R"(config={"threshold":0.75,"flags":[True,False]})"
        R"())"
        R"(]<|tool_call_end|>)";
    const auto get_weather_call =
        R"(<|tool_call_start|>[)"
        R"(get_weather(location="Seoul"))"
        R"(]<|tool_call_end|>)";
    const auto multiple_tool_calls =
        R"(<|tool_call_start|>[)"
        R"(get_weather(location="Seoul"),)"
        R"(get_weather(location="Henry's Cave"))"
        R"(]<|tool_call_end|>)";

    return accepts_complete_text(grammar.get(), fixture, complex_tool_call)
        && accepts_complete_text(grammar.get(), fixture, get_weather_call)
        && accepts_complete_text(grammar.get(), fixture, multiple_tool_calls);
}

static bool test_model_tools_gemma4_rejects_invalid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("gemma4", test_model_tools_json().c_str()));

    const auto missing_required_field =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>])"
        R"(}<tool_call|>)";
    const auto wrong_type =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:<|"|>3.5<|"|>,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_enum =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>preview<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_any_of =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:false,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_one_of =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{zone:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_pattern_properties =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{alpha:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_regex_string =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>abc-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_all_of =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:0,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_prefix_items_type =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,<|"|>2<|"|>,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto invalid_prefix_items_length =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true,<|"|>extra<|"|>],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(}<tool_call|>)";
    const auto extra_property =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]},)"
        R"(extra:1)"
        R"(}<tool_call|>)";
    const auto malformed_arguments =
        R"(<|tool_call>call:complex_tool{)"
        R"(title:<|"|>alpha<|"|>,)"
        R"(count:3.5,)"
        R"(enabled:true,)"
        R"(mode:<|"|>execute<|"|>,)"
        R"(ticket_id:<|"|>ABC-12<|"|>,)"
        R"(priority:4,)"
        R"(routing:{region:<|"|>us-west<|"|>},)"
        R"(labels:{ALPHA:1},)"
        R"(window:3,)"
        R"(tuple_args:[<|"|>alpha<|"|>,2,true],)"
        R"(optional_note:null,)"
        R"(tags:[<|"|>a<|"|>,<|"|>b<|"|>],)"
        R"(config:{threshold:0.75,flags:[true,false]})"
        R"(<tool_call|>)";

    return rejects_text(grammar.get(), fixture, missing_required_field)
        && rejects_text(grammar.get(), fixture, wrong_type)
        && rejects_text(grammar.get(), fixture, invalid_enum)
        && rejects_text(grammar.get(), fixture, invalid_any_of)
        && rejects_text(grammar.get(), fixture, invalid_one_of)
        && rejects_text(grammar.get(), fixture, invalid_pattern_properties)
        && rejects_text(grammar.get(), fixture, invalid_regex_string)
        && rejects_text(grammar.get(), fixture, invalid_all_of)
        && rejects_text(grammar.get(), fixture, invalid_prefix_items_type)
        && rejects_text(grammar.get(), fixture, invalid_prefix_items_length)
        && rejects_text(grammar.get(), fixture, extra_property)
        && rejects_text(grammar.get(), fixture, malformed_arguments);
}

static bool test_model_tools_qwen_rejects_invalid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("qwen3p5", test_model_tools_json().c_str()));

    const auto missing_required_field =
        R"(<tool_call>
{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"])"
        R"(}}
</tool_call>)";
    const auto wrong_type =
        R"(<tool_call>
{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":"3.5",)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]})"
        R"(}}
</tool_call>)";
    const auto extra_property =
        R"(<tool_call>
{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]},)"
        R"("extra":1)"
        R"(}}
</tool_call>)";
    const auto malformed_json =
        R"(<tool_call>
{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]})"
        R"(
</tool_call>)";
    const auto malformed_tag =
        R"(<tool_call>{)"
        R"("name":"get_weather",)"
        R"("arguments":{"location":"Seoul"})"
        R"(}</tool_call>)";

    return rejects_text(grammar.get(), fixture, missing_required_field)
        && rejects_text(grammar.get(), fixture, wrong_type)
        && rejects_text(grammar.get(), fixture, extra_property)
        && rejects_text(grammar.get(), fixture, malformed_json)
        && rejects_text(grammar.get(), fixture, malformed_tag);
}

static bool test_model_tools_lfm2_rejects_invalid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("lfm2", test_model_tools_json().c_str()));

    const auto missing_required_field =
        R"(<|tool_call_start|>[)"
        R"(complex_tool()"
        R"(title="alpha",)"
        R"(count=3.5,)"
        R"(enabled=True,)"
        R"(mode="execute",)"
        R"(ticket_id="ABC-12",)"
        R"(priority=4,)"
        R"(routing={"region":"us-west"},)"
        R"(labels={"ALPHA":1},)"
        R"(window=3,)"
        R"(tuple_args=["alpha",2,True],)"
        R"(optional_note=None,)"
        R"(tags=["a","b"])"
        R"())"
        R"(]<|tool_call_end|>)";
    const auto wrong_type =
        R"(<|tool_call_start|>[)"
        R"(complex_tool()"
        R"(title="alpha",)"
        R"(count="3.5",)"
        R"(enabled=True,)"
        R"(mode="execute",)"
        R"(ticket_id="ABC-12",)"
        R"(priority=4,)"
        R"(routing={"region":"us-west"},)"
        R"(labels={"ALPHA":1},)"
        R"(window=3,)"
        R"(tuple_args=["alpha",2,True],)"
        R"(optional_note=None,)"
        R"(tags=["a","b"],)"
        R"(config={"threshold":0.75,"flags":[True,False]})"
        R"())"
        R"(]<|tool_call_end|>)";
    const auto invalid_enum =
        R"(<|tool_call_start|>[)"
        R"(complex_tool()"
        R"(title="alpha",)"
        R"(count=3.5,)"
        R"(enabled=True,)"
        R"(mode="preview",)"
        R"(ticket_id="ABC-12",)"
        R"(priority=4,)"
        R"(routing={"region":"us-west"},)"
        R"(labels={"ALPHA":1},)"
        R"(window=3,)"
        R"(tuple_args=["alpha",2,True],)"
        R"(optional_note=None,)"
        R"(tags=["a","b"],)"
        R"(config={"threshold":0.75,"flags":[True,False]})"
        R"())"
        R"(]<|tool_call_end|>)";
    const auto invalid_pattern_properties =
        R"(<|tool_call_start|>[)"
        R"(complex_tool()"
        R"(title="alpha",)"
        R"(count=3.5,)"
        R"(enabled=True,)"
        R"(mode="execute",)"
        R"(ticket_id="ABC-12",)"
        R"(priority=4,)"
        R"(routing={"region":"us-west"},)"
        R"(labels={"alpha":1},)"
        R"(window=3,)"
        R"(tuple_args=["alpha",2,True],)"
        R"(optional_note=None,)"
        R"(tags=["a","b"],)"
        R"(config={"threshold":0.75,"flags":[True,False]})"
        R"())"
        R"(]<|tool_call_end|>)";
    const auto extra_property =
        R"(<|tool_call_start|>[)"
        R"(complex_tool()"
        R"(title="alpha",)"
        R"(count=3.5,)"
        R"(enabled=True,)"
        R"(mode="execute",)"
        R"(ticket_id="ABC-12",)"
        R"(priority=4,)"
        R"(routing={"region":"us-west"},)"
        R"(labels={"ALPHA":1},)"
        R"(window=3,)"
        R"(tuple_args=["alpha",2,True],)"
        R"(optional_note=None,)"
        R"(tags=["a","b"],)"
        R"(config={"threshold":0.75,"flags":[True,False]},)"
        R"(extra=1)"
        R"())"
        R"(]<|tool_call_end|>)";
    const auto malformed_call =
        R"(<|tool_call_start|>[)"
        R"(complex_tool()"
        R"(title="alpha",)"
        R"(count=3.5,)"
        R"(enabled=True,)"
        R"(mode="execute",)"
        R"(ticket_id="ABC-12",)"
        R"(priority=4,)"
        R"(routing={"region":"us-west"},)"
        R"(labels={"ALPHA":1},)"
        R"(window=3,)"
        R"(tuple_args=["alpha",2,True],)"
        R"(optional_note=None,)"
        R"(tags=["a","b"],)"
        R"(config={"threshold":0.75,"flags":[True,False]})"
        R"())"
        R"(<|tool_call_end|>)";

    return rejects_text(grammar.get(), fixture, missing_required_field)
        && rejects_text(grammar.get(), fixture, wrong_type)
        && rejects_text(grammar.get(), fixture, invalid_enum)
        && rejects_text(grammar.get(), fixture, invalid_pattern_properties)
        && rejects_text(grammar.get(), fixture, extra_property)
        && rejects_text(grammar.get(), fixture, malformed_call);
}

static bool test_model_tools_needle_rejects_invalid_calls(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("needle", test_model_tools_json().c_str()));

    const auto missing_required_field =
        R"(<tool_call>[{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"])"
        R"(}}])";
    const auto wrong_type =
        R"(<tool_call>[{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":"3.5",)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]})"
        R"(}}])";
    const auto invalid_pattern_properties =
        R"(<tool_call>[{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"alpha":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]})"
        R"(}}])";
    const auto extra_property =
        R"(<tool_call>[{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]},)"
        R"("extra":1)"
        R"(}}])";
    const auto malformed_json =
        R"(<tool_call>[{)"
        R"("name":"complex_tool",)"
        R"("arguments":{)"
        R"("title":"alpha",)"
        R"("count":3.5,)"
        R"("enabled":true,)"
        R"("mode":"execute",)"
        R"("ticket_id":"ABC-12",)"
        R"("priority":4,)"
        R"("routing":{"region":"us-west"},)"
        R"("labels":{"ALPHA":1},)"
        R"("window":3,)"
        R"("tuple_args":["alpha",2,true],)"
        R"("optional_note":null,)"
        R"("tags":["a","b"],)"
        R"("config":{"threshold":0.75,"flags":[true,false]})"
        R"(}])";

    return rejects_text(grammar.get(), fixture, missing_required_field)
        && rejects_text(grammar.get(), fixture, wrong_type)
        && rejects_text(grammar.get(), fixture, invalid_pattern_properties)
        && rejects_text(grammar.get(), fixture, extra_property)
        && rejects_text(grammar.get(), fixture, malformed_json);
}

static bool test_model_tools_gemma4_rejects_invalid_tool_name(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("gemma4", test_model_tools_json().c_str()));

    const auto invalid_tool_name =
        R"(<|tool_call>call:not_a_real_tool{)"
        R"(location:<|"|>Seoul<|"|>)"
        R"(}<tool_call|>)";

    return rejects_text(grammar.get(), fixture, invalid_tool_name);
}

static bool test_model_tools_qwen_rejects_invalid_tool_name(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("qwen3p5", test_model_tools_json().c_str()));

    const auto invalid_tool_name =
        R"(<tool_call>
{)"
        R"("name":"not_a_real_tool",)"
        R"("arguments":{"location":"Seoul"})"
        R"(}
</tool_call>)";

    return rejects_text(grammar.get(), fixture, invalid_tool_name);
}

static bool test_model_tools_needle_rejects_invalid_tool_name(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("needle", test_model_tools_json().c_str()));

    const auto invalid_tool_name =
        R"(<tool_call>[{)"
        R"("name":"not_a_real_tool",)"
        R"("arguments":{"location":"Seoul"})"
        R"(}])";

    return rejects_text(grammar.get(), fixture, invalid_tool_name);
}

static bool test_model_tools_lfm2_rejects_invalid_tool_name(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_tools("lfm2", test_model_tools_json().c_str()));

    const auto invalid_tool_name =
        R"(<|tool_call_start|>[)"
        R"(not_a_real_tool(location="Seoul"))"
        R"(]<|tool_call_end|>)";

    return rejects_text(grammar.get(), fixture, invalid_tool_name);
}

static bool test_regex_accepts_expected_text(const GrammarFixture& fixture) {
    auto regex = grammar_handle(cactus_grammar_init_regex("(cat|dog)"));

    return accepts_complete_text(regex.get(), fixture, "cat")
        && accepts_complete_text(regex.get(), fixture, "dog")
        && rejects_text(regex.get(), fixture, "cow");
}

static bool test_json_schema_accepts_expected_text(const GrammarFixture& fixture) {
    auto json_schema = grammar_handle(json_schema_grammar(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    ));

    return accepts_complete_text(json_schema.get(), fixture, R"({"name":"cactus"})")
        && rejects_text(json_schema.get(), fixture, R"({"age":1})");
}

static bool test_model_thinking_unsupported_types_return_empty_grammar() {
    auto unknown = grammar_handle(cactus_grammar_init_model_thinking("i made this up"));
    return cactus_grammar_is_empty(unknown.get());
}

static bool test_model_thinking_gemma4_uses_channel_tags(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_thinking("gemma4"));

    return !cactus_grammar_is_empty(grammar.get())
        && accepts_complete_text(grammar.get(), fixture, "<|channel>blah blah blah<channel|>")
        && rejects_text(grammar.get(), fixture, "<|channel>blah blah blah<channel|> more <channel|>")
        && rejects_eos_after_text(grammar.get(), fixture, "<|channel>blah blah blah");
}

static bool test_model_thinking_qwen_uses_think_tags(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_thinking("qwen"));

    return !cactus_grammar_is_empty(grammar.get())
        && accepts_complete_text(grammar.get(), fixture, "<think>blah blah blah</think>")
        && rejects_text(grammar.get(), fixture, "<think>blah blah blah</think> more </think>")
        && rejects_eos_after_text(grammar.get(), fixture, "<think>blah blah blah");
}

static bool test_model_thinking_lfm2_uses_think_tags(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_model_thinking("lfm2"));

    return !cactus_grammar_is_empty(grammar.get())
        && accepts_complete_text(grammar.get(), fixture, "<think>blah blah blah</think>")
        && rejects_text(grammar.get(), fixture, "<think>blah blah blah</think> more </think>")
        && rejects_eos_after_text(grammar.get(), fixture, "<think>blah blah blah");
}

static bool test_grammar_matcher_reset_restores_initial_state(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));

    if (!accept_text(matcher.get(), fixture, "hello")) return false;
    if (!cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token())) return false;

    cactus_grammar_matcher_reset(matcher.get());
    return rejects_text(grammar.get(), fixture, "goodbye")
        && accept_text(matcher.get(), fixture, "hello")
        && cactus_grammar_matcher_accept(matcher.get(), fixture.tokenizer->get_eos_token());
}

static bool test_grammar_matcher_rollback_restores_previous_state(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\" | \"hi\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    const std::vector<uint32_t> hello_tokens = fixture.tokenizer->encode("hello");
    const std::vector<uint32_t> hi_tokens = fixture.tokenizer->encode("hi");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    for (uint32_t token : hello_tokens) {
        if (!cactus_grammar_matcher_accept(matcher.get(), token)) return false;
    }
    if (!cactus_grammar_matcher_accept(matcher.get(), eos_token)) return false;

    cactus_grammar_matcher_rollback(matcher.get(), static_cast<int>(hello_tokens.size()) + 1);

    for (uint32_t token : hi_tokens) {
        if (!cactus_grammar_matcher_accept(matcher.get(), token)) return false;
    }
    return cactus_grammar_matcher_accept(matcher.get(), eos_token);
}

static bool test_grammar_matcher_completion_state(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (cactus_grammar_matcher_is_completed(matcher.get()) || cactus_grammar_matcher_is_terminated(matcher.get())) return false;
    if (!accept_text(matcher.get(), fixture, "hello")) return false;
    if (!cactus_grammar_matcher_is_completed(matcher.get()) || cactus_grammar_matcher_is_terminated(matcher.get())) return false;
    if (!cactus_grammar_matcher_accept(matcher.get(), eos_token)) return false;
    return cactus_grammar_matcher_is_completed(matcher.get()) && cactus_grammar_matcher_is_terminated(matcher.get());
}

static bool test_grammar_matcher_fork_preserves_accept_state(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello world\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    const std::vector<uint32_t> tokens = fixture.tokenizer->encode("hello world");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (tokens.size() < 2) return false;

    const size_t fork_point = tokens.size() / 2;

    for (size_t i = 0; i < fork_point; ++i) {
        if (!cactus_grammar_matcher_accept(matcher.get(), tokens[i])) return false;
    }

    auto forked = matcher_handle(cactus_grammar_matcher_fork(matcher.get()));
    if (!forked) return false;

    for (size_t i = fork_point; i < tokens.size(); ++i) {
        if (!cactus_grammar_matcher_accept(matcher.get(), tokens[i])) return false;
    }
    if (!cactus_grammar_matcher_accept(matcher.get(), eos_token)) return false;

    for (size_t i = fork_point; i < tokens.size(); ++i) {
        if (!cactus_grammar_matcher_accept(forked.get(), tokens[i])) return false;
    }
    return cactus_grammar_matcher_accept(forked.get(), eos_token);
}

static bool test_grammar_matcher_get_grammar_round_trips_source(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello world\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    auto retrieved = grammar_handle(cactus_grammar_matcher_get_grammar(matcher.get()));
    if (!retrieved) return false;

    return grammar_ebnf(retrieved.get()) == grammar_ebnf(grammar.get());
}

static bool test_grammar_matcher_fork_preserves_source_grammar(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello world\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    auto forked = matcher_handle(cactus_grammar_matcher_fork(matcher.get()));
    if (!forked) return false;

    auto retrieved = grammar_handle(cactus_grammar_matcher_get_grammar(forked.get()));
    if (!retrieved) return false;

    return grammar_ebnf(retrieved.get()) == grammar_ebnf(grammar.get());
}

static bool test_grammar_matcher_next_bitmask_tracks_simple_grammar(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    const std::vector<uint32_t> hello_tokens = fixture.tokenizer->encode("hello");
    const uint32_t eos_token = fixture.tokenizer->get_eos_token();

    if (hello_tokens.empty()) return false;

    std::vector<int32_t> bitmask(bitmask_size(fixture.vocab_size));
    if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) != 1) return false;
    if (!bitmask_allows_token(bitmask, hello_tokens.front())) return false;
    if (bitmask_allows_token(bitmask, eos_token)) return false;

    for (size_t i = 0; i < hello_tokens.size(); ++i) {
        if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) != 1) return false;
        if (!bitmask_allows_token(bitmask, hello_tokens[i])) return false;
        if (!cactus_grammar_matcher_accept(matcher.get(), hello_tokens[i])) return false;
    }

    if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size) != 1) return false;
    if (!bitmask_allows_token(bitmask, eos_token)) return false;
    return !bitmask_allows_token(bitmask, hello_tokens.front());
}

static bool test_grammar_matcher_next_bitmask_zeroes_overallocated_tail(const GrammarFixture& fixture) {
    auto grammar = grammar_handle(cactus_grammar_init_ebnf("root ::= \"hello\"", "root"));
    auto matcher = matcher_handle(make_matcher(grammar.get(), fixture.engine.get()));
    std::vector<int32_t> bitmask(bitmask_size(fixture.vocab_size + 1), -1);

    if (cactus_grammar_matcher_next_bitmask(matcher.get(), bitmask.data(), fixture.vocab_size + 1) != 1) return false;
    return (bitmask.back() & 0xFF000000u) == 0;
}

} // anonymous namespace

int main() {
    TestUtils::TestRunner runner("Grammar Tests");

    try {
        GrammarFixture fixture;
        runner.run_test("vocab_accessors", test_vocab_accessors(fixture));
        runner.run_test("empty_properties", test_empty_grammar_properties());
        runner.run_test("epsilon_language", test_epsilon_grammar_accepts_only_empty_string(fixture));
        runner.run_test("optional_language", test_optional_grammar_accepts_zero_or_one_occurrence(fixture));
        runner.run_test("optional_empty_stays_empty", test_optional_empty_grammar_stays_empty());
        runner.run_test("star_language", test_star_grammar_accepts_zero_or_more_occurrences(fixture));
        runner.run_test("star_empty_stays_empty", test_star_empty_grammar_stays_empty());
        runner.run_test("repeat_exact_language", test_repeat_exact_language(fixture));
        runner.run_test("repeat_range_language", test_repeat_range_language(fixture));
        runner.run_test("repeat_range_unbounded_language", test_repeat_range_unbounded_language(fixture));
        runner.run_test("repeat_empty_stays_empty", test_repeat_empty_grammar_stays_empty());
        runner.run_test("ebnf_export_matches", test_ebnf_string_export_matches_parenthesized_input_ebnf());
        runner.run_test("regex_json_schema_init", test_regex_and_json_schema_construction());
        runner.run_test("concat_language", test_concat_accepts_expected_language(fixture));
        runner.run_test("union_language", test_union_accepts_expected_language(fixture));
        runner.run_test("three_way_concat", test_three_way_concat(fixture));
        runner.run_test("three_way_union", test_three_way_union(fixture));
        runner.run_test("unordred_choice", test_unordered_choice(fixture));
        runner.run_test("regex_language", test_regex_accepts_expected_text(fixture));
        runner.run_test("json_schema_language", test_json_schema_accepts_expected_text(fixture));
        runner.run_test("model_thinking_unsupported_empty", test_model_thinking_unsupported_types_return_empty_grammar());
        runner.run_test("model_thinking_gemma4", test_model_thinking_gemma4_uses_channel_tags(fixture));
        runner.run_test("model_thinking_qwen", test_model_thinking_qwen_uses_think_tags(fixture));
        runner.run_test("model_thinking_lfm2", test_model_thinking_lfm2_uses_think_tags(fixture));
        runner.run_test("universal", test_universal_grammar_accepts_anything(fixture));
        runner.run_test("structural_tag_language", test_structural_tag_accepts_and_rejects_expected_text(fixture));
        runner.run_test("model_tools_gemma4_valid", test_model_tools_gemma4_accepts_valid_calls(fixture));
        runner.run_test("model_tools_functiongemma_valid", test_model_tools_functiongemma_accepts_valid_calls(fixture));
        runner.run_test("model_tools_qwen_valid", test_model_tools_qwen_accepts_valid_calls(fixture));
        runner.run_test("model_tools_qwen_simple", test_model_tools_qwen_accepts_simple_call(fixture));
        runner.run_test("model_tools_qwen_multiple", test_model_tools_qwen_accepts_multiple_calls(fixture));
        runner.run_test("model_tools_needle_valid", test_model_tools_needle_accepts_valid_calls(fixture));
        runner.run_test("model_tools_lfm2_valid", test_model_tools_lfm2_accepts_valid_calls(fixture));
        runner.run_test("model_tools_gemma4_invalid", test_model_tools_gemma4_rejects_invalid_calls(fixture));
        runner.run_test("model_tools_gemma4_invalid_tool_name", test_model_tools_gemma4_rejects_invalid_tool_name(fixture));
        runner.run_test("model_tools_qwen_invalid", test_model_tools_qwen_rejects_invalid_calls(fixture));
        runner.run_test("model_tools_qwen_invalid_tool_name", test_model_tools_qwen_rejects_invalid_tool_name(fixture));
        runner.run_test("model_tools_needle_invalid", test_model_tools_needle_rejects_invalid_calls(fixture));
        runner.run_test("model_tools_needle_invalid_tool_name", test_model_tools_needle_rejects_invalid_tool_name(fixture));
        runner.run_test("model_tools_lfm2_invalid", test_model_tools_lfm2_rejects_invalid_calls(fixture));
        runner.run_test("model_tools_lfm2_invalid_tool_name", test_model_tools_lfm2_rejects_invalid_tool_name(fixture));
        runner.run_test("grammar_matcher_reset", test_grammar_matcher_reset_restores_initial_state(fixture));
        runner.run_test("grammar_matcher_rollback", test_grammar_matcher_rollback_restores_previous_state(fixture));
        runner.run_test("grammar_matcher_completion_state", test_grammar_matcher_completion_state(fixture));
        runner.run_test("grammar_matcher_fork", test_grammar_matcher_fork_preserves_accept_state(fixture));
        runner.run_test("grammar_matcher_get_grammar", test_grammar_matcher_get_grammar_round_trips_source(fixture));
        runner.run_test("grammar_matcher_fork_preserves_grammar", test_grammar_matcher_fork_preserves_source_grammar(fixture));
        runner.run_test("grammar_matcher_next_bitmask", test_grammar_matcher_next_bitmask_tracks_simple_grammar(fixture));
        runner.run_test("grammar_matcher_next_bitmask_overallocated_tail", test_grammar_matcher_next_bitmask_zeroes_overallocated_tail(fixture));
    } catch (const std::exception& e) {
        std::cerr << "[✗] Grammar test setup failed: " << e.what() << "\n";
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
