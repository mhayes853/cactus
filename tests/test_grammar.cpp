#include "test_utils.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>

using cactus::engine::Config;
using cactus::engine::Grammar;
using cactus::engine::GrammarMatcher;
using cactus::engine::ToolDefinition;
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
        if (!matcher.accept(token, false)) {
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
    return matcher.accept(fixture.tokenizer->get_eos_token(), false);
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
    return !matcher.accept(fixture.tokenizer->get_eos_token(), false);
}

static bool test_empty_grammar_properties() {
    Grammar empty;
    Grammar empty2;
    Grammar simple = Grammar::gbnf("root ::= \"hello\"");

    if (!empty.is_empty() || !empty2.is_empty() || simple.is_empty()) {
        return false;
    }

    if (empty.is_universal() || empty2.is_universal() || simple.is_universal()) {
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

static bool test_universal_grammar_accepts_anything(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::universal();
    return grammar.is_universal()
        && accepts_complete_text(grammar, fixture, "")
        && accepts_complete_text(grammar, fixture, "blob says hello from cactus")
        && accepts_complete_text(grammar, fixture, "line one\nline two\nline three");
}

static bool test_union_with_universal_returns_universal(const GrammarFixture& fixture) {
    Grammar universal = Grammar::universal();
    Grammar specific = Grammar::gbnf("root ::= \"hello\"");
    return Grammar::unite({specific, universal}).is_universal();
}

static bool test_concat_with_leading_universal_returns_universal(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::concatenate({
        Grammar(),
        Grammar::universal(),
        Grammar::gbnf("root ::= \"hello\"")
    });

    return grammar.is_universal()
        && accepts_complete_text(grammar, fixture, "")
        && accepts_complete_text(grammar, fixture, "anything goes here");
}

static bool test_concat_ignores_grammars_after_universal(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::concatenate({
        Grammar::gbnf("root ::= \"hello\"") ,
        Grammar::universal(),
        Grammar::gbnf("root ::= \" world\"")
    });

    return accepts_complete_text(grammar, fixture, "hello")
        && accepts_complete_text(grammar, fixture, "hello there")
        && accepts_complete_text(grammar, fixture, "hello world")
        && rejects_text(grammar, fixture, "goodbye");
}

static std::string tool_call_structural_tag_json() {
    return R"({
        "type": "structural_tag",
        "format": {
            "type": "triggered_tags",
            "triggers": ["<｜tool▁calls▁begin｜>"],
            "tags": [
                {
                    "begin": "<｜tool▁calls▁begin｜>",
                    "end": "<｜tool▁calls▁end｜>",
                    "content": {
                        "type": "tags_with_separator",
                        "separator": "\n",
                        "tags": [
                            {
                                "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n",
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
                                "end": "\n```<｜tool▁call▁end｜>"
                            },
                            {
                                "begin": "<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n",
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
                                "end": "\n```<｜tool▁call▁end｜>"
                            }
                        ]
                    }
                }
            ],
            "stop_after_first": true
        }
    })";
}

const static std::string tools_json = R"([
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["recipient", "message"],
                "additionalProperties": false
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
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

static bool test_tool_definition_parse_tools_json() {
    std::vector<ToolDefinition> tools = ToolDefinition::parse_tools_json(tools_json);
    return tools.size() == 2
        && tools[0].name == "send_message"
        && tools[0].arguments_schema.is<picojson::object>()
        && tools[1].name == "get_weather"
        && tools[1].arguments_schema.is<picojson::object>();
}

static bool test_tool_definition_rejects_invalid_tools_json() {
    bool rejected_missing_type = false;
    try {
        ToolDefinition::parse_tools_json(R"([{"function":{"name":"send_message","parameters":{"type":"object"}}}])");
    } catch (const std::runtime_error&) {
        rejected_missing_type = true;
    }

    bool rejected_wrong_type = false;
    try {
        ToolDefinition::parse_tools_json(R"([{"type":"not_function","function":{"name":"send_message","parameters":{"type":"object"}}}])");
    } catch (const std::runtime_error&) {
        rejected_wrong_type = true;
    }

    bool rejected_missing_function = false;
    try {
        ToolDefinition::parse_tools_json(R"([{"type":"function"}])");
    } catch (const std::runtime_error&) {
        rejected_missing_function = true;
    }

    bool rejected_missing_name = false;
    try {
        ToolDefinition::parse_tools_json(R"([{"type":"function","function":{"parameters":{"type":"object"}}}])");
    } catch (const std::runtime_error&) {
        rejected_missing_name = true;
    }

    bool rejected_missing_parameters = false;
    try {
        ToolDefinition::parse_tools_json(R"([{"type":"function","function":{"name":"send_message"}}])");
    } catch (const std::runtime_error&) {
        rejected_missing_parameters = true;
    }

    return rejected_missing_type
        && rejected_wrong_type
        && rejected_missing_function
        && rejected_missing_name
        && rejected_missing_parameters;
}

static Grammar tool_call_grammar_from_tools_json(Config::ModelType model_type) {
    std::vector<ToolDefinition> tools = ToolDefinition::parse_tools_json(tools_json);
    return Grammar::model_decode_grammar(Grammar(), true, false, model_type, tools);
}

static bool test_qwen_style_tool_call_accepts_single_tool_call(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::QWEN);

    return accepts_complete_text(
        grammar,
        fixture,
        "<tool_call>\n{\"name\": \"send_message\", \"arguments\": {\"recipient\":\"Blob\",\"message\":\"Hello Blob!\"}}\n</tool_call>"
    );
}

static bool test_qwen_style_tool_call_accepts_repeated_tool_calls(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::QWEN);

    return accepts_complete_text(
        grammar,
        fixture,
        "<tool_call>\n{\"name\": \"send_message\", \"arguments\": {\"recipient\":\"Blob\",\"message\":\"Hello Blob!\"}}\n</tool_call><tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"San Francisco\"}}\n</tool_call>"
    );
}

static bool test_qwen_style_tool_call_rejects_unknown_tool(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::QWEN);

    return rejects_text(
        grammar,
        fixture,
        "<tool_call>\n{\"name\": \"unknown_tool\", \"arguments\": {\"recipient\":\"Blob\",\"message\":\"Hello Blob!\"}}\n</tool_call>"
    );
}

static bool test_qwen_style_tool_call_rejects_invalid_arguments(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::QWEN);

    return rejects_eos_after_text(
        grammar,
        fixture,
        "<tool_call>\n{\"name\": \"send_message\", \"arguments\": {\"recipient\":\"Blob\",\"message\":1}}\n</tool_call>"
    );
}

static bool test_qwen_style_tool_call_rejects_malformed_wrapper(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::QWEN);

    return rejects_eos_after_text(
        grammar,
        fixture,
        "<tool_call>\n{\"name\": \"send_message\", \"arguments\": {\"recipient\":\"Blob\",\"message\":\"Hello Blob!\"}}"
    );
}

static bool test_lfm2_style_tool_call_accepts_single_tool_call(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::LFM2);

    return accepts_complete_text(
        grammar,
        fixture,
        "<|tool_call_start|>[send_message(recipient=\"Blob\",message=\"Hello Blob!\")]<|tool_call_end|>"
    );
}

static bool test_lfm2_style_tool_call_accepts_repeated_tool_calls(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::LFM2);

    return accepts_complete_text(
        grammar,
        fixture,
        "<|tool_call_start|>[send_message(recipient=\"Blob\",message=\"Hello Blob!\"),get_weather(location=\"San Francisco\")]<|tool_call_end|>"
    );
}

static bool test_lfm2_style_tool_call_accepts_pythonic_literals(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::LFM2);

    return accepts_complete_text(
        grammar,
        fixture,
        "<|tool_call_start|>[send_message(recipient=\"Blob\",message={attempts:2,urgent:True,aliases:[\"A\",\"B\"],metadata:{active:False,score:-1.25e2,note:null}})]<|tool_call_end|>"
    );
}

static bool test_lfm2_style_tool_call_rejects_unknown_tool(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::LFM2);

    return rejects_text(
        grammar,
        fixture,
        "<|tool_call_start|>[unknown_tool(recipient=\"Blob\")]<|tool_call_end|>"
    );
}

static bool test_lfm2_style_tool_call_rejects_malformed_wrapper(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::LFM2);

    return rejects_eos_after_text(
        grammar,
        fixture,
        "<|tool_call_start|>[send_message(recipient=\"Blob\",message=\"Hello Blob!\")]"
    );
}

static bool test_lfm2_style_tool_call_rejects_malformed_literal(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::LFM2);

    return rejects_text(
        grammar,
        fixture,
        "<|tool_call_start|>[send_message(recipient=\"Blob\",message={text:})]<|tool_call_end|>"
    );
}

static bool test_gemma_style_tool_call_accepts_single_tool_call(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA);

    return accepts_complete_text(
        grammar,
        fixture,
        "<start_function_call>call:send_message{recipient:<escape>Blob<escape>,message:<escape>Hello Blob!<escape>}<end_function_call>"
    );
}

static bool test_gemma_style_tool_call_accepts_single_tool_call_with_pipe_tags(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA4);

    return accepts_complete_text(
        grammar,
        fixture,
        "<|tool_call>call:send_message{recipient:<escape>Blob<escape>,message:<escape>Hello Blob!<escape>}<tool_call|>"
    );
}

static bool test_gemma_style_tool_call_accepts_repeated_tool_calls(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA);

    return accepts_complete_text(
        grammar,
        fixture,
        "<start_function_call>call:send_message{recipient:<escape>Blob<escape>,message:<escape>Hello Blob!<escape>}<end_function_call><start_function_call>call:get_weather{location:<escape>San Francisco<escape>}<end_function_call>"
    );
}

static bool test_gemma_style_tool_call_accepts_mixed_value_types(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA);

    return accepts_complete_text(
        grammar,
        fixture,
        "<start_function_call>call:send_message{recipient:<escape>Blob<escape>,message:{attempts:2,urgent:true,aliases:[1,2],metadata:{active:false,score:-1.25e2,note:null}}}<end_function_call>"
    );
}

static bool test_gemma_style_tool_call_rejects_unknown_tool(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA);

    return rejects_text(
        grammar,
        fixture,
        "<start_function_call>call:unknown_tool{recipient:<escape>Blob<escape>}<end_function_call>"
    );
}

static bool test_gemma_style_tool_call_rejects_malformed_wrapper(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA);

    return rejects_eos_after_text(
        grammar,
        fixture,
        "<start_function_call>call:send_message{recipient:<escape>Blob<escape>,message:<escape>Hello Blob!<escape>}"
    );
}

static bool test_gemma_style_tool_call_rejects_malformed_escaped_string(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA);

    return rejects_eos_after_text(
        grammar,
        fixture,
        "<start_function_call>call:send_message{recipient:<escape>Blob<escape>,message:<escape>Hello Blob!}<end_function_call>"
    );
}

static bool test_gemma_style_tool_call_rejects_malformed_mixed_value_types(const GrammarFixture& fixture) {
    Grammar grammar = tool_call_grammar_from_tools_json(Config::ModelType::GEMMA);

    return rejects_text(
        grammar,
        fixture,
        "<start_function_call>call:send_message{recipient:<escape>Blob<escape>,message:{attempts:2,urgent:true,aliases:[1,],metadata:{active:false,score:-1.25e2,note:null}}}<end_function_call>"
    );
}

static bool test_structural_tag_accepts_and_rejects_expected_text(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::structural_tag(tool_call_structural_tag_json());

    return accepts_complete_text(
        grammar,
        fixture,
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
        && accepts_complete_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜>\n<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n{\"count\":2}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_3\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_eos_after_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":1}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        )
        && rejects_text(
            grammar,
            fixture,
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_1\n```jsonc\n{\"city\":\"Oakland\"}\n```<｜tool▁call▁end｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name_2\n```jsonc\n{\"count\":2}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
        );
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
    Grammar user_grammar = Grammar::universal();
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return accepts_complete_text(combined, fixture, "hello");
}

static bool test_model_decode_accepts_thinking_then_output(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return accepts_complete_text(combined, fixture, "<think>\nreasoning\n</think>\n\nhello");
}

static bool test_model_decode_accepts_less_than_in_thinking_payload(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return accepts_complete_text(combined, fixture, "<think>\n1 < 2\n</think>\n\nhello");
}

static bool test_model_decode_rejects_thinking_only_eos(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return rejects_eos_after_text(combined, fixture, "<think>\nreasoning\n</think>\n\n");
}

static bool test_model_decode_rejects_thinking_when_reasoning_disabled(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, false, Config::ModelType::QWEN, {});

    return rejects_text(combined, fixture, "<think>\nreasoning\n</think>\n\nhello")
        && accepts_complete_text(combined, fixture, "hello");
}

static bool test_model_decode_accepts_thinking_then_json_schema(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    );
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return accepts_complete_text(combined, fixture, "<think>\nreasoning\n</think>\n\n{\"name\":\"cactus\"}");
}

static bool test_model_decode_rejects_invalid_json_schema_after_thinking(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::json_schema(
        R"({"type":"object","properties":{"name":{"type":"string"}},"required":["name"]})"
    );
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return rejects_eos_after_text(combined, fixture, "<think>\nreasoning\n</think>\n\n{\"name\":1}")
        && rejects_text(combined, fixture, "<think>\nreasoning\n</think>\n\nthis is some random text");
}

static bool test_model_decode_rejects_nested_thinking_close_tag(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return rejects_text(combined, fixture, "<think>\nreasoning</think>\n</think>\n\nhello");
}

static bool test_model_decode_rejects_invalid_close_only_thinking_string(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, {});

    return rejects_text(combined, fixture, "</think>\nreasoning\n</think>\nhello");
}

static bool test_model_decode_qwen_with_tools_accepts_tool_call_after_thinking(const GrammarFixture& fixture) {
    std::vector<ToolDefinition> tools = ToolDefinition::parse_tools_json(tools_json);
    Grammar combined = Grammar::model_decode_grammar(Grammar(), true, true, Config::ModelType::QWEN, tools);

    return accepts_complete_text(
        combined,
        fixture,
        "<think>\nreasoning\n</think>\n\n<tool_call>\n{\"name\": \"send_message\", \"arguments\": {\"recipient\":\"Blob\",\"message\":\"Hello Blob!\"}}\n</tool_call>"
    );
}

static bool test_model_decode_qwen_with_tools_allows_plain_text_when_not_forced(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    std::vector<ToolDefinition> tools = ToolDefinition::parse_tools_json(tools_json);
    Grammar combined = Grammar::model_decode_grammar(user_grammar, false, true, Config::ModelType::QWEN, tools);

    return accepts_complete_text(combined, fixture, "hello");
}

static bool test_model_decode_qwen_with_tools_rejects_plain_text_when_forced(const GrammarFixture& fixture) {
    Grammar user_grammar = Grammar::gbnf("root ::= \"hello\"");
    std::vector<ToolDefinition> tools = ToolDefinition::parse_tools_json(tools_json);
    Grammar combined = Grammar::model_decode_grammar(user_grammar, true, true, Config::ModelType::QWEN, tools);

    return rejects_eos_after_text(combined, fixture, "hello");
}

static bool test_model_decode_lfm2_with_tools_accepts_tool_call_after_thinking(const GrammarFixture& fixture) {
    std::vector<ToolDefinition> tools = ToolDefinition::parse_tools_json(tools_json);
    Grammar combined = Grammar::model_decode_grammar(Grammar(), true, true, Config::ModelType::LFM2, tools);

    return accepts_complete_text(
        combined,
        fixture,
        "<think>\nreasoning\n</think>\n\n<|tool_call_start|>[send_message(recipient=\"Blob\",message=\"Hello Blob!\")]<|tool_call_end|>"
    );
}

static bool test_model_decode_gemma_with_tools_accepts_tool_call_after_thinking(const GrammarFixture& fixture) {
    std::vector<ToolDefinition> tools = ToolDefinition::parse_tools_json(tools_json);
    Grammar combined = Grammar::model_decode_grammar(Grammar(), true, true, Config::ModelType::GEMMA, tools);

    return accepts_complete_text(
        combined,
        fixture,
        "<think>\nreasoning\n</think>\n\n<start_function_call>call:send_message{recipient:<escape>Blob<escape>,message:<escape>Hello Blob!<escape>}<end_function_call>"
    );
}

static bool test_grammar_matcher_reset_restores_initial_state(const GrammarFixture& fixture) {
    Grammar grammar = Grammar::gbnf("root ::= \"hello\"");
    GrammarMatcher matcher(&grammar, fixture.tokenizer_info);

    if (!accept_text(matcher, fixture, "hello")) {
        return false;
    }
    if (!matcher.accept(fixture.tokenizer->get_eos_token(), false)) {
        return false;
    }

    matcher.reset();
    return rejects_text(grammar, fixture, "goodbye")
        && accept_text(matcher, fixture, "hello")
        && matcher.accept(fixture.tokenizer->get_eos_token(), false);
}


} // anonymous namespace

int main() {
    TestUtils::TestRunner runner("Grammar Tests");

    runner.run_test("empty_properties", test_empty_grammar_properties());
    runner.run_test("regex_json_schema_init", test_regex_and_json_schema_construction());
    runner.run_test("tool_definition_parse_tools_json", test_tool_definition_parse_tools_json());
    runner.run_test("tool_definition_rejects_invalid_tools_json", test_tool_definition_rejects_invalid_tools_json());

    try {
        GrammarFixture fixture;
        runner.run_test("concat_language", test_concat_accepts_expected_language(fixture));
        runner.run_test("union_language", test_union_accepts_expected_language(fixture));
        runner.run_test("three_way_concat", test_three_way_concat(fixture));
        runner.run_test("three_way_union", test_three_way_union(fixture));
        runner.run_test("unordred_choice", test_unordered_choice(fixture));
        runner.run_test("regex_language", test_regex_accepts_expected_text(fixture));
        runner.run_test("json_schema_language", test_json_schema_accepts_expected_text(fixture));
        runner.run_test("universal", test_universal_grammar_accepts_anything(fixture));
        runner.run_test("union_with_universal_returns_universal", test_union_with_universal_returns_universal(fixture));
        runner.run_test("concat_with_leading_universal_returns_universal", test_concat_with_leading_universal_returns_universal(fixture));
        runner.run_test("concat_ignores_grammars_after_universal", test_concat_ignores_grammars_after_universal(fixture));
        runner.run_test("structural_tag_language", test_structural_tag_accepts_and_rejects_expected_text(fixture));
        runner.run_test("qwen_style_tool_call_single", test_qwen_style_tool_call_accepts_single_tool_call(fixture));
        runner.run_test("qwen_style_tool_call_repeated", test_qwen_style_tool_call_accepts_repeated_tool_calls(fixture));
        runner.run_test("qwen_style_tool_call_unknown_tool", test_qwen_style_tool_call_rejects_unknown_tool(fixture));
        runner.run_test("qwen_style_tool_call_invalid_arguments", test_qwen_style_tool_call_rejects_invalid_arguments(fixture));
        runner.run_test("qwen_style_tool_call_malformed_wrapper", test_qwen_style_tool_call_rejects_malformed_wrapper(fixture));
        runner.run_test("lfm2_style_tool_call_single", test_lfm2_style_tool_call_accepts_single_tool_call(fixture));
        runner.run_test("lfm2_style_tool_call_repeated", test_lfm2_style_tool_call_accepts_repeated_tool_calls(fixture));
        runner.run_test("lfm2_style_tool_call_pythonic_literals", test_lfm2_style_tool_call_accepts_pythonic_literals(fixture));
        runner.run_test("lfm2_style_tool_call_unknown_tool", test_lfm2_style_tool_call_rejects_unknown_tool(fixture));
        runner.run_test("lfm2_style_tool_call_malformed_wrapper", test_lfm2_style_tool_call_rejects_malformed_wrapper(fixture));
        runner.run_test("lfm2_style_tool_call_malformed_literal", test_lfm2_style_tool_call_rejects_malformed_literal(fixture));
        runner.run_test("gemma_style_tool_call_single", test_gemma_style_tool_call_accepts_single_tool_call(fixture));
        runner.run_test("gemma_style_tool_call_single_pipe_tags", test_gemma_style_tool_call_accepts_single_tool_call_with_pipe_tags(fixture));
        runner.run_test("gemma_style_tool_call_repeated", test_gemma_style_tool_call_accepts_repeated_tool_calls(fixture));
        runner.run_test("gemma_style_tool_call_mixed_value_types", test_gemma_style_tool_call_accepts_mixed_value_types(fixture));
        runner.run_test("gemma_style_tool_call_unknown_tool", test_gemma_style_tool_call_rejects_unknown_tool(fixture));
        runner.run_test("gemma_style_tool_call_malformed_wrapper", test_gemma_style_tool_call_rejects_malformed_wrapper(fixture));
        runner.run_test("gemma_style_tool_call_malformed_escaped_string", test_gemma_style_tool_call_rejects_malformed_escaped_string(fixture));
        runner.run_test("gemma_style_tool_call_malformed_mixed_value_types", test_gemma_style_tool_call_rejects_malformed_mixed_value_types(fixture));
        runner.run_test("model_decode_direct_output_reasoning", test_model_decode_accepts_direct_output_when_reasoning_enabled(fixture));
        runner.run_test("model_decode_thinking_then_output", test_model_decode_accepts_thinking_then_output(fixture));
        runner.run_test("model_decode_thinking_payload_less_than", test_model_decode_accepts_less_than_in_thinking_payload(fixture));
        runner.run_test("model_decode_rejects_thinking_only_eos", test_model_decode_rejects_thinking_only_eos(fixture));
        runner.run_test("model_decode_no_reasoning_prefix", test_model_decode_rejects_thinking_when_reasoning_disabled(fixture));
        runner.run_test("model_decode_thinking_then_json_schema", test_model_decode_accepts_thinking_then_json_schema(fixture));
        runner.run_test("model_decode_rejects_invalid_json_schema_after_thinking", test_model_decode_rejects_invalid_json_schema_after_thinking(fixture));
        runner.run_test("model_decode_invalid_close_only", test_model_decode_rejects_invalid_close_only_thinking_string(fixture));
        runner.run_test("model_decode_rejects_nested_thinking_close_tag", test_model_decode_rejects_nested_thinking_close_tag(fixture));
        runner.run_test("model_decode_qwen_with_tools_tool_call_after_thinking", test_model_decode_qwen_with_tools_accepts_tool_call_after_thinking(fixture));
        runner.run_test("model_decode_qwen_with_tools_allows_plain_text_when_not_forced", test_model_decode_qwen_with_tools_allows_plain_text_when_not_forced(fixture));
        runner.run_test("model_decode_qwen_with_tools_rejects_plain_text_when_forced", test_model_decode_qwen_with_tools_rejects_plain_text_when_forced(fixture));
        runner.run_test("model_decode_lfm2_with_tools_tool_call_after_thinking", test_model_decode_lfm2_with_tools_accepts_tool_call_after_thinking(fixture));
        runner.run_test("model_decode_gemma_with_tools_tool_call_after_thinking", test_model_decode_gemma_with_tools_accepts_tool_call_after_thinking(fixture));
        runner.run_test("grammar_matcher_reset", test_grammar_matcher_reset_restores_initial_state(fixture));
    } catch (const std::exception& e) {
        std::cerr << "[✗] Grammar test setup failed: " << e.what() << "\n";
    }

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
