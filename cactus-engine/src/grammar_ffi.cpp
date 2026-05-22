#include "../cactus_engine.h"
#include "ebnf_syntax.h"
#include "gemma_tools.h"
#include "utils.h"
#include "engine.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <picojson/picojson.h>

using namespace cactus::ffi;
using namespace cactus::engine;

namespace cactus {
namespace ffi {

struct CactusGrammarHandle {
    std::unique_ptr<cactus::engine::Grammar> grammar;
};

struct CactusGrammarVocabularyHandle {
    std::unique_ptr<cactus::engine::GrammarVocabulary> vocabulary;
};

struct CactusGrammarEngineHandle {
    std::unique_ptr<cactus::engine::GrammarEngine> engine;
};

struct CactusGrammarMatcherHandle {
    std::unique_ptr<cactus::engine::GrammarMatcher> matcher;
};

} // namespace ffi
} // namespace cactus

namespace {

static cactus_grammar_t handle_exception(const char* operation, const std::string& error) {
    last_error_message = std::string(operation) + ": " + error;
    CACTUS_LOG_ERROR("grammar", last_error_message);
    return nullptr;
}

static int handle_int_exception(const char* operation, const std::string& error) {
    last_error_message = std::string(operation) + ": " + error;
    CACTUS_LOG_ERROR("grammar", last_error_message);
    return -1;
}

template <typename Handle, typename RawHandle>
static Handle* require_handle(const char* operation, RawHandle handle, const char* label) {
    if (!handle) {
        handle_exception(operation, std::string(label) + " is null");
        return nullptr;
    }
    return static_cast<Handle*>(handle);
}

static std::vector<Grammar> collect_grammars(cactus_grammar_t* grammars, size_t num_grammars) {
    std::vector<Grammar> collected;
    collected.reserve(num_grammars);
    for (size_t i = 0; i < num_grammars; ++i) {
        if (!grammars || !grammars[i]) continue;
        collected.push_back(*static_cast<CactusGrammarHandle*>(grammars[i])->grammar);
    }
    return collected;
}

static CactusGrammarVocabularyHandle* require_vocabulary_handle(
    const char* operation,
    cactus_grammar_vocabulary_t vocabulary
) {
    return require_handle<CactusGrammarVocabularyHandle>(operation, vocabulary, "grammar vocabulary");
}

static CactusGrammarEngineHandle* require_engine_handle(
    const char* operation,
    cactus_grammar_engine_t engine
) {
    return require_handle<CactusGrammarEngineHandle>(operation, engine, "grammar engine");
}

static CactusGrammarMatcherHandle* require_matcher_handle(
    const char* operation,
    cactus_grammar_matcher_t matcher
) {
    return require_handle<CactusGrammarMatcherHandle>(operation, matcher, "grammar matcher");
}

static CactusGrammarHandle* require_grammar_handle(const char* operation, cactus_grammar_t grammar) {
    return require_handle<CactusGrammarHandle>(operation, grammar, "grammar");
}

template <typename Factory>
static cactus_grammar_t make_grammar(const char* operation, Factory&& factory) {
    try {
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(std::forward<Factory>(factory)())
        };
    } catch (const std::exception& e) {
        return handle_exception(operation, e.what());
    }
}

static const std::string& require_tool_schema(const ToolFunction& tool) {
    if (tool.name.empty()) throw std::runtime_error("Tool name is required");

    auto schema_it = tool.parameters.find("schema");
    if (schema_it == tool.parameters.end() || schema_it->second.empty()) {
        throw std::runtime_error("Tool '" + tool.name + "' is missing a parameters schema");
    }

    return schema_it->second;
}

static std::string require_normalized_tool_name(const ToolFunction& tool, const char* model_name) {
    const std::string normalized_name = to_snake_case(tool.name);
    if (normalized_name.empty()) {
        throw std::runtime_error("Tool '" + tool.name + "' normalized to an empty " + std::string(model_name) + " name");
    }
    return normalized_name;
}

static Grammar gemma_tool_grammar(const std::vector<ToolFunction>& tools, bool use_pipe_tags) {
    std::vector<std::pair<std::string, EbnfSyntax>> tool_rule_sets;
    tool_rule_sets.reserve(tools.size());

    for (const auto& tool : tools) {
        const std::string& schema = require_tool_schema(tool);

        picojson::value schema_value;
        const std::string parse_error = picojson::parse(schema_value, schema);
        if (!parse_error.empty()) {
            throw std::runtime_error("Tool '" + tool.name + "' schema parse failed: " + parse_error);
        }

        std::unordered_set<std::string> property_names;
        gemma::collect_schema_property_names(schema_value, property_names);
        std::unordered_set<std::string> string_literals;
        gemma::collect_schema_string_literals(schema_value, string_literals);

        EbnfSyntax tool_syntax = gemma::xgrammar_tools_ebnf_to_gemma_tools_ebnf(
            Grammar::json_schema(schema, false, 0).ebnf(),
            property_names,
            string_literals,
            use_pipe_tags
        );

        const std::string args_rule_name = tool.name + "_args";
        tool_syntax.rename_rules({{"root", args_rule_name}});
        tool_rule_sets.push_back({tool.name, std::move(tool_syntax)});
    }

    EbnfSyntax merged;
    merged.merge_with(tool_rule_sets);

    std::vector<std::string> call_rule_names;
    call_rule_names.reserve(tools.size());
    for (const auto& tool : tools) {
        const std::string call_rule_name = tool.name + "_call";
        const std::string args_rule_name = tool.name + "_args";
        merged.rules[call_rule_name] = "(\"" + EbnfSyntax::escape_string_literal(tool.name) + "\" " + args_rule_name + ")";
        call_rule_names.push_back(call_rule_name);
    }

    std::string call_body_expr;
    for (size_t i = 0; i < call_rule_names.size(); ++i) {
        if (i != 0) call_body_expr += " | ";
        call_body_expr += call_rule_names[i];
    }
    merged.rules["call_body"] = call_body_expr;

    const std::string tool_call_start =
        EbnfSyntax::escape_string_literal(gemma::tool_call_start_tag(use_pipe_tags) + "call:");
    const std::string tool_call_end =
        EbnfSyntax::escape_string_literal(gemma::tool_call_end_tag(use_pipe_tags));
    merged.rules["root"] = "\"" + tool_call_start + "\" call_body \"" + tool_call_end + "\"";

    return Grammar::repeat_range(Grammar::ebnf(merged.ebnf()), 1, -1);
}

static Grammar needle_tool_grammar(const std::vector<ToolFunction>& tools) {
    std::vector<std::pair<std::string, EbnfSyntax>> tool_rule_sets;
    tool_rule_sets.reserve(tools.size());

    for (const auto& tool : tools) {
        const std::string& schema = require_tool_schema(tool);
        const std::string normalized_name = require_normalized_tool_name(tool, "Needle");

        EbnfSyntax tool_syntax = EbnfSyntax::from_string(Grammar::json_schema(schema, false, 0).ebnf());
        tool_syntax.remove_json_whitespaces();
        tool_syntax.rename_rules({{"root", normalized_name + "_args"}});
        tool_rule_sets.push_back({normalized_name, std::move(tool_syntax)});
    }

    EbnfSyntax merged;
    merged.merge_with(tool_rule_sets);

    std::vector<std::string> call_rule_names;
    call_rule_names.reserve(tool_rule_sets.size());
    for (const auto& [normalized_name, _] : tool_rule_sets) {
        const std::string call_rule_name = normalized_name + "_call";
        const std::string args_rule_name = normalized_name + "_args";
        const std::string call_prefix =
            EbnfSyntax::escape_string_literal("{\"name\":\"" + normalized_name + "\",\"arguments\":");
        merged.rules[call_rule_name] = "(\"" + call_prefix + "\" " + args_rule_name + " \"}\")";
        call_rule_names.push_back(call_rule_name);
    }

    std::string call_body_expr;
    for (size_t i = 0; i < call_rule_names.size(); ++i) {
        if (i != 0) call_body_expr += " | ";
        call_body_expr += call_rule_names[i];
    }
    merged.rules["call_body"] = call_body_expr;
    merged.rules["root"] = "\"<tool_call>[\" call_body (\",\" call_body)* \"]\"";
    return Grammar::ebnf(merged.ebnf());
}

static Grammar qwen_tool_grammar(const std::vector<ToolFunction>& tools) {
    std::vector<std::pair<std::string, EbnfSyntax>> tool_rule_sets;
    tool_rule_sets.reserve(tools.size());

    for (const auto& tool : tools) {
        const std::string& schema = require_tool_schema(tool);

        EbnfSyntax tool_syntax = EbnfSyntax::from_string(Grammar::json_schema(schema, false, 0).ebnf());
        tool_syntax.remove_json_whitespaces();
        tool_syntax.rename_rules({{"root", tool.name + "_args"}});
        tool_rule_sets.push_back({tool.name, std::move(tool_syntax)});
    }

    EbnfSyntax merged;
    merged.merge_with(tool_rule_sets);

    std::vector<std::string> call_rule_names;
    call_rule_names.reserve(tool_rule_sets.size());
    for (const auto& [normalized_name, _] : tool_rule_sets) {
        const std::string call_rule_name = normalized_name + "_call";
        const std::string args_rule_name = normalized_name + "_args";
        const std::string call_prefix = EbnfSyntax::escape_multiline_literal(
            "<tool_call>\n{\"name\":\"" + normalized_name + "\",\"arguments\":"
        );
        const std::string call_suffix = EbnfSyntax::escape_multiline_literal("}\n</tool_call>");
        merged.rules[call_rule_name] = "(\"" + call_prefix + "\" " + args_rule_name + " \"" + call_suffix + "\")";
        call_rule_names.push_back(call_rule_name);
    }

    std::string call_body_expr;
    for (size_t i = 0; i < call_rule_names.size(); ++i) {
        if (i != 0) call_body_expr += " | ";
        call_body_expr += call_rule_names[i];
    }
    merged.rules["call_body"] = call_body_expr;
    merged.rules["root"] = "call_body (\"\\n\" call_body)*";
    return Grammar::ebnf(merged.ebnf());
}

static Grammar thinking_structural_tag(const std::string& begin, const std::string& end) {
    return Grammar::structural_tag(R"({
        "type": "structural_tag",
        "format": {
            "type": "tag",
            "begin": ")" + begin + R"(",
            "content": { "type": "any_text" },
            "end": ")" + end + R"("
        }
    })");
}

} // anonymous namespace

extern "C" {

cactus_grammar_json_schema_options_t cactus_grammar_json_schema_default_options() {
    return { true, 2, {",", ":"}, true, 1, };
}

cactus_grammar_vocabulary_t cactus_grammar_vocabulary_init(const char* model_path) {
    if (!model_path) return handle_exception(__func__, "model path is null");

    try {
        const auto vocabulary = GrammarVocabulary::from_model_dir(model_path);
        return new CactusGrammarVocabularyHandle{
            std::make_unique<GrammarVocabulary>(vocabulary)
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

cactus_grammar_vocabulary_t cactus_grammar_vocabulary_init_from_model(cactus_model_t model) {
    if (!model) return handle_exception(__func__, "model is null");

    try {
        auto handle = static_cast<CactusModelHandle*>(model);
        auto tokenizer = handle->model ? handle->model->get_tokenizer() : nullptr;
        if (!tokenizer) return handle_exception(__func__, "model tokenizer is null");
        const auto vocabulary = GrammarVocabulary::from_tokenizer(*tokenizer);
        return new CactusGrammarVocabularyHandle{
            std::make_unique<GrammarVocabulary>(vocabulary)
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

int cactus_grammar_vocabulary_get_add_prefix_space(cactus_grammar_vocabulary_t vocabulary) {
    auto* handle = require_vocabulary_handle(__func__, vocabulary);
    if (!handle) return -1;

    return handle->vocabulary->add_prefix_space() ? 1 : 0;
}

size_t cactus_grammar_vocabulary_get_size(cactus_grammar_vocabulary_t vocabulary) {
    auto* handle = require_vocabulary_handle(__func__, vocabulary);
    if (!handle) return 0;

    return handle->vocabulary->vocab_size();
}

int cactus_grammar_vocabulary_get_stop_token_ids(
    cactus_grammar_vocabulary_t vocabulary,
    uint32_t* buffer,
    size_t buffer_size,
    size_t* out_token_count
) {
    auto* handle = require_vocabulary_handle(__func__, vocabulary);
    if (!handle) return -1;
    if (!out_token_count) return handle_int_exception(__func__, "out_token_count is null");

    const auto& stop_token_ids = handle->vocabulary->stop_token_ids();
    *out_token_count = stop_token_ids.size();
    if (!buffer) return handle_int_exception(__func__, "buffer is null");
    if (buffer_size < stop_token_ids.size()) {
        return handle_int_exception(__func__, "buffer too small");
    }

    std::copy(stop_token_ids.begin(), stop_token_ids.end(), buffer);
    return 0;
}

void cactus_grammar_vocabulary_destroy(cactus_grammar_vocabulary_t vocabulary) {
    if (vocabulary) delete static_cast<CactusGrammarVocabularyHandle*>(vocabulary);
}

cactus_grammar_t cactus_grammar_init_ebnf(const char* ebnf, const char* start_symbol) {
    if (!ebnf) return handle_exception(__func__, "ebnf is null");

    return make_grammar(__func__, [&] {
        return Grammar::ebnf(ebnf, start_symbol ? start_symbol : "root");
    });
}

cactus_grammar_t cactus_grammar_init_json() {
    return make_grammar(__func__, [] {
        return Grammar::json();
    });
}

cactus_grammar_t cactus_grammar_init_empty() {
    return make_grammar(__func__, [] {
        return Grammar();
    });
}

cactus_grammar_t cactus_grammar_init_epsilon() {
    return make_grammar(__func__, [] {
        return Grammar::epsilon();
    });
}

cactus_grammar_t cactus_grammar_init_universal() {
    return make_grammar(__func__, [] {
        return Grammar::universal();
    });
}

cactus_grammar_t cactus_grammar_init_json_schema(
    const char* json_schema,
    cactus_grammar_json_schema_options_t options
) {
    if (!json_schema) return handle_exception(__func__, "json_schema is null");

    if ((!options.any_whitespace) && (!options.separators[0] || !options.separators[1])) {
        return handle_exception(
            __func__,
            "json schema separators must have 2 strings"
        );
    }

    return make_grammar(__func__, [&] {
        return Grammar::json_schema(
            json_schema,
            options.any_whitespace,
            options.indent,
            {
                options.separators[0] ? options.separators[0] : ",",
                options.separators[1] ? options.separators[1] : ":"
            },
            options.strict_mode,
            options.max_whitespace_count
        );
    });
}

cactus_grammar_t cactus_grammar_init_regex(const char* regex) {
    if (!regex) return handle_exception(__func__, "regex is null");

    return make_grammar(__func__, [&] {
        return Grammar::regex(regex);
    });
}

cactus_grammar_t cactus_grammar_init_structural_tag(
    const char* structural_tag_json,
    cactus_grammar_vocabulary_t vocabulary
) {
    if (!structural_tag_json) return handle_exception(__func__, "structural_tag_json is null");

    return make_grammar(__func__, [&] {
        auto* handle = static_cast<CactusGrammarVocabularyHandle*>(vocabulary);
        return Grammar::structural_tag(
            structural_tag_json,
            handle ? handle->vocabulary.get() : nullptr
        );
    });
}

cactus_grammar_t cactus_grammar_init_model_tools(const char* model_type, const char* tools_json) {
    if (!model_type) return handle_exception(__func__, "model_type is null");
    if (!tools_json) return handle_exception(__func__, "tools_json is null");

    return make_grammar(__func__, [&] {
        const std::string type(model_type);
        const auto tools = parse_tools_json(tools_json);
        if (tools.empty()) return Grammar();

        const auto is_function_gemma = type.find("functiongemma") != std::string::npos;
        if (gemma::is_gemma4_model_type(type) || is_function_gemma) {
            return gemma_tool_grammar(tools, !is_function_gemma);
        } else if (type.find("qwen") != std::string::npos) {
            return qwen_tool_grammar(tools);
        } else if (type.find("needle") != std::string::npos) {
            return needle_tool_grammar(tools);
        }
        return Grammar();
    });
}

cactus_grammar_t cactus_grammar_init_model_thinking(const char* model_type) {
    if (!model_type) return handle_exception(__func__, "model_type is null");

    return make_grammar(__func__, [&] {
        std::string type(model_type);
        if (gemma::is_gemma4_model_type(type)) {
            return thinking_structural_tag("<|channel>", "<channel|>");
        } else if (type.find("lfm2") != std::string::npos || type.find("qwen") != std::string::npos) {
            return thinking_structural_tag("<think>", "</think>");
        }
        return Grammar();
    });
}

cactus_grammar_t cactus_grammar_union(cactus_grammar_t* grammars, size_t num_grammars) {
    return make_grammar(__func__, [&] {
        return Grammar::unite(collect_grammars(grammars, num_grammars));
    });
}

cactus_grammar_t cactus_grammar_concatenate(cactus_grammar_t* grammars, size_t num_grammars) {
    return make_grammar(__func__, [&] {
        return Grammar::concatenate(collect_grammars(grammars, num_grammars));
    });
}

cactus_grammar_t cactus_grammar_optional(cactus_grammar_t grammar) {
    auto* handle = require_grammar_handle(__func__, grammar);
    if (!handle) return nullptr;

    return make_grammar(__func__, [&] {
        return Grammar::optional(*handle->grammar);
    });
}

cactus_grammar_t cactus_grammar_star(cactus_grammar_t grammar) {
    auto* handle = require_grammar_handle(__func__, grammar);
    if (!handle) return nullptr;

    return make_grammar(__func__, [&] {
        return Grammar::star(*handle->grammar);
    });
}

cactus_grammar_t cactus_grammar_repeat(cactus_grammar_t grammar, int count) {
    auto* handle = require_grammar_handle(__func__, grammar);
    if (!handle) return nullptr;

    return make_grammar(__func__, [&] {
        return Grammar::repeat(*handle->grammar, count);
    });
}

cactus_grammar_t cactus_grammar_repeat_range(cactus_grammar_t grammar, int min_count, int max_count) {
    auto* handle = require_grammar_handle(__func__, grammar);
    if (!handle) return nullptr;

    return make_grammar(__func__, [&] {
        return Grammar::repeat_range(*handle->grammar, min_count, max_count);
    });
}

int cactus_grammar_get_ebnf(cactus_grammar_t grammar, char* buffer, size_t buffer_size) {
    if (!grammar) return handle_int_exception(__func__, "grammar is null");
    if (!buffer || buffer_size == 0) {
        return handle_int_exception(__func__, "buffer is null or buffer_size is 0");
    }

    try {
        std::string ebnf = static_cast<CactusGrammarHandle*>(grammar)->grammar->ebnf();
        if (ebnf.size() >= buffer_size) return handle_int_exception(__func__, "buffer too small");
        std::strcpy(buffer, ebnf.c_str());
        return 0;
    } catch (const std::exception& e) {
        return handle_int_exception(__func__, e.what());
    }
}

int cactus_grammar_is_empty(cactus_grammar_t grammar) {
    if (!grammar) return handle_int_exception(__func__, "grammar is null");
    return static_cast<CactusGrammarHandle*>(grammar)->grammar->is_empty() ? 1 : 0;
}

void cactus_grammar_destroy(cactus_grammar_t grammar) {
    if (grammar) delete static_cast<CactusGrammarHandle*>(grammar);
}

cactus_grammar_engine_t cactus_grammar_engine_init(cactus_grammar_vocabulary_t vocabulary) {
    auto* handle = require_vocabulary_handle(__func__, vocabulary);
    if (!handle) return nullptr;

    try {
        return new CactusGrammarEngineHandle{
            std::make_unique<GrammarEngine>(*handle->vocabulary)
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

void cactus_grammar_engine_destroy(cactus_grammar_engine_t engine) {
    if (engine) delete static_cast<CactusGrammarEngineHandle*>(engine);
}

cactus_grammar_matcher_t cactus_grammar_engine_compile_matcher(
    cactus_grammar_engine_t engine,
    cactus_grammar_t grammar
) {
    auto* engine_handle = require_engine_handle(__func__, engine);
    if (!engine_handle) return nullptr;
    auto* grammar_handle = require_grammar_handle(__func__, grammar);
    if (!grammar_handle) return nullptr;

    try {
        return new CactusGrammarMatcherHandle{
            std::make_unique<GrammarMatcher>(
                engine_handle->engine->compile_matcher(*grammar_handle->grammar)
            )
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

void cactus_grammar_matcher_destroy(cactus_grammar_matcher_t matcher) {
    if (matcher) delete static_cast<CactusGrammarMatcherHandle*>(matcher);
}

void cactus_grammar_matcher_reset(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return;
    handle->matcher->reset();
}

void cactus_grammar_matcher_rollback(cactus_grammar_matcher_t matcher, int tokens) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return;
    handle->matcher->rollback(tokens);
}

cactus_grammar_matcher_t cactus_grammar_matcher_fork(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return nullptr;

    try {
        return new CactusGrammarMatcherHandle{
            std::make_unique<GrammarMatcher>(handle->matcher->fork())
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

cactus_grammar_t cactus_grammar_matcher_get_grammar(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return nullptr;

    try {
        return new CactusGrammarHandle{
            std::make_unique<Grammar>(handle->matcher->grammar())
        };
    } catch (const std::exception& e) {
        return handle_exception(__func__, e.what());
    }
}

int cactus_grammar_matcher_is_completed(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    return handle ? (handle->matcher->is_completed() ? 1 : 0) : -1;
}

int cactus_grammar_matcher_is_terminated(cactus_grammar_matcher_t matcher) {
    auto* handle = require_matcher_handle(__func__, matcher);
    return handle ? (handle->matcher->is_terminated() ? 1 : 0) : -1;
}

int cactus_grammar_matcher_accept(cactus_grammar_matcher_t matcher, uint32_t token_id) {
    auto* handle = require_matcher_handle(__func__, matcher);
    return handle ? (handle->matcher->accept(token_id) ? 1 : 0) : -1;
}

int cactus_grammar_matcher_next_bitmask(
    cactus_grammar_matcher_t matcher,
    int32_t* bitmask,
    size_t logits_buffer_size
) {
    auto* handle = require_matcher_handle(__func__, matcher);
    if (!handle) return -1;
    if (!bitmask) return handle_int_exception(__func__, "bitmask is null");

    try {
        std::vector<int32_t> next_bitmask;
        const bool should_apply = handle->matcher->next_bitmask(next_bitmask, logits_buffer_size);
        std::copy(next_bitmask.begin(), next_bitmask.end(), bitmask);
        return should_apply ? 1 : 0;
    } catch (const std::exception& e) {
        return handle_int_exception(__func__, e.what());
    }
}

}
