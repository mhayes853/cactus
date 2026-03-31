#include "engine.h"

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cactus {
namespace engine {

Grammar::Grammar() : grammar(xgrammar::NullObj{}), is_universal_(false) {}

Grammar::Grammar(xgrammar::Grammar raw_grammar)
    : grammar(raw_grammar), is_universal_(false) {}

static const picojson::object& require_object_field(
    const picojson::object& object,
    const std::string& field_name,
    const char* context
) {
    if (!object.count(field_name) || !object.at(field_name).is<picojson::object>()) {
        throw std::runtime_error(std::string(context) + " must contain an object field '" + field_name + "'");
    }
    return object.at(field_name).get<picojson::object>();
}

static const std::string& require_string_field(
    const picojson::object& object,
    const std::string& field_name,
    const char* context
) {
    if (!object.count(field_name) || !object.at(field_name).is<std::string>()) {
        throw std::runtime_error(std::string(context) + " must contain a string field '" + field_name + "'");
    }
    return object.at(field_name).get<std::string>();
}

std::vector<ToolDefinition> ToolDefinition::parse_tools_json(const std::string& tools_json) {
    if (tools_json.empty()) {
        throw std::runtime_error("tools_json must not be empty");
    }

    picojson::value parsed;
    std::string parse_error;
    auto end = picojson::parse(parsed, tools_json.begin(), tools_json.end(), &parse_error);
    if (!parse_error.empty()) {
        throw std::runtime_error("failed to parse tools_json: " + parse_error);
    }
    if (end != tools_json.end()) {
        throw std::runtime_error("failed to parse tools_json: trailing content after JSON payload");
    }
    if (!parsed.is<picojson::array>()) {
        throw std::runtime_error("tools_json must be a JSON array");
    }

    std::vector<ToolDefinition> tools;
    for (const auto& item : parsed.get<picojson::array>()) {
        if (!item.is<picojson::object>()) {
            throw std::runtime_error("each tool entry must be a JSON object");
        }

        const auto& tool_object = item.get<picojson::object>();
        const std::string& type = require_string_field(tool_object, "type", "tool entry");
        if (type != "function") {
            throw std::runtime_error("tool entry field 'type' must be 'function'");
        }

        const auto& function_object = require_object_field(tool_object, "function", "tool entry");
        const std::string& name = require_string_field(function_object, "name", "tool entry function");
        if (!function_object.count("parameters") || !function_object.at("parameters").is<picojson::object>()) {
            throw std::runtime_error("tool entry function must contain an object field 'parameters'");
        }

        tools.push_back(ToolDefinition{name, function_object.at("parameters")});
    }

    if (tools.empty()) {
        throw std::runtime_error("tools_json must contain at least one tool entry");
    }

    return tools;
}

Grammar Grammar::gbnf(const std::string& gbnf, const std::string& start_symbol) {
    return Grammar(xgrammar::Grammar::FromEBNF(gbnf, start_symbol));
}

Grammar Grammar::json() {
    return Grammar(xgrammar::Grammar::BuiltinJSONGrammar());
}

Grammar Grammar::universal() {
    static auto grammar = Grammar::structural_tag(R"({
        "type": "structural_tag",
        "format": {
            "type": "any_text"
        }
    })");
    grammar.is_universal_ = true;
    return grammar;
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

static std::string qwen_tool_structural_tag_json(const std::vector<ToolDefinition>& tools, bool force_tools) {
    if (tools.empty()) {
        throw std::runtime_error("qwen_style_tool_call requires at least one tool definition");
    }

    picojson::array tags;

    for (const auto& tool : tools) {
        if (!tool.arguments_schema.is<picojson::object>()) {
            throw std::runtime_error("tool '" + tool.name + "' arguments schema must be a JSON object");
        }
        if (tool.name.empty()) {
            throw std::runtime_error("tool definitions must have a non-empty name");
        }

        picojson::object content;
        content["type"] = picojson::value("json_schema");
        content["json_schema"] = tool.arguments_schema;

        picojson::object tag;
        tag["begin"] = picojson::value(
            std::string("<tool_call>\n{\"name\": \"") + tool.name + "\", \"arguments\": "
        );
        tag["content"] = picojson::value(content);
        tag["end"] = picojson::value("}\n</tool_call>");
        tags.push_back(picojson::value(tag));
    }

    picojson::object format;
    if (force_tools) {
        format["type"] = picojson::value("tags_with_separator");
        format["separator"] = picojson::value("");
        format["tags"] = picojson::value(tags);
    } else {
        picojson::array triggers;
        triggers.push_back(picojson::value("<tool_call>"));
        format["type"] = picojson::value("triggered_tags");
        format["triggers"] = picojson::value(triggers);
        format["tags"] = picojson::value(tags);
    }

    picojson::object root;
    root["type"] = picojson::value("structural_tag");
    root["format"] = picojson::value(format);
    return picojson::value(root).serialize();
}

static Grammar qwen_style_tool_call_grammar(const std::vector<ToolDefinition>& tools, bool force_tools) {
    return Grammar::structural_tag(qwen_tool_structural_tag_json(tools, force_tools));
}

static Grammar gemma_style_tool_call_grammar(const std::vector<ToolDefinition>& tools) {
    (void)tools;
    return Grammar();
}

static std::string gbnf_escape_literal(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        if (c == '\\' || c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return escaped;
}

static std::string lfm2_tool_name_rule(const std::vector<ToolDefinition>& tools) {
    if (tools.empty()) {
        throw std::runtime_error("lfm2_style_tool_call requires at least one tool definition");
    }

    std::string rule = "tool_name ::= ";
    for (size_t i = 0; i < tools.size(); ++i) {
        const auto& tool = tools[i];
        if (tool.name.empty()) {
            throw std::runtime_error("tool definitions must have a non-empty name");
        }
        if (i > 0) {
            rule += " | ";
        }
        rule += "\"" + gbnf_escape_literal(tool.name) + "\"";
    }
    return rule;
}

static std::string lfm2_tool_call_gbnf(const std::vector<ToolDefinition>& tools) {
    return std::string(R"GRAMMAR(root ::= tool_block+
tool_block ::= tool_call_start "[" call_list "]" tool_call_end
call_list ::= call ("," call)*
call ::= tool_name "(" kwarg_list? ")"
kwarg_list ::= kwarg ("," kwarg)*
kwarg ::= ident "=" value
value ::= string | number | boolean | none | list | dict
list ::= "[" list_items? "]"
list_items ::= value ("," value)*
dict ::= "{" dict_items? "}"
dict_items ::= dict_item ("," dict_item)*
dict_item ::= dict_key ":" value
dict_key ::= string | ident
boolean ::= "True" | "False" | "true" | "false"
none ::= "None" | "null"
number ::= "-"? int frac? exp?
int ::= "0" | nonzero_digit digit*
frac ::= "." digit+
exp ::= ("e" | "E") ("+" | "-")? digit+
string ::= "\"" string_char* "\""
string_char ::= unescaped | "\\" escape
unescaped ::= [^"\\]
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t"
ident ::= ident_start ident_continue*
ident_start ::= [A-Za-z_]
ident_continue ::= [A-Za-z0-9_]
digit ::= [0-9]
nonzero_digit ::= [1-9]
tool_call_start ::= "<|tool_call_start|>"
tool_call_end ::= "<|tool_call_end|>"
)GRAMMAR") + lfm2_tool_name_rule(tools) + "\n";
}

static Grammar lfm2_style_tool_call_grammar(const std::vector<ToolDefinition>& tools) {
    return Grammar::gbnf(lfm2_tool_call_gbnf(tools));
}

static Grammar model_tool_call_grammar(
    Config::ModelType model_type,
    bool force_tools,
    const std::vector<ToolDefinition>& tools
) {
    if (tools.empty()) return Grammar();
    switch (model_type) {
        case Config::ModelType::QWEN:
        case Config::ModelType::QWEN3P5:
        case Config::ModelType::YOUTU:
            return qwen_style_tool_call_grammar(tools, force_tools);
        case Config::ModelType::GEMMA:
            return gemma_style_tool_call_grammar(tools);
        case Config::ModelType::LFM2:
            return lfm2_style_tool_call_grammar(tools);
        default:
            return Grammar();
    }
}

Grammar Grammar::unite(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) continue;
        if (grammar.is_universal()) return grammar;
        handles.push_back(grammar.raw_value());
    }
    return handles.empty() ? Grammar() : Grammar(xgrammar::Grammar::Union(handles));
}

Grammar Grammar::concatenate(const std::vector<Grammar>& grammars) {
    std::vector<xgrammar::Grammar> handles;
    handles.reserve(grammars.size());
    for (const auto& grammar : grammars) {
        if (grammar.is_empty()) continue;
        if (grammar.is_universal()) {
            if (handles.empty()) return grammar;
            handles.push_back(grammar.raw_value());
            break;
        }
        handles.push_back(grammar.raw_value());
    }
    return handles.empty() ? Grammar() : Grammar(xgrammar::Grammar::Concat(handles));
}

static Grammar reasoning_grammar() {
    static const auto grammar = Grammar::gbnf(R"(
        root ::= think?
        think ::= "<think>\n" any_non_thinking_character* "\n</think>\n\n"
        any_non_thinking_character ::= (
            [^<]
            | "<" [^/]
            | "</" [^t]
            | "</t" [^h]
            | "</th" [^i]
            | "</thi" [^n]
            | "</thin" [^k]
            | "</think" [^>]
        )
    )");
    return grammar;
}

Grammar Grammar::model_decode_grammar(
    const Grammar& grammar,
    bool force_tools,
    bool supports_reasoning,
    Config::ModelType model_type,
    const std::vector<ToolDefinition>& tools
) {
    if (grammar.is_empty() && tools.empty()) {
        return Grammar();
    }
    auto tool_grammar = model_tool_call_grammar(model_type, force_tools, tools);
    auto content_grammar = force_tools ? tool_grammar : Grammar::unite({grammar, tool_grammar});
    return supports_reasoning
        ? Grammar::concatenate({reasoning_grammar(), content_grammar})
        : content_grammar;
}

bool Grammar::is_empty() const {
    return grammar.IsNull();
}

bool Grammar::is_universal() const {
    return is_universal_;
}

const xgrammar::Grammar& Grammar::raw_value() const {
    return grammar;
}

namespace {

static xgrammar::VocabType to_xgrammar_vocab_type(VocabType vocab_type) {
    switch (vocab_type) {
        case VocabType::RAW:
            return xgrammar::VocabType::RAW;
        case VocabType::BYTE_LEVEL:
            return xgrammar::VocabType::BYTE_LEVEL;
        case VocabType::BYTE_FALLBACK:
            return xgrammar::VocabType::BYTE_FALLBACK;
    }
}

} // anonymous namespace

GrammarMatcher::GrammarMatcher(const Grammar* grammar, const TokenizerInfo& tokenizer_info)
    : matcher(nullptr), tokenizer_info(nullptr) {
    if (grammar->is_empty()) {
        throw std::runtime_error("Cannot create GrammarMatcher with empty grammar");
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

    auto compiled_grammar = compiler.CompileGrammar(grammar->raw_value());
    matcher = xgrammar::GrammarMatcher(compiled_grammar);
    this->tokenizer_info = xgrammar_tokenizer_info;
}

void GrammarMatcher::reset() {
    matcher.Reset();
}

bool GrammarMatcher::accept(uint32_t token_id, bool log_rejection) {
    const bool accepted = matcher.AcceptToken(token_id);
    if (!accepted && log_rejection) {
        CACTUS_LOG_WARN("model decode", "Token id: " << token_id << " was not accepted by grammar matcher.");
    }
    return accepted;
}

bool GrammarMatcher::next_bitmask(std::vector<int32_t>& token_bitmask) {
    const int32_t bitmask_size = xgrammar::GetBitmaskSize(tokenizer_info.GetVocabSize());
    token_bitmask.resize(bitmask_size);

    int64_t bitmask_shape[1];
    int64_t bitmask_strides[1];
    bitmask_shape[0] = static_cast<int64_t>(token_bitmask.size());
    bitmask_strides[0] = 1;

    DLTensor bitmask_tensor;
    bitmask_tensor.data = token_bitmask.data();
    bitmask_tensor.device = DLDevice{kDLCPU, 0};
    bitmask_tensor.ndim = 1;
    bitmask_tensor.dtype = xgrammar::GetBitmaskDLType();
    bitmask_tensor.shape = bitmask_shape;
    bitmask_tensor.strides = bitmask_strides;
    bitmask_tensor.byte_offset = 0;

    return matcher.FillNextTokenBitmask(&bitmask_tensor, 0);
}

} // namespace engine
} // namespace cactus
