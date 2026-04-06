#ifndef CACTUS_LFM2_TOOLS_H
#define CACTUS_LFM2_TOOLS_H

#include <picojson.h>

#include <cctype>
#include <cstring>
#include <string>
#include <vector>

namespace cactus {
namespace ffi {

static inline std::string trim_lfm2_slice(const std::string& value, size_t begin, size_t end) {
    size_t start = begin;
    while (start < end && std::isspace(static_cast<unsigned char>(value[start]))) {
        start++;
    }
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }
    return value.substr(start, end - start);
}

static inline bool is_lfm2_identifier_start(char c) {
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isalpha(uc) || c == '_';
}

static inline bool is_lfm2_identifier_char(char c) {
    unsigned char uc = static_cast<unsigned char>(c);
    return std::isalnum(uc) || c == '_' || c == '-';
}

struct LFM2ToolCallParserState {
    const std::string& text;
    size_t pos = 0;

    LFM2ToolCallParserState(const std::string& text) : text(text) {}

    bool at_end() const { return pos >= text.size(); }
    char peek() const { return at_end() ? '\0' : text[pos]; }
    char consume() { return at_end() ? '\0' : text[pos++]; }

    void skip_whitespace() {
        while (!at_end() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            pos++;
        }
    }

    bool consume_if(char expected) {
        skip_whitespace();
        if (peek() != expected) return false;
        pos++;
        return true;
    }

    bool consume_literal(const char* literal) {
        skip_whitespace();
        size_t len = std::strlen(literal);
        if (text.compare(pos, len, literal) != 0) return false;
        pos += len;
        return true;
    }
};

static inline bool parse_lfm2_identifier(LFM2ToolCallParserState& state, std::string& out) {
    state.skip_whitespace();
    if (state.at_end() || !is_lfm2_identifier_start(state.peek())) return false;

    size_t start = state.pos++;
    while (!state.at_end() && is_lfm2_identifier_char(state.peek())) {
        state.pos++;
    }
    out = state.text.substr(start, state.pos - start);
    return true;
}

static inline bool parse_lfm2_string(LFM2ToolCallParserState& state, picojson::value& out) {
    state.skip_whitespace();
    if (state.peek() != '"') return false;

    state.consume();
    std::string result;
    while (!state.at_end()) {
        char c = state.consume();
        if (c == '"') {
            out = picojson::value(result);
            return true;
        }
        if (c == '\\') {
            if (state.at_end()) return false;
            char escaped = state.consume();
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                default: return false;
            }
            continue;
        }
        result.push_back(c);
    }
    return false;
}

static inline bool parse_lfm2_number(LFM2ToolCallParserState& state, picojson::value& out) {
    state.skip_whitespace();
    size_t start = state.pos;
    if (state.peek() == '-') state.pos++;

    if (state.at_end()) return false;
    if (state.peek() == '0') {
        state.pos++;
    } else {
        if (!std::isdigit(static_cast<unsigned char>(state.peek()))) {
            state.pos = start;
            return false;
        }
        while (!state.at_end() && std::isdigit(static_cast<unsigned char>(state.peek()))) {
            state.pos++;
        }
    }

    if (!state.at_end() && state.peek() == '.') {
        state.pos++;
        if (state.at_end() || !std::isdigit(static_cast<unsigned char>(state.peek()))) {
            state.pos = start;
            return false;
        }
        while (!state.at_end() && std::isdigit(static_cast<unsigned char>(state.peek()))) {
            state.pos++;
        }
    }

    if (!state.at_end() && (state.peek() == 'e' || state.peek() == 'E')) {
        state.pos++;
        if (!state.at_end() && (state.peek() == '+' || state.peek() == '-')) {
            state.pos++;
        }
        if (state.at_end() || !std::isdigit(static_cast<unsigned char>(state.peek()))) {
            state.pos = start;
            return false;
        }
        while (!state.at_end() && std::isdigit(static_cast<unsigned char>(state.peek()))) {
            state.pos++;
        }
    }

    try {
        out = picojson::value(std::stod(state.text.substr(start, state.pos - start)));
        return true;
    } catch (...) {
        state.pos = start;
        return false;
    }
}

static inline bool parse_lfm2_value(LFM2ToolCallParserState& state, picojson::value& out);

static inline bool parse_lfm2_list(LFM2ToolCallParserState& state, picojson::value& out) {
    state.skip_whitespace();
    if (!state.consume_if('[')) return false;

    picojson::array values;
    state.skip_whitespace();
    if (state.consume_if(']')) {
        out = picojson::value(values);
        return true;
    }

    while (true) {
        picojson::value item;
        if (!parse_lfm2_value(state, item)) return false;
        values.push_back(item);

        state.skip_whitespace();
        if (state.consume_if(']')) {
            out = picojson::value(values);
            return true;
        }
        if (!state.consume_if(',')) return false;
    }
}

static inline bool parse_lfm2_dict(LFM2ToolCallParserState& state, picojson::value& out) {
    state.skip_whitespace();
    if (!state.consume_if('{')) return false;

    picojson::object object;
    state.skip_whitespace();
    if (state.consume_if('}')) {
        out = picojson::value(object);
        return true;
    }

    while (true) {
        std::string key;
        picojson::value key_value;
        if (!parse_lfm2_string(state, key_value)) return false;
        key = key_value.get<std::string>();

        if (!state.consume_if(':')) return false;

        picojson::value value;
        if (!parse_lfm2_value(state, value)) return false;
        object[key] = value;

        state.skip_whitespace();
        if (state.consume_if('}')) {
            out = picojson::value(object);
            return true;
        }
        if (!state.consume_if(',')) return false;
    }
}

static inline bool parse_lfm2_value(LFM2ToolCallParserState& state, picojson::value& out) {
    state.skip_whitespace();
    if (state.at_end()) return false;

    char c = state.peek();
    if (c == '"') return parse_lfm2_string(state, out);
    if (c == '{') return parse_lfm2_dict(state, out);
    if (c == '[') return parse_lfm2_list(state, out);
    if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return parse_lfm2_number(state, out);
    if (state.consume_literal("True")) {
        out = picojson::value(true);
        return true;
    }
    if (state.consume_literal("False")) {
        out = picojson::value(false);
        return true;
    }
    if (state.consume_literal("None") || state.consume_literal("null")) {
        out = picojson::value();
        return true;
    }

    return false;
}

static inline bool parse_lfm2_call(const std::string& entry, std::string& out_json_call) {
    std::string trimmed_entry = trim_lfm2_slice(entry, 0, entry.size());
    if (trimmed_entry.empty()) return false;

    LFM2ToolCallParserState state(trimmed_entry);
    std::string func_name;
    if (!parse_lfm2_identifier(state, func_name)) return false;
    if (!state.consume_if('(')) return false;

    picojson::object args;
    state.skip_whitespace();
    if (!state.consume_if(')')) {
        while (true) {
            std::string arg_name;
            if (!parse_lfm2_identifier(state, arg_name)) return false;
            if (!state.consume_if('=')) return false;

            picojson::value arg_value;
            if (!parse_lfm2_value(state, arg_value)) return false;
            args[arg_name] = arg_value;

            state.skip_whitespace();
            if (state.consume_if(')')) break;
            if (!state.consume_if(',')) return false;
        }
    }

    state.skip_whitespace();
    if (!state.at_end()) return false;

    picojson::object call;
    call["name"] = picojson::value(func_name);
    call["arguments"] = picojson::value(args);
    out_json_call = picojson::value(call).serialize();
    return true;
}

struct LFM2TopLevelParserState {
    const std::string& text;
    size_t pos = 0;
    int paren_depth = 0;
    int bracket_depth = 0;
    int brace_depth = 0;
    bool in_string = false;
    bool escape = false;
    size_t obj_end_pos = 0;

    LFM2TopLevelParserState(const std::string& text) : text(text) {}

    bool at_end() const { return pos >= text.size(); }

    void skip_whitespace() {
        while (!at_end() && std::isspace(static_cast<unsigned char>(text[pos]))) {
            pos++;
        }
    }

    bool starts_with_json_object_list() {
        skip_whitespace();
        return !at_end() && text[pos] == '{';
    }

    bool next_tool_call(std::string& out_json) {
        size_t pos = obj_end_pos;
        while (pos < text.size() && text[pos] != '{') {
            pos++;
        }
        if (pos >= text.size() || text[pos] != '{') return false;

        int brace_depth = 1;
        size_t obj_start = pos;
        pos++;
        in_string = false;
        escape = false;
        while (pos < text.size() && brace_depth > 0) {
            char c = text[pos];
            if (escape) {
                escape = false;
            } else if (in_string) {
                if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    in_string = false;
                }
            } else {
                if (c == '"') {
                    in_string = true;
                } else if (c == '{') {
                    brace_depth++;
                } else if (c == '}') {
                    brace_depth--;
                }
            }
            pos++;
        }
        if (brace_depth == 0) {
            out_json = text.substr(obj_start, pos - obj_start);
            obj_end_pos = pos;
            return true;
        }
        return false;
    }

    std::string next_entry() {
        skip_whitespace();
        const size_t start = pos;

        while (!at_end()) {
            char c = text[pos++];
            if (in_string) {
                if (escape) {
                    escape = false;
                } else if (c == '\\') {
                    escape = true;
                } else if (c == '"') {
                    in_string = false;
                }
                continue;
            }

            if (c == '"') {
                in_string = true;
            } else if (c == '(') {
                paren_depth++;
            } else if (c == ')' && paren_depth > 0) {
                paren_depth--;
            } else if (c == '[') {
                bracket_depth++;
            } else if (c == ']' && bracket_depth > 0) {
                bracket_depth--;
            } else if (c == '{') {
                brace_depth++;
            } else if (c == '}' && brace_depth > 0) {
                brace_depth--;
            } else if (c == ',' && paren_depth == 0 && bracket_depth == 0 && brace_depth == 0) {
                return trim_lfm2_slice(text, start, pos - 1);
            }
        }

        return trim_lfm2_slice(text, start, pos);
    }
};

static inline void parse_lfm2_function_calls(const std::string& tool_content,
                                             std::vector<std::string>& function_calls) {
    std::string content = trim_lfm2_slice(tool_content, 0, tool_content.size());

    auto append_call_if_valid = [&](const std::string& entry) {
        std::string json_call;
        if (parse_lfm2_call(entry, json_call)) {
            function_calls.push_back(json_call);
        }
    };

    if (!content.empty() && content.front() == '[' && content.back() == ']') {
        std::string inner = content.substr(1, content.size() - 2);
        LFM2TopLevelParserState state(inner);

        if (state.starts_with_json_object_list()) {
            std::string json_obj;
            while (state.next_tool_call(json_obj)) {
                if (json_obj.find("\"name\"") != std::string::npos) {
                    function_calls.push_back(json_obj);
                }
            }
            return;
        }

        while (!state.at_end()) {
            std::string entry = state.next_entry();
            if (!entry.empty()) append_call_if_valid(entry);
        }
    } else if (!content.empty()) {
        append_call_if_valid(content);
    }
}

}  // namespace ffi
}  // namespace cactus

#endif
