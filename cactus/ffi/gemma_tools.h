#pragma once

#include "../engine/engine.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace gemma {

inline std::string to_upper(const std::string& s) {
    std::string result = s;
    for (auto& c : result) c = std::toupper(c);
    return result;
}

inline std::string escape(const std::string& s) {
    return "<escape>" + s + "<escape>";
}

inline void append_field(std::string& result, bool& first, const std::string& field) {
    if (!first) result += ",";
    first = false;
    result += field;
}

inline std::string format_literal(const picojson::value& value) {
    if (value.is<std::string>()) {
        return escape(value.get<std::string>());
    }
    if (value.is<picojson::array>()) {
        std::string result = "[";
        bool first = true;
        for (const auto& item : value.get<picojson::array>()) {
            if (!first) result += ",";
            first = false;
            result += format_literal(item);
        }
        result += "]";
        return result;
    }
    if (value.is<picojson::object>()) {
        std::string result = "{";
        bool first = true;
        for (const auto& [key, item] : value.get<picojson::object>()) {
            if (!first) result += ",";
            first = false;
            result += escape(key) + ":" + format_literal(item);
        }
        result += "}";
        return result;
    }
    return value.serialize();
}

inline std::string format_required(const picojson::value& required_value) {
    const auto& required = required_value.get<picojson::array>();
    std::string result = "[";
    bool first = true;
    for (const auto& item : required) {
        if (!first) result += ",";
        first = false;
        result += escape(item.get<std::string>());
    }
    result += "]";
    return result;
}

inline std::string format_schema_body(const picojson::object& schema);

inline std::string format_schema_properties(const picojson::object& properties) {
    std::string result;
    bool first = true;
    for (const auto& [key, property_value] : properties) {
        const auto& property = property_value.get<picojson::object>();
        if (!first) result += ",";
        first = false;
        result += key + ":{" + format_schema_body(property) + "}";
    }
    return result;
}

inline std::string format_schema_body(const picojson::object& schema) {
    std::string result;
    bool first = true;

    if (schema.count("description") && schema.at("description").is<std::string>()) {
        append_field(result, first, "description:" + escape(schema.at("description").get<std::string>()));
    }

    auto properties_it = schema.find("properties");
    if (properties_it != schema.end()) {
        append_field(
            result,
            first,
            "properties:{" + format_schema_properties(properties_it->second.get<picojson::object>()) + "}"
        );
    }

    auto required_it = schema.find("required");
    if (required_it != schema.end()) {
        append_field(result, first, "required:" + format_required(required_it->second));
    }

    auto enum_it = schema.find("enum");
    if (enum_it != schema.end()) {
        append_field(result, first, "enum:" + format_literal(enum_it->second));
    }

    auto items_it = schema.find("items");
    if (items_it != schema.end()) {
        append_field(
            result,
            first,
            "items:{" + format_schema_body(items_it->second.get<picojson::object>()) + "}"
        );
    }

    static const std::vector<std::string> reserved_keys = {
        "description", "type", "properties", "required", "enum", "items", "additionalProperties", "nullable"
    };
    for (const auto& [key, value] : schema) {
        if (std::find(reserved_keys.begin(), reserved_keys.end(), key) != reserved_keys.end()) {
            continue;
        }
        append_field(result, first, key + ":" + format_literal(value));
    }

    if (schema.count("type") && schema.at("type").is<std::string>()) {
        append_field(result, first, "type:" + escape(to_upper(schema.at("type").get<std::string>())));
    }

    return result;
}

inline std::string format_function_declaration(const cactus::engine::ToolDefinition& tool) {
    std::string result = "declaration:" + tool.name + "{";
    result += "description:" + escape(tool.description);

    const auto& schema = tool.arguments_schema.get<picojson::object>();
    if (!schema.empty()) {
        result += ",parameters:{";
        result += format_schema_body(schema);
        result += "}";
    }

    result += "}";
    return result;
}

inline std::string format_tools(const std::vector<cactus::engine::ToolDefinition>& tools, bool use_pipe_tags = false) {
    if (tools.empty()) return "";

    const char* decl_start = use_pipe_tags ? "<|tool>" : "<start_function_declaration>";
    const char* decl_end = use_pipe_tags ? "<tool|>" : "<end_function_declaration>";

    std::string result;
    for (const auto& tool : tools) {
        result += decl_start;
        result += format_function_declaration(tool);
        result += decl_end;
    }
    return result;
}

inline size_t match_quote_tag(const std::string& s, size_t pos) {
    if (s.compare(pos, 8, "<escape>") == 0) return 8;
    if (s.compare(pos, 5, "<|\"|>") == 0) return 5;
    return 0;
}

inline size_t find_quote_tag(const std::string& s, size_t pos) {
    size_t e = s.find("<escape>", pos);
    size_t t = s.find("<|\"|>", pos);
    if (e == std::string::npos) return t;
    if (t == std::string::npos) return e;
    return std::min(e, t);
}

inline std::string unescape(const std::string& s) {
    const std::string ESCAPE_TAG = "<escape>";
    std::string result = s;
    size_t pos = 0;
    while ((pos = result.find(ESCAPE_TAG, pos)) != std::string::npos) {
        result.erase(pos, ESCAPE_TAG.length());
    }
    return result;
}

inline std::string args_to_json(const std::string& args_content) {
    std::string result = "{";
    size_t pos = 0;
    bool first = true;

    if (!args_content.empty() && args_content[0] == '{') pos = 1;

    while (pos < args_content.length()) {
        while (pos < args_content.length() && std::isspace(static_cast<unsigned char>(args_content[pos]))) pos++;
        if (pos >= args_content.length() || args_content[pos] == '}') break;
        if (args_content[pos] == ',') { pos++; continue; }

        size_t key_start = pos;
        while (pos < args_content.length() && args_content[pos] != ':') pos++;
        std::string key = args_content.substr(key_start, pos - key_start);
        if (pos < args_content.length()) pos++;

        std::string value;
        while (pos < args_content.length() && std::isspace(static_cast<unsigned char>(args_content[pos]))) pos++;

        if (pos < args_content.length()) {
            size_t qtag_len = match_quote_tag(args_content, pos);
            if (qtag_len > 0) {
                pos += qtag_len;
                size_t val_end = find_quote_tag(args_content, pos);
                if (val_end != std::string::npos) {
                    value = "\"" + args_content.substr(pos, val_end - pos) + "\"";
                    pos = val_end + match_quote_tag(args_content, val_end);
                }
            } else if (args_content[pos] == '{') {
                int depth = 1;
                size_t start = pos;
                pos++;
                while (pos < args_content.length() && depth > 0) {
                    if (args_content[pos] == '{') depth++;
                    else if (args_content[pos] == '}') depth--;
                    pos++;
                }
                value = args_to_json(args_content.substr(start, pos - start));
            } else if (args_content[pos] == '[') {
                int depth = 1;
                size_t start = pos;
                pos++;
                while (pos < args_content.length() && depth > 0) {
                    if (args_content[pos] == '[') depth++;
                    else if (args_content[pos] == ']') depth--;
                    pos++;
                }
                std::string arr_content = args_content.substr(start + 1, pos - start - 2);
                value = "[";
                size_t arr_pos = 0;
                bool first_item = true;
                while (arr_pos < arr_content.length()) {
                    while (arr_pos < arr_content.length() &&
                           (std::isspace(static_cast<unsigned char>(arr_content[arr_pos])) || arr_content[arr_pos] == ',')) {
                        arr_pos++;
                    }
                    if (arr_pos >= arr_content.length()) break;

                    if (!first_item) value += ",";
                    first_item = false;

                    size_t aq_len = match_quote_tag(arr_content, arr_pos);
                    if (aq_len > 0) {
                        arr_pos += aq_len;
                        size_t end = find_quote_tag(arr_content, arr_pos);
                        if (end != std::string::npos) {
                            value += "\"" + arr_content.substr(arr_pos, end - arr_pos) + "\"";
                            arr_pos = end + match_quote_tag(arr_content, end);
                        }
                    } else {
                        size_t end = arr_content.find_first_of(",]", arr_pos);
                        if (end == std::string::npos) end = arr_content.length();
                        value += arr_content.substr(arr_pos, end - arr_pos);
                        arr_pos = end;
                    }
                }
                value += "]";
            } else {
                size_t val_start = pos;
                while (pos < args_content.length() && args_content[pos] != ',' && args_content[pos] != '}') {
                    pos++;
                }
                value = args_content.substr(val_start, pos - val_start);
                while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) value.pop_back();
            }
        }

        if (!first) result += ",";
        first = false;
        result += "\"" + key + "\":" + value;
    }

    result += "}";
    return result;
}

inline void parse_function_calls(std::string& response, std::vector<std::string>& function_calls) {
    const std::string CALL_START = (response.find("<|tool_call>") != std::string::npos)
        ? "<|tool_call>" : "<start_function_call>";
    const std::string CALL_END = (CALL_START == "<|tool_call>")
        ? "<tool_call|>" : "<end_function_call>";
    size_t pos = 0;

    while ((pos = response.find(CALL_START, pos)) != std::string::npos) {
        size_t content_start = pos + CALL_START.length();
        size_t call_end_pos = response.find(CALL_END, content_start);

        size_t content_end = (call_end_pos != std::string::npos) ? call_end_pos : response.length();
        std::string call_content = response.substr(content_start, content_end - content_start);

        if (call_content.compare(0, 5, "call:") == 0) {
            size_t brace_pos = call_content.find('{');

            if (brace_pos == std::string::npos) {
                size_t sep_pos = call_content.find_first_of(", ", 5);
                if (sep_pos != std::string::npos) {
                    std::string func_name = call_content.substr(5, sep_pos - 5);
                    size_t args_start = sep_pos + 1;
                    while (args_start < call_content.length() &&
                           (call_content[args_start] == ' ' || call_content[args_start] == ',')) {
                        args_start++;
                    }
                    std::string args_content = "{" + call_content.substr(args_start);
                    if (args_content.back() != '}') args_content += "}";

                    std::string args_json = args_to_json(args_content);
                    std::string json_call = "{\"name\":\"" + func_name + "\",\"arguments\":" + args_json + "}";
                    function_calls.push_back(json_call);
                }
            } else {
                std::string func_name = call_content.substr(5, brace_pos - 5);
                std::string args_content = call_content.substr(brace_pos);
                if (args_content.back() != '}') args_content += "}";

                std::string args_json = args_to_json(args_content);
                std::string json_call = "{\"name\":\"" + func_name + "\",\"arguments\":" + args_json + "}";
                function_calls.push_back(json_call);
            }
        }

        size_t erase_end = (call_end_pos != std::string::npos)
            ? call_end_pos + CALL_END.length() : response.length();
        response.erase(pos, erase_end - pos);
    }
}

} // namespace gemma
