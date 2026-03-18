#include "test_utils.h"
#include <cstddef>
#include <picojson/picojson.h>
#include <cctype>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <thread>
#include <chrono>

#if __has_include(<curl/curl.h>)
#include <curl/curl.h>
#define CACTUS_ENGINE_TEST_HAS_CURL 1
#else
#define CACTUS_ENGINE_TEST_HAS_CURL 0
#endif

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");

static const char* g_options = R"({
        "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
    })";

template<typename TestFunc>
bool run_test(const char* title, const char* messages, TestFunc test_logic,
              const char* tools = nullptr, int stop_at = -1) {
    return EngineTestUtils::run_test(title, g_model_path, messages, g_options, test_logic, tools, stop_at);
}

bool test_streaming() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "      STREAMING & FOLLOW-UP TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* messages1 = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "My name is Henry Ndubuaku, how are you?"}
    ])";

    StreamingData data1;
    data1.model = model;
    char response1[4096];

    std::cout << "\n[Turn 1]\n";
    std::cout << "User: My name is Henry Ndubuaku, how are you?\n";
    std::cout << "Assistant: ";

    int result1 = cactus_complete(model, messages1, response1, sizeof(response1),
                                 g_options, nullptr, stream_callback, &data1, nullptr);

    std::cout << "\n\n[Results - Turn 1]\n";
    Metrics metrics1;
    metrics1.parse(response1);
    metrics1.print_json();

    bool success1 = result1 > 0 && data1.token_count > 0;

    if (!success1) {
        std::cout << "└─ Status: FAILED ✗\n";
        cactus_destroy(model);
        return false;
    }

    std::string assistant_response;
    for(const auto& token : data1.tokens) {
        assistant_response += token;
    }

    std::string messages2_str = R"([
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "My name is Henry Ndubuaku, how are you?"},
        {"role": "assistant", "content": ")" + escape_json(assistant_response) + R"("},
        {"role": "user", "content": "What is my name?"}
    ])";

    StreamingData data2;
    data2.model = model;
    char response2[4096];

    std::cout << "\n[Turn 2]\n";
    std::cout << "User: What is my name?\n";
    std::cout << "Assistant: ";

    int result2 = cactus_complete(model, messages2_str.c_str(), response2, sizeof(response2),
                                 g_options, nullptr, stream_callback, &data2, nullptr);

    std::cout << "\n\n[Results - Turn 2]\n";
    Metrics metrics2;
    metrics2.parse(response2);
    metrics2.print_json();

    bool success2 = result2 > 0 && data2.token_count > 0;

    cactus_destroy(model);
    return success1 && success2;
}

bool test_tool_call() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("TOOL CALL TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What's the weather in San Francisco?");
}

bool test_multiple_tool_call_invocations() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Send a message to Blob and get the weather for San Francisco."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Name of the person to send the message to"},
                    "message": {"type": "string", "description": "The message content to send"}
                },
                "required": ["recipient", "message"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("MULTIPLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_weather_tool = has_function
                && (response.find("\"name\":\"get_weather\"") != std::string::npos
                    || response.find("\"name\": \"get_weather\"") != std::string::npos);
            bool has_message_tool = has_function
                && (response.find("\"name\":\"send_message\"") != std::string::npos
                    || response.find("\"name\": \"send_message\"") != std::string::npos);
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_weather_tool && has_message_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_weather_tool && has_message_tool;
        }, tools, -1, "Send a message to Blob and get the weather for San Francisco.");
}

bool test_tool_call_with_three_tools() {
    const char* messages = R"([
        {"role": "system", "content": "You are a helpful assistant that can use tools."},
        {"role": "user", "content": "Send a message to John saying hello."}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, State, Country"}
                },
                "required": ["location"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "set_alarm",
            "description": "Set an alarm for a given time",
            "parameters": {
                "type": "object",
                "properties": {
                    "hour": {"type": "integer", "description": "Hour to set the alarm for"},
                    "minute": {"type": "integer", "description": "Minute to set the alarm for"}
                },
                "required": ["hour", "minute"]
            }
        }
    }, {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a contact",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Name of the person to send the message to"},
                    "message": {"type": "string", "description": "The message content to send"}
                },
                "required": ["recipient", "message"]
            }
        }
    }])";

    const char* options_with_force_tools = R"({
        "max_tokens": 256,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "force_tools": true
    })";

    return EngineTestUtils::run_test("TRIPLE TOOLS TEST", g_model_path, messages, options_with_force_tools,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("send_message") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "Send a message to John saying hello.");
}

bool test_1k_context() {
    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < 50; i++) {
        msg += "Context " + std::to_string(i) + ": Background knowledge. ";
    }
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < 50; i++) {
        msg += "Data " + std::to_string(i) + " = " + std::to_string(i * 3.14159) + ". ";
    }
    msg += "Explain the data.\"}]";

    return run_test("1K CONTEXT TEST", msg.c_str(),
        [](int result, const StreamingData&, const std::string&, const Metrics& m) {
            m.print_json();
            return result > 0;
        }, nullptr, 100);
}

bool is_valid_json_document(const std::string& text, std::string& error) {
    picojson::value value;
    auto begin = text.begin();
    auto end = picojson::parse(value, begin, text.end(), &error);
    if (!error.empty()) {
        return false;
    }
    while (end != text.end() && std::isspace(static_cast<unsigned char>(*end))) {
        ++end;
    }
    if (end != text.end()) {
        error = "Trailing characters after JSON document";
        return false;
    }
    return true;
}

bool test_json_grammar_outputs_valid_json() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "      JSON GRAMMAR TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    cactus_grammar_t grammar = cactus_grammar_init_json();
    if (!grammar) {
        std::cerr << "[✗] Failed to initialize JSON grammar\n";
        cactus_destroy(model);
        return false;
    }

    const std::vector<std::string> prompts = {
        "Tell me three short facts about otters.",
        "Write a haiku about the moon.",
        "What is 17 multiplied by 19?"
    };

    bool success = true;

    for (size_t i = 0; i < prompts.size(); ++i) {
        std::string messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ")" + escape_json(prompts[i]) + R"("}
    ])";

        StreamingData data;
        data.model = model;
        char response[4096];

        std::cout << "\n[Prompt " << (i + 1) << "]\n";
        std::cout << "User: " << prompts[i] << "\n";
        std::cout << "Assistant: ";

        int result = cactus_complete(model, messages.c_str(), response, sizeof(response),
                                     g_options, nullptr, stream_callback, &data, grammar);

        std::string output;
        for (const auto& token : data.tokens) {
            output += token;
        }

        std::string json_error;
        bool valid_json = is_valid_json_document(output, json_error);

        std::cout << "\n\n[Results - Prompt " << (i + 1) << "]\n";
        Metrics metrics;
        metrics.parse(response);
        std::cout << "├─ Valid JSON: " << (valid_json ? "YES" : "NO") << "\n";
        if (!valid_json) {
            std::cout << "├─ JSON error: " << json_error << "\n";
            std::cout << "├─ Raw output: " << output << "\n";
        }
        metrics.print_json();

        success = success && result > 0 && data.token_count > 0 && valid_json;
    }

    cactus_grammar_destroy(grammar);
    cactus_destroy(model);
    return success;
}

bool test_regex_grammar_outputs_address() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "     REGEX ADDRESS TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* address_regex = R"(^[0-9]{3,5} [A-Za-z0-9.'-]+( [A-Za-z0-9.'-]+){0,4}, [A-Za-z.'-]+( [A-Za-z.'-]+){0,2}, [A-Z]{2} [0-9]{5}$)";
    cactus_grammar_t grammar = cactus_grammar_init_regex(address_regex);
    if (!grammar) {
        std::cerr << "[✗] Failed to initialize regex grammar\n";
        cactus_destroy(model);
        return false;
    }

    const char* prompt = "What is the address of the Space Needle in Seattle?";
    std::string messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ")" + escape_json(prompt) + R"("}
    ])";

    StreamingData data;
    data.model = model;
    char response[4096];

    std::cout << "\n[Prompt]\n";
    std::cout << "User: " << prompt << "\n";
    std::cout << "Assistant: ";

    int result = cactus_complete(model, messages.c_str(), response, sizeof(response),
                                 g_options, nullptr, stream_callback, &data, grammar);

    std::string output;
    for (const auto& token : data.tokens) {
        output += token;
    }

    bool matches_regex = !output.empty()
        && output.find(',') != std::string::npos
        && output.find("Seattle") != std::string::npos
        && output.find("WA") != std::string::npos;

    std::cout << "\n\n[Results]\n";
    Metrics metrics;
    metrics.parse(response);
    std::cout << "├─ Regex match: " << (matches_regex ? "YES" : "NO") << "\n";
    if (!matches_regex) {
        std::cout << "├─ Raw output: " << output << "\n";
    }
    metrics.print_json();

    cactus_grammar_destroy(grammar);
    cactus_destroy(model);
    return result > 0 && data.token_count > 0 && matches_regex;
}

bool test_json_schema_grammar_outputs_person() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "    JSON SCHEMA PERSON TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* person_schema = R"({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0, "maximum": 130},
            "bio": {"type": "string"}
        },
        "required": ["name", "age", "bio"],
        "additionalProperties": false
    })";

    cactus_grammar_t grammar = cactus_grammar_init_json_schema(person_schema, CACTUS_GRAMMAR_JSON_SCHEMA_DEFAULT_OPTIONS);
    if (!grammar) {
        std::cerr << "[✗] Failed to initialize JSON Schema grammar\n";
        cactus_destroy(model);
        return false;
    }

    const char* prompt = "Create a fictional founder who lives in San Francisco.";
    std::string messages = R"([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": ")" + escape_json(prompt) + R"("}
    ])";

    StreamingData data;
    data.model = model;
    char response[4096];

    std::cout << "\n[Prompt]\n";
    std::cout << "User: " << prompt << "\n";
    std::cout << "Assistant: ";

    int result = cactus_complete(model, messages.c_str(), response, sizeof(response),
                                 g_options, nullptr, stream_callback, &data, grammar);

    std::string output;
    for (const auto& token : data.tokens) {
        output += token;
    }

    std::string json_error;
    bool valid_json = is_valid_json_document(output, json_error);
    bool has_name = false;
    bool has_age = false;
    bool age_is_integer = false;
    bool has_bio = false;

    if (valid_json) {
        picojson::value value;
        auto begin = output.begin();
        picojson::parse(value, begin, output.end(), &json_error);
        if (value.is<picojson::object>()) {
            has_name = value.contains("name") && value.get("name").is<std::string>()
                && !value.get("name").get<std::string>().empty();
            has_age = value.contains("age") && value.get("age").is<double>();
            has_bio = value.contains("bio") && value.get("bio").is<std::string>()
                && !value.get("bio").get<std::string>().empty();

            if (has_age) {
                double age_value = value.get("age").get<double>();
                age_is_integer = age_value >= 0.0
                    && age_value <= 130.0
                    && std::floor(age_value) == age_value;
            }
        }
    }

    std::cout << "\n\n[Results]\n";
    Metrics metrics;
    metrics.parse(response);
    std::cout << "├─ Valid JSON: " << (valid_json ? "YES" : "NO") << "\n"
              << "├─ Has name: " << (has_name ? "YES" : "NO") << "\n"
              << "├─ Has age: " << (has_age ? "YES" : "NO") << "\n"
              << "├─ Age is integer: " << (age_is_integer ? "YES" : "NO") << "\n"
              << "├─ Has bio: " << (has_bio ? "YES" : "NO") << "\n";
    if (!valid_json) {
        std::cout << "├─ JSON error: " << json_error << "\n";
    }
    if (!valid_json || !has_name || !has_age || !age_is_integer || !has_bio) {
        std::cout << "├─ Raw output: " << output << "\n";
    }
    metrics.print_json();

    cactus_grammar_destroy(grammar);
    cactus_destroy(model);
    return result > 0 && data.token_count > 0 && valid_json && has_name && has_age && age_is_integer && has_bio;
}

int main() {
    TestUtils::TestRunner runner("LLM Tests");
    // runner.run_test("1k_context", test_1k_context());
    // runner.run_test("streaming", test_streaming());
    // runner.run_test("json_grammar_outputs_valid_json", test_json_grammar_outputs_valid_json());
    runner.run_test("regex_grammar_outputs_address", test_regex_grammar_outputs_address());
    // runner.run_test("json_schema_grammar_outputs_person", test_json_schema_grammar_outputs_person());
    // runner.run_test("tool_calls", test_tool_call());
    // runner.run_test("tool_multiple_tool_call_invocations", test_multiple_tool_call_invocations());
    // runner.run_test("tool_calls_with_three_tools", test_tool_call_with_three_tools());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
