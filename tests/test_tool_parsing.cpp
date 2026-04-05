#include "test_utils.h"

#include "../cactus/ffi/cactus_utils.h"

static bool test_lfm2_parser_preserves_mixed_argument_types() {
    std::string response =
        "prelude<|tool_call_start|>[emit_blob(payload={\"type\":\"blob\",\"meta\":{\"enabled\":True,\"note\":None}},otinet=True,flag=False,message=\"hello\",count=7,ratio=-1.25e2,empty=null,items=[\"A\",2,{\"inner\":\"v\"}])]<|tool_call_end|>epilogue";
    std::string regular_response;
    std::vector<std::string> function_calls;
    cactus::ffi::parse_function_calls_from_response(response, regular_response, function_calls);

    if (function_calls.size() != 1) return false;
    if (regular_response != "preludeepilogue") return false;

    picojson::value parsed;
    std::string error;
    picojson::parse(parsed, function_calls[0].begin(), function_calls[0].end(), &error);
    if (!error.empty() || !parsed.is<picojson::object>()) return false;

    const picojson::value& args = parsed.get("arguments");

    auto has = [](const picojson::value& value, const char* key) {
        return value.is<picojson::object>() && value.contains(key);
    };

    if (!parsed.contains("name") || !parsed.get("name").is<std::string>()) return false;
    if (parsed.get("name").get<std::string>() != "emit_blob") return false;
    if (!args.is<picojson::object>()) return false;

    if (!has(args, "payload") || !args.get("payload").is<picojson::object>()) return false;
    const picojson::value& payload = args.get("payload");
    if (!has(payload, "type") || !payload.get("type").is<std::string>()) return false;
    if (payload.get("type").get<std::string>() != "blob") return false;
    if (!has(payload, "meta") || !payload.get("meta").is<picojson::object>()) return false;
    const picojson::value& meta = payload.get("meta");
    if (!has(meta, "enabled") || !meta.get("enabled").is<bool>() || !meta.get("enabled").get<bool>()) return false;
    if (!has(meta, "note") || !meta.get("note").is<picojson::null>()) return false;

    if (!has(args, "otinet") || !args.get("otinet").is<bool>() || !args.get("otinet").get<bool>()) return false;
    if (!has(args, "flag") || !args.get("flag").is<bool>() || args.get("flag").get<bool>()) return false;
    if (!has(args, "message") || !args.get("message").is<std::string>()) return false;
    if (args.get("message").get<std::string>() != "hello") return false;
    if (!has(args, "count") || !args.get("count").is<double>() || args.get("count").get<double>() != 7.0) return false;
    if (!has(args, "ratio") || !args.get("ratio").is<double>() || args.get("ratio").get<double>() != -125.0) return false;
    if (!has(args, "empty") || !args.get("empty").is<picojson::null>()) return false;

    if (!has(args, "items") || !args.get("items").is<picojson::array>()) return false;
    const picojson::value& items = args.get("items");
    if (!items.contains(0) || !items.contains(1) || !items.contains(2) || items.contains(3)) return false;
    if (!items.get(0).is<std::string>() || items.get(0).get<std::string>() != "A") return false;
    if (!items.get(1).is<double>() || items.get(1).get<double>() != 2.0) return false;
    if (!items.get(2).is<picojson::object>() || !items.get(2).contains("inner")) return false;
    if (!items.get(2).get("inner").is<std::string>() || items.get(2).get("inner").get<std::string>() != "v") return false;

    return !args.contains("unset") && !args.contains("config");
}

int main() {
    TestUtils::TestRunner runner("Tool Parsing Tests");
    runner.run_test("lfm2_mixed_argument_types", test_lfm2_parser_preserves_mixed_argument_types());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
