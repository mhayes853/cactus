#include "test_utils.h"
#include <fstream>
#include <cstdlib>
#include <iostream>

using namespace EngineTestUtils;

static const char* g_model_path = std::getenv("CACTUS_TEST_MODEL");
static const char* g_assets_path = std::getenv("CACTUS_TEST_ASSETS");

static const char* g_options = R"({
        "max_tokens": 256,
    "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
    "telemetry_enabled": false
    })";

bool test_vlm_multiturn() {
    std::string model_path_str(g_model_path ? g_model_path : "");

    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "Skipping VLM multi-turn test: vision weights not found." << std::endl;
        return true;
    }
    vf.close();

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║       VLM MULTI-TURN TEST                ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "Failed to initialize model for VLM multi-turn test" << std::endl;
        return false;
    }

    std::string img_path = std::string(g_assets_path) + "/test_monkey.png";

    std::string messages1 = "[{\"role\": \"user\", "
        "\"content\": \"Describe what is happening in this image in two sentences.\", "
        "\"images\": [\"" + img_path + "\"]}]";

    StreamingData stream_data1;
    stream_data1.model = model;
    char response1[4096];

    std::cout << "\n[Turn 1]\n";
    std::cout << "User: Describe what is happening in this image in two sentences.\n";
    std::cout << "Assistant: ";

    int result1 = cactus_complete(model, messages1.c_str(), response1, sizeof(response1),
                                  g_options, nullptr, stream_callback, &stream_data1, nullptr, 0, nullptr);

    std::cout << "\n\n[Results - Turn 1]\n";
    Metrics metrics1;
    metrics1.parse(response1);
    metrics1.print_json();

    bool success1 = result1 > 0 && stream_data1.token_count > 0;

    if (!success1) {
        std::cout << "└─ Status: FAILED ✗\n";
        cactus_destroy(model);
        return false;
    }

    std::string assistant_response;
    for (const auto& token : stream_data1.tokens) {
        assistant_response += token;
    }

    std::string messages2 = "[{\"role\": \"user\", "
        "\"content\": \"Describe what is happening in this image in two sentences.\", "
        "\"images\": [\"" + img_path + "\"]}, "
        "{\"role\": \"assistant\", \"content\": \"" + escape_json(assistant_response) + "\"}, "
        "{\"role\": \"user\", \"content\": \"Describe the image once again.\"}]";

    StreamingData stream_data2;
    stream_data2.model = model;
    char response2[4096];

    std::cout << "\n[Turn 2]\n";
    std::cout << "User: Describe the image once again.\n";
    std::cout << "Assistant: ";

    int result2 = cactus_complete(model, messages2.c_str(), response2, sizeof(response2),
                                  g_options, nullptr, stream_callback, &stream_data2, nullptr, 0, nullptr);

    std::cout << "\n\n[Results - Turn 2]\n";
    Metrics metrics2;
    metrics2.parse(response2);
    metrics2.print_json();

    bool success2 = result2 > 0 && stream_data2.token_count > 0;

    if (!success2) {
        std::cout << "└─ Status: FAILED ✗ (Follow-up message failed)\n";
    }

    cactus_destroy(model);
    return success1 && success2;
}

bool test_prefill_invalidated_on_message_change_vlm() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << " PREFILL INVALIDATION (VLM) TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    std::string model_path_str(g_model_path ? g_model_path : "");
    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "⊘ SKIP │ Vision weights not found\n";
        return true;
    }
    vf.close();

    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    std::string prefill_img_path = std::string(g_assets_path) + "/test_monkey.png";
    std::ifstream imgf(prefill_img_path);
    if (!imgf.good()) {
        std::cout << "⊘ SKIP │ test_monkey.png not found\n";
        return true;
    }
    imgf.close();

    std::string complete_img_path = std::string(g_assets_path) + "/test_thing.png";
    std::ifstream imgf2(complete_img_path);
    if (!imgf2.good()) {
        std::cout << "⊘ SKIP │ test_thing.png not found\n";
        return true;
    }
    imgf2.close();

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string prefill_messages = "[{\"role\": \"user\", \"content\": \"Describe this image in one short sentence.\", \"images\": [\""
        + prefill_img_path + "\"]}]";

    std::string complete_messages = "[{\"role\": \"user\", \"content\": \"Describe this image in one short sentence.\", \"images\": [\""
        + complete_img_path + "\"]}]";

    const char* options = R"({
        "max_tokens": 128,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "confidence_threshold": 0.0,
        "telemetry_enabled": false
    })";

    char prefill_response[2048] = {0};
    int prefill_result = cactus_prefill(model, prefill_messages.c_str(), prefill_response, sizeof(prefill_response), nullptr, nullptr, nullptr, 0);
    PrefillMetrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response_warm[4096] = {0};
    int complete_result_warm = cactus_complete(model, complete_messages.c_str(), complete_response_warm, sizeof(complete_response_warm),
                                               options, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    Metrics warm_metrics;
    warm_metrics.parse(complete_response_warm);

    cactus_reset(model);

    char complete_response_cold[4096] = {0};
    int complete_result_cold = cactus_complete(model, complete_messages.c_str(), complete_response_cold, sizeof(complete_response_cold),
                                               options, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    Metrics cold_metrics;
    cold_metrics.parse(complete_response_cold);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill success: " << ((prefill_result > 0 && prefill_metrics.success) ? "YES" : "NO") << "\n"
              << "├─ Complete(warm mismatched) prefill_tokens: " << warm_metrics.prefill_tokens << "\n"
              << "├─ Complete(cold) prefill_tokens: " << cold_metrics.prefill_tokens << "\n";

    bool all_success = prefill_result > 0 && prefill_metrics.success
        && complete_result_warm > 0 && warm_metrics.success
        && complete_result_cold > 0 && cold_metrics.success;
    bool invalidated = warm_metrics.prefill_tokens == cold_metrics.prefill_tokens;

    std::cout << "├─ Calls successful: " << (all_success ? "YES" : "NO") << "\n"
              << "└─ Mismatch invalidated cache: " << (invalidated ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return all_success && invalidated;
}

bool test_prefill_with_images() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "      PREFILL WITH IMAGES TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    std::string model_path_str(g_model_path ? g_model_path : "");
    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "⊘ SKIP │ Vision weights not found\n";
        return true;
    }
    vf.close();

    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    std::string prefill_img_path = std::string(g_assets_path) + "/test_monkey.png";
    std::ifstream imgf(prefill_img_path);
    if (!imgf.good()) {
        std::cout << "⊘ SKIP │ test_monkey.png not found\n";
        return true;
    }
    imgf.close();

    std::string extension_img_path = std::string(g_assets_path) + "/test_thing.png";
    std::ifstream imgf2(extension_img_path);
    if (!imgf2.good()) {
        std::cout << "⊘ SKIP │ test_thing.png not found\n";
        return true;
    }
    imgf2.close();

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string prefill_messages = "["
        "{\"role\": \"system\", \"content\": \"You are a helpful assistant. Be concise.\"},"
        "{\"role\": \"user\", \"content\": \"Describe this image in one short sentence.\", \"images\": [\""
        + prefill_img_path + "\"]},"
        "{\"role\": \"assistant\", \"content\": \"This image shows a close-up of a monkey face.\"}"
        "]";

    std::string complete_messages = "["
        "{\"role\": \"system\", \"content\": \"You are a helpful assistant. Be concise.\"},"
        "{\"role\": \"user\", \"content\": \"Describe this image in one short sentence.\", \"images\": [\""
        + prefill_img_path + "\"]},"
        "{\"role\": \"assistant\", \"content\": \"This image shows a close-up of a monkey face.\"},"
        "{\"role\": \"user\", \"content\": \"Now describe this second image in one short sentence.\", \"images\": [\""
        + extension_img_path + "\"]}"
        "]";

    const char* options = R"({
        "max_tokens": 128,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "confidence_threshold": 0.0,
        "telemetry_enabled": false
    })";

    char prefill_response[2048] = {0};
    int prefill_result = cactus_prefill(model, prefill_messages.c_str(), prefill_response, sizeof(prefill_response), nullptr, nullptr, nullptr, 0);
    PrefillMetrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response_warm[4096] = {0};
    int complete_result_warm = cactus_complete(model, complete_messages.c_str(), complete_response_warm, sizeof(complete_response_warm),
                                               options, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    Metrics warm_metrics;
    warm_metrics.parse(complete_response_warm);

    cactus_reset(model);

    char complete_response_cold[4096] = {0};
    int complete_result_cold = cactus_complete(model, complete_messages.c_str(), complete_response_cold, sizeof(complete_response_cold),
                                               options, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    Metrics cold_metrics;
    cold_metrics.parse(complete_response_cold);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill success: " << ((prefill_result > 0 && prefill_metrics.success) ? "YES" : "NO") << "\n"
              << "├─ Prefill metrics: ";
    prefill_metrics.print_line();
    std::cout << "\n";
    std::cout << "├─ Complete warm metrics:\n";
    warm_metrics.print_json();
    std::cout << "├─ Complete cold metrics:\n";
    cold_metrics.print_json();

    bool all_success = prefill_result > 0 && prefill_metrics.success
        && complete_result_warm > 0 && warm_metrics.success
        && complete_result_cold > 0 && cold_metrics.success;
    bool warm_prefilled_less = warm_metrics.prefill_tokens < cold_metrics.prefill_tokens;

    std::cout << "├─ Calls successful: " << (all_success ? "YES" : "NO") << "\n"
              << "└─ Warm prefilled less than cold: " << (warm_prefilled_less ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return all_success && warm_prefilled_less;
}

bool test_prefill_prefix_extension_reuse_vlm() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "  PREFILL PREFIX EXTENSION (VLM)" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    std::string model_path_str(g_model_path ? g_model_path : "");
    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "⊘ SKIP │ Vision weights not found\n";
        return true;
    }
    vf.close();

    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    std::string base_img_path = std::string(g_assets_path) + "/test_monkey.png";
    std::ifstream imgf(base_img_path);
    if (!imgf.good()) {
        std::cout << "⊘ SKIP │ test_monkey.png not found\n";
        return true;
    }
    imgf.close();

    std::string extension_img_path = std::string(g_assets_path) + "/test_thing.png";
    std::ifstream imgf2(extension_img_path);
    if (!imgf2.good()) {
        std::cout << "⊘ SKIP │ test_thing.png not found\n";
        return true;
    }
    imgf2.close();

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string base_messages = "["
        "{\"role\": \"system\", \"content\": \"You are a helpful assistant. Be concise.\"},"
        "{\"role\": \"user\", \"content\": \"Describe this image in one short sentence.\", \"images\": [\""
        + base_img_path + "\"]},"
        "{\"role\": \"assistant\", \"content\": \"This image shows a close-up of a monkey face.\"}"
        "]";

    std::string extended_messages = "["
        "{\"role\": \"system\", \"content\": \"You are a helpful assistant. Be concise.\"},"
        "{\"role\": \"user\", \"content\": \"Describe this image in one short sentence.\", \"images\": [\""
        + base_img_path + "\"]},"
        "{\"role\": \"assistant\", \"content\": \"This image shows a close-up of a monkey face.\"},"
        "{\"role\": \"user\", \"content\": \"Now describe this second image in one short sentence.\", \"images\": [\""
        + extension_img_path + "\"]}"
        "]";

    char prefill_response1[2048] = {0};
    int prefill_result1 = cactus_prefill(model, base_messages.c_str(), prefill_response1, sizeof(prefill_response1), nullptr, nullptr, nullptr, 0);
    PrefillMetrics prefill_metrics1;
    prefill_metrics1.parse(prefill_response1);

    char prefill_response2[2048] = {0};
    int prefill_result2 = cactus_prefill(model, extended_messages.c_str(), prefill_response2, sizeof(prefill_response2), nullptr, nullptr, nullptr, 0);
    PrefillMetrics prefill_metrics2;
    prefill_metrics2.parse(prefill_response2);

    cactus_reset(model);

    char prefill_response3[2048] = {0};
    int prefill_result3 = cactus_prefill(model, extended_messages.c_str(), prefill_response3, sizeof(prefill_response3), nullptr, nullptr, nullptr, 0);
    PrefillMetrics prefill_metrics3;
    prefill_metrics3.parse(prefill_response3);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill#1 (base): ";
    prefill_metrics1.print_line();
    std::cout << "\n"
              << "├─ Prefill#2 (extended, warm): ";
    prefill_metrics2.print_line();
    std::cout << "\n"
              << "├─ Prefill#3 (extended, cold): ";
    prefill_metrics3.print_line();
    std::cout << "\n";

    bool prefill_success = prefill_result1 > 0 && prefill_result2 > 0 && prefill_result3 > 0
        && prefill_metrics1.success && prefill_metrics2.success && prefill_metrics3.success;
    bool second_call_prefilled = prefill_metrics2.prefill_tokens > 0;
    bool warm_reused_prefix = prefill_metrics2.prefill_tokens < prefill_metrics3.prefill_tokens;

    std::cout << "├─ Prefill calls success: " << (prefill_success ? "YES" : "NO") << "\n"
              << "├─ Warm extension prefilled tokens: " << (second_call_prefilled ? "YES" : "NO") << "\n"
              << "└─ Warm extension < cold extension: " << (warm_reused_prefix ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return prefill_success && second_call_prefilled && warm_reused_prefix;
}

bool test_json_schema_grammar_with_image() {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << " JSON SCHEMA IMAGE TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    std::string model_path_str(g_model_path ? g_model_path : "");
    std::string vision_file = model_path_str + "/vision_patch_embedding.weights";
    std::ifstream vf(vision_file);
    if (!vf.good()) {
        std::cout << "⊘ SKIP │ Vision weights not found\n";
        return true;
    }
    vf.close();

    if (!g_assets_path) {
        std::cout << "⊘ SKIP │ CACTUS_TEST_ASSETS not set\n";
        return true;
    }

    std::string img_path = std::string(g_assets_path) + "/test_thing.png";
    std::ifstream imgf(img_path);
    if (!imgf.good()) {
        std::cout << "⊘ SKIP │ test_thing.png not found\n";
        return true;
    }
    imgf.close();

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* image_schema = R"({
        "type": "object",
        "properties": {
            "shape": {"type": "string"},
            "pose": {"type": "string"}
        },
        "required": ["shape", "pose", "dominant_color"],
        "additionalProperties": false
    })";

    cactus_grammar_t grammar = cactus_grammar_init_json_schema(image_schema, CACTUS_GRAMMAR_JSON_SCHEMA_DEFAULT_OPTIONS);
    if (!grammar) {
        std::cerr << "[✗] Failed to initialize JSON Schema grammar\n";
        cactus_destroy(model);
        return false;
    }

    std::string messages = "[{\"role\": \"system\", \"content\": \"You are a helpful assistant that answers with concise factual image descriptions.\"}, "
        "{\"role\": \"user\", \"content\": \"Look at this image and describe its shape and pose.\", \"images\": [\""
        + img_path + "\"]}]";

    StreamingData data;
    data.model = model;
    char response[4096];

    std::cout << "\n[Prompt]\n";
    std::cout << "User: Look at this image and describe its shape and pose.\n";
    std::cout << "Assistant: ";

    int result = cactus_complete(model, messages.c_str(), response, sizeof(response),
                                 g_options, nullptr, stream_callback, &data, nullptr, 0, grammar);

    Metrics metrics;
    metrics.parse(response);
    const std::string& output = metrics.response;

    std::string json_error;
    bool valid_json = is_valid_json_document(output, json_error);
    bool has_shape = false;
    bool has_pose = false;

    if (valid_json) {
        picojson::value value;
        auto begin = output.begin();
        picojson::parse(value, begin, output.end(), &json_error);
        if (value.is<picojson::object>()) {
            has_shape = value.contains("shape") && value.get("shape").is<std::string>()
                && !value.get("shape").get<std::string>().empty();
            has_pose = value.contains("pose") && value.get("pose").is<std::string>()
                && !value.get("pose").get<std::string>().empty();
        }
    }

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Valid JSON: " << (valid_json ? "YES" : "NO") << "\n"
              << "├─ Has shape: " << (has_shape ? "YES" : "NO") << "\n"
              << "├─ Has pose: " << (has_pose ? "YES" : "NO") << "\n";
    if (!valid_json) {
        std::cout << "├─ JSON error: " << json_error << "\n";
    }
    if (!valid_json || !has_shape || !has_pose) {
        std::cout << "├─ Raw output: " << output << "\n";
    }
    metrics.print_json();

    cactus_grammar_destroy(grammar);
    cactus_destroy(model);
    return result > 0 && data.token_count > 0 && valid_json && has_shape && has_pose;
}

int main() {
    TestUtils::TestRunner runner("VLM Tests");
    runner.run_test("prefill_with_images", test_prefill_with_images());
    runner.run_test("prefill_prefix_extension_reuse", test_prefill_prefix_extension_reuse_vlm());
    runner.run_test("prefill_invalidated_on_message_change", test_prefill_invalidated_on_message_change_vlm());
    runner.run_test("vlm_multiturn", test_vlm_multiturn());
    runner.run_test("json_schema_grammar_with_image", test_json_schema_grammar_with_image());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
