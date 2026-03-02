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

    std::string img_path = std::string(g_assets_path) + "/test_monkey.png";
    std::ifstream imgf(img_path);
    if (!imgf.good()) {
        std::cout << "⊘ SKIP │ test_monkey.png not found\n";
        return true;
    }
    imgf.close();

    cactus_model_t model = cactus_init(g_model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    std::string messages = "[{\"role\": \"user\", \"content\": \"Describe this image in one sentence.\", \"images\": [\""
        + img_path + "\"]}]";

    const char* complete_options = R"({
        "max_tokens": 128,
        "stop_sequences": ["<|im_end|>", "<end_of_turn>"],
        "confidence_threshold": 0.0,
        "telemetry_enabled": false
    })";

    char prefill_response[2048] = {0};
    Timer prefill_timer;
    int prefill_result = cactus_prefill(model, messages.c_str(), prefill_response, sizeof(prefill_response), nullptr, nullptr);
    double prefill_elapsed_ms = prefill_timer.elapsed_ms();
    Metrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response1[4096] = {0};
    Timer complete_timer1;
    int complete_result1 = cactus_complete(model, messages.c_str(), complete_response1, sizeof(complete_response1),
                                           complete_options, nullptr, nullptr, nullptr);
    double complete_elapsed_ms1 = complete_timer1.elapsed_ms();
    Metrics complete_metrics1;
    complete_metrics1.parse(complete_response1);

    cactus_reset(model);

    char complete_response2[4096] = {0};
    Timer complete_timer2;
    int complete_result2 = cactus_complete(model, messages.c_str(), complete_response2, sizeof(complete_response2),
                                           complete_options, nullptr, nullptr, nullptr);
    double complete_elapsed_ms2 = complete_timer2.elapsed_ms();
    Metrics complete_metrics2;
    complete_metrics2.parse(complete_response2);

    std::cout << "\n\n[Results]\n";
    std::cout << "├─ Prefill status: " << (prefill_result > 0 ? "OK" : "FAILED") << "\n"
              << "├─ Prefill benchmark: elapsed_ms=" << std::fixed << std::setprecision(2) << prefill_elapsed_ms
              << ", prefill_tokens=" << std::setprecision(0) << prefill_metrics.prefill_tokens
              << ", prefill_tps=" << std::setprecision(2) << prefill_metrics.prefill_tps
              << ", total_time_ms=" << std::setprecision(2) << prefill_metrics.total_ms << "\n";
    std::cout << "├─ Complete#1 metrics:\n";
    complete_metrics1.print_json();
    std::cout << "├─ Complete#2 metrics:\n";
    complete_metrics2.print_json();
    std::cout << "├─ Complete elapsed ms: #1=" << complete_elapsed_ms1
              << ", #2=" << complete_elapsed_ms2 << "\n";

    bool prefill_success = prefill_result > 0 && prefill_metrics.success;
    bool complete_success = complete_result1 > 0 && complete_result2 > 0
        && complete_metrics1.success && complete_metrics2.success;
    bool prefill_reduced = complete_metrics1.prefill_tokens < complete_metrics2.prefill_tokens;

    std::cout << "├─ Prefill call success: " << (prefill_success ? "YES" : "NO") << "\n"
              << "├─ Complete calls success: " << (complete_success ? "YES" : "NO") << "\n"
              << "└─ Prefill tokens #1 <= #2: " << (prefill_reduced ? "YES" : "NO") << std::endl;

    cactus_destroy(model);
    return prefill_success && complete_success && prefill_reduced;
}

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
                                  g_options, nullptr, stream_callback, &stream_data1);

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
                                  g_options, nullptr, stream_callback, &stream_data2);

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
    int prefill_result = cactus_prefill(model, prefill_messages.c_str(), prefill_response, sizeof(prefill_response), nullptr, nullptr);
    Metrics prefill_metrics;
    prefill_metrics.parse(prefill_response);

    char complete_response_warm[4096] = {0};
    int complete_result_warm = cactus_complete(model, complete_messages.c_str(), complete_response_warm, sizeof(complete_response_warm),
                                               options, nullptr, nullptr, nullptr);
    Metrics warm_metrics;
    warm_metrics.parse(complete_response_warm);

    cactus_reset(model);

    char complete_response_cold[4096] = {0};
    int complete_result_cold = cactus_complete(model, complete_messages.c_str(), complete_response_cold, sizeof(complete_response_cold),
                                               options, nullptr, nullptr, nullptr);
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

int main() {
    TestUtils::TestRunner runner("VLM Tests");
    runner.run_test("prefill_with_images", test_prefill_with_images());
    runner.run_test("prefill_invalidated_on_message_change", test_prefill_invalidated_on_message_change_vlm());
    runner.run_test("vlm_multiturn", test_vlm_multiturn());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
