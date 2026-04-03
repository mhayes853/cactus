#include "test_utils.h"
#include "../cactus/models/gemma4/model_gemma4.h"
#include "../cactus/ffi/cactus_utils.h"
#include "../libs/audio/wav.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <sys/stat.h>

using namespace cactus::engine;
using namespace cactus::audio;
using namespace EngineTestUtils;

static const char* get_model_path() {
    const char* path = std::getenv("CACTUS_TEST_GEMMA4_MODEL");
    if (path) return path;
    return std::getenv("CACTUS_TEST_MODEL");
}

static const char* get_image_path() {
    return std::getenv("CACTUS_TEST_IMAGE");
}

static std::string get_assets_dir() {
    const char* dir = std::getenv("CACTUS_TEST_ASSETS");
    if (dir) return dir;
    return "../assets";
}

static bool has_npu_package(const char* model_path, const std::string& name) {
    struct stat st;
    return stat((std::string(model_path) + "/" + name).c_str(), &st) == 0;
}

static const char* g_options = R"({
    "max_tokens": 256,
    "temperature": 0,
    "top_k": 1,
    "stop_sequences": ["<turn|>", "<eos>", "<end_of_turn>", "<|im_end|>"],
    "enable_thinking_if_supported": false,
    "auto_handoff": false,
    "telemetry_enabled": false
})";

struct LocalPerfMetrics {
    double time_to_first_token_ms = 0.0;
    double total_time_ms = 0.0;
    double prefill_tps = 0.0;
    double decode_tps = 0.0;
    size_t prompt_tokens = 0;
    size_t completion_tokens = 0;
};

struct EncoderOnlyMetrics {
    bool collected = false;
    bool npu_enabled = false;
    std::string input_label = "in_units";
    std::string output_label = "out_units";
    size_t input_units = 0;
    size_t output_units = 0;
    double encoder_ms = 0.0;
    double output_units_per_s = 0.0;
};

static LocalPerfMetrics compute_local_perf_metrics(size_t prompt_tokens,
                                                   size_t completion_tokens,
                                                   double ttft_ms,
                                                   double total_time_ms) {
    LocalPerfMetrics m;
    m.time_to_first_token_ms = ttft_ms;
    m.total_time_ms = total_time_ms;
    m.prompt_tokens = prompt_tokens;
    m.completion_tokens = completion_tokens;
    m.prefill_tps = ttft_ms > 0.0 ? (static_cast<double>(prompt_tokens) * 1000.0) / ttft_ms : 0.0;
    const double decode_time_ms = std::max(0.0, total_time_ms - ttft_ms);
    m.decode_tps = (completion_tokens > 1 && decode_time_ms > 0.0)
                   ? ((static_cast<double>(completion_tokens) - 1.0) * 1000.0) / decode_time_ms
                   : 0.0;
    return m;
}

static void print_local_perf_metrics(const LocalPerfMetrics& m,
                                     const EncoderOnlyMetrics* encoder_metrics = nullptr) {
    std::streamsize old_precision = std::cout.precision();
    std::ios::fmtflags old_flags = std::cout.flags();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Performance:\n";
    if (encoder_metrics && encoder_metrics->collected) {
        std::cout << "    encoder_only : " << encoder_metrics->encoder_ms << " ms"
                  << " | mode=" << (encoder_metrics->npu_enabled ? "npu" : "cpu")
                  << " | " << encoder_metrics->input_label << "=" << encoder_metrics->input_units
                  << " | " << encoder_metrics->output_label << "=" << encoder_metrics->output_units
                  << " | " << encoder_metrics->output_label << "_per_s=" << encoder_metrics->output_units_per_s
                  << "\n";
    }
    std::cout << "    end_to_end   : "
              << "ttft_ms=" << m.time_to_first_token_ms
              << " | total_time_ms=" << m.total_time_ms
              << "\n";
    std::cout << "    throughput   : "
              << "prefill_tps=" << m.prefill_tps
              << " | decode_tps=" << m.decode_tps
              << "\n";
    std::cout << "    token_counts : "
              << "prompt=" << m.prompt_tokens
              << " | completion=" << m.completion_tokens
              << "\n";
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

static std::string preview_text(const std::string& text, size_t max_chars = 240) {
    if (text.size() <= max_chars) return text;
    return text.substr(0, max_chars) + "...";
}

static void print_modality_box(const std::string& title, const std::string& prompt) {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << std::string("          ") + title << "║\n"
              << "╚══════════════════════════════════════════╝\n";
    if (!prompt.empty()) {
        std::cout << "├─ User prompt: " << prompt << "\n";
    }
}


bool test_text_generation() {
    const char* model_path = get_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    const char* messages = R"([
        {"role": "system", "content": "/no_think You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France?"}
    ])";

    return EngineTestUtils::run_test("TEXT GENERATION", model_path, messages, g_options,
        [](int result, const StreamingData& data, const std::string& /*response*/, const Metrics& m) {
            std::string text;
            for (const auto& t : data.tokens) text += t;
            std::string lower_text;
            for (char c : text) lower_text += std::tolower(c);

            bool has_paris = lower_text.find("paris") != std::string::npos;
            std::cout << "├─ Output: " << text.substr(0, 200) << "\n"
                      << "├─ Contains 'paris': " << (has_paris ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && data.token_count > 0 && has_paris;
        }, nullptr, -1, "What is the capital of France?");
}


bool test_tool_call() {
    const char* model_path = get_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    const char* messages = R"([
        {"role": "system", "content": "/no_think You are a helpful assistant that can use tools."},
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ])";

    const char* tools = R"([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City, Country"}
                },
                "required": ["location"]
            }
        }
    }])";

    const char* options = R"({
        "max_tokens": 256,
        "temperature": 0,
        "top_k": 1,
        "stop_sequences": ["<turn|>", "<eos>", "<end_of_turn>", "<|im_end|>"],
        "enable_thinking_if_supported": false,
        "auto_handoff": false,
        "force_tools": true,
        "telemetry_enabled": false
    })";

    return EngineTestUtils::run_test("TOOL CALL", model_path, messages, options,
        [](int result, const StreamingData&, const std::string& response, const Metrics& m) {
            bool has_function = response.find("\"function_calls\":[") != std::string::npos;
            bool has_tool = has_function && response.find("get_weather") != std::string::npos;
            std::cout << "├─ Function call: " << (has_function ? "YES" : "NO") << "\n"
                      << "├─ Correct tool: " << (has_tool ? "YES" : "NO") << "\n";
            m.print_json();
            return result > 0 && has_function && has_tool;
        }, tools, -1, "What's the weather in Tokyo?");
}


bool test_1k_context() {
    const char* model_path = get_model_path();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    std::string msg = "[{\"role\": \"system\", \"content\": \"/no_think You are helpful. ";
    for (int i = 0; i < 50; i++)
        msg += "Context " + std::to_string(i) + ": The quick brown fox jumps over the lazy dog. ";
    msg += "\"}, {\"role\": \"user\", \"content\": \"";
    for (int i = 0; i < 50; i++)
        msg += "Data point " + std::to_string(i) + " = " + std::to_string(i * 2.71828) + ". ";
    msg += "Summarize the data briefly.\"}]";

    return EngineTestUtils::run_test("1K CONTEXT", model_path, msg.c_str(), g_options,
        [](int result, const StreamingData& data, const std::string&, const Metrics& m) {
            std::cout << "├─ Tokens generated: " << data.token_count << "\n";
            m.print_json();
            return result > 0 && data.token_count > 0;
        }, nullptr, 100, "Summarize the data briefly.");
}


bool test_gemma4_vision(bool expect_npu) {
    const char* model_path = get_model_path();
    const char* image_path = get_image_path();
    if (!model_path || !image_path) {
        std::cerr << "  SKIP: CACTUS_TEST_GEMMA4_MODEL or CACTUS_TEST_IMAGE not set\n";
        return true;
    }

    if (expect_npu && !has_npu_package(model_path, "vision_encoder.mlpackage")) {
        std::cerr << "  SKIP: vision_encoder.mlpackage not found in model folder\n";
        return true;
    }

    print_modality_box(expect_npu ? "VISION NPU" : "VISION", "Describe this image briefly.");

    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "  FAIL: create_model returned null\n";
        return false;
    }

    auto* mm = dynamic_cast<Gemma4MmModel*>(model.get());
    if (mm && !expect_npu) {
        mm->vision_encoder().disable_npu_ = true;
    }

    if (!model->init(model_path, 2048, "", false)) {
        std::cerr << "  FAIL: model init\n";
        return false;
    }

    auto* tokenizer = model->get_tokenizer();
    if (!tokenizer) {
        std::cerr << "  FAIL: no tokenizer\n";
        return false;
    }

    std::vector<ChatMessage> messages;
    messages.push_back({"user", "Describe this image briefly.", "", {image_path}, {}});
    std::string prompt = tokenizer->format_chat_prompt(messages, true, "", false);
    auto tokens = tokenizer->encode(prompt);

    uint32_t image_token_id = model->get_config().image_token_id;
    if (image_token_id == 0)
        image_token_id = 258880;

    size_t vision_count = 0;
    for (auto t : tokens) {
        if (t == image_token_id)
            vision_count++;
    }
    std::cout << "  tokens: " << tokens.size()
              << ", image_token_id: " << image_token_id
              << ", vision soft tokens: " << vision_count << "\n";
    const size_t prompt_token_count = tokens.size();

    EncoderOnlyMetrics encoder_only_metrics;
    encoder_only_metrics.input_label = "in_patches";
    encoder_only_metrics.output_label = "out_steps";
    if (mm) {
        auto preprocessed = mm->vision_encoder().preprocess_image(image_path);
        encoder_only_metrics.input_units = preprocessed.num_patches;
        encoder_only_metrics.npu_enabled = mm->vision_encoder().use_npu_encoder_;
        if (model->graph_handle_) {
            auto* gb = static_cast<CactusGraph*>(model->graph_handle_);
            auto backend = model->get_config().default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
            gb->soft_reset();
            auto encoder_start = std::chrono::high_resolution_clock::now();
            size_t vision_output = mm->vision_encoder().forward_vision(gb, preprocessed, backend);
            gb->execute();
            auto encoder_end = std::chrono::high_resolution_clock::now();

            encoder_only_metrics.encoder_ms =
                std::chrono::duration_cast<std::chrono::microseconds>(encoder_end - encoder_start).count() / 1000.0;
            const auto& out_buf = gb->get_output_buffer(vision_output);
            encoder_only_metrics.output_units = out_buf.shape.empty() ? 0 : out_buf.shape[0];
            encoder_only_metrics.output_units_per_s = encoder_only_metrics.encoder_ms > 0.0
                ? (static_cast<double>(encoder_only_metrics.output_units) * 1000.0) / encoder_only_metrics.encoder_ms
                : 0.0;
            encoder_only_metrics.collected = true;
        }
    }

    std::vector<std::string> images = {image_path};
    std::string output;
    size_t completion_tokens = 0;
    double ttft_ms = 0.0;
    bool saw_first_token = false;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 150; i++) {
        uint32_t token = model->decode_with_images(tokens, images, 0.0f, 1.0f, 1, "");
        if (!saw_first_token) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
            saw_first_token = true;
        }
        std::string piece = tokenizer->decode({token});
        output += piece;
        tokens.push_back(token);
        completion_tokens++;
        if (piece.find("<turn|>") != std::string::npos || piece.find("<eos>") != std::string::npos)
            break;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    auto metrics = compute_local_perf_metrics(prompt_token_count, completion_tokens, ttft_ms, total_time_ms);

    print_local_perf_metrics(metrics, &encoder_only_metrics);
    std::cout << "  Output: " << output.substr(0, 300) << "\n";
    std::cout << "  NPU" << (expect_npu ? " (expected)" : " (not required)") << ": "
              << (has_npu_package(model_path, "vision_encoder.mlpackage") ? "available" : "not available") << "\n";

    if (output.empty()) {
        std::cerr << "  FAIL: empty output\n";
        return false;
    }

    bool has_content = false;
    for (char c : output)
        if (std::isalpha(c)) { has_content = true; break; }

    if (!has_content) {
        std::cerr << "  FAIL: output has no alphabetic content\n";
        return false;
    }

    return true;
}


bool test_gemma4_vision_json_schema() {
    const char* model_path = get_model_path();
    const char* image_path = get_image_path();
    if (!model_path || !image_path) {
        std::cerr << "  SKIP: CACTUS_TEST_GEMMA4_MODEL or CACTUS_TEST_IMAGE not set\n";
        return true;
    }

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << "   VISION JSON SCHEMA TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";
    std::cout << "├─ User prompt: Describe the image.\n";

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    const char* schema = R"({
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 1},
            "dominant_colors": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 3
            }
        },
        "required": ["summary", "dominant_colors"],
        "additionalProperties": false
    })";

    cactus_grammar_t grammar = cactus_grammar_init_json_schema(
        schema,
        CACTUS_GRAMMAR_JSON_SCHEMA_DEFAULT_OPTIONS
    );
    if (!grammar) {
        std::cerr << "[✗] Failed to initialize JSON Schema grammar\n";
        cactus_destroy(model);
        return false;
    }

    const char* prompt = "Describe this image.";
    std::string messages = R"([
        {"role": "system", "content": "/no_think You are a precise vision assistant."},
        {"role": "user", "content": ")" + escape_json(prompt) + R"(", "images": [")"
        + escape_json(image_path) + R"("]}
    ])";

    StreamingData data;
    data.model = model;
    char response[4096];

    std::cout << "Response: ";
    int result = cactus_complete(model, messages.c_str(), response, sizeof(response),
                                 g_options, nullptr, stream_callback, &data, nullptr, 0, grammar);

    std::cout << "\n\n[Results]\n";

    Metrics metrics;
    metrics.parse(response);
    const std::string& output = metrics.response;

    std::string json_error;
    bool valid_json = is_valid_json_document(output, json_error);
    bool has_summary = false;
    bool has_colors = false;

    if (valid_json) {
        picojson::value value;
        auto begin = output.begin();
        picojson::parse(value, begin, output.end(), &json_error);
        if (value.is<picojson::object>()) {
            has_summary = value.contains("summary")
                && value.get("summary").is<std::string>()
                && !value.get("summary").get<std::string>().empty();

            const picojson::value& colors_value = value.get("dominant_colors");
            if (value.contains("dominant_colors") && colors_value.is<picojson::array>()) {
                const picojson::array& colors = colors_value.get<picojson::array>();
                has_colors = !colors.empty();
                for (const auto& color : colors) {
                    if (!color.is<std::string>() || color.get<std::string>().empty()) {
                        has_colors = false;
                        break;
                    }
                }
            }
        }
    }

    std::cout << "├─ Valid JSON: " << (valid_json ? "YES" : "NO") << "\n"
              << "├─ Has summary: " << (has_summary ? "YES" : "NO") << "\n"
              << "├─ Has dominant_colors: " << (has_colors ? "YES" : "NO") << "\n";
    if (!valid_json) {
        std::cout << "├─ JSON error: " << json_error << "\n";
    }
    if (!valid_json || !has_summary || !has_colors) {
        std::cout << "├─ Raw output: " << output << "\n";
    }
    metrics.print_json();

    cactus_grammar_destroy(grammar);
    cactus_destroy(model);
    return result > 0 && data.token_count > 0 && valid_json && has_summary && has_colors;
}


bool test_gemma4_audio(bool expect_npu) {
    const char* model_path = get_model_path();
    std::string assets = get_assets_dir();
    if (!model_path) {
        std::cerr << "  SKIP: model path not set\n";
        return true;
    }

    if (expect_npu && !has_npu_package(model_path, "audio_encoder.mlpackage")) {
        std::cerr << "  SKIP: audio_encoder.mlpackage not found in model folder\n";
        return true;
    }

    print_modality_box(expect_npu ? "AUDIO NPU" : "AUDIO", "Transcribe the audio.");

    std::string audio_path = assets + "/test.wav";
    struct stat st;
    if (stat(audio_path.c_str(), &st) != 0) {
        std::cerr << "  SKIP: test.wav not found in assets\n";
        return true;
    }

    auto model = create_model(model_path);
    if (!model) {
        std::cerr << "  FAIL: create_model returned null\n";
        return false;
    }

    auto* mm = dynamic_cast<Gemma4MmModel*>(model.get());
    if (mm) {
        if (!expect_npu)
            mm->audio_encoder().disable_npu_ = true;
    }

    if (!model->init(model_path, 2048, "", true)) {
        std::cerr << "  FAIL: model init\n";
        return false;
    }

    AudioFP32 wav = load_wav(audio_path.c_str());
    auto audio_samples = resample_to_16k_fp32(wav.samples, wav.sample_rate);

    size_t pad_amt = 320 - (audio_samples.size() % 320);
    if (pad_amt < 320)
        audio_samples.resize(audio_samples.size() + pad_amt, 0.0f);

    const auto& cfg = model->get_config();
    size_t mel_bins = cfg.audio_input_feat_size;
    uint32_t audio_token_id = cfg.audio_token_id;
    if (audio_token_id == 0) audio_token_id = 258881;

    auto spec_cfg = get_htk_spectrogram_config();
    AudioProcessor ap;
    size_t fft_for_mel = spec_cfg.fft_override > 0 ? spec_cfg.fft_override : spec_cfg.n_fft;
    ap.init_mel_filters(fft_for_mel / 2 + 1, mel_bins, 0.0f, 8000.0f, 16000, nullptr, "htk");
    auto mel = ap.compute_spectrogram(audio_samples, spec_cfg);

    size_t num_frames = mel.size() / mel_bins;

#ifdef __APPLE__
    {
        static constexpr float LN2 = 0.693147180559945f;
        for (auto& v : mel) v -= LN2;
    }
#endif

    auto audio_features = transpose_mel_to_frame_major(mel, mel_bins, num_frames);

    EncoderOnlyMetrics encoder_only_metrics;
    encoder_only_metrics.input_label = "in_frames";
    encoder_only_metrics.output_label = "out_steps";
    encoder_only_metrics.input_units = num_frames;
    if (mm) {
        encoder_only_metrics.npu_enabled = mm->audio_encoder().use_npu_encoder_;
        if (model->graph_handle_) {
            auto* gb = static_cast<CactusGraph*>(model->graph_handle_);
            auto backend = cfg.default_backend == Config::Backend::CPU ? ComputeBackend::CPU : ComputeBackend::NPU;
            gb->soft_reset();

            auto encoder_start = std::chrono::high_resolution_clock::now();
            size_t encoded_audio = mm->audio_encoder().forward_audio(gb, audio_features, num_frames, backend);
            gb->execute();
            auto encoder_end = std::chrono::high_resolution_clock::now();

            encoder_only_metrics.encoder_ms = std::chrono::duration_cast<std::chrono::microseconds>(encoder_end - encoder_start).count() / 1000.0;
            const auto& encoded_buf = gb->get_output_buffer(encoded_audio);
            encoder_only_metrics.output_units = encoded_buf.shape.empty() ? 0 : encoded_buf.shape[0];
            encoder_only_metrics.output_units_per_s = encoder_only_metrics.encoder_ms > 0.0
                ? (static_cast<double>(encoder_only_metrics.output_units) * 1000.0) / encoder_only_metrics.encoder_ms
                : 0.0;
            encoder_only_metrics.collected = true;
        }
    }

    size_t after_stage1 = (num_frames + 1) / 2;
    size_t num_soft_tokens = (after_stage1 + 1) / 2;

    auto* tokenizer = model->get_tokenizer();
    auto prefix = tokenizer->encode("<bos><|turn>user\nTranscribe the audio.<|audio>");
    auto suffix = tokenizer->encode("<audio|><turn|>\n<|turn>model\n");

    std::vector<uint32_t> tokens;
    tokens.insert(tokens.end(), prefix.begin(), prefix.end());
    for (size_t i = 0; i < num_soft_tokens; i++)
        tokens.push_back(audio_token_id);
    tokens.insert(tokens.end(), suffix.begin(), suffix.end());
    const size_t prompt_token_count = tokens.size();

    std::string output;
    size_t completion_tokens = 0;
    double ttft_ms = 0.0;
    bool saw_first_token = false;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 200; i++) {
        uint32_t token = model->decode_with_audio(tokens, audio_features, 0.0f, 1.0f, 1, "");
        if (!saw_first_token) {
            auto t_first = std::chrono::high_resolution_clock::now();
            ttft_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_first - start_time).count() / 1000.0;
            saw_first_token = true;
        }
        std::string piece = tokenizer->decode({token});
        output += piece;
        tokens.push_back(token);
        completion_tokens++;
        if (output.find("<turn|>") != std::string::npos || output.find("<eos>") != std::string::npos)
            break;
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    auto metrics = compute_local_perf_metrics(prompt_token_count, completion_tokens, ttft_ms, total_time_ms);

    print_local_perf_metrics(metrics, &encoder_only_metrics);
    std::cout << "  Transcript: " << preview_text(output) << "\n";

    if (output.empty()) {
        std::cerr << "  FAIL: empty output\n";
        return false;
    }

    bool has_content = false;
    for (char c : output)
        if (std::isalpha(c)) { has_content = true; break; }

    if (!has_content) {
        std::cerr << "  FAIL: output has no alphabetic content\n";
        return false;
    }

    return true;
}


bool test_gemma4_audio_json_schema_pcm() {
    const char* model_path = get_model_path();
    std::string assets = get_assets_dir();
    if (!model_path) {
        std::cerr << "  SKIP: CACTUS_TEST_GEMMA4_MODEL or CACTUS_TEST_MODEL not set\n";
        return true;
    }

    std::string audio_path = assets + "/test.wav";
    struct stat st;
    if (stat(audio_path.c_str(), &st) != 0) {
        std::cerr << "  SKIP: test.wav not found in assets\n";
        return true;
    }

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║" << std::setw(42) << std::left << " AUDIO JSON SCHEMA PCM TEST" << "║\n"
              << "╚══════════════════════════════════════════╝\n";
    std::cout << "├─ User prompt: Describe the audio.\n";

    cactus_model_t model = cactus_init(model_path, nullptr, false);
    if (!model) {
        std::cerr << "[✗] Failed to initialize model\n";
        return false;
    }

    AudioFP32 wav = load_wav(audio_path.c_str());
    if (wav.samples.empty()) {
        std::cerr << "[✗] Failed to load audio\n";
        cactus_destroy(model);
        return false;
    }

    std::vector<int16_t> pcm16(wav.samples.size());
    for (size_t i = 0; i < wav.samples.size(); i++) {
        float sample = wav.samples[i];
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        pcm16[i] = static_cast<int16_t>(sample * 32767.0f);
    }

    const char* schema = R"({
        "type": "object",
        "properties": {
            "summary": {"type": "string", "minLength": 1},
            "keywords": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "maxItems": 4
            }
        },
        "required": ["summary", "keywords"],
        "additionalProperties": false
    })";

    cactus_grammar_t grammar = cactus_grammar_init_json_schema(
        schema,
        CACTUS_GRAMMAR_JSON_SCHEMA_DEFAULT_OPTIONS
    );
    if (!grammar) {
        std::cerr << "[✗] Failed to initialize JSON Schema grammar\n";
        cactus_destroy(model);
        return false;
    }

    const char* prompt = "Describe the audio.";
    std::string messages = R"([
        {"role": "system", "content": "/no_think You are a precise audio assistant."},
        {"role": "user", "content": ")" + escape_json(prompt) + R"("}
    ])";

    StreamingData data;
    data.model = model;
    char response[4096];

    std::cout << "Response: ";
    int result = cactus_complete(
        model,
        messages.c_str(),
        response,
        sizeof(response),
        g_options,
        nullptr,
        stream_callback,
        &data,
        reinterpret_cast<const uint8_t*>(pcm16.data()),
        pcm16.size() * sizeof(int16_t),
        grammar
    );

    std::cout << "\n\n[Results]\n";

    Metrics metrics;
    metrics.parse(response);
    const std::string& output = metrics.response;

    std::string json_error;
    bool valid_json = is_valid_json_document(output, json_error);
    bool has_summary = false;
    bool has_keywords = false;

    if (valid_json) {
        picojson::value value;
        auto begin = output.begin();
        picojson::parse(value, begin, output.end(), &json_error);
        if (value.is<picojson::object>()) {
            has_summary = value.contains("summary")
                && value.get("summary").is<std::string>()
                && !value.get("summary").get<std::string>().empty();

            const picojson::value& keywords_value = value.get("keywords");
            if (value.contains("keywords") && keywords_value.is<picojson::array>()) {
                const picojson::array& keywords = keywords_value.get<picojson::array>();
                has_keywords = !keywords.empty();
                for (const auto& keyword : keywords) {
                    if (!keyword.is<std::string>() || keyword.get<std::string>().empty()) {
                        has_keywords = false;
                        break;
                    }
                }
            }
        }
    }

    std::cout << "├─ Valid JSON: " << (valid_json ? "YES" : "NO") << "\n"
              << "├─ Has summary: " << (has_summary ? "YES" : "NO") << "\n"
              << "├─ Has keywords: " << (has_keywords ? "YES" : "NO") << "\n";
    if (!valid_json) {
        std::cout << "├─ JSON error: " << json_error << "\n";
    }
    if (!valid_json || !has_summary || !has_keywords) {
        std::cout << "├─ Raw output: " << output << "\n";
    }
    metrics.print_json();

    cactus_grammar_destroy(grammar);
    cactus_destroy(model);
    return result > 0 && data.token_count > 0 && valid_json && has_summary && has_keywords;
}


int main() {
    TestUtils::TestRunner runner("Gemma4 Suite");

    runner.run_test("text_generation", test_text_generation());
    runner.run_test("tool_call", test_tool_call());
    runner.run_test("1k_context", test_1k_context());
    runner.run_test("vision", test_gemma4_vision(false));
    runner.run_test("vision_npu", test_gemma4_vision(true));
    runner.run_test("vision_json_schema", test_gemma4_vision_json_schema());
    runner.run_test("audio_json_schema_pcm", test_gemma4_audio_json_schema_pcm());
    runner.run_test("audio", test_gemma4_audio(false));
    runner.run_test("audio_npu", test_gemma4_audio(true));

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
