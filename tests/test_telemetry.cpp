#include "test_utils.h"
#include "../cactus/telemetry/telemetry.h"

#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

std::string make_temp_dir(const char* prefix) {
    char pattern[256] = {0};
    std::snprintf(pattern, sizeof(pattern), "/tmp/%s_XXXXXX", prefix);
    return std::string(mkdtemp(pattern));
}

int count_events(const std::string& file_path) {
    std::ifstream in(file_path);
    if (!in.is_open()) return 0;
    int count = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++count;
    }
    return count;
}

bool test_record_many_then_flush() {
    const std::string cache_dir = make_temp_dir("cactus_record_many_flush");

    cactus::telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    cactus::telemetry::setCloudDisabled(true);
    cactus::telemetry::init("telemetry-test-project", "record-many", nullptr);

    constexpr int expected_event_count = 200;
    for (int i = 0; i < expected_event_count; ++i) {
        cactus::telemetry::recordCompletion("test-model", true, 10.0, 25.0, 30.0, 32, "ok");
    }

    cactus::telemetry::flush();

    const std::string completion_log = cache_dir + "/completion.log";
    const int event_count = count_events(completion_log);

    cactus::telemetry::shutdown();
    rmdir(cache_dir.c_str());
    return event_count == expected_event_count;
}

bool test_shutdown_then_reinit_then_record() {
    const std::string cache_dir = make_temp_dir("cactus_shutdown_reinit");

    cactus::telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    cactus::telemetry::setCloudDisabled(true);
    cactus::telemetry::init("telemetry-test-project", "shutdown-reinit", nullptr);

    cactus::telemetry::recordCompletion("test-model", true, 5.0, 20.0, 18.0, 16, "before-shutdown");
    cactus::telemetry::flush();

    const std::string completion_log = cache_dir + "/completion.log";
    const int lines_before_shutdown = count_events(completion_log);

    cactus::telemetry::shutdown();

    cactus::telemetry::init("telemetry-test-project", "shutdown-reinit", nullptr);
    cactus::telemetry::recordCompletion("test-model", true, 6.0, 21.0, 19.0, 17, "after-reinit");
    cactus::telemetry::flush();

    const int lines_after_reinit = count_events(completion_log);

    cactus::telemetry::shutdown();
    rmdir(cache_dir.c_str());
    return lines_before_shutdown > 0 && lines_after_reinit > lines_before_shutdown;
}

int main() {
    TestUtils::TestRunner runner("Telemetry Tests");
    runner.run_test("Record many then Flush", test_record_many_then_flush());
    runner.run_test("Shutdown then Reinit", test_shutdown_then_reinit_then_record());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
