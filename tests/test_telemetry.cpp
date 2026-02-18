#include "test_utils.h"
#include "../cactus/telemetry/telemetry.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <sstream>
#include <unistd.h>
#include <vector>

std::string make_temp_dir(const char* prefix) {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::ostringstream name;
    name << "/tmp/" << prefix << "_" << now << "_XXXXXX";
    std::string pattern = name.str();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* dir_ptr = mkdtemp(writable.data());
    if (!dir_ptr) return "";
    std::string dir = dir_ptr;
    return dir;
}

int count_lines(const std::string& file_path) {
    std::ifstream in(file_path);
    if (!in.is_open()) return 0;
    int count = 0;
    std::string line;
    while (std::getline(in, line)) {
        ++count;
    }
    return count;
}

void remove_dir_recursive(const std::string& path) {
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        std::remove(path.c_str());
        return;
    }

    struct dirent* entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
        if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        std::string child = path + "/" + entry->d_name;
        struct stat st;
        if (stat(child.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            remove_dir_recursive(child);
        } else {
            std::remove(child.c_str());
        }
    }

    closedir(dir);
    rmdir(path.c_str());
}

bool test_record_many_then_flush() {
    namespace telemetry = cactus::telemetry;
    const std::string cache_dir = make_temp_dir("cactus_record_many_flush");
    if (cache_dir.empty()) return false;

    telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    telemetry::setCloudDisabled(true);
    telemetry::init("telemetry-test-project", "record-many", nullptr);

    constexpr int kEventCount = 200;
    for (int i = 0; i < kEventCount; ++i) {
        telemetry::recordCompletion("test-model", true, 10.0, 25.0, 30.0, 32, "ok");
    }

    telemetry::flush();

    const std::string completion_log = cache_dir + "/completion.log";
    const int lines = count_lines(completion_log);

    telemetry::shutdown();
    remove_dir_recursive(cache_dir);
    return lines >= kEventCount;
}

bool test_shutdown_then_reinit_then_record() {
    namespace telemetry = cactus::telemetry;
    const std::string cache_dir = make_temp_dir("cactus_shutdown_reinit");
    if (cache_dir.empty()) return false;

    telemetry::setTelemetryEnvironment("cpp-test", cache_dir.c_str());
    telemetry::setCloudDisabled(true);
    telemetry::init("telemetry-test-project", "shutdown-reinit", nullptr);

    telemetry::recordCompletion("test-model", true, 5.0, 20.0, 18.0, 16, "before-shutdown");
    telemetry::flush();

    const std::string completion_log = cache_dir + "/completion.log";
    const int lines_before_shutdown = count_lines(completion_log);

    telemetry::shutdown();

    telemetry::init("telemetry-test-project", "shutdown-reinit", nullptr);
    telemetry::recordCompletion("test-model", true, 6.0, 21.0, 19.0, 17, "after-reinit");
    telemetry::flush();

    const int lines_after_reinit = count_lines(completion_log);

    telemetry::shutdown();
    remove_dir_recursive(cache_dir);
    return lines_before_shutdown > 0 && lines_after_reinit > lines_before_shutdown;
}

int main() {
    TestUtils::TestRunner runner("Telemetry Tests");
    runner.run_test("record_many_then_flush", test_record_many_then_flush());
    runner.run_test("shutdown_then_reinit", test_shutdown_then_reinit_then_record());
    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
