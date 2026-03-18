#include "gs_logger.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/os/os.h"
#include <mutex>
#include <unordered_map>

namespace gs_logger {

namespace {

std::once_flag s_init_flag;
std::array<std::atomic<int>, static_cast<size_t>(Category::CATEGORY_MAX)> s_levels;
std::mutex s_rate_limit_mutex;

struct RateLimitKey {
    uint8_t category = 0;
    uint8_t level = 0;
    uint64_t fingerprint = 0;

    bool operator==(const RateLimitKey &p_other) const {
        return category == p_other.category && level == p_other.level && fingerprint == p_other.fingerprint;
    }
};

struct RateLimitKeyHasher {
    size_t operator()(const RateLimitKey &p_key) const {
        const uint64_t metadata = (uint64_t(p_key.category) << 8) | uint64_t(p_key.level);
        uint64_t mixed = p_key.fingerprint;
        mixed ^= metadata + 0x9e3779b97f4a7c15ULL + (mixed << 6) + (mixed >> 2);
        return static_cast<size_t>(mixed);
    }
};

std::unordered_map<RateLimitKey, uint64_t, RateLimitKeyHasher> s_last_log_usec_by_key;
std::atomic<int> s_global_verbosity;
std::atomic<uint64_t> s_rate_limit_usec;

// Thread-local fast-path cache to reduce mutex contention.  Each thread
// remembers the last rate-limited key and its timestamp so repeat messages
// on the same thread skip the global lock entirely.
struct ThreadLocalRateCache {
    RateLimitKey last_key;
    uint64_t last_usec = 0;
};
thread_local ThreadLocalRateCache tl_rate_cache;

constexpr Level k_dev_default_info = Level::WARN;

constexpr Level k_default_levels[] = {
        k_dev_default_info, // GENERAL
        k_dev_default_info, // RENDERER
        Level::WARN,        // STREAMING
        Level::WARN,        // GPU_SORT
        Level::WARN,        // GPU_MEMORY
        Level::WARN,        // COMPOSITOR
        Level::WARN,        // COMMAND_BUFFER
        k_dev_default_info, // TESTS
};

const char *category_setting_key(Category p_category) {
    switch (p_category) {
        case Category::GENERAL:
            return "rendering/gaussian_splatting/logging/general";
        case Category::RENDERER:
            return "rendering/gaussian_splatting/logging/renderer";
        case Category::STREAMING:
            return "rendering/gaussian_splatting/logging/streaming";
        case Category::GPU_SORT:
            return "rendering/gaussian_splatting/logging/gpu_sort";
        case Category::GPU_MEMORY:
            return "rendering/gaussian_splatting/logging/gpu_memory";
        case Category::COMPOSITOR:
            return "rendering/gaussian_splatting/logging/compositor";
        case Category::COMMAND_BUFFER:
            return "rendering/gaussian_splatting/logging/command_buffer";
        case Category::TESTS:
            return "rendering/gaussian_splatting/logging/tests";
        default:
            return "rendering/gaussian_splatting/logging/general";
    }
}

Level string_to_level(const String &p_value, Level p_default) {
    String lower = p_value.to_lower();
    if (lower == "off") {
        return Level::OFF;
    }
    if (lower == "silent") {
        return Level::WARN;
    }
    if (lower == "error") {
        return Level::ERROR;
    }
    if (lower == "errors") {
        return Level::ERROR;
    }
    if (lower == "warn" || lower == "warning") {
        return Level::WARN;
    }
    if (lower == "warnings") {
        return Level::WARN;
    }
    if (lower == "info") {
        return Level::INFO;
    }
    if (lower == "debug") {
        return Level::DEBUG;
    }
    if (lower == "trace" || lower == "verbose") {
        return Level::TRACE;
    }
    return p_default;
}

void ensure_initialized() {
    std::call_once(s_init_flag, []() {
        for (size_t i = 0; i < static_cast<size_t>(Category::CATEGORY_MAX); i++) {
            s_levels[i].store(static_cast<int>(k_default_levels[i]), std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(s_rate_limit_mutex);
            s_last_log_usec_by_key.clear();
        }
        s_global_verbosity.store(static_cast<int>(Level::WARN), std::memory_order_relaxed);
        s_rate_limit_usec.store(1000000, std::memory_order_relaxed);

        ProjectSettings *settings = ProjectSettings::get_singleton();
        if (!settings) {
            return;
        }

        if (settings->has_setting("rendering/gaussian_splatting/logging/verbosity")) {
            Variant value = settings->get_setting_with_override("rendering/gaussian_splatting/logging/verbosity");
            Level level = Level::WARN;
            if (value.get_type() == Variant::STRING) {
                level = string_to_level((String)value, level);
            } else if (value.is_num()) {
                int enum_value = int(value);
                enum_value = CLAMP(enum_value, int(Level::OFF), int(Level::TRACE));
                level = static_cast<Level>(enum_value);
            }
            s_global_verbosity.store(static_cast<int>(level), std::memory_order_relaxed);
        }
        if (settings->has_setting("rendering/gaussian_splatting/logging/rate_limit_ms")) {
            Variant value = settings->get_setting_with_override("rendering/gaussian_splatting/logging/rate_limit_ms");
            uint64_t rate_ms = 1000;
            if (value.is_num()) {
                int64_t ms_value = int64_t(value);
                if (ms_value < 0) {
                    ms_value = 0;
                }
                rate_ms = static_cast<uint64_t>(ms_value);
            }
            s_rate_limit_usec.store(rate_ms * 1000, std::memory_order_relaxed);
        }

        for (size_t i = 0; i < static_cast<size_t>(Category::CATEGORY_MAX); i++) {
            Category category = static_cast<Category>(i);
            const char *setting_key = category_setting_key(category);
            if (!settings->has_setting(setting_key)) {
                continue;
            }
            Variant value = settings->get_setting_with_override(setting_key);
            Level level = k_default_levels[i];
            if (value.get_type() == Variant::STRING) {
                level = string_to_level((String)value, level);
            } else if (value.is_num()) {
                int enum_value = int(value);
                enum_value = CLAMP(enum_value, int(Level::OFF), int(Level::TRACE));
                level = static_cast<Level>(enum_value);
            }
            s_levels[i].store(static_cast<int>(level), std::memory_order_relaxed);
        }
    });
}

String build_prefix(Category p_category, Level p_level) {
    return "[" + category_to_string(p_category) + "][" + level_to_string(p_level) + "] ";
}

RateLimitKey make_rate_limit_key(Category p_category, Level p_level, const String &p_message) {
    RateLimitKey key;
    key.category = static_cast<uint8_t>(p_category);
    key.level = static_cast<uint8_t>(p_level);
    key.fingerprint = p_message.hash64();
    return key;
}

bool should_rate_limit(Level p_level) {
    return p_level == Level::INFO || p_level == Level::DEBUG || p_level == Level::TRACE;
}

bool check_rate_limit_key(const RateLimitKey &p_key, uint64_t p_now_usec, uint64_t p_rate_limit_usec) {
    if (p_rate_limit_usec == 0) {
        return true;
    }

    // Thread-local fast path: if this thread recently suppressed the same key,
    // skip the global lock entirely.
    if (tl_rate_cache.last_key == p_key && tl_rate_cache.last_usec > 0 &&
            p_now_usec >= tl_rate_cache.last_usec &&
            (p_now_usec - tl_rate_cache.last_usec) < p_rate_limit_usec) {
        return false;
    }

    std::lock_guard<std::mutex> lock(s_rate_limit_mutex);
    auto it = s_last_log_usec_by_key.find(p_key);
    if (it != s_last_log_usec_by_key.end()) {
        const uint64_t last = it->second;
        if (last > 0 && p_now_usec >= last && (p_now_usec - last) < p_rate_limit_usec) {
            // Update thread-local cache so subsequent calls avoid the lock.
            tl_rate_cache.last_key = p_key;
            tl_rate_cache.last_usec = last;
            return false;
        }
    }
    if (s_last_log_usec_by_key.size() > 4096) {
        s_last_log_usec_by_key.clear();
    }
    s_last_log_usec_by_key[p_key] = p_now_usec;

    // Update thread-local cache with the accepted timestamp.
    tl_rate_cache.last_key = p_key;
    tl_rate_cache.last_usec = p_now_usec;
    return true;
}

} // namespace

void initialize() {
    ensure_initialized();
}

void set_level(Category p_category, Level p_level) {
    ensure_initialized();
    size_t index = static_cast<size_t>(p_category);
    if (index >= s_levels.size()) {
        return;
    }
    s_levels[index].store(static_cast<int>(p_level), std::memory_order_relaxed);
}

Level get_level(Category p_category) {
    ensure_initialized();
    size_t index = static_cast<size_t>(p_category);
    if (index >= s_levels.size()) {
        return Level::OFF;
    }
    return static_cast<Level>(s_levels[index].load(std::memory_order_relaxed));
}

bool is_enabled(Category p_category, Level p_level) {
#ifdef GS_SILENCE_LOGS
    (void)p_category;
    (void)p_level;
    return false;
#endif
    ensure_initialized();
    size_t index = static_cast<size_t>(p_category);
    if (index >= s_levels.size()) {
        return false;
    }
    Level current = static_cast<Level>(s_levels[index].load(std::memory_order_relaxed));
    Level global = static_cast<Level>(s_global_verbosity.load(std::memory_order_relaxed));
    if (global == Level::OFF) {
        return false;
    }
    Level effective = static_cast<int>(current) <= static_cast<int>(global) ? current : global;
    return static_cast<int>(p_level) <= static_cast<int>(effective) && effective != Level::OFF;
}

String level_to_string(Level p_level) {
    switch (p_level) {
        case Level::OFF:
            return "OFF";
        case Level::ERROR:
            return "ERROR";
        case Level::WARN:
            return "WARN";
        case Level::INFO:
            return "INFO";
        case Level::DEBUG:
            return "DEBUG";
        case Level::TRACE:
            return "TRACE";
        default:
            return "UNKNOWN";
    }
}

String category_to_string(Category p_category) {
    switch (p_category) {
        case Category::GENERAL:
            return "GENERAL";
        case Category::RENDERER:
            return "RENDERER";
        case Category::STREAMING:
            return "STREAMING";
        case Category::GPU_SORT:
            return "GPU_SORT";
        case Category::GPU_MEMORY:
            return "GPU_MEMORY";
        case Category::COMPOSITOR:
            return "COMPOSITOR";
        case Category::COMMAND_BUFFER:
            return "COMMAND_BUFFER";
        case Category::TESTS:
            return "TESTS";
        default:
            return "UNKNOWN";
    }
}

void log_message(Category p_category, Level p_level, const String &p_message) {
#ifdef GS_SILENCE_LOGS
    (void)p_category;
    (void)p_level;
    (void)p_message;
    return;
#endif
    ensure_initialized();
    uint64_t rate_limit = s_rate_limit_usec.load(std::memory_order_relaxed);
    if (rate_limit > 0 && should_rate_limit(p_level)) {
        OS *os = OS::get_singleton();
        if (!os) {
            rate_limit = 0;
        }
        if (rate_limit > 0) {
            const uint64_t now = os->get_ticks_usec();
            const RateLimitKey rate_key = make_rate_limit_key(p_category, p_level, p_message);
            if (!check_rate_limit_key(rate_key, now, rate_limit)) {
                return;
            }
        }
    }
    String formatted = build_prefix(p_category, p_level) + p_message;
    if (p_level == Level::INFO || p_level == Level::DEBUG || p_level == Level::TRACE) {
        formatted += " [enable: rendering/gaussian_splatting/logging/verbosity, rate_limit_ms]";
    }

    switch (p_level) {
        case Level::ERROR:
            ERR_PRINT(formatted);
            break;
        case Level::WARN:
            WARN_PRINT(formatted);
            break;
        case Level::INFO:
            print_line(formatted);
            break;
        case Level::DEBUG:
        case Level::TRACE:
#ifdef DEBUG_ENABLED
            print_verbose(formatted);
#else
            print_line(formatted);
#endif
            break;
        case Level::OFF:
        default:
            break;
    }
}

namespace test {

void reset_rate_limiter() {
    ensure_initialized();
    std::lock_guard<std::mutex> lock(s_rate_limit_mutex);
    s_last_log_usec_by_key.clear();
}

bool check_rate_limit(Category p_category, Level p_level, const String &p_message, uint64_t p_now_usec, uint64_t p_rate_limit_usec) {
    ensure_initialized();
    if (!should_rate_limit(p_level)) {
        return true;
    }
    const RateLimitKey rate_key = make_rate_limit_key(p_category, p_level, p_message);
    return check_rate_limit_key(rate_key, p_now_usec, p_rate_limit_usec);
}

} // namespace test

} // namespace gs_logger
