#ifndef GS_LOGGER_H
#define GS_LOGGER_H

#include "core/string/ustring.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>

// Windows defines ERROR as 0 in WinGDI.h - undefine to avoid enum conflict
#ifdef ERROR
#undef ERROR
#endif

#ifndef GS_LOG_MAX_LEVEL
#if defined(DEBUG_ENABLED) || defined(DEV_ENABLED)
// Enable INFO logs in dev/debug builds for diagnostics.
#define GS_LOG_MAX_LEVEL ((gs_logger::Level)3)  // Level::INFO
#else
// Disable all non-error logs for production.
#define GS_LOG_MAX_LEVEL ((gs_logger::Level)1)  // Level::ERROR
#endif
#endif

namespace gs_logger {

enum class Level : uint8_t {
    OFF = 0,
    ERROR,
    WARN,
    INFO,
    DEBUG,
    TRACE,
};

enum class Category : uint8_t {
    GENERAL = 0,
    RENDERER,
    STREAMING,
    GPU_SORT,
    GPU_MEMORY,
    COMPOSITOR,
    COMMAND_BUFFER,
    TESTS,
    CATEGORY_MAX
};

void initialize();
void set_level(Category p_category, Level p_level);
Level get_level(Category p_category);
bool is_enabled(Category p_category, Level p_level);
String level_to_string(Level p_level);
String category_to_string(Category p_category);

void log_message(Category p_category, Level p_level, const String &p_message);

namespace test {
void reset_rate_limiter();
bool check_rate_limit(Category p_category, Level p_level, const String &p_message, uint64_t p_now_usec, uint64_t p_rate_limit_usec);
} // namespace test

template <typename Callable>
void log(Category p_category, Level p_level, Callable &&p_callable) {
    if (!is_enabled(p_category, p_level)) {
        return;
    }
    log_message(p_category, p_level, p_callable());
}

} // namespace gs_logger

#define GS_LOG_ENABLED(cat, lvl) (gs_logger::is_enabled((cat), (lvl)))

#define GS_LOG_CALL(cat, lvl, expr)                                                  \
    do {                                                                             \
        if constexpr ((int)(lvl) <= (int)(GS_LOG_MAX_LEVEL)) {                       \
            if (gs_logger::is_enabled((cat), (lvl))) {                               \
                gs_logger::log((cat), (lvl), [&]() -> String { return (expr); });    \
            }                                                                        \
        }                                                                            \
    } while (false)

#define GS_LOG_ERROR(cat, expr) GS_LOG_CALL((cat), gs_logger::Level::ERROR, (expr))
#define GS_LOG_WARN(cat, expr) GS_LOG_CALL((cat), gs_logger::Level::WARN, (expr))
#define GS_LOG_INFO(cat, expr) GS_LOG_CALL((cat), gs_logger::Level::INFO, (expr))
#define GS_LOG_DEBUG(cat, expr) GS_LOG_CALL((cat), gs_logger::Level::DEBUG, (expr))
#define GS_LOG_TRACE(cat, expr) GS_LOG_CALL((cat), gs_logger::Level::TRACE, (expr))

#define GS_LOG_INFO_DEFAULT(expr) GS_LOG_INFO(gs_logger::Category::GENERAL, (expr))
#define GS_LOG_WARN_DEFAULT(expr) GS_LOG_WARN(gs_logger::Category::GENERAL, (expr))
#define GS_LOG_ERROR_DEFAULT(expr) GS_LOG_ERROR(gs_logger::Category::GENERAL, (expr))

#define GS_LOG_RENDERER_ERROR(expr) GS_LOG_ERROR(gs_logger::Category::RENDERER, (expr))
#define GS_LOG_RENDERER_WARN(expr) GS_LOG_WARN(gs_logger::Category::RENDERER, (expr))
#define GS_LOG_RENDERER_INFO(expr) GS_LOG_INFO(gs_logger::Category::RENDERER, (expr))
#define GS_LOG_RENDERER_DEBUG(expr) GS_LOG_DEBUG(gs_logger::Category::RENDERER, (expr))
#define GS_LOG_RENDERER_TRACE(expr) GS_LOG_TRACE(gs_logger::Category::RENDERER, (expr))

#define GS_LOG_STREAMING_ERROR(expr) GS_LOG_ERROR(gs_logger::Category::STREAMING, (expr))
#define GS_LOG_STREAMING_WARN(expr) GS_LOG_WARN(gs_logger::Category::STREAMING, (expr))
#define GS_LOG_STREAMING_INFO(expr) GS_LOG_INFO(gs_logger::Category::STREAMING, (expr))
#define GS_LOG_STREAMING_DEBUG(expr) GS_LOG_DEBUG(gs_logger::Category::STREAMING, (expr))

#define GS_LOG_GPU_SORT_ERROR(expr) GS_LOG_ERROR(gs_logger::Category::GPU_SORT, (expr))
#define GS_LOG_GPU_SORT_WARN(expr) GS_LOG_WARN(gs_logger::Category::GPU_SORT, (expr))
#define GS_LOG_GPU_SORT_INFO(expr) GS_LOG_INFO(gs_logger::Category::GPU_SORT, (expr))
#define GS_LOG_GPU_SORT_DEBUG(expr) GS_LOG_DEBUG(gs_logger::Category::GPU_SORT, (expr))

#define GS_LOG_GPU_MEMORY_INFO(expr) GS_LOG_INFO(gs_logger::Category::GPU_MEMORY, (expr))
#define GS_LOG_GPU_MEMORY_DEBUG(expr) GS_LOG_DEBUG(gs_logger::Category::GPU_MEMORY, (expr))
#define GS_LOG_GPU_MEMORY_WARN(expr) GS_LOG_WARN(gs_logger::Category::GPU_MEMORY, (expr))

#define GS_LOG_COMPOSITOR_INFO(expr) GS_LOG_INFO(gs_logger::Category::COMPOSITOR, (expr))
#define GS_LOG_COMPOSITOR_DEBUG(expr) GS_LOG_DEBUG(gs_logger::Category::COMPOSITOR, (expr))

#define GS_LOG_COMMAND_BUFFER_WARN(expr) GS_LOG_WARN(gs_logger::Category::COMMAND_BUFFER, (expr))
#define GS_LOG_COMMAND_BUFFER_DEBUG(expr) GS_LOG_DEBUG(gs_logger::Category::COMMAND_BUFFER, (expr))
#define GS_LOG_COMMAND_BUFFER_TRACE(expr) GS_LOG_TRACE(gs_logger::Category::COMMAND_BUFFER, (expr))

#define GS_LOG_TESTS_INFO(expr) GS_LOG_INFO(gs_logger::Category::TESTS, (expr))
#define GS_LOG_TESTS_DEBUG(expr) GS_LOG_DEBUG(gs_logger::Category::TESTS, (expr))
#define GS_LOG_TESTS_WARN(expr) GS_LOG_WARN(gs_logger::Category::TESTS, (expr))
#define GS_LOG_TESTS_ERROR(expr) GS_LOG_ERROR(gs_logger::Category::TESTS, (expr))
#define GS_LOG_TESTS_TRACE(expr) GS_LOG_TRACE(gs_logger::Category::TESTS, (expr))

#define GS_LOG_EVERY_N(cat, lvl, counter_expr, interval, expr)                   \
    do {                                                                         \
        static uint64_t counter_expr = 0;                                        \
        if ((interval) > 0 && (counter_expr++ % uint64_t(interval)) == 0) {      \
            GS_LOG_CALL((cat), (lvl), (expr));                                   \
        }                                                                        \
    } while (false)

#define GS_LOG_INFO_DEFAULT_EVERY(counter_expr, interval, expr) \
    GS_LOG_EVERY_N(gs_logger::Category::GENERAL, gs_logger::Level::INFO, counter_expr, interval, (expr))
#define GS_LOG_RENDERER_INFO_EVERY(counter_expr, interval, expr) \
    GS_LOG_EVERY_N(gs_logger::Category::RENDERER, gs_logger::Level::INFO, counter_expr, interval, (expr))

#ifdef GS_SILENCE_LOGS
#undef GS_LOG_ENABLED
#undef GS_LOG_CALL
#undef GS_LOG_EVERY_N
#define GS_LOG_ENABLED(cat, lvl) (false)
#define GS_LOG_CALL(cat, lvl, expr) ((void)0)
#define GS_LOG_EVERY_N(cat, lvl, counter_expr, interval, expr) ((void)0)
#endif

#endif // GS_LOGGER_H
