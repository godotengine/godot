#pragma once

#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/variant/variant.h"
#include "servers/rendering/rendering_device.h"
#include "../core/gs_project_settings.h"
#include "../logger/gs_logger.h"

namespace GaussianSplatting {

// Legacy aliases -- delegate to the canonical gs::settings helpers so that
// existing call sites inside this namespace continue to compile unchanged.
static inline bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
    return gs::settings::get_bool(ps, name, fallback);
}

static inline int _get_int_setting(ProjectSettings *ps, const StringName &name, int fallback) {
    ERR_FAIL_NULL_V(ps, fallback);
    if (!ps->has_setting(name)) {
        return fallback;
    }
    Variant value = ps->get_setting_with_override(name);
    if (value.get_type() == Variant::INT) {
        return static_cast<int>(value.operator int64_t());
    }
    if (value.get_type() == Variant::FLOAT) {
        return static_cast<int>(value.operator double());
    }
    if (value.get_type() == Variant::BOOL) {
        return value.operator bool() ? 1 : 0;
    }
    return fallback;
}

static inline int get_debug_frame_log_frequency(int fallback = 0) {
#ifdef GS_SILENCE_LOGS
    return 0;
#else
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        int freq = _get_int_setting(ps, "rendering/gaussian_splatting/debug/frame_log_frequency", fallback);
        if (gs::settings::is_all_debug_enabled(ps) && freq <= 0) {
            freq = 1;
        }
        return freq;
    }
    return fallback;
#endif
}

static inline bool is_debug_frame_logging_enabled() {
#ifdef GS_SILENCE_LOGS
    return false;
#else
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        if (gs::settings::is_all_debug_enabled(ps)) {
            return true;
        }
        return get_debug_frame_log_frequency(0) > 0;
    }
    return false;
#endif
}

static inline bool is_debug_resource_logging_enabled() {
    return gs::settings::is_data_log_enabled();
}

static inline bool is_debug_force_unclustered_lights_enabled() {
#ifdef GS_SILENCE_LOGS
    return false;
#else
    if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
        return gs::settings::get_bool(ps, "rendering/gaussian_splatting/debug/force_unclustered_lights", false);
    }
    return false;
#endif
}

/**
 * Color scheme for different GPU pass types.
 * These colors are optimized for visibility in RenderDoc's event browser.
 */
struct PassColors {
    static constexpr Color FRAME_BRACKET = Color(0.2f, 0.6f, 0.9f, 1.0f);  // Blue - frame boundaries
    static constexpr Color CULLING = Color(0.9f, 0.6f, 0.2f, 1.0f);        // Orange - culling passes
    static constexpr Color BINNING = Color(0.4f, 0.8f, 0.4f, 1.0f);        // Green - tile binning
    static constexpr Color PREFIX = Color(0.8f, 0.4f, 0.8f, 1.0f);         // Purple - prefix scan
    static constexpr Color SORTING = Color(0.9f, 0.4f, 0.4f, 1.0f);        // Red - sorting passes
    static constexpr Color RASTER = Color(0.4f, 0.4f, 0.9f, 1.0f);         // Indigo - rasterization
    static constexpr Color RESOLVE = Color(0.6f, 0.6f, 0.6f, 1.0f);        // Gray - resolve passes
};

/**
 * RAII helper for GPU debug markers.
 * Automatically begins a labeled region on construction and ends it on destruction.
 * Safe to use with null device pointers.
 */
class ScopedGpuMarker {
    RenderingDevice *device = nullptr;

public:
    ScopedGpuMarker(RenderingDevice *p_device, const char *p_label, const Color &p_color = Color(0.4f, 0.7f, 1.0f, 1.0f)) {
        device = p_device;
        if (device && p_label) {
            device->_draw_command_begin_label(String(p_label), p_color);
        }
    }

    ~ScopedGpuMarker() {
        if (device) {
            device->draw_command_end_label();
        }
    }

    // Non-copyable, non-movable
    ScopedGpuMarker(const ScopedGpuMarker &) = delete;
    ScopedGpuMarker &operator=(const ScopedGpuMarker &) = delete;
    ScopedGpuMarker(ScopedGpuMarker &&) = delete;
    ScopedGpuMarker &operator=(ScopedGpuMarker &&) = delete;
};

/**
 * Extended RAII helper for GPU debug markers with explicit color support.
 * Identical to ScopedGpuMarker but with a more descriptive name for use with PassColors.
 */
class ScopedGpuMarkerEx {
    RenderingDevice *device = nullptr;

public:
    ScopedGpuMarkerEx(RenderingDevice *p_device, const char *p_label, const Color &p_color) {
        device = p_device;
        if (device && p_label) {
            device->_draw_command_begin_label(String(p_label), p_color);
        }
    }

    ~ScopedGpuMarkerEx() {
        if (device) {
            device->draw_command_end_label();
        }
    }

    // Non-copyable, non-movable
    ScopedGpuMarkerEx(const ScopedGpuMarkerEx &) = delete;
    ScopedGpuMarkerEx &operator=(const ScopedGpuMarkerEx &) = delete;
    ScopedGpuMarkerEx(ScopedGpuMarkerEx &&) = delete;
    ScopedGpuMarkerEx &operator=(ScopedGpuMarkerEx &&) = delete;
};

// Convenience macros for different pass types with color coding.
// These create unique variable names using __LINE__ to avoid conflicts in the same scope.

#define GS_GPU_MARKER_CULL(device, label) \
    GaussianSplatting::ScopedGpuMarkerEx _gs_marker_cull_##__LINE__(device, label, GaussianSplatting::PassColors::CULLING)

#define GS_GPU_MARKER_BINNING(device, label) \
    GaussianSplatting::ScopedGpuMarkerEx _gs_marker_binning_##__LINE__(device, label, GaussianSplatting::PassColors::BINNING)

#define GS_GPU_MARKER_PREFIX(device, label) \
    GaussianSplatting::ScopedGpuMarkerEx _gs_marker_prefix_##__LINE__(device, label, GaussianSplatting::PassColors::PREFIX)

#define GS_GPU_MARKER_SORT(device, label) \
    GaussianSplatting::ScopedGpuMarkerEx _gs_marker_sort_##__LINE__(device, label, GaussianSplatting::PassColors::SORTING)

#define GS_GPU_MARKER_RASTER(device, label) \
    GaussianSplatting::ScopedGpuMarkerEx _gs_marker_raster_##__LINE__(device, label, GaussianSplatting::PassColors::RASTER)

#define GS_GPU_MARKER_RESOLVE(device, label) \
    GaussianSplatting::ScopedGpuMarkerEx _gs_marker_resolve_##__LINE__(device, label, GaussianSplatting::PassColors::RESOLVE)

} // namespace GaussianSplatting

// ============================================================================
// Debug RID Free Macros - Phase 6: Track source of invalid ID frees
// ============================================================================

// Enable/disable detailed free tracking (set to 1 for debugging invalid ID errors)
#ifndef GS_DEBUG_FREE_TRACKING
#define GS_DEBUG_FREE_TRACKING 1
#endif

#if GS_DEBUG_FREE_TRACKING

// Safe free with source tracking - logs before freeing so we can see where invalid frees come from
#define GS_SAFE_FREE(device, rid, tag) \
    do { \
        if ((rid).is_valid() && (device) != nullptr) { \
            if (GaussianSplatting::is_debug_resource_logging_enabled()) { \
                GS_LOG_GPU_MEMORY_DEBUG(vformat("[GS_FREE] %s:%d [%s] freeing RID %llu", __FILE__, __LINE__, tag, (uint64_t)(rid).get_id())); \
            } \
            (device)->free(rid); \
            rid = RID(); \
        } \
    } while (0)

// Conditional free - only frees if RID is valid, with tracking
#define GS_SAFE_FREE_IF_VALID(device, rid, tag) \
    do { \
        if ((rid).is_valid()) { \
            if ((device) != nullptr) { \
                if (GaussianSplatting::is_debug_resource_logging_enabled()) { \
                    GS_LOG_GPU_MEMORY_DEBUG(vformat("[GS_FREE] %s:%d [%s] freeing RID %llu", __FILE__, __LINE__, tag, (uint64_t)(rid).get_id())); \
                } \
                (device)->free(rid); \
            } else { \
                if (GaussianSplatting::is_debug_resource_logging_enabled()) { \
                    GS_LOG_GPU_MEMORY_DEBUG(vformat("[GS_FREE] %s:%d [%s] SKIP (null device) RID %llu", __FILE__, __LINE__, tag, (uint64_t)(rid).get_id())); \
                } \
            } \
            rid = RID(); \
        } \
    } while (0)

#else // !GS_DEBUG_FREE_TRACKING

// Production versions without logging
#define GS_SAFE_FREE(device, rid, tag) \
    do { \
        if ((rid).is_valid() && (device) != nullptr) { \
            (device)->free(rid); \
            rid = RID(); \
        } \
    } while (0)

#define GS_SAFE_FREE_IF_VALID(device, rid, tag) \
    do { \
        if ((rid).is_valid() && (device) != nullptr) { \
            (device)->free(rid); \
            rid = RID(); \
        } \
    } while (0)

#endif // GS_DEBUG_FREE_TRACKING

