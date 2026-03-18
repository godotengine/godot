#include "sorting_config.h"

#include "../core/gs_project_settings.h"
#include "core/config/project_settings.h"
#include "core/math/math_funcs.h"
#include "core/object/callable_method_pointer.h"
#include "core/templates/hash_map.h"
#include "core/variant/variant.h"

namespace {
// Project settings helpers provided by gs_project_settings.h (gs::settings namespace).
static uint32_t _get_uint_setting(ProjectSettings *ps, const StringName &name, uint32_t fallback) {
    return gs::settings::get_uint(ps, name, fallback);
}

static float _get_float_setting(ProjectSettings *ps, const StringName &name, float fallback) {
    return gs::settings::get_float(ps, name, fallback);
}

static bool _get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
    return gs::settings::get_bool(ps, name, fallback);
}

static int _get_int_setting(ProjectSettings *ps, const StringName &name, int fallback) {
    return static_cast<int>(gs::settings::get_uint(ps, name, static_cast<uint32_t>(fallback)));
}
} // namespace

namespace {
static SortingStrategyConfig g_cached_sort_config;
static bool g_cached_sort_config_valid = false;
static bool g_sort_config_connected = false;

static void _on_sort_settings_changed() {
    g_cached_sort_config_valid = false;
}

static void _ensure_sort_config_callback() {
    if (g_sort_config_connected) {
        return;
    }
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        return;
    }
    ps->connect("settings_changed", callable_mp_static(&_on_sort_settings_changed));
    g_sort_config_connected = true;
}

static void _refresh_sort_config() {
    SortingStrategyConfig config;
    ProjectSettings *ps = ProjectSettings::get_singleton();
    if (!ps) {
        // Don't cache defaults - wait for ProjectSettings to exist.
        // Return default config but leave cache invalid so we retry later.
        return;
    }

    config.bitonic_max_elements = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/bitonic_max_elements", config.bitonic_max_elements);
    config.radix_max_elements = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/radix_max_elements", config.radix_max_elements);
    config.onesweep_max_elements = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/onesweep_max_elements", config.onesweep_max_elements);
    config.hybrid_trigger_elements = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/hybrid_trigger_elements", config.hybrid_trigger_elements);
    config.hybrid_batch_size = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/hybrid_batch_size", config.hybrid_batch_size);
    config.history_size = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/history_size", config.history_size);
    config.log_interval_frames = _get_uint_setting(ps, "rendering/gaussian_splatting/sorting/log_interval_frames", config.log_interval_frames);
    config.target_sort_time_ms = _get_float_setting(ps, "rendering/gaussian_splatting/sorting/target_sort_time_ms", config.target_sort_time_ms);
    config.log_metrics = _get_bool_setting(ps, "rendering/gaussian_splatting/sorting/log_metrics", config.log_metrics);
    config.force_algorithm = static_cast<SortingStrategyConfig::ForcedAlgorithm>(_get_int_setting(ps,
            "rendering/gaussian_splatting/sorting/force_algorithm",
            static_cast<int>(config.force_algorithm)));
    config.force_cpu_sort = _get_bool_setting(ps, "rendering/gaussian_splatting/sorting/force_cpu_sort", config.force_cpu_sort);

    config.sanitize();
    g_cached_sort_config = config;
    g_cached_sort_config_valid = true;
}
} // namespace

void SortingStrategyConfig::sanitize() {
    if (bitonic_max_elements == 0) {
        bitonic_max_elements = 1;
    }

    if (radix_max_elements < bitonic_max_elements) {
        radix_max_elements = bitonic_max_elements;
    }

    if (onesweep_max_elements < radix_max_elements) {
        onesweep_max_elements = radix_max_elements;
    }

    if (hybrid_trigger_elements < radix_max_elements) {
        hybrid_trigger_elements = radix_max_elements;
    }

    if (hybrid_batch_size == 0) {
        hybrid_batch_size = radix_max_elements;
    }

    if (history_size == 0) {
        history_size = 120;
    }

    if (log_interval_frames == 0) {
        log_interval_frames = 60;
    }

    if (target_sort_time_ms < 0.0f) {
        target_sort_time_ms = 0.0f;
    }

    const int force_algorithm_value = static_cast<int>(force_algorithm);
    if (force_algorithm_value < static_cast<int>(ForcedAlgorithm::AUTO) ||
            force_algorithm_value > static_cast<int>(ForcedAlgorithm::ONESWEEP)) {
        force_algorithm = ForcedAlgorithm::AUTO;
    }
}

String SortingStrategyConfig::describe_thresholds() const {
    return vformat("bitonic≤%d | radix≤%d | onesweep≤%d | hybrid>%d(batch %d)",
            bitonic_max_elements,
            radix_max_elements,
            onesweep_max_elements,
            hybrid_trigger_elements,
            hybrid_batch_size);
}

bool SortingStrategyConfig::is_algorithm_forced() const {
    return force_algorithm != ForcedAlgorithm::AUTO;
}

String SortingStrategyConfig::get_forced_algorithm_name() const {
    switch (force_algorithm) {
        case ForcedAlgorithm::AUTO:
            return "auto";
        case ForcedAlgorithm::RADIX:
            return "radix";
        case ForcedAlgorithm::BITONIC:
            return "bitonic";
        case ForcedAlgorithm::ONESWEEP:
            return "onesweep";
    }
    return "auto";
}

SortingStrategyConfig SortingStrategyConfig::load_from_project_settings() {
    _ensure_sort_config_callback();
    if (!g_cached_sort_config_valid) {
        _refresh_sort_config();
    }
    // If cache still invalid (ProjectSettings not available), return sanitized defaults.
    if (!g_cached_sort_config_valid) {
        SortingStrategyConfig defaults;
        defaults.sanitize();
        return defaults;
    }
    return g_cached_sort_config;
}
