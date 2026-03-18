/**
 * @file gs_project_settings.h
 * @brief Shared ProjectSettings helper utilities for the Gaussian Splatting module.
 *
 * This header provides canonical, type-safe accessors for reading Godot
 * ProjectSettings values.  It replaces the many per-file static copies of
 * _get_bool_setting / _ps_get_uint / _ps_get_float that were duplicated
 * across the module.
 *
 * Usage:
 *   #include "core/gs_project_settings.h"
 *   bool val = gs::settings::get_bool(ps, "rendering/gaussian_splatting/debug/enable_all_debug", false);
 */

#ifndef GS_PROJECT_SETTINGS_H
#define GS_PROJECT_SETTINGS_H

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/variant/variant.h"

namespace gs {
namespace settings {

/**
 * @brief Read a boolean from ProjectSettings with type coercion.
 *
 * Handles BOOL, INT (non-zero == true) and FLOAT (non-zero-approx == true).
 * Returns @p p_fallback when @p p_ps is null, the setting does not exist, or the
 * stored type cannot be coerced.
 */
static inline bool get_bool(ProjectSettings *p_ps, const StringName &p_name, bool p_fallback) {
	ERR_FAIL_NULL_V(p_ps, p_fallback);

	if (!p_ps->has_setting(p_name)) {
		return p_fallback;
	}

	Variant value = p_ps->get_setting_with_override(p_name);
	if (value.get_type() == Variant::BOOL) {
		return (bool)value;
	}
	if (value.get_type() == Variant::INT) {
		return value.operator int64_t() != 0;
	}
	if (value.get_type() == Variant::FLOAT) {
		return !Math::is_zero_approx((float)value.operator double());
	}

	return p_fallback;
}

/**
 * @brief Read an unsigned 32-bit integer from ProjectSettings with type coercion.
 *
 * Handles INT and FLOAT types.  Negative values fall back to @p p_fallback.
 */
static inline uint32_t get_uint(ProjectSettings *p_ps, const StringName &p_name, uint32_t p_fallback) {
	if (!p_ps || !p_ps->has_setting(p_name)) {
		return p_fallback;
	}

	Variant value = p_ps->get_setting_with_override(p_name);
	if (value.get_type() == Variant::INT) {
		int64_t v = value.operator int64_t();
		return v < 0 ? p_fallback : uint32_t(v);
	}
	if (value.get_type() == Variant::FLOAT) {
		double v = value.operator double();
		return v < 0.0 ? p_fallback : uint32_t(Math::round(v));
	}

	return p_fallback;
}

/**
 * @brief Read a float from ProjectSettings with type coercion.
 *
 * Handles FLOAT and INT types.
 */
static inline float get_float(ProjectSettings *p_ps, const StringName &p_name, float p_fallback) {
	if (!p_ps || !p_ps->has_setting(p_name)) {
		return p_fallback;
	}

	Variant value = p_ps->get_setting_with_override(p_name);
	if (value.get_type() == Variant::FLOAT) {
		return (float)value.operator double();
	}
	if (value.get_type() == Variant::INT) {
		return (float)value.operator int64_t();
	}

	return p_fallback;
}

/**
 * @brief Convenience: check whether "all debug" is enabled for the GS module.
 */
static inline bool is_all_debug_enabled(ProjectSettings *p_ps) {
	return get_bool(p_ps, "rendering/gaussian_splatting/debug/enable_all_debug", false);
}

/**
 * @brief Check whether data-level debug logging is enabled.
 */
static inline bool is_data_log_enabled() {
#ifdef GS_SILENCE_LOGS
	return false;
#else
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (!ps) {
		return false;
	}
	if (is_all_debug_enabled(ps)) {
		return true;
	}
	return get_bool(ps, "rendering/gaussian_splatting/debug/enable_data_logging", false);
#endif
}

/**
 * @brief Check whether per-frame debug logging is enabled.
 */
static inline bool is_frame_log_enabled() {
#ifdef GS_SILENCE_LOGS
	return false;
#else
	ProjectSettings *ps = ProjectSettings::get_singleton();
	if (!ps) {
		return false;
	}
	if (is_all_debug_enabled(ps)) {
		return true;
	}
	return get_bool(ps, "rendering/gaussian_splatting/debug/enable_frame_logging", false);
#endif
}

} // namespace settings
} // namespace gs

#endif // GS_PROJECT_SETTINGS_H
