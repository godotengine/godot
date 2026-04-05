#ifndef GS_SORTING_SETTINGS_UTILS_H
#define GS_SORTING_SETTINGS_UTILS_H

#include "core/config/project_settings.h"
#include "../core/gs_project_settings.h"

#include "core/error/error_macros.h"
#include "core/string/string_name.h"

namespace gs {
namespace sorting_settings {

static inline const StringName &target_sort_time_path() {
	static const StringName path("rendering/gaussian_splatting/sorting/target_sort_time_ms");
	return path;
}

static inline const StringName &legacy_gpu_target_sort_time_path() {
	static const StringName path("rendering/gaussian_splatting/gpu_sorting/target_sort_time_ms");
	return path;
}

static inline bool has_explicit_target_sort_time_override(ProjectSettings *p_ps) {
	if (!p_ps) {
		return false;
	}
	if (!p_ps->has_setting(String(target_sort_time_path()))) {
		return false;
	}
	if (!p_ps->is_builtin_setting(String(target_sort_time_path()))) {
		return true;
	}
	if (!p_ps->property_can_revert(target_sort_time_path())) {
		return true;
	}
	return p_ps->get_setting_with_override(target_sort_time_path()) != p_ps->property_get_revert(target_sort_time_path());
}

static inline void register_canonical_target_sort_time_setting(ProjectSettings *p_ps, float p_default_value) {
	const bool had_project_target_override = p_ps && p_ps->has_setting(String(target_sort_time_path())) &&
			!p_ps->is_builtin_setting(String(target_sort_time_path()));
	const int prior_target_order = had_project_target_override ? p_ps->get_order(target_sort_time_path()) : -1;
	GLOBAL_DEF(String(target_sort_time_path()), p_default_value);
	if (had_project_target_override) {
		p_ps->set_order(target_sort_time_path(), prior_target_order);
	}
}

static inline float get_target_sort_time_ms(ProjectSettings *p_ps, float p_fallback) {
	if (!p_ps) {
		return p_fallback;
	}
	if (has_explicit_target_sort_time_override(p_ps)) {
		return gs::settings::get_float(p_ps, target_sort_time_path(), p_fallback);
	}
	if (p_ps->has_setting(legacy_gpu_target_sort_time_path())) {
		WARN_PRINT_ONCE(vformat("[GaussianSplatting] Project setting '%s' is deprecated; use '%s' instead. Legacy alias support is read-only compatibility and will be removed after project migration.",
				String(legacy_gpu_target_sort_time_path()),
				String(target_sort_time_path())));
		return gs::settings::get_float(p_ps, legacy_gpu_target_sort_time_path(), p_fallback);
	}
	if (p_ps->has_setting(String(target_sort_time_path()))) {
		return gs::settings::get_float(p_ps, target_sort_time_path(), p_fallback);
	}
	return p_fallback;
}

} // namespace sorting_settings
} // namespace gs

#endif // GS_SORTING_SETTINGS_UTILS_H
