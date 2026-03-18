#ifndef GAUSSIAN_SPLATTING_IO_SETTINGS_UTILS_H
#define GAUSSIAN_SPLATTING_IO_SETTINGS_UTILS_H

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/math/math_funcs.h"
#include "core/variant/variant.h"

namespace GaussianSplattingIO {

inline bool get_bool_setting(ProjectSettings *ps, const StringName &name, bool fallback) {
	ERR_FAIL_NULL_V(ps, fallback);

	if (!ps->has_setting(name)) {
		return fallback;
	}

	Variant value = ps->get_setting_with_override(name);
	if (value.get_type() == Variant::BOOL) {
		return (bool)value;
	}
	if (value.get_type() == Variant::INT) {
		return value.operator int64_t() != 0;
	}
	if (value.get_type() == Variant::FLOAT) {
		return !Math::is_zero_approx((float)value.operator double());
	}

	return fallback;
}

inline bool is_data_log_enabled() {
#ifdef GS_SILENCE_LOGS
	return false;
#else
	if (ProjectSettings *ps = ProjectSettings::get_singleton()) {
		if (get_bool_setting(ps, "rendering/gaussian_splatting/debug/enable_all_debug", false)) {
			return true;
		}
		return get_bool_setting(ps, "rendering/gaussian_splatting/debug/enable_data_logging", false);
	}
	return false;
#endif
}

} // namespace GaussianSplattingIO

#endif // GAUSSIAN_SPLATTING_IO_SETTINGS_UTILS_H
