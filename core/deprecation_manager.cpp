#include "deprecation_manager.h"
#include "core/class_db.h"

#ifdef DEBUG_ENABLED

DeprecationInfo::DeprecationInfo() :
		warned(false),
		custom_warning("") {
}

DeprecationInfo::DeprecationInfo(String p_custom_warning) :
		warned(false),
		custom_warning(p_custom_warning) {
}

HashMap<String, DeprecationManager::ClassInfo> DeprecationManager::classes;

void DeprecationManager::warn_deprecated(const StringName &p_name, const StringName &p_type, const DeprecationInfo &p_info) {
	String warning = "The " + p_type + " '" + p_name + "' has been deprecated and will be removed in the future.";

	if (!p_info.custom_warning.empty())
		warning = warning + " " + p_info.custom_warning;

	WARN_PRINTS(warning);
}

void DeprecationManager::deprecate_method(const StringName &p_class, const StringName &p_method, DeprecationInfo p_info) {
	classes[p_class].deprecated_methods[p_method] = p_info;
}

void DeprecationManager::deprecate_constant(const StringName &p_class, const StringName &p_constant, DeprecationInfo p_info) {
	classes[p_class].deprecated_constants[p_constant] = p_info;
}

void DeprecationManager::deprecate_signal(const StringName &p_class, const StringName &p_signal, DeprecationInfo p_info) {
	classes[p_class].deprecated_signals[p_signal] = p_info;
}

void DeprecationManager::deprecate_property(const StringName &p_class, const StringName &p_property, DeprecationInfo p_info) {
	classes[p_class].deprecated_properties[p_property] = p_info;
}

void DeprecationManager::test_method_deprecated(const StringName &p_class, const StringName &p_method_name) {
	if (classes[p_class].deprecated_methods.has(p_method_name)) {
		DeprecationInfo *info = &classes[p_class].deprecated_properties[p_method_name];

		if (!info->warned) {
			warn_deprecated(p_method_name, "method", *info);
			info->warned = true;
		}
	}
}

void DeprecationManager::test_constant_deprecated(const StringName &p_class, const StringName &p_const) {
	if (classes[p_class].deprecated_constants.has(p_const)) {
		DeprecationInfo *info = &classes[p_class].deprecated_constants[p_const];

		if (!info->warned) {
			warn_deprecated(p_const, "constant", *info);
			info->warned = true;
		}
	}
}

void DeprecationManager::test_property_deprecated(const StringName &p_class, const StringName &p_property) {
	if (classes[p_class].deprecated_properties.has(p_property)) {
		DeprecationInfo *info = &classes[p_class].deprecated_properties[p_property];

		if (!info->warned) {
			warn_deprecated(p_property, "property", *info);
			info->warned = true;
		}
	}
}

void DeprecationManager::test_signal_deprecated(const StringName &p_class, const StringName &p_signal) {
	if (classes[p_class].deprecated_signals.has(p_signal)) {
		DeprecationInfo *info = &classes[p_class].deprecated_signals[p_signal];

		if (!info->warned) {
			warn_deprecated(p_signal, "signal", *info);
			info->warned = true;
		}
	}
}

#endif
