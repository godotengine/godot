#include "core/hash_map.h"
#include "core/ustring.h"

#ifndef DEPRECATION_MANAGER_H
#define DEPRECATION_MANAGER_H

#ifdef DEBUG_ENABLED

struct DeprecationInfo {
	bool warned;
	String custom_warning;

	DeprecationInfo();
	DeprecationInfo(String p_custom_message);
};

class DeprecationManager {
	struct ClassInfo {
		HashMap<String, DeprecationInfo> deprecated_methods;
		HashMap<String, DeprecationInfo> deprecated_properties;
		HashMap<String, DeprecationInfo> deprecated_constants;
		HashMap<String, DeprecationInfo> deprecated_signals;
	};

	static HashMap<String, ClassInfo> classes;
	static HashMap<int, DeprecationInfo> deprecated_methods;

public:
	static void warn_deprecated(const StringName &p_name, const StringName &p_type, const DeprecationInfo &p_info);
	static void deprecate_method(const StringName &p_class, const StringName &p_method, DeprecationInfo p_info);
	static void deprecate_property(const StringName &p_class, const StringName &p_property, DeprecationInfo p_info);
	static void deprecate_constant(const StringName &p_class, const StringName &p_const, DeprecationInfo p_info);
	static void deprecate_signal(const StringName &p_class, const StringName &p_signal, DeprecationInfo p_info);
	static void test_method_deprecated(const StringName &p_class, const StringName &p_method_name);
	static void test_constant_deprecated(const StringName &p_class, const StringName &p_const);
	static void test_property_deprecated(const StringName &p_class, const StringName &p_property);
	static void test_signal_deprecated(const StringName &p_class, const StringName &p_signal);
};

#define DEPRECATE_METHOD(p_method, p_info) \
	DeprecationManager::deprecate_method(get_class_static(), p_method, p_info);
#define DEPRECATE_PROPERTY(p_property, p_info) \
	DeprecationManager::deprecate_property(get_class_static(), p_property, p_info);
#define DEPRECATE_SIGNAL(p_signal, p_info) \
	DeprecationManager::deprecate_signal(get_class_static(), p_signal, p_info);
#define DEPRECATE_CONSTANT(p_const, p_info) \
	DeprecationManager::deprecate_constant(get_class_static(), #p_const, p_info);

#define TEST_METHOD_DEPRECATED(p_class_name, p_method_name) \
	DeprecationManager::test_method_deprecated(p_class_name, p_method_name);
#define TEST_CONSTANT_DEPRECATED(p_class_name, p_const) \
	DeprecationManager::test_constant_deprecated(p_class_name, p_const);
#define TEST_SIGNAL_DEPRECATED(p_class_name, p_signal) \
	DeprecationManager::test_signal_deprecated(p_class_name, p_signal);
#define TEST_PROPERTY_DEPRECATED(p_class_name, p_property) \
	DeprecationManager::test_property_deprecated(p_class_name, p_property);

#else

#define DEPRECATE_METHOD(p_method, p_info)
#define DEPRECATE_PROPERTY(p_property, p_info)
#define DEPRECATE_SIGNAL(p_signal, p_info)
#define DEPRECATE_CONSTANT(p_const, p_info)

#define TEST_METHOD_DEPRECATED(p_class_name, p_method_name)
#define TEST_CONSTANT_DEPRECATED(p_class_name, p_const)
#define TEST_SIGNAL_DEPRECATED(p_class_name, p_signal)
#define TEST_PROPERTY_DEPRECATED(p_class_name, p_property)

#endif

#endif // DEPRECATION_MANAGER_H
