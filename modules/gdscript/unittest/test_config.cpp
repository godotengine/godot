/*************************************************************************/
/*  test_config.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "test_config.h"

#include "editor/editor_node.h"

TestConfig *TestConfig::singleton = NULL;

TestConfig::TestConfig() {
	singleton = this;

	GLOBAL_DEF("debug/testing/test/directory", "");
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/test/directory", PropertyInfo(Variant::STRING, "debug/testing/test/directory", PROPERTY_HINT_DIR));
	GLOBAL_DEF("debug/testing/test/stop_on_error", false);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/test/stop_on_error", PropertyInfo(Variant::BOOL, "debug/testing/test/stop_on_error", PROPERTY_HINT_NONE));
	GLOBAL_DEF("debug/testing/test/file_match", "");
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/test/file_match", PropertyInfo(Variant::STRING, "debug/testing/test/file_match", PROPERTY_HINT_PLACEHOLDER_TEXT, "*_test.gd"));
	GLOBAL_DEF("debug/testing/test/func_match", "");
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/test/func_match", PropertyInfo(Variant::STRING, "debug/testing/test/func_match", PROPERTY_HINT_PLACEHOLDER_TEXT, "test_*"));
	GLOBAL_DEF("debug/testing/test/default_test_result", "");
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/test/default_test_result", PropertyInfo(Variant::STRING, "debug/testing/test/default_test_result", PROPERTY_HINT_PLACEHOLDER_TEXT, "TextTestResult"));

	/*GLOBAL_DEF("debug/testing/documentation/should_test", true);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/documentation/should_test", PropertyInfo(Variant::BOOL, "debug/testing/documentation/should_test", PROPERTY_HINT_NONE));
	GLOBAL_DEF("debug/testing/documentation/directory", "");
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/documentation/directory", PropertyInfo(Variant::STRING, "debug/testing/documentation/directory", PROPERTY_HINT_DIR));

	GLOBAL_DEF("debug/testing/coverage/should_compute", true);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/coverage/should_compute", PropertyInfo(Variant::BOOL, "debug/testing/coverage/should_compute", PROPERTY_HINT_NONE));
	GLOBAL_DEF("debug/testing/coverage/directory", "");
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/coverage/directory", PropertyInfo(Variant::STRING, "debug/testing/coverage/directory", PROPERTY_HINT_DIR));
	GLOBAL_DEF("debug/testing/coverage/minimum_percent", 0);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/coverage/minimum_percent", PropertyInfo(Variant::INT, "debug/testing/coverage/minimum_percent", PROPERTY_HINT_RANGE, "0,100,1"));*/

	GLOBAL_DEF("debug/testing/log/console", true);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/log/console", PropertyInfo(Variant::BOOL, "debug/testing/log/console", PROPERTY_HINT_NONE));
	GLOBAL_DEF("debug/testing/log/on_success", true);
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/log/on_success", PropertyInfo(Variant::BOOL, "debug/testing/log/on_success", PROPERTY_HINT_NONE));
	GLOBAL_DEF("debug/testing/log/fail_greater_equal", 4); // Fail on ERROR or FATAL
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/log/fail_greater_equal", PropertyInfo(Variant::INT, "debug/testing/log/fail_greater_equal", PROPERTY_HINT_ENUM, "Trace,Debug,Info,Warn,Error,Fatal"));
	GLOBAL_DEF("debug/testing/log/filter_below", 2); // Don't show TRACE or DEBUG
	ProjectSettings::get_singleton()->set_custom_property_info("debug/testing/log/filter_below", PropertyInfo(Variant::INT, "debug/testing/log/filter_below", PROPERTY_HINT_ENUM, "Trace,Debug,Info,Warn,Error,Fatal"));
}

TestConfig *TestConfig::get_singleton() {
	return singleton;
}

String get_setting(const String &p_name, const String &p_default) {
	String value = ProjectSettings::get_singleton()->get(p_name);
	if (value.empty()) {
		value = p_default;
	}
	return value;
}

String TestConfig::test_directory() const {
	return get_setting("debug/testing/test/directory", "res://");
}

String TestConfig::test_file_match() const {
	return get_setting("debug/testing/test/file_match", "*_test.gd");
}

String TestConfig::test_func_match() const {
	return get_setting("debug/testing/test/func_match", "test_*");
}

Ref<TestResult> TestConfig::make_result() const {
	const String &script = get_setting("debug/testing/test/default_test_result", "TextTestResult");
	Object *obj = NULL;
	if (ClassDB::class_exists(script)) {
		obj = ClassDB::instance(script);
	} else {
		Ref<Script> script_res = ResourceLoader::load(script);
		if (script_res.is_valid() && script_res->can_instance()) {
			StringName instance_type = script_res->get_instance_base_type();
			obj = ClassDB::instance(instance_type);
		}
	}
	ERR_FAIL_COND_V(!obj, NULL);
	Ref<TestResult> test_result = Object::cast_to<TestResult>(obj);
	ERR_EXPLAIN("Can't load script '" + script + "', it does not inherit from a TestResult type");
	ERR_FAIL_COND_V(test_result.is_null(), NULL);

	return test_result;
}

bool TestConfig::log_console() const {
	return ProjectSettings::get_singleton()->get("debug/testing/log/console");
}

bool TestConfig::log_on_success() const {
	return ProjectSettings::get_singleton()->get("debug/testing/log/on_success");
}

TestLog::LogLevel TestConfig::log_fail_greater_equal() const {
	return (TestLog::LogLevel)(int)ProjectSettings::get_singleton()->get("debug/testing/log/fail_greater_equal");
}

void TestConfig::_bind_methods() {
}
