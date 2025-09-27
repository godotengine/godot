/**************************************************************************/
/*  test_project_settings.h                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/variant/variant.h"
#include "tests/test_macros.h"

class TestProjectSettingsInternalsAccessor {
public:
	static String &resource_path() {
		return ProjectSettings::get_singleton()->resource_path;
	}
};

namespace TestProjectSettings {

TEST_CASE("[ProjectSettings] Get existing setting") {
	CHECK(ProjectSettings::get_singleton()->has_setting("application/run/main_scene"));

	Variant variant = ProjectSettings::get_singleton()->get_setting("application/run/main_scene");
	CHECK_EQ(variant.get_type(), Variant::STRING);

	String name = variant;
	CHECK_EQ(name, String());
}

TEST_CASE("[ProjectSettings] Default value is ignored if setting exists") {
	CHECK(ProjectSettings::get_singleton()->has_setting("application/run/main_scene"));

	Variant variant = ProjectSettings::get_singleton()->get_setting("application/run/main_scene", "SomeDefaultValue");
	CHECK_EQ(variant.get_type(), Variant::STRING);

	String name = variant;
	CHECK_EQ(name, String());
}

TEST_CASE("[ProjectSettings] Non existing setting is null") {
	CHECK_FALSE(ProjectSettings::get_singleton()->has_setting("not_existing_setting"));

	Variant variant = ProjectSettings::get_singleton()->get_setting("not_existing_setting");
	CHECK_EQ(variant.get_type(), Variant::NIL);
}

TEST_CASE("[ProjectSettings] Non existing setting should return default value") {
	CHECK_FALSE(ProjectSettings::get_singleton()->has_setting("not_existing_setting"));

	Variant variant = ProjectSettings::get_singleton()->get_setting("not_existing_setting");
	CHECK_EQ(variant.get_type(), Variant::NIL);

	variant = ProjectSettings::get_singleton()->get_setting("not_existing_setting", "my_nice_default_value");
	CHECK_EQ(variant.get_type(), Variant::STRING);

	String name = variant;
	CHECK_EQ(name, "my_nice_default_value");

	CHECK_FALSE(ProjectSettings::get_singleton()->has_setting("not_existing_setting"));
}

TEST_CASE("[ProjectSettings] Set value should be returned when retrieved") {
	CHECK_FALSE(ProjectSettings::get_singleton()->has_setting("my_custom_setting"));

	Variant variant = ProjectSettings::get_singleton()->get_setting("my_custom_setting");
	CHECK_EQ(variant.get_type(), Variant::NIL);

	ProjectSettings::get_singleton()->set_setting("my_custom_setting", true);
	CHECK(ProjectSettings::get_singleton()->has_setting("my_custom_setting"));

	variant = ProjectSettings::get_singleton()->get_setting("my_custom_setting");
	CHECK_EQ(variant.get_type(), Variant::BOOL);

	bool value = variant;
	CHECK_EQ(true, value);

	CHECK(ProjectSettings::get_singleton()->has_setting("my_custom_setting"));
}

TEST_CASE("[ProjectSettings] localize_path") {
	String old_resource_path = TestProjectSettingsInternalsAccessor::resource_path();
	TestProjectSettingsInternalsAccessor::resource_path() = DirAccess::create(DirAccess::ACCESS_FILESYSTEM)->get_current_dir();
	String root_path = ProjectSettings::get_singleton()->get_resource_path();
#ifdef WINDOWS_ENABLED
	String root_path_win = ProjectSettings::get_singleton()->get_resource_path().replace_char('/', '\\');
#endif

	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("filename"), "res://filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("path/filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("path/something/../filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("path/./filename"), "res://path/filename");
#ifdef WINDOWS_ENABLED
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("path\\filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("path\\something\\..\\filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("path\\.\\filename"), "res://path/filename");
#endif

	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("../filename"), "../filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("../path/filename"), "../path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("..\\path\\filename"), "../path/filename");

	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("/testroot/filename"), "/testroot/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("/testroot/path/filename"), "/testroot/path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("/testroot/path/something/../filename"), "/testroot/path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("/testroot/path/./filename"), "/testroot/path/filename");
#ifdef WINDOWS_ENABLED
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:/testroot/filename"), "C:/testroot/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:/testroot/path/filename"), "C:/testroot/path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:/testroot/path/something/../filename"), "C:/testroot/path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:/testroot/path/./filename"), "C:/testroot/path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:\\testroot\\filename"), "C:/testroot/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:\\testroot\\path\\filename"), "C:/testroot/path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:\\testroot\\path\\something\\..\\filename"), "C:/testroot/path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path("C:\\testroot\\path\\.\\filename"), "C:/testroot/path/filename");
#endif

	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path + "/filename"), "res://filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path + "/path/filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path + "/path/something/../filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path + "/path/./filename"), "res://path/filename");
#ifdef WINDOWS_ENABLED
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path_win + "\\filename"), "res://filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path_win + "\\path\\filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path_win + "\\path\\something\\..\\filename"), "res://path/filename");
	CHECK_EQ(ProjectSettings::get_singleton()->localize_path(root_path_win + "\\path\\.\\filename"), "res://path/filename");
#endif

	TestProjectSettingsInternalsAccessor::resource_path() = old_resource_path;
}

TEST_CASE("[SceneTree][ProjectSettings] settings_changed signal") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));

	ProjectSettings::get_singleton()->set_setting("test_signal_setting", "test_value");
	MessageQueue::get_singleton()->flush();

	SIGNAL_CHECK("settings_changed", { {} });

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
}

TEST_CASE("[SceneTree][ProjectSettings] setting_changed signal for new setting") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));

	String setting_name = "test_new_setting";
	String new_value = "new_value";

	CHECK_FALSE(ProjectSettings::get_singleton()->has_setting(setting_name));

	SIGNAL_DISCARD("setting_changed");

	ProjectSettings::get_singleton()->set_setting(setting_name, new_value);
	MessageQueue::get_singleton()->flush();

	Array expected_args;
	expected_args.push_back(StringName(setting_name));
	expected_args.push_back(Variant());
	expected_args.push_back(new_value);
	Array signal_args;
	signal_args.push_back(expected_args);
	SIGNAL_CHECK("setting_changed", signal_args);

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));
}

TEST_CASE("[SceneTree][ProjectSettings] setting_changed signal with parameters") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));

	String setting_name = "test_old_value_signal";
	String old_value = "old_value";
	String new_value = "new_value";

	ProjectSettings::get_singleton()->set_setting(setting_name, old_value);
	MessageQueue::get_singleton()->flush();

	SIGNAL_DISCARD("setting_changed");

	ProjectSettings::get_singleton()->set_setting(setting_name, new_value);
	MessageQueue::get_singleton()->flush();

	Array expected_args;
	expected_args.push_back(StringName(setting_name));
	expected_args.push_back(old_value);
	expected_args.push_back(new_value);
	Array signal_args;
	signal_args.push_back(expected_args);
	SIGNAL_CHECK("setting_changed", signal_args);

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));
}

TEST_CASE("[SceneTree][ProjectSettings] setting_changed signal for setting removal") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));

	String setting_name = "test_removal_setting";
	String initial_value = "initial_value";

	ProjectSettings::get_singleton()->set_setting(setting_name, initial_value);
	MessageQueue::get_singleton()->flush();
	SIGNAL_DISCARD("setting_changed");

	ProjectSettings::get_singleton()->set_setting(setting_name, Variant());
	MessageQueue::get_singleton()->flush();

	Array expected_args;
	expected_args.push_back(StringName(setting_name));
	expected_args.push_back(initial_value);
	expected_args.push_back(Variant());
	Array signal_args;
	signal_args.push_back(expected_args);
	SIGNAL_CHECK("setting_changed", signal_args);

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));
}

TEST_CASE("[SceneTree][ProjectSettings] Both signals emitted together") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));

	String setting_name = "test_both_signals";
	String old_value = "old_both";
	String new_value = "new_both";

	ProjectSettings::get_singleton()->set_setting(setting_name, old_value);
	MessageQueue::get_singleton()->flush();
	SIGNAL_DISCARD("settings_changed");
	SIGNAL_DISCARD("setting_changed");

	ProjectSettings::get_singleton()->set_setting(setting_name, new_value);
	MessageQueue::get_singleton()->flush();

	SIGNAL_CHECK("settings_changed", { {} });

	Array expected_args;
	expected_args.push_back(StringName(setting_name));
	expected_args.push_back(old_value);
	expected_args.push_back(new_value);
	Array signal_args;
	signal_args.push_back(expected_args);
	SIGNAL_CHECK("setting_changed", signal_args);

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));
}

TEST_CASE("[SceneTree][ProjectSettings] No signals when setting same value") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));

	String setting_name = "test_same_value";
	String test_value = "same_value";

	ProjectSettings::get_singleton()->set_setting(setting_name, test_value);
	MessageQueue::get_singleton()->flush();
	SIGNAL_DISCARD("settings_changed");
	SIGNAL_DISCARD("setting_changed");

	// Set the same value again. This should not trigger any signals.
	ProjectSettings::get_singleton()->set_setting(setting_name, test_value);
	MessageQueue::get_singleton()->flush();

	SIGNAL_CHECK_FALSE("settings_changed");
	SIGNAL_CHECK_FALSE("setting_changed");

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));
}

TEST_CASE("[SceneTree][ProjectSettings] Multiple setting changes in same frame") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));

	SIGNAL_DISCARD("settings_changed");
	SIGNAL_DISCARD("setting_changed");

	// Change multiple settings in the same frame,
	ProjectSettings::get_singleton()->set_setting("setting_1", "value_1");
	ProjectSettings::get_singleton()->set_setting("setting_2", "value_2");

	MessageQueue::get_singleton()->flush();

	Array expected_args1;
	expected_args1.push_back(StringName("setting_1"));
	expected_args1.push_back(Variant());
	expected_args1.push_back("value_1");

	Array expected_args2;
	expected_args2.push_back(StringName("setting_2"));
	expected_args2.push_back(Variant());
	expected_args2.push_back("value_2");

	Array all_signal_args;
	all_signal_args.push_back(expected_args1);
	all_signal_args.push_back(expected_args2);

	// Should have 2 setting_changed signals.
	SIGNAL_CHECK("setting_changed", all_signal_args);
	// Should have 1 settings_changed signal.
	SIGNAL_CHECK("settings_changed", { {} });

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));
}

TEST_CASE("[SceneTree][ProjectSettings] Multiple changes to the same setting only fire for actual changes") {
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_WATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));

	SIGNAL_DISCARD("settings_changed");
	SIGNAL_DISCARD("setting_changed");

	// Change setting multiple times in the same frame.
	ProjectSettings::get_singleton()->set_setting("reused_setting", "value_1");
	ProjectSettings::get_singleton()->set_setting("reused_setting", "value_2");
	ProjectSettings::get_singleton()->set_setting("reused_setting", "value_2");
	ProjectSettings::get_singleton()->set_setting("reused_setting", "value_3");

	MessageQueue::get_singleton()->flush();

	Array expected_args1;
	expected_args1.push_back(StringName("reused_setting"));
	expected_args1.push_back(Variant());
	expected_args1.push_back("value_1");

	Array expected_args2;
	expected_args2.push_back(StringName("reused_setting"));
	expected_args2.push_back("value_1");
	expected_args2.push_back("value_2");

	Array expected_args3;
	expected_args3.push_back(StringName("reused_setting"));
	expected_args3.push_back("value_2");
	expected_args3.push_back("value_3");

	Array all_signal_args;
	all_signal_args.push_back(expected_args1);
	all_signal_args.push_back(expected_args2);
	all_signal_args.push_back(expected_args3);

	// Should have 3 setting_changed signals as the duplicate value_2 should be ignored.
	SIGNAL_CHECK("setting_changed", all_signal_args);
	// Should have 1 settings_changed signal.
	SIGNAL_CHECK("settings_changed", { {} });

	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("settings_changed"));
	SIGNAL_UNWATCH(ProjectSettings::get_singleton(), SNAME("setting_changed"));
}

} // namespace TestProjectSettings
