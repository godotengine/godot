/*************************************************************************/
/*  test_plugin.cpp                                                      */
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

#include "test_plugin.h"

#include "core/io/json.h"
#include "editor/editor_data.h"
#include "editor/editor_node.h"

DebugButton::DebugButton() {
	set_text(TTR("Test"));
	m_is_testing = false;
	add_user_signal(MethodInfo("display_results"));
}

ToolButton *find_play_button() {
	return Object::cast_to<ToolButton>(EditorNode::get_menu_hb()->get_child(4)->get_child(0));
}

void DebugButton::run() {
	const String customArgs = "--main-loop-type TestRunner";
	ProjectSettings *project_settings = ProjectSettings::get_singleton();
	const String args = project_settings->get("editor/main_run_args");
	ToolButton *play_button = find_play_button();
	project_settings->set("editor/main_run_args", customArgs);
	play_button->emit_signal("pressed");
	project_settings->set("editor/main_run_args", args);
	m_is_testing = true;

	REF ref;
	ScriptEditor::get_singleton()->get_debugger()->emit_signal("goto_script_line", ref, 2);
}

void DebugButton::enable() {
	this->set_disabled(false);
}

void DebugButton::disable() {
	this->set_disabled(true);
}

void DebugButton::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_READY: {
			connect("pressed", this, "run");
			EditorNode *editor_node = EditorNode::get_singleton();
			editor_node->connect("play_pressed", this, "disable");
			editor_node->connect("stop_pressed", this, "enable");
			editor_node->connect("stop_pressed", this, "_display_results");
			break;
		}
	}
}

void DebugButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("run"), &DebugButton::run);
	ClassDB::bind_method(D_METHOD("enable"), &DebugButton::enable);
	ClassDB::bind_method(D_METHOD("_display_results"), &DebugButton::_display_results);
	ClassDB::bind_method(D_METHOD("disable"), &DebugButton::disable);
}

void DebugButton::_display_results() {
	if (m_is_testing) {
		emit_signal("display_results");
	}
	m_is_testing = false;
}

TestPanel::TestPanel() {
	m_files = memnew(VBoxContainer);
	add_child(m_files);
	m_tests = memnew(VBoxContainer);
	add_child(m_tests);
	m_logs = memnew(VBoxContainer);
	add_child(m_logs);
}

void TestPanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_display_results"), &TestPanel::_display_results);
}

void TestPanel::_display_results() {
	m_results.clear();
	Error error;
	FileAccessRef file = FileAccess::open("res://Testing/results.json", FileAccess::READ, &error);
	if (error == Error::OK) {
		String text = file->get_as_utf8_string();
		Variant ret;
		String err_str;
		int err_line;
		error = JSON::parse(text, ret, err_str, err_line);
		if (error == Error::OK) {
			if (ret.get_type() == Variant::ARRAY) {
				Array logs = ret;
				for (int i = 0; i < logs.size(); i++) {
					if (logs[i].get_type() == Variant::DICTIONARY) {
						const Dictionary &data = logs[i];
						Ref<TestLog::LogMessage> message = TestLog::LogMessage::log((TestLog::LogLevel)(int)data["level"], data["time"], data["script_path"], data["test_func"], data["message"]);
						if (!m_results.has(message->script_path())) {
							m_results.set(message->script_path(), TestFuncLogMap());
							Label *script_path = new Label();
							script_path->set_text(message->script_path());
							m_logs->add_child(script_path);
						}
						TestFuncLogMap test_func_log_map = m_results.get(message->script_path());
						if (!test_func_log_map.has(message->test_func())) {
							test_func_log_map.set(message->test_func(), RefTestLog())->value().instance();
						}
						RefTestLog log = test_func_log_map.get(message->test_func());
						log->add_message(message);
					}
				}
			}
		}
	}
}

void TestPanel::_display_tests(String p_filename) {
}

void TestPanel::_display_logs(String p_filename, String p_method) {
}

TestPlugin::TestPlugin() {
	m_debug_button = memnew(DebugButton);
	add_control_to_container(CONTAINER_TOOLBAR, m_debug_button);
	EditorNode::get_menu_hb()->move_child(m_debug_button, 5);
	EditorNode *editor_node = EditorNode::get_singleton();
	TestPanel *test_panel = memnew(TestPanel);
	m_debug_button->connect("display_results", test_panel, "_display_results");
	m_test_panel = editor_node->add_bottom_panel_item(TTR("Test"), test_panel);
}

TestPlugin::~TestPlugin() {
}

void TestPlugin::_bind_methods() {
}
