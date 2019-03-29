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
	add_user_signal(MethodInfo("display_test_panel"));
	set_custom_minimum_size(Size2(50, 250));

	VBoxContainer *files = memnew(VBoxContainer);
	files->set_name("test_files");
	m_file_filter = memnew(LineEdit);
	m_file_filter->connect("text_changed", this, "_filter_results");
	files->add_child(m_file_filter);
	m_files = memnew(ItemList);
	m_files->set_name("list");
	files->add_child(m_files);
	add_child(files);
	files->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	m_files->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	m_files->connect("item_selected", this, "_display_tests");

	VBoxContainer *tests = memnew(VBoxContainer);
	tests->set_name("test_tests");
	m_tests_filter = memnew(LineEdit);
	m_tests_filter->connect("text_changed", this, "_filter_tests");
	tests->add_child(m_tests_filter);
	m_tests = memnew(ItemList);
	m_tests->set_name("list");
	tests->add_child(m_tests);
	add_child(tests);
	tests->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	m_tests->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	m_tests->connect("item_selected", this, "_display_messages");

	VBoxContainer *messages = memnew(VBoxContainer);
	messages->set_name("test_messages");
	m_messages = memnew(ItemList);
	m_messages->set_name("list");
	messages->add_child(m_messages);
	add_child(messages);
	messages->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	m_messages->set_v_size_flags(Control::SIZE_EXPAND_FILL);
}

void TestPanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_display_results"), &TestPanel::_display_results);
	ClassDB::bind_method(D_METHOD("_filter_results"), &TestPanel::_filter_results);
	ClassDB::bind_method(D_METHOD("_display_tests"), &TestPanel::_display_tests);
	ClassDB::bind_method(D_METHOD("_filter_tests"), &TestPanel::_filter_tests);
	ClassDB::bind_method(D_METHOD("_display_messages"), &TestPanel::_display_messages);
}

void TestPanel::_display_results() {
	m_files->clear();
	m_tests->clear();
	m_messages->clear();
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
						const String &script_path = message->script_path();
						if (!m_results.has(script_path)) {
							m_results.set(script_path, TestFuncLogMap());
						}
						TestFuncLogMap* test_func_log_map = m_results.getptr(script_path);
						const String &test_func = message->test_func();
						if (!test_func_log_map->has(test_func)) {
							test_func_log_map->set(test_func, RefTestLog())->value().instance();
						}
						test_func_log_map->get(test_func)->add_message(message);
					}
				}
			}
		}
	}
	if (!m_results.empty()) {
		_filter_results("");
		emit_signal("display_test_panel");
	}
}

String _wildcard(const LineEdit *edit) {
	return "*" + edit->get_text() + "*";
}


TestLog::LogLevel max_level(const TestPanel::RefTestLog &map) {
	TestLog::LogLevel max = TestLog::LogLevel::TRACE;
	List<const TestLog::LogMessage *> messages;
	map->get_messages(&messages);
	for (int i = 0; i < messages.size(); i++) {
		if (messages[i]->level() > max) {
			max = messages[i]->level();
		}
	}
	return max;
}
TestLog::LogLevel max_level(const TestPanel::TestFuncLogMap &map) {
	TestLog::LogLevel max = TestLog::LogLevel::TRACE;
	const String *key = map.next(NULL);
	while (key) {
		TestLog::LogLevel level = max_level(map.get(*key));
		if (level > max) {
			max = level;
		}
		key = map.next(key);
	}
	return max;
}

void TestPanel::_filter_results(String p_ignore) {
	m_files->clear();
	List<String> keys;
	m_results.get_key_list(&keys);
	while (!keys.empty()) {
		const String &text = keys.front()->get();
		if (text.match(_wildcard(m_file_filter))) {
			m_files->add_item(text);
			m_files->set_item_custom_fg_color(m_files->get_item_count() - 1, TestLog::LogMessage::level_to_color(max_level(m_results.get(text))));
		}
		keys.pop_front();
	}
}

void TestPanel::_filter_tests(String p_ignore) {
	m_tests->set_block_signals(true);
	Vector<int> selection = m_files->get_selected_items();
	m_tests->clear();
	if (selection.size() > 0) {
		const String &filename = m_files->get_item_text(selection[0]);
		const TestFuncLogMap &map = m_results.get(filename);
		List<String> keys;
		map.get_key_list(&keys);
		while (!keys.empty()) {
			const String &text = keys.front()->get();
			if (text.match(_wildcard(m_tests_filter))) {
				m_tests->add_item(text);
				m_tests->set_item_custom_fg_color(m_tests->get_item_count() - 1, TestLog::LogMessage::level_to_color(max_level(map.get(text))));
			}
			keys.pop_front();
		}
	}
	m_tests->set_block_signals(false);
}

void TestPanel::_display_tests(int p_ignore) {
	m_tests->clear();
	m_messages->clear();
	Vector<int> selection = m_files->get_selected_items();
	if (selection.size() > 0) {
		const String &filename = m_files->get_item_text(selection[0]);
		const TestFuncLogMap &map = m_results.get(filename);
		List<String> keys;
		map.get_key_list(&keys);
		while (!keys.empty()) {
			const String &text = keys.front()->get();
			if (text.match(_wildcard(m_tests_filter))) {
				m_tests->add_item(text);
				m_tests->set_item_custom_fg_color(m_tests->get_item_count() - 1, TestLog::LogMessage::level_to_color(max_level(map.get(text))));
			}
			keys.pop_front();
		}

		REF ref = ResourceLoader::load(filename);
		ScriptEditor::get_singleton()->get_debugger()->emit_signal("goto_script_line", ref, 0);
	}
}

void TestPanel::_display_messages(int p_ignore) {
	m_messages->clear();
	Vector<int> file = m_files->get_selected_items();
	Vector<int> test = m_tests->get_selected_items();
	if (file.size() > 0 && test.size() > 0) {
		const String &filename = m_files->get_item_text(file[0]);
		const String &testname = m_tests->get_item_text(test[0]);
		const RefTestLog &log = m_results.get(filename).get(testname);
		List<const TestLog::LogMessage *> messages;
		log->get_messages(&messages);
		while (!messages.empty()) {
			const TestLog::LogMessage *message = messages.front()->get();
			const String &text = message->message();
			m_messages->add_item(text);
			m_messages->set_item_custom_fg_color(m_messages->get_item_count() - 1, TestLog::LogMessage::level_to_color(message->level()));
			messages.pop_front();
		}
		Ref<Script> ref = ResourceLoader::load(filename);
		int line = ref->get_member_line(testname);
		ScriptEditor::get_singleton()->get_debugger()->emit_signal("goto_script_line", ref, line);
	}
}

TestPlugin::TestPlugin() {
	m_debug_button = memnew(DebugButton);
	add_control_to_container(CONTAINER_TOOLBAR, m_debug_button);
	EditorNode::get_menu_hb()->move_child(m_debug_button, 5);
	EditorNode *editor_node = EditorNode::get_singleton();
	m_test_panel = memnew(TestPanel);
	m_test_panel->connect("display_test_panel", this, "_display_test_panel");
	m_debug_button->connect("display_results", m_test_panel, "_display_results");
	m_test_panel_button = editor_node->add_bottom_panel_item(TTR("Test"), m_test_panel);
	m_test_panel_button->set_visible(false);
}

TestPlugin::~TestPlugin() {
}

void TestPlugin::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_display_test_panel"), &TestPlugin::_display_test_panel);
}

void TestPlugin::_display_test_panel() {
	m_test_panel_button->set_visible(true);
	m_test_panel->set_visible(true);
}
