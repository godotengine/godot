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
#include "test_config.h"

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
	const String customArgs = "-s TestRunner";
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

	m_tree = memnew(Tree);
	m_tree->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	m_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	m_tree->connect("item_selected", this, "_item_selected");
	add_child(m_tree);
}

void TestPanel::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_display_results"), &TestPanel::_display_results);
	ClassDB::bind_method(D_METHOD("_item_selected"), &TestPanel::_item_selected);
}

void TestPanel::_display_results() {
	m_tree->clear();
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
						TestFuncLogMap *test_func_log_map = m_results.getptr(script_path);
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
		TreeItem *root_item = m_tree->create_item();
		root_item->set_text(0, "Results");
		const String *filename = m_results.next(NULL);
		while (filename != NULL) {
			TreeItem *filename_item = m_tree->create_item(root_item);
			filename_item->set_text(0, *filename);
			TestLog::LogLevel filename_level = TestLog::LogLevel::TRACE;
			const TestFuncLogMap &func_log_map = m_results.get(*filename);
			const String *test = func_log_map.next(NULL);
			while (test != NULL) {
				TreeItem *test_item = m_tree->create_item(filename_item);
				test_item->set_text(0, *test);
				test_item->set_collapsed(true);
				TestLog::LogLevel test_level = TestLog::LogLevel::TRACE;
				const RefTestLog &test_log = func_log_map.get(*test);
				List<const TestLog::LogMessage *> messages;
				test_log->get_messages(&messages);
				while (!messages.empty()) {
					TreeItem *message_item = m_tree->create_item(test_item);
					const TestLog::LogMessage *message = messages.front()->get();
					message_item->set_text(0, message->message());
					message_item->set_custom_color(0, TestLog::LogMessage::level_to_color(message->level()));
					if (message->level() > test_level) {
						if (message->level() >= TestConfig::get_singleton()->log_fail_greater_equal()) {
							test_item->set_collapsed(false);
						}
						test_level = message->level();
						if (message->level() > filename_level) {
							filename_level = message->level();
						}
					}
					messages.pop_front();
				}
				test_item->set_custom_color(0, TestLog::LogMessage::level_to_color(test_level));
				test = func_log_map.next(test);
			}
			filename_item->set_custom_color(0, TestLog::LogMessage::level_to_color(filename_level));
			filename = m_results.next(filename);
		}

		emit_signal("display_test_panel");
	}
}

void TestPanel::_item_selected() {
	TreeItem *selected = m_tree->get_selected();
	if (selected != NULL) {
		TreeItem *parent = selected->get_parent();
		if (parent != NULL) {
			String filename;
			String testname;
			TreeItem *grandparent = parent->get_parent();
			if (grandparent != NULL) {
				TreeItem *greatgrandparent = grandparent->get_parent();
				if (greatgrandparent != NULL) {
					filename = grandparent->get_text(0);
					testname = parent->get_text(0);
				} else {
					filename = parent->get_text(0);
					testname = selected->get_text(0);
				}
			} else {
				filename = selected->get_text(0);
			}

			Ref<Script> ref = ResourceLoader::load(filename);
			int line = ref->get_member_line(testname);
			ScriptEditor::get_singleton()->get_debugger()->emit_signal("goto_script_line", ref, line);
		}
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
