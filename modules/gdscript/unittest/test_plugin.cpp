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

#include "editor/editor_data.h"
#include "editor/editor_node.h"

DebugButton::DebugButton() {
	set_text(TTR("Test"));
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
			break;
		}
	}
}

void DebugButton::_bind_methods() {
	ClassDB::bind_method(D_METHOD("run"), &DebugButton::run);
	ClassDB::bind_method(D_METHOD("enable"), &DebugButton::enable);
	ClassDB::bind_method(D_METHOD("disable"), &DebugButton::disable);
}

TestPanel::TestPanel() {
}

void TestPanel::_bind_methods() {
}

TestPlugin::TestPlugin() {
	m_debug_button = memnew(DebugButton);
	add_control_to_container(CONTAINER_TOOLBAR, m_debug_button);
	EditorNode::get_menu_hb()->add_child(m_debug_button);
	EditorNode *editor_node = EditorNode::get_singleton();
	m_test_panel = editor_node->add_bottom_panel_item(TTR("Test"), memnew(TestPanel));
}

TestPlugin::~TestPlugin() {
}

void TestPlugin::_bind_methods() {
}
