/**************************************************************************/
/*  game_editor_plugin.cpp                                                */
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

#include "game_editor_plugin.h"

#include "editor/debugger/editor_debugger_node.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/button.h"
#include "scene/gui/panel.h"
#include "scene/gui/separator.h"

void GameEditorDebugger::_session_started(Ref<EditorDebuggerSession> p_session) {
	p_session->send_message("scene:runtime_node_select_setup", Array());

	Array type;
	type.append(node_type);
	p_session->send_message("scene:runtime_node_select_set_type", type);
	Array mode;
	mode.append(select_mode);
	p_session->send_message("scene:runtime_node_select_set_mode", mode);

	emit_signal(SNAME("session_started"));
}

void GameEditorDebugger::_session_stopped() {
	emit_signal(SNAME("session_stopped"));
}

void GameEditorDebugger::set_suspend(bool p_enabled) {
	Array message;
	message.append(p_enabled);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:suspend_changed", message);
		}
	}
}

void GameEditorDebugger::next_frame() {
	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:next_frame", Array());
		}
	}
}

void GameEditorDebugger::set_node_type(int p_type) {
	node_type = p_type;

	Array message;
	message.append(p_type);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_set_type", message);
		}
	}
}

void GameEditorDebugger::set_select_mode(int p_mode) {
	select_mode = p_mode;

	Array message;
	message.append(p_mode);

	for (Ref<EditorDebuggerSession> &I : sessions) {
		if (I->is_active()) {
			I->send_message("scene:runtime_node_select_set_mode", message);
		}
	}
}

void GameEditorDebugger::setup_session(int p_session_id) {
	Ref<EditorDebuggerSession> session = get_session(p_session_id);
	ERR_FAIL_COND(session.is_null());

	sessions.append(session);

	session->connect("started", callable_mp(this, &GameEditorDebugger::_session_started).bind(session));
	session->connect("stopped", callable_mp(this, &GameEditorDebugger::_session_stopped));
}

void GameEditorDebugger::_bind_methods() {
	ADD_SIGNAL(MethodInfo("session_started"));
	ADD_SIGNAL(MethodInfo("session_stopped"));
}

///////

void GameEditor::_sessions_changed() {
	// The debugger session's `session_started/stopped` signal can be unreliable, so count it manually.
	active_sessions = 0;
	Array sessions = debugger->get_sessions();
	for (int i = 0; i < sessions.size(); i++) {
		if (Object::cast_to<EditorDebuggerSession>(sessions[i])->is_active()) {
			active_sessions++;
		}
	}

	_update_debugger_buttons();
}

void GameEditor::_update_debugger_buttons() {
	bool empty = active_sessions == 0;
	suspend_button->set_disabled(empty);
	if (empty) {
		suspend_button->set_pressed(false);
	}

	next_frame_button->set_disabled(!suspend_button->is_pressed());
}

void GameEditor::_suspend_button_toggled(bool p_pressed) {
	if (p_pressed) {
		_update_debugger_buttons();
	}

	debugger->set_suspend(p_pressed);
}

void GameEditor::_node_type_pressed(int p_option) {
	RuntimeNodeSelect::NodeType type = (RuntimeNodeSelect::NodeType)p_option;
	for (int i = 0; i < RuntimeNodeSelect::NODE_TYPE_MAX; i++) {
		node_type_button[i]->set_pressed(i == type);
	}

	debugger->set_node_type(type);
}

void GameEditor::_select_mode_pressed(int p_option) {
	RuntimeNodeSelect::SelectMode mode = (RuntimeNodeSelect::SelectMode)p_option;
	for (int i = 0; i < RuntimeNodeSelect::SELECT_MODE_MAX; i++) {
		select_mode_button[i]->set_pressed(i == mode);
	}

	debugger->set_select_mode(mode);
}

void GameEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			suspend_button->set_icon(get_editor_theme_icon(SNAME("Pause")));
			next_frame_button->set_icon(get_editor_theme_icon(SNAME("NextFrame")));

			node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_icon(get_editor_theme_icon(SNAME("2DNodes")));
			node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_icon(get_editor_theme_icon(SNAME("Node3D")));

			select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_icon(get_editor_theme_icon(SNAME("ToolSelect")));
			select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_icon(get_editor_theme_icon(SNAME("ListSelect")));

			panel->set_theme_type_variation("GamePanel");
		} break;
	}
}

GameEditor::GameEditor(Ref<GameEditorDebugger> p_debugger) {
	debugger = p_debugger;

	// Add some margin to the sides for better aesthetics.
	// This prevents the first button's hover/pressed effect from "touching" the panel's border,
	// which looks ugly.
	MarginContainer *toolbar_margin = memnew(MarginContainer);
	toolbar_margin->add_theme_constant_override("margin_left", 4 * EDSCALE);
	toolbar_margin->add_theme_constant_override("margin_right", 4 * EDSCALE);
	add_child(toolbar_margin);

	HBoxContainer *main_menu_hbox = memnew(HBoxContainer);
	toolbar_margin->add_child(main_menu_hbox);

	suspend_button = memnew(Button);
	main_menu_hbox->add_child(suspend_button);
	suspend_button->set_toggle_mode(true);
	suspend_button->set_theme_type_variation("FlatButton");
	suspend_button->connect(SceneStringName(toggled), callable_mp(this, &GameEditor::_suspend_button_toggled));
	suspend_button->set_tooltip_text(TTR("Suspend"));

	next_frame_button = memnew(Button);
	main_menu_hbox->add_child(next_frame_button);
	next_frame_button->set_theme_type_variation("FlatButton");
	next_frame_button->connect(SceneStringName(pressed), callable_mp(*debugger, &GameEditorDebugger::next_frame));
	next_frame_button->set_tooltip_text(TTR("Next Frame"));

	_update_debugger_buttons();

	main_menu_hbox->add_child(memnew(VSeparator));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D] = memnew(Button);
	main_menu_hbox->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_text(TTR("2D"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_pressed(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_theme_type_variation("FlatButton");
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->connect(SceneStringName(pressed), callable_mp(this, &GameEditor::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_2D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_tooltip_text(TTR("Select Node2Ds and Controls"));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D] = memnew(Button);
	main_menu_hbox->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_text(TTR("3D"));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_theme_type_variation("FlatButton");
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->connect(SceneStringName(pressed), callable_mp(this, &GameEditor::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_3D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_tooltip_text(TTR("Select Node3Ds"));

	main_menu_hbox->add_child(memnew(VSeparator));

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE] = memnew(Button);
	main_menu_hbox->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_pressed(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_theme_type_variation("FlatButton");
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->connect(SceneStringName(pressed), callable_mp(this, &GameEditor::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_SINGLE));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_select", TTR("Select Mode"), Key::Q));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut_context(this);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_tooltip_text(keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL) + TTR("Alt+RMB: Show list of all nodes at position clicked."));

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST] = memnew(Button);
	main_menu_hbox->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_theme_type_variation("FlatButton");
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->connect(SceneStringName(pressed), callable_mp(this, &GameEditor::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_LIST));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_tooltip_text(TTR("Show list of selectable nodes at position clicked."));

	panel = memnew(Panel);
	add_child(panel);
	panel->set_v_size_flags(SIZE_EXPAND_FILL);

	p_debugger->connect("session_started", callable_mp(this, &GameEditor::_sessions_changed));
	p_debugger->connect("session_stopped", callable_mp(this, &GameEditor::_sessions_changed));
}

///////

void GameEditorPlugin::make_visible(bool p_visible) {
	game_editor->set_visible(p_visible);
}

void GameEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			add_debugger_plugin(debugger);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			remove_debugger_plugin(debugger);
		} break;
	}
}

GameEditorPlugin::GameEditorPlugin() {
	debugger.instantiate();

	game_editor = memnew(GameEditor(debugger));
	game_editor->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(game_editor);
	game_editor->hide();
}

GameEditorPlugin::~GameEditorPlugin() {
}
