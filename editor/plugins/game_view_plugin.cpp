/**************************************************************************/
/*  game_view_plugin.cpp                                                  */
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

#include "game_view_plugin.h"

#include "core/debugger/debugger_marshalls.h"
#include "editor/editor_main_screen.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "scene/gui/button.h"
#include "scene/gui/grid_container.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/panel.h"
#include "scene/gui/separator.h"

void GameViewDebugger::setup_session(int p_session_id) {
	Ref<EditorDebuggerSession> session = get_session(p_session_id);
	ERR_FAIL_COND(session.is_null());

	GameTab *tab = memnew(GameTab(session));
	session->add_session_tab(tab);

	tabs[p_session_id] = tab;
}

bool GameViewDebugger::capture(const String &p_message, const Array &p_data, int p_session) {
	ERR_FAIL_COND_V(!tabs.has(p_session), false);
	GameTab *tab = tabs[p_session];

	if (p_message == "scene:click_ctrl") {
		ERR_FAIL_COND_V(p_data.size() < 2, false);
		tab->on_click_ctrl(p_data[0], p_data[1]);
	}
	return true;
}

bool GameViewDebugger::has_capture(const String &p_capture) const {
	return p_capture == "scene";
}

void GameViewDebugger::set_state(const Dictionary &p_state) {
	for (const KeyValue<int, GameTab *> &E : tabs) {
		if (p_state.has(E.key)) {
			E.value->set_state(p_state[E.key]);
		}
	}
}

Dictionary GameViewDebugger::get_state() const {
	Dictionary d;
	for (const KeyValue<int, GameTab *> &E : tabs) {
		d[E.key] = E.value->get_state();
	}
	return d;
}

///////

void GameTab::_update_debugger_buttons() {
	bool empty = !session->is_active();

	suspend_button->set_disabled(empty);
	camera_override_button->set_disabled(empty);

	const Tree *editor_remote_tree = session->get_editor_remote_tree();
	live_edit_set_button->set_disabled(empty || !editor_remote_tree || !editor_remote_tree->get_selected());
	live_edit_clear_button->set_disabled(empty);

	PopupMenu *menu = camera_override_menu->get_popup();

	bool disable_camera_reset = empty || !camera_override_button->is_pressed() || !menu->is_item_checked(menu->get_item_index(CAMERA_MODE_INGAME));
	menu->set_item_disabled(CAMERA_RESET_2D, disable_camera_reset);
	menu->set_item_disabled(CAMERA_RESET_3D, disable_camera_reset);

	if (empty) {
		suspend_button->set_pressed(false);
		camera_override_button->set_pressed(false);
	}
	next_frame_button->set_disabled(!suspend_button->is_pressed());
}

void GameTab::_suspend_button_toggled(bool p_pressed) {
	_update_debugger_buttons();

	Array message;
	message.append(p_pressed);
	session->send_message("scene:suspend_changed", message);
}

void GameTab::_next_frame_button_pressed() {
	session->send_message("scene:next_frame", Array());
}

void GameTab::_node_type_pressed(int p_option) {
	node_type = (RuntimeNodeSelect::NodeType)p_option;
	for (int i = 0; i < RuntimeNodeSelect::NODE_TYPE_MAX; i++) {
		node_type_button[i]->set_pressed_no_signal(i == node_type);
	}

	_update_debugger_buttons();

	Array message;
	message.append(node_type);
	session->send_message("scene:runtime_node_select_set_type", message);
}

void GameTab::_select_mode_pressed(int p_option) {
	select_mode = (RuntimeNodeSelect::SelectMode)p_option;
	for (int i = 0; i < RuntimeNodeSelect::SELECT_MODE_MAX; i++) {
		select_mode_button[i]->set_pressed_no_signal(i == select_mode);
	}

	Array message;
	message.append(select_mode);
	session->send_message("scene:runtime_node_select_set_mode", message);
}

void GameTab::_hide_selection_toggled(bool p_pressed) {
	hide_selection->set_button_icon(get_editor_theme_icon(p_pressed ? SNAME("GuiVisibilityHidden") : SNAME("GuiVisibilityVisible")));
	selection_visible = !p_pressed;

	Array message;
	message.append(selection_visible);
	session->send_message("scene:runtime_node_select_set_visible", message);
}

void GameTab::_camera_override_button_toggled(bool p_pressed) {
	_update_debugger_buttons();
	session->set_camera_override(p_pressed ? camera_override_mode : ScriptEditorDebugger::OVERRIDE_NONE);
}

void GameTab::_camera_override_menu_id_pressed(int p_id) {
	PopupMenu *menu = camera_override_menu->get_popup();
	if (p_id != CAMERA_RESET_2D && p_id != CAMERA_RESET_3D) {
		for (int i = 0; i < menu->get_item_count(); i++) {
			menu->set_item_checked(i, false);
		}
	}

	switch (p_id) {
		case CAMERA_RESET_2D: {
			session->send_message("scene:runtime_node_select_reset_camera_2d", Array());
		} break;

		case CAMERA_RESET_3D: {
			session->send_message("scene:runtime_node_select_reset_camera_3d", Array());
		} break;

		case CAMERA_MODE_INGAME: {
			camera_override_mode = ScriptEditorDebugger::OVERRIDE_INGAME;
			if (session->get_camera_override() != ScriptEditorDebugger::OVERRIDE_NONE) {
				session->set_camera_override(camera_override_mode);
			}
			menu->set_item_checked(menu->get_item_index(p_id), true);
			_update_debugger_buttons();
		} break;

		case CAMERA_MODE_EDITORS: {
			camera_override_mode = ScriptEditorDebugger::OVERRIDE_EDITORS;
			if (session->get_camera_override() != ScriptEditorDebugger::OVERRIDE_NONE) {
				session->set_camera_override(camera_override_mode);
			}
			menu->set_item_checked(menu->get_item_index(p_id), true);
			_update_debugger_buttons();
		} break;
	}
}

void GameTab::_live_edit_set_button_pressed() {
	if (!session->is_active()) {
		return;
	}
	const Tree *editor_remote_tree = session->get_editor_remote_tree();
	if (!editor_remote_tree) {
		return;
	}

	TreeItem *ti = editor_remote_tree->get_selected();
	if (!ti) {
		return;
	}

	String path;

	while (ti) {
		String lp = ti->get_text(0);
		path = "/" + lp + path;
		ti = ti->get_parent();
	}

	NodePath np = path;

	EditorNode::get_editor_data().set_edited_scene_live_edit_root(np);

	session->update_live_edit_root();
	live_edit_root->set_text(EditorNode::get_editor_data().get_edited_scene_live_edit_root());
}

void GameTab::_live_edit_clear_button_pressed() {
	NodePath np = NodePath("/root");
	EditorNode::get_editor_data().set_edited_scene_live_edit_root(np);

	session->update_live_edit_root();
	live_edit_root->set_text(EditorNode::get_editor_data().get_edited_scene_live_edit_root());
}

void GameTab::_on_start() {
	{
		Array params;
		Dictionary settings;
		settings["editors/panning/2d_editor_panning_scheme"] = EDITOR_GET("editors/panning/2d_editor_panning_scheme");
		settings["editors/panning/simple_panning"] = EDITOR_GET("editors/panning/simple_panning");
		settings["editors/panning/warped_mouse_panning"] = EDITOR_GET("editors/panning/warped_mouse_panning");
		settings["editors/panning/2d_editor_pan_speed"] = EDITOR_GET("editors/panning/2d_editor_pan_speed");
		settings["canvas_item_editor/pan_view"] = DebuggerMarshalls::serialize_key_shortcut(ED_GET_SHORTCUT("canvas_item_editor/pan_view"));
		params.append(settings);
		session->send_message("scene:runtime_node_select_setup", params);
	}
	{
		Array params;
		params.append(node_type);
		session->send_message("scene:runtime_node_select_set_type", params);
	}
	{
		Array params;
		params.append(selection_visible);
		session->send_message("scene:runtime_node_select_set_visible", params);
	}
	{
		Array params;
		params.append(select_mode);
		session->send_message("scene:runtime_node_select_set_mode", params);
	}

	_update_debugger_buttons();

	clicked_ctrl->set_text(String());
	clicked_ctrl_type->set_text(String());
}

void GameTab::_on_stop() {
	_update_debugger_buttons();
}

void GameTab::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_THEME_CHANGED: {
			suspend_button->set_button_icon(get_editor_theme_icon(SNAME("Pause")));
			next_frame_button->set_button_icon(get_editor_theme_icon(SNAME("NextFrame")));

			node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_button_icon(get_editor_theme_icon(SNAME("InputEventJoypadMotion")));
			node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_button_icon(get_editor_theme_icon(SNAME("2DNodes")));
#ifndef _3D_DISABLED
			node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_button_icon(get_editor_theme_icon(SNAME("Node3D")));
#endif // _3D_DISABLED

			select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_button_icon(get_editor_theme_icon(SNAME("ToolSelect")));
			select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_button_icon(get_editor_theme_icon(SNAME("ListSelect")));

			hide_selection->set_button_icon(get_editor_theme_icon(hide_selection->is_pressed() ? SNAME("GuiVisibilityHidden") : SNAME("GuiVisibilityVisible")));

			camera_override_button->set_button_icon(get_editor_theme_icon(SNAME("Camera")));
			camera_override_menu->set_button_icon(get_editor_theme_icon(SNAME("GuiTabMenuHl")));
		} break;
	}
}

void GameTab::set_state(const Dictionary &p_state) {
	if (p_state.has("hide_selection")) {
		hide_selection->set_pressed(p_state["hide_selection"]);
		_hide_selection_toggled(hide_selection->is_pressed());
	}
	if (p_state.has("select_mode")) {
		_select_mode_pressed(p_state["select_mode"]);
	}
	if (p_state.has("camera_override_mode")) {
		_camera_override_menu_id_pressed(p_state["camera_override_mode"]);
	}
}

Dictionary GameTab::get_state() const {
	Dictionary d;
	d["hide_selection"] = hide_selection->is_pressed();

	for (int i = 0; i < RuntimeNodeSelect::SELECT_MODE_MAX; i++) {
		if (select_mode_button[i]->is_pressed()) {
			d["select_mode"] = i;
			break;
		}
	}

	PopupMenu *menu = camera_override_menu->get_popup();
	for (int i = CAMERA_MODE_INGAME; i < CAMERA_MODE_EDITORS + 1; i++) {
		if (menu->is_item_checked(menu->get_item_index(i))) {
			d["camera_override_mode"] = i;
			break;
		}
	}

	return d;
}

void GameTab::on_click_ctrl(const String &p_ctrl, const String &p_ctrl_type) {
	clicked_ctrl->set_text(p_ctrl);
	clicked_ctrl_type->set_text(p_ctrl_type);
}

GameTab::GameTab(Ref<EditorDebuggerSession> p_session) {
	set_name(TTR("Game"));

	session = p_session;

	session->connect("started", callable_mp(this, &GameTab::_on_start));
	session->connect("stopped", callable_mp(this, &GameTab::_on_stop));

	HBoxContainer *toolbar = memnew(HBoxContainer);
	add_child(toolbar);

	suspend_button = memnew(Button);
	suspend_button->set_tooltip_text(TTR("Suspend"));
	suspend_button->set_toggle_mode(true);
	suspend_button->set_theme_type_variation(SceneStringName(FlatButton));
	suspend_button->connect(SceneStringName(toggled), callable_mp(this, &GameTab::_suspend_button_toggled));
	toolbar->add_child(suspend_button);

	next_frame_button = memnew(Button);
	next_frame_button->set_tooltip_text(TTR("Next Frame"));
	next_frame_button->set_theme_type_variation("FlatButton");
	next_frame_button->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_next_frame_button_pressed));
	toolbar->add_child(next_frame_button);

	toolbar->add_child(memnew(VSeparator));
	toolbar->add_child(memnew(Label(TTR("Picking:"))));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE] = memnew(Button(TTRC("Off")));
	toolbar->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_pressed(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_NONE));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_NONE]->set_tooltip_text(TTR("Allow game input."));

	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D] = memnew(Button(TTRC("2D")));
	toolbar->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_2D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_2D]->set_tooltip_text(TTR("Disable game input and allow to select Node2Ds, Controls, and manipulate the 2D camera."));

#ifndef _3D_DISABLED
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D] = memnew(Button(TTRC("3D")));
	toolbar->add_child(node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_toggle_mode(true);
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_theme_type_variation(SceneStringName(FlatButton));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_node_type_pressed).bind(RuntimeNodeSelect::NODE_TYPE_3D));
	node_type_button[RuntimeNodeSelect::NODE_TYPE_3D]->set_tooltip_text(TTR("Disable game input and allow to select Node3Ds and manipulate the 3D camera."));
#endif // _3D_DISABLED

	toolbar->add_child(memnew(VSeparator));

	hide_selection = memnew(Button);
	toolbar->add_child(hide_selection);
	hide_selection->set_toggle_mode(true);
	hide_selection->set_theme_type_variation(SceneStringName(FlatButton));
	hide_selection->connect(SceneStringName(toggled), callable_mp(this, &GameTab::_hide_selection_toggled));
	hide_selection->set_tooltip_text(TTR("Toggle Selection Visibility"));

	toolbar->add_child(memnew(VSeparator));

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE] = memnew(Button);
	toolbar->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_pressed(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_theme_type_variation(SceneStringName(FlatButton));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_SINGLE));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut(ED_SHORTCUT("spatial_editor/tool_select", TTR("Select Mode"), Key::Q));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_shortcut_context(this);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_SINGLE]->set_tooltip_text(keycode_get_string((Key)KeyModifierMask::CMD_OR_CTRL) + TTR("Alt+RMB: Show list of all nodes at position clicked."));

	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST] = memnew(Button);
	toolbar->add_child(select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_toggle_mode(true);
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_theme_type_variation(SceneStringName(FlatButton));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_select_mode_pressed).bind(RuntimeNodeSelect::SELECT_MODE_LIST));
	select_mode_button[RuntimeNodeSelect::SELECT_MODE_LIST]->set_tooltip_text(TTR("Show list of selectable nodes at position clicked."));

	toolbar->add_child(memnew(VSeparator));

	camera_override_button = memnew(Button);
	toolbar->add_child(camera_override_button);
	camera_override_button->set_toggle_mode(true);
	camera_override_button->set_theme_type_variation(SceneStringName(FlatButton));
	camera_override_button->connect(SceneStringName(toggled), callable_mp(this, &GameTab::_camera_override_button_toggled));
	camera_override_button->set_tooltip_text(TTR("Override the in-game camera."));

	camera_override_menu = memnew(MenuButton);
	toolbar->add_child(camera_override_menu);
	camera_override_menu->set_flat(false);
	camera_override_menu->set_theme_type_variation("FlatMenuButton");
	camera_override_menu->set_h_size_flags(SIZE_SHRINK_END);
	camera_override_menu->set_tooltip_text(TTR("Camera Override Options"));

	PopupMenu *menu = camera_override_menu->get_popup();
	menu->connect(SceneStringName(id_pressed), callable_mp(this, &GameTab::_camera_override_menu_id_pressed));
	menu->add_item(TTR("Reset 2D Camera"), CAMERA_RESET_2D);
	menu->add_item(TTR("Reset 3D Camera"), CAMERA_RESET_3D);
	menu->add_separator();
	menu->add_radio_check_item(TTR("Manipulate In-Game"), CAMERA_MODE_INGAME);
	menu->set_item_checked(menu->get_item_index(CAMERA_MODE_INGAME), true);
	menu->add_radio_check_item(TTR("Manipulate From Editors"), CAMERA_MODE_EDITORS);

	add_child(memnew(HSeparator));

	GridContainer *grid = memnew(GridContainer);
	grid->set_columns(2);
	add_child(grid);

	grid->add_child(memnew(Label(TTRC("Clicked Control:"))));
	clicked_ctrl = memnew(LineEdit);
	clicked_ctrl->set_editable(false);
	clicked_ctrl->set_h_size_flags(SIZE_EXPAND_FILL);
	grid->add_child(clicked_ctrl);

	grid->add_child(memnew(Label(TTRC("Clicked Control Type:"))));
	clicked_ctrl_type = memnew(LineEdit);
	clicked_ctrl_type->set_editable(false);
	grid->add_child(clicked_ctrl_type);

	grid->add_child(memnew(Label(TTRC("Live Edit Root:"))));

	HBoxContainer *lehb = memnew(HBoxContainer);
	grid->add_child(lehb);

	live_edit_root = memnew(LineEdit);
	live_edit_root->set_editable(false);
	live_edit_root->set_h_size_flags(SIZE_EXPAND_FILL);
	lehb->add_child(live_edit_root);

	live_edit_set_button = memnew(Button(TTR("Set From Tree")));
	lehb->add_child(live_edit_set_button);
	live_edit_set_button->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_live_edit_set_button_pressed));

	live_edit_clear_button = memnew(Button(TTR("Clear")));
	lehb->add_child(live_edit_clear_button);
	live_edit_clear_button->connect(SceneStringName(pressed), callable_mp(this, &GameTab::_live_edit_clear_button_pressed));

	_update_debugger_buttons();
}

///////

GameView::GameView(Ref<GameViewDebugger> p_debugger) {
	panel = memnew(Panel);
	add_child(panel);
	panel->set_theme_type_variation("GamePanel");
	panel->set_v_size_flags(SIZE_EXPAND_FILL);
}

///////

void GameViewPlugin::make_visible(bool p_visible) {
	game_view->set_visible(p_visible);
}

void GameViewPlugin::set_state(const Dictionary &p_state) {
	debugger->set_state(p_state);
}

Dictionary GameViewPlugin::get_state() const {
	return debugger->get_state();
}

void GameViewPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			add_debugger_plugin(debugger);
		} break;
		case NOTIFICATION_EXIT_TREE: {
			remove_debugger_plugin(debugger);
		} break;
	}
}

GameViewPlugin::GameViewPlugin() {
	debugger.instantiate();

	game_view = memnew(GameView(debugger));
	game_view->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	EditorNode::get_singleton()->get_editor_main_screen()->get_control()->add_child(game_view);
	game_view->hide();
}

GameViewPlugin::~GameViewPlugin() {
}
