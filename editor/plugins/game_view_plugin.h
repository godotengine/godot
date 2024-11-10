/**************************************************************************/
/*  game_view_plugin.h                                                    */
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

#ifndef GAME_VIEW_PLUGIN_H
#define GAME_VIEW_PLUGIN_H

#include "editor/plugins/editor_debugger_plugin.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/gui/box_container.h"

class GameTab : public VBoxContainer {
	GDCLASS(GameTab, VBoxContainer);

	enum {
		CAMERA_RESET_2D,
		CAMERA_RESET_3D,
		CAMERA_MODE_INGAME,
		CAMERA_MODE_EDITORS,
	};

	Ref<EditorDebuggerSession> session;

	int node_type = RuntimeNodeSelect::NODE_TYPE_NONE;
	bool selection_visible = true;
	int select_mode = RuntimeNodeSelect::SELECT_MODE_SINGLE;
	ScriptEditorDebugger::CameraOverride camera_override_mode = ScriptEditorDebugger::OVERRIDE_INGAME;

	// Toolbar.
	Button *suspend_button = nullptr;
	Button *next_frame_button = nullptr;
	Button *node_type_button[RuntimeNodeSelect::NODE_TYPE_MAX];
	Button *select_mode_button[RuntimeNodeSelect::SELECT_MODE_MAX];
	Button *hide_selection = nullptr;
	Button *camera_override_button = nullptr;
	MenuButton *camera_override_menu = nullptr;

	// Game input controls.
	LineEdit *clicked_ctrl = nullptr;
	LineEdit *clicked_ctrl_type = nullptr;
	LineEdit *live_edit_root = nullptr;
	Button *live_edit_set_button = nullptr;
	Button *live_edit_clear_button = nullptr;

	void _sessions_changed();

	void _update_debugger_buttons();

	void _suspend_button_toggled(bool p_pressed);
	void _next_frame_button_pressed();

	void _node_type_pressed(int p_option);
	void _select_mode_pressed(int p_option);

	void _hide_selection_toggled(bool p_pressed);

	void _camera_override_button_toggled(bool p_pressed);
	void _camera_override_menu_id_pressed(int p_id);

	void _live_edit_set_button_pressed();
	void _live_edit_clear_button_pressed();

	void _on_start();
	void _on_stop();

protected:
	void _notification(int p_what);

public:
	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;

	void on_click_ctrl(const String &p_ctrl, const String &p_ctrl_type);

	GameTab(Ref<EditorDebuggerSession> p_session);
};

class GameViewDebugger : public EditorDebuggerPlugin {
	GDCLASS(GameViewDebugger, EditorDebuggerPlugin);

	HashMap<int, GameTab *> tabs;

public:
	virtual void setup_session(int p_session_id) override;
	virtual bool capture(const String &p_message, const Array &p_data, int p_session) override;
	virtual bool has_capture(const String &p_capture) const override;

	void set_state(const Dictionary &p_state);
	Dictionary get_state() const;
};

class GameView : public VBoxContainer {
	GDCLASS(GameView, VBoxContainer);

	Panel *panel = nullptr;

public:
	GameView(Ref<GameViewDebugger> p_debugger);
};

class GameViewPlugin : public EditorPlugin {
	GDCLASS(GameViewPlugin, EditorPlugin);

	GameView *game_view = nullptr;

	Ref<GameViewDebugger> debugger;

protected:
	void _notification(int p_what);

public:
	virtual String get_name() const override { return "Game"; }
	bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override {}
	virtual bool handles(Object *p_object) const override { return false; }
	virtual void make_visible(bool p_visible) override;

	virtual void set_state(const Dictionary &p_state) override;
	virtual Dictionary get_state() const override;

	GameViewPlugin();
	~GameViewPlugin();
};

#endif // GAME_VIEW_PLUGIN_H
