/**************************************************************************/
/*  game_editor_plugin.h                                                  */
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

#ifndef GAME_EDITOR_PLUGIN_H
#define GAME_EDITOR_PLUGIN_H

#include "editor/plugins/editor_debugger_plugin.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/debugger/scene_debugger.h"
#include "scene/gui/box_container.h"

class GameEditorDebugger : public EditorDebuggerPlugin {
	GDCLASS(GameEditorDebugger, EditorDebuggerPlugin);

	Vector<Ref<EditorDebuggerSession>> sessions;

	int node_type = RuntimeNodeSelect::NODE_TYPE_2D;
	int select_mode = RuntimeNodeSelect::SELECT_MODE_SINGLE;

	void _session_started(Ref<EditorDebuggerSession> p_session);
	void _session_stopped();

protected:
	static void _bind_methods();

public:
	void set_suspend(bool p_enabled);
	void next_frame();

	void set_node_type(int p_type);
	void set_select_mode(int p_mode);

	virtual void setup_session(int p_session_id) override;

	GameEditorDebugger() {}
};

class GameEditor : public VBoxContainer {
	GDCLASS(GameEditor, VBoxContainer);

private:
	Ref<GameEditorDebugger> debugger;

	int active_sessions = 0;

	Button *suspend_button = nullptr;
	Button *next_frame_button = nullptr;

	Button *node_type_button[RuntimeNodeSelect::NODE_TYPE_MAX];
	Button *select_mode_button[RuntimeNodeSelect::SELECT_MODE_MAX];

	Panel *panel = nullptr;

	void _sessions_changed();

	void _update_debugger_buttons();

	void _suspend_button_toggled(bool p_pressed);

	void _node_type_pressed(int p_option);
	void _select_mode_pressed(int p_option);

protected:
	void _notification(int p_what);

public:
	GameEditor(Ref<GameEditorDebugger> p_debugger);
};

class GameEditorPlugin : public EditorPlugin {
	GDCLASS(GameEditorPlugin, EditorPlugin);

	GameEditor *game_editor = nullptr;

	Ref<GameEditorDebugger> debugger;

protected:
	void _notification(int p_what);

public:
	virtual String get_name() const override { return "Game"; }
	bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override {}
	virtual bool handles(Object *p_object) const override { return false; }
	virtual void make_visible(bool p_visible) override;

	GameEditorPlugin();
	~GameEditorPlugin();
};

#endif // GAME_EDITOR_PLUGIN_H
