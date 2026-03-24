/**************************************************************************/
/*  game_object_editor_plugin.h                                           */
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

#include "editor/gui/create_dialog.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/inspector/editor_inspector.h"
#include "editor/plugins/editor_plugin.h"
#include "scene/gui/box_container.h"
#include "scene/main/game_object.h"

class Button;

class GameObjectComponentList : public VBoxContainer {
	GDCLASS(GameObjectComponentList, VBoxContainer);

	Node *game_object = nullptr;
	VBoxContainer *component_container = nullptr;
	CreateDialog *create_dialog = nullptr;
	EditorFileDialog *script_file_dialog = nullptr;

	void _rebuild_list();
	void _on_add_component_pressed();
	void _on_add_script_pressed();
	void _on_create_confirmed();
	void _on_script_file_selected(const String &p_path);
	void _on_child_order_changed();

	void _toggle_component(int p_index);
	void _delete_component(int p_index);

public:
	void set_game_object(Node *p_game_object);
	GameObjectComponentList();
};

class GameObjectInspectorPlugin : public EditorInspectorPlugin {
	GDCLASS(GameObjectInspectorPlugin, EditorInspectorPlugin);

public:
	virtual bool can_handle(Object *p_object) override;
	virtual void parse_begin(Object *p_object) override;
};

class GameObjectEditorPlugin : public EditorPlugin {
	GDCLASS(GameObjectEditorPlugin, EditorPlugin);

public:
	virtual String get_plugin_name() const override { return "GameObjectEditorPlugin"; }
	GameObjectEditorPlugin();
};
