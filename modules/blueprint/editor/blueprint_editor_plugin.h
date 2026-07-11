/**************************************************************************/
/*  blueprint_editor_plugin.h                                             */
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
/* included in all copies or substantial portions of the Software.       */
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

#include "../blueprint.h"

#include "editor/plugins/editor_plugin.h"
#include "scene/gui/box_container.h"

class Button;
class EditorFileDialog;
class GraphEdit;
class GraphNode;
class Label;
class MenuButton;
class PopupMenu;
class TabBar;

class BlueprintEditor : public VBoxContainer {
	GDCLASS(BlueprintEditor, VBoxContainer);

	Vector<Ref<Blueprint>> open_blueprints;
	Ref<Blueprint> blueprint; // The one currently shown (open_blueprints[current tab]).

	TabBar *tabs = nullptr;
	GraphEdit *graph = nullptr;
	MenuButton *add_menu = nullptr;
	PopupMenu *context_menu = nullptr;
	Label *info_label = nullptr;
	Label *empty_hint = nullptr;
	Button *save_button = nullptr;
	EditorFileDialog *new_dialog = nullptr;
	EditorFileDialog *open_dialog = nullptr;

	Vector2 pending_add_position;

	void _rebuild();
	void _make_graph_node(const Dictionary &p_data);
	void _style_graph_node(GraphNode *p_gn, const String &p_category);
	void _fill_node_menu(PopupMenu *p_menu);
	Vector2 _center_graph_position() const;
	String _tab_title(const Ref<Blueprint> &p_blueprint) const;
	void _update_state();

	void _on_tab_changed(int p_tab);
	void _on_tab_close_pressed(int p_tab);
	void _on_new_pressed();
	void _on_open_pressed();
	void _on_save_pressed();
	void _on_new_file_selected(const String &p_path);
	void _on_open_file_selected(const String &p_path);

	void _on_connection_request(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port);
	void _on_disconnection_request(const StringName &p_from, int p_from_port, const StringName &p_to, int p_to_port);
	void _on_delete_nodes_request(const TypedArray<StringName> &p_nodes);
	void _on_popup_request(const Vector2 &p_position);
	void _on_add_menu_pressed(int p_id);
	void _on_context_menu_pressed(int p_id);
	void _add_node_of_type(int p_type_index, const Vector2 &p_graph_position);
	void _on_node_dragged(const Vector2 &p_from, const Vector2 &p_to, int p_id);
	void _on_param_changed(const String &p_text, int p_id, const String &p_key);

public:
	void edit(const Ref<Blueprint> &p_blueprint);

	BlueprintEditor();
};

class BlueprintEditorPlugin : public EditorPlugin {
	GDCLASS(BlueprintEditorPlugin, EditorPlugin);

	BlueprintEditor *blueprint_editor = nullptr;

public:
	virtual String get_plugin_name() const override { return "Blueprint"; }
	virtual const Ref<Texture2D> get_plugin_icon() const override;
	virtual bool has_main_screen() const override { return true; }
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	BlueprintEditorPlugin();
};
