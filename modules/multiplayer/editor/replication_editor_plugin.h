/*************************************************************************/
/*  replication_editor_plugin.h                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef REPLICATION_EDITOR_PLUGIN_H
#define REPLICATION_EDITOR_PLUGIN_H

#include "editor/editor_plugin.h"

#include "editor/editor_spin_slider.h"
#include "editor/property_selector.h"

#include "../scene_replication_config.h"

class ConfirmationDialog;
class MultiplayerSynchronizer;
class SceneTreeDialog;
class Tree;
class TreeItem;

class ReplicationEditor : public VBoxContainer {
	GDCLASS(ReplicationEditor, VBoxContainer);

private:
	MultiplayerSynchronizer *current = nullptr;

	AcceptDialog *error_dialog = nullptr;
	ConfirmationDialog *delete_dialog = nullptr;
	Button *add_pick_button = nullptr;
	Button *add_from_path_button = nullptr;
	LineEdit *np_line_edit = nullptr;

	Label *drop_label = nullptr;

	Ref<SceneReplicationConfig> config;
	NodePath deleting;
	Tree *tree = nullptr;

	PropertySelector *prop_selector = nullptr;
	SceneTreeDialog *pick_node = nullptr;
	NodePath adding_node_path;

	Button *pin = nullptr;

	Ref<Texture2D> _get_class_icon(const Node *p_node);

	void _add_pressed();
	void _tree_item_edited();
	void _tree_button_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button);
	void _update_checked(const NodePath &p_prop, int p_column, bool p_checked);
	void _update_config();
	void _dialog_closed(bool p_confirmed);
	void _add_property(const NodePath &p_property, bool p_spawn = true, bool p_sync = true);

	void _pick_node_filter_text_changed(const String &p_newtext);
	void _pick_node_select_recursive(TreeItem *p_item, const String &p_filter, Vector<Node *> &p_select_candidates);
	void _pick_node_filter_input(const Ref<InputEvent> &p_ie);
	void _pick_node_selected(NodePath p_path);

	void _pick_new_property();
	void _pick_node_property_selected(String p_name);

	bool _can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void _drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _add_sync_property(String p_path);

protected:
	static void _bind_methods();

	void _notification(int p_what);

public:
	void edit(MultiplayerSynchronizer *p_object);
	MultiplayerSynchronizer *get_current() const { return current; }

	Button *get_pin() { return pin; }
	ReplicationEditor();
	~ReplicationEditor() {}
};

class ReplicationEditorPlugin : public EditorPlugin {
	GDCLASS(ReplicationEditorPlugin, EditorPlugin);

private:
	Button *button = nullptr;
	ReplicationEditor *repl_editor = nullptr;

	void _node_removed(Node *p_node);

	void _pinned();

protected:
	void _notification(int p_what);

public:
	virtual void edit(Object *p_object) override;
	virtual bool handles(Object *p_object) const override;
	virtual void make_visible(bool p_visible) override;

	ReplicationEditorPlugin();
	~ReplicationEditorPlugin();
};

#endif // REPLICATION_EDITOR_PLUGIN_H
