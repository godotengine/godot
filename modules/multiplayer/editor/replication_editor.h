/**************************************************************************/
/*  replication_editor.h                                                  */
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

#include "../scene_replication_config.h"

#include "editor/docks/editor_dock.h"
#include "editor/plugins/editor_plugin.h"

class ConfirmationDialog;
class MultiplayerSynchronizer;
class AcceptDialog;
class LineEdit;
class Tree;
class TreeItem;
class PropertySelector;
class SceneTreeDialog;

class ReplicationEditor : public EditorDock {
	GDCLASS(ReplicationEditor, EditorDock);

private:
	MultiplayerSynchronizer *multiplayer_synchronizer = nullptr;
	SceneReplicationConfig *replication_config = nullptr;

	bool read_only = false;
	TreeItem *ti_edited = nullptr;

	ConfirmationDialog *delete_dialog = nullptr;
	Button *add_pick_button = nullptr;
	Button *move_up = nullptr;
	Button *move_down = nullptr;
	Button *remove = nullptr;
	Button *add_from_path_button = nullptr;
	LineEdit *np_line_edit = nullptr;

	Label *drop_label = nullptr;

	Ref<SceneReplicationConfig> replication_config_ref;
	NodePath deleting;

	MarginContainer *tree_mc = nullptr;
	Tree *tree = nullptr;

	PropertySelector *prop_selector = nullptr;
	SceneTreeDialog *pick_node = nullptr;
	NodePath adding_node_path;

	Button *pin = nullptr;

	Ref<Texture2D> _get_class_icon(const Node *p_node);

	void _add_pressed();
	void _np_text_submitted(const String &p_newtext);
	void _tree_item_edited();
	void _move_up_pressed();
	void _move_down_pressed();
	void _call_swap_property_by_index(SceneReplicationConfig *config, int new_index, int old_index);
	void _delete_button_pressed();
	void _tree_item_selected();
	void _update_config();
	void _dialog_closed(bool p_confirmed);
	void _rename_tree_item(const NodePath &p_old, const NodePath &p_new);
	void _swap_tree_item(const int new_index, const int old_index);
	void _update_tree_item_by_node_path(const NodePath &p_property);
	void _update_tree_item(TreeItem &t_ti);
	void _set_tree_item(TreeItem &item, int p_index, const NodePath &p_property, const bool p_spawn, const SceneReplicationConfig::ReplicationMode p_mode);

	void _pick_node_filter_text_changed(const String &p_newtext);
	void _pick_node_select_recursive(TreeItem *p_item, const String &p_filter, Vector<Node *> &p_select_candidates);
	void _pick_node_selected(NodePath p_path);

	void _pick_new_property();
	void _pick_node_property_selected(String p_name);

	void _pinned();

	bool _can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const;
	void _drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from);

	void _add_sync_property(String p_path);

	void _clear_multiplayer_sync_node();
	void _setup_multiplayer_sync_node();
	void _replication_config_changed();
	void _multiplayer_synchronizer_exit_tree();

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual void update_layout(EditorDock::DockLayout p_layout, int p_slot) override;

public:
	void edit(Object *p_object);
	Button *get_pin() { return pin; }
	ReplicationEditor();
};
