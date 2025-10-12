/**************************************************************************/
/*  node_view.h                                                           */
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

#include "../snapshot_data.h"
#include "shared_controls.h"
#include "snapshot_view.h"

class Tree;

// When diffing in split view, we have two trees/filters
// so this struct is used to group their properties together.
struct NodeTreeElements {
	NodeTreeElements() {
		tree = nullptr;
		filter_bar = nullptr;
		root = nullptr;
	}
	Tree *tree = nullptr;
	TreeSortAndFilterBar *filter_bar = nullptr;
	VBoxContainer *root = nullptr;
};

class SnapshotNodeView : public SnapshotView {
	GDCLASS(SnapshotNodeView, SnapshotView);

	enum DiffGroup {
		DIFF_GROUP_NONE,
		DIFF_GROUP_ADDED,
		DIFF_GROUP_REMOVED
	};

	NodeTreeElements main_tree;
	NodeTreeElements diff_tree;
	Tree *active_tree = nullptr;
	PopupMenu *choose_object_menu = nullptr;
	bool combined_diff_view = true;
	HashMap<TreeItem *, LocalVector<SnapshotDataObject *>> tree_item_data;

	void _node_selected(Tree *p_tree_selected_from);
	void _notification(int p_what);
	NodeTreeElements _make_node_tree(const String &p_tree_name);
	void _apply_filters();
	void _refresh_icons();
	void _toggle_diff_mode(bool p_state);
	void _choose_object_pressed(int p_object_idx, bool p_confirm_override);
	void _show_choose_object_menu();

	void _add_snapshot_to_tree(Tree *p_tree, GameStateSnapshot *p_snapshot, DiffGroup p_diff_group = DIFF_GROUP_NONE);
	void _add_children_to_tree(TreeItem *p_parent_item, SnapshotDataObject *p_data, DiffGroup p_diff_group = DIFF_GROUP_NONE);
	TreeItem *_add_item_to_tree(Tree *p_tree, TreeItem *p_parent, const String &p_item_name, DiffGroup p_diff_group = DIFF_GROUP_NONE);
	TreeItem *_add_item_to_tree(Tree *p_tree, TreeItem *p_parent, SnapshotDataObject *p_data, DiffGroup p_diff_group = DIFF_GROUP_NONE);

public:
	SnapshotNodeView();
	virtual void show_snapshot(GameStateSnapshot *p_data, GameStateSnapshot *p_diff_data) override;
	virtual void clear_snapshot() override;
};
