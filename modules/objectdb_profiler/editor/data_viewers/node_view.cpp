/**************************************************************************/
/*  node_view.cpp                                                         */
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

#include "node_view.h"

#include "editor/editor_node.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/check_button.h"
#include "scene/gui/popup_menu.h"
#include "scene/gui/split_container.h"

SnapshotNodeView::SnapshotNodeView() {
	set_name(TTRC("Nodes"));
}

void SnapshotNodeView::show_snapshot(GameStateSnapshot *p_data, GameStateSnapshot *p_diff_data) {
	SnapshotView::show_snapshot(p_data, p_diff_data);

	set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);

	HSplitContainer *diff_sides = memnew(HSplitContainer);
	diff_sides->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
	add_child(diff_sides);

	main_tree = _make_node_tree(diff_data && !combined_diff_view ? TTRC("A Nodes") : TTRC("Nodes"));
	diff_sides->add_child(main_tree.root);
	_add_snapshot_to_tree(main_tree.tree, snapshot_data, diff_data && combined_diff_view ? DIFF_GROUP_REMOVED : DIFF_GROUP_NONE);

	if (diff_data) {
		CheckButton *diff_mode_toggle = memnew(CheckButton(TTRC("Combine Diff")));
		diff_mode_toggle->set_pressed(combined_diff_view);
		diff_mode_toggle->connect(SceneStringName(toggled), callable_mp(this, &SnapshotNodeView::_toggle_diff_mode));
		main_tree.filter_bar->add_child(diff_mode_toggle);
		main_tree.filter_bar->move_child(diff_mode_toggle, 0);

		if (combined_diff_view) {
			// Merge the snapshots together and add a diff.
			_add_snapshot_to_tree(main_tree.tree, diff_data, DIFF_GROUP_ADDED);
		} else {
			// Add a second column with the diff snapshot.
			diff_tree = _make_node_tree(TTRC("B Nodes"));
			diff_sides->add_child(diff_tree.root);
			_add_snapshot_to_tree(diff_tree.tree, diff_data, DIFF_GROUP_NONE);
		}
	}

	_refresh_icons();
	main_tree.filter_bar->apply();
	if (diff_tree.filter_bar) {
		diff_tree.filter_bar->apply();
		diff_sides->set_split_offset(diff_sides->get_size().x * 0.5);
	}

	choose_object_menu = memnew(PopupMenu);
	add_child(choose_object_menu);
	choose_object_menu->connect(SceneStringName(id_pressed), callable_mp(this, &SnapshotNodeView::_choose_object_pressed).bind(false));
}

NodeTreeElements SnapshotNodeView::_make_node_tree(const String &p_tree_name) {
	NodeTreeElements elements;
	elements.root = memnew(VBoxContainer);
	elements.root->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);
	elements.tree = memnew(Tree);
	elements.filter_bar = memnew(TreeSortAndFilterBar(elements.tree, TTRC("Filter Nodes")));
	elements.root->add_child(elements.filter_bar);
	elements.tree->set_select_mode(Tree::SelectMode::SELECT_ROW);
	elements.tree->set_custom_minimum_size(Size2(150, 0) * EDSCALE);
	elements.tree->set_hide_folding(false);
	elements.root->add_child(elements.tree);
	elements.tree->set_hide_root(true);
	elements.tree->set_allow_reselect(true);
	elements.tree->set_columns(1);
	elements.tree->set_column_titles_visible(true);
	elements.tree->set_column_title(0, p_tree_name);
	elements.tree->set_column_expand(0, true);
	elements.tree->set_column_clip_content(0, false);
	elements.tree->set_column_custom_minimum_width(0, 150 * EDSCALE);
	elements.tree->connect(SceneStringName(item_selected), callable_mp(this, &SnapshotNodeView::_node_selected).bind(elements.tree));
	elements.tree->set_h_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	elements.tree->set_v_size_flags(SizeFlags::SIZE_EXPAND_FILL);
	elements.tree->set_anchors_preset(LayoutPreset::PRESET_FULL_RECT);

	elements.tree->create_item();

	return elements;
}

void SnapshotNodeView::_node_selected(Tree *p_tree_selected_from) {
	active_tree = p_tree_selected_from;
	if (diff_tree.tree) {
		// Deselect nodes in non-active tree, if needed.
		if (active_tree == main_tree.tree) {
			diff_tree.tree->deselect_all();
		}
		if (active_tree == diff_tree.tree) {
			main_tree.tree->deselect_all();
		}
	}

	const LocalVector<SnapshotDataObject *> &item_data = tree_item_data[p_tree_selected_from->get_selected()];
	if (item_data.is_empty()) {
		return;
	} else if (item_data.size() == 1) {
		EditorNode::get_singleton()->push_item(static_cast<Object *>(item_data[0]));
	} else if (item_data.size() == 2) {
		// This happens if we're in the combined diff view and the node exists in both trees
		// The user has to specify which version of the node they want to see in the inspector.
		_show_choose_object_menu();
	}
}

void SnapshotNodeView::_toggle_diff_mode(bool p_state) {
	combined_diff_view = p_state;
	show_snapshot(snapshot_data, diff_data); // Redraw everything when we toggle views.
}

void SnapshotNodeView::_notification(int p_what) {
	if (p_what == NOTIFICATION_THEME_CHANGED) {
		_refresh_icons();
	}
}

void SnapshotNodeView::_add_snapshot_to_tree(Tree *p_tree, GameStateSnapshot *p_snapshot, DiffGroup p_diff_group) {
	SnapshotDataObject *scene_root = nullptr;
	LocalVector<SnapshotDataObject *> orphan_nodes;

	for (const KeyValue<ObjectID, SnapshotDataObject *> &kv : p_snapshot->objects) {
		if (kv.value->is_node() && !kv.value->extra_debug_data.has("node_parent")) {
			if (kv.value->extra_debug_data["node_is_scene_root"]) {
				scene_root = kv.value;
			} else {
				orphan_nodes.push_back(kv.value);
			}
		}
	}

	if (scene_root != nullptr) {
		TreeItem *root_item = _add_item_to_tree(p_tree, p_tree->get_root(), scene_root, p_diff_group);
		_add_children_to_tree(root_item, scene_root, p_diff_group);
	}

	if (!orphan_nodes.is_empty()) {
		TreeItem *orphans_item = _add_item_to_tree(p_tree, p_tree->get_root(), TTRC("Orphan Nodes"), p_diff_group);
		for (SnapshotDataObject *orphan_node : orphan_nodes) {
			TreeItem *orphan_item = _add_item_to_tree(p_tree, orphans_item, orphan_node, p_diff_group);
			_add_children_to_tree(orphan_item, orphan_node, p_diff_group);
		}
	}
}

void SnapshotNodeView::_add_children_to_tree(TreeItem *p_parent_item, SnapshotDataObject *p_data, DiffGroup p_diff_group) {
	for (const Variant &child_id : (Array)p_data->extra_debug_data["node_children"]) {
		SnapshotDataObject *child_object = p_data->snapshot->objects[ObjectID((uint64_t)child_id)];
		TreeItem *child_item = _add_item_to_tree(p_parent_item->get_tree(), p_parent_item, child_object, p_diff_group);
		_add_children_to_tree(child_item, child_object, p_diff_group);
	}
}

TreeItem *SnapshotNodeView::_add_item_to_tree(Tree *p_tree, TreeItem *p_parent, const String &p_item_name, DiffGroup p_diff_group) {
	// Find out if this node already exists.
	TreeItem *item = nullptr;
	if (p_diff_group != DIFF_GROUP_NONE) {
		for (int idx = 0; idx < p_parent->get_child_count(); idx++) {
			TreeItem *child = p_parent->get_child(idx);
			if (child->get_text(0) == p_item_name) {
				item = child;
				break;
			}
		}
	}

	if (item) {
		// If it exists, clear the background color because we now know it exists in both trees.
		item->clear_custom_bg_color(0);
	} else {
		// Add the new node and set its background color to green or red depending on which snapshot it's a part of.
		item = p_tree->create_item(p_parent);

		if (p_diff_group == DIFF_GROUP_ADDED) {
			item->set_custom_bg_color(0, Color(0, 1, 0, 0.1));
		} else if (p_diff_group == DIFF_GROUP_REMOVED) {
			item->set_custom_bg_color(0, Color(1, 0, 0, 0.1));
		}
	}

	item->set_text(0, p_item_name);
	item->set_auto_translate_mode(0, AUTO_TRANSLATE_MODE_DISABLED);

	return item;
}

TreeItem *SnapshotNodeView::_add_item_to_tree(Tree *p_tree, TreeItem *p_parent, SnapshotDataObject *p_data, DiffGroup p_diff_group) {
	String node_name = p_data->extra_debug_data["node_name"];
	TreeItem *child_item = _add_item_to_tree(p_tree, p_parent, node_name, p_diff_group);
	tree_item_data[child_item].push_back(p_data);
	return child_item;
}

void SnapshotNodeView::_refresh_icons() {
	for (TreeItem *item : _get_children_recursive(main_tree.tree)) {
		HashMap<TreeItem *, LocalVector<SnapshotDataObject *>>::Iterator E = tree_item_data.find(item);
		if (E && !E->value.is_empty()) {
			item->set_icon(0, EditorNode::get_singleton()->get_class_icon(E->value[0]->type_name));
		} else {
			item->set_icon(0, EditorNode::get_singleton()->get_class_icon("MissingNode"));
		}
	}

	if (diff_tree.tree) {
		for (TreeItem *item : _get_children_recursive(diff_tree.tree)) {
			HashMap<TreeItem *, LocalVector<SnapshotDataObject *>>::Iterator E = tree_item_data.find(item);
			if (E && !E->value.is_empty()) {
				item->set_icon(0, EditorNode::get_singleton()->get_class_icon(E->value[0]->type_name));
			} else {
				item->set_icon(0, EditorNode::get_singleton()->get_class_icon("MissingNode"));
			}
		}
	}
}

void SnapshotNodeView::clear_snapshot() {
	SnapshotView::clear_snapshot();

	tree_item_data.clear();
	main_tree.tree = nullptr;
	main_tree.filter_bar = nullptr;
	main_tree.root = nullptr;
	diff_tree.tree = nullptr;
	diff_tree.filter_bar = nullptr;
	diff_tree.root = nullptr;
	active_tree = nullptr;
}

void SnapshotNodeView::_choose_object_pressed(int p_object_idx, bool p_confirm_override) {
	EditorNode::get_singleton()->push_item(static_cast<Object *>(tree_item_data[active_tree->get_selected()][p_object_idx]));
}

void SnapshotNodeView::_show_choose_object_menu() {
	remove_child(choose_object_menu);
	add_child(choose_object_menu);
	choose_object_menu->clear(false);
	choose_object_menu->add_item(TTRC("Snapshot A"), 0);
	choose_object_menu->add_item(TTRC("Snapshot B"), 1);
	choose_object_menu->reset_size();
	choose_object_menu->set_position(get_screen_position() + get_local_mouse_position());
	choose_object_menu->popup();
}
