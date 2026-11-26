/**************************************************************************/
/*  viewport_bookmarks.cpp                                                */
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

#include "viewport_bookmarks.h"

#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/templates/hash_set.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/box_container.h"
#include "scene/gui/button.h"
#include "scene/gui/item_list.h"
#include "scene/gui/label.h"
#include "scene/gui/line_edit.h"

static bool _is_finite_number(const Variant &p_value) {
	return (p_value.get_type() == Variant::FLOAT || p_value.get_type() == Variant::INT) && Math::is_finite(double(p_value));
}

Array ViewportBookmarks::get_bookmarks(const Node *p_scene_root, Type p_type) {
	Array valid;
	if (!p_scene_root) {
		return valid;
	}

	Variant metadata_variant = p_scene_root->get_meta(META_KEY, Dictionary());
	if (metadata_variant.get_type() != Variant::DICTIONARY) {
		return valid;
	}
	Dictionary metadata = metadata_variant;
	if (int(metadata.get("version", 0)) != 1) {
		return valid;
	}

	const StringName collection_key = p_type == TYPE_2D ? SNAME("2d") : SNAME("3d");
	Variant bookmarks_variant = metadata.get(collection_key, Array());
	if (bookmarks_variant.get_type() != Variant::ARRAY) {
		return valid;
	}

	Array bookmarks = bookmarks_variant;
	HashSet<String> names;
	for (const Variant &bookmark_variant : bookmarks) {
		if (bookmark_variant.get_type() != Variant::DICTIONARY) {
			continue;
		}
		Dictionary bookmark = bookmark_variant;
		Variant name_variant = bookmark.get("name", Variant());
		if (name_variant.get_type() != Variant::STRING) {
			continue;
		}
		String name = String(name_variant).strip_edges();
		if (name.is_empty() || names.has(name)) {
			continue;
		}

		bool is_valid = false;
		if (p_type == TYPE_2D) {
			Variant offset = bookmark.get("offset", Variant());
			Variant zoom = bookmark.get("zoom", Variant());
			is_valid = offset.get_type() == Variant::VECTOR2 && Vector2(offset).is_finite() && _is_finite_number(zoom) && double(zoom) > 0.0;
		} else {
			Variant position = bookmark.get("position", Variant());
			Variant x_rotation = bookmark.get("x_rotation", Variant());
			Variant y_rotation = bookmark.get("y_rotation", Variant());
			Variant distance = bookmark.get("distance", Variant());
			Variant orthogonal = bookmark.get("orthogonal", Variant());
			Variant view_type = bookmark.get("view_type", Variant());
			is_valid = position.get_type() == Variant::VECTOR3 && Vector3(position).is_finite() && _is_finite_number(x_rotation) && _is_finite_number(y_rotation) && _is_finite_number(distance) && double(distance) > 0.0 && orthogonal.get_type() == Variant::BOOL && view_type.get_type() == Variant::INT;
		}
		if (!is_valid) {
			continue;
		}

		bookmark["name"] = name;
		valid.push_back(bookmark);
		names.insert(name);
	}
	return valid;
}

Dictionary ViewportBookmarks::make_metadata(const Node *p_scene_root, Type p_type, const Array &p_bookmarks) {
	Dictionary metadata;
	if (p_scene_root) {
		Variant existing = p_scene_root->get_meta(META_KEY, Dictionary());
		if (existing.get_type() == Variant::DICTIONARY) {
			metadata = Dictionary(existing).duplicate(true);
		}
	}
	metadata["version"] = 1;
	metadata[p_type == TYPE_2D ? "2d" : "3d"] = p_bookmarks;
	return metadata;
}

bool ViewportBookmarks::is_valid_name(const Array &p_bookmarks, const String &p_name, int p_except) {
	String name = p_name.strip_edges();
	if (name.is_empty()) {
		return false;
	}
	for (int i = 0; i < p_bookmarks.size(); i++) {
		if (i != p_except && Dictionary(p_bookmarks[i]).get("name", "") == name) {
			return false;
		}
	}
	return true;
}

String ViewportBookmarks::make_unique_name(const Array &p_bookmarks) {
	for (int i = 1;; i++) {
		String candidate = vformat(TTR("Bookmark %d"), i);
		if (is_valid_name(p_bookmarks, candidate)) {
			return candidate;
		}
	}
}

Node *ViewportBookmarkManager::_get_scene_root() const {
	return EditorNode::get_singleton()->get_edited_scene();
}

Array ViewportBookmarkManager::_get_bookmarks() const {
	return ViewportBookmarks::get_bookmarks(_get_scene_root(), type);
}

void ViewportBookmarkManager::_commit_bookmarks(const Array &p_bookmarks, const String &p_action) {
	Node *root = _get_scene_root();
	ERR_FAIL_NULL(root);

	Dictionary new_metadata = ViewportBookmarks::make_metadata(root, type, p_bookmarks);
	Array other_bookmarks = ViewportBookmarks::get_bookmarks(root, type == ViewportBookmarks::TYPE_2D ? ViewportBookmarks::TYPE_3D : ViewportBookmarks::TYPE_2D);
	const bool remove_new_metadata = p_bookmarks.is_empty() && other_bookmarks.is_empty();
	const bool had_old_metadata = root->has_meta(ViewportBookmarks::META_KEY);
	Variant old_metadata;
	if (had_old_metadata) {
		old_metadata = root->get_meta(ViewportBookmarks::META_KEY);
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(p_action);
	if (remove_new_metadata) {
		undo_redo->add_do_method(root, "remove_meta", ViewportBookmarks::META_KEY);
	} else {
		undo_redo->add_do_method(root, "set_meta", ViewportBookmarks::META_KEY, new_metadata);
	}
	if (had_old_metadata) {
		undo_redo->add_undo_method(root, "set_meta", ViewportBookmarks::META_KEY, old_metadata);
	} else {
		undo_redo->add_undo_method(root, "remove_meta", ViewportBookmarks::META_KEY);
	}
	undo_redo->add_do_method(this, "refresh");
	undo_redo->add_undo_method(this, "refresh");
	undo_redo->commit_action();
}

void ViewportBookmarkManager::_selection_changed(int p_index) {
	const bool selected = p_index >= 0;
	go_to_button->set_disabled(!selected);
	overwrite_button->set_disabled(!selected);
	rename_button->set_disabled(!selected);
	delete_button->set_disabled(!selected);
}

void ViewportBookmarkManager::_item_activated(int p_index) {
	activate(p_index);
}

void ViewportBookmarkManager::_go_to_selected() {
	PackedInt32Array selected = bookmark_list->get_selected_items();
	if (!selected.is_empty()) {
		activate(selected[0]);
	}
}

void ViewportBookmarkManager::_overwrite_selected() {
	PackedInt32Array selected = bookmark_list->get_selected_items();
	if (selected.is_empty()) {
		return;
	}
	Array bookmarks = _get_bookmarks();
	const int index = selected[0];
	ERR_FAIL_INDEX(index, bookmarks.size());
	Dictionary bookmark = capture_view.call();
	bookmark["name"] = Dictionary(bookmarks[index])["name"];
	bookmarks[index] = bookmark;
	_commit_bookmarks(bookmarks, TTR("Overwrite Viewport Bookmark"));
	refresh();
	bookmark_list->select(index);
	_selection_changed(index);
}

void ViewportBookmarkManager::_rename_selected() {
	PackedInt32Array selected = bookmark_list->get_selected_items();
	if (!selected.is_empty()) {
		_show_name_dialog(selected[0]);
	}
}

void ViewportBookmarkManager::_show_name_dialog(int p_index) {
	if (is_visible()) {
		hide();
	}
	Array bookmarks = _get_bookmarks();
	rename_index = p_index;
	if (p_index >= 0) {
		ERR_FAIL_INDEX(p_index, bookmarks.size());
		name_dialog->set_title(TTR("Rename Viewport Bookmark"));
		name_edit->set_text(Dictionary(bookmarks[p_index]).get("name", ""));
	} else {
		name_dialog->set_title(TTR("Add Viewport Bookmark"));
		name_edit->set_text(ViewportBookmarks::make_unique_name(bookmarks));
	}
	_name_changed(name_edit->get_text());
	name_edit->select_all();
	name_dialog->popup_centered(Size2(360, 0) * EDSCALE);
	name_edit->grab_focus();
}

void ViewportBookmarkManager::_name_changed(const String &p_name) {
	const bool valid = ViewportBookmarks::is_valid_name(_get_bookmarks(), p_name, rename_index);
	name_dialog->get_ok_button()->set_disabled(!valid);
	name_error->set_text(valid ? String() : TTR("The name must be non-empty and unique."));
}

void ViewportBookmarkManager::_name_confirmed() {
	Array bookmarks = _get_bookmarks();
	String name = name_edit->get_text().strip_edges();
	if (!ViewportBookmarks::is_valid_name(bookmarks, name, rename_index)) {
		return;
	}

	if (rename_index >= 0) {
		ERR_FAIL_INDEX(rename_index, bookmarks.size());
		Dictionary bookmark = bookmarks[rename_index];
		bookmark["name"] = name;
		bookmarks[rename_index] = bookmark;
		_commit_bookmarks(bookmarks, TTR("Rename Viewport Bookmark"));
	} else {
		Dictionary bookmark = capture_view.call();
		bookmark["name"] = name;
		bookmarks.push_back(bookmark);
		_commit_bookmarks(bookmarks, TTR("Add Viewport Bookmark"));
	}
	refresh();
}

void ViewportBookmarkManager::_show_delete_dialog() {
	if (is_visible()) {
		hide();
	}
	PackedInt32Array selected = bookmark_list->get_selected_items();
	if (selected.is_empty()) {
		return;
	}
	Array bookmarks = _get_bookmarks();
	const int index = selected[0];
	ERR_FAIL_INDEX(index, bookmarks.size());
	delete_dialog->set_text(vformat(TTR("Delete viewport bookmark \"%s\"?"), Dictionary(bookmarks[index]).get("name", "")));
	delete_dialog->popup_centered();
}

void ViewportBookmarkManager::_delete_confirmed() {
	PackedInt32Array selected = bookmark_list->get_selected_items();
	if (selected.is_empty()) {
		return;
	}
	Array bookmarks = _get_bookmarks();
	const int index = selected[0];
	ERR_FAIL_INDEX(index, bookmarks.size());
	bookmarks.remove_at(index);
	_commit_bookmarks(bookmarks, TTR("Delete Viewport Bookmark"));
	refresh();
}

void ViewportBookmarkManager::_bind_methods() {
	ClassDB::bind_method(D_METHOD("refresh"), &ViewportBookmarkManager::refresh);
}

void ViewportBookmarkManager::refresh() {
	bookmark_list->clear();
	Array bookmarks = _get_bookmarks();
	for (const Variant &bookmark_variant : bookmarks) {
		bookmark_list->add_item(Dictionary(bookmark_variant).get("name", ""));
	}
	_selection_changed(-1);
}

void ViewportBookmarkManager::popup_add() {
	if (!_get_scene_root()) {
		return;
	}
	_show_name_dialog(-1);
}

void ViewportBookmarkManager::popup_manage() {
	refresh();
	popup_centered(Size2(520, 360) * EDSCALE);
}

void ViewportBookmarkManager::activate(int p_index) {
	Array bookmarks = _get_bookmarks();
	ERR_FAIL_INDEX(p_index, bookmarks.size());
	active_index = p_index;
	activate_view.call(Dictionary(bookmarks[p_index]));
}

void ViewportBookmarkManager::activate_next() {
	Array bookmarks = _get_bookmarks();
	if (bookmarks.is_empty()) {
		return;
	}
	activate((active_index + 1) % bookmarks.size());
}

void ViewportBookmarkManager::activate_previous() {
	Array bookmarks = _get_bookmarks();
	if (bookmarks.is_empty()) {
		return;
	}
	activate(active_index <= 0 ? bookmarks.size() - 1 : active_index - 1);
}

ViewportBookmarkManager::ViewportBookmarkManager(ViewportBookmarks::Type p_type, const Callable &p_capture_view, const Callable &p_activate_view) {
	type = p_type;
	capture_view = p_capture_view;
	activate_view = p_activate_view;

	set_title(TTRC("Viewport Bookmarks"));
	set_ok_button_text(TTRC("Close"));
	set_min_size(Size2(520, 360) * EDSCALE);

	VBoxContainer *vbox = memnew(VBoxContainer);
	add_child(vbox);

	bookmark_list = memnew(ItemList);
	bookmark_list->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	bookmark_list->set_allow_reselect(true);
	bookmark_list->connect(SceneStringName(item_selected), callable_mp(this, &ViewportBookmarkManager::_selection_changed));
	bookmark_list->connect("item_activated", callable_mp(this, &ViewportBookmarkManager::_item_activated));
	vbox->add_child(bookmark_list);

	HBoxContainer *buttons = memnew(HBoxContainer);
	vbox->add_child(buttons);
	go_to_button = memnew(Button(TTRC("Go To")));
	go_to_button->connect(SceneStringName(pressed), callable_mp(this, &ViewportBookmarkManager::_go_to_selected));
	buttons->add_child(go_to_button);
	overwrite_button = memnew(Button(TTRC("Overwrite")));
	overwrite_button->connect(SceneStringName(pressed), callable_mp(this, &ViewportBookmarkManager::_overwrite_selected));
	buttons->add_child(overwrite_button);
	rename_button = memnew(Button(TTRC("Rename")));
	rename_button->connect(SceneStringName(pressed), callable_mp(this, &ViewportBookmarkManager::_rename_selected));
	buttons->add_child(rename_button);
	delete_button = memnew(Button(TTRC("Delete")));
	delete_button->connect(SceneStringName(pressed), callable_mp(this, &ViewportBookmarkManager::_show_delete_dialog));
	buttons->add_child(delete_button);

	name_dialog = memnew(ConfirmationDialog);
	name_dialog->set_ok_button_text(TTRC("Save"));
	name_dialog->set_exclusive(false);
	// Nested windows may not render correctly, so use the editor root.
	EditorNode::get_singleton()->get_gui_base()->add_child(name_dialog);
	VBoxContainer *name_vbox = memnew(VBoxContainer);
	name_dialog->add_child(name_vbox);
	name_edit = memnew(LineEdit);
	name_edit->set_custom_minimum_size(Size2(300, 0) * EDSCALE);
	name_edit->set_placeholder(TTRC("Bookmark name"));
	name_edit->connect(SceneStringName(text_changed), callable_mp(this, &ViewportBookmarkManager::_name_changed));
	name_vbox->add_margin_child(TTRC("Name:"), name_edit);
	name_error = memnew(Label);
	name_error->add_theme_color_override(SceneStringName(font_color), Color(1, 0.4, 0.4));
	name_vbox->add_child(name_error);
	name_dialog->register_text_enter(name_edit);
	name_dialog->connect(SceneStringName(confirmed), callable_mp(this, &ViewportBookmarkManager::_name_confirmed));

	delete_dialog = memnew(ConfirmationDialog);
	delete_dialog->set_title(TTRC("Delete Viewport Bookmark"));
	delete_dialog->set_exclusive(false);
	delete_dialog->set_ok_button_text(TTRC("Delete"));
	delete_dialog->connect(SceneStringName(confirmed), callable_mp(this, &ViewportBookmarkManager::_delete_confirmed));
	EditorNode::get_singleton()->get_gui_base()->add_child(delete_dialog);

	_selection_changed(-1);
}
