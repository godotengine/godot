/**************************************************************************/
/*  resource_preloader_editor_plugin.cpp                                  */
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

#include "resource_preloader_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "editor/docks/editor_dock_manager.h"
#include "editor/editor_interface.h"
#include "editor/editor_node.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/settings/editor_command_palette.h"
#include "editor/settings/editor_settings.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"
#include "scene/main/resource_preloader.h"

void ResourcePreloaderEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_TRANSLATION_CHANGED: {
			if (preloader) {
				_update_library();
			}
		} break;

		case NOTIFICATION_THEME_CHANGED: {
			load->set_button_icon(get_editor_theme_icon(SNAME("Folder")));
		} break;
	}
}

void ResourcePreloaderEditor::_files_load_request(const Vector<String> &p_paths) {
	for (int i = 0; i < p_paths.size(); i++) {
		const String &path = p_paths[i];

		Ref<Resource> resource;
		resource = ResourceLoader::load(path);

		if (resource.is_null()) {
			dialog->set_text(TTRC("ERROR: Couldn't load resource!"));
			dialog->popup_centered();
			return; /// Beh, should show an error I guess.
		}

		String basename = path.get_file().get_basename();
		String name = basename;
		int counter = 1;
		while (preloader->has_resource(name)) {
			counter++;
			name = basename + " " + itos(counter);
		}

		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Add Resource"));
		undo_redo->add_do_method(preloader, "add_resource", name, resource);
		undo_redo->add_undo_method(preloader, "remove_resource", name);
		undo_redo->add_do_method(this, "_update_library");
		undo_redo->add_undo_method(this, "_update_library");
		undo_redo->commit_action();
	}
}

void ResourcePreloaderEditor::_load_pressed() {
	loading_scene = false;

	file->clear_filters();
	List<String> extensions;
	ResourceLoader::get_recognized_extensions_for_type("", &extensions);
	for (const String &extension : extensions) {
		file->add_filter("*." + extension);
	}

	file->set_file_mode(EditorFileDialog::FILE_MODE_OPEN_FILES);
	file->popup_file_dialog();
}

void ResourcePreloaderEditor::_item_edited() {
	if (!tree->get_selected()) {
		return;
	}

	TreeItem *s = tree->get_selected();

	if (tree->get_selected_column() == 0) {
		// renamed
		String old_name = s->get_metadata(0);
		String new_name = s->get_text(0);
		if (old_name == new_name) {
			return;
		}

		if (new_name.is_empty() || new_name.contains_char('\\') || new_name.contains_char('/') || preloader->has_resource(new_name)) {
			s->set_text(0, old_name);
			return;
		}

		Ref<Resource> samp = preloader->get_resource(old_name);
		EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
		undo_redo->create_action(TTR("Rename Resource"));
		undo_redo->add_do_method(preloader, "remove_resource", old_name);
		undo_redo->add_do_method(preloader, "add_resource", new_name, samp);
		undo_redo->add_undo_method(preloader, "remove_resource", new_name);
		undo_redo->add_undo_method(preloader, "add_resource", old_name, samp);
		undo_redo->add_do_method(this, "_update_library");
		undo_redo->add_undo_method(this, "_update_library");
		undo_redo->commit_action();
	}
}

void ResourcePreloaderEditor::_remove_resource(const String &p_to_remove) {
	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Delete Resource"));
	undo_redo->add_do_method(preloader, "remove_resource", p_to_remove);
	undo_redo->add_undo_method(preloader, "add_resource", p_to_remove, preloader->get_resource(p_to_remove));
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void ResourcePreloaderEditor::_paste_pressed() {
	Ref<Resource> r = EditorSettings::get_singleton()->get_resource_clipboard();
	if (r.is_null()) {
		dialog->set_text(TTRC("Resource clipboard is empty!"));
		dialog->popup_centered();
		return; /// Beh, should show an error I guess.
	}

	String name = r->get_name();
	if (name.is_empty()) {
		name = r->get_path().get_file();
	}
	if (name.is_empty()) {
		name = r->get_class();
	}

	String basename = name;
	int counter = 1;
	while (preloader->has_resource(name)) {
		counter++;
		name = basename + " " + itos(counter);
	}

	EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
	undo_redo->create_action(TTR("Paste Resource"));
	undo_redo->add_do_method(preloader, "add_resource", name, r);
	undo_redo->add_undo_method(preloader, "remove_resource", name);
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void ResourcePreloaderEditor::_update_library() {
	tree->clear();
	if (!preloader) {
		return;
	}

	TreeItem *root = tree->create_item(nullptr);

	List<StringName> rnames;
	preloader->get_resource_list(&rnames);

	List<String> names;
	for (const StringName &E : rnames) {
		names.push_back(E);
	}

	names.sort();

	for (const String &E : names) {
		TreeItem *ti = tree->create_item(root);
		ti->set_cell_mode(0, TreeItem::CELL_MODE_STRING);
		ti->set_editable(0, true);
		ti->set_selectable(0, true);
		ti->set_text(0, E);
		ti->set_metadata(0, E);

		Ref<Resource> r = preloader->get_resource(E);

		ERR_CONTINUE(r.is_null());

		String type = r->get_class();
		ti->set_icon(0, EditorNode::get_singleton()->get_class_icon(type));
		ti->set_tooltip_text(0, TTR("Instance:") + " " + r->get_path() + "\n" + TTR("Type:") + " " + type);

		ti->set_text(1, r->get_path());
		ti->set_editable(1, false);
		ti->set_selectable(1, false);

		if (type == "PackedScene") {
			ti->add_button(1, get_editor_theme_icon(SNAME("InstanceOptions")), BUTTON_OPEN_SCENE, false, TTR("Open in Editor"));
		} else {
			ti->add_button(1, get_editor_theme_icon(SNAME("Load")), BUTTON_EDIT_RESOURCE, false, TTR("Open in Editor"));
		}
		ti->add_button(1, get_editor_theme_icon(SNAME("Remove")), BUTTON_REMOVE, false, TTR("Remove"));
	}
}

void ResourcePreloaderEditor::_cell_button_pressed(Object *p_item, int p_column, int p_id, MouseButton p_button) {
	if (p_button != MouseButton::LEFT) {
		return;
	}

	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_NULL(item);

	if (p_id == BUTTON_OPEN_SCENE) {
		String rpath = item->get_text(p_column);
		EditorInterface::get_singleton()->open_scene_from_path(rpath);

	} else if (p_id == BUTTON_EDIT_RESOURCE) {
		Ref<Resource> r = preloader->get_resource(item->get_text(0));
		EditorInterface::get_singleton()->edit_resource(r);

	} else if (p_id == BUTTON_REMOVE) {
		_remove_resource(item->get_text(0));
	}
}

void ResourcePreloaderEditor::edit(ResourcePreloader *p_preloader) {
	preloader = p_preloader;
	_update_library();
}

Variant ResourcePreloaderEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *ti = (p_point == Vector2(Math::INF, Math::INF)) ? tree->get_selected() : tree->get_item_at_position(p_point);
	if (!ti) {
		return Variant();
	}

	String name = ti->get_metadata(0);

	Ref<Resource> res = preloader->get_resource(name);
	if (res.is_null()) {
		return Variant();
	}

	return EditorNode::get_singleton()->drag_resource(res, p_from);
}

bool ResourcePreloaderEditor::can_drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) const {
	Dictionary d = p_data;

	if (!d.has("type")) {
		return false;
	}

	if (d.has("from") && (Object *)(d["from"]) == tree) {
		return false;
	}

	if (String(d["type"]) == "resource" && d.has("resource")) {
		Ref<Resource> r = d["resource"];

		return r.is_valid();
	}

	if (String(d["type"]) == "files") {
		Vector<String> files = d["files"];

		return files.size() != 0;
	}
	return false;
}

void ResourcePreloaderEditor::drop_data_fw(const Point2 &p_point, const Variant &p_data, Control *p_from) {
	if (!can_drop_data_fw(p_point, p_data, p_from)) {
		return;
	}

	Dictionary d = p_data;

	if (!d.has("type")) {
		return;
	}

	if (String(d["type"]) == "resource" && d.has("resource")) {
		Ref<Resource> r = d["resource"];

		if (r.is_valid()) {
			String basename;
			if (!r->get_name().is_empty()) {
				basename = r->get_name();
			} else if (r->get_path().is_resource_file()) {
				basename = r->get_path().get_basename();
			} else {
				basename = "Resource";
			}

			String name = basename;
			int counter = 0;
			while (preloader->has_resource(name)) {
				counter++;
				name = basename + "_" + itos(counter);
			}

			EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
			undo_redo->create_action(TTR("Add Resource"));
			undo_redo->add_do_method(preloader, "add_resource", name, r);
			undo_redo->add_undo_method(preloader, "remove_resource", name);
			undo_redo->add_do_method(this, "_update_library");
			undo_redo->add_undo_method(this, "_update_library");
			undo_redo->commit_action();
		}
	}

	if (String(d["type"]) == "files") {
		Vector<String> files = d["files"];

		_files_load_request(files);
	}
}

void ResourcePreloaderEditor::update_layout(EditorDock::DockLayout p_layout) {
	bool new_horizontal = (p_layout == EditorDock::DOCK_LAYOUT_HORIZONTAL);
	if (horizontal == new_horizontal) {
		return;
	}
	horizontal = new_horizontal;

	if (horizontal) {
		mc->set_theme_type_variation("NoBorderHorizontal");
		tree->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_BOTH);
	} else {
		mc->set_theme_type_variation("NoBorderHorizontalBottom");
		tree->set_scroll_hint_mode(Tree::SCROLL_HINT_MODE_TOP);
	}
}

void ResourcePreloaderEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_update_library"), &ResourcePreloaderEditor::_update_library);
	ClassDB::bind_method(D_METHOD("_remove_resource", "to_remove"), &ResourcePreloaderEditor::_remove_resource);
}

ResourcePreloaderEditor::ResourcePreloaderEditor() {
	set_name(TTRC("ResourcePreloader"));
	set_icon_name("ResourcePreloader");
	set_dock_shortcut(ED_SHORTCUT_AND_COMMAND("bottom_panels/toggle_resource_preloader_bottom_panel", TTRC("Toggle ResourcePreloader Dock")));
	set_default_slot(DockConstants::DOCK_SLOT_BOTTOM);
	set_available_layouts(EditorDock::DOCK_LAYOUT_ALL);
	set_global(false);
	set_transient(true);

	set_custom_minimum_size(Size2(0, 250 * EDSCALE));

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	load = memnew(Button);
	load->set_tooltip_text(TTRC("Load Resource"));
	hbc->add_child(load);

	paste = memnew(Button);
	paste->set_text(TTRC("Paste"));
	hbc->add_child(paste);

	file = memnew(EditorFileDialog);
	add_child(file);

	mc = memnew(MarginContainer);
	mc->set_v_size_flags(SIZE_EXPAND_FILL);
	vbc->add_child(mc);

	tree = memnew(Tree);
	tree->connect("button_clicked", callable_mp(this, &ResourcePreloaderEditor::_cell_button_pressed));
	tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	tree->set_hide_root(true);
	tree->set_columns(2);
	tree->set_column_expand_ratio(0, 2);
	tree->set_column_clip_content(0, true);
	tree->set_column_expand_ratio(1, 3);
	tree->set_column_clip_content(1, true);

	SET_DRAG_FORWARDING_GCD(tree, ResourcePreloaderEditor);
	mc->add_child(tree);

	dialog = memnew(AcceptDialog);
	dialog->set_title(TTRC("Error!"));
	dialog->set_ok_button_text(TTRC("Close"));
	add_child(dialog);

	load->connect(SceneStringName(pressed), callable_mp(this, &ResourcePreloaderEditor::_load_pressed));
	paste->connect(SceneStringName(pressed), callable_mp(this, &ResourcePreloaderEditor::_paste_pressed));
	file->connect("files_selected", callable_mp(this, &ResourcePreloaderEditor::_files_load_request));
	tree->connect("item_edited", callable_mp(this, &ResourcePreloaderEditor::_item_edited));
	loading_scene = false;
}

void ResourcePreloaderEditorPlugin::edit(Object *p_object) {
	ResourcePreloader *s = Object::cast_to<ResourcePreloader>(p_object);
	if (!s) {
		return;
	}

	preloader_editor->edit(s);
}

bool ResourcePreloaderEditorPlugin::handles(Object *p_object) const {
	return Object::cast_to<ResourcePreloader>(p_object);
}

void ResourcePreloaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		preloader_editor->make_visible();
	} else {
		preloader_editor->close();
	}
}

ResourcePreloaderEditorPlugin::ResourcePreloaderEditorPlugin() {
	preloader_editor = memnew(ResourcePreloaderEditor);
	EditorDockManager::get_singleton()->add_dock(preloader_editor);
	preloader_editor->close();
}
