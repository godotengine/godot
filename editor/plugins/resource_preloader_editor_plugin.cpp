/*************************************************************************/
/*  resource_preloader_editor_plugin.cpp                                 */
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

#include "resource_preloader_editor_plugin.h"

#include "core/io/resource_loader.h"
#include "core/project_settings.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"

void ResourcePreloaderEditor::_gui_input(Ref<InputEvent> p_event) {
}

void ResourcePreloaderEditor::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			load->set_icon(get_icon("Folder", "EditorIcons"));
		} break;
	}
}

void ResourcePreloaderEditor::_files_load_request(const Vector<String> &p_paths) {
	for (int i = 0; i < p_paths.size(); i++) {
		String path = p_paths[i];

		RES resource;
		resource = ResourceLoader::load(path);

		if (resource.is_null()) {
			dialog->set_text(TTR("ERROR: Couldn't load resource!"));
			dialog->set_title(TTR("Error!"));
			//dialog->get_cancel()->set_text("Close");
			dialog->get_ok()->set_text(TTR("Close"));
			dialog->popup_centered_minsize();
			return; ///beh should show an error i guess
		}

		String basename = path.get_file().get_basename();
		String name = basename;
		int counter = 1;
		while (preloader->has_resource(name)) {
			counter++;
			name = basename + " " + itos(counter);
		}

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
	for (int i = 0; i < extensions.size(); i++) {
		file->add_filter("*." + extensions[i]);
	}

	file->set_mode(EditorFileDialog::MODE_OPEN_FILES);

	file->popup_centered_ratio();
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

		if (new_name == "" || new_name.find("\\") != -1 || new_name.find("/") != -1 || preloader->has_resource(new_name)) {
			s->set_text(0, old_name);
			return;
		}

		RES samp = preloader->get_resource(old_name);
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
	undo_redo->create_action(TTR("Delete Resource"));
	undo_redo->add_do_method(preloader, "remove_resource", p_to_remove);
	undo_redo->add_undo_method(preloader, "add_resource", p_to_remove, preloader->get_resource(p_to_remove));
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void ResourcePreloaderEditor::_paste_pressed() {
	RES r = EditorSettings::get_singleton()->get_resource_clipboard();
	if (!r.is_valid()) {
		dialog->set_text(TTR("Resource clipboard is empty!"));
		dialog->set_title(TTR("Error!"));
		dialog->get_ok()->set_text(TTR("Close"));
		dialog->popup_centered_minsize();
		return; ///beh should show an error i guess
	}

	String name = r->get_name();
	if (name == "") {
		name = r->get_path().get_file();
	}
	if (name == "") {
		name = r->get_class();
	}

	String basename = name;
	int counter = 1;
	while (preloader->has_resource(name)) {
		counter++;
		name = basename + " " + itos(counter);
	}

	undo_redo->create_action(TTR("Paste Resource"));
	undo_redo->add_do_method(preloader, "add_resource", name, r);
	undo_redo->add_undo_method(preloader, "remove_resource", name);
	undo_redo->add_do_method(this, "_update_library");
	undo_redo->add_undo_method(this, "_update_library");
	undo_redo->commit_action();
}

void ResourcePreloaderEditor::_update_library() {
	tree->clear();
	tree->set_hide_root(true);
	TreeItem *root = tree->create_item(nullptr);

	List<StringName> rnames;
	preloader->get_resource_list(&rnames);

	List<String> names;
	for (List<StringName>::Element *E = rnames.front(); E; E = E->next()) {
		names.push_back(E->get());
	}

	names.sort();

	for (List<String>::Element *E = names.front(); E; E = E->next()) {
		TreeItem *ti = tree->create_item(root);
		ti->set_cell_mode(0, TreeItem::CELL_MODE_STRING);
		ti->set_editable(0, true);
		ti->set_selectable(0, true);
		ti->set_text(0, E->get());
		ti->set_metadata(0, E->get());

		RES r = preloader->get_resource(E->get());

		ERR_CONTINUE(r.is_null());

		String type = r->get_class();
		ti->set_icon(0, EditorNode::get_singleton()->get_class_icon(type, "Object"));
		ti->set_tooltip(0, TTR("Instance:") + " " + r->get_path() + "\n" + TTR("Type:") + " " + type);

		ti->set_text(1, r->get_path());
		ti->set_editable(1, false);
		ti->set_selectable(1, false);

		if (type == "PackedScene") {
			ti->add_button(1, get_icon("InstanceOptions", "EditorIcons"), BUTTON_OPEN_SCENE, false, TTR("Open in Editor"));
		} else {
			ti->add_button(1, get_icon("Load", "EditorIcons"), BUTTON_EDIT_RESOURCE, false, TTR("Open in Editor"));
		}
		ti->add_button(1, get_icon("Remove", "EditorIcons"), BUTTON_REMOVE, false, TTR("Remove"));
	}

	//player->add_resource("default",resource);
}

void ResourcePreloaderEditor::_cell_button_pressed(Object *p_item, int p_column, int p_id) {
	TreeItem *item = Object::cast_to<TreeItem>(p_item);
	ERR_FAIL_COND(!item);

	if (p_id == BUTTON_OPEN_SCENE) {
		String rpath = item->get_text(p_column);
		EditorInterface::get_singleton()->open_scene_from_path(rpath);

	} else if (p_id == BUTTON_EDIT_RESOURCE) {
		RES r = preloader->get_resource(item->get_text(0));
		EditorInterface::get_singleton()->edit_resource(r);

	} else if (p_id == BUTTON_REMOVE) {
		_remove_resource(item->get_text(0));
	}
}

void ResourcePreloaderEditor::edit(ResourcePreloader *p_preloader) {
	preloader = p_preloader;

	if (p_preloader) {
		_update_library();
	} else {
		hide();
		set_physics_process(false);
	}
}

Variant ResourcePreloaderEditor::get_drag_data_fw(const Point2 &p_point, Control *p_from) {
	TreeItem *ti = tree->get_item_at_position(p_point);
	if (!ti) {
		return Variant();
	}

	String name = ti->get_metadata(0);

	RES res = preloader->get_resource(name);
	if (!res.is_valid()) {
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
		RES r = d["resource"];

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
		RES r = d["resource"];

		if (r.is_valid()) {
			String basename;
			if (r->get_name() != "") {
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

void ResourcePreloaderEditor::_bind_methods() {
	ClassDB::bind_method(D_METHOD("_gui_input"), &ResourcePreloaderEditor::_gui_input);
	ClassDB::bind_method(D_METHOD("_load_pressed"), &ResourcePreloaderEditor::_load_pressed);
	ClassDB::bind_method(D_METHOD("_item_edited"), &ResourcePreloaderEditor::_item_edited);
	ClassDB::bind_method(D_METHOD("_paste_pressed"), &ResourcePreloaderEditor::_paste_pressed);
	ClassDB::bind_method(D_METHOD("_files_load_request"), &ResourcePreloaderEditor::_files_load_request);
	ClassDB::bind_method(D_METHOD("_update_library"), &ResourcePreloaderEditor::_update_library);
	ClassDB::bind_method(D_METHOD("_cell_button_pressed"), &ResourcePreloaderEditor::_cell_button_pressed);
	ClassDB::bind_method(D_METHOD("_remove_resource", "to_remove"), &ResourcePreloaderEditor::_remove_resource);

	ClassDB::bind_method(D_METHOD("get_drag_data_fw"), &ResourcePreloaderEditor::get_drag_data_fw);
	ClassDB::bind_method(D_METHOD("can_drop_data_fw"), &ResourcePreloaderEditor::can_drop_data_fw);
	ClassDB::bind_method(D_METHOD("drop_data_fw"), &ResourcePreloaderEditor::drop_data_fw);
}

ResourcePreloaderEditor::ResourcePreloaderEditor() {
	//add_style_override("panel", EditorNode::get_singleton()->get_gui_base()->get_stylebox("panel","Panel"));

	VBoxContainer *vbc = memnew(VBoxContainer);
	add_child(vbc);

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	load = memnew(Button);
	load->set_tooltip(TTR("Load Resource"));
	hbc->add_child(load);

	paste = memnew(Button);
	paste->set_text(TTR("Paste"));
	hbc->add_child(paste);

	file = memnew(EditorFileDialog);
	add_child(file);

	tree = memnew(Tree);
	tree->connect("button_pressed", this, "_cell_button_pressed");
	tree->set_columns(2);
	tree->set_column_min_width(0, 2);
	tree->set_column_min_width(1, 3);
	tree->set_column_expand(0, true);
	tree->set_column_expand(1, true);
	tree->set_v_size_flags(SIZE_EXPAND_FILL);

	tree->set_drag_forwarding(this);
	vbc->add_child(tree);

	dialog = memnew(AcceptDialog);
	add_child(dialog);

	load->connect("pressed", this, "_load_pressed");
	paste->connect("pressed", this, "_paste_pressed");
	file->connect("files_selected", this, "_files_load_request");
	tree->connect("item_edited", this, "_item_edited");
	loading_scene = false;
}

void ResourcePreloaderEditorPlugin::edit(Object *p_object) {
	preloader_editor->set_undo_redo(&get_undo_redo());
	ResourcePreloader *s = Object::cast_to<ResourcePreloader>(p_object);
	if (!s) {
		return;
	}

	preloader_editor->edit(s);
}

bool ResourcePreloaderEditorPlugin::handles(Object *p_object) const {
	return p_object->is_class("ResourcePreloader");
}

void ResourcePreloaderEditorPlugin::make_visible(bool p_visible) {
	if (p_visible) {
		//preloader_editor->show();
		button->show();
		editor->make_bottom_panel_item_visible(preloader_editor);
		//preloader_editor->set_process(true);
	} else {
		if (preloader_editor->is_visible_in_tree()) {
			editor->hide_bottom_panel();
		}
		button->hide();
		//preloader_editor->hide();
		//preloader_editor->set_process(false);
	}
}

ResourcePreloaderEditorPlugin::ResourcePreloaderEditorPlugin(EditorNode *p_node) {
	editor = p_node;
	preloader_editor = memnew(ResourcePreloaderEditor);
	preloader_editor->set_custom_minimum_size(Size2(0, 250) * EDSCALE);

	button = editor->add_bottom_panel_item(TTR("ResourcePreloader"), preloader_editor);
	button->hide();

	//preloader_editor->set_anchor( MARGIN_TOP, Control::ANCHOR_END);
	//preloader_editor->set_margin( MARGIN_TOP, 120 );
}

ResourcePreloaderEditorPlugin::~ResourcePreloaderEditorPlugin() {
}
