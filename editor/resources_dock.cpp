/*************************************************************************/
/*  resources_dock.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "resources_dock.h"

#include "editor_file_system.h"
#include "editor_node.h"
#include "editor_settings.h"
#include "global_config.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "project_settings.h"

void ResourcesDock::_tool_selected(int p_tool) {

	current_action = p_tool;

	switch (p_tool) {

		case TOOL_NEW: {

			create_dialog->popup_centered_ratio();
		} break;
		case TOOL_OPEN: {
			editor->open_resource();

		} break;
		case TOOL_SAVE: {

			TreeItem *ti = resources->get_selected();
			if (!ti)
				break;
			Ref<Resource> current_res = ti->get_metadata(0);

			if (current_res->get_path() != "" && current_res->get_path().find("::") == -1) {
				_file_action(current_res->get_path());
				break;
			};

		}; /* fallthrough */
		case TOOL_SAVE_AS: {

			TreeItem *ti = resources->get_selected();
			if (!ti)
				break;

			save_resource_as(ti->get_metadata(0));

		} break;
		case TOOL_MAKE_LOCAL: {

			TreeItem *ti = resources->get_selected();
			if (!ti)
				break;
			Ref<Resource> current_res = ti->get_metadata(0);
			current_res->set_path("");
			_update_name(ti);
		} break;
		case TOOL_COPY: {

			TreeItem *ti = resources->get_selected();
			if (!ti)
				break;
			Ref<Resource> current_res = ti->get_metadata(0);
			EditorSettings::get_singleton()->set_resource_clipboard(current_res);

		} break;
		case TOOL_PASTE: {

			add_resource(EditorSettings::get_singleton()->get_resource_clipboard());
		} break;
	}
}

void ResourcesDock::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			button_new->set_icon(get_icon("New", "EditorIcons"));
			button_open->set_icon(get_icon("Folder", "EditorIcons"));
			button_save->set_icon(get_icon("Save", "EditorIcons"));
			button_tools->set_icon(get_icon("Tools", "EditorIcons"));

		} break;
	}
}

void ResourcesDock::save_resource(const String &p_path, const Ref<Resource> &p_resource) {

	editor->get_editor_data().apply_changes_in_editors();
	int flg = 0;
	if (EditorSettings::get_singleton()->get("on_save/compress_binary_resources"))
		flg |= ResourceSaver::FLAG_COMPRESS;
	/*
	if (EditorSettings::get_singleton()->get("on_save/save_paths_as_relative"))
		flg|=ResourceSaver::FLAG_RELATIVE_PATHS;
	*/

	String path = GlobalConfig::get_singleton()->localize_path(p_path);
	Error err = ResourceSaver::save(path, p_resource, flg | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS);

	if (err != OK) {
		accept->set_text(TTR("Error saving resource!"));
		accept->popup_centered_minsize();
		return;
	}
	//EditorFileSystem::get_singleton()->update_file(path,p_resource->get_type());

	((Resource *)p_resource.ptr())->set_path(path);
	editor->emit_signal("resource_saved", p_resource);
}

void ResourcesDock::save_resource_as(const Ref<Resource> &p_resource) {

	current_action = TOOL_SAVE_AS;

	RES res(p_resource);

	List<String> extensions;
	ResourceSaver::get_recognized_extensions(res, &extensions);
	file->set_mode(EditorFileDialog::MODE_SAVE_FILE);

	if (p_resource->get_path() != "" && p_resource->get_path().find("::") == -1) {

		file->set_current_path(p_resource->get_path());
	} else {

		String existing;
		if (extensions.size()) {
			existing = "new_" + res->get_class().to_lower() + "." + extensions.front()->get().to_lower();
		}

		file->set_current_file(existing);
	}

	file->clear_filters();
	for (int i = 0; i < extensions.size(); i++) {

		file->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
	}

	file->popup_centered_ratio();
}

void ResourcesDock::_file_action(const String &p_path) {

	switch (current_action) {

		case TOOL_OPEN: {

		} break;
		case TOOL_SAVE:
		case TOOL_SAVE_AS: {

			TreeItem *ti = resources->get_selected();
			if (!ti)
				break;
			Ref<Resource> current_res = ti->get_metadata(0);

			RES res(current_res);

			save_resource(p_path, res);

			_update_name(ti);

		} break;
	}
}

void ResourcesDock::_update_name(TreeItem *item) {

	Ref<Resource> res = item->get_metadata(0);

	if (res->get_name() != "")
		item->set_text(0, res->get_name());
	else if (res->get_path() != "" && res->get_path().find("::") == -1)
		item->set_text(0, res->get_path().get_file());
	else
		item->set_text(0, res->get_class() + " (" + itos(res->get_instance_ID()) + ")");
}

void ResourcesDock::remove_resource(const Ref<Resource> &p_resource) {

	TreeItem *root = resources->get_root();
	ERR_FAIL_COND(!root);

	TreeItem *existing = root->get_children();

	while (existing) {

		Ref<Resource> r = existing->get_metadata(0);
		if (r == p_resource) {
			//existing->move_to_top();
			memdelete(existing);
			return;
		}
		existing = existing->get_next();
	}
}

void ResourcesDock::add_resource(const Ref<Resource> &p_resource) {

	if (block_add)
		return;
	if (!p_resource.is_valid())
		return;

	TreeItem *root = resources->get_root();
	ERR_FAIL_COND(!root);

	TreeItem *existing = root->get_children();

	while (existing) {

		Ref<Resource> r = existing->get_metadata(0);
		if (r == p_resource) {
			//existing->move_to_top();
			existing->select(0);
			resources->ensure_cursor_is_visible();
			return; // existing
		}
		existing = existing->get_next();
	}

	TreeItem *res = resources->create_item(root);
	res->set_metadata(0, p_resource);

	if (has_icon(p_resource->get_class(), "EditorIcons")) {
		res->set_icon(0, get_icon(p_resource->get_class(), "EditorIcons"));
	}

	_update_name(res);
	res->add_button(0, get_icon("Del", "EditorIcons"));
	res->move_to_top();
	res->select(0);
	resources->ensure_cursor_is_visible();
}

void ResourcesDock::_resource_selected() {

	TreeItem *sel = resources->get_selected();
	ERR_FAIL_COND(!sel);

	Ref<Resource> r = sel->get_metadata(0);
	if (r.is_null())
		return;
	block_add = true;
	editor->push_item(r.ptr());
	block_add = false;
}

void ResourcesDock::_delete(Object *p_item, int p_column, int p_id) {

	TreeItem *ti = p_item->cast_to<TreeItem>();
	ERR_FAIL_COND(!ti);

	call_deferred("remove_resource", ti->get_metadata(0));
}

void ResourcesDock::_create() {

	Object *c = create_dialog->instance_selected();

	ERR_FAIL_COND(!c);
	Resource *r = c->cast_to<Resource>();
	ERR_FAIL_COND(!r);

	REF res(r);

	editor->push_item(c);
}

void ResourcesDock::_bind_methods() {

	ClassDB::bind_method(D_METHOD("_tool_selected"), &ResourcesDock::_tool_selected);
	ClassDB::bind_method(D_METHOD("_create"), &ResourcesDock::_create);
	ClassDB::bind_method(D_METHOD("_resource_selected"), &ResourcesDock::_resource_selected);
	ClassDB::bind_method(D_METHOD("_delete"), &ResourcesDock::_delete);
	ClassDB::bind_method(D_METHOD("remove_resource"), &ResourcesDock::remove_resource);
	ClassDB::bind_method(D_METHOD("_file_action"), &ResourcesDock::_file_action);
}

void ResourcesDock::cleanup() {

	resources->clear();
	resources->create_item(); //root
}

ResourcesDock::ResourcesDock(EditorNode *p_editor) {

	editor = p_editor;

	VBoxContainer *vbc = this;

	HBoxContainer *hbc = memnew(HBoxContainer);
	vbc->add_child(hbc);

	Button *b;
	b = memnew(ToolButton);
	b->set_tooltip(TTR("Create New Resource"));
	b->connect("pressed", this, "_tool_selected", make_binds(TOOL_NEW));
	hbc->add_child(b);
	button_new = b;

	b = memnew(ToolButton);
	b->set_tooltip(TTR("Open Resource"));
	b->connect("pressed", this, "_tool_selected", make_binds(TOOL_OPEN));
	hbc->add_child(b);
	button_open = b;

	MenuButton *mb = memnew(MenuButton);
	mb->set_tooltip(TTR("Save Resource"));
	mb->get_popup()->add_item(TTR("Save Resource"), TOOL_SAVE);
	mb->get_popup()->add_item(TTR("Save Resource As.."), TOOL_SAVE_AS);
	mb->get_popup()->connect("id_pressed", this, "_tool_selected");
	hbc->add_child(mb);
	button_save = mb;

	hbc->add_spacer();

	mb = memnew(MenuButton);
	mb->set_tooltip(TTR("Resource Tools"));
	mb->get_popup()->add_item(TTR("Make Local"), TOOL_MAKE_LOCAL);
	mb->get_popup()->add_item(TTR("Copy"), TOOL_COPY);
	mb->get_popup()->add_item(TTR("Paste"), TOOL_PASTE);
	mb->get_popup()->connect("id_pressed", this, "_tool_selected");
	hbc->add_child(mb);
	button_tools = mb;

	resources = memnew(Tree);
	vbc->add_child(resources);
	resources->set_v_size_flags(SIZE_EXPAND_FILL);
	resources->create_item(); //root
	resources->set_hide_root(true);
	resources->connect("cell_selected", this, "_resource_selected");
	resources->connect("button_pressed", this, "_delete");

	create_dialog = memnew(CreateDialog);
	add_child(create_dialog);
	create_dialog->set_base_type("Resource");
	create_dialog->connect("create", this, "_create");
	accept = memnew(AcceptDialog);
	add_child(accept);

	file = memnew(EditorFileDialog);
	add_child(file);
	file->connect("file_selected", this, "_file_action");

	block_add = false;
}
