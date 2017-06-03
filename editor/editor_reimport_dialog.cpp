/*************************************************************************/
/*  editor_reimport_dialog.cpp                                           */
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
#include "editor_reimport_dialog.h"

#include "editor_file_system.h"
#include "editor_node.h"

#if 0
void EditorReImportDialog::popup_reimport() {

	if (EditorFileSystem::get_singleton()->is_scanning()) {
		error->set_text(TTR("Please wait for scan to complete."));
		error->popup_centered_minsize();
		return;
	}

	tree->clear();
	items.clear();
	List<String> ril;
	EditorFileSystem::get_singleton()->get_changed_sources(&ril);

	scene_must_save=false;


	TreeItem *root = tree->create_item();
	for(List<String>::Element *E=ril.front();E;E=E->next()) {

		TreeItem *item = tree->create_item(root);
		item->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		item->set_metadata(0,E->get());
		item->set_text(0,E->get().replace_first("res://",""));
		item->set_tooltip(0,E->get());
		item->set_checked(0,true);
		item->set_editable(0,true);
		items.push_back(item);

		String name = E->get();

		if (EditorFileSystem::get_singleton()->get_file_type(name)=="PackedScene" && EditorNode::get_singleton()->is_scene_in_use(name)) {

			scene_must_save=true;
		}
	}


	if (scene_must_save) {
		if (EditorNode::get_singleton()->get_edited_scene() && EditorNode::get_singleton()->get_edited_scene()->get_filename()=="") {

			error->set_text(TTR("Current scene must be saved to re-import."));
			error->popup_centered_minsize();
			get_ok()->set_text(TTR("Re-Import"));
			get_ok()->set_disabled(true);
			return;

		}
		get_ok()->set_disabled(false);
		get_ok()->set_text(TTR("Save & Re-Import"));
	} else {
		get_ok()->set_text(TTR("Re-Import"));
		get_ok()->set_disabled(false);
	}

	popup_centered(Size2(600,400));


}


void EditorReImportDialog::ok_pressed() {

	if (EditorFileSystem::get_singleton()->is_scanning()) {
		error->set_text(TTR("Please wait for scan to complete."));
		error->popup_centered_minsize();
		return;
	}



	EditorProgress ep("reimport",TTR("Re-Importing"),items.size());
	String reload_fname;
	if (scene_must_save && EditorNode::get_singleton()->get_edited_scene()) {
		reload_fname = EditorNode::get_singleton()->get_edited_scene()->get_filename();
		EditorNode::get_singleton()->save_scene(reload_fname);
		EditorNode::get_singleton()->clear_scene();
	}

	for(int i=0;i<items.size();i++) {

		String it = items[i]->get_metadata(0);
		ep.step(items[i]->get_text(0),i);
		print_line("reload import from: "+it);
		Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(it);
		ERR_CONTINUE(rimd.is_null());
		String editor = rimd->get_editor();
		Ref<EditorImportPlugin> eip = EditorImportExport::get_singleton()->get_import_plugin_by_name(editor);
		ERR_CONTINUE(eip.is_null());
		Error err = eip->import(it,rimd);
		if (err!=OK) {
			EditorNode::add_io_error("Error Importing:\n  "+it);
		}

	}
	if (reload_fname!="") {
		EditorNode::get_singleton()->load_scene(reload_fname);
	}

	EditorFileSystem::get_singleton()->scan_sources();
}

EditorReImportDialog::EditorReImportDialog() {

	tree = memnew( Tree );
	add_child(tree);
	tree->set_hide_root(true);
	//set_child_rect(tree);
	set_title(TTR("Re-Import Changed Resources"));
	error = memnew( AcceptDialog);
	add_child(error);
	scene_must_save=false;

}
#endif
