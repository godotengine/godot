/*************************************************************************/
/*  import_settings.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "import_settings.h"
#include "os/os.h"
#include "editor_node.h"

void ImportSettingsDialog::_item_pressed(int p_idx) {

	if (!edited)
		return;

	String p=edited->get_metadata(0);
#if 0
	if (EditorImportDB::get_singleton()->is_image(p)) {

		uint32_t flags = EditorImportDB::get_singleton()->get_image_flags(p);
		bool pressed = !popup->is_item_checked(p_idx);
		if (pressed)
			flags|=(1<<p_idx);
		else
			flags&=~(1<<p_idx);


		EditorImportDB::get_singleton()->set_image_flags(p,flags);
		edited->set_text(2,_str_from_flags(flags));
	}
#endif
}

void ImportSettingsDialog::_item_edited() {

	if (updating)
		return;
	TreeItem *it=tree->get_selected();
	int ec=tree->get_edited_column();

	String p=it->get_metadata(0);
#if 0
	if (EditorImportDB::get_singleton()->is_image(p)) {

		if (ec==1) {


			EditorImportDB::get_singleton()->set_image_format(p,EditorImport::ImageFormat(int(it->get_range(1))));

		} else if (ec==2) {


			int flags = EditorImportDB::get_singleton()->get_image_flags(p);
			for(int i=0;i<popup->get_item_count();i++) {

				popup->set_item_checked(i,flags&(1<<i));
			}

			Rect2 r = tree->get_custom_popup_rect();
			popup->set_size(Size2(r.size.width,1));
			popup->set_pos(/*tree->get_global_pos()+*/r.pos+Point2(0,r.size.height));
			popup->popup();
		}

		edited=it;

	}
#endif


}


void ImportSettingsDialog::_button_pressed(Object *p_button, int p_col, int p_id) {

	TreeItem *ti=p_button->cast_to<TreeItem>();
	if (!ti)
		return;
	String path = ti->get_metadata(0);
	print_line("PATH: "+path);
	Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(path);
	ERR_FAIL_COND(rimd.is_null());
	Ref<EditorImportPlugin> rimp = EditorImportExport::get_singleton()->get_import_plugin_by_name(rimd->get_editor());
	ERR_FAIL_COND(!rimp.is_valid());
	rimp->import_dialog(path);
	hide();
}

bool ImportSettingsDialog::_generate_fs(TreeItem *p_parent,EditorFileSystemDirectory *p_dir) {

	bool valid=false;


	for(int i=0;i<p_dir->get_subdir_count();i++) {

		EditorFileSystemDirectory *sd=p_dir->get_subdir(i);
		TreeItem *ti = tree->create_item(p_parent);
		ti->set_text(0,sd->get_name()+"/");
		ti->set_icon(0,get_icon("Folder","EditorIcons"));

		if (!_generate_fs(ti,sd)) {
			memdelete(ti);
		} else {
			valid=true;
		}
	}


	for(int i=0;i<p_dir->get_file_count();i++) {

		String path=p_dir->get_file_path(i);
		if (!p_dir->get_file_meta(i))
			continue;

		valid=true;

		String f = p_dir->get_file(i);
		TreeItem *ti = tree->create_item(p_parent);
		String type = p_dir->get_file_type(i);
		Ref<Texture> t;
		if (has_icon(type,"EditorIcons"))
			t = get_icon(type,"EditorIcons");
		else
			t = get_icon("Object","EditorIcons");


		ti->set_icon(0,t);
		ti->set_text(0,f);
//		ti->add_button(0,get_icon("Reload","EditorIcons"));
		ti->set_metadata(0,p_dir->get_file_path(i));
		String tt = p_dir->get_file_path(i);

		if (p_dir->is_missing_sources(i)) {
			ti->set_icon(1,get_icon("ImportFail","EditorIcons"));
			Vector<String> missing = p_dir->get_missing_sources(i);
			for(int j=0;j<missing.size();j++) {
				tt+="\nmissing: "+missing[j];
			}

		} else
			ti->set_icon(1,get_icon("ImportCheck","EditorIcons"));

		ti->set_tooltip(0,tt);
		ti->set_tooltip(1,tt);

	}

#if 0
		if (!EditorImportDB::get_singleton()->is_image(path) && !EditorImportDB::get_singleton()->is_scene(path))
			continue;

		String f = p_dir->get_file(i);
		TreeItem *ti = tree->create_item(p_parent);
		ti->set_text(0,f);
		String type = p_dir->get_file_type(i);
		Ref<Texture> icon = get_icon( (has_icon(type,"EditorIcons")?type:String("Object")),"EditorIcons");


		if (EditorImportDB::get_singleton()->is_image(path)) {


			ti->set_tooltip(0,"Type: Image\nSource: "+EditorImportDB::get_singleton()->get_file_source(path)+"\nSource MD5: "+EditorImportDB::get_singleton()->get_file_md5(path));
			ti->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
			ti->set_editable(1,true);
			ti->set_range_config(1,0,3,1);
			ti->set_range(1,EditorImportDB::get_singleton()->get_image_format(path));
			ti->set_text(1,texformat);
			ti->set_cell_mode(2,TreeItem::CELL_MODE_CUSTOM);
			ti->set_editable(2,true);
			ti->set_metadata(0,path);

			String txt;
			uint32_t flags=EditorImportDB::get_singleton()->get_image_flags(path);
			txt=_str_from_flags(flags);

			ti->set_text(2,txt);
		}

		ti->set_icon(0, icon);
		valid=true;
#endif


	return valid;
}

void ImportSettingsDialog::update_tree() {

	updating=true;
	tree->clear();
	edited=NULL;


	TreeItem *root = tree->create_item();
	EditorFileSystemDirectory *fs = EditorFileSystem::get_singleton()->get_filesystem();

	_generate_fs(root,fs);
	updating=false;


}

void ImportSettingsDialog::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		EditorFileSystem::get_singleton()->connect("filesystem_changed",this,"update_tree");
	}
}


void ImportSettingsDialog::_bind_methods() {

	ObjectTypeDB::bind_method("update_tree",&ImportSettingsDialog::update_tree);
	ObjectTypeDB::bind_method("_item_edited",&ImportSettingsDialog::_item_edited);
	ObjectTypeDB::bind_method("_item_pressed",&ImportSettingsDialog::_item_pressed);
	ObjectTypeDB::bind_method("_button_pressed",&ImportSettingsDialog::_button_pressed);


}


void ImportSettingsDialog::popup_import_settings() {

	update_tree();
	popup_centered_ratio();
}

void ImportSettingsDialog::ok_pressed() {


	TreeItem *ti = tree->get_selected();
	if (!ti)
		return;

	String path = ti->get_metadata(0);
	print_line("PATH: "+path);
	Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(path);
	ERR_FAIL_COND(rimd.is_null());
	Ref<EditorImportPlugin> rimp = EditorImportExport::get_singleton()->get_import_plugin_by_name(rimd->get_editor());
	ERR_FAIL_COND(!rimp.is_valid());
	rimp->import_dialog(path);
	hide();


}

ImportSettingsDialog::ImportSettingsDialog(EditorNode *p_editor) {

	editor=p_editor;

	get_ok()->set_text("Close");

	tree = memnew( Tree );
	add_child(tree);
	set_child_rect(tree);
	set_title("Imported Resources");

	texformat="Keep,None,Disk,VRAM";

	tree->set_hide_root(true);
	tree->set_columns(2);
	tree->set_column_expand(1,false);
	tree->set_column_min_width(1,20);

	tree->connect("item_edited",this,"_item_edited");
	tree->connect("button_pressed",this,"_button_pressed");

//	add_button("Re-Import","reimport");
	get_ok()->set_text("Re-Import");
	get_cancel()->set_text("Close");

	updating=false;
	edited=NULL;
	set_hide_on_ok(false);


}

