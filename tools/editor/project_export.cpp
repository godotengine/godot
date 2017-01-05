/*************************************************************************/
/*  project_export.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "project_export.h"

#include "os/dir_access.h"
#include "os/file_access.h"
#include "globals.h"

#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/os.h"
#include "scene/gui/box_container.h"

#include "scene/gui/tab_container.h"
#include "scene/gui/scroll_container.h"
#include "editor_data.h"
#include "io/image_loader.h"
#include "compressed_translation.h"
#include "editor_node.h"
#include "io_plugins/editor_texture_import_plugin.h"
#include "editor_settings.h"

const char *ProjectExportDialog::da_string[ProjectExportDialog::ACTION_MAX]={
	"",
	"Copy",
	"Bundle"
};

bool ProjectExportDialog::_create_tree(TreeItem *p_parent,EditorFileSystemDirectory *p_dir) {

	TreeItem *item = tree->create_item(p_parent);
	item->set_text(0,p_dir->get_name()+"/");
	item->set_icon(0,get_icon("Folder","EditorIcons"));


	bool has_items=false;

	for(int i=0;i<p_dir->get_subdir_count();i++) {

		if (_create_tree(item,p_dir->get_subdir(i)))
			has_items=true;
	}

//	int cc = p_options.get_slice_count(",");

	for (int i=0;i<p_dir->get_file_count();i++) {

		TreeItem *fitem = tree->create_item(item);
		//fitem->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		//fitem->set_editable(0,true);
	//	fitem->set_checked(0,isfave);
		fitem->set_text(0,p_dir->get_file(i));
		String path = p_dir->get_file_path(i);
		fitem->set_tooltip(0,path);
		fitem->set_metadata(0,path);
		Ref<Texture> icon = get_icon( (has_icon(p_dir->get_file_type(i),ei)?p_dir->get_file_type(i):ot),ei);
		fitem->set_icon(0,icon);

		fitem->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
		fitem->set_range_config(1,0,2,1);
		fitem->set_text(1,expopt);
		fitem->set_editable(1,true);

		EditorImportExport::FileAction fa = EditorImportExport::get_singleton()->get_export_file_action(path);
		fitem->set_range(1,fa);

		has_items=true;

	}

	if (!has_items) {

		memdelete(item);
		return false;

	}

	return true;
}


void ProjectExportDialog::_tree_changed() {

	TreeItem *t=tree->get_selected();
	if (!t)
		return;

	String selected = t->get_metadata(0);

	EditorImportExport::get_singleton()->set_export_file_action(selected,EditorImportExport::FileAction(int(t->get_range(1))));
	_save_export_cfg();

	//editor->save_import_export(true);
	//EditorImportDB::get_singleton()->save_settings();

}

void ProjectExportDialog::popup_export() {
	popup_centered_ratio();
	if (pending_update_tree) {
		_update_tree();
		_update_group_tree();
		pending_update_tree=false;
	}
}

void ProjectExportDialog::_update_tree() {



	updating_tree=true;
	tree->clear();
	EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->get_filesystem();

	if (efsd) {
		_create_tree(NULL,efsd);
	}

	updating_tree=false;
}




void ProjectExportDialog::_update_platform() {

	_validate_platform();
	TreeItem *selected = platforms->get_selected();
	if (!selected)
		return;

	String platform = selected->get_metadata(0);
	Ref<EditorExportPlatform> exporter = EditorImportExport::get_singleton()->get_export_platform(platform);
	platform_options->edit( exporter.ptr() );
}

void ProjectExportDialog::_platform_selected() {

	String p =platforms->get_selected()->get_metadata(0);
	_update_platform();
//	editor->save_import_export();
//	EditorFileSystem::get_singleton()->scan();

}

void ProjectExportDialog::_scan_finished() {

/*	print_line("**********SCAN DONEEE********");
	print_line("**********SCAN DONEEE********");
	print_line("**********SCAN DONEEE********");
	print_line("**********SCAN DONEEE********");*/

	if (!is_visible()) {
		pending_update_tree=true;
		return;
	}

	_update_tree();
	_update_group_tree();
}

void ProjectExportDialog::_rescan() {

	EditorFileSystem::get_singleton()->scan();

}

void ProjectExportDialog::_update_exporter() {


}


void ProjectExportDialog::_save_export_cfg() {

	EditorImportExport::get_singleton()->save_config();
}

void ProjectExportDialog::_prop_edited(String what) {

	_save_export_cfg();

	_validate_platform();

}

void ProjectExportDialog::_filters_edited(String what) {

	EditorImportExport::get_singleton()->set_export_custom_filter(what);
	_save_export_cfg();
}

void ProjectExportDialog::_filters_exclude_edited(String what) {
	EditorImportExport::get_singleton()->set_export_custom_filter_exclude(what);
	_save_export_cfg();
}

void ProjectExportDialog::_quality_edited(float what) {

	EditorImportExport::get_singleton()->set_export_image_quality(what);
	_save_export_cfg();
}

void ProjectExportDialog::_shrink_edited(float what) {

	EditorImportExport::get_singleton()->set_export_image_shrink(what);
	_save_export_cfg();
}

void ProjectExportDialog::_image_export_edited(int what) {

	EditorImportExport::get_singleton()->set_export_image_action(EditorImportExport::ImageAction(what));
	_save_export_cfg();
}

void ProjectExportDialog::_format_toggled() {

	EditorImportExport::get_singleton()->get_image_formats().clear();

	for(int i=0;i<formats.size();i++) {
		if (formats[i]->is_checked(0))
			EditorImportExport::get_singleton()->get_image_formats().insert( formats[i]->get_text(0));

	}
	_save_export_cfg();
}


void ProjectExportDialog::_script_edited(Variant v) {

	if (updating_script)
		return;
	updating_script=true;
	EditorNode::get_undo_redo()->create_action(TTR("Edit Script Options"));
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"script_set_action",script_mode->get_selected());
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"script_set_action",EditorImportExport::get_singleton()->script_get_action());
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"script_set_encryption_key",script_key->get_text());
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"script_set_encryption_key",EditorImportExport::get_singleton()->script_get_encryption_key());
	EditorNode::get_undo_redo()->add_do_method(this,"_update_script");
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_script");
	EditorNode::get_undo_redo()->add_do_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->add_undo_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->commit_action();
	updating_script=false;


}

void ProjectExportDialog::_sample_convert_edited(int what) {
	EditorImportExport::get_singleton()->sample_set_action( EditorImportExport::SampleAction(sample_mode->get_selected()));
	EditorImportExport::get_singleton()->sample_set_max_hz(  sample_max_hz->get_value() );
	EditorImportExport::get_singleton()->sample_set_trim(  sample_trim->is_pressed() );
	_save_export_cfg();

}

void ProjectExportDialog::_notification(int p_what) {

	switch(p_what) {

		case NOTIFICATION_ENTER_TREE: {


			CenterContainer *cc = memnew( CenterContainer );
			TextureFrame *tf = memnew( TextureFrame);
			tf->set_texture(get_icon("ErrorSign","EditorIcons"));
			cc->add_child(tf);
			plat_errors->add_child(cc);
			platform_error_string->raise();

			TreeItem *root = platforms->create_item(NULL);
			List<StringName> ep;
			EditorImportExport::get_singleton()->get_export_platforms(&ep);
			ep.sort_custom<StringName::AlphCompare>();

			for(List<StringName>::Element *E=ep.front();E;E=E->next()) {


				Ref<EditorExportPlatform> eep = EditorImportExport::get_singleton()->get_export_platform(E->get());
				TreeItem *p = platforms->create_item(root);
				p->set_text(0,eep->get_name());
				p->set_icon(0,eep->get_logo());
				p->set_metadata(0,eep->get_name());
				if (eep->get_name()==OS::get_singleton()->get_name())
					p->select(0);

			}

			EditorFileSystem::get_singleton()->connect("filesystem_changed",this,"_scan_finished");
//			_rescan();
			_update_platform();
			export_mode->select( EditorImportExport::get_singleton()->get_export_filter() );
			convert_text_scenes->set_pressed( EditorImportExport::get_singleton()->get_convert_text_scenes() );
			filters->set_text( EditorImportExport::get_singleton()->get_export_custom_filter() );
			filters_exclude->set_text( EditorImportExport::get_singleton()->get_export_custom_filter_exclude() );
			if (EditorImportExport::get_singleton()->get_export_filter()!=EditorImportExport::EXPORT_SELECTED)
				tree_vb->hide();
			else
				tree_vb->show();

			image_action->select(EditorImportExport::get_singleton()->get_export_image_action());
			image_quality->set_value(EditorImportExport::get_singleton()->get_export_image_quality());
			image_shrink->set_value(EditorImportExport::get_singleton()->get_export_image_shrink());
			_update_script();


			image_quality->connect("value_changed",this,"_quality_edited");
			image_shrink->connect("value_changed",this,"_shrink_edited");
			image_action->connect("item_selected",this,"_image_export_edited");

			script_mode->connect("item_selected",this,"_script_edited");
			script_key->connect("text_changed",this,"_script_edited");

			for(int i=0;i<formats.size();i++) {
				if (EditorImportExport::get_singleton()->get_image_formats().has(formats[i]->get_text(0)))
					formats[i]->set_checked(0,true);
			}
			image_formats->connect("item_edited",this,"_format_toggled");
			group_add->set_icon(get_icon("Add","EditorIcons"));
//			group_del->set_icon(get_icon("Del","EditorIcons"));

			_update_group_list();
			_update_group();
			_update_group_tree();

			sample_mode->select( EditorImportExport::get_singleton()->sample_get_action() );
			sample_max_hz->set_value( EditorImportExport::get_singleton()->sample_get_max_hz() );
			sample_trim->set_pressed( EditorImportExport::get_singleton()->sample_get_trim() );

			sample_mode->connect("item_selected",this,"_sample_convert_edited");
			sample_max_hz->connect("value_changed",this,"_sample_convert_edited");
			sample_trim->connect("toggled",this,"_sample_convert_edited");


		} break;
		case NOTIFICATION_EXIT_TREE: {

		} break;
		case MainLoop::NOTIFICATION_WM_FOCUS_IN: {
			//something may have changed
			_validate_platform();

		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible())
				_validate_platform();

		} break;
		case NOTIFICATION_PROCESS: {

		} break;
	}

}


void ProjectExportDialog::_validate_platform() {

	get_ok()->set_disabled(true);
	button_export->set_disabled(true);
	TreeItem *selected = platforms->get_selected();
	plat_errors->hide();
	if (!selected) {
		return;
	}

	String platform = selected->get_metadata(0);
	Ref<EditorExportPlatform> exporter = EditorImportExport::get_singleton()->get_export_platform(platform);
	if (!exporter.is_valid()) {
		return;
	}

	String err;
	if (!exporter->can_export(&err)) {
		Vector<String> items = err.strip_edges().split("\n");
		err="";
		for(int i=0;i<items.size();i++) {
			if (i!=0)
				err+="\n";
			err+="  -"+items[i];
		}

		platform_error_string->set_text(err);
		plat_errors->show();
		return;
	}

	List<String> pl;
	EditorFileSystem::get_singleton()->get_changed_sources(&pl);

	if (false && pl.size()) {
		if (pl.size()==1)
			platform_error_string->set_text(" -One Resource is pending re-import.");
		else
			platform_error_string->set_text("  "+itos(pl.size())+" Resources are pending re-import.");

		plat_errors->show();
		return;
	}

	get_ok()->set_disabled(false);
	button_export->set_disabled(false);

}

void ProjectExportDialog::_export_mode_changed(int p_idx) {

	if (EditorImportExport::get_singleton()->get_export_filter()==p_idx)
		return;
	EditorImportExport::get_singleton()->set_export_filter(EditorImportExport::ExportFilter(p_idx));

	if (p_idx!=EditorImportExport::EXPORT_SELECTED)
		tree_vb->hide();
	else
		tree_vb->show();

	EditorImportExport::get_singleton()->set_convert_text_scenes( convert_text_scenes->is_pressed() );

	_save_export_cfg();

}

void ProjectExportDialog::_export_action(const String& p_file) {

	String location = GlobalConfig::get_singleton()->globalize_path(p_file).get_base_dir().replace("\\","/");

	while(true) {

		print_line("TESTING: "+location.plus_file("engine.cfg"));
		if (FileAccess::exists(location.plus_file("engine.cfg"))) {

			error->set_text(TTR("Please export outside the project folder!"));
			error->popup_centered_minsize();
			return;
		}
		String nl = (location+"/..").simplify_path();
		if (nl.find("/")==location.find_last("/"))
			break;
		location=nl;
	}

	/* Checked if the export location is outside the project directory,
	 * now will check if a file name has been entered */
	if (p_file.ends_with("/")) {

		error->set_text("Please enter a file name!");
		error->popup_centered_minsize();
		return;
	}

	TreeItem *selected = platforms->get_selected();
	if (!selected)
		return;

	String platform = selected->get_metadata(0);
	bool debugging_enabled = EditorImportExport::get_singleton()->get_export_platform(platform)->is_debugging_enabled();
	Error err = export_platform(platform,p_file,debugging_enabled,file_export_password->get_text(),false);
	if (err!=OK) {
		error->set_text(TTR("Error exporting project!"));
		error->popup_centered_minsize();
	}

}

void ProjectExportDialog::_export_action_pck(const String& p_file) {

	TreeItem *selected = platforms->get_selected();
	if (!selected)
		return;

	Ref<EditorExportPlatform> exporter = EditorImportExport::get_singleton()->get_export_platform(selected->get_metadata(0));
	if (exporter.is_null()) {
		ERR_PRINT("Invalid platform for export of PCK");
		return;
	}

	if (p_file.ends_with(".pck")) {
		FileAccess *f = FileAccess::open(p_file,FileAccess::WRITE);
		if (!f) {
			error->set_text(TTR("Error writing the project PCK!"));
			error->popup_centered_minsize();
		}
		ERR_FAIL_COND(!f);

		Error err = exporter->save_pack(f,false);
		memdelete(f);

		if (err!=OK) {
			error->set_text(TTR("Error exporting project!"));
			error->popup_centered_minsize();
			return;
		}
	} else if (p_file.ends_with(".zip")) {

		Error err = exporter->save_zip(p_file,false);

		if (err!=OK) {
			error->set_text(TTR("Error exporting project!"));
			error->popup_centered_minsize();
			return;
		}
	}
}


Error ProjectExportDialog::export_platform(const String& p_platform, const String& p_path, bool p_debug,const String& p_password, bool p_quit_after) {

	Ref<EditorExportPlatform> exporter = EditorImportExport::get_singleton()->get_export_platform(p_platform);
	if (exporter.is_null()) {
		ERR_PRINT("Invalid platform for export");

		List<StringName> platforms;
		EditorImportExport::get_singleton()->get_export_platforms(&platforms);
		print_line("Valid export plaftorms are:");
		for (List<StringName>::Element *E=platforms.front();E;E=E->next())
			print_line("    \""+E->get()+"\"");

		if (p_quit_after) {
			OS::get_singleton()->set_exit_code(255);
			get_tree()->quit();
		}

		return ERR_INVALID_PARAMETER;
	}
	Error err = exporter->export_project(p_path,p_debug);
	if (err!=OK) {
		error->set_text(TTR("Error exporting project!"));
		error->popup_centered_minsize();
		ERR_PRINT("Exporting failed!");
		if (p_quit_after) {
			OS::get_singleton()->set_exit_code(255);
			get_tree()->quit();
		}
		return ERR_CANT_CREATE;
	} else {
		if (p_quit_after) {
			get_tree()->quit();
		}
	}

	return OK;

}

void ProjectExportDialog::ok_pressed() {
	//export pck
	pck_export->popup_centered_ratio();

}
void ProjectExportDialog::custom_action(const String&) {
	//real export

	TreeItem *selected = platforms->get_selected();
	if (!selected)
		return;

	String platform = selected->get_metadata(0);
	Ref<EditorExportPlatform> exporter = EditorImportExport::get_singleton()->get_export_platform(platform);

	if (exporter.is_null()) {
		error->set_text(vformat(TTR("No exporter for platform '%s' yet."),platform));
		error->popup_centered_minsize();
		return;
	}

	if (platform.to_lower()=="android" && _check_android_setting(exporter)==false){
		// not filled all field for Android release
		return;
	}

	String extension = exporter->get_binary_extension();

	file_export_password->set_editable( exporter->requires_password(exporter->is_debugging_enabled()) );

	file_export->clear_filters();
	if (extension!="") {
		file_export->add_filter("*."+extension);
	}
	file_export->popup_centered_ratio();


}

LineEdit* ProjectExportDialog::_create_keystore_input(Control* container, const String& p_label, const String& name) {

	HBoxContainer* hb=memnew(HBoxContainer);
	Label* lb=memnew(Label);
	LineEdit* input=memnew(LineEdit);

	lb->set_text(p_label);
	lb->set_custom_minimum_size(Size2(140*EDSCALE,0));
	lb->set_align(Label::ALIGN_RIGHT);

	input->set_custom_minimum_size(Size2(170*EDSCALE,0));
	input->set_name(name);

	hb->add_constant_override("separation", 10*EDSCALE);
	hb->add_child(lb);
	hb->add_child(input);
	container->add_child(hb);

	return input;

}

void ProjectExportDialog::_create_android_keystore_window() {

	keystore_file_dialog = memnew( EditorFileDialog );
	add_child(keystore_file_dialog);
	keystore_file_dialog->set_mode(EditorFileDialog::MODE_OPEN_DIR);
	keystore_file_dialog->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	keystore_file_dialog->set_current_dir( "res://" );

	keystore_file_dialog->set_title(TTR("Target Path:"));
	keystore_file_dialog->connect("dir_selected", this,"_keystore_dir_selected");

	keystore_create_dialog=memnew(ConfirmationDialog);
	VBoxContainer* vb=memnew(VBoxContainer);
	vb->set_size(Size2(340*EDSCALE,0));
	keystore_create_dialog->set_title(TTR("Create Android keystore"));

	_create_keystore_input(vb, TTR("Full name"), "name");
	_create_keystore_input(vb, TTR("Organizational unit"), "unit");
	_create_keystore_input(vb, TTR("Organization"), "org");
	_create_keystore_input(vb, TTR("City"), "city");
	_create_keystore_input(vb, TTR("State"), "state");
	_create_keystore_input(vb, TTR("2 letter country code"), "code");
	_create_keystore_input(vb, TTR("User alias"), "alias");
	LineEdit* pass=_create_keystore_input(vb, TTR("Password"), "pass");
	pass->set_placeholder(TTR("at least 6 characters"));
	_create_keystore_input(vb, TTR("File name"), "file");

	Label* lb_path=memnew(Label);
	LineEdit* path=memnew(LineEdit);
	Button* btn=memnew(Button);
	HBoxContainer* hb=memnew(HBoxContainer);

	lb_path->set_text(TTR("Path : (better to save outside of project)"));
	path->set_h_size_flags(SIZE_EXPAND_FILL);
	path->set_name("path");
	btn->set_text(" .. ");
	btn->connect("pressed", keystore_file_dialog, "popup_centered_ratio");

	vb->add_spacer();
	vb->add_child(lb_path);
	hb->add_child(path);
	hb->add_child(btn);
	vb->add_child(hb);

	keystore_create_dialog->add_child(vb);
	keystore_create_dialog->set_child_rect(vb);
	add_child(keystore_create_dialog);

	keystore_create_dialog->connect("confirmed", this, "_create_android_keystore");
	path->connect("text_changed", this, "_check_keystore_path");

	confirm_keystore = memnew(ConfirmationDialog);
	confirm_keystore->connect("confirmed", keystore_create_dialog, "popup_centered_minsize");
	add_child(confirm_keystore);

}

void ProjectExportDialog::_keystore_dir_selected(const String& path) {

	LineEdit* edit=keystore_create_dialog->find_node("path", true, false)->cast_to<LineEdit>();
	edit->set_text(path.simplify_path());

}

void ProjectExportDialog::_keystore_created() {

	if (error->is_connected("popup_hide", this, "_keystore_created")){
		error->disconnect("popup_hide", this, "_keystore_created");
	}
	custom_action("export_pck");

}

void ProjectExportDialog::_check_keystore_path(const String& path) {

	LineEdit* edit=keystore_create_dialog->find_node("path", true, false)->cast_to<LineEdit>();
	bool exists = DirAccess::exists(path);
	if (!exists) {
		edit->add_color_override("font_color", Color(1,0,0,1));
	} else {
		edit->add_color_override("font_color", Color(0,1,0,1));
	}

}

void ProjectExportDialog::_create_android_keystore() {

	Vector<String> names=String("name,unit,org,city,state,code,alias,pass").split(",");
	String path=keystore_create_dialog->find_node("path", true, false)->cast_to<LineEdit>()->get_text();
	String file=keystore_create_dialog->find_node("file", true, false)->cast_to<LineEdit>()->get_text();

	if (file.ends_with(".keystore")==false) {
		file+=".keystore";
	}
	String fullpath=path.plus_file(file);
	String info="CN=$name, OU=$unit, O=$org, L=$city, S=$state, C=$code";
	Dictionary dic;

	for (int i=0;i<names.size();i++){
		LineEdit* edit = keystore_create_dialog->find_node(names[i], true, false)->cast_to<LineEdit>();
		dic[names[i]]=edit->get_text();
		info=info.replace("$"+names[i], edit->get_text());
	}

	String jarsigner=EditorSettings::get_singleton()->get("export/android/jarsigner");
	String keytool=jarsigner.get_base_dir().plus_file("keytool");
	String os_name=OS::get_singleton()->get_name();
	if (os_name.to_lower()=="windows") {
		keytool+=".exe";
	}

	bool exist=FileAccess::exists(keytool);
	if (!exist) {
		error->set_text("Can't find 'keytool'");
		error->popup_centered_minsize();
		return;
	}

	List<String> args;
	args.push_back("-genkey");
	args.push_back("-v");
	args.push_back("-keystore");
	args.push_back(fullpath);
	args.push_back("-alias");
	args.push_back(dic["alias"]);
	args.push_back("-storepass");
	args.push_back(dic["pass"]);
	args.push_back("-keypass");
	args.push_back(dic["pass"]);
	args.push_back("-keyalg");
	args.push_back("RSA");
	args.push_back("-keysize");
	args.push_back("2048");
	args.push_back("-validity");
	args.push_back("10000");
	args.push_back("-dname");
	args.push_back(info);
	int retval;
	OS::get_singleton()->execute(keytool,args,true,NULL,NULL,&retval);

	if (retval==0) { // success
		platform_options->_edit_set("keystore/release", fullpath);
		platform_options->_edit_set("keystore/release_user", dic["alias"]);
		platform_options->_edit_set("keystore/release_password", dic["pass"]);

		error->set_text("Android keystore created at \n"+fullpath);
		error->connect("popup_hide", this, "_keystore_created");
		error->popup_centered_minsize();
	} else { // fail
		error->set_text("Fail to create android keystore at \n"+fullpath);
		error->popup_centered_minsize();
	}

}

bool ProjectExportDialog::_check_android_setting(const Ref<EditorExportPlatform>& exporter) {

	bool is_debugging = exporter->get("debug/debugging_enabled");
	String release = exporter->get("keystore/release");
	String user = exporter->get("keystore/release_user");
	String password = exporter->get("keystore/release_password");

	if (!is_debugging && (release=="" || user=="" || password=="")){
		if (release==""){
			confirm_keystore->set_text(TTR("Release keystore is not set.\nDo you want to create one?"));
			confirm_keystore->popup_centered_minsize();
		} else {
			error->set_text(TTR("Fill Keystore/Release User and Release Password"));
			error->popup_centered_minsize();
		}
		return false;
	}

	return true;

}

void ProjectExportDialog::_group_selected() {


	_update_group(); //?

	_update_group_tree();
}

String ProjectExportDialog::_get_selected_group() {

	TreeItem *sel = groups->get_selected();
	if (!sel)
		return String();

	return sel->get_text(0);


}

void ProjectExportDialog::_update_group_list() {

	String current = _get_selected_group();

	groups->clear();
	List<StringName> grouplist;
	EditorImportExport::get_singleton()->image_export_get_groups(&grouplist);
	grouplist.sort_custom<StringName::AlphCompare>();

	TreeItem *r = groups->create_item();
	for (List<StringName>::Element *E=grouplist.front();E;E=E->next()) {

		TreeItem *ti = groups->create_item(r);
		ti->set_text(0,E->get());
		ti->add_button(0,get_icon("Remove","EditorIcons"));
		if (E->get()==current) {
			ti->select(0);
		}
	}

	_update_group();
}

void ProjectExportDialog::_select_group(const String& p_by_name) {

	TreeItem *c = groups->get_root();
	if (!c)
		return;
	c=c->get_children();

	if (!c)
		return;

	while(c) {

		if (c->get_text(0)==p_by_name) {
			c->select(0);
			_update_group();
			return;
		}
		c=c->get_next();
	}
}

void ProjectExportDialog::_update_group() {

	if (updating)
		return;
	updating=true;


	if (_get_selected_group()=="") {
		group_options->hide();
		//group_del->set_disabled(true);

	} else {
		group_options->show();
		//group_del->set_disabled(false);
		StringName name = _get_selected_group();
		group_image_action->select(EditorImportExport::get_singleton()->image_export_group_get_image_action(name));
		group_atlas->set_pressed(EditorImportExport::get_singleton()->image_export_group_get_make_atlas(name));
		group_shrink->set_value(EditorImportExport::get_singleton()->image_export_group_get_shrink(name));
		group_lossy_quality->set_value(EditorImportExport::get_singleton()->image_export_group_get_lossy_quality(name));
		if (group_atlas->is_pressed())
			atlas_preview->show();
		else
			atlas_preview->hide();

	}

	_update_group_tree();

	updating=false;


}

bool ProjectExportDialog::_update_group_treef(TreeItem *p_parent,EditorFileSystemDirectory *p_dir,const Set<String>& p_extensions,const String& p_groups,const Map<StringName,int>& p_group_index) {

	TreeItem *ti = group_images->create_item(p_parent);
	ti->set_text(0,p_dir->get_name()+"/");
	bool has_child=false;
	for(int i=0;i<p_dir->get_subdir_count();i++) {

		if (_update_group_treef(ti,p_dir->get_subdir(i),p_extensions,p_groups,p_group_index)) {
			has_child=true;
		}
	}

	String filter=group_images_filter->get_text();
	StringName current_group = _get_selected_group();
	String check_text=TTR("Include");

	for(int i=0;i<p_dir->get_file_count();i++) {

		String fname = p_dir->get_file(i);
		if (p_extensions.has(fname.to_lower().extension())) {
			String path = p_dir->get_file_path(i);

			if (filter!=String() && path.find(filter)==-1)
				continue;

			has_child=true;
			TreeItem *file = group_images->create_item(ti);
			file->set_text(0,fname);

			StringName g = EditorImportExport::get_singleton()->image_get_export_group(path);

			if (current_group==g || g==StringName()) {

				file->set_cell_mode(1,TreeItem::CELL_MODE_CHECK);
				file->set_text(1,check_text);
				file->set_editable(1,true);
				file->set_checked(1,current_group==g);
			} else {

				file->set_text(1,g);
				file->set_editable(1,false);
				file->set_selectable(1,false);
			}

			file->set_metadata(0,path);
		}
	}

	if (!has_child) {
		memdelete(ti);
		return false;
	}

	return true;

}
void ProjectExportDialog::_update_group_tree() {

	if (updating)
		return;

	group_images->clear();

	if (_get_selected_group()=="")
		return;

	updating=true;
	print_line("****UGT");
	List<String> img_extensions;
	ImageLoader::get_recognized_extensions(&img_extensions);
	Set<String> extensions;
	for(List<String>::Element *E=img_extensions.front();E;E=E->next()) {

		extensions.insert(E->get());
	}

	List<StringName> grouplist;
	EditorImportExport::get_singleton()->image_export_get_groups(&grouplist);
	grouplist.sort_custom<StringName::AlphCompare>();
	Map<StringName,int> group_index;
	group_index[StringName()]=0;
	int idx=1;
	String groupenum="--";
	for(List<StringName>::Element *E=grouplist.front();E;E=E->next()) {

		group_index[E->get()]=idx++;
		groupenum+=","+String(E->get());
	}

	updating=false;


	_update_group_treef(NULL,EditorFileSystem::get_singleton()->get_filesystem(),extensions,groupenum,group_index);

}

void ProjectExportDialog::_group_changed(Variant v) {

	if (updating)
		return;
	if (_get_selected_group()=="")
		return;
	updating=true;
	StringName name = _get_selected_group();
	EditorNode::get_undo_redo()->create_action(TTR("Change Image Group"));
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_export_group_set_image_action",name,group_image_action->get_selected());
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_export_group_set_make_atlas",name,group_atlas->is_pressed());
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_export_group_set_shrink",name,group_shrink->get_value());
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_export_group_set_lossy_quality",name,group_lossy_quality->get_value());
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_image_action",name,EditorImportExport::get_singleton()->image_export_group_get_image_action(name));
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_make_atlas",name,EditorImportExport::get_singleton()->image_export_group_get_make_atlas(name));
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_shrink",name,EditorImportExport::get_singleton()->image_export_group_get_shrink(name));
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_lossy_quality",name,EditorImportExport::get_singleton()->image_export_group_get_lossy_quality(name));
	EditorNode::get_undo_redo()->add_do_method(this,"_update_group");
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group");
	EditorNode::get_undo_redo()->add_do_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->add_undo_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->commit_action();
	updating=false;
	// update atlas preview button
	_update_group();
}

void ProjectExportDialog::_group_item_edited() {

	TreeItem *item = group_images->get_edited();
	if (!item)
		return;
	if (_get_selected_group()==String())
		return;

	StringName path = item->get_metadata(0);
	String group;
	if (item->is_checked(1)) {
		group=_get_selected_group();
	} else {
		group=String();
	}

	print_line("changed "+path+" to group: "+group);
	EditorNode::get_undo_redo()->create_action(TTR("Change Image Group"));
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_add_to_export_group",path,group);
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_add_to_export_group",path,EditorImportExport::get_singleton()->image_get_export_group(path));
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_do_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_do_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->add_undo_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->commit_action();

}

void ProjectExportDialog::_group_add() {

	String name = group_new_name->get_text();

	if (name=="") {
		group_new_name_error->show();
		group_new_name_error->set_text(TTR("Group name can't be empty!"));
		return;
	}
	if (name.find("/")!=-1 || name.find(":")!=-1 || name.find(",")!=-1 || name.find("-")!=-1) {
		group_new_name_error->set_text(TTR("Invalid character in group name!"));
		group_new_name_error->show();
		return;
	}

	if (EditorImportExport::get_singleton()->image_export_has_group(name)) {
		group_new_name_error->set_text(TTR("Group name already exists!"));
		group_new_name_error->show();
		return;
	}
	group_new_name_error->hide();

	String current=_get_selected_group();


	EditorNode::get_undo_redo()->create_action(TTR("Add Image Group"));
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_export_group_create",name);
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_remove",name);
	EditorNode::get_undo_redo()->add_do_method(this,"_update_group_list");
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group_list");
	EditorNode::get_undo_redo()->add_do_method(this,"_select_group",name);
	if (current!="")
		EditorNode::get_undo_redo()->add_undo_method(this,"_select_group",current);

	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_do_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_do_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->add_undo_method(this,"_save_export_cfg");

	EditorNode::get_undo_redo()->commit_action();

}


void ProjectExportDialog::_group_del(Object *p_item, int p_column, int p_button){

	TreeItem *item = p_item->cast_to<TreeItem>();
	if (!item)
		return;
	String name = item->get_text(0);

	EditorNode::get_undo_redo()->create_action(TTR("Delete Image Group"));
	List<StringName> images_used;
	EditorImportExport::get_singleton()->image_export_get_images_in_group(name,&images_used);
	for (List<StringName>::Element*E=images_used.front();E;E=E->next()) {

		EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_add_to_export_group",E->get(),StringName());

	}
	EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_export_group_remove",name);


	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_create",name);
	for (List<StringName>::Element*E=images_used.front();E;E=E->next()) {

		EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_add_to_export_group",E->get(),name);

	}

	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_image_action",name,EditorImportExport::get_singleton()->image_export_group_get_image_action(name));
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_make_atlas",name,EditorImportExport::get_singleton()->image_export_group_get_make_atlas(name));
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_shrink",name,EditorImportExport::get_singleton()->image_export_group_get_shrink(name));
	EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_export_group_set_lossy_quality",name,EditorImportExport::get_singleton()->image_export_group_get_lossy_quality(name));

	EditorNode::get_undo_redo()->add_do_method(this,"_update_group_list");
	EditorNode::get_undo_redo()->add_do_method(this,"_update_group");
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group_list");
	EditorNode::get_undo_redo()->add_undo_method(this,"_select_group",name);

	EditorNode::get_undo_redo()->add_do_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_do_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->add_undo_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->commit_action();

}

void ProjectExportDialog::_group_select_all() {


	String group = _get_selected_group();
	if (group=="")
		return;

	TreeItem *item = group_images->get_root();
	if (!item)
		return;

	List<StringName> items;
	while(item) {

		if (item->get_cell_mode(1)==TreeItem::CELL_MODE_CHECK && !item->is_checked(1))
			items.push_back(item->get_metadata(0));
		item=item->get_next_visible();
	}


	if (items.size()==0)
		return;

	EditorNode::get_undo_redo()->create_action(TTR("Select All"));

	for (List<StringName>::Element *E=items.front();E;E=E->next()) {

		EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_add_to_export_group",E->get(),group);
		EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_add_to_export_group",E->get(),String());

	}
	EditorNode::get_undo_redo()->add_do_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_do_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->add_undo_method(this,"_save_export_cfg");

	EditorNode::get_undo_redo()->commit_action();

}

void ProjectExportDialog::_group_select_none(){

	String group = _get_selected_group();
	if (group=="")
		return;

	TreeItem *item = group_images->get_root();
	if (!item)
		return;

	List<StringName> items;
	while(item) {

		if (item->get_cell_mode(1)==TreeItem::CELL_MODE_CHECK && item->is_checked(1))
			items.push_back(item->get_metadata(0));
		item=item->get_next_visible();
	}


	if (items.size()==0)
		return;

	EditorNode::get_undo_redo()->create_action(TTR("Select All"));

	for (List<StringName>::Element *E=items.front();E;E=E->next()) {

		EditorNode::get_undo_redo()->add_do_method(EditorImportExport::get_singleton(),"image_add_to_export_group",E->get(),String());
		EditorNode::get_undo_redo()->add_undo_method(EditorImportExport::get_singleton(),"image_add_to_export_group",E->get(),group);

	}
	EditorNode::get_undo_redo()->add_do_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_undo_method(this,"_update_group_tree");
	EditorNode::get_undo_redo()->add_do_method(this,"_save_export_cfg");
	EditorNode::get_undo_redo()->add_undo_method(this,"_save_export_cfg");

	EditorNode::get_undo_redo()->commit_action();

}

void ProjectExportDialog::_group_atlas_preview() {

	StringName group = _get_selected_group();
	if (!group)
		return;

	atlas_preview_frame->set_texture(Ref<Texture>()); //clear previous

	List<StringName> images;
	EditorImportExport::get_singleton()->image_export_get_images_in_group(group,&images);
	images.sort_custom<StringName::AlphCompare>();

	String dst_file = EditorSettings::get_singleton()->get_settings_path()+"/tmp/atlas-preview.tex";
	Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );
	//imd->set_editor();

	for (List<StringName>::Element *F=images.front();F;F=F->next()) {

		imd->add_source(EditorImportPlugin::validate_source_path(F->get()));
	}


	int flags=0;

	if (GlobalConfig::get_singleton()->get("image_loader/filter"))
		flags|=EditorTextureImportPlugin::IMAGE_FLAG_FILTER;
	if (!GlobalConfig::get_singleton()->get("image_loader/gen_mipmaps"))
		flags|=EditorTextureImportPlugin::IMAGE_FLAG_NO_MIPMAPS;
	if (!GlobalConfig::get_singleton()->get("image_loader/repeat"))
		flags|=EditorTextureImportPlugin::IMAGE_FLAG_REPEAT;

	flags|=EditorTextureImportPlugin::IMAGE_FLAG_FIX_BORDER_ALPHA;

	imd->set_option("format",EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS);
	imd->set_option("flags",flags);
	imd->set_option("quality",0.7);
	imd->set_option("atlas",true);
	imd->set_option("crop",true);

	Ref<EditorTextureImportPlugin> plugin = EditorImportExport::get_singleton()->get_import_plugin_by_name("texture");
	Error err = plugin->import2(dst_file,imd,EditorExportPlatform::IMAGE_COMPRESSION_NONE,true);
	if (err) {

		EditorNode::add_io_error(TTR("Error saving atlas:")+" "+dst_file.get_file());
		return;
	}

	Ref<Texture> tex = ResourceLoader::load(dst_file);
	atlas_preview_frame->set_texture(tex); //clear previous
	atlas_preview_dialog->set_title(TTR("Atlas Preview")+" ("+itos(tex->get_width())+"x"+itos(tex->get_height())+")");
	atlas_preview_dialog->popup_centered_ratio(0.9);

}

void ProjectExportDialog::_update_script() {

	if (updating_script)
		return;
	updating_script=true;
	script_mode->select(EditorImportExport::get_singleton()->script_get_action());
	script_key->set_text(EditorImportExport::get_singleton()->script_get_encryption_key());
	updating_script=false;

}

void ProjectExportDialog::_image_filter_changed(String) {

	_update_group_tree();
}

void ProjectExportDialog::_bind_methods() {


	ClassDB::bind_method(_MD("_rescan"),&ProjectExportDialog::_rescan);
	ClassDB::bind_method(_MD("_tree_changed"),&ProjectExportDialog::_tree_changed);
	ClassDB::bind_method(_MD("_scan_finished"),&ProjectExportDialog::_scan_finished);
	ClassDB::bind_method(_MD("_platform_selected"),&ProjectExportDialog::_platform_selected);
	ClassDB::bind_method(_MD("_prop_edited"),&ProjectExportDialog::_prop_edited);
	ClassDB::bind_method(_MD("_export_mode_changed"),&ProjectExportDialog::_export_mode_changed);
	ClassDB::bind_method(_MD("_filters_edited"),&ProjectExportDialog::_filters_edited);
	ClassDB::bind_method(_MD("_filters_exclude_edited"),&ProjectExportDialog::_filters_exclude_edited);
	ClassDB::bind_method(_MD("_export_action"),&ProjectExportDialog::_export_action);
	ClassDB::bind_method(_MD("_export_action_pck"),&ProjectExportDialog::_export_action_pck);
	ClassDB::bind_method(_MD("_quality_edited"),&ProjectExportDialog::_quality_edited);
	ClassDB::bind_method(_MD("_shrink_edited"),&ProjectExportDialog::_shrink_edited);
	ClassDB::bind_method(_MD("_image_export_edited"),&ProjectExportDialog::_image_export_edited);
	ClassDB::bind_method(_MD("_format_toggled"),&ProjectExportDialog::_format_toggled);
	ClassDB::bind_method(_MD("_group_changed"),&ProjectExportDialog::_group_changed);
	ClassDB::bind_method(_MD("_group_add"),&ProjectExportDialog::_group_add);
	ClassDB::bind_method(_MD("_group_del"),&ProjectExportDialog::_group_del);
	ClassDB::bind_method(_MD("_group_selected"),&ProjectExportDialog::_group_selected);
	ClassDB::bind_method(_MD("_update_group"),&ProjectExportDialog::_update_group);
	ClassDB::bind_method(_MD("_update_group_list"),&ProjectExportDialog::_update_group_list);
	ClassDB::bind_method(_MD("_select_group"),&ProjectExportDialog::_select_group);
	ClassDB::bind_method(_MD("_update_group_tree"),&ProjectExportDialog::_update_group_tree);
	ClassDB::bind_method(_MD("_group_item_edited"),&ProjectExportDialog::_group_item_edited);
	ClassDB::bind_method(_MD("_save_export_cfg"),&ProjectExportDialog::_save_export_cfg);
	ClassDB::bind_method(_MD("_image_filter_changed"),&ProjectExportDialog::_image_filter_changed);
	ClassDB::bind_method(_MD("_group_atlas_preview"),&ProjectExportDialog::_group_atlas_preview);
	ClassDB::bind_method(_MD("_group_select_all"),&ProjectExportDialog::_group_select_all);
	ClassDB::bind_method(_MD("_group_select_none"),&ProjectExportDialog::_group_select_none);
	ClassDB::bind_method(_MD("_script_edited"),&ProjectExportDialog::_script_edited);
	ClassDB::bind_method(_MD("_update_script"),&ProjectExportDialog::_update_script);
	ClassDB::bind_method(_MD("_sample_convert_edited"),&ProjectExportDialog::_sample_convert_edited);


	ClassDB::bind_method(_MD("export_platform"),&ProjectExportDialog::export_platform);
	ClassDB::bind_method(_MD("_create_android_keystore"),&ProjectExportDialog::_create_android_keystore);
	ClassDB::bind_method(_MD("_check_keystore_path"),&ProjectExportDialog::_check_keystore_path);
	ClassDB::bind_method(_MD("_keystore_dir_selected"),&ProjectExportDialog::_keystore_dir_selected);
	ClassDB::bind_method(_MD("_keystore_created"),&ProjectExportDialog::_keystore_created);


//	ADD_SIGNAL(MethodInfo("instance"));
//	ADD_SIGNAL(MethodInfo("open"));

}


ProjectExportDialog::ProjectExportDialog(EditorNode *p_editor) {

	editor=p_editor;
	set_title(TTR("Project Export Settings"));

	sections = memnew( TabContainer );
	add_child(sections);
	set_child_rect(sections);

	VBoxContainer *pvbox = memnew( VBoxContainer );
	sections->add_child(pvbox);
	pvbox->set_name(TTR("Target"));

	HBoxContainer *phbox = memnew( HBoxContainer );
	pvbox->add_child(phbox);
	phbox->set_v_size_flags(SIZE_EXPAND_FILL);

	plat_errors = memnew( HBoxContainer );
	pvbox->add_child(plat_errors);
	platform_error_string = memnew( Label );
	platform_error_string->set_h_size_flags(SIZE_EXPAND_FILL);
	plat_errors->add_child(platform_error_string);

	VBoxContainer *vb = memnew( VBoxContainer );
	vb->set_h_size_flags(SIZE_EXPAND_FILL);
	vb->set_v_size_flags(SIZE_EXPAND_FILL);
	phbox->add_child(vb);
	platforms = memnew( Tree );
	platforms->set_hide_root(true);
	vb->add_margin_child(TTR("Export to Platform"),platforms,true);

	platforms->connect("cell_selected",this,"_platform_selected");


	vb = memnew(VBoxContainer );
	phbox->add_child(vb);
	vb->set_h_size_flags(SIZE_EXPAND_FILL);
	vb->set_v_size_flags(SIZE_EXPAND_FILL);
	platform_options = memnew( PropertyEditor() );
	platform_options->hide_top_label();
	vb->add_margin_child(TTR("Options"),platform_options,true);
	platform_options->connect("property_edited",this,"_prop_edited");



	//////////////////

	vb = memnew( VBoxContainer );
	vb->set_name(TTR("Resources"));
	sections->add_child(vb);

	export_mode = memnew( OptionButton );
	export_mode->add_item(TTR("Export selected resources (including dependencies)."));
	export_mode->add_item(TTR("Export all resources in the project."));
	export_mode->add_item(TTR("Export all files in the project directory."));
	export_mode->connect("item_selected",this,"_export_mode_changed");

	vb->add_margin_child(TTR("Export Mode:"),export_mode);



	tree_vb = memnew( VBoxContainer );
	vb->add_child(tree_vb);
	tree_vb->set_v_size_flags(SIZE_EXPAND_FILL);

	tree = memnew( Tree );
	tree_vb->add_margin_child(TTR("Resources to Export:"),tree,true);

	tree->set_v_size_flags(SIZE_EXPAND_FILL);
	tree->connect("item_edited",this,"_tree_changed");
	tree->set_columns(2);
	tree->set_column_titles_visible(true);
	tree->set_column_title(0,TTR("File"));
	tree->set_column_title(1,TTR("Action"));
	tree->set_column_expand(1,false);
	tree->set_column_min_width(1,90);

	filters = memnew( LineEdit );
	vb->add_margin_child(TTR("Filters to export non-resource files (comma-separated, e.g.: *.json, *.txt):"),filters);
	filters->connect("text_changed",this,"_filters_edited");
	filters_exclude = memnew( LineEdit );
	vb->add_margin_child(TTR("Filters to exclude from export (comma-separated, e.g.: *.json, *.txt):"),filters_exclude);
	filters_exclude->connect("text_changed",this,"_filters_exclude_edited");

	convert_text_scenes = memnew( CheckButton );
	convert_text_scenes->set_text(TTR("Convert text scenes to binary on export."));
	vb->add_child(convert_text_scenes);
	convert_text_scenes->connect("toggled",this,"_export_mode_changed");

	image_vb = memnew( VBoxContainer );
	image_vb->set_name(TTR("Images"));
	image_action = memnew( OptionButton );
	image_action->add_item(TTR("Keep Original"));
	image_action->add_item(TTR("Compress for Disk (Lossy, WebP)"));
	image_action->add_item(TTR("Compress for RAM (BC/PVRTC/ETC)"));
	image_vb->add_margin_child(TTR("Convert Images (*.png):"),image_action);
	HBoxContainer *qhb = memnew( HBoxContainer );
	image_quality = memnew( HSlider );
	qhb->add_child(image_quality);
	image_quality->set_h_size_flags(SIZE_EXPAND_FILL);
	SpinBox *qspin = memnew( SpinBox );
	image_quality->share(qspin);
	qhb->add_child(qspin);
	image_quality->set_min(0);
	image_quality->set_max(1);
	image_quality->set_step(0.01);
	image_vb->add_margin_child(TTR("Compress for Disk (Lossy) Quality:"),qhb);
	image_shrink = memnew( SpinBox );
	image_shrink->set_min(1);
	image_shrink->set_max(8);
	image_shrink->set_step(0.1);
	image_vb->add_margin_child(TTR("Shrink All Images:"),image_shrink);
	sections->add_child(image_vb);

	image_formats=memnew(Tree);
	image_formats->set_hide_root(true);
	TreeItem *root = image_formats->create_item(NULL);
	List<String> fmts;
	ImageLoader::get_recognized_extensions(&fmts);
	for(List<String>::Element *E=fmts.front();E;E=E->next()) {

		TreeItem *fmt = image_formats->create_item(root);
		fmt->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		fmt->set_text(0,E->get());
		fmt->set_editable(0,true);
		formats.push_back(fmt);
	}
	image_vb->add_margin_child(TTR("Compress Formats:")+" ",image_formats,true);

	/// groups
	HBoxContainer *group_hb = memnew( HBoxContainer );
	group_hb->set_name(TTR("Image Groups"));
	sections->add_child(group_hb);
	VBoxContainer *group_vb_left = memnew( VBoxContainer);
	group_hb->add_child(group_vb_left);

	VBoxContainer *gvb = memnew(VBoxContainer);
	HBoxContainer *ghb = memnew(HBoxContainer);
	gvb->add_child(ghb);

	group_new_name = memnew( LineEdit );
	group_new_name->set_h_size_flags(SIZE_EXPAND_FILL);
	ghb->add_child(group_new_name);

	group_add = memnew(ToolButton);
	group_add->connect("pressed",this,"_group_add");
	ghb->add_child(group_add);

	group_new_name_error = memnew( Label );
	group_new_name_error->add_color_override("font_color",Color(1,0.4,0.4));
	gvb->add_child(group_new_name_error);
	group_new_name_error->hide();

	groups=memnew(Tree);
	groups->set_v_size_flags(SIZE_EXPAND_FILL);
	groups->connect("cell_selected",this,"_group_selected",varray(),CONNECT_DEFERRED);
	groups->connect("button_pressed",this,"_group_del",varray(),CONNECT_DEFERRED);
	groups->set_hide_root(true);
	gvb->add_child(groups);

	group_vb_left->add_margin_child(TTR("Groups:"),gvb,true);
	//group_vb_left->add_child( memnew( HSeparator));
	group_options = memnew(VBoxContainer);
	group_vb_left->add_child(group_options);


	group_image_action = memnew(OptionButton);
	group_image_action->add_item(TTR("Default"));
	group_image_action->add_item(TTR("Compress Disk"));
	group_image_action->add_item(TTR("Compress RAM"));
	group_image_action->add_item(TTR("Keep Original"));
	group_options->add_margin_child(TTR("Compress Mode:"),group_image_action);
	group_image_action->connect("item_selected",this,"_group_changed");

	group_lossy_quality = memnew( HSlider );
	group_lossy_quality->set_min(0.1);
	group_lossy_quality->set_max(1.0);
	group_lossy_quality->set_step(0.01);
	group_lossy_quality->set_value(0.7);
	group_lossy_quality->connect("value_changed",this,"_quality_edited");

	HBoxContainer *gqhb = memnew( HBoxContainer );
	SpinBox *gqspin = memnew( SpinBox );
	group_lossy_quality->share(gqspin);
	group_lossy_quality->set_h_size_flags(SIZE_EXPAND_FILL);
	gqhb->add_child(group_lossy_quality);
	gqhb->add_child(gqspin);
	group_options->add_margin_child(TTR("Lossy Quality:"),gqhb);

	group_atlas = memnew(CheckButton);
	group_atlas->set_pressed(true);
	group_options->add_margin_child(TTR("Atlas:"),group_atlas);
	group_atlas->connect("toggled",this,"_group_changed");

	group_shrink = memnew(SpinBox);
	group_shrink->set_min(1);
	group_shrink->set_max(8);
	group_shrink->set_value(1);
	group_shrink->set_step(0.001);
	group_options->add_margin_child(TTR("Shrink By:"),group_shrink);
	group_shrink->connect("value_changed",this,"_group_changed");

	atlas_preview = memnew( Button );
	atlas_preview->set_text(TTR("Preview Atlas"));
	group_options->add_child(atlas_preview);
	atlas_preview->show();
	atlas_preview->connect("pressed",this,"_group_atlas_preview");
	Control *ec = memnew(Control );
	ec->set_custom_minimum_size(Size2(150,1));
	gvb->add_child(ec);

	VBoxContainer *group_vb_right = memnew( VBoxContainer );
	group_hb->add_child(group_vb_right);
	group_vb_right->set_h_size_flags(SIZE_EXPAND_FILL);

	HBoxContainer *filter_hb = memnew (HBoxContainer);

	group_images_filter = memnew( LineEdit );
	group_vb_right->add_margin_child(TTR("Image Filter:"),filter_hb);
	filter_hb->add_child(group_images_filter);
	group_images_filter->set_h_size_flags(SIZE_EXPAND_FILL);
	group_images_filter->connect("text_changed",this,"_image_filter_changed");
	group_images = memnew( Tree );
	group_images->set_v_size_flags(SIZE_EXPAND_FILL);
	group_vb_right->add_margin_child(TTR("Images:"),group_images,true);

	Button *filt_select_all = memnew( Button );
	filt_select_all->set_text(TTR("Select All"));
	filter_hb->add_child(filt_select_all);
	filt_select_all->connect("pressed",this,"_group_select_all");

	Button *filt_select_none = memnew( Button );
	filt_select_none->set_text(TTR("Select None"));
	filter_hb->add_child(filt_select_none);
	filt_select_none->connect("pressed",this,"_group_select_none");

	atlas_preview_dialog = memnew( AcceptDialog );
	ScrollContainer *scroll = memnew( ScrollContainer );
	atlas_preview_dialog->add_child(scroll);
	atlas_preview_dialog->set_child_rect(scroll);
	atlas_preview_frame = memnew( TextureFrame );
	scroll->add_child(atlas_preview_frame);
	add_child(atlas_preview_dialog);


	group_images->set_hide_root(true);
	group_images->set_columns(2);
	group_images->set_column_expand(0,true);
	group_images->set_column_expand(1,false);
	group_images->set_column_min_width(1,100);
	group_images->set_column_titles_visible(true);
	group_images->set_column_title(0,TTR("Images"));
	group_images->set_column_title(1,TTR("Group"));
	group_images->connect("item_edited",this,"_group_item_edited",varray(),CONNECT_DEFERRED);

/*	SpinBox *group_shrink;
	CheckButton *group_atlas;
	OptionButton *group_image_action;*/


/*	progress = memnew( Label );
	add_child(progress);
	progress->set_area_as_parent_rect();
	progress->set_anchor_and_margin(MARGIN_TOP,Control::ANCHOR_BEGIN,25);
	progress->hide();
	progress->set_align(Label::ALIGN_CENTER);*/

/*
	button_reload = memnew( Button );
	button_reload->set_pos(Point2(3,2));
	button_reload->set_size(Point2(20,5));
	button_reload->set_flat(true);
	//add_child(button_reload);
	button_reload->connect("pressed",this,"_rescan");
	hbc->add_child(button_reload);
*/


	sample_vbox = memnew( VBoxContainer );
	sample_vbox->set_name(TTR("Samples"));
	sections->add_child(sample_vbox);
	sample_mode = memnew( OptionButton );
	sample_vbox->add_margin_child(TTR("Sample Conversion Mode: (.wav files):"),sample_mode);
	sample_mode->add_item(TTR("Keep"));
	sample_mode->add_item(TTR("Compress (RAM - IMA-ADPCM)"));
	sample_max_hz = memnew( SpinBox );
	sample_max_hz->set_max(192000);
	sample_max_hz->set_min(8000);
	sample_vbox->add_margin_child(TTR("Sampling Rate Limit (Hz):"),sample_max_hz);
	sample_trim = memnew( CheckButton );
	sample_trim->set_text(TTR("Trim"));
	sample_vbox->add_margin_child(TTR("Trailing Silence:"),sample_trim);

	script_vbox = memnew( VBoxContainer );
	script_vbox->set_name(TTR("Script"));
	sections->add_child(script_vbox);
	script_mode = memnew( OptionButton );
	script_vbox->add_margin_child(TTR("Script Export Mode:"),script_mode);
	script_mode->add_item(TTR("Text"));
	script_mode->add_item(TTR("Compiled"));
	script_mode->add_item(TTR("Encrypted (Provide Key Below)"));
	script_key = memnew( LineEdit );
	script_vbox->add_margin_child(TTR("Script Encryption Key (256-bits as hex):"),script_key);



	updating=false;

	error = memnew( AcceptDialog );
	add_child(error);

	confirm = memnew( ConfirmationDialog );
	add_child(confirm);
	confirm->connect("confirmed",this,"_confirmed");

	get_ok()->set_text(TTR("Export PCK/Zip"));


	expopt="--,Export,Bundle";

	file_export = memnew( EditorFileDialog );
	add_child(file_export);
	file_export->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	file_export->set_current_dir( EditorSettings::get_singleton()->get("filesystem/directories/default_project_export_path") );

	file_export->set_title(TTR("Export Project"));
	file_export->connect("file_selected", this,"_export_action");

	file_export_password = memnew( LineEdit );
	file_export_password->set_secret(true);
	file_export_password->set_editable(false);
	file_export->get_vbox()->add_margin_child(TTR("Password:"),file_export_password);

	pck_export = memnew( EditorFileDialog );
	pck_export->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	pck_export->set_current_dir( EditorSettings::get_singleton()->get("filesystem/directories/default_project_export_path") );
	pck_export->set_title(TTR("Export Project PCK"));
	pck_export->connect("file_selected", this,"_export_action_pck");
	pck_export->add_filter("*.pck ; Data Pack");
	pck_export->add_filter("*.zip ; Zip");
	add_child(pck_export);

	button_export = add_button(TTR("Export.."),!OS::get_singleton()->get_swap_ok_cancel(),"export_pck");
	updating_script=false;

	ei="EditorIcons";
	ot="Object";
	pending_update_tree=true;

	_create_android_keystore_window();
}


ProjectExportDialog::~ProjectExportDialog() {


}

void ProjectExport::popup_export() {

	Set<String> presets;
	presets.insert("default");

	List<PropertyInfo> pi;
	GlobalConfig::get_singleton()->get_property_list(&pi);
	export_preset->clear();

	for (List<PropertyInfo>::Element *E=pi.front();E;E=E->next()) {

		if (!E->get().name.begins_with("export_presets/"))
			continue;
		presets.insert(E->get().name.get_slice("/",1));
	}

	for(Set<String>::Element *E=presets.front();E;E=E->next()) {

		export_preset->add_item(E->get());
	}



	popup_centered(Size2(300,100));



}
Error ProjectExport::export_project(const String& p_preset) {

	return OK;

#if 0

	String selected=p_preset;

	DVector<String> preset_settings = GlobalConfig::get_singleton()->get("export_presets/"+selected);
	String preset_path=GlobalConfig::get_singleton()->get("export_presets_path/"+selected);
	if (preset_path=="") {

		error->set_text("Export path empty, see export options");
		error->popup_centered_minsize(Size2(300,100));
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	int pc=preset_settings.size();

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	if (d->change_dir(preset_path)!=OK) {

		memdelete(d);
		error->set_text("Can't access to export path:\n "+preset_path);
		error->popup_centered_minsize(Size2(300,100));
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	if (pc==0) {
		memdelete(d);
		return OK;
	}
	if (pc%3 != 0 ) {
		memdelete(d);
		error->set_text("Corrupted export data..");
		error->popup_centered_minsize(Size2(300,100));
		ERR_EXPLAIN("Corrupted export data...");
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	Map<String,ProjectExportSettings::ItemData> export_action;


	Map<String,Map<String,String> > remapped_paths;

	Set<String> scene_extensions;
	Set<String> resource_extensions;

	{

		List<String> l;
//		SceneLoader::get_recognized_extensions(&l);
//		for(List<String>::Element *E=l.front();E;E=E->next()) {
//
//			scene_extensions.insert(E->get());
//		}
		ResourceLoader::get_recognized_extensions_for_type("",&l);
		for(List<String>::Element *E=l.front();E;E=E->next()) {

			resource_extensions.insert(E->get());
		}
	}

	Vector<String> names = GlobalConfig::get_singleton()->get_optimizer_presets();

	//prepare base paths

	for(int i=0;i<pc;i+=3) {


		String name = preset_settings[i+0];
		String pname=preset_settings[i+1];
		String deps=preset_settings[i+2];
		int idx=1;
		if (pname=="") {
			pname="copy";
		} else {

			for(int j=0;j<names.size();j++) {
				if (pname==names[j]) {
					idx=j+2;
					break;
				}
			}
		}

		int dep_idx=0;

		for(int j=0;j<ProjectExportSettings::DA_MAX;j++) {
			if (ProjectExportSettings::da_string[j]==deps) {
				dep_idx=j;
				break;
			}
		}

		if (idx>=0) {
			export_action[name].action=idx;
			export_action[name].depaction=dep_idx;
		}

	}


	Set<String> bundle_exceptions;
	for (Map<String,ProjectExportSettings::ItemData>::Element *E=export_action.front();E;E=E->next()) {
		bundle_exceptions.insert(E->key());
	}


	{

		// find dependencies and add them to export

		Map<String,ProjectExportSettings::ItemData> dependencies;

		for (Map<String,ProjectExportSettings::ItemData>::Element *E=export_action.front();E;E=E->next()) {

			ProjectExportSettings::ItemData &id=E->get();

			if (id.depaction!=ProjectExportSettings::DA_COPY && id.depaction!=ProjectExportSettings::DA_OPTIMIZE)
				continue; //no copy or optimize, go on
			List<String> deplist;
			ResourceLoader::get_dependencies(E->key(),&deplist);

			while (deplist.size()) {

				String dependency = deplist.front()->get();
				deplist.pop_front();
				if (export_action.has(dependency))
					continue; //taged to export, will not override
				if (dependencies.has(dependency)) {

					if (id.action <= dependencies[dependency].action )
						continue;
				}

				ProjectExportSettings::ItemData depid;
				if (id.depaction==ProjectExportSettings::DA_COPY || id.action==ProjectExportSettings::DA_COPY)
					depid.action=ProjectExportSettings::DA_COPY;
				else if (id.depaction==ProjectExportSettings::DA_OPTIMIZE)
					depid.action=id.action;
				depid.depaction=0;

				dependencies[dependency]=depid;

				ResourceLoader::get_dependencies(dependency,&deplist);
			}


		}

		for (Map<String,ProjectExportSettings::ItemData>::Element *E=dependencies.front();E;E=E->next()) {
			export_action[E->key()]=E->get();
		}
	}



	int idx=0;
	for (Map<String,ProjectExportSettings::ItemData>::Element *E=export_action.front();E;E=E->next(),idx++) {


		String path=E->key();
		if (E->get().action==0)
			continue; //nothing to do here
		String preset;
		if (E->get().action==1)
			preset="";
		else
			preset=names[E->get().action-2];

		print_line("Exporting "+itos(idx)+"/"+itos(export_action.size())+": "+path);

		String base_dir = GlobalConfig::get_singleton()->localize_path(path.get_base_dir()).replace("\\","/").replace("res://","");
		DirAccess *da=DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		String cwd = d->get_current_dir();
		da->change_dir(cwd);
		print_line("base dir: "+base_dir);
		String remap_platform="all";

		for(int j=0;j<base_dir.get_slice_count("/");j++) {

			String p = base_dir.get_slice("/",j);
			if (da->change_dir(p)!=OK) {

				Error err = da->make_dir(p);
				if (err!=OK) {
					memdelete(da);
					memdelete(d);
					ERR_EXPLAIN("Cannot make dir: "+cwd+"/"+p);
					ERR_FAIL_V(ERR_CANT_CREATE);
				}

				if (da->change_dir(p)!=OK) {

					memdelete(da);
					memdelete(d);
					ERR_EXPLAIN("Cannot change to dir: "+cwd+"/"+p);
					ERR_FAIL_V(ERR_CANT_CREATE);
				}

			}

			cwd=da->get_current_dir();
		}

		memdelete(da);
		//cwd is the target dir

		String source_file;

		print_line("Exporting: "+source_file);
		bool delete_source=false;
		if (preset=="") {
			//just copy!

			source_file=path;
			delete_source=false;
		} else {

			delete_source=true;
			//create an optimized source file

			if (!GlobalConfig::get_singleton()->has("optimizer_presets/"+preset)) {
				memdelete(d);
				ERR_EXPLAIN("Unknown optimizer preset: "+preset);
				ERR_FAIL_V(ERR_INVALID_DATA);
			}


			Dictionary dc = GlobalConfig::get_singleton()->get("optimizer_presets/"+preset);

			ERR_FAIL_COND_V(!dc.has("__type__"),ERR_INVALID_DATA);
			String type=dc["__type__"];

			Ref<EditorOptimizedSaver> saver;

			for(int i=0;i<editor_data->get_optimized_saver_count();i++) {

				if (editor_data->get_optimized_saver(i)->get_target_name()==type) {
					saver=editor_data->get_optimized_saver(i);
				}
			}

			if (saver.is_null()) {
				memdelete(d);
				ERR_EXPLAIN("Preset '"+preset+"' references nonexistent saver: "+type);
				ERR_FAIL_COND_V(saver.is_null(),ERR_INVALID_DATA);
			}

			List<Variant> keys;
			dc.get_key_list(&keys);

			saver->clear();

			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				saver->set(E->get(),dc[E->get()]);
			}


			remap_platform=saver->get_target_platform();
			if (remap_platform=="")
				remap_platform="all";


			if (resource_extensions.has(path.extension().to_lower())) {

				uint32_t flags=0;

//				if (saver->is_bundle_scenes_enabled())
//					flags|=Reso::FLAG_BUNDLE_INSTANCED_SCENES;
				saver->set_bundle_exceptions(NULL);
				if (E->get().depaction>=ProjectExportSettings::DA_BUNDLE) {
					flags|=ResourceSaver::FLAG_BUNDLE_RESOURCES;
					if (E->get().depaction==ProjectExportSettings::DA_BUNDLE)
						saver->set_bundle_exceptions(&bundle_exceptions);

				}

				if (saver->is_remove_editor_data_enabled())
					flags|=ResourceSaver::FLAG_OMIT_EDITOR_PROPERTIES;
				if (saver->is_big_endian_data_enabled())
					flags|=ResourceSaver::FLAG_SAVE_BIG_ENDIAN;

				RES res = ResourceLoader::load(path);

				if (res.is_null()) {

					memdelete(d);
					ERR_EXPLAIN("Error loading resource to optimize: "+path);
					ERR_FAIL_V(ERR_INVALID_DATA);
				}

				if (saver->is_compress_translations_enabled() && res->get_type()=="Translation") {

					Ref<PHashTranslation> ct = Ref<PHashTranslation>( memnew( PHashTranslation ) );
					ct->generate(res);
					res=ct;
				}


				//dst_file=path.get_file();
				//dst_file = cwd+"/"+dst_file.substr(0,dst_file.length()-dst_file.extension().length())+"opt.scn";

				//String write_file = path.substr(0,path.length()-path.extension().length())+"optimized.res";
				String write_file = path+".opt.res";


				print_line("DST RES FILE: "+write_file);
				Error err = ResourceSaver::save(write_file,res,flags,saver);
				if (err) {
					memdelete(d);
					ERR_EXPLAIN("Error saving optimized resource: "+write_file);
					ERR_FAIL_COND_V(err,ERR_CANT_OPEN);
				}
				source_file=write_file;
	//			project_settings->add_remapped_path(src_scene,path,platform);

			}


		}

		String dst_file;
		dst_file=cwd+"/"+source_file.get_file();
		print_line("copying from: "+source_file);
		print_line("copying to: "+dst_file);
		Error err = d->copy(source_file,dst_file);

		if (delete_source)
			d->remove(source_file);

		if (err) {


			ERR_EXPLAIN("Error copying from: "+source_file+" to "+dst_file+".");
			ERR_FAIL_COND_V(err,err);
		}

		String src_remap=path;
		String dst_remap=source_file;
		print_line("remap from: "+src_remap);
		print_line("remap to: "+dst_remap);
		if (src_remap!=dst_remap) {


			remapped_paths[remap_platform][src_remap]=dst_remap;
		}

		//do the copy man...

	}

	Map<String,Variant> added_settings;


	for (Map<String,Map<String,String> >::Element *E=remapped_paths.front();E;E=E->next()) {

		String platform=E->key();
		DVector<String> remaps;
		for(Map<String,String>::Element *F=E->get().front();F;F=F->next() ) {

			remaps.push_back(F->key());
			remaps.push_back(F->get());
		}



//		added_settings["remap/"+platform]=remaps;`
		added_settings["remap/"+platform]=Variant(remaps).operator Array();
	}

	String engine_cfg_path=d->get_current_dir()+"/engine.cfg";
	print_line("enginecfg: "+engine_cfg_path);
	GlobalConfig::get_singleton()->save_custom(engine_cfg_path,added_settings);

	memdelete(d);
	return OK;
#endif
}

ProjectExport::ProjectExport(EditorData* p_data) {

	editor_data=p_data;
	VBoxContainer *vbc = memnew( VBoxContainer );
	add_child(vbc);
	set_child_rect(vbc);
	set_title(TTR("Project Export"));
	label = memnew( Label );
	label->set_text(TTR("Export Preset:"));
	vbc->add_child(label);
	export_preset = memnew (OptionButton);
	vbc->add_child(export_preset);
	get_ok()->set_text(TTR("Export"));
	set_hide_on_ok(false);
	error = memnew( AcceptDialog );
	add_child(error);


}
