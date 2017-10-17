/*************************************************************************/
/*  export_template_manager.cpp                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "export_template_manager.h"

#include "editor_node.h"
#include "editor_scale.h"
#include "io/zip_io.h"
#include "os/dir_access.h"
#include "version.h"

void ExportTemplateManager::_update_template_list() {

	while (current_hb->get_child_count()) {
		memdelete(current_hb->get_child(0));
	}

	while (installed_vb->get_child_count()) {
		memdelete(installed_vb->get_child(0));
	}

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->change_dir(EditorSettings::get_singleton()->get_settings_path().plus_file("templates"));

	d->list_dir_begin();
	Set<String> templates;

	if (err == OK) {

		bool isdir;
		String c = d->get_next(&isdir);
		while (c != String()) {
			if (isdir && !c.begins_with(".")) {
				templates.insert(c);
			}
			c = d->get_next(&isdir);
		}
	}
	d->list_dir_end();

	memdelete(d);

	String current_version = itos(VERSION_MAJOR) + "." + itos(VERSION_MINOR) + "-" + _MKSTR(VERSION_STATUS);

	Label *current = memnew(Label);
	current->set_h_size_flags(SIZE_EXPAND_FILL);
	current_hb->add_child(current);

	if (templates.has(current_version)) {
		current->add_color_override("font_color", get_color("success_color", "Editor"));
		Button *redownload = memnew(Button);
		redownload->set_text(TTR("Re-Download"));
		current_hb->add_child(redownload);
		redownload->connect("pressed", this, "_download_template", varray(current_version));

		Button *uninstall = memnew(Button);
		uninstall->set_text(TTR("Uninstall"));
		current_hb->add_child(uninstall);
		current->set_text(current_version + " " + TTR("(Installed)"));
		uninstall->connect("pressed", this, "_uninstall_template", varray(current_version));

	} else {
		current->add_color_override("font_color", get_color("error_color", "Editor"));
		Button *redownload = memnew(Button);
		redownload->set_text(TTR("Download"));
		redownload->connect("pressed", this, "_download_template", varray(current_version));
		current_hb->add_child(redownload);
		current->set_text(current_version + " " + TTR("(Missing)"));
	}

	for (Set<String>::Element *E = templates.back(); E; E = E->prev()) {

		HBoxContainer *hbc = memnew(HBoxContainer);
		Label *version = memnew(Label);
		version->set_modulate(get_color("disabled_font_color", "Editor"));
		String text = E->get();
		if (text == current_version) {
			text += " " + TTR("(Current)");
		}
		version->set_text(text);
		version->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(version);

		Button *uninstall = memnew(Button);

		uninstall->set_text(TTR("Uninstall"));
		hbc->add_child(uninstall);
		uninstall->connect("pressed", this, "_uninstall_template", varray(E->get()));

		installed_vb->add_child(hbc);
	}
}

void ExportTemplateManager::_download_template(const String &p_version) {

	print_line("download " + p_version);
}

void ExportTemplateManager::_uninstall_template(const String &p_version) {

	remove_confirm->set_text(vformat(TTR("Remove template version '%s'?"), p_version));
	remove_confirm->popup_centered_minsize();
	to_remove = p_version;
}

void ExportTemplateManager::_uninstall_template_confirm() {

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->change_dir(EditorSettings::get_singleton()->get_settings_path().plus_file("templates"));

	ERR_FAIL_COND(err != OK);

	err = d->change_dir(to_remove);

	ERR_FAIL_COND(err != OK);

	Vector<String> files;

	d->list_dir_begin();

	bool isdir;
	String c = d->get_next(&isdir);
	while (c != String()) {
		if (!isdir) {
			files.push_back(c);
		}
		c = d->get_next(&isdir);
	}

	d->list_dir_end();

	for (int i = 0; i < files.size(); i++) {
		d->remove(files[i]);
	}

	d->change_dir("..");
	d->remove(to_remove);

	_update_template_list();
}

void ExportTemplateManager::_install_from_file(const String &p_file) {

	FileAccess *fa = NULL;
	zlib_filefunc_def io = zipio_create_io_from_file(&fa);

	unzFile pkg = unzOpen2(p_file.utf8().get_data(), &io);
	if (!pkg) {

		EditorNode::get_singleton()->show_warning(TTR("Can't open export templates zip."));
		return;
	}
	int ret = unzGoToFirstFile(pkg);

	int fc = 0; //count them and find version
	String version;

	while (ret == UNZ_OK) {

		unz_file_info info;
		char fname[16384];
		ret = unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

		String file = fname;

		if (file.ends_with("version.txt")) {

			Vector<uint8_t> data;
			data.resize(info.uncompressed_size);

			//read
			unzOpenCurrentFile(pkg);
			ret = unzReadCurrentFile(pkg, data.ptr(), data.size());
			unzCloseCurrentFile(pkg);

			String data_str;
			data_str.parse_utf8((const char *)data.ptr(), data.size());
			data_str = data_str.strip_edges();

			if (data_str.get_slice_count("-") != 2 || data_str.get_slice_count(".") != 2) {
				EditorNode::get_singleton()->show_warning(TTR("Invalid version.txt format inside templates."));
				unzClose(pkg);
				return;
			}

			String ver = data_str.get_slice("-", 0);

			int major = ver.get_slice(".", 0).to_int();
			int minor = ver.get_slice(".", 1).to_int();
			String rev = data_str.get_slice("-", 1);

			if (!rev.is_valid_identifier()) {
				EditorNode::get_singleton()->show_warning(TTR("Invalid version.txt format inside templates. Revision is not a valid identifier."));
				unzClose(pkg);
				return;
			}

			version = itos(major) + "." + itos(minor) + "-" + rev;
		}

		fc++;
		ret = unzGoToNextFile(pkg);
	}

	if (version == String()) {
		EditorNode::get_singleton()->show_warning(TTR("No version.txt found inside templates."));
		unzClose(pkg);
		return;
	}

	String template_path = EditorSettings::get_singleton()->get_settings_path().plus_file("templates").plus_file(version);

	DirAccess *d = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	Error err = d->make_dir_recursive(template_path);
	if (err != OK) {
		EditorNode::get_singleton()->show_warning(TTR("Error creating path for templates:\n") + template_path);
		unzClose(pkg);
		return;
	}

	memdelete(d);

	ret = unzGoToFirstFile(pkg);

	EditorProgress p("ltask", TTR("Extracting Export Templates"), fc);

	fc = 0;

	while (ret == UNZ_OK) {

		//get filename
		unz_file_info info;
		char fname[16384];
		unzGetCurrentFileInfo(pkg, &info, fname, 16384, NULL, 0, NULL, 0);

		String file = fname;

		Vector<uint8_t> data;
		data.resize(info.uncompressed_size);

		//read
		unzOpenCurrentFile(pkg);
		unzReadCurrentFile(pkg, data.ptr(), data.size());
		unzCloseCurrentFile(pkg);

		print_line(fname);
		/*
		for(int i=0;i<512;i++) {
			print_line(itos(data[i]));
		}
		*/

		file = file.get_file();

		p.step(TTR("Importing:") + " " + file, fc);

		FileAccess *f = FileAccess::open(template_path.plus_file(file), FileAccess::WRITE);

		ERR_CONTINUE(!f);
		f->store_buffer(data.ptr(), data.size());

		memdelete(f);

		ret = unzGoToNextFile(pkg);
		fc++;
	}

	unzClose(pkg);

	_update_template_list();
}

void ExportTemplateManager::popup_manager() {

	_update_template_list();
	popup_centered_minsize(Size2(400, 400) * EDSCALE);
}

void ExportTemplateManager::ok_pressed() {

	template_open->popup_centered_ratio();
}

void ExportTemplateManager::_bind_methods() {

	ClassDB::bind_method("_download_template", &ExportTemplateManager::_download_template);
	ClassDB::bind_method("_uninstall_template", &ExportTemplateManager::_uninstall_template);
	ClassDB::bind_method("_uninstall_template_confirm", &ExportTemplateManager::_uninstall_template_confirm);
	ClassDB::bind_method("_install_from_file", &ExportTemplateManager::_install_from_file);
}

ExportTemplateManager::ExportTemplateManager() {

	VBoxContainer *main_vb = memnew(VBoxContainer);
	add_child(main_vb);

	current_hb = memnew(HBoxContainer);
	main_vb->add_margin_child(TTR("Current Version:"), current_hb, false);

	installed_scroll = memnew(ScrollContainer);
	main_vb->add_margin_child(TTR("Installed Versions:"), installed_scroll, true);

	installed_vb = memnew(VBoxContainer);
	installed_scroll->add_child(installed_vb);
	installed_scroll->set_enable_v_scroll(true);
	installed_scroll->set_enable_h_scroll(false);
	installed_vb->set_h_size_flags(SIZE_EXPAND_FILL);

	get_cancel()->set_text(TTR("Close"));
	get_ok()->set_text(TTR("Install From File"));

	remove_confirm = memnew(ConfirmationDialog);
	remove_confirm->set_title(TTR("Remove Template"));
	add_child(remove_confirm);
	remove_confirm->connect("confirmed", this, "_uninstall_template_confirm");

	template_open = memnew(FileDialog);
	template_open->set_title(TTR("Select template file"));
	template_open->add_filter("*.tpz ; Godot Export Templates");
	template_open->set_access(FileDialog::ACCESS_FILESYSTEM);
	template_open->set_mode(FileDialog::MODE_OPEN_FILE);
	template_open->connect("file_selected", this, "_install_from_file");
	add_child(template_open);

	set_title(TTR("Export Template Manager"));
	set_hide_on_ok(false);
}
