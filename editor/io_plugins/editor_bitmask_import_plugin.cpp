/*************************************************************************/
/*  editor_bitmask_import_plugin.cpp                                     */
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
#include "editor_bitmask_import_plugin.h"
#if 0
#include "editor/editor_dir_dialog.h"
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/property_editor.h"
#include "io/image_loader.h"
#include "io/marshalls.h"
#include "io/resource_saver.h"
#include "os/file_access.h"

class _EditorBitMaskImportOptions : public Object {

	GDCLASS(_EditorBitMaskImportOptions, Object);
public:

	bool _set(const StringName& p_name, const Variant& p_value) {

		return false;
	}

	bool _get(const StringName& p_name, Variant &r_ret) const{

		return false;
	}

	void _get_property_list(List<PropertyInfo> *p_list) const{

	}

	static void _bind_methods() {

		ADD_SIGNAL(MethodInfo("changed"));
	}


	_EditorBitMaskImportOptions() {

	}

};

class EditorBitMaskImportDialog : public ConfirmationDialog {

	GDCLASS(EditorBitMaskImportDialog, ConfirmationDialog);

	EditorBitMaskImportPlugin *plugin;

	LineEdit *import_path;
	LineEdit *save_path;
	EditorFileDialog *file_select;
	EditorDirDialog *save_select;
	ConfirmationDialog *error_dialog;
	PropertyEditor *option_editor;

public:

	void _choose_files(const Vector<String>& p_path) {

		String files;
		for (int i = 0; i<p_path.size(); i++) {

			if (i>0)
				files += ",";
			files += p_path[i];
		}

		import_path->set_text(files);

	}
	void _choose_save_dir(const String& p_path) {

		save_path->set_text(p_path);
	}

	void _browse() {

		file_select->popup_centered_ratio();
	}

	void _browse_target() {

		save_select->popup_centered_ratio();

	}


	void popup_import(const String& p_path) {

		popup_centered(Size2(400, 100)*EDSCALE);
		if (p_path != "") {

			Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_path);
			ERR_FAIL_COND(!rimd.is_valid());

			save_path->set_text(p_path.get_base_dir());

			String src = "";
			for (int i = 0; i<rimd->get_source_count(); i++) {
				if (i>0)
					src += ",";
				src += EditorImportPlugin::expand_source_path(rimd->get_source_path(i));
			}
			import_path->set_text(src);
		}
	}


	void _import() {

		Vector<String> bitmasks = import_path->get_text().split(",");

		if (bitmasks.size() == 0) {
			error_dialog->set_text(TTR("No bit masks to import!"));
			error_dialog->popup_centered(Size2(200, 100)*EDSCALE);
		}

		if (save_path->get_text().strip_edges() == "") {
			error_dialog->set_text(TTR("Target path is empty."));
			error_dialog->popup_centered_minsize();
			return;
		}

		if (!save_path->get_text().begins_with("res://")) {
			error_dialog->set_text(TTR("Target path must be a complete resource path."));
			error_dialog->popup_centered_minsize();
			return;
		}

		if (!DirAccess::exists(save_path->get_text())) {
			error_dialog->set_text(TTR("Target path must exist."));
			error_dialog->popup_centered_minsize();
			return;
		}

		for (int i = 0; i<bitmasks.size(); i++) {

			Ref<ResourceImportMetadata> imd = memnew(ResourceImportMetadata);

			imd->add_source(EditorImportPlugin::validate_source_path(bitmasks[i]));

			String dst = save_path->get_text();
			if (dst == "") {
				error_dialog->set_text(TTR("Save path is empty!"));
				error_dialog->popup_centered(Size2(200, 100)*EDSCALE);
			}

			dst = dst.plus_file(bitmasks[i].get_file().get_basename() + ".pbm");

			plugin->import(dst, imd);
		}

		hide();

	}


	void _notification(int p_what) {

	}

	static void _bind_methods() {


		ClassDB::bind_method("_choose_files", &EditorBitMaskImportDialog::_choose_files);
		ClassDB::bind_method("_choose_save_dir", &EditorBitMaskImportDialog::_choose_save_dir);
		ClassDB::bind_method("_import", &EditorBitMaskImportDialog::_import);
		ClassDB::bind_method("_browse", &EditorBitMaskImportDialog::_browse);
		ClassDB::bind_method("_browse_target", &EditorBitMaskImportDialog::_browse_target);
		//ADD_SIGNAL( MethodInfo("imported",PropertyInfo(Variant::OBJECT,"scene")) );
	}

	EditorBitMaskImportDialog(EditorBitMaskImportPlugin *p_plugin) {

		plugin = p_plugin;


		set_title(TTR("Import BitMasks"));

		VBoxContainer *vbc = memnew(VBoxContainer);
		add_child(vbc);
		//set_child_rect(vbc);


		HBoxContainer *hbc = memnew(HBoxContainer);
		vbc->add_margin_child(TTR("Source Texture(s):"), hbc);

		import_path = memnew(LineEdit);
		import_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(import_path);

		Button * import_choose = memnew(Button);
		import_choose->set_text(" .. ");
		hbc->add_child(import_choose);

		import_choose->connect("pressed", this, "_browse");

		hbc = memnew(HBoxContainer);
		vbc->add_margin_child(TTR("Target Path:"), hbc);

		save_path = memnew(LineEdit);
		save_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(save_path);

		Button * save_choose = memnew(Button);
		save_choose->set_text(" .. ");
		hbc->add_child(save_choose);

		save_choose->connect("pressed", this, "_browse_target");

		file_select = memnew(EditorFileDialog);
		file_select->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		add_child(file_select);
		file_select->set_mode(EditorFileDialog::MODE_OPEN_FILES);
		file_select->connect("files_selected", this, "_choose_files");

		List<String> extensions;
		ImageLoader::get_recognized_extensions(&extensions);
		file_select->clear_filters();
		for (int i = 0; i<extensions.size(); i++) {

			file_select->add_filter("*." + extensions[i] + " ; " + extensions[i].to_upper());
		}

		save_select = memnew(EditorDirDialog);
		add_child(save_select);

		//save_select->set_mode(EditorFileDialog::MODE_OPEN_DIR);
		save_select->connect("dir_selected", this, "_choose_save_dir");

		get_ok()->connect("pressed", this, "_import");
		get_ok()->set_text(TTR("Import"));


		error_dialog = memnew(ConfirmationDialog);
		add_child(error_dialog);
		error_dialog->get_ok()->set_text(TTR("Accept"));
		//error_dialog->get_cancel()->hide();

		set_hide_on_ok(false);
	}

	~EditorBitMaskImportDialog() {
	}

};


String EditorBitMaskImportPlugin::get_name() const {

	return "bitmask";
}
String EditorBitMaskImportPlugin::get_visible_name() const{

	return TTR("Bit Mask");
}
void EditorBitMaskImportPlugin::import_dialog(const String& p_from){

	dialog->popup_import(p_from);
}
Error EditorBitMaskImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from){

	ERR_FAIL_COND_V(p_from->get_source_count() != 1, ERR_INVALID_PARAMETER);

	Ref<ResourceImportMetadata> from = p_from;

	String src_path = EditorImportPlugin::expand_source_path(from->get_source_path(0));
	Ref<ImageTexture> it = ResourceLoader::load(src_path);
	ERR_FAIL_COND_V(it.is_null(), ERR_CANT_OPEN);

	Ref<BitMap> target = memnew(BitMap);
	target->create_from_image_alpha(it.ptr()->get_data());

	from->set_source_md5(0, FileAccess::get_md5(src_path));
	from->set_editor(get_name());
	target->set_import_metadata(from);


	Error err = ResourceSaver::save(p_path, target);

	return err;

}


EditorBitMaskImportPlugin* EditorBitMaskImportPlugin::singleton = NULL;


void EditorBitMaskImportPlugin::import_from_drop(const Vector<String>& p_drop, const String &p_dest_path) {

	Vector<String> files;

	List<String> valid_extensions;
	ImageLoader::get_recognized_extensions(&valid_extensions);
	for(int i=0;i<p_drop.size();i++) {

		String extension=p_drop[i].get_extension().to_lower();

		for (List<String>::Element *E=valid_extensions.front();E;E=E->next()) {

			if (E->get()==extension) {
				files.push_back(p_drop[i]);
				break;
			}
		}
	}

	if (files.size()) {
		import_dialog();
		dialog->_choose_files(files);
		dialog->_choose_save_dir(p_dest_path);
	}
}

void EditorBitMaskImportPlugin::reimport_multiple_files(const Vector<String>& p_list) {

	if (p_list.size() == 0)
		return;

	Vector<String> sources;
	for (int i = 0; i<p_list.size(); i++) {
		int idx;
		EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->find_file(p_list[i], &idx);
		if (efsd) {
			for (int j = 0; j<efsd->get_source_count(idx); j++) {
				String file = expand_source_path(efsd->get_source_file(idx, j));
				if (sources.find(file) == -1) {
					sources.push_back(file);
				}

			}
		}
	}

	if (sources.size()) {

		dialog->popup_import(p_list[0]);
		dialog->_choose_files(sources);
		dialog->_choose_save_dir(p_list[0].get_base_dir());
	}
}

bool EditorBitMaskImportPlugin::can_reimport_multiple_files() const {

	return true;
}

EditorBitMaskImportPlugin::EditorBitMaskImportPlugin(EditorNode* p_editor) {

	singleton = this;
	dialog = memnew(EditorBitMaskImportDialog(this));
	p_editor->get_gui_base()->add_child(dialog);
}

EditorBitMaskExportPlugin::EditorBitMaskExportPlugin() {

}
#endif
