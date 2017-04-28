/*************************************************************************/
/*  editor_translation_import_plugin.cpp                                 */
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
#include "editor_translation_import_plugin.h"

#if 0
#include "editor/editor_dir_dialog.h"
#include "editor/editor_node.h"
#include "editor/property_editor.h"
#include "scene/gui/file_dialog.h"
//#include "scene/resources/sample.h"
#include "compressed_translation.h"
#include "editor/project_settings.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "translation.h"


class EditorTranslationImportDialog : public ConfirmationDialog {

	GDCLASS(EditorTranslationImportDialog,ConfirmationDialog);

	EditorTranslationImportPlugin *plugin;

	LineEdit *import_path;
	LineEdit *save_path;
	EditorFileDialog *file_select;
	CheckButton *ignore_first;
	CheckButton *compress;
	CheckButton *add_to_project;
	EditorDirDialog *save_select;
	ConfirmationDialog *error_dialog;
	Vector<TreeItem*> items;
	Tree *columns;

public:

	void _choose_file(const String& p_path) {

		import_path->set_text(p_path);
		FileAccess *f = FileAccess::open(p_path,FileAccess::READ);
		if (!f) {

			error_dialog->set_text(TTR("Invalid source!"));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;

		}

		Vector<String> csvh = f->get_csv_line();
		memdelete(f);

		if (csvh.size()<2) {

			error_dialog->set_text(TTR("Invalid translation source!"));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;

		}

		columns->clear();
		columns->set_columns(2);
		TreeItem *root = columns->create_item();
		columns->set_hide_root(true);
		columns->set_column_titles_visible(true);
		columns->set_column_title(0,TTR("Column"));
		columns->set_column_title(1,TTR("Language"));
		Vector<String> langs = TranslationServer::get_all_locales();
		Vector<String> names = TranslationServer::get_all_locale_names();
		if (csvh[0]=="")
			ignore_first->set_pressed(true);


		items.clear();

		for(int i=1;i<csvh.size();i++) {

			TreeItem *ti = columns->create_item(root);

			ti->set_editable(0,true);
			ti->set_selectable(0,false);
			ti->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
			ti->set_checked(0,true);
			ti->set_text(0,itos(i));
			items.push_back(ti);

			String lname = csvh[i].to_lower().strip_edges();
			int idx=-1;
			String hint;
			for(int j=0;j<langs.size();j++) {

				if (langs[j]==lname.substr(0,langs[j].length()).to_lower()) {
					idx=j;
				}
				if (j>0) {
					hint+=",";
				}
				hint+=names[j].replace(","," ");
			}

			ti->set_cell_mode(1,TreeItem::CELL_MODE_RANGE);
			ti->set_text(1,hint);
			ti->set_editable(1,true);


			if (idx!=-1) {
				ignore_first->set_pressed(true);
				ti->set_range(1,idx);
			} else {

				//not found, maybe used stupid name
				if (lname.begins_with("br")) //brazilian
					ti->set_range(1,langs.find("pt"));
				else if (lname.begins_with("ch")) //chinese
					ti->set_range(1,langs.find("zh"));
				else if (lname.begins_with("sp")) //spanish
					ti->set_range(1,langs.find("es"));
				else if (lname.begins_with("kr"))// kprean
					ti->set_range(1,langs.find("ko"));
				else if (i==0)
					ti->set_range(1,langs.find("en"));
				else
					ti->set_range(1,langs.find("es"));
			}

			ti->set_metadata(1,names[ti->get_range(1)]);
		}



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


	void popup_import(const String& p_from) {

		popup_centered(Size2(400,400)*EDSCALE);

		if (p_from!="") {

			Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_from);
			ERR_FAIL_COND(!rimd.is_valid());
			ERR_FAIL_COND(rimd->get_source_count()!=1);
			_choose_file(EditorImportPlugin::expand_source_path(rimd->get_source_path(0)));
			_choose_save_dir(p_from.get_base_dir());
			String locale = rimd->get_option("locale");
			bool skip_first=rimd->get_option("skip_first");
			bool compressed = rimd->get_option("compress");

			int idx=-1;

			for(int i=0;i<items.size();i++) {

				String il = TranslationServer::get_all_locales()[items[i]->get_range(1)];
				if (il==locale) {
					idx=i;
					break;
				}
			}

			if (idx!=-1) {
				idx=rimd->get_option("index");
			}

			for(int i=0;i<items.size();i++) {

				if (i==idx) {

					Vector<String> locs = TranslationServer::get_all_locales();
					for(int j=0;j<locs.size();j++) {
						if (locs[j]==locale) {
							items[i]->set_range(1,j);
						}

					}
					items[i]->set_checked(0,true);
				} else {
					items[i]->set_checked(0,false);

				}
			}

			ignore_first->set_pressed(skip_first);
			compress->set_pressed(compressed);



		}

	}


	void _import() {


		if (items.size()==0) {
			error_dialog->set_text(TTR("No items to import!"));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
		}

		if (!save_path->get_text().begins_with("res://")) {
			error_dialog->set_text(TTR("No target path!"));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
		}

		EditorProgress progress("import_xl",TTR("Import Translations"),items.size());
		for(int i=0;i<items.size();i++) {

			progress.step(items[i]->get_metadata(1),i);
			if (!items[i]->is_checked(0))
				continue;

			String locale = TranslationServer::get_all_locales()[items[i]->get_range(1)];
			Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );
			imd->add_source(EditorImportPlugin::validate_source_path(import_path->get_text()));
			imd->set_option("locale",locale);
			imd->set_option("index",i);
			imd->set_option("skip_first",ignore_first->is_pressed());
			imd->set_option("compress",compress->is_pressed());

			String savefile = save_path->get_text().plus_file(import_path->get_text().get_file().get_basename()+"."+locale+".xl");
			Error err = plugin->import(savefile,imd);
			if (err!=OK) {
				error_dialog->set_text(TTR("Couldn't import!"));
				error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			} else if (add_to_project->is_pressed()) {

				ProjectSettings::get_singleton()->add_translation(savefile);
			}
		}
		hide();

	}


	void _notification(int p_what) {


		if (p_what==NOTIFICATION_ENTER_TREE) {


		}
	}

	static void _bind_methods() {


		ClassDB::bind_method("_choose_file",&EditorTranslationImportDialog::_choose_file);
		ClassDB::bind_method("_choose_save_dir",&EditorTranslationImportDialog::_choose_save_dir);
		ClassDB::bind_method("_import",&EditorTranslationImportDialog::_import);
		ClassDB::bind_method("_browse",&EditorTranslationImportDialog::_browse);
		ClassDB::bind_method("_browse_target",&EditorTranslationImportDialog::_browse_target);
		//ADD_SIGNAL( MethodInfo("imported",PropertyInfo(Variant::OBJECT,"scene")) );
	}

	EditorTranslationImportDialog(EditorTranslationImportPlugin *p_plugin) {

		plugin=p_plugin;


		set_title(TTR("Import Translation"));

		VBoxContainer *vbc = memnew( VBoxContainer );
		add_child(vbc);
		//set_child_rect(vbc);



		VBoxContainer *csvb = memnew( VBoxContainer );

		HBoxContainer *hbc = memnew( HBoxContainer );
		csvb->add_child(hbc);
		vbc->add_margin_child(TTR("Source CSV:"),csvb);

		import_path = memnew( LineEdit );
		import_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(import_path);
		ignore_first = memnew( CheckButton );
		ignore_first->set_text(TTR("Ignore First Row"));
		csvb->add_child(ignore_first);

		Button * import_choose = memnew( Button );
		import_choose->set_text(" .. ");
		hbc->add_child(import_choose);

		import_choose->connect("pressed", this,"_browse");

		VBoxContainer *tcomp = memnew( VBoxContainer);
		hbc = memnew( HBoxContainer );
		tcomp->add_child(hbc);
		vbc->add_margin_child(TTR("Target Path:"),tcomp);

		save_path = memnew( LineEdit );
		save_path->set_h_size_flags(SIZE_EXPAND_FILL);
		hbc->add_child(save_path);

		Button * save_choose = memnew( Button );
		save_choose->set_text(" .. ");
		hbc->add_child(save_choose);

		save_choose->connect("pressed", this,"_browse_target");

		compress = memnew( CheckButton);
		compress->set_pressed(true);
		compress->set_text(TTR("Compress"));
		tcomp->add_child(compress);

		add_to_project = memnew( CheckButton);
		add_to_project->set_pressed(true);
		add_to_project->set_text(TTR("Add to Project (godot.cfg)"));
		tcomp->add_child(add_to_project);

		file_select = memnew(EditorFileDialog);
		file_select->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		add_child(file_select);
		file_select->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		file_select->connect("file_selected", this,"_choose_file");
		file_select->add_filter("*.csv ; Translation CSV");
		save_select = memnew(	EditorDirDialog );
		add_child(save_select);

		//save_select->set_mode(EditorFileDialog::MODE_OPEN_DIR);
		save_select->connect("dir_selected", this,"_choose_save_dir");

		get_ok()->connect("pressed", this,"_import");
		get_ok()->set_text(TTR("Import"));


		error_dialog = memnew ( ConfirmationDialog );
		add_child(error_dialog);
		error_dialog->get_ok()->set_text(TTR("Accept"));
		//error_dialog->get_cancel()->hide();

		set_hide_on_ok(false);

		columns = memnew( Tree );
		vbc->add_margin_child(TTR("Import Languages:"),columns,true);
	}

	~EditorTranslationImportDialog() {

	}

};


String EditorTranslationImportPlugin::get_name() const {

	return "translation";
}
String EditorTranslationImportPlugin::get_visible_name() const {

	return TTR("Translation");
}
void EditorTranslationImportPlugin::import_dialog(const String& p_from) {

	dialog->popup_import(p_from);
}



void EditorTranslationImportPlugin::import_from_drop(const Vector<String>& p_drop, const String &p_dest_path) {


	for(int i=0;i<p_drop.size();i++) {
		String ext = p_drop[i].get_extension().to_lower();

		if (ext=="csv") {

			import_dialog();
			dialog->_choose_file(p_drop[i]);
			dialog->_choose_save_dir(p_dest_path);
			break;
		}
	}


}

Error EditorTranslationImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from) {

	Ref<ResourceImportMetadata> from = p_from;
	ERR_FAIL_COND_V( from->get_source_count()!=1, ERR_INVALID_PARAMETER);

	String source = EditorImportPlugin::expand_source_path( from->get_source_path(0) );

	FileAccessRef f = FileAccess::open(source,FileAccess::READ);

	ERR_FAIL_COND_V( !f, ERR_INVALID_PARAMETER );

	bool skip_first = from->get_option("skip_first");
	int index = from->get_option("index");
	index+=1;
	String locale = from->get_option("locale");

	Ref<Translation> translation = memnew( Translation );

	translation->set_locale( locale );

	Vector<String> line = f->get_csv_line();

	while(line.size()>1) {

		if (!skip_first) {
			ERR_FAIL_INDEX_V(index,line.size(),ERR_INVALID_DATA );
			translation->add_message(line[0].strip_edges(),line[index]);

		} else {

			skip_first=false;
		}

		line = f->get_csv_line();
	}

	from->set_source_md5(0,FileAccess::get_md5(source));
	from->set_editor(get_name());

	String dst_path = p_path;

	if (from->get_option("compress")) {

		Ref<PHashTranslation> cxl = memnew( PHashTranslation );
		cxl->generate( translation );
		translation=cxl;
	}

	translation->set_import_metadata(from);
	return ResourceSaver::save(dst_path,translation);

}


EditorTranslationImportPlugin::EditorTranslationImportPlugin(EditorNode* p_editor) {

	dialog = memnew(EditorTranslationImportDialog(this));
	p_editor->get_gui_base()->add_child(dialog);
}

#endif
