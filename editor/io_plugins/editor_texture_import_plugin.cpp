/*************************************************************************/
/*  editor_texture_import_plugin.cpp                                     */
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
#include "editor_texture_import_plugin.h"
#if 0
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor_atlas.h"
#include "global_config.h"
#include "io/image_loader.h"
#include "io/marshalls.h"
#include "io/md5.h"
#include "io/resource_saver.h"
#include "scene/gui/button_group.h"
#include "scene/gui/check_button.h"
#include "scene/gui/margin_container.h"
#include "scene/io/resource_format_image.h"

static const char *flag_names[]={
	("Streaming Format"),
	("Fix Border Alpha"),
	("Alpha Bit Hint"),
	("Compress Extra (PVRTC2)"),
	("No MipMaps"),
	("Repeat"),
	("Filter (Magnifying)"),
	("Premultiply Alpha"),
	("Convert SRGB->Linear"),
	("Convert NormalMap to XY"),
	("Use Anisotropy"),
	NULL
};

#if 0 // not used
static const char *flag_short_names[]={
	"Stream",
	"FixBorder",
	"AlphBit",
	"ExtComp",
	"NoMipMap",
	"Repeat",
	"Filter",
	"PMAlpha",
	"ToLinear",
	"ToRG",
	"Anisoropic",
	NULL
};
#endif


void EditorImportTextureOptions::set_format(EditorTextureImportPlugin::ImageFormat p_format) {

	updating=true;
	format->select(p_format);
	if (p_format==EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSY) {
		quality_vb->show();
	} else {
		quality_vb->hide();
	}

	updating=false;

}

EditorTextureImportPlugin::ImageFormat EditorImportTextureOptions::get_format() const{

	return (EditorTextureImportPlugin::ImageFormat)format->get_selected();

}

void EditorImportTextureOptions::set_flags(uint32_t p_flags){

	updating=true;
	for(int i=0;i<items.size();i++) {

		items[i]->set_checked(0,p_flags&(1<<i));
	}
	updating=false;

}

void EditorImportTextureOptions::set_quality(float p_quality) {

	quality->set_value(p_quality);
}

float EditorImportTextureOptions::get_quality() const {

    return quality->get_value();
}


uint32_t EditorImportTextureOptions::get_flags() const{

	uint32_t f=0;
	for(int i=0;i<items.size();i++) {

		if (items[i]->is_checked(0))
			f|=(1<<i);
	}

	return f;
}

void EditorImportTextureOptions::_changedp(int p_value) {

	_changed();
}

void EditorImportTextureOptions::_changed() {

	if (updating)
		return;
	if (format->get_selected()==EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSY) {
		quality_vb->show();
	} else {
		quality_vb->hide();
	}

	emit_signal("changed");
}


void EditorImportTextureOptions::_bind_methods() {

	ClassDB::bind_method("_changed",&EditorImportTextureOptions::_changed);
	ClassDB::bind_method("_changedp",&EditorImportTextureOptions::_changedp);

	ADD_SIGNAL(MethodInfo("changed"));
}


void EditorImportTextureOptions::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		flags->connect("item_edited",this,"_changed");
		format->connect("item_selected",this,"_changedp");
	}
}

void EditorImportTextureOptions::show_2d_notice() {

	//notice_for_2d->show();
}

EditorImportTextureOptions::EditorImportTextureOptions() {


	add_constant_override("separation",3);
	updating=false;
	format = memnew( OptionButton );

	format->add_item(TTR("Uncompressed"),EditorTextureImportPlugin::IMAGE_FORMAT_UNCOMPRESSED);
	format->add_item(TTR("Compress Lossless (PNG)"),EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS);
	format->add_item(TTR("Compress Lossy (WebP)"),EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSY);
	format->add_item(TTR("Compress (VRAM)"),EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM);


	add_margin_child(TTR("Texture Format"),format);

	quality_vb = memnew( VBoxContainer );

	HBoxContainer *quality_hb = memnew(HBoxContainer);
	HSlider *hs = memnew( HSlider );
	hs->set_h_size_flags(SIZE_EXPAND_FILL);
	hs->set_stretch_ratio(0.8);
	quality_hb->add_child(hs);
	quality_hb->set_h_size_flags(SIZE_EXPAND_FILL);
	SpinBox *sb = memnew( SpinBox );
	sb->set_h_size_flags(SIZE_EXPAND_FILL);
	sb->set_stretch_ratio(0.2);
	quality_hb->add_child(sb);
	sb->share(hs);
	hs->set_min(0);
	hs->set_max(1.0);
	hs->set_step(0.01);
	hs->set_value(0.7);
	quality=hs;
	quality_vb->add_margin_child(TTR("Texture Compression Quality (WebP):"),quality_hb);

	add_child(quality_vb);

	flags = memnew( Tree );
	flags->set_hide_root(true);
	TreeItem *root = flags->create_item();



	const char ** fname=flag_names;

	while( *fname ) {

		TreeItem*ti = flags->create_item(root);
		ti->set_cell_mode(0,TreeItem::CELL_MODE_CHECK);
		ti->set_text(0,*fname);
		ti->set_editable(0,true);
		items.push_back(ti);
		fname++;
	}

	add_margin_child(TTR("Texture Options"),flags,true);


}

///////////////////////////////////////////////////////////




class EditorTextureImportDialog : public ConfirmationDialog  {

	GDCLASS(EditorTextureImportDialog,ConfirmationDialog);



	HBoxContainer *mode_hb;
	CheckBox *mode_check[EditorTextureImportPlugin::MODE_MAX];

	EditorImportTextureOptions *texture_options;

	EditorTextureImportPlugin::Mode mode;
	//EditorNode *editor;

	LineEdit *import_path;
	LineEdit *save_path;
	EditorFileDialog *file_select;
	EditorFileDialog *save_file_select;
	EditorDirDialog *save_select;
	OptionButton *texture_action;
	ConfirmationDialog *error_dialog;
	CheckButton *crop_source;
	SpinBox *size;

	MarginContainer *size_mc;
	Label* size_label;

	Label* source_label;
	Label *notice_for_2d;

	EditorTextureImportPlugin *plugin;

	void _mode_changed(int p_mode);
	void _choose_files(const Vector<String>& p_path);
	void _choose_file(const String& p_path);
	void _choose_save_dir(const String& p_path);
	void _browse();
	void _browse_target();
	void _import();


protected:

	void _notification(int p_what);
	static void _bind_methods();
public:


	void setup_multiple_import_3d(const Vector<String>& p_path,const String& p_dest) {

		_mode_changed(EditorTextureImportPlugin::MODE_TEXTURE_3D);
		_choose_files(p_path);
		_choose_save_dir(p_dest);
	}

	void add_sources_and_dest(const Vector<String>& p_path,const String& p_dest) {

		_choose_files(p_path);
		_choose_save_dir(p_dest);
	}

	Error import(const String& p_from, const String& p_to, const String& p_preset);
	void popup_import(const String &p_from=String());
	EditorTextureImportDialog(EditorTextureImportPlugin *p_plugin=NULL);
};


/////////////////////////////////////////////////////////




void EditorTextureImportDialog::_choose_files(const Vector<String>& p_path) {

	String files;
	for(int i=0;i<p_path.size();i++) {

		if (i>0)
			files+=",";
		files+=p_path[i];
	}
	/*
	if (p_path.size()) {
		String srctex=p_path[0];
		String ipath = EditorImportDB::get_singleton()->find_source_path(srctex);

		if (ipath!="")
			save_path->set_text(ipath.get_base_dir());
	}*/
	import_path->set_text(files);

}



void EditorTextureImportDialog::_choose_file(const String& p_path) {


	import_path->set_text(p_path);

}
void EditorTextureImportDialog::_choose_save_dir(const String& p_path) {

	save_path->set_text(p_path);
}


void EditorTextureImportDialog::_import() {


	//ImportMonitorBlock imb;

	Vector<String> files=import_path->get_text().split(",");

	if (!files.size()) {

		error_dialog->set_text(TTR("Please specify some files!"));
		error_dialog->popup_centered(Size2(200,100)*EDSCALE);
		return;
	}

	String dst_path=save_path->get_text();

	if (save_path->get_text().strip_edges()=="") {
		error_dialog->set_text(TTR("Target path is empty."));
		error_dialog->popup_centered_minsize();
		return;
	}

	if (!save_path->get_text().begins_with("res://")) {
		error_dialog->set_text(TTR("Target path must be a complete resource path."));
		error_dialog->popup_centered_minsize();
		return;
	}


	if (mode!=EditorTextureImportPlugin::MODE_ATLAS && mode!=EditorTextureImportPlugin::MODE_LARGE && !DirAccess::exists(save_path->get_text())) {
		error_dialog->set_text(TTR("Target path must exist."));
		error_dialog->popup_centered_minsize();
		return;
	}

	if (mode==EditorTextureImportPlugin::MODE_ATLAS) { //atlas

		if (files.size()==0) {

			error_dialog->set_text(TTR("At least one file needed for Atlas."));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;

		}
		String dst_file = dst_path;
		//dst_file=dst_file.basename()+".tex";
		Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );
		//imd->set_editor();
		for(int i=0;i<files.size();i++) {
			imd->add_source(EditorImportPlugin::validate_source_path(files[i]));
		}
		imd->set_option("format",texture_options->get_format());
		imd->set_option("flags",texture_options->get_flags());
		imd->set_option("quality",texture_options->get_quality());
		imd->set_option("atlas",true);
		imd->set_option("atlas_size",int(size->get_value()));
		imd->set_option("large",false);
		imd->set_option("crop",crop_source->is_pressed());
		imd->set_option("mode",mode);

		Error err = plugin->import(dst_file,imd);
		if (err) {

			error_dialog->set_text(TTR("Error importing:")+" "+dst_file.get_file());
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;

		}
	} else if (mode==EditorTextureImportPlugin::MODE_LARGE) { //large

		if (files.size()!=1) {

			error_dialog->set_text(TTR("Only one file is required for large texture."));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;

		}
		String dst_file = dst_path;
		//dst_file=dst_file.basename()+".tex";
		Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );
		//imd->set_editor();
		for(int i=0;i<files.size();i++) {
			imd->add_source(EditorImportPlugin::validate_source_path(files[i]));
		}
		imd->set_option("format",texture_options->get_format());
		imd->set_option("flags",texture_options->get_flags());
		imd->set_option("quality",texture_options->get_quality());
		imd->set_option("atlas",false);
		imd->set_option("large",true);
		imd->set_option("large_cell_size",int(size->get_value()));
		imd->set_option("crop",crop_source->is_pressed());
		imd->set_option("mode",mode);

		Error err = plugin->import(dst_file,imd);
		if (err) {

			error_dialog->set_text(TTR("Error importing:")+" "+dst_file.get_file());
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;

		}
	} else {


		for(int i=0;i<files.size();i++) {

			String dst_file = dst_path.plus_file(files[i].get_file());
			dst_file=dst_file.get_basename()+".tex";
			Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );
			//imd->set_editor();
			imd->add_source(EditorImportPlugin::validate_source_path(files[i]));
			imd->set_option("format",texture_options->get_format());
			imd->set_option("flags",texture_options->get_flags());
			imd->set_option("quality",texture_options->get_quality());
			imd->set_option("atlas",false);
			imd->set_option("large",false);
			imd->set_option("mode",mode);

			Error err = plugin->import(dst_file,imd);
			if (err) {

				error_dialog->set_text(TTR("Error importing:")+" "+dst_file.get_file());
				error_dialog->popup_centered(Size2(200,100)*EDSCALE);
				return;

			}
		}
	}

	hide();
}

void EditorTextureImportDialog::_browse() {

	file_select->popup_centered_ratio();
}

void EditorTextureImportDialog::_browse_target() {

	if (mode==EditorTextureImportPlugin::MODE_ATLAS || mode==EditorTextureImportPlugin::MODE_LARGE) {
		save_file_select->popup_centered_ratio();
	} else {
		save_select->popup_centered_ratio();
	}

}


void EditorTextureImportDialog::popup_import(const String& p_from) {

	popup_centered(Size2(600,500)*EDSCALE);
	if (p_from!="") {
		Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_from);
		ERR_FAIL_COND(!rimd.is_valid());

		if (rimd->has_option("mode")) {
			//new imported stuff uses this option
			_mode_changed(rimd->get_option("mode"));
		} else {
			//this one is for compatibility, will have to guess it
			if (rimd->has_option("atlas") && rimd->get_option("atlas")) {
				_mode_changed(EditorTextureImportPlugin::MODE_ATLAS);
			} else if (rimd->has_option("large") && rimd->get_option("large")) {
				_mode_changed(EditorTextureImportPlugin::MODE_LARGE);
			} else {
				//guess by usage of mipmaps..?
				_mode_changed(EditorTextureImportPlugin::MODE_TEXTURE_2D);
			}

		}

		if (mode==EditorTextureImportPlugin::MODE_ATLAS || mode==EditorTextureImportPlugin::MODE_LARGE)
			save_path->set_text(p_from);
		else
			save_path->set_text(p_from.get_base_dir());

		texture_options->set_format(EditorTextureImportPlugin::ImageFormat(int(rimd->get_option("format"))));
		texture_options->set_flags(rimd->get_option("flags"));
		texture_options->set_quality(rimd->get_option("quality"));
		String src = "";
		for(int i=0;i<rimd->get_source_count();i++) {
			if (i>0)
				src+=",";
			src+=EditorImportPlugin::expand_source_path(rimd->get_source_path(i));
		}
		import_path->set_text(src);
	}
}


void EditorTextureImportDialog::_notification(int p_what) {


	if (p_what==NOTIFICATION_ENTER_TREE) {


		List<String> extensions;
		ImageLoader::get_recognized_extensions(&extensions);
		//ResourceLoader::get_recognized_extensions_for_type("PackedTexture",&extensions);
		file_select->clear_filters();
		for(int i=0;i<extensions.size();i++) {

			file_select->add_filter("*."+extensions[i]+" ; "+extensions[i].to_upper());
		}
	}
}

Error EditorTextureImportDialog::import(const String& p_from, const String& p_to, const String& p_preset) {


	import_path->set_text(p_from);
	save_path->set_text(p_to);
	_import();

	return OK;
}

void EditorTextureImportDialog::_mode_changed(int p_mode) {

	mode = EditorTextureImportPlugin::Mode(p_mode);

	for(int i=0;i<EditorTextureImportPlugin::MODE_MAX;i++) {
		mode_check[i]->set_pressed(i==mode);
	}

	if (p_mode==EditorTextureImportPlugin::MODE_ATLAS) {

		size_label->set_text(TTR("Max Texture Size:"));
		size->set_value(2048);
		crop_source->show();
		size_label->show();
		size->show();

		texture_options->set_flags(EditorTextureImportPlugin::IMAGE_FLAG_FIX_BORDER_ALPHA|EditorTextureImportPlugin::IMAGE_FLAG_NO_MIPMAPS|EditorTextureImportPlugin::IMAGE_FLAG_FILTER);
		texture_options->set_quality(0.7);
		texture_options->set_format(EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSY);
		set_title(TTR("Import Textures for Atlas (2D)"));

	} else {
		crop_source->hide();
	}


	if (p_mode==EditorTextureImportPlugin::MODE_LARGE) {

		size_label->set_text(TTR("Cell Size:"));
		size->set_value(256);
		size_label->show();
		size->show();

		file_select->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		save_file_select->add_filter("*.ltex;"+TTR("Large Texture"));

		texture_options->set_flags(EditorTextureImportPlugin::IMAGE_FLAG_FIX_BORDER_ALPHA|EditorTextureImportPlugin::IMAGE_FLAG_NO_MIPMAPS|EditorTextureImportPlugin::IMAGE_FLAG_FILTER);
		texture_options->set_quality(0.7);
		texture_options->set_format(EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS);
		set_title(TTR("Import Large Textures (2D)"));
		source_label->set_text(TTR("Source Texture"));

	} else {
		file_select->set_mode(EditorFileDialog::MODE_OPEN_FILES);
		save_file_select->add_filter("*.tex;"+TTR("Base Atlas Texture"));
		source_label->set_text(TTR("Source Texture(s)"));
	}

	if (p_mode==EditorTextureImportPlugin::MODE_TEXTURE_2D) {

		size_label->hide();
		size->hide();

		texture_options->set_flags(EditorTextureImportPlugin::IMAGE_FLAG_NO_MIPMAPS|EditorTextureImportPlugin::IMAGE_FLAG_FIX_BORDER_ALPHA|EditorTextureImportPlugin::IMAGE_FLAG_FILTER);
		texture_options->set_quality(0.7);
		texture_options->set_format(EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSY);
		notice_for_2d->show();
		set_title(TTR("Import Textures for 2D"));

	} else {
		notice_for_2d->hide();
	}

	if (p_mode==EditorTextureImportPlugin::MODE_TEXTURE_3D) {

		size_label->hide();
		size->hide();
		//texture_options->set_flags(EditorTextureImportPlugin::IMAGE_FLAG_);
		//texture_options->set_flags(EditorTextureImportPlugin::IMAGE_FLAG_NO_MIPMAPS);
		texture_options->set_flags(EditorTextureImportPlugin::IMAGE_FLAG_FIX_BORDER_ALPHA|EditorTextureImportPlugin::IMAGE_FLAG_FILTER|EditorTextureImportPlugin::IMAGE_FLAG_REPEAT);
		texture_options->set_format(EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM);
		set_title(TTR("Import Textures for 3D"));
	}
}

void EditorTextureImportDialog::_bind_methods() {


	ClassDB::bind_method("_choose_files",&EditorTextureImportDialog::_choose_files);
	ClassDB::bind_method("_choose_file",&EditorTextureImportDialog::_choose_file);
	ClassDB::bind_method("_choose_save_dir",&EditorTextureImportDialog::_choose_save_dir);
	ClassDB::bind_method("_import",&EditorTextureImportDialog::_import);
	ClassDB::bind_method("_browse",&EditorTextureImportDialog::_browse);
	ClassDB::bind_method("_browse_target",&EditorTextureImportDialog::_browse_target);
	ClassDB::bind_method("_mode_changed",&EditorTextureImportDialog::_mode_changed);
	//ADD_SIGNAL( MethodInfo("imported",PropertyInfo(Variant::OBJECT,"scene")) );
}

EditorTextureImportDialog::EditorTextureImportDialog(EditorTextureImportPlugin* p_plugin) {





	plugin=p_plugin;
	set_title(TTR("Import Textures"));

	mode_hb = memnew( HBoxContainer );
	add_child(mode_hb);
	//set_child_rect(mode_hb);

	VBoxContainer *vbcg = memnew( VBoxContainer);


	mode_hb->add_child(vbcg);
	mode_hb->add_constant_override("separation",15);
	VBoxContainer *bg = memnew( VBoxContainer );
	vbcg->add_margin_child("Import Mode",bg);

	for(int i=0;i<EditorTextureImportPlugin::MODE_MAX;i++) {
		String mode_name[EditorTextureImportPlugin::MODE_MAX]={
			TTR("2D Texture"),
			TTR("3D Texture"),
			TTR("Atlas Texture"),
			TTR("Large Texture")
		};


		mode_check[i]=memnew(CheckBox);
		bg->add_child(mode_check[i]);
		mode_check[i]->set_text(mode_name[i]);
		mode_check[i]->connect("pressed",this,"_mode_changed",varray(i));
	}

	VBoxContainer *vbc = memnew(VBoxContainer);
	mode_hb->add_child(vbc);
	vbc->set_h_size_flags(SIZE_EXPAND_FILL);
	vbc->add_constant_override("separation",4);

	notice_for_2d = memnew( Label );
	notice_for_2d->set_text(TTR("NOTICE: Importing 2D textures is not mandatory. Just copy png/jpg files to the project."));
	//notice_for_2d->set_custom_minimum_size(Size2(0,50));
	notice_for_2d->set_autowrap(true);
	notice_for_2d->hide();
	vbcg->add_child(notice_for_2d);
	notice_for_2d->set_v_size_flags(SIZE_EXPAND_FILL);
	notice_for_2d->set_valign(Label::VALIGN_BOTTOM);

	VBoxContainer *source_vb=memnew(VBoxContainer);
	MarginContainer *source_mc = vbc->add_margin_child(TTR("Source Texture(s):"),source_vb);

	source_label = vbc->get_child(source_mc->get_index()-1)->cast_to<Label>();

	HBoxContainer *hbc = memnew( HBoxContainer );
	source_vb->add_child(hbc);

	import_path = memnew( LineEdit );
	import_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(import_path);
	crop_source = memnew( CheckButton );
	crop_source->set_pressed(true);
	source_vb->add_child(crop_source);
	crop_source->set_text(TTR("Crop empty space."));


	Button * import_choose = memnew( Button );
	import_choose->set_text(" .. ");
	hbc->add_child(import_choose);

	import_choose->connect("pressed", this,"_browse");

	hbc = memnew( HBoxContainer );
	vbc->add_margin_child(TTR("Target Path:"),hbc);

	size = memnew( SpinBox );
	size->set_min(128);
	size->set_max(16384);


	size->set_value(256);
	size_mc=vbc->add_margin_child(TTR("Cell Size:"),size);
	size_label=vbc->get_child(size_mc->get_index()-1)->cast_to<Label>();


	save_path = memnew( LineEdit );
	save_path->set_h_size_flags(SIZE_EXPAND_FILL);
	hbc->add_child(save_path);

	Button * save_choose = memnew( Button );
	save_choose->set_text(" .. ");
	hbc->add_child(save_choose);

	save_choose->connect("pressed", this,"_browse_target");

	file_select = memnew(EditorFileDialog);
	file_select->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
	add_child(file_select);

	file_select->connect("files_selected", this,"_choose_files");
	file_select->connect("file_selected", this,"_choose_file");

	save_file_select = memnew(EditorFileDialog);
	save_file_select->set_access(EditorFileDialog::ACCESS_RESOURCES);
	add_child(save_file_select);
	save_file_select->set_mode(EditorFileDialog::MODE_SAVE_FILE);
	save_file_select->clear_filters();

	save_file_select->connect("file_selected", this,"_choose_save_dir");

	save_select = memnew(	EditorDirDialog );
	add_child(save_select);

	//save_select->set_mode(EditorFileDialog::MODE_OPEN_DIR);
	save_select->connect("dir_selected", this,"_choose_save_dir");

	get_ok()->connect("pressed", this,"_import");
	get_ok()->set_text(TTR("Import"));

	//move stuff up
	/*
	for(int i=0;i<4;i++)
		vbc->move_child( vbc->get_child( vbc->get_child_count() -1), 0);
	*/

	error_dialog = memnew ( ConfirmationDialog );
	add_child(error_dialog);
	error_dialog->get_ok()->set_text(TTR("Accept"));
	//error_dialog->get_cancel()->hide();

	set_hide_on_ok(false);

	texture_options = memnew( EditorImportTextureOptions );
	vbc->add_child(texture_options);
	texture_options->set_v_size_flags(SIZE_EXPAND_FILL);

	_mode_changed(EditorTextureImportPlugin::MODE_TEXTURE_3D);


	//GLOBAL_DEF("import/shared_textures","res://");
	//Globals::get_singleton()->set_custom_property_info("import/shared_textures",PropertyInfo(Variant::STRING,"import/shared_textures",PROPERTY_HINT_DIR));


}



///////////////////////////////////////////////////////////


String EditorTextureImportPlugin::get_name() const {

	return "texture";
#if 0 //old names, kept for compatibility reference
	switch(mode) {
		case MODE_TEXTURE_2D: {

			return "texture_2d";
		} break;
		case MODE_TEXTURE_3D: {

			return "texture_3d";

		} break;
		case MODE_ATLAS: {

			return "texture_atlas";
		} break;
		case MODE_LARGE: {

			return "texture_large";
		} break;

	}


	return "";
#endif
}

String EditorTextureImportPlugin::get_visible_name() const {

	return TTR("Texture");

}
void EditorTextureImportPlugin::import_dialog(const String& p_from) {

	dialog->popup_import(p_from);
}

void EditorTextureImportPlugin::compress_image(EditorExportPlatform::ImageCompression p_mode,Image& image,bool p_smaller) {


	switch(p_mode) {
		case EditorExportPlatform::IMAGE_COMPRESSION_NONE: {

			//do absolutely nothing

		} break;	
		case EditorExportPlatform::IMAGE_COMPRESSION_BC: {


			// for maximum compatibility, BC shall always use mipmaps and be PO2
			image.resize_to_po2();
			if (!image.has_mipmaps())
				image.generate_mipmaps();

			image.compress(Image::COMPRESS_S3TC);
			/*
			if (has_alpha) {

				if (flags&IMAGE_FLAG_ALPHA_BIT) {
					image.convert(Image::FORMAT_DXT5);
				} else {
					image.convert(Image::FORMAT_DXT3);
				}
			} else {

				image.convert(Image::FORMAT_DXT1);
			}*/


		} break;
		case EditorExportPlatform::IMAGE_COMPRESSION_PVRTC:
		case EditorExportPlatform::IMAGE_COMPRESSION_PVRTC_SQUARE: {

			// for maximum compatibility (hi apple!), PVRT shall always
			// use mipmaps, be PO2 and square

			if (!image.has_mipmaps())
				image.generate_mipmaps();
			image.resize_to_po2(true);

			if (p_smaller) {

				image.compress(Image::COMPRESS_PVRTC2);
				//image.convert(has_alpha ? Image::FORMAT_PVRTC2A : Image::FORMAT_PVRTC2);
			} else {
				image.compress(Image::COMPRESS_PVRTC4);
				//image.convert(has_alpha ? Image::FORMAT_PVRTC4A : Image::FORMAT_PVRTC4);
			}

		} break;
		case EditorExportPlatform::IMAGE_COMPRESSION_ETC1: {

			image.resize_to_po2(); //square or not?
			if (!image.has_mipmaps())
				image.generate_mipmaps();
			if (!image.detect_alpha()) {
				//ETC1 is only opaque
				image.compress(Image::COMPRESS_ETC);
			}

		} break;
		case EditorExportPlatform::IMAGE_COMPRESSION_ETC2: {


		} break;
	}


}

Error EditorTextureImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from) {


	return import2(p_path,p_from,EditorExportPlatform::IMAGE_COMPRESSION_BC,false);
}


Error EditorTextureImportPlugin::_process_texture_data(Ref<ImageTexture> &texture,int format, float quality,int flags,EditorExportPlatform::ImageCompression p_compr,int tex_flags,float shrink)  {


	if (format==IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS || format==IMAGE_FORMAT_COMPRESS_DISK_LOSSY) {

		Image image=texture->get_data();
		ERR_FAIL_COND_V(image.empty(),ERR_INVALID_DATA);

		bool has_alpha=image.detect_alpha();
		if (!has_alpha && image.get_format()==Image::FORMAT_RGBA8) {

			image.convert(Image::FORMAT_RGB8);

		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_FIX_BORDER_ALPHA) {

			image.fix_alpha_edges();
		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_PREMULT_ALPHA) {

			image.premultiply_alpha();
		}

		if (flags&IMAGE_FLAG_CONVERT_NORMAL_TO_XY) {
			image.normalmap_to_xy();
		}

		/*
		if ((image.get_format()==Image::FORMAT_RGB8 || image.get_format()==Image::FORMAT_RGBA8) && flags&IMAGE_FLAG_CONVERT_TO_LINEAR) {

			image.srgb_to_linear();
		}
		*/

		if (shrink>1) {

			int orig_w=image.get_width();
			int orig_h=image.get_height();
			image.resize(orig_w/shrink,orig_h/shrink,Image::INTERPOLATE_CUBIC);
			texture->create_from_image(image,tex_flags);
			texture->set_size_override(Size2(orig_w,orig_h));


		} else {

			texture->create_from_image(image,tex_flags);
		}


		if (format==IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS) {
			texture->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSLESS);
		} else {
			texture->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
		}



		texture->set_lossy_storage_quality(quality);


	} else {


		Image image=texture->get_data();
		ERR_FAIL_COND_V(image.empty(),ERR_INVALID_DATA);


		bool has_alpha=image.detect_alpha();
		if (!has_alpha && image.get_format()==Image::FORMAT_RGBA8) {

			image.convert(Image::FORMAT_RGB8);

		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_FIX_BORDER_ALPHA) {

			image.fix_alpha_edges();
		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_PREMULT_ALPHA) {

			image.premultiply_alpha();
		}

		if (flags&IMAGE_FLAG_CONVERT_NORMAL_TO_XY) {
			image.normalmap_to_xy();
		}

		/*
		if ((image.get_format()==Image::FORMAT_RGB8 || image.get_format()==Image::FORMAT_RGBA8) && flags&IMAGE_FLAG_CONVERT_TO_LINEAR) {

			print_line("CONVERT BECAUSE: "+itos(flags));
			image.srgb_to_linear();
		}
		*/

		int orig_w=image.get_width();
		int orig_h=image.get_height();

		if (shrink>1) {
			image.resize(orig_w/shrink,orig_h/shrink,Image::INTERPOLATE_CUBIC);
			texture->create_from_image(image,tex_flags);
			texture->set_size_override(Size2(orig_w,orig_h));
		}

		if (!(flags&IMAGE_FLAG_NO_MIPMAPS)) {
			image.generate_mipmaps();

		}

		if (format!=IMAGE_FORMAT_UNCOMPRESSED) {

			compress_image(p_compr,image,flags&IMAGE_FLAG_COMPRESS_EXTRA);
		}


		texture->create_from_image(image,tex_flags);


		if (shrink>1 || (format!=IMAGE_FORMAT_UNCOMPRESSED && (image.get_width()!=orig_w || image.get_height()!=orig_h))) {
			texture->set_size_override(Size2(orig_w,orig_h));
		}

		//uint32_t save_flags=ResourceSaver::FLAG_COMPRESS;
	}

	return OK;
}


Error EditorTextureImportPlugin::import2(const String& p_path, const Ref<ResourceImportMetadata>& p_from,EditorExportPlatform::ImageCompression p_compr, bool p_external){



	ERR_FAIL_COND_V(p_from->get_source_count()==0,ERR_INVALID_PARAMETER);

	Ref<ResourceImportMetadata> from=p_from;

	Ref<ImageTexture> texture;
	Vector<Ref<AtlasTexture> > atlases;
	bool atlas = from->get_option("atlas");
	bool large = from->get_option("large");

	int flags=from->get_option("flags");
	int format=from->get_option("format");
	float quality=from->get_option("quality");

	uint32_t tex_flags=0;

	if (flags&EditorTextureImportPlugin::IMAGE_FLAG_REPEAT)
		tex_flags|=Texture::FLAG_REPEAT;
	if (flags&EditorTextureImportPlugin::IMAGE_FLAG_FILTER)
		tex_flags|=Texture::FLAG_FILTER;
	if (!(flags&EditorTextureImportPlugin::IMAGE_FLAG_NO_MIPMAPS))
		tex_flags|=Texture::FLAG_MIPMAPS;
	if (flags&EditorTextureImportPlugin::IMAGE_FLAG_CONVERT_TO_LINEAR)
		tex_flags|=Texture::FLAG_CONVERT_TO_LINEAR;
	if (flags&EditorTextureImportPlugin::IMAGE_FLAG_USE_ANISOTROPY)
		tex_flags|=Texture::FLAG_ANISOTROPIC_FILTER;

	print_line("path: "+p_path+" flags: "+itos(tex_flags));
	float shrink=1;
	if (from->has_option("shrink"))
		shrink=from->get_option("shrink");

	if (large) {
		ERR_FAIL_COND_V(from->get_source_count()!=1,ERR_INVALID_PARAMETER);

		String src_path = EditorImportPlugin::expand_source_path(from->get_source_path(0));


		int cell_size=from->get_option("large_cell_size");
		ERR_FAIL_COND_V(cell_size<128 || cell_size>16384,ERR_CANT_OPEN);

		EditorProgress pg("ltex",TTR("Import Large Texture"),3);

		pg.step(TTR("Load Source Image"),0);
		Image img;
		Error err = ImageLoader::load_image(src_path,&img);
		if (err) {
			return err;
		}

		pg.step(TTR("Slicing"),1);

		Map<Vector2,Image> pieces;
		for(int i=0;i<img.get_width();i+=cell_size) {
			int w = MIN(img.get_width()-i,cell_size);
			for(int j=0;j<img.get_height();j+=cell_size) {
				int h = MIN(img.get_height()-j,cell_size);

				Image piece(w,h,0,img.get_format());
				piece.blit_rect(img,Rect2(i,j,w,h),Point2(0,0));
				if (!piece.is_invisible()) {
					pieces[Vector2(i,j)]=piece;
					//print_line("ADDING PIECE AT "+Vector2(i,j));
				}
			}
		}

		Ref<LargeTexture> existing;
		if (ResourceCache::has(p_path)) {
			existing = ResourceCache::get(p_path);
		}

		if (existing.is_valid()) {
			existing->clear();
		} else {
			existing = Ref<LargeTexture>(memnew( LargeTexture ));
		}

		existing->set_size(Size2(img.get_width(),img.get_height()));
		pg.step(TTR("Inserting"),2);

		for (Map<Vector2,Image>::Element *E=pieces.front();E;E=E->next()) {

			Ref<ImageTexture> imgtex = Ref<ImageTexture>( memnew( ImageTexture ) );
			imgtex->create_from_image(E->get(),tex_flags);
			_process_texture_data(imgtex,format,quality,flags,p_compr,tex_flags,shrink);
			existing->add_piece(E->key(),imgtex);
		}

		if (!p_external) {
			from->set_editor(get_name());
			from->set_source_md5(0,FileAccess::get_md5(src_path));
			existing->set_path(p_path);
			existing->set_import_metadata(from);
		}
		pg.step(TTR("Saving"),3);

		err = ResourceSaver::save(p_path,existing);
		if (err!=OK) {
			EditorNode::add_io_error(TTR("Couldn't save large texture:")+" "+p_path);
			return err;
		}

		return OK;


	} else if (atlas) {

		//prepare atlas!
		Vector< Image > sources;
		Vector< Image > tsources;
		bool alpha=false;
		bool crop = from->get_option("crop");

		EditorProgress ep("make_atlas",TTR("Build Atlas For:")+" "+p_path.get_file(),from->get_source_count()+3);

		print_line("sources: "+itos(from->get_source_count()));

		for(int i=0;i<from->get_source_count();i++) {

			String path = EditorImportPlugin::expand_source_path(from->get_source_path(i));
			String md5 = FileAccess::get_md5(path);
			from->set_source_md5(i,FileAccess::get_md5(path));
			ep.step(TTR("Loading Image:")+" "+path,i);
			print_line("source path: "+path+" md5 "+md5);
			Image src;
			Error err = ImageLoader::load_image(path,&src);
			if (err) {
				EditorNode::add_io_error(TTR("Couldn't load image:")+" "+path);
				return err;
			}

			if (src.detect_alpha())
				alpha=true;

			tsources.push_back(src);
		}
		ep.step(TTR("Converting Images"),sources.size());


		Map<uint64_t,int> source_md5;
		Map<int,List<int> > source_map;

		for(int i=0;i<tsources.size();i++) {

			Image src = tsources[i];

			if (alpha) {
				src.convert(Image::FORMAT_RGBA8);
			} else {
				src.convert(Image::FORMAT_RGB8);
			}

			PoolVector<uint8_t> data = src.get_data();
			MD5_CTX md5;
			PoolVector<uint8_t>::Read r=data.read();
			MD5Init(&md5);
			int len=data.size();
			for(int j=0;j<len;j++) {
				uint8_t b = r[j];
				b>>=2; //to aid in comparing
				MD5Update(&md5,(unsigned char*)&b,1);
			}
			MD5Final(&md5);
			uint64_t *cmp = (uint64_t*)md5.digest; //less bits, but still useful for this

			tsources[i]=Image(); //clear

			if (source_md5.has(*cmp)) {
				int sidx=source_md5[*cmp];
				source_map[sidx].push_back(i);
				print_line("REUSING "+from->get_source_path(i));

			} else {
				int sidx=sources.size();
				source_md5[*cmp]=sidx;
				sources.push_back(src);
				List<int> sm;
				sm.push_back(i);
				source_map[sidx]=sm;

			}


		}

		//texturepacker is not really good for optimizing, so..
		//will at some point likely replace with my own
		//first, will find the nearest to a square packing
		int border=1;

		Vector<Size2i> src_sizes;
		Vector<Rect2> crops;

		ep.step(TTR("Cropping Images"),sources.size()+1);

		for(int j=0;j<sources.size();j++) {

			Size2i s;
			if (crop) {
				Rect2 crop = sources[j].get_used_rect();
				print_line("CROP: "+crop);
				s=crop.size;
				crops.push_back(crop);
			} else {

				s=Size2i(sources[j].get_width(),sources[j].get_height());
			}
			s+=Size2i(border*2,border*2);
			src_sizes.push_back(s); //add a line to constraint width
		}

		Vector<Point2i> dst_positions;
		Size2i dst_size;
		EditorAtlas::fit(src_sizes,dst_positions,dst_size);

		print_line("size that worked: "+itos(dst_size.width)+","+itos(dst_size.height));

		ep.step(TTR("Blitting Images"),sources.size()+2);

		bool blit_to_po2=tex_flags&Texture::FLAG_MIPMAPS;
		int atlas_w=dst_size.width;
		int atlas_h=dst_size.height;
		if (blit_to_po2) {
			atlas_w=nearest_power_of_2(dst_size.width);
			atlas_h=nearest_power_of_2(dst_size.height);
		}
		Image atlas;
		atlas.create(atlas_w,atlas_h,0,alpha?Image::FORMAT_RGBA8:Image::FORMAT_RGB8);


		atlases.resize(from->get_source_count());

		for(int i=0;i<sources.size();i++) {

			int x=dst_positions[i].x;
			int y=dst_positions[i].y;

			Size2 sz = Size2(sources[i].get_width(),sources[i].get_height());

			Rect2 region;
			Rect2 margin;

			if (crop && sz!=crops[i].size) {
				Rect2 rect = crops[i];
				rect.size=sz-rect.size;
				region=Rect2(x+border,y+border,crops[i].size.width,crops[i].size.height);
				margin=rect;
				atlas.blit_rect(sources[i],crops[i],Point2(x+border,y+border));
			} else {
				region=Rect2(x+border,y+border,sz.x,sz.y);
				atlas.blit_rect(sources[i],Rect2(0,0,sources[i].get_width(),sources[i].get_height()),Point2(x+border,y+border));
			}

			ERR_CONTINUE( !source_map.has(i) );
			for (List<int>::Element *E=source_map[i].front();E;E=E->next()) {

				String apath;
				String spath = from->get_source_path(E->get()).get_file();

				if (p_external) {
					apath = p_path.get_base_dir().plus_file(spath.get_basename()+"."+from->get_source_path(E->get()).md5_text()+".atex");
				} else {
					apath = p_path.get_base_dir().plus_file(spath.get_basename()+".atex");
				}

				Ref<AtlasTexture> at;

				if (ResourceCache::has(apath)) {

					at = Ref<AtlasTexture>( ResourceCache::get(apath)->cast_to<AtlasTexture>() );
				} else {

					at = Ref<AtlasTexture>( memnew( AtlasTexture ) );

				}
				at->set_region(region);
				at->set_margin(margin);
				at->set_path(apath);
				atlases[E->get()]=at;

			}
		}
		if (ResourceCache::has(p_path)) {
			texture = Ref<ImageTexture> ( ResourceCache::get(p_path)->cast_to<ImageTexture>() );
		} else {
			texture = Ref<ImageTexture>( memnew( ImageTexture ) );
		}
		texture->create_from_image(atlas,tex_flags);

	} else {
		ERR_FAIL_COND_V(from->get_source_count()!=1,ERR_INVALID_PARAMETER);

		String src_path = EditorImportPlugin::expand_source_path(from->get_source_path(0));

		if (ResourceCache::has(p_path)) {
			Resource *r = ResourceCache::get(p_path);

			texture = Ref<ImageTexture> ( r->cast_to<ImageTexture>() );

			Image img;
			Error err = img.load(src_path);
			ERR_FAIL_COND_V(err!=OK,ERR_CANT_OPEN);
			texture->create_from_image(img);
		} else {
			texture=ResourceLoader::load(src_path,"ImageTexture");
		}

		ERR_FAIL_COND_V(texture.is_null(),ERR_CANT_OPEN);
		if (!p_external)
			from->set_source_md5(0,FileAccess::get_md5(src_path));

	}



	if (!p_external) {
		from->set_editor(get_name());
		texture->set_path(p_path);
		texture->set_import_metadata(from);
	}

	if (atlas) {

		if (p_external) {
			//used by exporter
			Array rects;
			for(int i=0;i<atlases.size();i++) {
				rects.push_back(atlases[i]->get_region());
				rects.push_back(atlases[i]->get_margin());
			}
			from->set_option("rects",rects);

		} else {
			//used by importer
			for(int i=0;i<atlases.size();i++) {
				String apath = atlases[i]->get_path();
				atlases[i]->set_atlas(texture);
				Error err = ResourceSaver::save(apath,atlases[i]);
				if (err) {
					EditorNode::add_io_error(TTR("Couldn't save atlas image:")+" "+apath);
					return err;
				}
				//from->set_source_md5(i,FileAccess::get_md5(apath));
			}
		}
	}

	bool compress=false;
#if 1

	_process_texture_data(texture,format,quality,flags,p_compr,tex_flags,shrink);
#else
	if (format==IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS || format==IMAGE_FORMAT_COMPRESS_DISK_LOSSY) {

		Image image=texture->get_data();
		ERR_FAIL_COND_V(image.empty(),ERR_INVALID_DATA);

		bool has_alpha=image.detect_alpha();
		if (!has_alpha && image.get_format()==Image::FORMAT_RGBA8) {

			image.convert(Image::FORMAT_RGB8);

		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_FIX_BORDER_ALPHA) {

			image.fix_alpha_edges();
		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_PREMULT_ALPHA) {

			image.premultiply_alpha();
		}

		if (flags&IMAGE_FLAG_CONVERT_NORMAL_TO_XY) {
			image.normalmap_to_xy();
		}

		/*
		if ((image.get_format()==Image::FORMAT_RGB8 || image.get_format()==Image::FORMAT_RGBA8) && flags&IMAGE_FLAG_CONVERT_TO_LINEAR) {

			image.srgb_to_linear();
		}
		*/

		if (shrink>1) {

			int orig_w=image.get_width();
			int orig_h=image.get_height();
			image.resize(orig_w/shrink,orig_h/shrink);
			texture->create_from_image(image,tex_flags);
			texture->set_size_override(Size2(orig_w,orig_h));


		} else {

			texture->create_from_image(image,tex_flags);
		}


		if (format==IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS) {
			texture->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSLESS);
		} else {
			texture->set_storage(ImageTexture::STORAGE_COMPRESS_LOSSY);
		}



		texture->set_lossy_storage_quality(quality);


	} else {


		Image image=texture->get_data();
		ERR_FAIL_COND_V(image.empty(),ERR_INVALID_DATA);


		bool has_alpha=image.detect_alpha();
		if (!has_alpha && image.get_format()==Image::FORMAT_RGBA8) {

			image.convert(Image::FORMAT_RGB8);

		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_FIX_BORDER_ALPHA) {

			image.fix_alpha_edges();
		}

		if (image.get_format()==Image::FORMAT_RGBA8 && flags&IMAGE_FLAG_PREMULT_ALPHA) {

			image.premultiply_alpha();
		}

		if (flags&IMAGE_FLAG_CONVERT_NORMAL_TO_XY) {
			image.normalmap_to_xy();
		}

		/*
		if ((image.get_format()==Image::FORMAT_RGB8 || image.get_format()==Image::FORMAT_RGBA8) && flags&IMAGE_FLAG_CONVERT_TO_LINEAR) {

			print_line("CONVERT BECAUSE: "+itos(flags));
			image.srgb_to_linear();
		}
		*/

		int orig_w=image.get_width();
		int orig_h=image.get_height();

		if (shrink>1) {
			image.resize(orig_w/shrink,orig_h/shrink);
			texture->create_from_image(image,tex_flags);
			texture->set_size_override(Size2(orig_w,orig_h));
		}

		if (!(flags&IMAGE_FLAG_NO_MIPMAPS)) {
			image.generate_mipmaps();

		}

		if (format!=IMAGE_FORMAT_UNCOMPRESSED) {

			compress_image(p_compr,image,flags&IMAGE_FLAG_COMPRESS_EXTRA);
		}


		texture->create_from_image(image,tex_flags);


		if (shrink>1 || (format!=IMAGE_FORMAT_UNCOMPRESSED && (image.get_width()!=orig_w || image.get_height()!=orig_h))) {
			texture->set_size_override(Size2(orig_w,orig_h));
		}

		compress=true;


	}
#endif
	uint32_t save_flags=0;
	if (compress)
		save_flags=ResourceSaver::FLAG_COMPRESS;

	Error err = ResourceSaver::save(p_path,texture,save_flags);
	if (err!=OK) {
		EditorNode::add_io_error(TTR("Couldn't save converted texture:")+" "+p_path);
		return err;
	}

	return OK;
}

Vector<uint8_t> EditorTextureImportPlugin::custom_export(const String& p_path, const Ref<EditorExportPlatform> &p_platform) {


	Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_path);

	if (rimd.is_null()) {

		StringName group = EditorImportExport::get_singleton()->image_get_export_group(p_path);

		if (group!=StringName()) {
			//handled by export group
			rimd = Ref<ResourceImportMetadata>( memnew( ResourceImportMetadata ) );

			int group_format=0;
			float group_lossy_quality=EditorImportExport::get_singleton()->image_export_group_get_lossy_quality(group);
			int group_shrink=EditorImportExport::get_singleton()->image_export_group_get_shrink(group);
			group_shrink*=EditorImportExport::get_singleton()->get_export_image_shrink();

			switch(EditorImportExport::get_singleton()->image_export_group_get_image_action(group)) {
				case EditorImportExport::IMAGE_ACTION_NONE: {

					switch(EditorImportExport::get_singleton()->get_export_image_action()) {
						case EditorImportExport::IMAGE_ACTION_NONE: {

							group_format=EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSLESS; //?

						} break; //use default
						case EditorImportExport::IMAGE_ACTION_COMPRESS_DISK: {
							group_format=EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSY;
						} break; //use default
						case EditorImportExport::IMAGE_ACTION_COMPRESS_RAM: {
							group_format=EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM;
						} break; //use default
					}

					group_lossy_quality=EditorImportExport::get_singleton()->get_export_image_quality();

				} break; //use default
				case EditorImportExport::IMAGE_ACTION_COMPRESS_DISK: {
					group_format=EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_DISK_LOSSY;
				} break; //use default
				case EditorImportExport::IMAGE_ACTION_COMPRESS_RAM: {
					group_format=EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM;
				} break; //use default
				case EditorImportExport::IMAGE_ACTION_KEEP: {
					return Vector<uint8_t>();
				} break; //use default
			}

			String validated_path=EditorImportPlugin::validate_source_path(p_path);

			int flags=texture_flags_to_export_flags(ResourceFormatLoaderImage::load_image_flags(validated_path));
			flags|=IMAGE_FLAG_FIX_BORDER_ALPHA;

			print_line("group format"+itos(group_format));
			rimd->set_option("format",group_format);
			rimd->set_option("flags",flags);
			rimd->set_option("quality",group_lossy_quality);
			rimd->set_option("atlas",false);
			rimd->set_option("shrink",group_shrink);
			rimd->add_source(validated_path,FileAccess::get_md5(p_path));

		} else if (EditorImportExport::get_singleton()->get_image_formats().has(p_path.get_extension().to_lower()) && EditorImportExport::get_singleton()->get_export_image_action()!=EditorImportExport::IMAGE_ACTION_NONE) {
			//handled by general image export settings

			rimd = Ref<ResourceImportMetadata>( memnew( ResourceImportMetadata ) );

			switch(EditorImportExport::get_singleton()->get_export_image_action()) {
				case EditorImportExport::IMAGE_ACTION_COMPRESS_DISK: rimd->set_option("format",IMAGE_FORMAT_COMPRESS_DISK_LOSSY); break;
				case EditorImportExport::IMAGE_ACTION_COMPRESS_RAM: rimd->set_option("format",IMAGE_FORMAT_COMPRESS_RAM); break;
			}

			String validated_path=EditorImportPlugin::validate_source_path(p_path);

			int flags=texture_flags_to_export_flags(ResourceFormatLoaderImage::load_image_flags(validated_path));
			flags|=IMAGE_FLAG_FIX_BORDER_ALPHA;

			rimd->set_option("shrink",EditorImportExport::get_singleton()->get_export_image_shrink());
			rimd->set_option("flags",flags);
			rimd->set_option("quality",EditorImportExport::get_singleton()->get_export_image_quality());
			rimd->set_option("atlas",false);
			rimd->add_source(validated_path,FileAccess::get_md5(p_path));

		} else {
			return Vector<uint8_t>();
		}
	}

	int fmt = rimd->get_option("format");

	if (fmt!=IMAGE_FORMAT_COMPRESS_RAM && fmt!=IMAGE_FORMAT_COMPRESS_DISK_LOSSY)  {
		print_line("no compress ram or lossy");
		return Vector<uint8_t>(); //pointless to do anything, since no need to reconvert
	}

	uint32_t flags = rimd->get_option("flags");
	uint8_t shrink = rimd->has_option("shrink") ? rimd->get_option("shrink"): Variant(1);
	uint8_t format = rimd->get_option("format");
	uint8_t comp = (format==EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM)?uint8_t(p_platform->get_image_compression()):uint8_t(255);

	MD5_CTX ctx;
	uint8_t f4[4];
	encode_uint32(flags,&f4[0]);
	MD5Init(&ctx);
	String gp = GlobalConfig::get_singleton()->globalize_path(p_path);
	CharString cs = gp.utf8();
	MD5Update(&ctx,(unsigned char*)cs.get_data(),cs.length());
	MD5Update(&ctx,f4,4);
	MD5Update(&ctx,&format,1);
	MD5Update(&ctx,&comp,1);
	MD5Update(&ctx,&shrink,1);
	MD5Final(&ctx);



	uint64_t sd=0;
	String smd5;

	String md5 = String::md5(ctx.digest);
	print_line(p_path+" MD5: "+md5+" FLAGS: "+itos(flags));

	String tmp_path = EditorSettings::get_singleton()->get_settings_path().plus_file("tmp/");

	bool valid=false;
	{
		//if existing, make sure it's valid
		FileAccessRef f = FileAccess::open(tmp_path+"imgexp-"+md5+".txt",FileAccess::READ);
		if (f) {

			uint64_t d = f->get_line().strip_edges().to_int64();
			sd = FileAccess::get_modified_time(p_path);

			if (d==sd) {
				valid=true;
			} else {
				String cmd5 = f->get_line().strip_edges();
				smd5 = FileAccess::get_md5(p_path);
				if (cmd5==smd5) {
					valid=true;
				}
			}


		}
	}

	if (!valid) {
		//cache failed, convert
		Error err = import2(tmp_path+"imgexp-"+md5+".tex",rimd,p_platform->get_image_compression(),true);
		ERR_FAIL_COND_V(err!=OK,Vector<uint8_t>());
		FileAccessRef f = FileAccess::open(tmp_path+"imgexp-"+md5+".txt",FileAccess::WRITE);

		if (sd==0)
			sd = FileAccess::get_modified_time(p_path);
		if (smd5==String())
			smd5 = FileAccess::get_md5(p_path);

		f->store_line(String::num(sd));
		f->store_line(smd5);
		f->store_line(gp); //source path for reference
	}


	Vector<uint8_t> ret;
	FileAccessRef f = FileAccess::open(tmp_path+"imgexp-"+md5+".tex",FileAccess::READ);
	ERR_FAIL_COND_V(!f,ret);

	ret.resize(f->get_len());
	f->get_buffer(ret.ptr(),ret.size());

	return ret;
}

uint32_t EditorTextureImportPlugin::texture_flags_to_export_flags(uint32_t p_tex_flags) const {

	uint32_t flags=0;

	if (!(p_tex_flags&Texture::FLAG_MIPMAPS)) {
		flags|=IMAGE_FLAG_NO_MIPMAPS;
	}
	if (p_tex_flags&Texture::FLAG_REPEAT) {
		flags|=IMAGE_FLAG_REPEAT;
	}
	if (p_tex_flags&Texture::FLAG_FILTER) {
		flags|=IMAGE_FLAG_FILTER;
	}
	if (p_tex_flags&Texture::FLAG_ANISOTROPIC_FILTER) {
		flags|=IMAGE_FLAG_USE_ANISOTROPY;
	}
	if (p_tex_flags&Texture::FLAG_CONVERT_TO_LINEAR) {
		flags|=IMAGE_FLAG_CONVERT_TO_LINEAR;
	}
	/* // no correspondence yet
	if (p_tex_flags&Texture::TEXTURE_FLAG_MIRRORED_REPEAT) {
		flags|=;
	}*/

	return flags;
}

void EditorTextureImportPlugin::import_from_drop(const Vector<String>& p_drop,const String& p_dest_path) {

	Vector<String> valid;

	List<String> valid_extensions;
	ImageLoader::get_recognized_extensions(&valid_extensions);
	for(int i=0;i<p_drop.size();i++) {

		String extension=p_drop[i].get_extension().to_lower();

		for (List<String>::Element *E=valid_extensions.front();E;E=E->next()) {

			if (E->get()==extension) {
				valid.push_back(p_drop[i]);
				break;
			}
		}
	}

	if (valid.size()) {
		dialog->popup_import();
		dialog->setup_multiple_import_3d(valid,p_dest_path);
	}
}

void EditorTextureImportPlugin::reimport_multiple_files(const Vector<String>& p_list) {

	Vector<String> valid;


	for(int i=0;i<p_list.size();i++) {

		Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_list[i]);
		String type = rimd->get_editor();
		if (type=="texture" || type.begins_with("texture_")) {

			if ((rimd->has_option("atlas") && rimd->get_option("atlas")) || (rimd->has_option("large") && rimd->get_option("large"))) {
				continue;
			}

			valid.push_back(p_list[i]);
		}
	}

	if (valid.size()) {

		dialog->popup_import(valid[0]);

		Vector<String> sources;
		for(int i=0;i<valid.size();i++) {
			int idx;
			EditorFileSystemDirectory *efsd = EditorFileSystem::get_singleton()->find_file(valid[i],&idx);
			if (efsd) {
				for(int j=0;j<efsd->get_source_count(idx);j++) {
					String file = expand_source_path(efsd->get_source_file(idx,j));
					if (sources.find(file)==-1) {
						sources.push_back(file);
					}

				}
			}
		}

		if (sources.size()) {

			dialog->add_sources_and_dest(sources,valid[0].get_base_dir());
		}
	}
}

bool EditorTextureImportPlugin::can_reimport_multiple_files() const {

	return true;

}



EditorTextureImportPlugin *EditorTextureImportPlugin::singleton=NULL;

EditorTextureImportPlugin::EditorTextureImportPlugin(EditorNode *p_editor) {

	singleton=this;
	editor=p_editor;
	dialog = memnew( EditorTextureImportDialog(this) );
	editor->get_gui_base()->add_child(dialog);

}

////////////////////////////


 Vector<uint8_t> EditorTextureExportPlugin::custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform) {

	Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_path);

	if (rimd.is_valid()) {

		if (rimd->get_editor()!="") {
			int compression = rimd->get_option("format");
			if (compression!=EditorTextureImportPlugin::IMAGE_FORMAT_COMPRESS_RAM)
				return Vector<uint8_t>(); //only useful for RAM compression to reconvert
			Ref<EditorImportPlugin> pl = EditorImportExport::get_singleton()->get_import_plugin_by_name(rimd->get_editor());
			if (pl.is_valid()) {
				Vector<uint8_t> ce = pl->custom_export(p_path,p_platform);
				if (ce.size())
					return ce;
			}
		}
	} else if (EditorImportExport::get_singleton()->image_get_export_group(p_path)) {


		Ref<EditorImportPlugin> pl = EditorImportExport::get_singleton()->get_import_plugin_by_name("texture");
		if (pl.is_valid()) {
			Vector<uint8_t> ce = pl->custom_export(p_path,p_platform);
			if (ce.size()) {
				p_path=p_path.get_basename()+".converted.tex";
				return ce;
			}
		}

	} else if (EditorImportExport::get_singleton()->get_export_image_action()!=EditorImportExport::IMAGE_ACTION_NONE){

		String xt = p_path.get_extension().to_lower();
		if (EditorImportExport::get_singleton()->get_image_formats().has(xt)) { //should check for more I guess?

			Ref<EditorImportPlugin> pl = EditorImportExport::get_singleton()->get_import_plugin_by_name("texture");
			if (pl.is_valid()) {
				Vector<uint8_t> ce = pl->custom_export(p_path,p_platform);
				if (ce.size()) {
					p_path=p_path.get_basename()+".converted.tex";
					return ce;
				}
			}
		}
	}

	return Vector<uint8_t>();
}

EditorTextureExportPlugin::EditorTextureExportPlugin() {


}
#endif
