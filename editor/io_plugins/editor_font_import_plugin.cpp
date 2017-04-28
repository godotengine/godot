/*************************************************************************/
/*  editor_font_import_plugin.cpp                                        */
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
#include "editor_font_import_plugin.h"
#if 0
#include "editor/editor_file_dialog.h"
#include "editor/editor_node.h"
#include "editor_atlas.h"
#include "io/image_loader.h"
#include "io/resource_saver.h"
#include "os/file_access.h"
#include "scene/gui/dialogs.h"

#ifdef FREETYPE_ENABLED
#include <ft2build.h>
#include FT_FREETYPE_H
#endif


class _EditorFontImportOptions : public Object {

	GDCLASS(_EditorFontImportOptions,Object);
public:

	enum FontMode {

		FONT_BITMAP,
		FONT_DISTANCE_FIELD
	};

	enum ColorType {
		COLOR_WHITE,
		COLOR_CUSTOM,
		COLOR_GRADIENT_RANGE,
		COLOR_GRADIENT_IMAGE
	};


	int char_extra_spacing;
	int top_extra_spacing;
	int bottom_extra_spacing;
	int space_extra_spacing;

	enum CharacterSet {

		CHARSET_ASCII,
		CHARSET_LATIN,
		CHARSET_UNICODE,
		CHARSET_CUSTOM,
		CHARSET_CUSTOM_LATIN
	};


	FontMode font_mode;

	CharacterSet character_set;
	String custom_file;

	bool shadow;
	Vector2 shadow_offset;
	int shadow_radius;
	Color shadow_color;
	float shadow_transition;

	bool shadow2;
	Vector2 shadow2_offset;
	int shadow2_radius;
	Color shadow2_color;
	float shadow2_transition;

	ColorType color_type;
	Color color;
	Color gradient_begin;
	Color gradient_end;
	bool color_use_monochrome;
	String gradient_image;

	bool enable_filter;
	bool round_advance;
	bool premultiply_alpha;



	bool _set(const StringName& p_name, const Variant& p_value) {

		String n = p_name;
		if (n=="mode/mode") {
			font_mode=FontMode(int(p_value));
			_change_notify();
		} else if (n=="extra_space/char")
			char_extra_spacing=p_value;
		else if (n=="extra_space/space")
			space_extra_spacing=p_value;
		else if (n=="extra_space/top")
			top_extra_spacing=p_value;
		else if (n=="extra_space/bottom")
			bottom_extra_spacing=p_value;

		else if (n=="character_set/mode") {
			character_set=CharacterSet(int(p_value));
			_change_notify();
		} else if (n=="character_set/custom")
			custom_file=p_value;

		else if (n=="shadow/enabled") {
			shadow=p_value;
			_change_notify();
		}else if (n=="shadow/radius")
			shadow_radius=p_value;
		else if (n=="shadow/offset")
			shadow_offset=p_value;
		else if (n=="shadow/color")
			shadow_color=p_value;
		else if (n=="shadow/transition")
			shadow_transition=p_value;

		else if (n=="shadow2/enabled") {
			shadow2=p_value;
			_change_notify();
		}else if (n=="shadow2/radius")
			shadow2_radius=p_value;
		else if (n=="shadow2/offset")
			shadow2_offset=p_value;
		else if (n=="shadow2/color")
			shadow2_color=p_value;
		else if (n=="shadow2/transition")
			shadow2_transition=p_value;

		else if (n=="color/mode") {
			color_type=ColorType(int(p_value));
			_change_notify();
		}else if (n=="color/color")
			color=p_value;
		else if (n=="color/begin")
			gradient_begin=p_value;
		else if (n=="color/end")
			gradient_end=p_value;
		else if (n=="color/image")
			gradient_image=p_value;
		else if (n=="color/monochrome")
			color_use_monochrome=p_value;
		else if (n=="advanced/round_advance")
			round_advance=p_value;
		else if (n=="advanced/enable_filter")
			enable_filter=p_value;
		else if (n=="advanced/premultiply_alpha")
			premultiply_alpha=p_value;
		else
			return false;

		emit_signal("changed");


		return true;

	}

	bool _get(const StringName& p_name,Variant &r_ret) const{

		String n = p_name;
		if (n=="mode/mode")
			r_ret=font_mode;
		else if (n=="extra_space/char")
			r_ret=char_extra_spacing;
		else if (n=="extra_space/space")
			r_ret=space_extra_spacing;
		else if (n=="extra_space/top")
			r_ret=top_extra_spacing;
		else if (n=="extra_space/bottom")
			r_ret=bottom_extra_spacing;

		else if (n=="character_set/mode")
			r_ret=character_set;
		else if (n=="character_set/custom")
			r_ret=custom_file;

		else if (n=="shadow/enabled")
			r_ret=shadow;
		else if (n=="shadow/radius")
			r_ret=shadow_radius;
		else if (n=="shadow/offset")
			r_ret=shadow_offset;
		else if (n=="shadow/color")
			r_ret=shadow_color;
		else if (n=="shadow/transition")
			r_ret=shadow_transition;

		else if (n=="shadow2/enabled")
			r_ret=shadow2;
		else if (n=="shadow2/radius")
			r_ret=shadow2_radius;
		else if (n=="shadow2/offset")
			r_ret=shadow2_offset;
		else if (n=="shadow2/color")
			r_ret=shadow2_color;
		else if (n=="shadow2/transition")
			r_ret=shadow2_transition;


		else if (n=="color/mode")
			r_ret=color_type;
		else if (n=="color/color")
			r_ret=color;
		else if (n=="color/begin")
			r_ret=gradient_begin;
		else if (n=="color/end")
			r_ret=gradient_end;
		else if (n=="color/image")
			r_ret=gradient_image;
		else if (n=="color/monochrome")
			r_ret=color_use_monochrome;
		else if (n=="advanced/round_advance")
			r_ret=round_advance;
		else if (n=="advanced/enable_filter")
			r_ret=enable_filter;
		else if (n=="advanced/premultiply_alpha")
			r_ret=premultiply_alpha;
		else
			return false;

		return true;

	}

	void _get_property_list( List<PropertyInfo> *p_list) const{


		p_list->push_back(PropertyInfo(Variant::INT,"mode/mode",PROPERTY_HINT_ENUM,"Bitmap,Distance Field"));

		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/char",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/space",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/top",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/bottom",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"character_set/mode",PROPERTY_HINT_ENUM,"Ascii,Latin,Unicode,Custom,Custom&Latin"));

		if (character_set>=CHARSET_CUSTOM)
			p_list->push_back(PropertyInfo(Variant::STRING,"character_set/custom",PROPERTY_HINT_GLOBAL_FILE));

		int usage = PROPERTY_USAGE_DEFAULT;

		if (font_mode==FONT_DISTANCE_FIELD) {
			usage = PROPERTY_USAGE_NOEDITOR;
		}

		{

			p_list->push_back(PropertyInfo(Variant::BOOL,"shadow/enabled",PROPERTY_HINT_NONE,"",usage));
			if (shadow) {
				p_list->push_back(PropertyInfo(Variant::INT,"shadow/radius",PROPERTY_HINT_RANGE,"-64,64,1",usage));
				p_list->push_back(PropertyInfo(Variant::VECTOR2,"shadow/offset",PROPERTY_HINT_NONE,"",usage));
				p_list->push_back(PropertyInfo(Variant::COLOR,"shadow/color",PROPERTY_HINT_NONE,"",usage));
				p_list->push_back(PropertyInfo(Variant::REAL,"shadow/transition",PROPERTY_HINT_EXP_EASING,"",usage));
			}

			p_list->push_back(PropertyInfo(Variant::BOOL,"shadow2/enabled",PROPERTY_HINT_NONE,"",usage));
			if (shadow2) {
				p_list->push_back(PropertyInfo(Variant::INT,"shadow2/radius",PROPERTY_HINT_RANGE,"-64,64,1",usage));
				p_list->push_back(PropertyInfo(Variant::VECTOR2,"shadow2/offset",PROPERTY_HINT_NONE,"",usage));
				p_list->push_back(PropertyInfo(Variant::COLOR,"shadow2/color",PROPERTY_HINT_NONE,"",usage));
				p_list->push_back(PropertyInfo(Variant::REAL,"shadow2/transition",PROPERTY_HINT_EXP_EASING,"",usage));
			}

			p_list->push_back(PropertyInfo(Variant::INT,"color/mode",PROPERTY_HINT_ENUM,"White,Color,Gradient,Gradient Image",usage));
			if (color_type==COLOR_CUSTOM) {
				p_list->push_back(PropertyInfo(Variant::COLOR,"color/color",PROPERTY_HINT_NONE,"",usage));

			}
			if (color_type==COLOR_GRADIENT_RANGE) {
				p_list->push_back(PropertyInfo(Variant::COLOR,"color/begin",PROPERTY_HINT_NONE,"",usage));
				p_list->push_back(PropertyInfo(Variant::COLOR,"color/end",PROPERTY_HINT_NONE,"",usage));
			}
			if (color_type==COLOR_GRADIENT_IMAGE) {
				p_list->push_back(PropertyInfo(Variant::STRING,"color/image",PROPERTY_HINT_GLOBAL_FILE,"",usage));
			}
			p_list->push_back(PropertyInfo(Variant::BOOL,"color/monochrome",PROPERTY_HINT_NONE,"",usage));
		}

		p_list->push_back(PropertyInfo(Variant::BOOL,"advanced/round_advance"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"advanced/enable_filter"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"advanced/premultiply_alpha"));

	}


	static void _bind_methods() {


		ADD_SIGNAL( MethodInfo("changed"));
	}


	void reset() {

		char_extra_spacing=0;
		top_extra_spacing=0;
		bottom_extra_spacing=0;
		space_extra_spacing=0;

		character_set=CHARSET_LATIN;

		shadow=false;
		shadow_radius=2;
		shadow_color=Color(0,0,0,0.3);
		shadow_transition=1.0;

		shadow2=false;
		shadow2_radius=2;
		shadow2_color=Color(0,0,0,0.3);
		shadow2_transition=1.0;

		color_type=COLOR_WHITE;
		color=Color(1,1,1,1);
		gradient_begin=Color(1,1,1,1);
		gradient_end=Color(0.5,0.5,0.5,1);
		color_use_monochrome=false;

		font_mode=FONT_BITMAP;
		round_advance=true;
		enable_filter=true;
		premultiply_alpha=false;

	}

	_EditorFontImportOptions() {

		font_mode=FONT_BITMAP;

		char_extra_spacing=0;
		top_extra_spacing=0;
		bottom_extra_spacing=0;
		space_extra_spacing=0;

		character_set=CHARSET_LATIN;

		shadow=false;
		shadow_radius=2;
		shadow_color=Color(0,0,0,0.3);
		shadow_transition=1.0;

		shadow2=false;
		shadow2_radius=2;
		shadow2_color=Color(0,0,0,0.3);
		shadow2_transition=1.0;

		color_type=COLOR_WHITE;
		color=Color(1,1,1,1);
		gradient_begin=Color(1,1,1,1);
		gradient_end=Color(0.5,0.5,0.5,1);
		color_use_monochrome=false;

		round_advance=true;
		enable_filter=true;
		premultiply_alpha=false;
	}


};


class EditorFontImportDialog : public ConfirmationDialog {

	GDCLASS(EditorFontImportDialog, ConfirmationDialog);


	EditorLineEditFileChooser *source;
	EditorLineEditFileChooser *dest;
	SpinBox *font_size;
	LineEdit *test_string;
	ColorPickerButton *test_color;
	Label *test_label;
	PropertyEditor *prop_edit;
	Timer *timer;
	ConfirmationDialog *error_dialog;


	Ref<ResourceImportMetadata> get_rimd() {

		Ref<ResourceImportMetadata> imd = memnew( ResourceImportMetadata );
		List<PropertyInfo> pl;
		options->_get_property_list(&pl);
		for(List<PropertyInfo>::Element *E=pl.front();E;E=E->next()) {

			Variant v;
			String opt=E->get().name;
			options->_get(opt,v);
			if (opt=="color/image" || opt=="character_set/custom") {
				v = EditorImportPlugin::validate_source_path(v);
			}
			imd->set_option(opt,v);
		}

		String src_path = EditorImportPlugin::validate_source_path(source->get_line_edit()->get_text());
		//print_line("pre src path "+source->get_line_edit()->get_text());
		//print_line("src path "+src_path);
		imd->add_source(src_path);
		imd->set_option("font/size",font_size->get_value());

		return imd;

	}

	void _src_changed(String) {
		_prop_changed();
	}

	void _update_text2(String) {
		_update_text();
	}
	void _update_text3(Color) {
		_update_text();
	}

	void _update_text() {

		test_label->set_text("");
		test_label->set_text(test_string->get_text());
		test_label->add_color_override("font_color",test_color->get_pick_color());
	}

	void _update() {

		Ref<ResourceImportMetadata> imd = get_rimd();
		Ref<BitmapFont> font = plugin->generate_font(imd);
		test_label->add_font_override("font",font);
		_update_text();
	}

	void _font_size_changed(double) {

		_prop_changed();
	}

	void _prop_changed() {

		timer->start();
	}

	void _import_inc(String p_font) {

		Ref<BitmapFont> font = ResourceLoader::load(p_font);
		if (!font.is_valid())
			return;
		Ref<ImageTexture> tex = font->get_texture(0);
		if (tex.is_null())
			return;
		FileAccessRef f=FileAccess::open(p_font.get_basename()+".inc",FileAccess::WRITE);
		Vector<CharType> ck = font->get_char_keys();

		f->store_line("static const int _builtin_font_height="+itos(font->get_height())+";");
		f->store_line("static const int _builtin_font_ascent="+itos(font->get_ascent())+";");
		f->store_line("static const int _builtin_font_charcount="+itos(ck.size())+";");
		f->store_line("static const int _builtin_font_charrects["+itos(ck.size())+"][8]={");
		f->store_line("/* charidx , ofs_x, ofs_y, size_x, size_y, valign, halign, advance */");

		for(int i=0;i<ck.size();i++) {
			CharType k=ck[i];
			BitmapFont::Character c=font->get_character(k);
			f->store_line("{"+itos(k)+","+rtos(c.rect.pos.x)+","+rtos(c.rect.pos.y)+","+rtos(c.rect.size.x)+","+rtos(c.rect.size.y)+","+rtos(c.v_align)+","+rtos(c.h_align)+","+rtos(c.advance)+"},");
		}
		f->store_line("};");

		Vector<BitmapFont::KerningPairKey> kp=font->get_kerning_pair_keys();
		f->store_line("static const int _builtin_font_kerning_pair_count="+itos(kp.size())+";");
		f->store_line("static const int _builtin_font_kerning_pairs["+itos(kp.size())+"][3]={");
		for(int i=0;i<kp.size();i++) {

			int d = font->get_kerning_pair(kp[i].A,kp[i].B);
			f->store_line("{"+itos(kp[i].A)+","+itos(kp[i].B)+","+itos(d)+"},");
		}

		f->store_line("};");
		Image img  = tex->get_data();

		f->store_line("static const int _builtin_font_img_width="+itos(img.get_width())+";");
		f->store_line("static const int _builtin_font_img_height="+itos(img.get_height())+";");		

		String fname = p_font.get_basename()+".sv.png";
		ResourceSaver::save(fname,tex);
		Vector<uint8_t> data=FileAccess::get_file_as_array(fname);


		f->store_line("static const int _builtin_font_img_data_size="+itos(data.size())+";");
		f->store_line("static const unsigned char _builtin_font_img_data["+itos(data.size())+"]={");



		for(int i=0;i<data.size();i++) {

			f->store_line(itos(data[i])+",");

		}
		f->store_line("};");

	}

	void _import() {

		if (source->get_line_edit()->get_text()=="") {
			error_dialog->set_text(TTR("No source font file!"));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;
		}

		if (dest->get_line_edit()->get_text()=="") {
			error_dialog->set_text(TTR("No target font resource!"));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;
		}

		if (dest->get_line_edit()->get_text().get_file()==".fnt") {
			dest->get_line_edit()->set_text(dest->get_line_edit()->get_text().get_base_dir() + "/" + source->get_line_edit()->get_text().get_file().get_basename() + ".fnt" );
		}

		if (dest->get_line_edit()->get_text().get_extension() == dest->get_line_edit()->get_text()) {
			dest->get_line_edit()->set_text(dest->get_line_edit()->get_text() + ".fnt");
		}

		if (dest->get_line_edit()->get_text().get_extension().to_lower() != "fnt") {
			error_dialog->set_text(TTR("Invalid file extension.\nPlease use .fnt."));
			error_dialog->popup_centered(Size2(200,100));
			return;
		}

		Ref<ResourceImportMetadata> rimd = get_rimd();

		if (rimd.is_null()) {
			error_dialog->set_text(TTR("Can't load/process source font."));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;
		}

		Error err = plugin->import(dest->get_line_edit()->get_text(),rimd);

		if (err!=OK) {
			error_dialog->set_text(TTR("Couldn't save font."));
			error_dialog->popup_centered(Size2(200,100)*EDSCALE);
			return;
		}

		_import_inc(dest->get_line_edit()->get_text());

		hide();
	}

	EditorFontImportPlugin *plugin;
	_EditorFontImportOptions *options;

	static void _bind_methods() {

		ClassDB::bind_method("_update",&EditorFontImportDialog::_update);
		ClassDB::bind_method("_update_text",&EditorFontImportDialog::_update_text);
		ClassDB::bind_method("_update_text2",&EditorFontImportDialog::_update_text2);
		ClassDB::bind_method("_update_text3",&EditorFontImportDialog::_update_text3);
		ClassDB::bind_method("_prop_changed",&EditorFontImportDialog::_prop_changed);
		ClassDB::bind_method("_src_changed",&EditorFontImportDialog::_src_changed);
		ClassDB::bind_method("_font_size_changed",&EditorFontImportDialog::_font_size_changed);
		ClassDB::bind_method("_import",&EditorFontImportDialog::_import);

	}

public:

	void _notification(int p_what) {

		if (p_what==NOTIFICATION_ENTER_TREE) {
			prop_edit->edit(options);
			_update_text();
		}
	}

	void popup_import(const String& p_path) {

		popup_centered(Size2(600,500)*EDSCALE);

		if (p_path!="") {

			Ref<ResourceImportMetadata> rimd = ResourceLoader::load_import_metadata(p_path);
			ERR_FAIL_COND(!rimd.is_valid());

			dest->get_line_edit()->set_text(p_path);
			List<String> opts;
			rimd->get_options(&opts);
			options->reset();
			for(List<String>::Element *E=opts.front();E;E=E->next()) {

				options->_set(E->get(),rimd->get_option(E->get()));
			}

			String src = "";
			for(int i=0;i<rimd->get_source_count();i++) {
				if (i>0)
					src+=",";
				src+=EditorImportPlugin::expand_source_path(rimd->get_source_path(i));
			}
			source->get_line_edit()->set_text(src);

			font_size->set_value(rimd->get_option("font/size"));
		}
	}


	void set_source_and_dest(const String& p_font,const String& p_dest) {
		source->get_line_edit()->set_text(p_font);
		dest->get_line_edit()->set_text(p_dest);
		_prop_changed();
	}

	EditorFontImportDialog(EditorFontImportPlugin *p_plugin) {
		plugin=p_plugin;
		VBoxContainer *vbc = memnew( VBoxContainer );
		add_child(vbc);
		//set_child_rect(vbc);
		HBoxContainer *hbc = memnew( HBoxContainer);
		vbc->add_child(hbc);
		VBoxContainer *vbl = memnew( VBoxContainer );
		hbc->add_child(vbl);
		hbc->set_v_size_flags(SIZE_EXPAND_FILL);
		vbl->set_h_size_flags(SIZE_EXPAND_FILL);
		VBoxContainer *vbr = memnew( VBoxContainer );
		hbc->add_child(vbr);
		vbr->set_h_size_flags(SIZE_EXPAND_FILL);

		source = memnew( EditorLineEditFileChooser );
		source->get_file_dialog()->set_access(EditorFileDialog::ACCESS_FILESYSTEM);
		source->get_file_dialog()->set_mode(EditorFileDialog::MODE_OPEN_FILE);
		source->get_file_dialog()->add_filter("*.ttf;TrueType");
		source->get_file_dialog()->add_filter("*.otf;OpenType");
		source->get_file_dialog()->add_filter("*.fnt;BMFont");
		source->get_line_edit()->connect("text_entered",this,"_src_changed");

		vbl->add_margin_child(TTR("Source Font:"),source);
		font_size = memnew( SpinBox );
		vbl->add_margin_child(TTR("Source Font Size:"),font_size);
		font_size->set_min(3);
		font_size->set_max(256);
		font_size->set_value(16);
		font_size->connect("value_changed",this,"_font_size_changed");
		dest = memnew( EditorLineEditFileChooser );
		//
		List<String> fl;
		Ref<BitmapFont> font= memnew(BitmapFont);
		dest->get_file_dialog()->add_filter("*.fnt ; Font" );
		/*
		ResourceSaver::get_recognized_extensions(font,&fl);
		for(List<String>::Element *E=fl.front();E;E=E->next()) {
			dest->get_file_dialog()->add_filter("*."+E->get());
		}
		*/

		vbl->add_margin_child(TTR("Dest Resource:"),dest);
		HBoxContainer *testhb = memnew( HBoxContainer );
		test_string = memnew( LineEdit );
		test_string->set_text(TTR("The quick brown fox jumps over the lazy dog."));
		test_string->set_h_size_flags(SIZE_EXPAND_FILL);
		test_string->set_stretch_ratio(5);

		testhb->add_child(test_string);
		test_color = memnew( ColorPickerButton );
		test_color->set_pick_color(get_color("font_color","Label"));
		test_color->set_h_size_flags(SIZE_EXPAND_FILL);
		test_color->set_stretch_ratio(1);
		test_color->connect("color_changed",this,"_update_text3");
		testhb->add_child(test_color);

		vbl->add_spacer();
		vbl->add_margin_child(TTR("Test:")+" ",testhb);
		/*
		HBoxContainer *upd_hb = memnew( HBoxContainer );
		//vbl->add_child(upd_hb);
		upd_hb->add_spacer();
		Button *update = memnew( Button);
		upd_hb->add_child(update);
		update->set_text("Update");
		update->connect("pressed",this,"_update");
*/
		options = memnew( _EditorFontImportOptions );
		prop_edit = memnew( PropertyEditor() );
		vbr->add_margin_child(TTR("Options:"),prop_edit,true);
		options->connect("changed",this,"_prop_changed");

		prop_edit->hide_top_label();

		Panel *panel = memnew( Panel );
		vbc->add_child(panel);
		test_label = memnew( Label );
		test_label->set_autowrap(true);
		panel->add_child(test_label);
		test_label->set_area_as_parent_rect();
		panel->set_v_size_flags(SIZE_EXPAND_FILL);
		test_string->connect("text_changed",this,"_update_text2");
		set_title(TTR("Font Import"));
		timer = memnew( Timer );
		add_child(timer);
		timer->connect("timeout",this,"_update");
		timer->set_wait_time(0.4);
		timer->set_one_shot(true);

		get_ok()->connect("pressed", this,"_import");
		get_ok()->set_text(TTR("Import"));

		error_dialog = memnew ( ConfirmationDialog );
		add_child(error_dialog);
		error_dialog->get_ok()->set_text(TTR("Accept"));
		set_hide_on_ok(false);


	}

	~EditorFontImportDialog() {
		memdelete(options);
	}
};


///////////////////////////////////////



struct _EditorFontData {

	Vector<uint8_t> bitmap;
	int width,height;
	int ofs_x; //offset to center, from ABOVE
	int ofs_y; //offset to beginning, from LEFT
	int valign; //vertical alignment
	int halign;
	float advance;
	int character;
	int glyph;

	int texture;
	Image blit;
	Point2i blit_ofs;
	//bool printable;

};


struct _EditorFontDataSort {

	bool operator()(const _EditorFontData *p_A,const _EditorFontData *p_B)  const {
		return p_A->height > p_B->height;
	};
};

struct _EditorKerningKey {

	CharType A,B;
	bool operator<(const _EditorKerningKey& p_k) const { return (A==p_k.A)?(B<p_k.B):(A<p_k.A); }

};


static unsigned char get_SDF_radial(
		unsigned char *fontmap,
		int w, int h,
		int x, int y,
		int max_radius )
{
	//hideous brute force method
	float d2 = max_radius*max_radius+1.0;
	unsigned char v = fontmap[x+y*w];
	for( int radius = 1; (radius <= max_radius) && (radius*radius < d2); ++radius )
	{
		int line, lo, hi;
		//north
		line = y - radius;
		if( (line >= 0) && (line < h) )
		{
			lo = x - radius;
			hi = x + radius;
			if( lo < 0 ) { lo = 0; }
			if( hi >= w ) { hi = w-1; }
			int idx = line * w + lo;
			for( int i = lo; i <= hi; ++i )
			{
				//check this pixel
				if( fontmap[idx] != v )
				{
					float nx = i - x;
					float ny = line - y;
					float nd2 = nx*nx+ny*ny;
					if( nd2 < d2 )
					{
						d2 = nd2;
					}
				}
				//move on
				++idx;
			}
		}
		//south
		line = y + radius;
		if( (line >= 0) && (line < h) )
		{
			lo = x - radius;
			hi = x + radius;
			if( lo < 0 ) { lo = 0; }
			if( hi >= w ) { hi = w-1; }
			int idx = line * w + lo;
			for( int i = lo; i <= hi; ++i )
			{
				//check this pixel
				if( fontmap[idx] != v )
				{
					float nx = i - x;
					float ny = line - y;
					float nd2 = nx*nx+ny*ny;
					if( nd2 < d2 )
					{
						d2 = nd2;
					}
				}
				//move on
				++idx;
			}
		}
		//west
		line = x - radius;
		if( (line >= 0) && (line < w) )
		{
			lo = y - radius + 1;
			hi = y + radius - 1;
			if( lo < 0 ) { lo = 0; }
			if( hi >= h ) { hi = h-1; }
			int idx = lo * w + line;
			for( int i = lo; i <= hi; ++i )
			{
				//check this pixel
				if( fontmap[idx] != v )
				{
					float nx = line - x;
					float ny = i - y;
					float nd2 = nx*nx+ny*ny;
					if( nd2 < d2 )
					{
						d2 = nd2;
					}
				}
				//move on
				idx += w;
			}
		}
		//east
		line = x + radius;
		if( (line >= 0) && (line < w) )
		{
			lo = y - radius + 1;
			hi = y + radius - 1;
			if( lo < 0 ) { lo = 0; }
			if( hi >= h ) { hi = h-1; }
			int idx = lo * w + line;
			for( int i = lo; i <= hi; ++i )
			{
				//check this pixel
				if( fontmap[idx] != v )
				{
					float nx = line - x;
					float ny = i - y;
					float nd2 = nx*nx+ny*ny;
					if( nd2 < d2 )
					{
						d2 = nd2;
					}
				}
				//move on
				idx += w;
			}
		}
	}
	d2 = sqrtf( d2 );
	if( v==0 )
	{
		d2 = -d2;
	}
	d2 *= 127.5 / max_radius;
	d2 += 127.5;
	if( d2 < 0.0 ) d2 = 0.0;
	if( d2 > 255.0 ) d2 = 255.0;
	return (unsigned char)(d2 + 0.5);
}


Ref<BitmapFont> EditorFontImportPlugin::generate_font(const Ref<ResourceImportMetadata>& p_from, const String &p_existing) {



	Ref<ResourceImportMetadata> from = p_from;
	ERR_FAIL_COND_V(from->get_source_count()!=1,Ref<BitmapFont>());

	String src_path = EditorImportPlugin::expand_source_path(from->get_source_path(0));

	if (src_path.get_extension().to_lower()=="fnt") {

		if (ResourceLoader::load(src_path).is_valid()) {
			EditorNode::get_singleton()->show_warning(TTR("Path:")+" "+src_path+"\n"+TTR("This file is already a Godot font file, please supply a BMFont type file instead."));
			return Ref<BitmapFont>();
		}

		Ref<BitmapFont> font;
		font.instance();
		Error err = font->create_from_fnt(src_path);
		if (err) {
			EditorNode::get_singleton()->show_warning(TTR("Path:")+" "+src_path+"\n"+TTR("Failed opening as BMFont file."));
			return Ref<BitmapFont>();
		}

		return font;
	}

	int size = from->get_option("font/size");

#ifdef FREETYPE_ENABLED
	FT_Library   library;   /* handle to library     */
	FT_Face      face;      /* handle to face object */

	Vector<_EditorFontData*> font_data_list;

	int error = FT_Init_FreeType( &library );

	ERR_EXPLAIN(TTR("Error initializing FreeType."));
	ERR_FAIL_COND_V( error !=0, Ref<BitmapFont>() );

	print_line("loadfrom: "+src_path);
	error = FT_New_Face( library, src_path.utf8().get_data(),0,&face );

	if ( error == FT_Err_Unknown_File_Format ) {
		ERR_EXPLAIN(TTR("Unknown font format."));
		FT_Done_FreeType( library );
	} else if ( error ) {

		ERR_EXPLAIN(TTR("Error loading font."));
		FT_Done_FreeType( library );

	}

	ERR_FAIL_COND_V(error,Ref<BitmapFont>());


	int height=0;
	int ascent=0;
	int font_spacing=0;

	error = FT_Set_Char_Size(face,0,64*size,512,512);

	if ( error ) {
		FT_Done_FreeType( library );
		ERR_EXPLAIN(TTR("Invalid font size."));
		ERR_FAIL_COND_V( error,Ref<BitmapFont>() );

	}

	int font_mode = from->get_option("mode/mode");

	int scaler=(font_mode==_EditorFontImportOptions::FONT_DISTANCE_FIELD)?16:1;

	error = FT_Set_Pixel_Sizes(face,0,size*scaler);

	FT_GlyphSlot slot = face->glyph;

	//error = FT_Set_Charmap(face,ft_encoding_unicode );   /* encoding..         */


	/* PRINT CHARACTERS TO INDIVIDUAL BITMAPS */


	//int space_size=5; //size for space, if none found.. 5!
	//int min_valign=500; //some ridiculous number

	FT_ULong  charcode;
	FT_UInt   gindex;

	int max_up=-1324345; ///gibberish
	int max_down=124232;

	Map<_EditorKerningKey,int> kerning_map;

	charcode = FT_Get_First_Char( face, &gindex );

	Set<CharType> import_chars;

	int import_mode = from->get_option("character_set/mode");
	bool round_advance = from->get_option("advanced/round_advance");

	if (import_mode>=_EditorFontImportOptions::CHARSET_CUSTOM) {

		//load from custom text
		String path = from->get_option("character_set/custom");

		FileAccess *fa = FileAccess::open(EditorImportPlugin::expand_source_path(path),FileAccess::READ);

		if ( !fa ) {

			FT_Done_FreeType( library );
			ERR_EXPLAIN(TTR("Invalid font custom source."));
			ERR_FAIL_COND_V( !fa,Ref<BitmapFont>() );

		}


		while(!fa->eof_reached()) {

			String line = fa->get_line();
			for(int i=0;i<line.length();i++) {
				import_chars.insert(line[i]);
			}
		}

		if (import_mode==_EditorFontImportOptions::CHARSET_CUSTOM_LATIN) {

			for(int i=32;i<128;i++)
				import_chars.insert(i);
		}

		memdelete(fa);
	}

	int xsize=0;
	while ( gindex != 0 )
	{

		bool skip=false;
		error = FT_Load_Char( face, charcode, font_mode==_EditorFontImportOptions::FONT_BITMAP?FT_LOAD_RENDER:FT_LOAD_MONOCHROME );
		if (error) skip=true;
		else error = FT_Render_Glyph( face->glyph, font_mode==_EditorFontImportOptions::FONT_BITMAP?ft_render_mode_normal:ft_render_mode_mono );
		if (error) {
			skip=true;
		} else if (!skip) {

			switch(import_mode) {

				case _EditorFontImportOptions::CHARSET_ASCII: skip = charcode>127; break;
				case _EditorFontImportOptions::CHARSET_LATIN: skip = charcode>255 ;break;
				case _EditorFontImportOptions::CHARSET_UNICODE: break; //none
				case _EditorFontImportOptions::CHARSET_CUSTOM:
				case _EditorFontImportOptions::CHARSET_CUSTOM_LATIN: skip = !import_chars.has(charcode); break;

			}
		}

		if (charcode<=32) //??
			skip=true;

		if (skip) {
			charcode=FT_Get_Next_Char(face,charcode,&gindex);
			continue;
		}

		_EditorFontData * fdata = memnew( _EditorFontData );


		int w = slot->bitmap.width;
		int h = slot->bitmap.rows;
		int p = slot->bitmap.pitch;

		//print_line("W: "+itos(w)+" P: "+itos(slot->bitmap.pitch));

		if (font_mode==_EditorFontImportOptions::FONT_DISTANCE_FIELD) {

			//oversize the holding buffer so I can smooth it!
			int sw = w + scaler * 4;
			int sh = h + scaler * 4;
			//do the SDF
			int sdfw = sw / scaler;
			int sdfh = sh / scaler;

			fdata->width=sdfw;
			fdata->height=sdfh;
		} else {
			fdata->width=w;
			fdata->height=h;
		}

		fdata->character=charcode;
		fdata->glyph=FT_Get_Char_Index(face,charcode);
		if  (charcode=='x')
			xsize=w/scaler;



		fdata->valign=slot->bitmap_top;
		fdata->halign=slot->bitmap_left;

		if (round_advance)
			fdata->advance=(slot->advance.x+(1<<5))>>6;
		else
			fdata->advance=slot->advance.x/float(1<<6);

		if (font_mode==_EditorFontImportOptions::FONT_DISTANCE_FIELD) {

			fdata->halign = fdata->halign / scaler - 1.5;
			fdata->valign = fdata->valign / scaler + 1.5;
			fdata->advance/=scaler;

		}

		fdata->advance+=font_spacing;


		if (charcode<127) {
			int top = fdata->valign;
			int hmax = h/scaler;

			if (top>max_up) {

				max_up=top;
			}


			if ( (top - hmax)<max_down ) {

				max_down=top - hmax;
			}
		}

		if (font_mode==_EditorFontImportOptions::FONT_DISTANCE_FIELD) {


			//oversize the holding buffer so I can smooth it!
			int sw = w + scaler * 4;
			int sh = h + scaler * 4;

			unsigned char *smooth_buf = new unsigned char[sw*sh];

			for( int i = 0; i < sw*sh; ++i ) {
				smooth_buf[i] = 0;
			}

			// copy the glyph into the buffer to be smoothed
			unsigned char *buf = slot->bitmap.buffer;
			for( int j = 0; j < h; ++j ) {
				for( int i = 0; i < w; ++i ) {
					smooth_buf[scaler*2+i+(j+scaler*2)*sw] = 255 * ((buf[j*p+(i>>3)] >> (7 - (i & 7))) & 1);
				}
			}

			// do the SDF
			int sdfw = fdata->width;
			int sdfh = fdata->height;

			fdata->bitmap.resize( sdfw*sdfh );

			for( int j = 0; j < sdfh; ++j )	{
				for( int i = 0; i < sdfw; ++i )	{
					int pd_idx = j*sdfw+i;

					//fdata->bitmap[j*slot->bitmap.width+i]=slot->bitmap.buffer[j*slot->bitmap.width+i];

					fdata->bitmap[pd_idx] =
							//get_SDF
							get_SDF_radial
							( smooth_buf, sw, sh,
							  i*scaler + (scaler >>1), j*scaler + (scaler >>1),
							  2*scaler );

				}
			}

			delete [] smooth_buf;

		} else {
			fdata->bitmap.resize( slot->bitmap.width*slot->bitmap.rows );
			for (int i=0;i<slot->bitmap.width;i++) {
				for (int j=0;j<slot->bitmap.rows;j++) {

					fdata->bitmap[j*slot->bitmap.width+i]=slot->bitmap.buffer[j*slot->bitmap.width+i];
				}
			}
		}

		font_data_list.push_back(fdata);
		charcode=FT_Get_Next_Char(face,charcode,&gindex);
//                printf("reading char %i\n",charcode);
	}

	/* SPACE */

	_EditorFontData *spd = memnew( _EditorFontData );
	spd->advance=0;
	spd->character=' ';
	spd->halign=0;
	spd->valign=0;
	spd->width=0;
	spd->height=0;
	spd->ofs_x=0;
	spd->ofs_y=0;

	if (!FT_Load_Char( face, ' ', FT_LOAD_RENDER ) && !FT_Render_Glyph( face->glyph, font_mode==_EditorFontImportOptions::FONT_BITMAP?ft_render_mode_normal:ft_render_mode_mono )) {

		spd->advance = slot->advance.x>>6; //round to nearest or store as float
		spd->advance/=scaler;
		spd->advance+=font_spacing;
	} else {

		spd->advance=xsize;
		spd->advance+=font_spacing;
	}

	font_data_list.push_back(spd);

	Set<CharType> exported;
	for (int i=0; i<font_data_list.size(); i++) {
		exported.insert(font_data_list[i]->character);
	};
	int missing = 0;
	for(Set<CharType>::Element *E=import_chars.front();E;E=E->next()) {
		CharType c = E->get();
		if (!exported.has(c)) {
			CharType str[2] = {c, 0};
			printf("** Warning: character %i (%ls) not exported\n", (int)c, str);
			++missing;
		};
	};
	print_line("total_chars: "+itos(font_data_list.size()));

	/* KERNING */


	for(int i=0;i<font_data_list.size();i++) {

		if (font_data_list[i]->character>512)
			continue;
		for(int j=0;j<font_data_list.size();j++) {

			if (font_data_list[j]->character>512)
				continue;

			FT_Vector  delta;
			FT_Get_Kerning( face, font_data_list[i]->glyph,font_data_list[j]->glyph,  FT_KERNING_DEFAULT, &delta );

			if (delta.x!=0) {

				_EditorKerningKey kpk;
				kpk.A = font_data_list[i]->character;
				kpk.B = font_data_list[j]->character;
				int kern = ((-delta.x)+(1<<5))>>6;

				if (kern==0)
					continue;
				kerning_map[kpk]=kern/scaler;
			}
		}
	}

	height=max_up-max_down;
	ascent=max_up;

	/* FIND OUT WHAT THE FONT HEIGHT FOR THIS IS */

	/* ADJUST THE VALIGN FOR EACH CHARACTER */

	for (int i=0;i<(int)font_data_list.size();i++) {

		font_data_list[i]->valign=max_up-font_data_list[i]->valign;
	}



	/* ADD THE SPACEBAR CHARACTER */
/*
	_EditorFontData * fdata = new _EditorFontData;

	fdata->character=32;
	fdata->bitmap=0;
	fdata->width=xsize;
	fdata->height=1;
	fdata->valign=0;

	font_data_list.push_back(fdata);
*/
	/* SORT BY HEIGHT, SO THEY FIT BETTER ON THE TEXTURE */

	font_data_list.sort_custom<_EditorFontDataSort>();
	Color *color=memnew_arr(Color,height);

	int gradient_type=from->get_option("color/mode");
	switch(gradient_type) {
		case _EditorFontImportOptions::COLOR_WHITE: {

			for(int i=0;i<height;i++){
				color[i]=Color(1,1,1,1);
			}

		} break;
		case _EditorFontImportOptions::COLOR_CUSTOM: {

			Color cc = from->get_option("color/color");
			for(int i=0;i<height;i++){
				color[i]=cc;
			}

		} break;
		case _EditorFontImportOptions::COLOR_GRADIENT_RANGE: {

			Color src=from->get_option("color/begin");
			Color to=from->get_option("color/end");
			for(int i=0;i<height;i++){
				color[i]=src.linear_interpolate(to,i/float(height));
			}

		} break;
		case _EditorFontImportOptions::COLOR_GRADIENT_IMAGE: {

			String fp = EditorImportPlugin::expand_source_path(from->get_option("color/image"));
			Image img;
			Error err = ImageLoader::load_image(fp,&img);
			if (err==OK) {

				for(int i=0;i<height;i++){
					//color[i]=img.get_pixel(0,i*img.get_height()/height);
				}
			} else {

				for(int i=0;i<height;i++){
					color[i]=Color(1,1,1,1);
				}
			}

		} break;
	}


	for(int i=0;i<font_data_list.size();i++) {

		if (font_data_list[i]->bitmap.size()==0)
			continue;

		int margin[4]={0,0,0,0};

		if (from->get_option("shadow/enabled").operator bool()) {
			int r=from->get_option("shadow/radius");
			Point2i ofs=Point2(from->get_option("shadow/offset"));
			margin[ MARGIN_LEFT ] = MAX( r - ofs.x, 0);
			margin[ MARGIN_RIGHT ] = MAX( r + ofs.x, 0);
			margin[ MARGIN_TOP ] = MAX( r - ofs.y, 0);
			margin[ MARGIN_BOTTOM ] = MAX( r + ofs.y, 0);

		}

		if (from->get_option("shadow2/enabled").operator bool()) {
			int r=from->get_option("shadow2/radius");
			Point2i ofs=Point2(from->get_option("shadow2/offset"));
			margin[ MARGIN_LEFT ] = MAX( r - ofs.x, margin[ MARGIN_LEFT ]);
			margin[ MARGIN_RIGHT ] = MAX( r + ofs.x, margin[ MARGIN_RIGHT ]);
			margin[ MARGIN_TOP ] = MAX( r - ofs.y, margin[ MARGIN_TOP ]);
			margin[ MARGIN_BOTTOM ] = MAX( r + ofs.y, margin[ MARGIN_BOTTOM ]);

		}

		Size2i s;
		s.width=font_data_list[i]->width+margin[MARGIN_LEFT]+margin[MARGIN_RIGHT];
		s.height=font_data_list[i]->height+margin[MARGIN_TOP]+margin[MARGIN_BOTTOM];
		Point2i o;
		o.x=margin[MARGIN_LEFT];
		o.y=margin[MARGIN_TOP];

		int ow=font_data_list[i]->width;
		int oh=font_data_list[i]->height;

		PoolVector<uint8_t> pixels;
		pixels.resize(s.x*s.y*4);

		PoolVector<uint8_t>::Write w = pixels.write();
		//print_line("val: "+itos(font_data_list[i]->valign));
		for(int y=0;y<s.height;y++) {

			int yc=CLAMP(y-o.y+font_data_list[i]->valign,0,height-1);
			Color c=color[yc];
			c.a=0;

			for(int x=0;x<s.width;x++) {

				int ofs=y*s.x+x;
				w[ofs*4+0]=c.r*255.0;
				w[ofs*4+1]=c.g*255.0;
				w[ofs*4+2]=c.b*255.0;
				w[ofs*4+3]=c.a*255.0;
			}
		}


		for(int si=0;si<2;si++) {

#define S_VAR(m_v) (String(si == 0 ? "shadow/" : "shadow2/") + m_v)
			if (from->get_option(S_VAR("enabled")).operator bool()) {
				int r = from->get_option(S_VAR("radius"));

				Color sc = from->get_option(S_VAR("color"));
				Point2i so=Point2(from->get_option(S_VAR("offset")));

				float tr = from->get_option(S_VAR("transition"));
				print_line("shadow enabled: "+itos(si));

				Vector<uint8_t> s2buf;
				s2buf.resize(s.x*s.y);
				uint8_t *wa=s2buf.ptr();

				for(int j=0;j<s.x*s.y;j++){

					wa[j]=0;
				}

				// blit shadowa
				for(int x=0;x<ow;x++) {
					for(int y=0;y<oh;y++) {
						int ofs = (o.y+y+so.y)*s.x+x+o.x+so.x;
						wa[ofs]=font_data_list[i]->bitmap[y*ow+x];
					}
				}
				//blur shadow2 with separatable convolution

				if (r>0) {

					Vector<uint8_t> pixels2;
					pixels2.resize(s2buf.size());
					uint8_t *w2=pixels2.ptr();
					//vert
					for(int x=0;x<s.width;x++) {
						for(int y=0;y<s.height;y++) {

							int ofs = y*s.width+x;
							int sum=wa[ofs];

							for(int k=1;k<=r;k++) {

								int ofs_d=MIN(y+k,s.height-1)*s.width+x;
								int ofs_u=MAX(y-k,0)*s.width+x;
								sum+=wa[ofs_d];
								sum+=wa[ofs_u];
							}

							w2[ofs]=sum/(r*2+1);

						}
					}
					//horiz
					for(int x=0;x<s.width;x++) {
						for(int y=0;y<s.height;y++) {

							int ofs = y*s.width+x;
							int sum=w2[ofs];

							for(int k=1;k<=r;k++) {

								int ofs_r=MIN(x+k,s.width-1)+s.width*y;
								int ofs_l=MAX(x-k,0)+s.width*y;
								sum+=w2[ofs_r];
								sum+=w2[ofs_l];
							}

							wa[ofs]=Math::pow(float(sum/(r*2+1))/255.0f,tr)*255.0f;

						}
					}

				}

				//blend back

				for(int j=0;j<s.x*s.y;j++){
					Color wd(w[j*4+0]/255.0,w[j*4+1]/255.0,w[j*4+2]/255.0,w[j*4+3]/255.0);
					Color ws(sc.r,sc.g,sc.b,sc.a*(wa[j]/255.0));
					Color b = wd.blend(ws);

					w[j*4+0]=b.r*255.0;
					w[j*4+1]=b.g*255.0;
					w[j*4+2]=b.b*255.0;
					w[j*4+3]=b.a*255.0;

				}
			}
		}

		for(int y=0;y<oh;y++) {
			int yc=CLAMP(y+font_data_list[i]->valign,0,height-1);
			Color sc=color[yc];
			for(int x=0;x<ow;x++) {
				int ofs = (o.y+y)*s.x+x+o.x;
				float c = font_data_list[i]->bitmap[y*ow+x]/255.0;
				Color src_col=sc;
				src_col.a*=c;
				Color dst_col(w[ofs*4+0]/255.0,w[ofs*4+1]/255.0,w[ofs*4+2]/255.0,w[ofs*4+3]/255.0);
				dst_col = dst_col.blend(src_col);
				w[ofs*4+0]=dst_col.r*255.0;
				w[ofs*4+1]=dst_col.g*255.0;
				w[ofs*4+2]=dst_col.b*255.0;
				w[ofs*4+3]=dst_col.a*255.0;
			}
		}


		w=PoolVector<uint8_t>::Write();

		Image img(s.width,s.height,0,Image::FORMAT_RGBA8,pixels);

		font_data_list[i]->blit=img;
		font_data_list[i]->blit_ofs=o;

	}

	//make atlas
	int spacing=2;
	Vector<Size2i> sizes;
	sizes.resize(font_data_list.size());
	for(int i=0;i<font_data_list.size();i++) {

		sizes[i]=Size2(font_data_list[i]->blit.get_width()+spacing*2,font_data_list[i]->blit.get_height()+spacing*2);

	}
	Vector<Point2i> res;
	Size2i res_size;
	EditorAtlas::fit(sizes,res,res_size);
	res_size.x=nearest_power_of_2(res_size.x);
	res_size.y=nearest_power_of_2(res_size.y);
	print_line("Atlas size: "+res_size);

	Image atlas(res_size.x,res_size.y,0,Image::FORMAT_RGBA8);

	for(int i=0;i<font_data_list.size();i++) {

		if (font_data_list[i]->bitmap.size()==0)
			continue;
		atlas.blit_rect(font_data_list[i]->blit,Rect2(0,0,font_data_list[i]->blit.get_width(),font_data_list[i]->blit.get_height()),res[i]+Size2(spacing,spacing));
		font_data_list[i]->ofs_x=res[i].x+spacing;
		font_data_list[i]->ofs_y=res[i].y+spacing;


	}

	if (from->has_option("advanced/premultiply_alpha") && bool(from->get_option("advanced/premultiply_alpha"))) {

		PoolVector<uint8_t> data = atlas.get_data();
		int dl = data.size();
		{
			PoolVector<uint8_t>::Write w = data.write();

			for(int i=0;i<dl;i+=4) {

				w[i+0]= uint8_t(int(w[i+0])*int(w[i+3])/255);
				w[i+1]= uint8_t(int(w[i+1])*int(w[i+3])/255);
				w[i+2]= uint8_t(int(w[i+2])*int(w[i+3])/255);
			}
		}

		atlas=Image(res_size.x,res_size.y,0,Image::FORMAT_RGBA8,data);
	}

	if (from->has_option("color/monochrome") && bool(from->get_option("color/monochrome"))) {

		atlas.convert(Image::FORMAT_LA8);
	}


	if (0) {
		//debug the texture
		Ref<ImageTexture> atlast = memnew( ImageTexture );
		atlast->create_from_image(atlas);
		//atlast->create_from_image(font_data_list[5]->blit);
		TextureRect *tf = memnew( TextureRect );
		tf->set_texture(atlast);
		dialog->add_child(tf);
	}


	/* CREATE FONT */

	int char_space = from->get_option("extra_space/char");
	int space_space = from->get_option("extra_space/space");
	int top_space = from->get_option("extra_space/top");
	int bottom_space = from->get_option("extra_space/bottom");
	bool enable_filter = from->get_option("advanced/enable_filter");
	if (from->has_option("advanced/disable_filter")){ // this is a compatibility check for a deprecated option
		enable_filter = !from->get_option("advanced/disable_filter");
	}

	Ref<BitmapFont> font;

	if (p_existing!=String() && ResourceCache::has(p_existing)) {

		font = Ref<BitmapFont>( ResourceCache::get(p_existing)->cast_to<BitmapFont>());
	}

	if (font.is_null()) {
		 font = Ref<BitmapFont>( memnew( BitmapFont ) );
	}

	font->clear();
	font->set_height(height+bottom_space+top_space);
	font->set_ascent(ascent+top_space);
	font->set_distance_field_hint(font_mode==_EditorFontImportOptions::FONT_DISTANCE_FIELD);

	//register texures
	{
		Ref<ImageTexture> t = memnew(ImageTexture);
		int flags;
		if (!enable_filter)
			flags=0;
		else
			flags=Texture::FLAG_FILTER;
		t->create_from_image(atlas,flags);
		t->set_storage( ImageTexture::STORAGE_COMPRESS_LOSSLESS );
		font->add_texture(t);

	}
	//register characters


	for(int i=0;i<font_data_list.size();i++) {
		_EditorFontData *fd=font_data_list[i];
		int tex_idx=0;

		font->add_char(fd->character,tex_idx,Rect2( fd->ofs_x, fd->ofs_y, fd->blit.get_width(), fd->blit.get_height()),Point2(fd->halign-fd->blit_ofs.x,fd->valign-fd->blit_ofs.y+top_space), fd->advance+char_space+(fd->character==' '?space_space:0));
		memdelete(fd);
	}

	for(Map<_EditorKerningKey,int>::Element *E=kerning_map.front();E;E=E->next()) {

		font->add_kerning_pair(E->key().A,E->key().B,E->get());
	}

	FT_Done_FreeType( library );

	return font;
#else

	return Ref<BitmapFont>();
#endif
}


String EditorFontImportPlugin::get_name() const {

	return "font";
}
String EditorFontImportPlugin::get_visible_name() const{

	return TTR("Font");
}
void EditorFontImportPlugin::import_dialog(const String& p_from){

	dialog->popup_import(p_from);
}
Error EditorFontImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from){


	Ref<BitmapFont> font  = EditorFontImportPlugin::generate_font(p_from,p_path);
	if (!font.is_valid())
		return ERR_CANT_CREATE;

	Ref<ResourceImportMetadata> from=p_from;
	from->set_source_md5(0,FileAccess::get_md5(EditorImportPlugin::expand_source_path(from->get_source_path(0))));
	from->set_editor(get_name());
	font->set_import_metadata(from);

	return ResourceSaver::save(p_path,font);

}

void EditorFontImportPlugin::import_from_drop(const Vector<String>& p_drop, const String &p_dest_path) {

	for(int i=0;i<p_drop.size();i++) {
		String ext = p_drop[i].get_extension().to_lower();
		String file = p_drop[i].get_file();
		if (ext=="ttf" || ext=="otf" || ext=="fnt") {

			import_dialog();
			dialog->set_source_and_dest(p_drop[i],p_dest_path.plus_file(file.get_basename()+".fnt"));
			break;
		}
	}
}


EditorFontImportPlugin::EditorFontImportPlugin(EditorNode* p_editor) {

	dialog = memnew( EditorFontImportDialog(this) );
	p_editor->get_gui_base()->add_child(dialog);
}
#endif
