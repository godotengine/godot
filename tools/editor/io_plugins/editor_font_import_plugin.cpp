/*************************************************************************/
/*  editor_font_import_plugin.cpp                                        */
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
#include "editor_font_import_plugin.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "tools/editor/editor_node.h"
#include "os/file_access.h"
#include "editor_atlas.h"
#include "io/image_loader.h"
#include "io/resource_saver.h"
#ifdef FREETYPE_ENABLED

#include <ft2build.h>
#include FT_FREETYPE_H

#endif


class _EditorFontImportOptions : public Object {

	OBJ_TYPE(_EditorFontImportOptions,Object);
public:

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


	bool disable_filter;
	bool round_advance;



	bool _set(const StringName& p_name, const Variant& p_value) {

		String n = p_name;
		if (n=="extra_space/char")
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
		else if (n=="advanced/disable_filter")
			disable_filter=p_value;
		else
			return false;

		emit_signal("changed");


		return true;

	}

	bool _get(const StringName& p_name,Variant &r_ret) const{

		String n = p_name;
		if (n=="extra_space/char")
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
		else if (n=="advanced/disable_filter")
			r_ret=disable_filter;
		else
			return false;

		return true;

	}

	void _get_property_list( List<PropertyInfo> *p_list) const{

		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/char",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/space",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/top",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"extra_space/bottom",PROPERTY_HINT_RANGE,"-64,64,1"));
		p_list->push_back(PropertyInfo(Variant::INT,"character_set/mode",PROPERTY_HINT_ENUM,"Ascii,Latin,Unicode,Custom,Custom&Latin"));

		if (character_set>=CHARSET_CUSTOM)
			p_list->push_back(PropertyInfo(Variant::STRING,"character_set/custom",PROPERTY_HINT_FILE));

		p_list->push_back(PropertyInfo(Variant::BOOL,"shadow/enabled"));
		if (shadow) {
			p_list->push_back(PropertyInfo(Variant::INT,"shadow/radius",PROPERTY_HINT_RANGE,"-64,64,1"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2,"shadow/offset"));
			p_list->push_back(PropertyInfo(Variant::COLOR,"shadow/color"));
			p_list->push_back(PropertyInfo(Variant::REAL,"shadow/transition",PROPERTY_HINT_EXP_EASING));
		}

		p_list->push_back(PropertyInfo(Variant::BOOL,"shadow2/enabled"));
		if (shadow2) {
			p_list->push_back(PropertyInfo(Variant::INT,"shadow2/radius",PROPERTY_HINT_RANGE,"-64,64,1"));
			p_list->push_back(PropertyInfo(Variant::VECTOR2,"shadow2/offset"));
			p_list->push_back(PropertyInfo(Variant::COLOR,"shadow2/color"));
			p_list->push_back(PropertyInfo(Variant::REAL,"shadow2/transition",PROPERTY_HINT_EXP_EASING));
		}

		p_list->push_back(PropertyInfo(Variant::INT,"color/mode",PROPERTY_HINT_ENUM,"White,Color,Gradient,Gradient Image"));
		if (color_type==COLOR_CUSTOM) {
			p_list->push_back(PropertyInfo(Variant::COLOR,"color/color"));

		}
		if (color_type==COLOR_GRADIENT_RANGE) {
			p_list->push_back(PropertyInfo(Variant::COLOR,"color/begin"));
			p_list->push_back(PropertyInfo(Variant::COLOR,"color/end"));
		}
		if (color_type==COLOR_GRADIENT_IMAGE) {
			p_list->push_back(PropertyInfo(Variant::STRING,"color/image",PROPERTY_HINT_FILE));
		}
		p_list->push_back(PropertyInfo(Variant::BOOL,"color/monochrome"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"advanced/round_advance"));
		p_list->push_back(PropertyInfo(Variant::BOOL,"advanced/disable_filter"));

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

		round_advance=true;
		disable_filter=false;

	}

	_EditorFontImportOptions() {

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
		disable_filter=false;
	}


};


class EditorFontImportDialog : public ConfirmationDialog {

	OBJ_TYPE(EditorFontImportDialog, ConfirmationDialog);


	LineEditFileChooser *source;
	LineEditFileChooser *dest;
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

		imd->add_source(EditorImportPlugin::validate_source_path(source->get_line_edit()->get_text()));
		imd->set_option("font/size",font_size->get_val());

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
		test_label->add_color_override("font_color",test_color->get_color());
	}

	void _update() {

		Ref<ResourceImportMetadata> imd = get_rimd();
		Ref<Font> font = plugin->generate_font(imd);
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

		Ref<Font> font = ResourceLoader::load(p_font);
		if (!font.is_valid())
			return;
		Ref<ImageTexture> tex = font->get_texture(0);
		if (tex.is_null())
			return;
		FileAccessRef f=FileAccess::open(p_font.basename()+".inc",FileAccess::WRITE);
		Vector<CharType> ck = font->get_char_keys();

		f->store_line("static const int _builtin_font_height="+itos(font->get_height())+";");
		f->store_line("static const int _builtin_font_ascent="+itos(font->get_ascent())+";");
		f->store_line("static const int _builtin_font_charcount="+itos(ck.size())+";");
		f->store_line("static const int _builtin_font_charrects["+itos(ck.size())+"][8]={");
		f->store_line("/* charidx , ofs_x, ofs_y, size_x, size_y, valign, halign, advance */");

		for(int i=0;i<ck.size();i++) {
			CharType k=ck[i];
			Font::Character c=font->get_character(k);
			f->store_line("{"+itos(k)+","+rtos(c.rect.pos.x)+","+rtos(c.rect.pos.y)+","+rtos(c.rect.size.x)+","+rtos(c.rect.size.y)+","+rtos(c.v_align)+","+rtos(c.h_align)+","+rtos(c.advance)+"},");
		}
		f->store_line("};");

		Vector<Font::KerningPairKey> kp=font->get_kerning_pair_keys();
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
		f->store_line("static const unsigned char _builtin_font_img_data["+itos(img.get_width()*img.get_height()*2)+"]={");
		for(int i=0;i<img.get_height();i++) {

			for(int j=0;j<img.get_width();j++) {

				Color c = img.get_pixel(j,i);
				int v = CLAMP(((c.r+c.g+c.b)/3.0)*255,0,255);
				int a = CLAMP(c.a*255,0,255);

				f->store_line(itos(v)+","+itos(a)+",");
			}
		}
		f->store_line("};");

	}

	void _import() {

		if (source->get_line_edit()->get_text()=="") {
			error_dialog->set_text("No source font file!");
			error_dialog->popup_centered(Size2(200,100));
			return;
		}

		if (dest->get_line_edit()->get_text()=="") {
			error_dialog->set_text("No target font resource!");
			error_dialog->popup_centered(Size2(200,100));
			return;
		}

		Ref<ResourceImportMetadata> rimd = get_rimd();

		if (rimd.is_null()) {
			error_dialog->set_text("Can't load/process source font");
			error_dialog->popup_centered(Size2(200,100));
			return;
		}

		Error err = plugin->import(dest->get_line_edit()->get_text(),rimd);

		if (err!=OK) {
			error_dialog->set_text("Couldn't save font.");
			error_dialog->popup_centered(Size2(200,100));
			return;
		}

		//_import_inc(dest->get_line_edit()->get_text());

		hide();
	}

	EditorFontImportPlugin *plugin;
	_EditorFontImportOptions *options;

	static void _bind_methods() {

		ObjectTypeDB::bind_method("_update",&EditorFontImportDialog::_update);
		ObjectTypeDB::bind_method("_update_text",&EditorFontImportDialog::_update_text);
		ObjectTypeDB::bind_method("_update_text2",&EditorFontImportDialog::_update_text2);
		ObjectTypeDB::bind_method("_update_text3",&EditorFontImportDialog::_update_text3);
		ObjectTypeDB::bind_method("_prop_changed",&EditorFontImportDialog::_prop_changed);
		ObjectTypeDB::bind_method("_src_changed",&EditorFontImportDialog::_src_changed);
		ObjectTypeDB::bind_method("_font_size_changed",&EditorFontImportDialog::_font_size_changed);
		ObjectTypeDB::bind_method("_import",&EditorFontImportDialog::_import);

	}

public:

	void _notification(int p_what) {

		if (p_what==NOTIFICATION_ENTER_TREE) {
			prop_edit->edit(options);
			_update_text();
		}
	}

	void popup_import(const String& p_path) {

		popup_centered(Size2(600,500));

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

			font_size->set_val(rimd->get_option("font/size"));
		}
	}

	EditorFontImportDialog(EditorFontImportPlugin *p_plugin) {
		plugin=p_plugin;
		VBoxContainer *vbc = memnew( VBoxContainer );
		add_child(vbc);
		set_child_rect(vbc);
		HBoxContainer *hbc = memnew( HBoxContainer);
		vbc->add_child(hbc);
		VBoxContainer *vbl = memnew( VBoxContainer );
		hbc->add_child(vbl);
		hbc->set_v_size_flags(SIZE_EXPAND_FILL);
		vbl->set_h_size_flags(SIZE_EXPAND_FILL);
		VBoxContainer *vbr = memnew( VBoxContainer );
		hbc->add_child(vbr);
		vbr->set_h_size_flags(SIZE_EXPAND_FILL);

		source = memnew( LineEditFileChooser );
		source->get_file_dialog()->set_access(FileDialog::ACCESS_FILESYSTEM);
		source->get_file_dialog()->set_mode(FileDialog::MODE_OPEN_FILE);
		source->get_file_dialog()->add_filter("*.ttf;TrueType");
		source->get_file_dialog()->add_filter("*.otf;OpenType");
		source->get_line_edit()->connect("text_entered",this,"_src_changed");

		vbl->add_margin_child("Source Font:",source);
		font_size = memnew( SpinBox );
		vbl->add_margin_child("Source Font Size:",font_size);
		font_size->set_min(3);
		font_size->set_max(256);
		font_size->set_val(16);
		font_size->connect("value_changed",this,"_font_size_changed");
		dest = memnew( LineEditFileChooser );
		//
		List<String> fl;
		Ref<Font> font= memnew(Font);
		dest->get_file_dialog()->add_filter("*.fnt ; Font" );
		//ResourceSaver::get_recognized_extensions(font,&fl);
		//for(List<String>::Element *E=fl.front();E;E=E->next()) {
		//	dest->get_file_dialog()->add_filter("*."+E->get());
		//}

		vbl->add_margin_child("Dest Resource:",dest);
		HBoxContainer *testhb = memnew( HBoxContainer );
		test_string = memnew( LineEdit );
		test_string->set_text("The quick brown fox jumps over the lazy dog.");
		test_string->set_h_size_flags(SIZE_EXPAND_FILL);
		test_string->set_stretch_ratio(5);

		testhb->add_child(test_string);
		test_color = memnew( ColorPickerButton );
		test_color->set_color(get_color("font_color","Label"));
		test_color->set_h_size_flags(SIZE_EXPAND_FILL);
		test_color->set_stretch_ratio(1);
		test_color->connect("color_changed",this,"_update_text3");
		testhb->add_child(test_color);

		vbl->add_spacer();
		vbl->add_margin_child("Test: ",testhb);
		HBoxContainer *upd_hb = memnew( HBoxContainer );
//		vbl->add_child(upd_hb);
		upd_hb->add_spacer();
		Button *update = memnew( Button);
		upd_hb->add_child(update);
		update->set_text("Update");
		update->connect("pressed",this,"_update");

		options = memnew( _EditorFontImportOptions );
		prop_edit = memnew( PropertyEditor() );
		vbr->add_margin_child("Options:",prop_edit,true);
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
		set_title("Font Import");
		timer = memnew( Timer );
		add_child(timer);
		timer->connect("timeout",this,"_update");
		timer->set_wait_time(0.4);
		timer->set_one_shot(true);

		get_ok()->connect("pressed", this,"_import");
		get_ok()->set_text("Import");

		error_dialog = memnew ( ConfirmationDialog );
		add_child(error_dialog);
		error_dialog->get_ok()->set_text("Accept");
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
	int ofs_x; //ofset to center, from ABOVE
	int ofs_y; //ofset to begining, from LEFT
	int valign; //vertical alignment
	int halign;
	float advance;
	int character;
	int glyph;

	int texture;
	Image blit;
	Point2i blit_ofs;
//	bool printable;

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

Ref<Font> EditorFontImportPlugin::generate_font(const Ref<ResourceImportMetadata>& p_from, const String &p_existing) {

	Ref<ResourceImportMetadata> from = p_from;
	ERR_FAIL_COND_V(from->get_source_count()!=1,Ref<Font>());

	String src_path = EditorImportPlugin::expand_source_path(from->get_source_path(0));
	int size = from->get_option("font/size");

#ifdef FREETYPE_ENABLED
	FT_Library   library;   /* handle to library     */
	FT_Face      face;      /* handle to face object */

	Vector<_EditorFontData*> font_data_list;

	int error = FT_Init_FreeType( &library );

	ERR_EXPLAIN("Error initializing FreeType.");
	ERR_FAIL_COND_V( error !=0, Ref<Font>() );

	print_line("loadfrom: "+src_path);
	error = FT_New_Face( library, src_path.utf8().get_data(),0,&face );

	if ( error == FT_Err_Unknown_File_Format ) {
		ERR_EXPLAIN("Unknown font format.");
		FT_Done_FreeType( library );
	} else if ( error ) {

		ERR_EXPLAIN("Error loading font.");
		FT_Done_FreeType( library );

	}

	ERR_FAIL_COND_V(error,Ref<Font>());


	int height=0;
	int ascent=0;
	int font_spacing=0;

	error = FT_Set_Char_Size(face,0,64*size,512,512);

	if ( error ) {
		FT_Done_FreeType( library );
		ERR_EXPLAIN("Invalid font size. ");
		ERR_FAIL_COND_V( error,Ref<Font>() );

	}

	error = FT_Set_Pixel_Sizes(face,0,size);

	FT_GlyphSlot slot = face->glyph;

//	error = FT_Set_Charmap(face,ft_encoding_unicode );   /* encoding..         */


	/* PRINT CHARACTERS TO INDIVIDUAL BITMAPS */


//	int space_size=5; //size for space, if none found.. 5!
//	int min_valign=500; //some ridiculous number

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
			ERR_EXPLAIN("Invalid font custom source. ");
			ERR_FAIL_COND_V( !fa,Ref<Font>() );

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
		error = FT_Load_Char( face, charcode, FT_LOAD_RENDER );
		if (error) skip=true;
		else error = FT_Render_Glyph( face->glyph, ft_render_mode_normal );
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
		fdata->bitmap.resize( slot->bitmap.width*slot->bitmap.rows );
		fdata->width=slot->bitmap.width;
		fdata->height=slot->bitmap.rows;
		fdata->character=charcode;
		fdata->glyph=FT_Get_Char_Index(face,charcode);
		if  (charcode=='x')
			xsize=slot->bitmap.width;


		if (charcode<127) {
			if (slot->bitmap_top>max_up) {

				max_up=slot->bitmap_top;
			}


			if ( (slot->bitmap_top - fdata->height)<max_down ) {

				max_down=slot->bitmap_top - fdata->height;
			}
		}


		fdata->valign=slot->bitmap_top;
		fdata->halign=slot->bitmap_left;

		if (round_advance)
			fdata->advance=(slot->advance.x+(1<<5))>>6;
		else
			fdata->advance=slot->advance.x/float(1<<6);

		fdata->advance+=font_spacing;

		for (int i=0;i<slot->bitmap.width;i++) {
			for (int j=0;j<slot->bitmap.rows;j++) {

				fdata->bitmap[j*slot->bitmap.width+i]=slot->bitmap.buffer[j*slot->bitmap.width+i];
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

	if (!FT_Load_Char( face, ' ', FT_LOAD_RENDER ) && !FT_Render_Glyph( face->glyph, ft_render_mode_normal )) {

		spd->advance = slot->advance.x>>6; //round to nearest or store as float
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
				kerning_map[kpk]=kern;
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
					color[i]=img.get_pixel(0,i*img.get_height()/height);
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

		DVector<uint8_t> pixels;
		pixels.resize(s.x*s.y*4);

		DVector<uint8_t>::Write w = pixels.write();
		print_line("val: "+itos(font_data_list[i]->valign));
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

#define S_VAR(m_v) (String(si==0?"shadow/":"shadow2/")+m_v)
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

							wa[ofs]=Math::pow(float(sum/(r*2+1))/255.0,tr)*255.0;

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


		w=DVector<uint8_t>::Write();

		Image img(s.width,s.height,0,Image::FORMAT_RGBA,pixels);

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

	Image atlas(res_size.x,res_size.y,0,Image::FORMAT_RGBA);

	for(int i=0;i<font_data_list.size();i++) {

		if (font_data_list[i]->bitmap.size()==0)
			continue;
		atlas.blit_rect(font_data_list[i]->blit,Rect2(0,0,font_data_list[i]->blit.get_width(),font_data_list[i]->blit.get_height()),res[i]+Size2(spacing,spacing));
		font_data_list[i]->ofs_x=res[i].x+spacing;
		font_data_list[i]->ofs_y=res[i].y+spacing;


	}


	if (from->has_option("color/monochrome") && bool(from->get_option("color/monochrome"))) {

		atlas.convert(Image::FORMAT_GRAYSCALE_ALPHA);
	}

	if (0) {
		//debug the texture
		Ref<ImageTexture> atlast = memnew( ImageTexture );
		atlast->create_from_image(atlas);
//		atlast->create_from_image(font_data_list[5]->blit);
		TextureFrame *tf = memnew( TextureFrame );
		tf->set_texture(atlast);
		dialog->add_child(tf);
	}


	/* CREATE FONT */

	int char_space = from->get_option("extra_space/char");
	int space_space = from->get_option("extra_space/space");
	int top_space = from->get_option("extra_space/top");
	int bottom_space = from->get_option("extra_space/bottom");
	bool disable_filter = from->get_option("advanced/disable_filter");

	Ref<Font> font;

	if (p_existing!=String() && ResourceCache::has(p_existing)) {

		font = Ref<Font>( ResourceCache::get(p_existing)->cast_to<Font>());
	}

	if (font.is_null()) {
		 font = Ref<Font>( memnew( Font ) );
	}

	font->clear();
	font->set_height(height+bottom_space+top_space);
	font->set_ascent(ascent+top_space);

	//register texures
	{
		Ref<ImageTexture> t = memnew(ImageTexture);
		int flags;
		if (disable_filter)
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

	return Ref<Font>();
#endif
}


String EditorFontImportPlugin::get_name() const {

	return "font";
}
String EditorFontImportPlugin::get_visible_name() const{

	return "Font";
}
void EditorFontImportPlugin::import_dialog(const String& p_from){

	dialog->popup_import(p_from);
}
Error EditorFontImportPlugin::import(const String& p_path, const Ref<ResourceImportMetadata>& p_from){


	Ref<Font> font  = EditorFontImportPlugin::generate_font(p_from,p_path);
	if (!font.is_valid())
		return ERR_CANT_CREATE;

	Ref<ResourceImportMetadata> from=p_from;
	from->set_source_md5(0,FileAccess::get_md5(EditorImportPlugin::expand_source_path(from->get_source_path(0))));
	from->set_editor(get_name());
	font->set_import_metadata(from);

	return ResourceSaver::save(p_path,font);

}


EditorFontImportPlugin::EditorFontImportPlugin(EditorNode* p_editor) {

	dialog = memnew( EditorFontImportDialog(this) );
	p_editor->get_gui_base()->add_child(dialog);
}
