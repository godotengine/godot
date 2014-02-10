/*************************************************************************/
/*  font_editor_plugin.cpp                                               */
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
#include "font_editor_plugin.h"
#include "os/file_access.h"
#ifdef FREETYPE_ENABLED

#include <ft2build.h>
#include FT_FREETYPE_H

#endif

#include "core/io/resource_saver.h"

void FontEditor::edit(const Ref<Font>& p_font) {

	font=p_font;
	label->add_font_override("font",font);
}

void FontEditor::_preview_text_changed(const String& p_text) {

	label->set_text(p_text);
}
struct FontData {

	Vector<uint8_t> bitmap;
	int width,height;
	int ofs_x; //ofset to center, from ABOVE
	int ofs_y; //ofset to begining, from LEFT
	int valign; //vertical alignment
	int halign;
	int advance;
	int character;
	int glyph;

	int texture;
//	bool printable;

};


struct FontDataSort {

	bool operator()(const FontData *p_A,const FontData *p_B)  const {
		return p_A->height > p_B->height;
	};
};

struct KerningKey {

	CharType A,B;
	bool operator<(const KerningKey& p_k) const { return (A==p_k.A)?(B<p_k.B):(A<p_k.A); }

};

void FontEditor::_export_fnt(const String& p_name, Ref<Font> p_font) {

	String fnt_name = p_name + ".fnt";
	FileAccess* f = FileAccess::open(fnt_name, FileAccess::WRITE);
	ERR_FAIL_COND(!f);

	f->store_string(String("info face=\"") + p_font->get_name() + "\" size=" + String::num_real(font->get_height()) + " bold=0 italic=0 charset=\"\" unicode=0 stretchH=100 smooth=1 aa=1 padding=0,0,0,0 spacing=4,4\n");

	Vector2 size = p_font->get_texture(0)->get_size();
	f->store_string(String("common lineHeight=") + String::num(font->get_height()) + " base=" + String::num(font->get_ascent()) + " scaleW=" + String::num(size.x) + " scaleH=" + String::num(size.y) + " pages="+String::num(p_font->get_texture_count()) + " packed=0\n");

	for (int i=0; i<p_font->get_texture_count(); i++) {

		f->store_string(String("page id=")+String::num(i)+ " file=\""+ p_name.get_file() + "_" +String::num(i)+".png\"\n");
	};

	f->store_string(String("chars count=")+String::num(p_font->get_character_count()) + "\n");

	Vector<CharType> keys = p_font->get_char_keys();
	keys.sort();
	for (int i=0; i<keys.size(); i++) {

		Font::Character c = p_font->get_character(keys[i]);
		int width = c.rect.size.x;
		if (keys[i] == 32) {
			width = c.advance;
		};
		f->store_string(String("char id=") + String::num(keys[i]) + " x=" + String::num(c.rect.pos.x) + " y=" + String::num(c.rect.pos.y) +
														" width=" + String::num(width) + " height=" + String::num(c.rect.size.y) +
														" xoffset=" + String::num(c.h_align) + " yoffset=" + String::num(c.v_align) +
														" xadvance=" + String::num(c.advance) + " page=" + String::num(c.texture_idx) +
														" chnl=0 letter=\"\"\n");
	};

	f->close();

	for (int i=0; i<p_font->get_texture_count(); i++) {

		ResourceSaver::save(p_name + "_" + String::num(i) + ".png", p_font->get_texture(i));
	};
};

void FontEditor::_import_fnt(const String& p_string) {
	//fnt format used by angelcode bmfont
	//http://www.angelcode.com/products/bmfont/

	FileAccess *f = FileAccess::open(p_string,FileAccess::READ);

	if (!f) {

		ERR_EXPLAIN("Can't open font: "+p_string);
		ERR_FAIL();
	}


	font->clear();

	while(true) {

		String line=f->get_line();

		int delimiter=line.find(" ");
		String type=line.substr(0,delimiter);
		int pos = delimiter+1;
		Map<String,String> keys;

		while (pos < line.size() && line[pos]==' ')
			pos++;


		while(pos<line.size()) {

			int eq = line.find("=",pos);
			if (eq==-1)
				break;
			String key=line.substr(pos,eq-pos);
			int end=-1;
			String value;
			if (line[eq+1]=='"') {
				end=line.find("\"",eq+2);
				if (end==-1)
					break;
				value=line.substr(eq+2,end-1-eq-1);
				pos=end+1;
			} else {
				end=line.find(" ",eq+1);
				if (end==-1)
					end=line.size();

				value=line.substr(eq+1,end-eq);

				pos=end;

			}

			while (pos<line.size() && line[pos]==' ')
				pos++;


			keys[key]=value;

		}


		if (type=="info") {

			if (keys.has("face"))
				font->set_name(keys["face"]);
			//if (keys.has("size"))
			//	font->set_height(keys["size"].to_int());

		} else if (type=="common") {

			if (keys.has("lineHeight"))
				font->set_height(keys["lineHeight"].to_int());
			if (keys.has("base"))
				font->set_ascent(keys["base"].to_int());

		} else if (type=="page") {

			if (keys.has("file")) {

				String file = keys["file"];
				file=p_string.get_base_dir()+"/"+file;
				Ref<Texture> tex = ResourceLoader::load(file);
				if (tex.is_null()) {
					ERR_PRINT("Can't load font texture!");
				} else {
					font->add_texture(tex);
				}
			}
		} else if (type=="char") {

			CharType idx=0;
			if (keys.has("id"))
				idx=keys["id"].to_int();

			Rect2 rect;

			if (keys.has("x"))
				rect.pos.x=keys["x"].to_int();
			if (keys.has("y"))
				rect.pos.y=keys["y"].to_int();
			if (keys.has("width"))
				rect.size.width=keys["width"].to_int();
			if (keys.has("height"))
				rect.size.height=keys["height"].to_int();

			Point2 ofs;

			if (keys.has("xoffset"))
				ofs.x=keys["xoffset"].to_int();
			if (keys.has("yoffset"))
				ofs.y=keys["yoffset"].to_int();

			int texture=0;
			if (keys.has("page"))
				texture=keys["page"].to_int();
			int advance=-1;
			if (keys.has("xadvance"))
				advance=keys["xadvance"].to_int();

			font->add_char(idx,texture,rect,ofs,advance);

		}  else if (type=="kerning") {

			CharType first=0,second=0;
			int k=0;

			if (keys.has("first"))
				first=keys["first"].to_int();
			if (keys.has("second"))
				second=keys["second"].to_int();
			if (keys.has("amount"))
				k=keys["amount"].to_int();

			font->add_kerning_pair(first,second,-k);

		}

		if (f->eof_reached())
			break;
	}



	memdelete(f);



}

void FontEditor::_import_ttf(const String& p_string) {

#ifdef FREETYPE_ENABLED
	FT_Library   library;   /* handle to library     */
	FT_Face      face;      /* handle to face object */

	Vector<FontData*> font_data_list;

	int error = FT_Init_FreeType( &library );

	ERR_EXPLAIN("Error initializing FreeType.");
	ERR_FAIL_COND( error !=0 );


	error = FT_New_Face( library, p_string.utf8().get_data(),0,&face );

	if ( error == FT_Err_Unknown_File_Format ) {
		ERR_EXPLAIN("Unknown font format.");
		FT_Done_FreeType( library );
	} else if ( error ) {

		ERR_EXPLAIN("Error loading font.");
		FT_Done_FreeType( library );

	}

	ERR_FAIL_COND(error);


	int height=0;
	int ascent=0;
	int font_spacing=0;

	int size=font_size->get_text().to_int();

	error = FT_Set_Char_Size(face,0,64*size,512,512);

	if ( error ) {
		FT_Done_FreeType( library );
		ERR_EXPLAIN("Invalid font size. ");
		ERR_FAIL_COND( error );

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

	Map<KerningKey,int> kerning_map;

	charcode = FT_Get_First_Char( face, &gindex );

	int xsize=0;
	while ( gindex != 0 )
	{

		bool skip=false;
		error = FT_Load_Char( face, charcode, FT_LOAD_RENDER );
		if (error) skip=true;
		else error = FT_Render_Glyph( face->glyph, ft_render_mode_normal );
		if (error) skip=true;


		if (!skip && (import_chars.has(charcode) && import_chars[charcode] != 0)) {

			skip = false;

		} else {
			if (import_option->get_selected() == 0 && charcode>127)
				skip=true;
			if (import_option->get_selected() == 1 && charcode>0xFE)
				skip=true;
		};

		if (charcode<=32) //
			skip=true;

		if (skip) {
			charcode=FT_Get_Next_Char(face,charcode,&gindex);
			continue;
		}

		FontData * fdata = memnew( FontData );
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
		fdata->advance=(slot->advance.x+(1<<5))>>6;
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

	FontData *spd = memnew( FontData );
	spd->advance=0;
	spd->character=' ';
	spd->halign=0;
	spd->valign=0;
	spd->width=0;
	spd->height=0;
	spd->ofs_x=0;
	spd->ofs_y=0;

	if (!FT_Load_Char( face, ' ', FT_LOAD_RENDER ) && !FT_Render_Glyph( face->glyph, ft_render_mode_normal )) {

		spd->advance = slot->advance.x>>6;
		spd->advance+=font_spacing;
	} else {

		spd->advance=xsize;
		spd->advance+=font_spacing;
	}

	font_data_list.push_back(spd);

	Map<CharType, bool> exported;
	for (int i=0; i<font_data_list.size(); i++) {
		exported[font_data_list[i]->character] = true;
	};
	int missing = 0;
	for(Map<CharType,int>::Element *E=import_chars.front();E;E=E->next()) {
		CharType c = E->key();
		if (!exported.has(c)) {
			CharType str[2] = {c, 0};
			printf("** Warning: character %i (%ls) not exported\n", (int)c, str);
			++missing;
		};
	};
	printf("total %i/%i\n", missing, import_chars.size());

	/* KERNING */


	for(int i=0;i<font_data_list.size();i++) {

		for(int j=0;j<font_data_list.size();j++) {

			FT_Vector  delta;
			FT_Get_Kerning( face, font_data_list[i]->glyph,font_data_list[j]->glyph,  FT_KERNING_DEFAULT, &delta );

			if (delta.x!=0) {

				KerningKey kpk;
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
	FontData * fdata = new FontData;

	fdata->character=32;
	fdata->bitmap=0;
	fdata->width=xsize;
	fdata->height=1;
	fdata->valign=0;

	font_data_list.push_back(fdata);
*/
	/* SORT BY HEIGHT, SO THEY FIT BETTER ON THE TEXTURE */

	font_data_list.sort_custom<FontDataSort>();

	int spacing=2;


	int use_width=256;
	int use_max_height=256;
//	int surf_idx=-1;

	List<Size2> tex_sizes;
//	int current_texture=0;

	Size2 first(use_width,nearest_power_of_2( font_data_list[0]->height + spacing ));
	Size2 *curtex=&tex_sizes.push_back(first)->get();

	Point2 tex_ofs;

	/* FIT (NOT COPY YEY) FACES IN TEXTURES */

	int current_height=font_data_list[0]->height + spacing;

	int font_margin=2;


	for(int i=0;i<font_data_list.size();i++) {

		FontData *fd=font_data_list[i];

		if (tex_ofs.x+fd->width >= use_width) {
			//end of column, advance a row
			tex_ofs.x=0;
			tex_ofs.y+=current_height+font_margin;
			current_height=fd->height + spacing;

			int new_tex_h = curtex->height;

			while( tex_ofs.y+current_height > new_tex_h ) {

				if (curtex->height * 2 > use_max_height) {
					//oops, can't use this texture anymore..
					Size2 newtex( use_width, nearest_power_of_2( fd->height + spacing ));
					new_tex_h=newtex.height;					
					curtex=&tex_sizes.push_back(newtex)->get();
					tex_ofs=Point2(0,0);

				} else {

					new_tex_h*=2;
				}
			}

			curtex->height=new_tex_h;

		}

		fd->ofs_x=tex_ofs.x;
		fd->ofs_y=tex_ofs.y;
		fd->texture=tex_sizes.size()-1;

		tex_ofs.x+=fd->width+font_margin;


	}

	/* WRITE FACES IN TEXTURES */

	// create textures

	Vector<DVector<uint8_t> >image_data;
	Vector<int> image_widths;
	Vector<DVector<uint8_t>::Write> image_ptrs;
	image_ptrs.resize(tex_sizes.size());

	for(int i=0;i<tex_sizes.size();i++) {

		DVector<uint8_t> pixels;
		int texsize=tex_sizes[i].width * tex_sizes[i].height * 2;
		pixels.resize(texsize );

		image_data.push_back(pixels);
		image_widths.push_back( tex_sizes[i].width );
		image_ptrs[i] = image_data[i].write();
		for(int j=0;j<texsize;j++) {

			image_ptrs[i][j]=0;
		}

	}

	//blit textures with fonts
	for(int i=0;i<font_data_list.size();i++) {

		FontData *fd=font_data_list[i];

		uint8_t *pixels = image_ptrs[fd->texture].ptr();
		int width = image_widths[fd->texture];

		for(int y=0;y<fd->height;y++) {

			const uint8_t *src = &fd->bitmap[y*fd->width];
			uint8_t *dst = &pixels[((fd->ofs_y+y)*width+fd->ofs_x)*2];


			for(int x=0;x<fd->width;x++) {

				dst[x<<1]=255; //white always
				dst[(x<<1) +1]=src[x];

			}
		}
	}

	//unlock writing
	for(int i=0;i<image_ptrs.size();i++)
		image_ptrs[i]=DVector<uint8_t>::Write();

	/* CREATE FONT */

	font->clear();
	font->set_height(height);
	font->set_ascent(ascent);

	//register texures
	for(int i=0;i<tex_sizes.size();i++) {
		Image img(tex_sizes[i].width,tex_sizes[i].height,0,Image::FORMAT_GRAYSCALE_ALPHA,image_data[i]);
		Ref<ImageTexture> tex = memnew( ImageTexture );
		tex->create_from_image(img,0); //no filter, no repeat
		font->add_texture(tex);
		//tframe->set_texture(tex);

	}

	//register characters

	for(int i=0;i<font_data_list.size();i++) {
		FontData *fd=font_data_list[i];

		font->add_char(fd->character,fd->texture,Rect2( fd->ofs_x, fd->ofs_y, fd->width, fd->height),Point2(fd->halign,fd->valign), fd->advance);
		memdelete(fd);
	}

	for(Map<KerningKey,int>::Element *E=kerning_map.front();E;E=E->next()) {

		font->add_kerning_pair(E->key().A,E->key().B,E->get());
	}

	FT_Done_FreeType( library );
#endif
}

void FontEditor::_add_source() {

	_source_file->popup_centered_ratio();
};

void FontEditor::_add_source_accept(const String& p_file) {

	FileAccess* f = FileAccess::open(p_file, FileAccess::READ);
	ERR_FAIL_COND(!f);

	String line;
	while ( !f->eof_reached() ) {

		line = f->get_line();
		for (int i=0; i<line.length(); i++) {

			if (import_chars.has(line[i])) {
				import_chars[line[i]] = import_chars[line[i]] + 1;
			} else {
				import_chars[line[i]] = 1;
			};
		};
	};
};

void FontEditor::_export_fnt_pressed() {

	_export_file->popup_centered_ratio();
};

void FontEditor::_export_fnt_accept(const String& p_file) {

	String name = p_file.replace(".fnt", "");
	_export_fnt(name, font);
};

void FontEditor::_import_accept(const String& p_string) {

#ifdef FREETYPE_ENABLED

	if (p_string.extension().nocasecmp_to("ttf")==0 || p_string.extension().nocasecmp_to("otf")==0) {

		_import_ttf(p_string);
	}
#endif

	if (p_string.extension().nocasecmp_to("fnt")==0) {

		_import_fnt(p_string);
	}

	label->add_font_override("font",font);
	label->notification(Control::NOTIFICATION_THEME_CHANGED);
	label->update();
}

void FontEditor::_import() {


	file->popup_centered_ratio();
}

void FontEditor::_bind_methods() {

	ObjectTypeDB::bind_method("_import",&FontEditor::_import);
	ObjectTypeDB::bind_method("_import_accept",&FontEditor::_import_accept);
	ObjectTypeDB::bind_method("_preview_text_changed",&FontEditor::_preview_text_changed);
	ObjectTypeDB::bind_method("_add_source",&FontEditor::_add_source);
	ObjectTypeDB::bind_method("_add_source_accept",&FontEditor::_add_source_accept);
	ObjectTypeDB::bind_method("_export_fnt_pressed",&FontEditor::_export_fnt_pressed);
	ObjectTypeDB::bind_method("_export_fnt_accept",&FontEditor::_export_fnt_accept);
}

FontEditor::FontEditor() {

	panel = memnew( Panel );
	add_child(panel);
	panel->set_area_as_parent_rect();

	/*
	tframe = memnew( TextureFrame );

	tframe->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	tframe->set_anchor( MARGIN_BOTTOM, ANCHOR_END );

	tframe->set_begin( Point2(5, 40 ) );
	tframe->set_end( Point2(5,55 ) );

	panel->add_child(tframe);
*/

	Label *l = memnew( Label );
	l->set_pos( Point2(5,13 ) );
	l->set_text("Import: ");

	panel->add_child(l);

	l = memnew( Label );
	l->set_pos( Point2(25,37 ) );
	l->set_text("Size: ");

	panel->add_child(l);

	font_size = memnew( LineEdit );
	font_size->set_text("12");
	font_size->set_pos( Point2(70,35 ) );
	font_size->set_size( Size2(40,10 ) );
	panel->add_child(font_size);

	l = memnew( Label );
	l->set_pos( Point2(140,37 ) );
	l->set_text("Encoding: ");

	panel->add_child(l);

	import_option = memnew( OptionButton );
	import_option->add_item("Ascii");
	import_option->add_item("Latin");
	import_option->add_item("Full Unicode");
	import_option->select(1);

	import_option->set_pos( Point2( 215,35 ) );
	import_option->set_size( Point2( 100,12 ) );

	panel->add_child(import_option);

	Button* import = memnew( Button );
	import->set_text("Import:..");
	import->set_begin( Point2(80,35) );
	import->set_end( Point2(10,45) );

	import->set_anchor( MARGIN_LEFT, ANCHOR_END );
	import->set_anchor( MARGIN_RIGHT, ANCHOR_END );

	panel->add_child(import);

	Button* add_source = memnew( Button );
	add_source->set_text("Add Source...");
	add_source->set_begin( Point2(180,35) );
	add_source->set_end( Point2(90,45) );
	add_source->set_anchor( MARGIN_LEFT, ANCHOR_END );
	add_source->set_anchor( MARGIN_RIGHT, ANCHOR_END );

	panel->add_child(add_source);

	file = memnew( FileDialog );
	file->set_access(FileDialog::ACCESS_FILESYSTEM);

	_source_file = memnew( FileDialog );
	_source_file->set_access(FileDialog::ACCESS_FILESYSTEM);
	_source_file->set_mode(FileDialog::MODE_OPEN_FILE);
	_source_file->connect("file_selected", this, "_add_source_accept");
	panel->add_child( _source_file );

	Button* export_fnt = memnew(Button);
	export_fnt->set_text("Export fnt");
	export_fnt->set_begin(Point2(80, 65));
	export_fnt->set_end(Point2(10, 75));
	export_fnt->set_anchor( MARGIN_LEFT, ANCHOR_END );
	export_fnt->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	export_fnt->connect("pressed", this, "_export_fnt_pressed");
	panel->add_child( export_fnt );

	_export_file = memnew(FileDialog);
	_export_file->set_access(FileDialog::ACCESS_FILESYSTEM);
	_export_file->set_mode(FileDialog::MODE_SAVE_FILE);
	_export_file->connect("file_selected", this, "_export_fnt_accept");
	panel->add_child(_export_file);

	l = memnew( Label );
	l->set_pos( Point2(5,65 ) );
	l->set_text("Preview Text: ");

	panel->add_child(l);

	preview_text = memnew( LineEdit );
	preview_text->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	preview_text->set_begin( Point2(25,85 ) );
	preview_text->set_end( Point2(10,95 ) );
	panel->add_child(preview_text);
	preview_text->connect("text_changed", this,"_preview_text_changed");
	preview_text->set_text("The quick brown fox jumped over the lazy dog.");

	l = memnew( Label );
	l->set_pos( Point2(5,115 ) );
	l->set_text("Preview: ");

	panel->add_child(l);

	label = memnew( Label );
	label->set_autowrap(true);

	label->set_anchor( MARGIN_RIGHT, ANCHOR_END );
	label->set_anchor( MARGIN_BOTTOM, ANCHOR_END );

	label->set_begin( Point2(5, 135 ) );
	label->set_end( Point2(5,5 ) );

	label->set_text("The quick brown fox jumped over the lazy dog.");
	label->set_align( Label::ALIGN_CENTER );

	panel->add_child(label);

#ifdef FREETYPE_ENABLED

	file->add_filter("*.ttf");
	file->add_filter("*.otf");
#endif
	file->add_filter("*.fnt ; AngelCode BMFont");

	file->set_mode(FileDialog::MODE_OPEN_FILE);
	panel->add_child( file );

	import->connect("pressed", this,"_import");
	file->connect("file_selected", this,"_import_accept");
	add_source->connect("pressed", this, "_add_source");
}

void FontEditorPlugin::edit(Object *p_node) {

	if (p_node && p_node->cast_to<Font>()) {
		font_editor->edit( p_node->cast_to<Font>() );
		font_editor->show();
	} else
		font_editor->hide();
}

bool FontEditorPlugin::handles(Object *p_node) const{

	return p_node->is_type("Font");
}

void FontEditorPlugin::make_visible(bool p_visible){

	if (p_visible)
		font_editor->show();
	else
		font_editor->hide();
}

FontEditorPlugin::FontEditorPlugin(EditorNode *p_node) {

	font_editor = memnew( FontEditor );

	p_node->get_viewport()->add_child(font_editor);
	font_editor->set_area_as_parent_rect();
	font_editor->hide();




}

