/*************************************************************************/
/*  editor_fonts.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#include "editor_fonts.h"
#include "doc_font.h"
#include "doc_title_font.h"
#include "doc_code_font.h"
#include "builtin_fonts.h"
#include "editor_settings.h"
#include "scene/resources/dynamic_font.h"

static Ref<BitmapFont> make_font(int p_height,int p_ascent, int p_valign, int p_charcount, const int *p_chars,const Ref<Texture> &p_texture) {


	Ref<BitmapFont> font( memnew( BitmapFont ) );
	font->add_texture( p_texture );

	for (int i=0;i<p_charcount;i++) {

		const int *c = &p_chars[i*8];

		int chr=c[0];
		Rect2 frect;
		frect.pos.x=c[1];
		frect.pos.y=c[2];
		frect.size.x=c[3];
		frect.size.y=c[4];
		Point2 align( c[5], c[6]+p_valign);
		int advance=c[7];


		font->add_char( chr, 0, frect, align,advance );

	}

	font->set_height( p_height );
	font->set_ascent( p_ascent );

	return font;
}


void editor_register_fonts(Ref<Theme> p_theme) {

	Ref<DynamicFontData> dfd;
	dfd.instance();
	dfd->set_font_ptr(_font_droid_sans,_font_droid_sans_size);
	dfd->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> dfmono;
	dfmono.instance();
	dfmono->set_font_ptr(_font_source_code_pro,_font_source_code_pro_size);
	//dfd->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFont> df;
	df.instance();
	df->set_size(int(EditorSettings::get_singleton()->get("global/font_size")));
	df->set_font_data(dfd);


	p_theme->set_default_theme_font(df);

//	Ref<BitmapFont> doc_font = make_font(_bi_font_doc_font_height,_bi_font_doc_font_ascent,0,_bi_font_doc_font_charcount,_bi_font_doc_font_characters,p_theme->get_icon("DocFont","EditorIcons"));
//	Ref<BitmapFont> doc_title_font = make_font(_bi_font_doc_title_font_height,_bi_font_doc_title_font_ascent,0,_bi_font_doc_title_font_charcount,_bi_font_doc_title_font_characters,p_theme->get_icon("DocTitleFont","EditorIcons"));
//	Ref<BitmapFont> doc_code_font = make_font(_bi_font_doc_code_font_height,_bi_font_doc_code_font_ascent,0,_bi_font_doc_code_font_charcount,_bi_font_doc_code_font_characters,p_theme->get_icon("DocCodeFont","EditorIcons"));

	Ref<DynamicFont> df_title;
	df_title.instance();
	df_title->set_size(int(EDITOR_DEF("help/help_title_font_size",18)));
	df_title->set_font_data(dfd);

	Ref<DynamicFont> df_doc;
	df_doc.instance();
	df_doc->set_size(int(EDITOR_DEF("help/help_font_size",16)));
	df_doc->set_font_data(dfd);

	p_theme->set_font("doc","EditorFonts",df_doc);
	p_theme->set_font("doc_title","EditorFonts",df_title);


	Ref<DynamicFont> df_code;
	df_code.instance();
	df_code->set_size(int(EditorSettings::get_singleton()->get("global/source_font_size")));
	df_code->set_font_data(dfmono);

	p_theme->set_font("source","EditorFonts",df_code);

	Ref<DynamicFont> df_doc_code;
	df_doc_code.instance();
	df_doc_code->set_size(int(EDITOR_DEF("help/help_source_font_size",14)));
	df_doc_code->set_font_data(dfmono);

	p_theme->set_font("doc_source","EditorFonts",df_doc_code);

}
