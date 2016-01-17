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

static Ref<Font> make_font(int p_height,int p_ascent, int p_valign, int p_charcount, const int *p_chars,const Ref<Texture> &p_texture) {


	Ref<Font> font( memnew( Font ) );
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


	Ref<Font> doc_font = make_font(_bi_font_doc_font_height,_bi_font_doc_font_ascent,0,_bi_font_doc_font_charcount,_bi_font_doc_font_characters,p_theme->get_icon("DocFont","EditorIcons"));
	Ref<Font> doc_code_font = make_font(_bi_font_doc_code_font_height,_bi_font_doc_code_font_ascent,0,_bi_font_doc_code_font_charcount,_bi_font_doc_code_font_characters,p_theme->get_icon("DocCodeFont","EditorIcons"));
	Ref<Font> doc_title_font = make_font(_bi_font_doc_title_font_height,_bi_font_doc_title_font_ascent,0,_bi_font_doc_title_font_charcount,_bi_font_doc_title_font_characters,p_theme->get_icon("DocTitleFont","EditorIcons"));
	p_theme->set_font("doc","EditorFonts",doc_font);
	p_theme->set_font("doc_code","EditorFonts",doc_code_font);
	p_theme->set_font("doc_title","EditorFonts",doc_title_font);

}
