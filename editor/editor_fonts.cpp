/*************************************************************************/
/*  editor_fonts.cpp                                                     */
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
#include "editor_fonts.h"

#include "builtin_fonts.h"
#include "doc_code_font.h"
#include "doc_font.h"
#include "doc_title_font.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/dynamic_font.h"

static Ref<BitmapFont> make_font(int p_height, int p_ascent, int p_valign, int p_charcount, const int *p_chars, const Ref<Texture> &p_texture) {

	Ref<BitmapFont> font(memnew(BitmapFont));
	font->add_texture(p_texture);

	for (int i = 0; i < p_charcount; i++) {

		const int *c = &p_chars[i * 8];

		int chr = c[0];
		Rect2 frect;
		frect.pos.x = c[1];
		frect.pos.y = c[2];
		frect.size.x = c[3];
		frect.size.y = c[4];
		Point2 align(c[5], c[6] + p_valign);
		int advance = c[7];

		font->add_char(chr, 0, frect, align, advance);
	}

	font->set_height(p_height);
	font->set_ascent(p_ascent);

	return font;
}

#define MAKE_FALLBACKS(m_name)               \
	m_name->add_fallback(DroidSansFallback); \
	m_name->add_fallback(DroidSansJapanese); \
	m_name->add_fallback(DroidSansArabic);   \
	m_name->add_fallback(DroidSansHebrew);   \
	m_name->add_fallback(DroidSansThai);

#define MAKE_DROID_SANS(m_name, m_size) \
	Ref<DynamicFont> m_name;            \
	m_name.instance();                  \
	m_name->set_size(m_size);           \
	m_name->set_font_data(DroidSans);   \
	MAKE_FALLBACKS(m_name);

void editor_register_fonts(Ref<Theme> p_theme) {
	/* Droid Sans */

	Ref<DynamicFontData> DroidSans;
	DroidSans.instance();
	DroidSans->set_font_ptr(_font_DroidSans, _font_DroidSans_size);
	DroidSans->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> DroidSansFallback;
	DroidSansFallback.instance();
	DroidSansFallback->set_font_ptr(_font_DroidSansFallback, _font_DroidSansFallback_size);
	DroidSansFallback->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> DroidSansJapanese;
	DroidSansJapanese.instance();
	DroidSansJapanese->set_font_ptr(_font_DroidSansJapanese, _font_DroidSansJapanese_size);
	DroidSansJapanese->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> DroidSansArabic;
	DroidSansArabic.instance();
	DroidSansArabic->set_font_ptr(_font_DroidSansArabic, _font_DroidSansArabic_size);
	DroidSansArabic->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> DroidSansHebrew;
	DroidSansHebrew.instance();
	DroidSansHebrew->set_font_ptr(_font_DroidSansHebrew, _font_DroidSansHebrew_size);
	DroidSansHebrew->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> DroidSansThai;
	DroidSansThai.instance();
	DroidSansThai->set_font_ptr(_font_DroidSansThai, _font_DroidSansThai_size);
	DroidSansThai->set_force_autohinter(true); //just looks better..i think?

	/* Source Code Pro */

	Ref<DynamicFontData> dfmono;
	dfmono.instance();
	dfmono->set_font_ptr(_font_source_code_pro, _font_source_code_pro_size);
	//dfd->set_force_autohinter(true); //just looks better..i think?

	MAKE_DROID_SANS(df, int(EditorSettings::get_singleton()->get("interface/font_size")) * EDSCALE);

	p_theme->set_default_theme_font(df);

	//Ref<BitmapFont> doc_font = make_font(_bi_font_doc_font_height,_bi_font_doc_font_ascent,0,_bi_font_doc_font_charcount,_bi_font_doc_font_characters,p_theme->get_icon("DocFont","EditorIcons"));
	//Ref<BitmapFont> doc_title_font = make_font(_bi_font_doc_title_font_height,_bi_font_doc_title_font_ascent,0,_bi_font_doc_title_font_charcount,_bi_font_doc_title_font_characters,p_theme->get_icon("DocTitleFont","EditorIcons"));
	//Ref<BitmapFont> doc_code_font = make_font(_bi_font_doc_code_font_height,_bi_font_doc_code_font_ascent,0,_bi_font_doc_code_font_charcount,_bi_font_doc_code_font_characters,p_theme->get_icon("DocCodeFont","EditorIcons"));

	MAKE_DROID_SANS(df_title, int(EDITOR_DEF("text_editor/help/help_title_font_size", 18)) * EDSCALE);

	MAKE_DROID_SANS(df_doc, int(EDITOR_DEF("text_editor/help/help_font_size", 16)) * EDSCALE);

	p_theme->set_font("doc", "EditorFonts", df_doc);
	p_theme->set_font("doc_title", "EditorFonts", df_title);

	Ref<DynamicFont> df_code;
	df_code.instance();
	df_code->set_size(int(EditorSettings::get_singleton()->get("interface/source_font_size")) * EDSCALE);
	df_code->set_font_data(dfmono);
	MAKE_FALLBACKS(df_code);

	p_theme->set_font("source", "EditorFonts", df_code);

	Ref<DynamicFont> df_doc_code;
	df_doc_code.instance();
	df_doc_code->set_size(int(EDITOR_DEF("text_editor/help/help_source_font_size", 14)) * EDSCALE);
	df_doc_code->set_font_data(dfmono);
	MAKE_FALLBACKS(df_doc_code);

	p_theme->set_font("doc_source", "EditorFonts", df_doc_code);

	//replace default theme
	Ref<Texture> di;
	Ref<StyleBox> ds;
	fill_default_theme(p_theme, df, df_doc, di, ds, EDSCALE);
}
