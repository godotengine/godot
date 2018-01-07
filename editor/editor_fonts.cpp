/*************************************************************************/
/*  editor_fonts.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "builtin_fonts.gen.h"
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
		frect.position.x = c[1];
		frect.position.y = c[2];
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

#define MAKE_FALLBACKS(m_name)          \
	m_name->add_fallback(FontArabic);   \
	m_name->add_fallback(FontHebrew);   \
	m_name->add_fallback(FontThai);     \
	m_name->add_fallback(FontJapanese); \
	m_name->add_fallback(FontFallback);

// the custom spacings might only work with Noto Sans
#define MAKE_DEFAULT_FONT(m_name, m_size)                       \
	Ref<DynamicFont> m_name;                                    \
	m_name.instance();                                          \
	m_name->set_size(m_size);                                   \
	if (CustomFont.is_valid()) {                                \
		m_name->set_font_data(CustomFont);                      \
		m_name->add_fallback(DefaultFont);                      \
	} else {                                                    \
		m_name->set_font_data(DefaultFont);                     \
	}                                                           \
	m_name->set_spacing(DynamicFont::SPACING_TOP, -EDSCALE);    \
	m_name->set_spacing(DynamicFont::SPACING_BOTTOM, -EDSCALE); \
	MAKE_FALLBACKS(m_name);

#define MAKE_SOURCE_FONT(m_name, m_size)                        \
	Ref<DynamicFont> m_name;                                    \
	m_name.instance();                                          \
	m_name->set_size(m_size);                                   \
	if (CustomFontSource.is_valid()) {                          \
		m_name->set_font_data(CustomFontSource);                \
		m_name->add_fallback(dfmono);                           \
	} else {                                                    \
		m_name->set_font_data(dfmono);                          \
	}                                                           \
	m_name->set_spacing(DynamicFont::SPACING_TOP, -EDSCALE);    \
	m_name->set_spacing(DynamicFont::SPACING_BOTTOM, -EDSCALE); \
	MAKE_FALLBACKS(m_name);

void editor_register_fonts(Ref<Theme> p_theme) {
	/* Custom font */

	String custom_font = EditorSettings::get_singleton()->get("interface/editor/main_font");
	Ref<DynamicFontData> CustomFont;
	if (custom_font.length() > 0) {
		CustomFont.instance();
		CustomFont->set_font_path(custom_font);
		CustomFont->set_force_autohinter(true); //just looks better..i think?
	}

	/* Custom source code font */

	String custom_font_source = EditorSettings::get_singleton()->get("interface/editor/code_font");
	Ref<DynamicFontData> CustomFontSource;
	if (custom_font_source.length() > 0) {
		CustomFontSource.instance();
		CustomFontSource->set_font_path(custom_font_source);
	}

	/* Droid Sans */

	Ref<DynamicFontData> DefaultFont;
	DefaultFont.instance();
	DefaultFont->set_font_ptr(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size);
	DefaultFont->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontFallback;
	FontFallback.instance();
	FontFallback->set_font_ptr(_font_DroidSansFallback, _font_DroidSansFallback_size);
	FontFallback->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontJapanese;
	FontJapanese.instance();
	FontJapanese->set_font_ptr(_font_DroidSansJapanese, _font_DroidSansJapanese_size);
	FontJapanese->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontArabic;
	FontArabic.instance();
	FontArabic->set_font_ptr(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size);
	FontArabic->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontHebrew;
	FontHebrew.instance();
	FontHebrew->set_font_ptr(_font_NotoSansHebrew_Regular, _font_NotoSansHebrew_Regular_size);
	FontHebrew->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontThai;
	FontThai.instance();
	FontThai->set_font_ptr(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size);
	FontThai->set_force_autohinter(true); //just looks better..i think?

	/* Source Code Pro */

	Ref<DynamicFontData> dfmono;
	dfmono.instance();
	dfmono->set_font_ptr(_font_Hack_Regular, _font_Hack_Regular_size);
	//dfd->set_force_autohinter(true); //just looks better..i think?

	int default_font_size = int(EditorSettings::get_singleton()->get("interface/editor/main_font_size")) * EDSCALE;
	MAKE_DEFAULT_FONT(df, default_font_size);

	p_theme->set_default_theme_font(df);

	MAKE_DEFAULT_FONT(df_title, default_font_size + 2 * EDSCALE);
	p_theme->set_font("title", "EditorFonts", df_title);

	MAKE_DEFAULT_FONT(df_doc_title, int(EDITOR_DEF("text_editor/help/help_title_font_size", 23)) * EDSCALE);

	MAKE_DEFAULT_FONT(df_doc, int(EDITOR_DEF("text_editor/help/help_font_size", 15)) * EDSCALE);

	p_theme->set_font("doc", "EditorFonts", df_doc);
	p_theme->set_font("doc_title", "EditorFonts", df_doc_title);

	MAKE_DEFAULT_FONT(df_rulers, 8 * EDSCALE);
	p_theme->set_font("rulers", "EditorFonts", df_rulers);

	MAKE_SOURCE_FONT(df_code, int(EditorSettings::get_singleton()->get("interface/editor/code_font_size")) * EDSCALE);
	p_theme->set_font("source", "EditorFonts", df_code);

	MAKE_SOURCE_FONT(df_doc_code, int(EDITOR_DEF("text_editor/help/help_source_font_size", 14)) * EDSCALE);
	p_theme->set_font("doc_source", "EditorFonts", df_doc_code);

	MAKE_SOURCE_FONT(df_output_code, int(EDITOR_DEF("run/output/font_size", 13)) * EDSCALE);
	p_theme->set_font("output_source", "EditorFonts", df_output_code);

	MAKE_SOURCE_FONT(df_text_editor_status_code, 14 * EDSCALE);
	p_theme->set_font("status_source", "EditorFonts", df_text_editor_status_code);
}
