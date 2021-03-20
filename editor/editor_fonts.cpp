/*************************************************************************/
/*  editor_fonts.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/os/dir_access.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/font.h"

#define MAKE_FALLBACKS(m_name)       \
	m_name->add_data(FontArabic);    \
	m_name->add_data(FontBengali);   \
	m_name->add_data(FontGeorgian);  \
	m_name->add_data(FontMalayalam); \
	m_name->add_data(FontOriya);     \
	m_name->add_data(FontSinhala);   \
	m_name->add_data(FontTamil);     \
	m_name->add_data(FontTelugu);    \
	m_name->add_data(FontHebrew);    \
	m_name->add_data(FontThai);      \
	m_name->add_data(FontHindi);     \
	m_name->add_data(FontJapanese);  \
	m_name->add_data(FontFallback);

// the custom spacings might only work with Noto Sans
#define MAKE_DEFAULT_FONT(m_name)                        \
	Ref<Font> m_name;                                    \
	m_name.instance();                                   \
	if (CustomFont.is_valid()) {                         \
		m_name->add_data(CustomFont);                    \
		m_name->add_data(DefaultFont);                   \
	} else {                                             \
		m_name->add_data(DefaultFont);                   \
	}                                                    \
	m_name->set_spacing(Font::SPACING_TOP, -EDSCALE);    \
	m_name->set_spacing(Font::SPACING_BOTTOM, -EDSCALE); \
	MAKE_FALLBACKS(m_name);

#define MAKE_BOLD_FONT(m_name)                           \
	Ref<Font> m_name;                                    \
	m_name.instance();                                   \
	if (CustomFontBold.is_valid()) {                     \
		m_name->add_data(CustomFontBold);                \
		m_name->add_data(DefaultFontBold);               \
	} else {                                             \
		m_name->add_data(DefaultFontBold);               \
	}                                                    \
	m_name->set_spacing(Font::SPACING_TOP, -EDSCALE);    \
	m_name->set_spacing(Font::SPACING_BOTTOM, -EDSCALE); \
	MAKE_FALLBACKS(m_name);

#define MAKE_SOURCE_FONT(m_name)                         \
	Ref<Font> m_name;                                    \
	m_name.instance();                                   \
	if (CustomFontSource.is_valid()) {                   \
		m_name->add_data(CustomFontSource);              \
		m_name->add_data(dfmono);                        \
	} else {                                             \
		m_name->add_data(dfmono);                        \
	}                                                    \
	m_name->set_spacing(Font::SPACING_TOP, -EDSCALE);    \
	m_name->set_spacing(Font::SPACING_BOTTOM, -EDSCALE); \
	MAKE_FALLBACKS(m_name);

void editor_register_fonts(Ref<Theme> p_theme) {
	DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	/* Custom font */

	bool font_antialiased = (bool)EditorSettings::get_singleton()->get("interface/editor/font_antialiased");
	int font_hinting_setting = (int)EditorSettings::get_singleton()->get("interface/editor/font_hinting");

	TextServer::Hinting font_hinting;
	switch (font_hinting_setting) {
		case 0:
			// The "Auto" setting uses the setting that best matches the OS' font rendering:
			// - macOS doesn't use font hinting.
			// - Windows uses ClearType, which is in between "Light" and "Normal" hinting.
			// - Linux has configurable font hinting, but most distributions including Ubuntu default to "Light".
#ifdef OSX_ENABLED
			font_hinting = TextServer::HINTING_NONE;
#else
			font_hinting = TextServer::HINTING_LIGHT;
#endif
			break;
		case 1:
			font_hinting = TextServer::HINTING_NONE;
			break;
		case 2:
			font_hinting = TextServer::HINTING_LIGHT;
			break;
		default:
			font_hinting = TextServer::HINTING_NORMAL;
			break;
	}

	int default_font_size = int(EDITOR_GET("interface/editor/main_font_size")) * EDSCALE;

	String custom_font_path = EditorSettings::get_singleton()->get("interface/editor/main_font");
	Ref<FontData> CustomFont;
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		CustomFont.instance();
		CustomFont->load_resource(custom_font_path, default_font_size);
		CustomFont->set_antialiased(font_antialiased);
		CustomFont->set_hinting(font_hinting);
		CustomFont->set_force_autohinter(true); //just looks better..i think?
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font", "");
	}

	/* Custom Bold font */

	String custom_font_path_bold = EditorSettings::get_singleton()->get("interface/editor/main_font_bold");
	Ref<FontData> CustomFontBold;
	if (custom_font_path_bold.length() > 0 && dir->file_exists(custom_font_path_bold)) {
		CustomFontBold.instance();
		CustomFontBold->load_resource(custom_font_path_bold, default_font_size);
		CustomFontBold->set_antialiased(font_antialiased);
		CustomFontBold->set_hinting(font_hinting);
		CustomFontBold->set_force_autohinter(true); //just looks better..i think?
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font_bold", "");
	}

	/* Custom source code font */

	String custom_font_path_source = EditorSettings::get_singleton()->get("interface/editor/code_font");
	Ref<FontData> CustomFontSource;
	if (custom_font_path_source.length() > 0 && dir->file_exists(custom_font_path_source)) {
		CustomFontSource.instance();
		CustomFontSource->load_resource(custom_font_path_source, default_font_size);
		CustomFontSource->set_antialiased(font_antialiased);
		CustomFontSource->set_hinting(font_hinting);

		Vector<String> subtag = String(EditorSettings::get_singleton()->get("interface/editor/code_font_custom_variations")).split(",");
		for (int i = 0; i < subtag.size(); i++) {
			Vector<String> subtag_a = subtag[i].split("=");
			if (subtag_a.size() == 2) {
				CustomFontSource->set_variation(subtag_a[0], subtag_a[1].to_float());
			}
		}
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/code_font", "");
	}

	memdelete(dir);

	/* Droid Sans */

	Ref<FontData> DefaultFont;
	DefaultFont.instance();
	DefaultFont->load_memory(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size, "ttf", default_font_size);
	DefaultFont->set_antialiased(font_antialiased);
	DefaultFont->set_hinting(font_hinting);
	DefaultFont->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> DefaultFontBold;
	DefaultFontBold.instance();
	DefaultFontBold->load_memory(_font_NotoSansUI_Bold, _font_NotoSansUI_Bold_size, "ttf", default_font_size);
	DefaultFontBold->set_antialiased(font_antialiased);
	DefaultFontBold->set_hinting(font_hinting);
	DefaultFontBold->set_force_autohinter(true); // just looks better..i think?

	Ref<FontData> FontFallback;
	FontFallback.instance();
	FontFallback->load_memory(_font_DroidSansFallback, _font_DroidSansFallback_size, "ttf", default_font_size);
	FontFallback->set_antialiased(font_antialiased);
	FontFallback->set_hinting(font_hinting);
	FontFallback->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontJapanese;
	FontJapanese.instance();
	FontJapanese->load_memory(_font_DroidSansJapanese, _font_DroidSansJapanese_size, "ttf", default_font_size);
	FontJapanese->set_antialiased(font_antialiased);
	FontJapanese->set_hinting(font_hinting);
	FontJapanese->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontArabic;
	FontArabic.instance();
	FontArabic->load_memory(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size, "ttf", default_font_size);
	FontArabic->set_antialiased(font_antialiased);
	FontArabic->set_hinting(font_hinting);
	FontArabic->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontBengali;
	FontBengali.instance();
	FontBengali->load_memory(_font_NotoSansBengali_Regular, _font_NotoSansBengali_Regular_size, "ttf", default_font_size);
	FontBengali->set_antialiased(font_antialiased);
	FontBengali->set_hinting(font_hinting);
	FontBengali->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontGeorgian;
	FontGeorgian.instance();
	FontGeorgian->load_memory(_font_NotoSansGeorgian_Regular, _font_NotoSansGeorgian_Regular_size, "ttf", default_font_size);
	FontGeorgian->set_antialiased(font_antialiased);
	FontGeorgian->set_hinting(font_hinting);
	FontGeorgian->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontHebrew;
	FontHebrew.instance();
	FontHebrew->load_memory(_font_NotoSansHebrew_Regular, _font_NotoSansHebrew_Regular_size, "ttf", default_font_size);
	FontHebrew->set_antialiased(font_antialiased);
	FontHebrew->set_hinting(font_hinting);
	FontHebrew->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontMalayalam;
	FontMalayalam.instance();
	FontMalayalam->load_memory(_font_NotoSansMalayalamUI_Regular, _font_NotoSansMalayalamUI_Regular_size, "ttf", default_font_size);
	FontMalayalam->set_antialiased(font_antialiased);
	FontMalayalam->set_hinting(font_hinting);
	FontMalayalam->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontOriya;
	FontOriya.instance();
	FontOriya->load_memory(_font_NotoSansOriyaUI_Regular, _font_NotoSansOriyaUI_Regular_size, "ttf", default_font_size);
	FontOriya->set_antialiased(font_antialiased);
	FontOriya->set_hinting(font_hinting);
	FontOriya->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontSinhala;
	FontSinhala.instance();
	FontSinhala->load_memory(_font_NotoSansSinhalaUI_Regular, _font_NotoSansSinhalaUI_Regular_size, "ttf", default_font_size);
	FontSinhala->set_antialiased(font_antialiased);
	FontSinhala->set_hinting(font_hinting);
	FontSinhala->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontTamil;
	FontTamil.instance();
	FontTamil->load_memory(_font_NotoSansTamilUI_Regular, _font_NotoSansTamilUI_Regular_size, "ttf", default_font_size);
	FontTamil->set_antialiased(font_antialiased);
	FontTamil->set_hinting(font_hinting);
	FontTamil->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontTelugu;
	FontTelugu.instance();
	FontTelugu->load_memory(_font_NotoSansTeluguUI_Regular, _font_NotoSansTeluguUI_Regular_size, "ttf", default_font_size);
	FontTelugu->set_antialiased(font_antialiased);
	FontTelugu->set_hinting(font_hinting);
	FontTelugu->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontThai;
	FontThai.instance();
	FontThai->load_memory(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size, "ttf", default_font_size);
	FontThai->set_antialiased(font_antialiased);
	FontThai->set_hinting(font_hinting);
	FontThai->set_force_autohinter(true); //just looks better..i think?

	Ref<FontData> FontHindi;
	FontHindi.instance();
	FontHindi->load_memory(_font_NotoSansDevanagariUI_Regular, _font_NotoSansDevanagariUI_Regular_size, "ttf", default_font_size);
	FontHindi->set_antialiased(font_antialiased);
	FontHindi->set_hinting(font_hinting);
	FontHindi->set_force_autohinter(true); //just looks better..i think?

	/* Hack */

	Ref<FontData> dfmono;
	dfmono.instance();
	dfmono->load_memory(_font_Hack_Regular, _font_Hack_Regular_size, "ttf", default_font_size);
	dfmono->set_antialiased(font_antialiased);
	dfmono->set_hinting(font_hinting);

	Vector<String> subtag = String(EditorSettings::get_singleton()->get("interface/editor/code_font_custom_variations")).split(",");
	Dictionary ftrs;
	for (int i = 0; i < subtag.size(); i++) {
		Vector<String> subtag_a = subtag[i].split("=");
		if (subtag_a.size() == 2) {
			dfmono->set_variation(subtag_a[0], subtag_a[1].to_float());
		}
	}

	// Default font
	MAKE_DEFAULT_FONT(df);
	p_theme->set_default_theme_font(df); // Default theme font
	p_theme->set_default_theme_font_size(default_font_size);

	p_theme->set_font_size("main_size", "EditorFonts", default_font_size);
	p_theme->set_font("main", "EditorFonts", df);

	// Bold font
	MAKE_BOLD_FONT(df_bold);
	p_theme->set_font_size("bold_size", "EditorFonts", default_font_size);
	p_theme->set_font("bold", "EditorFonts", df_bold);

	// Title font
	p_theme->set_font_size("title_size", "EditorFonts", default_font_size + 2 * EDSCALE);
	p_theme->set_font("title", "EditorFonts", df_bold);

	// Documentation fonts
	MAKE_SOURCE_FONT(df_code);
	p_theme->set_font_size("doc_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_font_size")) * EDSCALE);
	p_theme->set_font("doc", "EditorFonts", df);
	p_theme->set_font_size("doc_bold_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_font_size")) * EDSCALE);
	p_theme->set_font("doc_bold", "EditorFonts", df_bold);
	p_theme->set_font_size("doc_title_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_title_font_size")) * EDSCALE);
	p_theme->set_font("doc_title", "EditorFonts", df_bold);
	p_theme->set_font_size("doc_source_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_source_font_size")) * EDSCALE);
	p_theme->set_font("doc_source", "EditorFonts", df_code);
	p_theme->set_font_size("doc_keyboard_size", "EditorFonts", (int(EDITOR_GET("text_editor/help/help_source_font_size")) - 1) * EDSCALE);
	p_theme->set_font("doc_keyboard", "EditorFonts", df_code);

	// Ruler font
	p_theme->set_font_size("rulers_size", "EditorFonts", 8 * EDSCALE);
	p_theme->set_font("rulers", "EditorFonts", df);

	// Rotation widget font
	p_theme->set_font_size("rotation_control_size", "EditorFonts", 14 * EDSCALE);
	p_theme->set_font("rotation_control", "EditorFonts", df);

	// Code font
	p_theme->set_font_size("source_size", "EditorFonts", int(EDITOR_GET("interface/editor/code_font_size")) * EDSCALE);
	p_theme->set_font("source", "EditorFonts", df_code);

	p_theme->set_font_size("expression_size", "EditorFonts", (int(EDITOR_GET("interface/editor/code_font_size")) - 1) * EDSCALE);
	p_theme->set_font("expression", "EditorFonts", df_code);

	p_theme->set_font_size("output_source_size", "EditorFonts", int(EDITOR_GET("run/output/font_size")) * EDSCALE);
	p_theme->set_font("output_source", "EditorFonts", df_code);

	p_theme->set_font_size("status_source_size", "EditorFonts", default_font_size);
	p_theme->set_font("status_source", "EditorFonts", df_code);
}
