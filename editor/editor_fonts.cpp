/**************************************************************************/
/*  editor_fonts.cpp                                                      */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "editor_fonts.h"

#include "builtin_fonts.gen.h"
#include "core/os/dir_access.h"
#include "editor_scale.h"
#include "editor_settings.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/dynamic_font.h"

#define MAKE_FALLBACKS(m_name)          \
	m_name->add_fallback(FontArabic);   \
	m_name->add_fallback(FontHebrew);   \
	m_name->add_fallback(FontThai);     \
	m_name->add_fallback(FontHindi);    \
	m_name->add_fallback(FontJapanese); \
	m_name->add_fallback(FontFallback);

// Enable filtering and mipmaps on the editor fonts to improve text appearance
// in editors that are zoomed in/out without having dedicated fonts to generate.
// This is the case in GraphEdit-based editors such as the visual script and
// visual shader editors.

// the custom spacings might only work with Noto Sans
#define MAKE_DEFAULT_FONT(m_name, m_size)                       \
	Ref<DynamicFont> m_name;                                    \
	m_name.instance();                                          \
	m_name->set_size(m_size);                                   \
	m_name->set_use_filter(true);                               \
	m_name->set_use_mipmaps(true);                              \
	if (CustomFont.is_valid()) {                                \
		m_name->set_font_data(CustomFont);                      \
		m_name->add_fallback(DefaultFont);                      \
	} else {                                                    \
		m_name->set_font_data(DefaultFont);                     \
	}                                                           \
	m_name->set_spacing(DynamicFont::SPACING_TOP, -EDSCALE);    \
	m_name->set_spacing(DynamicFont::SPACING_BOTTOM, -EDSCALE); \
	MAKE_FALLBACKS(m_name);

#define MAKE_BOLD_FONT(m_name, m_size)                          \
	Ref<DynamicFont> m_name;                                    \
	m_name.instance();                                          \
	m_name->set_size(m_size);                                   \
	m_name->set_use_filter(true);                               \
	m_name->set_use_mipmaps(true);                              \
	if (CustomFontBold.is_valid()) {                            \
		m_name->set_font_data(CustomFontBold);                  \
		m_name->add_fallback(DefaultFontBold);                  \
	} else {                                                    \
		m_name->set_font_data(DefaultFontBold);                 \
	}                                                           \
	m_name->set_spacing(DynamicFont::SPACING_TOP, -EDSCALE);    \
	m_name->set_spacing(DynamicFont::SPACING_BOTTOM, -EDSCALE); \
	MAKE_FALLBACKS(m_name);

#define MAKE_SOURCE_FONT(m_name, m_size)                        \
	Ref<DynamicFont> m_name;                                    \
	m_name.instance();                                          \
	m_name->set_size(m_size);                                   \
	m_name->set_use_filter(true);                               \
	m_name->set_use_mipmaps(true);                              \
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
	DirAccess *dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	/* Custom font */

	bool font_antialiased = (bool)EditorSettings::get_singleton()->get("interface/editor/font_antialiased");
	int font_hinting_setting = (int)EditorSettings::get_singleton()->get("interface/editor/font_hinting");

	DynamicFontData::Hinting font_hinting;
	switch (font_hinting_setting) {
		case 0:
			// The "Auto" setting uses the setting that best matches the OS' font rendering:
			// - macOS doesn't use font hinting.
			// - Windows uses ClearType, which is in between "Light" and "Normal" hinting.
			// - Linux has configurable font hinting, but most distributions including Ubuntu default to "Light".
#ifdef OSX_ENABLED
			font_hinting = DynamicFontData::HINTING_NONE;
#else
			font_hinting = DynamicFontData::HINTING_LIGHT;
#endif
			break;
		case 1:
			font_hinting = DynamicFontData::HINTING_NONE;
			break;
		case 2:
			font_hinting = DynamicFontData::HINTING_LIGHT;
			break;
		default:
			font_hinting = DynamicFontData::HINTING_NORMAL;
			break;
	}

	String custom_font_path = EditorSettings::get_singleton()->get("interface/editor/main_font");
	Ref<DynamicFontData> CustomFont;
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		CustomFont.instance();
		CustomFont->set_antialiased(font_antialiased);
		CustomFont->set_hinting(font_hinting);
		CustomFont->set_font_path(custom_font_path);
		CustomFont->set_force_autohinter(true); //just looks better..i think?
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font", "");
	}

	/* Custom Bold font */

	String custom_font_path_bold = EditorSettings::get_singleton()->get("interface/editor/main_font_bold");
	Ref<DynamicFontData> CustomFontBold;
	if (custom_font_path_bold.length() > 0 && dir->file_exists(custom_font_path_bold)) {
		CustomFontBold.instance();
		CustomFontBold->set_antialiased(font_antialiased);
		CustomFontBold->set_hinting(font_hinting);
		CustomFontBold->set_font_path(custom_font_path_bold);
		CustomFontBold->set_force_autohinter(true); //just looks better..i think?
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font_bold", "");
	}

	/* Custom source code font */

	String custom_font_path_source = EditorSettings::get_singleton()->get("interface/editor/code_font");
	Ref<DynamicFontData> CustomFontSource;
	if (custom_font_path_source.length() > 0 && dir->file_exists(custom_font_path_source)) {
		CustomFontSource.instance();
		CustomFontSource->set_antialiased(font_antialiased);
		CustomFontSource->set_hinting(font_hinting);
		CustomFontSource->set_font_path(custom_font_path_source);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/code_font", "");
	}

	memdelete(dir);

	/* Droid Sans */

	Ref<DynamicFontData> DefaultFont;
	DefaultFont.instance();
	DefaultFont->set_antialiased(font_antialiased);
	DefaultFont->set_hinting(font_hinting);
	DefaultFont->set_font_ptr(_font_NotoSansUI_Regular, _font_NotoSansUI_Regular_size);
	DefaultFont->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> DefaultFontBold;
	DefaultFontBold.instance();
	DefaultFontBold->set_antialiased(font_antialiased);
	DefaultFontBold->set_hinting(font_hinting);
	DefaultFontBold->set_font_ptr(_font_NotoSansUI_Bold, _font_NotoSansUI_Bold_size);
	DefaultFontBold->set_force_autohinter(true); // just looks better..i think?

	Ref<DynamicFontData> FontFallback;
	FontFallback.instance();
	FontFallback->set_antialiased(font_antialiased);
	FontFallback->set_hinting(font_hinting);
	FontFallback->set_font_ptr(_font_DroidSansFallback, _font_DroidSansFallback_size);
	FontFallback->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontJapanese;
	FontJapanese.instance();
	FontJapanese->set_antialiased(font_antialiased);
	FontJapanese->set_hinting(font_hinting);
	FontJapanese->set_font_ptr(_font_DroidSansJapanese, _font_DroidSansJapanese_size);
	FontJapanese->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontArabic;
	FontArabic.instance();
	FontArabic->set_antialiased(font_antialiased);
	FontArabic->set_hinting(font_hinting);
	FontArabic->set_font_ptr(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size);
	FontArabic->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontHebrew;
	FontHebrew.instance();
	FontHebrew->set_antialiased(font_antialiased);
	FontHebrew->set_hinting(font_hinting);
	FontHebrew->set_font_ptr(_font_NotoSansHebrew_Regular, _font_NotoSansHebrew_Regular_size);
	FontHebrew->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontThai;
	FontThai.instance();
	FontThai->set_antialiased(font_antialiased);
	FontThai->set_hinting(font_hinting);
	FontThai->set_font_ptr(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size);
	FontThai->set_force_autohinter(true); //just looks better..i think?

	Ref<DynamicFontData> FontHindi;
	FontHindi.instance();
	FontHindi->set_antialiased(font_antialiased);
	FontHindi->set_hinting(font_hinting);
	FontHindi->set_font_ptr(_font_NotoSansDevanagariUI_Regular, _font_NotoSansDevanagariUI_Regular_size);
	FontHindi->set_force_autohinter(true); //just looks better..i think?

	/* Hack */

	Ref<DynamicFontData> dfmono;
	dfmono.instance();
	dfmono->set_antialiased(font_antialiased);
	dfmono->set_hinting(font_hinting);
	dfmono->set_font_ptr(_font_Hack_Regular, _font_Hack_Regular_size);

	int default_font_size = int(EDITOR_GET("interface/editor/main_font_size")) * EDSCALE;

	// Default font
	MAKE_DEFAULT_FONT(df, default_font_size);
	p_theme->set_default_theme_font(df);
	p_theme->set_font("main", "EditorFonts", df);

	// Bold font
	MAKE_BOLD_FONT(df_bold, default_font_size);
	p_theme->set_font("bold", "EditorFonts", df_bold);

	// Title font
	MAKE_BOLD_FONT(df_title, default_font_size + 2 * EDSCALE);
	p_theme->set_font("title", "EditorFonts", df_title);

	// Documentation fonts
	MAKE_DEFAULT_FONT(df_doc, int(EDITOR_GET("text_editor/help/help_font_size")) * EDSCALE);
	MAKE_BOLD_FONT(df_doc_bold, int(EDITOR_GET("text_editor/help/help_font_size")) * EDSCALE);
	MAKE_BOLD_FONT(df_doc_title, int(EDITOR_GET("text_editor/help/help_title_font_size")) * EDSCALE);
	MAKE_SOURCE_FONT(df_doc_code, int(EDITOR_GET("text_editor/help/help_source_font_size")) * EDSCALE);
	MAKE_SOURCE_FONT(df_doc_kbd, (int(EDITOR_GET("text_editor/help/help_source_font_size")) - 1) * EDSCALE);
	p_theme->set_font("doc", "EditorFonts", df_doc);
	p_theme->set_font("doc_bold", "EditorFonts", df_doc_bold);
	p_theme->set_font("doc_title", "EditorFonts", df_doc_title);
	p_theme->set_font("doc_source", "EditorFonts", df_doc_code);
	p_theme->set_font("doc_keyboard", "EditorFonts", df_doc_kbd);

	// Ruler font
	MAKE_DEFAULT_FONT(df_rulers, 8 * EDSCALE);
	p_theme->set_font("rulers", "EditorFonts", df_rulers);

	// Rotation widget font
	MAKE_DEFAULT_FONT(df_rotation_control, 14 * EDSCALE);
	p_theme->set_font("rotation_control", "EditorFonts", df_rotation_control);

	// Code font
	MAKE_SOURCE_FONT(df_code, int(EDITOR_GET("interface/editor/code_font_size")) * EDSCALE);
	p_theme->set_font("source", "EditorFonts", df_code);

	MAKE_SOURCE_FONT(df_expression, (int(EDITOR_GET("interface/editor/code_font_size")) - 1) * EDSCALE);
	p_theme->set_font("expression", "EditorFonts", df_expression);

	MAKE_SOURCE_FONT(df_output_code, int(EDITOR_GET("run/output/font_size")) * EDSCALE);
	p_theme->set_font("output_source", "EditorFonts", df_output_code);

	MAKE_SOURCE_FONT(df_text_editor_status_code, default_font_size);
	p_theme->set_font("status_source", "EditorFonts", df_text_editor_status_code);
}
