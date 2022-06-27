/*************************************************************************/
/*  editor_fonts.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#include "core/io/dir_access.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/font.h"

#define MAKE_FALLBACKS(m_name)        \
	m_name->add_data(FontArabic);     \
	m_name->add_data(FontBengali);    \
	m_name->add_data(FontDevanagari); \
	m_name->add_data(FontGeorgian);   \
	m_name->add_data(FontHebrew);     \
	m_name->add_data(FontMalayalam);  \
	m_name->add_data(FontOriya);      \
	m_name->add_data(FontSinhala);    \
	m_name->add_data(FontTamil);      \
	m_name->add_data(FontTelugu);     \
	m_name->add_data(FontThai);       \
	m_name->add_data(FontJapanese);   \
	m_name->add_data(FontFallback);

// Note: In some languages, the use of italic/slanted fonts is controversial. Therefore, we are limiting simulated slant to the main font (Latin, Cyrillic, and Greek) and using bold fonts for the rest.

#define MAKE_FALLBACKS_SLANTED(m_name)    \
	m_name->add_data(FontArabicBold);     \
	m_name->add_data(FontBengaliBold);    \
	m_name->add_data(FontDevanagariBold); \
	m_name->add_data(FontGeorgianBold);   \
	m_name->add_data(FontHebrewBold);     \
	m_name->add_data(FontMalayalamBold);  \
	m_name->add_data(FontOriyaBold);      \
	m_name->add_data(FontSinhalaBold);    \
	m_name->add_data(FontTamilBold);      \
	m_name->add_data(FontTeluguBold);     \
	m_name->add_data(FontThaiBold);       \
	m_name->add_data(FontJapaneseBold);   \
	m_name->add_data(FontFallbackBold);

#define MAKE_FALLBACKS_BOLD(m_name)       \
	m_name->add_data(FontArabicBold);     \
	m_name->add_data(FontBengaliBold);    \
	m_name->add_data(FontDevanagariBold); \
	m_name->add_data(FontGeorgianBold);   \
	m_name->add_data(FontHebrewBold);     \
	m_name->add_data(FontMalayalamBold);  \
	m_name->add_data(FontOriyaBold);      \
	m_name->add_data(FontSinhalaBold);    \
	m_name->add_data(FontTamilBold);      \
	m_name->add_data(FontTeluguBold);     \
	m_name->add_data(FontThaiBold);       \
	m_name->add_data(FontJapaneseBold);   \
	m_name->add_data(FontFallbackBold);

#define MAKE_DEFAULT_FONT(m_name, m_variations)                       \
	Ref<Font> m_name;                                                 \
	m_name.instantiate();                                             \
	if (CustomFont.is_valid()) {                                      \
		m_name->add_data(CustomFont);                                 \
		m_name->add_data(DefaultFont);                                \
	} else {                                                          \
		m_name->add_data(DefaultFont);                                \
	}                                                                 \
	{                                                                 \
		Dictionary variations;                                        \
		if (!m_variations.is_empty()) {                               \
			Vector<String> variation_tags = m_variations.split(",");  \
			for (int i = 0; i < variation_tags.size(); i++) {         \
				Vector<String> tokens = variation_tags[i].split("="); \
				if (tokens.size() == 2) {                             \
					variations[tokens[0]] = tokens[1].to_float();     \
				}                                                     \
			}                                                         \
		}                                                             \
		m_name->set_variation_coordinates(variations);                \
	}                                                                 \
	m_name->set_spacing(TextServer::SPACING_TOP, -EDSCALE);           \
	m_name->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);        \
	MAKE_FALLBACKS(m_name);

#define MAKE_DEFAULT_FONT_MSDF(m_name, m_variations)                  \
	Ref<Font> m_name;                                                 \
	m_name.instantiate();                                             \
	if (CustomFont.is_valid()) {                                      \
		m_name->add_data(CustomFontMSDF);                             \
		m_name->add_data(DefaultFontMSDF);                            \
	} else {                                                          \
		m_name->add_data(DefaultFontMSDF);                            \
	}                                                                 \
	{                                                                 \
		Dictionary variations;                                        \
		if (!m_variations.is_empty()) {                               \
			Vector<String> variation_tags = m_variations.split(",");  \
			for (int i = 0; i < variation_tags.size(); i++) {         \
				Vector<String> tokens = variation_tags[i].split("="); \
				if (tokens.size() == 2) {                             \
					variations[tokens[0]] = tokens[1].to_float();     \
				}                                                     \
			}                                                         \
		}                                                             \
		m_name->set_variation_coordinates(variations);                \
	}                                                                 \
	m_name->set_spacing(TextServer::SPACING_TOP, -EDSCALE);           \
	m_name->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);        \
	MAKE_FALLBACKS(m_name);

#define MAKE_SLANTED_FONT(m_name, m_variations)                       \
	Ref<Font> m_name;                                                 \
	m_name.instantiate();                                             \
	m_name.instantiate();                                             \
	if (CustomFontSlanted.is_valid()) {                               \
		m_name->add_data(CustomFontSlanted);                          \
		m_name->add_data(DefaultFontSlanted);                         \
	} else {                                                          \
		m_name->add_data(DefaultFontSlanted);                         \
	}                                                                 \
	{                                                                 \
		Dictionary variations;                                        \
		if (!m_variations.is_empty()) {                               \
			Vector<String> variation_tags = m_variations.split(",");  \
			for (int i = 0; i < variation_tags.size(); i++) {         \
				Vector<String> tokens = variation_tags[i].split("="); \
				if (tokens.size() == 2) {                             \
					variations[tokens[0]] = tokens[1].to_float();     \
				}                                                     \
			}                                                         \
		}                                                             \
		m_name->set_variation_coordinates(variations);                \
	}                                                                 \
	m_name->set_spacing(TextServer::SPACING_TOP, -EDSCALE);           \
	m_name->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);        \
	MAKE_FALLBACKS_SLANTED(m_name);

#define MAKE_BOLD_FONT(m_name, m_variations)                          \
	Ref<Font> m_name;                                                 \
	m_name.instantiate();                                             \
	if (CustomFontBold.is_valid()) {                                  \
		m_name->add_data(CustomFontBold);                             \
		m_name->add_data(DefaultFontBold);                            \
	} else {                                                          \
		m_name->add_data(DefaultFontBold);                            \
	}                                                                 \
	{                                                                 \
		Dictionary variations;                                        \
		if (!m_variations.is_empty()) {                               \
			Vector<String> variation_tags = m_variations.split(",");  \
			for (int i = 0; i < variation_tags.size(); i++) {         \
				Vector<String> tokens = variation_tags[i].split("="); \
				if (tokens.size() == 2) {                             \
					variations[tokens[0]] = tokens[1].to_float();     \
				}                                                     \
			}                                                         \
		}                                                             \
		m_name->set_variation_coordinates(variations);                \
	}                                                                 \
	m_name->set_spacing(TextServer::SPACING_TOP, -EDSCALE);           \
	m_name->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);        \
	MAKE_FALLBACKS_BOLD(m_name);

#define MAKE_BOLD_FONT_MSDF(m_name, m_variations)                     \
	Ref<Font> m_name;                                                 \
	m_name.instantiate();                                             \
	if (CustomFontBold.is_valid()) {                                  \
		m_name->add_data(CustomFontBoldMSDF);                         \
		m_name->add_data(DefaultFontBoldMSDF);                        \
	} else {                                                          \
		m_name->add_data(DefaultFontBoldMSDF);                        \
	}                                                                 \
	{                                                                 \
		Dictionary variations;                                        \
		if (!m_variations.is_empty()) {                               \
			Vector<String> variation_tags = m_variations.split(",");  \
			for (int i = 0; i < variation_tags.size(); i++) {         \
				Vector<String> tokens = variation_tags[i].split("="); \
				if (tokens.size() == 2) {                             \
					variations[tokens[0]] = tokens[1].to_float();     \
				}                                                     \
			}                                                         \
		}                                                             \
		m_name->set_variation_coordinates(variations);                \
	}                                                                 \
	m_name->set_spacing(TextServer::SPACING_TOP, -EDSCALE);           \
	m_name->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);        \
	MAKE_FALLBACKS_BOLD(m_name);

#define MAKE_SOURCE_FONT(m_name, m_variations)                        \
	Ref<Font> m_name;                                                 \
	m_name.instantiate();                                             \
	if (CustomFontSource.is_valid()) {                                \
		m_name->add_data(CustomFontSource);                           \
		m_name->add_data(dfmono);                                     \
	} else {                                                          \
		m_name->add_data(dfmono);                                     \
	}                                                                 \
	{                                                                 \
		Dictionary variations;                                        \
		if (!m_variations.is_empty()) {                               \
			Vector<String> variation_tags = m_variations.split(",");  \
			for (int i = 0; i < variation_tags.size(); i++) {         \
				Vector<String> tokens = variation_tags[i].split("="); \
				if (tokens.size() == 2) {                             \
					variations[tokens[0]] = tokens[1].to_float();     \
				}                                                     \
			}                                                         \
		}                                                             \
		m_name->set_variation_coordinates(variations);                \
	}                                                                 \
	m_name->set_spacing(TextServer::SPACING_TOP, -EDSCALE);           \
	m_name->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);        \
	MAKE_FALLBACKS(m_name);

Ref<FontData> load_cached_external_font(const String &p_path, TextServer::Hinting p_hinting, bool p_aa, bool p_autohint, TextServer::SubpixelPositioning p_font_subpixel_positioning, bool p_msdf = false) {
	Ref<FontData> font;
	font.instantiate();

	Vector<uint8_t> data = FileAccess::get_file_as_array(p_path);

	font->set_data(data);
	font->set_multichannel_signed_distance_field(p_msdf);
	font->set_antialiased(p_aa);
	font->set_hinting(p_hinting);
	font->set_force_autohinter(p_autohint);
	font->set_subpixel_positioning(p_font_subpixel_positioning);

	return font;
}

Ref<FontData> load_cached_internal_font(const uint8_t *p_data, size_t p_size, TextServer::Hinting p_hinting, bool p_aa, bool p_autohint, TextServer::SubpixelPositioning p_font_subpixel_positioning, bool p_msdf = false) {
	Ref<FontData> font;
	font.instantiate();

	font->set_data_ptr(p_data, p_size);
	font->set_multichannel_signed_distance_field(p_msdf);
	font->set_antialiased(p_aa);
	font->set_hinting(p_hinting);
	font->set_force_autohinter(p_autohint);
	font->set_subpixel_positioning(p_font_subpixel_positioning);

	return font;
}

void editor_register_fonts(Ref<Theme> p_theme) {
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);

	/* Custom font */

	bool font_antialiased = (bool)EditorSettings::get_singleton()->get("interface/editor/font_antialiased");
	int font_hinting_setting = (int)EditorSettings::get_singleton()->get("interface/editor/font_hinting");
	TextServer::SubpixelPositioning font_subpixel_positioning = (TextServer::SubpixelPositioning)(int)EditorSettings::get_singleton()->get("interface/editor/font_subpixel_positioning");

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

	const int default_font_size = int(EDITOR_GET("interface/editor/main_font_size")) * EDSCALE;
	const float embolden_strength = 0.6;

	String custom_font_path = EditorSettings::get_singleton()->get("interface/editor/main_font");
	Ref<FontData> CustomFont;
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		CustomFont = load_cached_external_font(custom_font_path, font_hinting, font_antialiased, true, font_subpixel_positioning);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font", "");
	}

	Ref<FontData> CustomFontMSDF;
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		CustomFontMSDF = load_cached_external_font(custom_font_path, font_hinting, font_antialiased, true, font_subpixel_positioning, true);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font", "");
	}

	Ref<FontData> CustomFontSlanted;
	if (CustomFont.is_valid()) {
		CustomFontSlanted = CustomFont->duplicate();
		CustomFontSlanted->set_transform(Transform2D(1.0, 0.4, 0.0, 1.0, 0.0, 0.0));
	}

	/* Custom Bold font */

	String custom_font_path_bold = EditorSettings::get_singleton()->get("interface/editor/main_font_bold");
	Ref<FontData> CustomFontBold;
	if (custom_font_path_bold.length() > 0 && dir->file_exists(custom_font_path_bold)) {
		CustomFontBold = load_cached_external_font(custom_font_path_bold, font_hinting, font_antialiased, true, font_subpixel_positioning);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font_bold", "");
	}

	if (CustomFont.is_valid() && !CustomFontBold.is_valid()) {
		CustomFontBold = CustomFont->duplicate();
		CustomFontBold->set_embolden(embolden_strength);
	}

	Ref<FontData> CustomFontBoldMSDF;
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		CustomFontBoldMSDF = load_cached_external_font(custom_font_path, font_hinting, font_antialiased, true, font_subpixel_positioning, true);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font_bold", "");
	}

	/* Custom source code font */

	String custom_font_path_source = EditorSettings::get_singleton()->get("interface/editor/code_font");
	Ref<FontData> CustomFontSource;
	if (custom_font_path_source.length() > 0 && dir->file_exists(custom_font_path_source)) {
		CustomFontSource = load_cached_external_font(custom_font_path_source, font_hinting, font_antialiased, true, font_subpixel_positioning);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/code_font", "");
	}

	/* Noto Sans */

	Ref<FontData> DefaultFont = load_cached_internal_font(_font_NotoSans_Regular, _font_NotoSans_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> DefaultFontMSDF = load_cached_internal_font(_font_NotoSans_Regular, _font_NotoSans_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning, true);
	Ref<FontData> DefaultFontBold = load_cached_internal_font(_font_NotoSans_Bold, _font_NotoSans_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> DefaultFontBoldMSDF = load_cached_internal_font(_font_NotoSans_Bold, _font_NotoSans_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning, true);
	Ref<FontData> FontArabic = load_cached_internal_font(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontArabicBold = load_cached_internal_font(_font_NotoNaskhArabicUI_Bold, _font_NotoNaskhArabicUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontBengali = load_cached_internal_font(_font_NotoSansBengaliUI_Regular, _font_NotoSansBengaliUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontBengaliBold = load_cached_internal_font(_font_NotoSansBengaliUI_Bold, _font_NotoSansBengaliUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontDevanagari = load_cached_internal_font(_font_NotoSansDevanagariUI_Regular, _font_NotoSansDevanagariUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontDevanagariBold = load_cached_internal_font(_font_NotoSansDevanagariUI_Bold, _font_NotoSansDevanagariUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontGeorgian = load_cached_internal_font(_font_NotoSansGeorgian_Regular, _font_NotoSansGeorgian_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontGeorgianBold = load_cached_internal_font(_font_NotoSansGeorgian_Bold, _font_NotoSansGeorgian_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontHebrew = load_cached_internal_font(_font_NotoSansHebrew_Regular, _font_NotoSansHebrew_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontHebrewBold = load_cached_internal_font(_font_NotoSansHebrew_Bold, _font_NotoSansHebrew_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontMalayalam = load_cached_internal_font(_font_NotoSansMalayalamUI_Regular, _font_NotoSansMalayalamUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontMalayalamBold = load_cached_internal_font(_font_NotoSansMalayalamUI_Bold, _font_NotoSansMalayalamUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontOriya = load_cached_internal_font(_font_NotoSansOriyaUI_Regular, _font_NotoSansOriyaUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontOriyaBold = load_cached_internal_font(_font_NotoSansOriyaUI_Bold, _font_NotoSansOriyaUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontSinhala = load_cached_internal_font(_font_NotoSansSinhalaUI_Regular, _font_NotoSansSinhalaUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontSinhalaBold = load_cached_internal_font(_font_NotoSansSinhalaUI_Bold, _font_NotoSansSinhalaUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontTamil = load_cached_internal_font(_font_NotoSansTamilUI_Regular, _font_NotoSansTamilUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontTamilBold = load_cached_internal_font(_font_NotoSansTamilUI_Bold, _font_NotoSansTamilUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontTelugu = load_cached_internal_font(_font_NotoSansTeluguUI_Regular, _font_NotoSansTeluguUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontTeluguBold = load_cached_internal_font(_font_NotoSansTeluguUI_Bold, _font_NotoSansTeluguUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontThai = load_cached_internal_font(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontThaiBold = load_cached_internal_font(_font_NotoSansThaiUI_Bold, _font_NotoSansThaiUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);

	Ref<FontData> DefaultFontSlanted = DefaultFont->duplicate();
	DefaultFontSlanted->set_transform(Transform2D(1.0, 0.3, 0.0, 1.0, 0.0, 0.0));

	/* Droid Sans */

	Ref<FontData> FontFallback = load_cached_internal_font(_font_DroidSansFallback, _font_DroidSansFallback_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<FontData> FontJapanese = load_cached_internal_font(_font_DroidSansJapanese, _font_DroidSansJapanese_size, font_hinting, font_antialiased, true, font_subpixel_positioning);

	Ref<FontData> FontFallbackBold = FontFallback->duplicate();
	FontFallbackBold->set_embolden(embolden_strength);
	Ref<FontData> FontJapaneseBold = FontJapanese->duplicate();
	FontJapaneseBold->set_embolden(embolden_strength);

	/* Hack */

	Ref<FontData> dfmono = load_cached_internal_font(_font_JetBrainsMono_Regular, _font_JetBrainsMono_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Dictionary opentype_features;
	opentype_features["calt"] = 0;
	dfmono->set_opentype_feature_overrides(opentype_features); // Disable contextual alternates (coding ligatures).

	// Default font
	MAKE_DEFAULT_FONT(df, String());
	p_theme->set_default_font(df); // Default theme font
	p_theme->set_default_font_size(default_font_size);

	p_theme->set_font_size("main_size", "EditorFonts", default_font_size);
	p_theme->set_font("main", "EditorFonts", df);

	MAKE_DEFAULT_FONT_MSDF(df_msdf, String());
	p_theme->set_font("main_msdf", "EditorFonts", df_msdf);

	// Bold font
	MAKE_BOLD_FONT(df_bold, String());
	MAKE_SLANTED_FONT(df_italic, String());
	p_theme->set_font_size("bold_size", "EditorFonts", default_font_size);
	p_theme->set_font("bold", "EditorFonts", df_bold);

	MAKE_BOLD_FONT_MSDF(df_bold_msdf, String());
	p_theme->set_font("main_bold_msdf", "EditorFonts", df_bold_msdf);

	// Title font
	p_theme->set_font_size("title_size", "EditorFonts", default_font_size + 1 * EDSCALE);
	p_theme->set_font("title", "EditorFonts", df_bold);

	p_theme->set_font_size("main_button_font_size", "EditorFonts", default_font_size + 1 * EDSCALE);
	p_theme->set_font("main_button_font", "EditorFonts", df_bold);

	p_theme->set_font("font", "Label", df);

	p_theme->set_type_variation("HeaderSmall", "Label");
	p_theme->set_font("font", "HeaderSmall", df_bold);
	p_theme->set_font_size("font_size", "HeaderSmall", default_font_size);

	p_theme->set_type_variation("HeaderMedium", "Label");
	p_theme->set_font("font", "HeaderMedium", df_bold);
	p_theme->set_font_size("font_size", "HeaderMedium", default_font_size + 1 * EDSCALE);

	p_theme->set_type_variation("HeaderLarge", "Label");
	p_theme->set_font("font", "HeaderLarge", df_bold);
	p_theme->set_font_size("font_size", "HeaderLarge", default_font_size + 3 * EDSCALE);

	// Documentation fonts
	String code_font_custom_variations = EditorSettings::get_singleton()->get("interface/editor/code_font_custom_variations");
	MAKE_SOURCE_FONT(df_code, code_font_custom_variations);
	p_theme->set_font_size("doc_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_font_size")) * EDSCALE);
	p_theme->set_font("doc", "EditorFonts", df);
	p_theme->set_font("doc_bold", "EditorFonts", df_bold);
	p_theme->set_font("doc_italic", "EditorFonts", df_italic);
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
