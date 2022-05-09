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

Ref<Font> load_external_font(const String &p_path, TextServer::Hinting p_hinting, bool p_aa, bool p_autohint, TextServer::SubpixelPositioning p_font_subpixel_positioning, bool p_msdf = false) {
	Ref<Font> font;
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

Ref<Font> load_internal_font(const uint8_t *p_data, size_t p_size, TextServer::Hinting p_hinting, bool p_aa, bool p_autohint, TextServer::SubpixelPositioning p_font_subpixel_positioning, bool p_msdf = false) {
	Ref<Font> font;
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

	// Load built-in fonts.

	Ref<Font> default_font = load_internal_font(_font_NotoSans_Regular, _font_NotoSans_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<Font> default_font_msdf = load_internal_font(_font_NotoSans_Regular, _font_NotoSans_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning, true);

	Ref<Font> default_font_bold = load_internal_font(_font_NotoSans_Bold, _font_NotoSans_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	Ref<Font> default_font_bold_msdf = load_internal_font(_font_NotoSans_Bold, _font_NotoSans_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning, true);

	Ref<Font> default_font_mono = load_internal_font(_font_JetBrainsMono_Regular, _font_JetBrainsMono_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning);
	{
		Dictionary opentype_features_mono;
		opentype_features_mono["calt"] = 0;
		default_font_mono->set_opentype_feature_overrides(opentype_features_mono); // Disable contextual alternates (coding ligatures).
	}

	TypedArray<Font> fallback;
	fallback.push_back(load_internal_font(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 0
	fallback.push_back(load_internal_font(_font_NotoSansBengaliUI_Regular, _font_NotoSansBengaliUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 1
	fallback.push_back(load_internal_font(_font_NotoSansDevanagariUI_Regular, _font_NotoSansDevanagariUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 2
	fallback.push_back(load_internal_font(_font_NotoSansGeorgian_Regular, _font_NotoSansGeorgian_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 3
	fallback.push_back(load_internal_font(_font_NotoSansHebrew_Regular, _font_NotoSansHebrew_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 4
	fallback.push_back(load_internal_font(_font_NotoSansMalayalamUI_Regular, _font_NotoSansMalayalamUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 5
	fallback.push_back(load_internal_font(_font_NotoSansOriyaUI_Regular, _font_NotoSansOriyaUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 6
	fallback.push_back(load_internal_font(_font_NotoSansSinhalaUI_Regular, _font_NotoSansSinhalaUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 7
	fallback.push_back(load_internal_font(_font_NotoSansTamilUI_Regular, _font_NotoSansTamilUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 8
	fallback.push_back(load_internal_font(_font_NotoSansTeluguUI_Regular, _font_NotoSansTeluguUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 9
	fallback.push_back(load_internal_font(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 10
	fallback.push_back(load_internal_font(_font_DroidSansFallback, _font_DroidSansFallback_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 11
	fallback.push_back(load_internal_font(_font_DroidSansJapanese, _font_DroidSansJapanese_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 12
	default_font->set_fallbacks(fallback);
	default_font_msdf->set_fallbacks(fallback);
	default_font_mono->set_fallbacks(fallback);

	TypedArray<Font> fallback_bold;
	fallback_bold.push_back(load_internal_font(_font_NotoNaskhArabicUI_Bold, _font_NotoNaskhArabicUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 0
	fallback_bold.push_back(load_internal_font(_font_NotoSansBengaliUI_Bold, _font_NotoSansBengaliUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 1
	fallback_bold.push_back(load_internal_font(_font_NotoSansDevanagariUI_Bold, _font_NotoSansDevanagariUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 2
	fallback_bold.push_back(load_internal_font(_font_NotoSansGeorgian_Bold, _font_NotoSansGeorgian_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 3
	fallback_bold.push_back(load_internal_font(_font_NotoSansHebrew_Bold, _font_NotoSansHebrew_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 4
	fallback_bold.push_back(load_internal_font(_font_NotoSansMalayalamUI_Bold, _font_NotoSansMalayalamUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 5
	fallback_bold.push_back(load_internal_font(_font_NotoSansOriyaUI_Bold, _font_NotoSansOriyaUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 6
	fallback_bold.push_back(load_internal_font(_font_NotoSansSinhalaUI_Bold, _font_NotoSansSinhalaUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 7
	fallback_bold.push_back(load_internal_font(_font_NotoSansTamilUI_Bold, _font_NotoSansTamilUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 8
	fallback_bold.push_back(load_internal_font(_font_NotoSansTeluguUI_Bold, _font_NotoSansTeluguUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 9
	fallback_bold.push_back(load_internal_font(_font_NotoSansThaiUI_Bold, _font_NotoSansThaiUI_Bold_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 10
	fallback_bold.push_back(load_internal_font(_font_DroidSansFallback, _font_DroidSansFallback_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 11
	fallback_bold.push_back(load_internal_font(_font_DroidSansJapanese, _font_DroidSansJapanese_size, font_hinting, font_antialiased, true, font_subpixel_positioning)); // 12
	default_font_bold->set_fallbacks(fallback_bold);
	default_font_bold_msdf->set_fallbacks(fallback_bold);

	// Init base font configs and load custom fonts.
	String custom_font_path = EditorSettings::get_singleton()->get("interface/editor/main_font");
	String custom_font_path_bold = EditorSettings::get_singleton()->get("interface/editor/main_font_bold");
	String custom_font_path_source = EditorSettings::get_singleton()->get("interface/editor/code_font");

	const int default_font_size = int(EDITOR_GET("interface/editor/main_font_size")) * EDSCALE;
	const float embolden_strength = 0.6;

	Ref<FontConfig> default_fc;
	default_fc.instantiate();
	default_fc->set_variation_customize_fallbacks(true); // Apply variations to each of the fallback individually, some languages do not use bold and italic..
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		Ref<Font> custom_font = load_external_font(custom_font_path, font_hinting, font_antialiased, true, font_subpixel_positioning);
		{
			TypedArray<Font> fallback_custom;
			fallback_custom.push_back(default_font);
			custom_font->set_fallbacks(fallback_custom);
		}
		default_fc->set_font(custom_font);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font", "");
		default_fc->set_font(default_font);
	}
	default_fc->set_spacing(TextServer::SPACING_TOP, -EDSCALE);
	default_fc->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);

	Ref<FontConfig> default_fc_msdf;
	default_fc_msdf.instantiate();
	default_fc_msdf->set_variation_customize_fallbacks(true); // Apply variations to each of the fallback individually, some languages do not use bold and italic..
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		Ref<Font> custom_font = load_external_font(custom_font_path, font_hinting, font_antialiased, true, font_subpixel_positioning);
		{
			TypedArray<Font> fallback_custom;
			fallback_custom.push_back(default_font_msdf);
			custom_font->set_fallbacks(fallback_custom);
		}
		default_fc_msdf->set_font(custom_font);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font", "");
		default_fc_msdf->set_font(default_font_msdf);
	}
	default_fc_msdf->set_spacing(TextServer::SPACING_TOP, -EDSCALE);
	default_fc_msdf->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);

	Ref<FontConfig> bold_fc;
	bold_fc.instantiate();
	bold_fc->set_variation_customize_fallbacks(true); // Apply variations to each of the fallback individually, some languages do not use bold and italic..
	if (custom_font_path_bold.length() > 0 && dir->file_exists(custom_font_path_bold)) {
		Ref<Font> custom_font = load_external_font(custom_font_path_bold, font_hinting, font_antialiased, true, font_subpixel_positioning);
		{
			TypedArray<Font> fallback_custom;
			fallback_custom.push_back(default_font_bold);
			custom_font->set_fallbacks(fallback_custom);
		}
		bold_fc->set_font(custom_font);
		bold_fc->set("variation_fallback/0/11/variation_embolden", embolden_strength); // DroidSansFallback
		bold_fc->set("variation_fallback/0/12/variation_embolden", embolden_strength); // DroidSansJapanese
	} else if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		Ref<Font> custom_font = load_external_font(custom_font_path, font_hinting, font_antialiased, true, font_subpixel_positioning);
		{
			TypedArray<Font> fallback_custom;
			fallback_custom.push_back(default_font_bold);
			custom_font->set_fallbacks(fallback_custom);
		}
		bold_fc->set_font(custom_font);
		bold_fc->set_variation_embolden(embolden_strength);
		bold_fc->set("variation_fallback/0/11/variation_embolden", embolden_strength); // DroidSansFallback
		bold_fc->set("variation_fallback/0/12/variation_embolden", embolden_strength); // DroidSansJapanese
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font_bold", "");
		bold_fc->set_font(default_font_bold);
		bold_fc->set("variation_fallback/11/variation_embolden", embolden_strength); // DroidSansFallback
		bold_fc->set("variation_fallback/12/variation_embolden", embolden_strength); // DroidSansJapanese
	}
	bold_fc->set_spacing(TextServer::SPACING_TOP, -EDSCALE);
	bold_fc->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);

	Ref<FontConfig> bold_fc_msdf;
	bold_fc_msdf.instantiate();
	bold_fc_msdf->set_variation_customize_fallbacks(true); // Apply variations to each of the fallback individually, some languages do not use bold and italic..
	if (custom_font_path_bold.length() > 0 && dir->file_exists(custom_font_path_bold)) {
		Ref<Font> custom_font = load_external_font(custom_font_path_bold, font_hinting, font_antialiased, true, font_subpixel_positioning);
		{
			TypedArray<Font> fallback_custom;
			fallback_custom.push_back(default_font_bold_msdf);
			custom_font->set_fallbacks(fallback_custom);
		}
		bold_fc_msdf->set_font(custom_font);
		bold_fc_msdf->set("variation_fallback/0/11/variation_embolden", embolden_strength); // DroidSansFallback
		bold_fc_msdf->set("variation_fallback/0/12/variation_embolden", embolden_strength); // DroidSansJapanese
	} else if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		Ref<Font> custom_font = load_external_font(custom_font_path, font_hinting, font_antialiased, true, font_subpixel_positioning);
		{
			TypedArray<Font> fallback_custom;
			fallback_custom.push_back(default_font_bold_msdf);
			custom_font->set_fallbacks(fallback_custom);
		}
		bold_fc_msdf->set_font(custom_font);
		bold_fc_msdf->set_variation_embolden(embolden_strength);
		bold_fc_msdf->set("variation_fallback/0/11/variation_embolden", embolden_strength); // DroidSansFallback
		bold_fc_msdf->set("variation_fallback/0/12/variation_embolden", embolden_strength); // DroidSansJapanese
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font_bold", "");
		bold_fc_msdf->set_font(default_font_bold_msdf);
		bold_fc_msdf->set("variation_fallback/11/variation_embolden", embolden_strength); // DroidSansFallback
		bold_fc_msdf->set("variation_fallback/12/variation_embolden", embolden_strength); // DroidSansJapanese
	}
	bold_fc_msdf->set_spacing(TextServer::SPACING_TOP, -EDSCALE);
	bold_fc_msdf->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);

	Ref<FontConfig> mono_fc;
	mono_fc.instantiate();
	mono_fc->set_variation_customize_fallbacks(true); // Apply variations to each of the fallback individually, some languages do not use bold and italic..
	if (custom_font_path_source.length() > 0 && dir->file_exists(custom_font_path_source)) {
		Ref<Font> custom_font = load_external_font(custom_font_path_source, font_hinting, font_antialiased, true, font_subpixel_positioning);
		{
			TypedArray<Font> fallback_custom;
			fallback_custom.push_back(default_font_mono);
			custom_font->set_fallbacks(fallback_custom);
		}
		mono_fc->set_font(custom_font);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/code_font", "");
		mono_fc->set_font(default_font_mono);
	}
	String code_font_custom_variations = EditorSettings::get_singleton()->get("interface/editor/code_font_custom_variations");
	Dictionary variations_mono;
	if (!code_font_custom_variations.is_empty()) {
		Vector<String> variation_tags = code_font_custom_variations.split(",");
		for (int i = 0; i < variation_tags.size(); i++) {
			Vector<String> subtag_a = variation_tags[i].split("=");
			if (subtag_a.size() == 2) {
				variations_mono[TS->name_to_tag(subtag_a[0])] = subtag_a[1].to_float();
			} else if (subtag_a.size() == 1) {
				variations_mono[TS->name_to_tag(subtag_a[0])] = 1;
			}
		}
		mono_fc->set_variation_opentype(variations_mono);
	}
	mono_fc->set_spacing(TextServer::SPACING_TOP, -EDSCALE);
	mono_fc->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);

	Ref<FontConfig> italic_fc = default_fc->duplicate();
	italic_fc->set_variation_transform(Transform2D(1.0, 0.2, 0.0, 1.0, 0.0, 0.0));

	// Setup theme.

	p_theme->set_default_font(default_fc); // Default theme font config.
	p_theme->set_default_font_size(default_font_size);

	// Main font.

	p_theme->set_font("main", "EditorFonts", default_fc);
	p_theme->set_font("main_msdf", "EditorFonts", default_fc_msdf);
	p_theme->set_font_size("main_size", "EditorFonts", default_font_size);

	p_theme->set_font("bold", "EditorFonts", bold_fc);
	p_theme->set_font("main_bold_msdf", "EditorFonts", bold_fc_msdf);
	p_theme->set_font_size("bold_size", "EditorFonts", default_font_size);

	// Title font.

	p_theme->set_font("title", "EditorFonts", bold_fc);
	p_theme->set_font_size("title_size", "EditorFonts", default_font_size + 1 * EDSCALE);

	p_theme->set_font("main_button_font", "EditorFonts", bold_fc);
	p_theme->set_font_size("main_button_font_size", "EditorFonts", default_font_size + 1 * EDSCALE);

	p_theme->set_font("font", "Label", default_fc);

	p_theme->set_type_variation("HeaderSmall", "Label");
	p_theme->set_font("font", "HeaderSmall", bold_fc);
	p_theme->set_font_size("font_size", "HeaderSmall", default_font_size);

	p_theme->set_type_variation("HeaderMedium", "Label");
	p_theme->set_font("font", "HeaderMedium", bold_fc);
	p_theme->set_font_size("font_size", "HeaderMedium", default_font_size + 1 * EDSCALE);

	p_theme->set_type_variation("HeaderLarge", "Label");
	p_theme->set_font("font", "HeaderLarge", bold_fc);
	p_theme->set_font_size("font_size", "HeaderLarge", default_font_size + 3 * EDSCALE);

	// Documentation fonts
	p_theme->set_font_size("doc_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_font_size")) * EDSCALE);
	p_theme->set_font("doc", "EditorFonts", default_fc);
	p_theme->set_font("doc_bold", "EditorFonts", bold_fc);
	p_theme->set_font("doc_italic", "EditorFonts", italic_fc);
	p_theme->set_font_size("doc_title_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_title_font_size")) * EDSCALE);
	p_theme->set_font("doc_title", "EditorFonts", bold_fc);
	p_theme->set_font_size("doc_source_size", "EditorFonts", int(EDITOR_GET("text_editor/help/help_source_font_size")) * EDSCALE);
	p_theme->set_font("doc_source", "EditorFonts", mono_fc);
	p_theme->set_font_size("doc_keyboard_size", "EditorFonts", (int(EDITOR_GET("text_editor/help/help_source_font_size")) - 1) * EDSCALE);
	p_theme->set_font("doc_keyboard", "EditorFonts", mono_fc);

	// Ruler font
	p_theme->set_font_size("rulers_size", "EditorFonts", 8 * EDSCALE);
	p_theme->set_font("rulers", "EditorFonts", default_fc);

	// Rotation widget font
	p_theme->set_font_size("rotation_control_size", "EditorFonts", 14 * EDSCALE);
	p_theme->set_font("rotation_control", "EditorFonts", default_fc);

	// Code font
	p_theme->set_font_size("source_size", "EditorFonts", int(EDITOR_GET("interface/editor/code_font_size")) * EDSCALE);
	p_theme->set_font("source", "EditorFonts", mono_fc);

	p_theme->set_font_size("expression_size", "EditorFonts", (int(EDITOR_GET("interface/editor/code_font_size")) - 1) * EDSCALE);
	p_theme->set_font("expression", "EditorFonts", mono_fc);

	p_theme->set_font_size("output_source_size", "EditorFonts", int(EDITOR_GET("run/output/font_size")) * EDSCALE);
	p_theme->set_font("output_source", "EditorFonts", mono_fc);

	p_theme->set_font_size("status_source_size", "EditorFonts", default_font_size);
	p_theme->set_font("status_source", "EditorFonts", mono_fc);
}
