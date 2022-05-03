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
#include "core/crypto/crypto_core.h"
#include "core/io/dir_access.h"
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "editor/editor_paths.h"
#include "editor/editor_scale.h"
#include "editor/editor_settings.h"
#include "scene/resources/default_theme/default_theme.h"
#include "scene/resources/font.h"

EditorFonts *EditorFonts::singleton = nullptr;

EditorFonts::InternalFontData::InternalFontData(const uint8_t *p_ptr, size_t p_size) {
	ptr = p_ptr;
	size = p_size;

	CryptoCore::MD5Context ctx;
	ctx.start();
	ctx.update(p_ptr, p_size);
	unsigned char file_hash[16];
	ctx.finish(file_hash);

	hash = String::md5(file_hash);
}

EditorFonts::ExternalFontData::ExternalFontData(const String &p_path) {
	data = FileAccess::get_file_as_array(p_path);
	hash = FileAccess::get_md5(p_path);
}

Ref<FontData> EditorFonts::load_cached_font(const HashMap<String, Ref<FontData>> &p_old_cache, const String &p_id, TextServer::Hinting p_hinting, bool p_aa, bool p_autohint, TextServer::SubpixelPositioning p_font_subpixel_positioning, bool p_msdf, bool p_embolden, bool p_slanted) {
	double start_time = OS::get_singleton()->get_unix_time();

	uint16_t config = 0xf000;
	config |= p_hinting; // bits 0-1
	config |= p_autohint << 2; // bit 2
	config |= p_aa << 3; // bit 3
	if (!p_msdf) {
		config |= p_font_subpixel_positioning << 4; // bits 4-5
	}
	config |= p_msdf << 6; // bit 6
	config |= p_embolden << 7; // bit 7
	config |= p_slanted << 8; // bit 8

	String file_hash;
	if (internal_fonts.has(p_id)) {
		file_hash = internal_fonts[p_id].hash;
	} else if (external_fonts.has(p_id)) {
		file_hash = external_fonts[p_id].hash;
	} else {
		return Ref<FontData>();
	}

	String hash = file_hash + String::num_int64(config, 16);
	String info = vformat("EditorFont load id: %s, hash: %s", p_id, hash);

	// Already loaded from previous config.
	if (p_old_cache.has(hash)) {
		print_verbose(vformat("%s -> already loaded, font: %s, in %f s", info, p_old_cache[hash]->get_font_name(), OS::get_singleton()->get_unix_time() - start_time));
		cache[hash] = p_old_cache[hash];
		return p_old_cache[hash];
	}

	// Try loading cached version.
	String cache_prefix = EditorPaths::get_singleton()->get_config_dir().plus_file("editor_font_cache");
	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	const String fname = cache_prefix.plus_file(hash + ".fontdata");
	if (dir->file_exists(fname)) {
		Ref<FontData> font = ResourceLoader::load(fname);
		if (font.is_valid()) {
			print_verbose(vformat("%s -> cache loaded, font: %s, in %f s", info, font->get_font_name(), OS::get_singleton()->get_unix_time() - start_time));
			cache[hash] = font;
			return font;
		}
	}

	// Load new font data.
	Ref<FontData> font;
	font.instantiate();

	font->_set_editor_font_id(p_id);
	font->set_antialiased(p_aa);
	font->set_hinting(p_hinting);
	font->set_force_autohinter(p_autohint);
	font->set_subpixel_positioning(p_font_subpixel_positioning);
	font->set_multichannel_signed_distance_field(p_msdf);
	if (p_slanted) {
		font->set_transform(Transform2D(1.0, 0.3, 0.0, 1.0, 0.0, 0.0));
	}
	if (p_embolden) {
		font->set_embolden(embolden_strength);
	}
	print_verbose(vformat("%s -> new, font: %s, in %f s", info, font->get_font_name(), OS::get_singleton()->get_unix_time() - start_time));
	cache[hash] = font;
	return font;
}

void EditorFonts::stop_pre_render() {
	print_verbose("EditorFonts: stopping pre-render thread...");
	pre_render_exit.store(true);
	pre_render_thread.wait_to_finish();
	pre_render_rq.clear();
}

void EditorFonts::start_pre_render() {
	print_verbose("EditorFonts: starting pre-render thread...");
	pre_render_exit.store(false);
	pre_render_thread.start(&EditorFonts::_pre_render_func, nullptr);
}

void EditorFonts::_pre_render_func(void *) {
	double start_time = OS::get_singleton()->get_unix_time();
	for (PreRenderRequest &E : singleton->pre_render_rq) {
		if (E.data.is_valid()) {
			if (E.start != 0 && E.end != 0) {
				print_verbose(vformat("EditorFonts:[T] pre-rendering range: U+%x...U+%x, font: %s, size: %d", E.start, E.end, E.data->get_font_name(), E.size));
				for (char32_t c = E.start; c <= E.end; c++) {
					if (singleton->pre_render_exit.load()) {
						return;
					}
					E.data->render_range(0, Size2(E.size, 0), c, c);
					if (((c - E.start) % 100) == 0) {
						print_verbose(vformat("EditorFonts:[T] %d of %d", c - E.start + 1, E.end - E.start + 1));
					}
				}
			} else {
				print_verbose(vformat("EditorFonts:[T] pre-rendering all chars, font: %s, size: %d", E.data->get_font_name(), E.size));
				String chars = E.data->get_supported_chars();
				for (int i = 0; i < chars.size(); i++) {
					if (singleton->pre_render_exit.load()) {
						return;
					}
					E.data->render_range(0, Size2(E.size, 0), chars[i], chars[i]);
					if ((i % 100) == 0) {
						print_verbose(vformat("EditorFonts:[T] %d of %d", i + 1, chars.size()));
					}
				}
			}
		}
	}
	singleton->pre_render_rq.clear();
	print_verbose(vformat("EditorFonts:[T] pre-render thread done in: %f s", OS::get_singleton()->get_unix_time() - start_time));
}

EditorFonts *EditorFonts::get_singleton() {
	return singleton;
}

Ref<Font> EditorFonts::make_font(const Ref<FontData> &p_default, const Ref<FontData> &p_custom, const Vector<Ref<FontData>> &p_fallback, const String &p_variations) {
	Ref<Font> font;
	font.instantiate();
	if (p_custom.is_valid()) {
		font->add_data(p_custom);
	}
	font->add_data(p_default);
	for (int i = 0; i < p_fallback.size(); i++) {
		font->add_data(p_fallback[i]);
	}

	Dictionary variations;
	if (!p_variations.is_empty()) {
		Vector<String> variation_tags = p_variations.split(",");
		for (int i = 0; i < variation_tags.size(); i++) {
			Vector<String> tokens = variation_tags[i].split("=");
			if (tokens.size() == 2) {
				variations[tokens[0]] = tokens[1].to_float();
			}
		}
	}
	font->set_variation_coordinates(variations);

	font->set_spacing(TextServer::SPACING_TOP, -EDSCALE);
	font->set_spacing(TextServer::SPACING_BOTTOM, -EDSCALE);

	return font;
}

bool EditorFonts::has_external_editor_font_data(const String &p_id) const {
	return external_fonts.has(p_id);
}

PackedByteArray EditorFonts::get_external_editor_font_data(const String &p_id) const {
	if (external_fonts.has(p_id)) {
		return external_fonts[p_id].data;
	} else {
		return PackedByteArray();
	}
}

bool EditorFonts::has_internal_editor_font_data(const String &p_id) const {
	return internal_fonts.has(p_id);
}

const uint8_t *EditorFonts::get_internal_editor_font_data_ptr(const String &p_id) const {
	if (internal_fonts.has(p_id)) {
		return internal_fonts[p_id].ptr;
	} else {
		return nullptr;
	}
}

size_t EditorFonts::get_internal_editor_font_data_size(const String &p_id) const {
	if (internal_fonts.has(p_id)) {
		return internal_fonts[p_id].size;
	} else {
		return 0;
	}
}

void EditorFonts::load_fonts(Ref<Theme> &p_theme) {
	stop_pre_render();

	HashMap<String, Ref<FontData>> old_cache = cache; // Keep old cache loaded, to avoid reloading the same fonts.
	cache.clear();
	external_fonts.clear();

	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String cache_prefix = EditorPaths::get_singleton()->get_config_dir().plus_file("editor_font_cache");

	// Get font settings.

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

	// Load custom regular font files.
	String custom_font_path = EditorSettings::get_singleton()->get("interface/editor/main_font");
	if (custom_font_path.length() > 0 && dir->file_exists(custom_font_path)) {
		external_fonts["Custom"] = ExternalFontData(custom_font_path);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font", "");
	}

	String custom_font_path_bold = EditorSettings::get_singleton()->get("interface/editor/main_font_bold");
	if (custom_font_path_bold.length() > 0 && dir->file_exists(custom_font_path_bold)) {
		external_fonts["Custom_Bold"] = ExternalFontData(custom_font_path_bold);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/main_font_bold", "");
	}

	String custom_font_path_source = EditorSettings::get_singleton()->get("interface/editor/code_font");
	if (custom_font_path_source.length() > 0 && dir->file_exists(custom_font_path_source)) {
		external_fonts["Custom_Source"] = ExternalFontData(custom_font_path_bold);
	} else {
		EditorSettings::get_singleton()->set_manually("interface/editor/code_font", "");
	}

	// Load custom font data.

	Ref<FontData> custom_font = load_cached_font(old_cache, "Custom", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false);
	Ref<FontData> custom_font_msdf = load_cached_font(old_cache, "Custom", font_hinting, font_antialiased, true, font_subpixel_positioning, true, false, false);
	Ref<FontData> custom_font_slanted = load_cached_font(old_cache, "Custom", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, true);

	Ref<FontData> custom_font_bold = load_cached_font(old_cache, "Custom_Bold", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false);
	if (custom_font_bold.is_null()) {
		custom_font_bold = load_cached_font(old_cache, "Custom", font_hinting, font_antialiased, true, font_subpixel_positioning, false, true, false);
	}

	Ref<FontData> custom_font_bold_msdf = load_cached_font(old_cache, "Custom_Bold", font_hinting, font_antialiased, true, font_subpixel_positioning, true, false, false);
	if (custom_font_bold_msdf.is_null()) {
		custom_font_bold_msdf = load_cached_font(old_cache, "Custom", font_hinting, font_antialiased, true, font_subpixel_positioning, true, true, false);
	}

	Ref<FontData> custom_font_source = load_cached_font(old_cache, "Custom_Source", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false);

	// Load Noto Sans font data.
	Ref<FontData> default_font = load_cached_font(old_cache, "Default", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false);
	Ref<FontData> default_font_msdf = load_cached_font(old_cache, "Default", font_hinting, font_antialiased, true, font_subpixel_positioning, true, false, false);
	Ref<FontData> default_font_slanted = load_cached_font(old_cache, "Default", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, true);
	Ref<FontData> default_font_bold = load_cached_font(old_cache, "Default_Bold", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false);
	Ref<FontData> default_font_bold_msdf = load_cached_font(old_cache, "Default_Bold", font_hinting, font_antialiased, true, font_subpixel_positioning, true, false, false);

	Vector<Ref<FontData>> fallback;
	Vector<Ref<FontData>> fallback_bold;
	Vector<Ref<FontData>> fallback_msdf;
	Vector<Ref<FontData>> fallback_bold_msdf;

	for (const String &E : fallback_list) {
		fallback.push_back(load_cached_font(old_cache, E, font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false));
		fallback_bold.push_back(load_cached_font(old_cache, E + "_Bold", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false));
		fallback_msdf.push_back(load_cached_font(old_cache, E, font_hinting, font_antialiased, true, font_subpixel_positioning, true, false, false));
		fallback_bold_msdf.push_back(load_cached_font(old_cache, E + "_Bold", font_hinting, font_antialiased, true, font_subpixel_positioning, true, false, false));
	}

	// Load JetBrains Mono font data.
	Ref<FontData> source_font = load_cached_font(old_cache, "Source", font_hinting, font_antialiased, true, font_subpixel_positioning, false, false, false);
	Dictionary opentype_features;
	opentype_features["calt"] = 0;
	source_font->set_opentype_feature_overrides(opentype_features); // Disable contextual alternates (coding ligatures).

	// Pre-render fonts.
	pre_render_rq.push_back(PreRenderRequest(default_font_msdf, 0x20, 0x1FF)); // Pre-render Latin.
	pre_render_rq.push_back(PreRenderRequest(default_font_bold_msdf, 0x20, 0x1FF)); // Pre-render Latin.

	// Setup theme font configs.

	// Default font.
	Ref<Font> df = make_font(default_font, custom_font, fallback);
	p_theme->set_default_font(df); // Default theme font.
	p_theme->set_default_font_size(default_font_size);

	Ref<Font> df_msdf = make_font(default_font_msdf, custom_font_msdf, fallback_msdf);
	p_theme->set_font("main_msdf", "EditorFonts", df_msdf);

	p_theme->set_font_size("main_size", "EditorFonts", default_font_size);
	p_theme->set_font("main", "EditorFonts", df);

	// Bold font.
	Ref<Font> df_bold = make_font(default_font_bold, custom_font_bold, fallback_bold);
	Ref<Font> df_italic = make_font(default_font_slanted, custom_font_slanted, fallback_bold);
	p_theme->set_font_size("bold_size", "EditorFonts", default_font_size);
	p_theme->set_font("bold", "EditorFonts", df_bold);

	Ref<Font> df_bold_msdf = make_font(default_font_bold_msdf, custom_font_bold_msdf, fallback_bold_msdf);
	p_theme->set_font("main_bold_msdf", "EditorFonts", df_bold_msdf);

	// Title font.
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

	// Documentation fonts.
	String code_font_custom_variations = EditorSettings::get_singleton()->get("interface/editor/code_font_custom_variations");
	Ref<Font> df_code = make_font(source_font, custom_font_source, fallback, code_font_custom_variations);
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

	// Ruler font.
	p_theme->set_font_size("rulers_size", "EditorFonts", 8 * EDSCALE);
	p_theme->set_font("rulers", "EditorFonts", df);

	// Rotation widget font.
	p_theme->set_font_size("rotation_control_size", "EditorFonts", 14 * EDSCALE);
	p_theme->set_font("rotation_control", "EditorFonts", df);

	// Code font.
	p_theme->set_font_size("source_size", "EditorFonts", int(EDITOR_GET("interface/editor/code_font_size")) * EDSCALE);
	p_theme->set_font("source", "EditorFonts", df_code);

	p_theme->set_font_size("expression_size", "EditorFonts", (int(EDITOR_GET("interface/editor/code_font_size")) - 1) * EDSCALE);
	p_theme->set_font("expression", "EditorFonts", df_code);

	p_theme->set_font_size("output_source_size", "EditorFonts", int(EDITOR_GET("run/output/font_size")) * EDSCALE);
	p_theme->set_font("output_source", "EditorFonts", df_code);

	p_theme->set_font_size("status_source_size", "EditorFonts", default_font_size);
	p_theme->set_font("status_source", "EditorFonts", df_code);

	// Delete unused font cache files from old config.
	for (KeyValue<String, Ref<FontData>> &E : old_cache) {
		if (!cache.has(E.key)) {
			const String fname = cache_prefix.plus_file(E.key + ".fontdata");
			print_verbose(vformat("EditorFonts: deleting old font cache: %s, font: %s", fname, E.value->get_font_name()));
			dir->remove(fname);
		}
	}
	old_cache.clear();

	start_pre_render();
}

EditorFonts::EditorFonts() {
	singleton = this;

	// Main font.
	internal_fonts["Default"] = InternalFontData(_font_NotoSans_Regular, _font_NotoSans_Regular_size);
	internal_fonts["Default_Bold"] = InternalFontData(_font_NotoSans_Bold, _font_NotoSans_Bold_size);

	// Fallback fonts.
	fallback_list.push_back("Arabic");
	fallback_list.push_back("Bengali");
	fallback_list.push_back("Devanagari");
	fallback_list.push_back("Georgian");
	fallback_list.push_back("Hebrew");
	fallback_list.push_back("Malayalam");
	fallback_list.push_back("Oriya");
	fallback_list.push_back("Sinhala");
	fallback_list.push_back("Tamil");
	fallback_list.push_back("Telugu");
	fallback_list.push_back("Thai");
	fallback_list.push_back("CJK_Extra");
	fallback_list.push_back("CJK_Main");

	internal_fonts["Arabic"] = InternalFontData(_font_NotoNaskhArabicUI_Regular, _font_NotoNaskhArabicUI_Regular_size);
	internal_fonts["Bengali"] = InternalFontData(_font_NotoSansBengaliUI_Regular, _font_NotoSansBengaliUI_Regular_size);
	internal_fonts["Devanagari"] = InternalFontData(_font_NotoSansDevanagariUI_Regular, _font_NotoSansDevanagariUI_Regular_size);
	internal_fonts["Georgian"] = InternalFontData(_font_NotoSansGeorgian_Regular, _font_NotoSansGeorgian_Regular_size);
	internal_fonts["Hebrew"] = InternalFontData(_font_NotoSansHebrew_Regular, _font_NotoSansHebrew_Regular_size);
	internal_fonts["Malayalam"] = InternalFontData(_font_NotoSansMalayalamUI_Regular, _font_NotoSansMalayalamUI_Regular_size);
	internal_fonts["Oriya"] = InternalFontData(_font_NotoSansOriyaUI_Regular, _font_NotoSansOriyaUI_Regular_size);
	internal_fonts["Sinhala"] = InternalFontData(_font_NotoSansSinhalaUI_Regular, _font_NotoSansSinhalaUI_Regular_size);
	internal_fonts["Tamil"] = InternalFontData(_font_NotoSansTamilUI_Regular, _font_NotoSansTamilUI_Regular_size);
	internal_fonts["Telugu"] = InternalFontData(_font_NotoSansTeluguUI_Regular, _font_NotoSansTeluguUI_Regular_size);
	internal_fonts["Thai"] = InternalFontData(_font_NotoSansThaiUI_Regular, _font_NotoSansThaiUI_Regular_size);
	internal_fonts["CJK_Extra"] = InternalFontData(_font_DroidSansFallback, _font_DroidSansFallback_size);
	internal_fonts["CJK_Main"] = InternalFontData(_font_DroidSansJapanese, _font_DroidSansJapanese_size);

	internal_fonts["Arabic_Bold"] = InternalFontData(_font_NotoNaskhArabicUI_Bold, _font_NotoNaskhArabicUI_Bold_size);
	internal_fonts["Bengali_Bold"] = InternalFontData(_font_NotoSansBengaliUI_Bold, _font_NotoSansBengaliUI_Bold_size);
	internal_fonts["Devanagari_Bold"] = InternalFontData(_font_NotoSansDevanagariUI_Bold, _font_NotoSansDevanagariUI_Bold_size);
	internal_fonts["Georgian_Bold"] = InternalFontData(_font_NotoSansGeorgian_Bold, _font_NotoSansGeorgian_Bold_size);
	internal_fonts["Hebrew_Bold"] = InternalFontData(_font_NotoSansHebrew_Bold, _font_NotoSansHebrew_Bold_size);
	internal_fonts["Malayalam_Bold"] = InternalFontData(_font_NotoSansMalayalamUI_Bold, _font_NotoSansMalayalamUI_Bold_size);
	internal_fonts["Oriya_Bold"] = InternalFontData(_font_NotoSansOriyaUI_Bold, _font_NotoSansOriyaUI_Bold_size);
	internal_fonts["Sinhala_Bold"] = InternalFontData(_font_NotoSansSinhalaUI_Bold, _font_NotoSansSinhalaUI_Bold_size);
	internal_fonts["Tamil_Bold"] = InternalFontData(_font_NotoSansTamilUI_Bold, _font_NotoSansTamilUI_Bold_size);
	internal_fonts["Telugu_Bold"] = InternalFontData(_font_NotoSansTeluguUI_Bold, _font_NotoSansTeluguUI_Bold_size);
	internal_fonts["Thai_Bold"] = InternalFontData(_font_NotoSansThaiUI_Bold, _font_NotoSansThaiUI_Bold_size);
	internal_fonts["CJK_Extra_Bold"] = InternalFontData(_font_DroidSansFallback, _font_DroidSansFallback_size);
	internal_fonts["CJK_Main_Bold"] = InternalFontData(_font_DroidSansJapanese, _font_DroidSansJapanese_size);

	// Code editor font.
	internal_fonts["Source"] = InternalFontData(_font_JetBrainsMono_Regular, _font_JetBrainsMono_Regular_size);
}

EditorFonts::~EditorFonts() {
	stop_pre_render();

	Ref<DirAccess> dir = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
	String cache_prefix = EditorPaths::get_singleton()->get_config_dir().plus_file("editor_font_cache");
	dir->make_dir_recursive(cache_prefix); //ensure base dir exists

	int flg = ResourceSaver::SaverFlags::FLAG_BUNDLE_RESOURCES | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS | ResourceSaver::FLAG_COMPRESS;

	double start_time = OS::get_singleton()->get_unix_time();
	for (KeyValue<String, Ref<FontData>> &E : cache) {
		const String fname = cache_prefix.plus_file(E.key + ".fontdata");
		print_verbose(vformat("EditorFonts: saving font cache to: %s, font: %s.", E.key, E.value->get_font_name()));
		Error err = ResourceSaver::save(fname, E.value, flg);
		if (err != OK) {
			WARN_PRINT(vformat("Unable to save editor font cache to: %s, font: %s.", E.value->get_font_name(), fname));
		}
	}
	print_verbose(vformat("EditorFonts: cache saved in: %f s", OS::get_singleton()->get_unix_time() - start_time));

	singleton = nullptr;
}
