/*************************************************************************/
/*  resource_importer_dynamicfont.cpp                                    */
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

#include "resource_importer_dynamicfont.h"

#include "core/io/file_access.h"
#include "core/io/resource_saver.h"
#include "dynamicfont_import_settings.h"
#include "editor/editor_node.h"

#include "modules/modules_enabled.gen.h" // For freetype.

String ResourceImporterDynamicFont::get_importer_name() const {
	return "font_data_dynamic";
}

String ResourceImporterDynamicFont::get_visible_name() const {
	return "Font Data (Dynamic Font)";
}

void ResourceImporterDynamicFont::get_recognized_extensions(List<String> *p_extensions) const {
	if (p_extensions) {
#ifdef MODULE_FREETYPE_ENABLED
		p_extensions->push_back("ttf");
		p_extensions->push_back("otf");
		p_extensions->push_back("woff");
		//p_extensions->push_back("woff2");
		p_extensions->push_back("pfb");
		p_extensions->push_back("pfm");
#endif
	}
}

String ResourceImporterDynamicFont::get_save_extension() const {
	return "fontdata";
}

String ResourceImporterDynamicFont::get_resource_type() const {
	return "FontData";
}

bool ResourceImporterDynamicFont::get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) const {
	if (p_option == "msdf_pixel_range" && !bool(p_options["multichannel_signed_distance_field"])) {
		return false;
	}
	if (p_option == "msdf_size" && !bool(p_options["multichannel_signed_distance_field"])) {
		return false;
	}
	if (p_option == "oversampling" && bool(p_options["multichannel_signed_distance_field"])) {
		return false;
	}
	return true;
}

int ResourceImporterDynamicFont::get_preset_count() const {
	return PRESET_MAX;
}

String ResourceImporterDynamicFont::get_preset_name(int p_idx) const {
	switch (p_idx) {
		case PRESET_DYNAMIC:
			return TTR("Dynamically rendered TrueType/OpenType font");
		case PRESET_MSDF:
			return TTR("Prerendered multichannel(+true) signed distance field");
		default:
			return String();
	}
}

void ResourceImporterDynamicFont::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	bool msdf = p_preset == PRESET_MSDF;

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "antialiased"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "multichannel_signed_distance_field", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), (msdf) ? true : false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "msdf_pixel_range", PROPERTY_HINT_RANGE, "1,100,1"), 8));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "msdf_size", PROPERTY_HINT_RANGE, "1,250,1"), 48));

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force_autohinter"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "hinting", PROPERTY_HINT_ENUM, "None,Light,Normal"), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "oversampling", PROPERTY_HINT_RANGE, "0,10,0.1"), 0.0));

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::DICTIONARY, "opentype_feature_overrides"), Dictionary()));

	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "preload/char_ranges"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "preload/glyph_ranges"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "preload/configurations"), Vector<String>()));

	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/language_enabled"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/language_disabled"), Vector<String>()));

	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/script_enabled"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/script_disabled"), Vector<String>()));
}

bool ResourceImporterDynamicFont::_decode_variation(const String &p_token, Dictionary &r_variations, Vector2i &r_size, String &r_name, Vector2i &r_spacing) {
	Vector<String> tokens = p_token.split("=");
	if (tokens.size() == 2) {
		if (tokens[0] == "name") {
			r_name = tokens[1];
		} else if (tokens[0] == "size") {
			r_size.x = tokens[1].to_int();
		} else if (tokens[0] == "outline_size") {
			r_size.y = tokens[1].to_int();
		} else if (tokens[0] == "spacing_space") {
			r_spacing.x = tokens[1].to_int();
		} else if (tokens[0] == "spacing_glyph") {
			r_spacing.y = tokens[1].to_int();
		} else {
			r_variations[tokens[0]] = tokens[1].to_float();
		}
		return true;
	} else {
		WARN_PRINT("Invalid variation: '" + p_token + "'.");
		return false;
	}
}

bool ResourceImporterDynamicFont::_decode_range(const String &p_token, int32_t &r_pos) {
	if (p_token.begins_with("U+") || p_token.begins_with("u+") || p_token.begins_with("0x")) {
		// Unicode character hex index.
		r_pos = p_token.substr(2).hex_to_int();
		return true;
	} else if (p_token.length() == 3 && p_token[0] == '\'' && p_token[2] == '\'') {
		// Unicode character.
		r_pos = p_token.unicode_at(1);
		return true;
	} else if (p_token.is_numeric()) {
		// Unicode character decimal index.
		r_pos = p_token.to_int();
		return true;
	} else {
		return false;
	}
}

bool ResourceImporterDynamicFont::has_advanced_options() const {
	return true;
}
void ResourceImporterDynamicFont::show_advanced_options(const String &p_path) {
	DynamicFontImportSettings::get_singleton()->open_settings(p_path);
}

Error ResourceImporterDynamicFont::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	print_verbose("Importing dynamic font from: " + p_source_file);

	bool antialiased = p_options["antialiased"];
	bool msdf = p_options["multichannel_signed_distance_field"];
	int px_range = p_options["msdf_pixel_range"];
	int px_size = p_options["msdf_size"];
	Dictionary ot_ov = p_options["opentype_feature_overrides"];

	bool autohinter = p_options["force_autohinter"];
	int hinting = p_options["hinting"];
	real_t oversampling = p_options["oversampling"];

	// Load base font data.
	Vector<uint8_t> data = FileAccess::get_file_as_array(p_source_file);

	// Create font.
	Ref<FontData> font;
	font.instantiate();
	font->set_data(data);
	font->set_antialiased(antialiased);
	font->set_multichannel_signed_distance_field(msdf);
	font->set_msdf_pixel_range(px_range);
	font->set_msdf_size(px_size);
	font->set_opentype_feature_overrides(ot_ov);
	font->set_fixed_size(0);
	font->set_force_autohinter(autohinter);
	font->set_hinting((TextServer::Hinting)hinting);
	font->set_oversampling(oversampling);

	Vector<String> lang_en = p_options["support_overrides/language_enabled"];
	for (int i = 0; i < lang_en.size(); i++) {
		font->set_language_support_override(lang_en[i], true);
	}

	Vector<String> lang_dis = p_options["support_overrides/language_disabled"];
	for (int i = 0; i < lang_dis.size(); i++) {
		font->set_language_support_override(lang_dis[i], false);
	}

	Vector<String> scr_en = p_options["support_overrides/script_enabled"];
	for (int i = 0; i < scr_en.size(); i++) {
		font->set_script_support_override(scr_en[i], true);
	}

	Vector<String> scr_dis = p_options["support_overrides/script_disabled"];
	for (int i = 0; i < scr_dis.size(); i++) {
		font->set_script_support_override(scr_dis[i], false);
	}

	Vector<String> variations = p_options["preload/configurations"];
	Vector<String> char_ranges = p_options["preload/char_ranges"];
	Vector<String> gl_ranges = p_options["preload/glyph_ranges"];

	for (int i = 0; i < variations.size(); i++) {
		String name;
		Dictionary var;
		Vector2i size = Vector2(16, 0);
		Vector2i spacing;

		Vector<String> variation_tags = variations[i].split(",");
		for (int j = 0; j < variation_tags.size(); j++) {
			if (!_decode_variation(variation_tags[j], var, size, name, spacing)) {
				WARN_PRINT(vformat(TTR("Invalid variation: \"%s\""), variations[i]));
				continue;
			}
		}
		RID conf = font->find_cache(var);

		for (int j = 0; j < char_ranges.size(); j++) {
			int32_t start, end;
			Vector<String> tokens = char_ranges[j].split("-");
			if (tokens.size() == 2) {
				if (!_decode_range(tokens[0], start) || !_decode_range(tokens[1], end)) {
					WARN_PRINT(vformat(TTR("Invalid range: \"%s\""), char_ranges[j]));
					continue;
				}
			} else if (tokens.size() == 1) {
				if (!_decode_range(tokens[0], start)) {
					WARN_PRINT(vformat(TTR("Invalid range: \"%s\""), char_ranges[j]));
					continue;
				}
				end = start;
			} else {
				WARN_PRINT(vformat(TTR("Invalid range: \"%s\""), char_ranges[j]));
				continue;
			}

			// Preload character ranges for each variations / sizes.
			print_verbose(vformat(TTR("Pre-rendering range U+%s...%s from configuration \"%s\" (%d / %d)..."), String::num_int64(start, 16), String::num_int64(end, 16), name, i + 1, variations.size()));
			TS->font_render_range(conf, size, start, end);
		}

		for (int j = 0; j < gl_ranges.size(); j++) {
			int32_t start, end;
			Vector<String> tokens = gl_ranges[j].split("-");
			if (tokens.size() == 2) {
				if (!_decode_range(tokens[0], start) || !_decode_range(tokens[1], end)) {
					WARN_PRINT(vformat(TTR("Invalid range: \"%s\""), gl_ranges[j]));
					continue;
				}
			} else if (tokens.size() == 1) {
				if (!_decode_range(tokens[0], start)) {
					WARN_PRINT(vformat(TTR("Invalid range: \"%s\""), gl_ranges[j]));
					continue;
				}
				end = start;
			} else {
				WARN_PRINT(vformat(TTR("Invalid range: \"%s\""), gl_ranges[j]));
				continue;
			}

			// Preload glyph range for each variations / sizes.
			print_verbose(vformat(TTR("Pre-rendering glyph range 0x%s...%s from configuration \"%s\" (%d / %d)..."), String::num_int64(start, 16), String::num_int64(end, 16), name, i + 1, variations.size()));
			for (int32_t k = start; k <= end; k++) {
				TS->font_render_glyph(conf, size, k);
			}
		}

		TS->font_set_spacing(conf, size.x, TextServer::SPACING_SPACE, spacing.x);
		TS->font_set_spacing(conf, size.x, TextServer::SPACING_GLYPH, spacing.y);
	}

	int flg = ResourceSaver::SaverFlags::FLAG_BUNDLE_RESOURCES | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;
	if ((bool)p_options["compress"]) {
		flg |= ResourceSaver::SaverFlags::FLAG_COMPRESS;
	}

	print_verbose("Saving to: " + p_save_path + ".fontdata");
	Error err = ResourceSaver::save(p_save_path + ".fontdata", font, flg);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save font to file \"" + p_save_path + ".res\".");
	print_verbose("Done saving to: " + p_save_path + ".fontdata");
	return OK;
}

ResourceImporterDynamicFont::ResourceImporterDynamicFont() {
}
