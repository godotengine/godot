/*************************************************************************/
/*  resource_importer_fontdata.cpp                                       */
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

#include "resource_importer_fontdata.h"

#include "core/os/file_access.h"
#include "editor/editor_node.h"

String ResourceImporterFontData::get_importer_name() const {
	return "font_data";
}

String ResourceImporterFontData::get_visible_name() const {
	return "Font Data";
}

void ResourceImporterFontData::get_recognized_extensions(List<String> *p_extensions) const {
	TS->font_get_recognized_extensions(p_extensions);
}

String ResourceImporterFontData::get_save_extension() const {
	return "fontdata";
}

String ResourceImporterFontData::get_resource_type() const {
	return "FontData";
}

bool ResourceImporterFontData::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {
	if (p_option == "msdf_px_range" && !bool(p_options["msdf"])) {
		return false;
	}
	if (p_option == "oversampling" && bool(p_options["msdf"])) {
		return false;
	}
	if (p_option == "antialiased" && bool(p_options["msdf"])) {
		return false;
	}
	if (p_option == "preload/variations" && p_options["convert_to_bitmap"]) {
		return false;
	}
	return true;
}

int ResourceImporterFontData::get_preset_count() const {
	return PRESET_MAX;
}

String ResourceImporterFontData::get_preset_name(int p_idx) const {
	switch (p_idx) {
		case PRESET_MSDF:
			return TTR("M(T)SDF font");
		case PRESET_DYNAMIC:
			return TTR("Dynamic font");
		default:
			return String();
	}
}

void ResourceImporterFontData::get_import_options(List<ImportOption> *r_options, int p_preset) const {
	bool msdf = p_preset == PRESET_MSDF;

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "antialiased"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "force_autohinter"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "msdf", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), (msdf) ? true : false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "oversampling", PROPERTY_HINT_RANGE, "0,100,0.1"), 0.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "msdf_px_range", PROPERTY_HINT_RANGE, "1,100,0.1"), 8.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "hinting", PROPERTY_HINT_ENUM, "None,Light,Normal"), 1));

	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "base_variation"), (msdf) ? "size=64" : "size=16"));

	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "preload/ranges"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "preload/variations"), Vector<String>()));

	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/language_enabled"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/language_disabled"), Vector<String>()));

	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/script_enabled"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "support_overrides/script_disabled"), Vector<String>()));

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "convert_to_bitmap", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "extra_spacing_glyph"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "extra_spacing_space"), 0));
}

bool ResourceImporterFontData::_decode_variation(const String &p_token, Map<int32_t, double> *r_variation, int *r_size, int *r_outline_size) const {
	Vector<String> tokens = p_token.split("=");
	if (tokens.size() == 2) {
		if (tokens[0] == "size") {
			*r_size = tokens[1].to_int();
		} else if (tokens[0] == "outline_size") {
			*r_outline_size = tokens[1].to_int();
		} else {
			uint32_t tag = TS->name_to_tag(tokens[0]);
			double value = tokens[1].to_float();
			r_variation->insert(tag, value);
		}
		return true;
	} else {
		WARN_PRINT("Invalid variation: '" + p_token + "'.");
		return false;
	}
}

bool ResourceImporterFontData::_decode_range(const String &p_token, bool *r_gl_index, int32_t *r_pos) const {
	if (p_token.begins_with("G+") || p_token.begins_with("g+")) {
		// Glyph hex index.
		if (r_gl_index) {
			*r_gl_index = true;
		}
		if (r_pos) {
			*r_pos = p_token.substr(2).hex_to_int();
		}
		return true;
	} else if (p_token.begins_with("U+") || p_token.begins_with("u+") || p_token.begins_with("0x")) {
		// Unicode character hex index.
		if (r_gl_index) {
			*r_gl_index = false;
		}
		if (r_pos) {
			*r_pos = p_token.substr(2).hex_to_int();
		}
		return true;
	} else if (p_token.length() == 3 && p_token[0] == '\'' && p_token[2] == '\'') {
		// Unicode character.
		if (r_gl_index) {
			*r_gl_index = false;
		}
		if (r_pos) {
			*r_pos = p_token.unicode_at(1);
		}
		return true;
	} else if (p_token.is_numeric()) {
		// Unicode character decimal index.
		if (r_gl_index) {
			*r_gl_index = false;
		}
		if (r_pos) {
			*r_pos = p_token.to_int();
		}
		return true;
	} else {
		return false;
	}
}

Error ResourceImporterFontData::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	bool antialiased = p_options["antialiased"];
	bool msdf = p_options["msdf"];

	bool autohinter = p_options["force_autohinter"];
	int hinting = p_options["hinting"];

	bool convert_to_bmp = p_options["convert_to_bitmap"];

	int extra_spacing_glyph = p_options["extra_spacing_glyph"];
	int extra_spacing_space = p_options["extra_spacing_space"];

	double px_range = p_options["msdf_px_range"];
	double oversampling = p_options["oversampling"];

	EditorProgress progress("import", TTR("Import Font"), 104);
	progress.step(TTR("Importing Font..."), 0);

	int base_size = 16;
	Vector<String> base_variation_tags = String(p_options["base_variation"]).split(",");
	for (int i = 0; i < base_variation_tags.size(); i++) {
		Vector<String> tokens = base_variation_tags[i].split("=");
		if (tokens.size() == 2) {
			if (tokens[0] == "size") {
				base_size = tokens[1].to_int();
			}
		}
	}

	// Load base font data.
	Ref<FontData> data;
	data.instantiate();
	data->load_resource(p_source_file, base_size);
	data->set_antialiased(antialiased);
	data->set_force_autohinter(autohinter);
	data->set_hinting((TextServer::Hinting)hinting);
	data->set_oversampling(oversampling);
	data->set_distance_field_hint(msdf);
	data->set_spacing(FontData::SPACING_GLYPH, extra_spacing_glyph);
	data->set_spacing(FontData::SPACING_SPACE, extra_spacing_space);
	data->set_msdf_px_range(px_range);
	for (int i = 0; i < base_variation_tags.size(); i++) {
		Vector<String> tokens = base_variation_tags[i].split("=");
		if (tokens.size() == 2) {
			if (tokens[0] != "size") {
				data->set_variation(tokens[0], tokens[1].to_float());
			}
		}
	}

	Vector<String> lang_en = p_options["support_overrides/language_enabled"];
	for (int i = 0; i < lang_en.size(); i++) {
		data->set_language_support_override(lang_en[i], true);
	}

	Vector<String> lang_dis = p_options["support_overrides/language_disabled"];
	for (int i = 0; i < lang_dis.size(); i++) {
		data->set_language_support_override(lang_dis[i], false);
	}

	Vector<String> scr_en = p_options["support_overrides/script_enabled"];
	for (int i = 0; i < scr_en.size(); i++) {
		data->set_script_support_override(scr_en[i], true);
	}

	Vector<String> scr_dis = p_options["support_overrides/script_disabled"];
	for (int i = 0; i < scr_dis.size(); i++) {
		data->set_script_support_override(scr_dis[i], false);
	}

	Vector<String> variations = p_options["preload/variations"];
	for (int i = 0; i < variations.size(); i++) {
		Map<int32_t, double> variation;
		int size, outline_size;
		Vector<String> variation_tags = variations[i].split(",");
		for (int j = 0; j < variation_tags.size(); j++) {
			if (!_decode_variation(variation_tags[j], &variation, &size, &outline_size)) {
				WARN_PRINT("Invalid variation: \"" + variations[i] + "\"");
				continue;
			}
		}
		// Preload variations / sizes.
		data->add_to_cache(variation, size, outline_size);
	}

	Vector<String> ranges = p_options["preload/ranges"];
	for (int i = 0; i < ranges.size(); i++) {
		int32_t start, end;
		bool gl_start, gl_end;
		Vector<String> tokens = ranges[i].split("-");
		if (tokens.size() == 2) {
			if (!_decode_range(tokens[0], &gl_start, &start) || !_decode_range(tokens[1], &gl_end, &end) || gl_start != gl_end) {
				WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
				continue;
			}
		} else if (tokens.size() == 1) {
			if (!_decode_range(tokens[0], &gl_start, &start)) {
				WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
				continue;
			}
			end = start;
		} else {
			WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
			continue;
		}
		// Preload char/glyph ranges for each variations / sizes.
		for (int32_t c = MIN(start, end); c <= MAX(start, end); c++) {
			progress.step(TTR("Pre-rendering") + " " + (gl_start ? TTR("glyph") : TTR("character")) + " U+" + String::num_int64(c, 16, true) + " " + TTR("from range") + " '" + ranges[i] + "' (" + itos(i + 1) + " " + TTR("of") + " " + itos(ranges.size()) + ")...", i * 100 / ranges.size());
			data->preload_range(c, c, gl_start);
		}
	}

	uint8_t flags = TextServer::FONT_CACHE_FLAGS_DEFAULT;
	if (convert_to_bmp) {
		flags |= TextServer::FONT_CACHE_FLAGS_CONVERT_TO_BITMAP;
	}

	progress.step(TTR("Saving..."), 104);

	return data->save_cache(p_save_path, flags, r_gen_files);
}

ResourceImporterFontData::ResourceImporterFontData() {
}
