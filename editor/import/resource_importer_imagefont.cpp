/*************************************************************************/
/*  resource_importer_imagefont.cpp                                      */
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

#include "resource_importer_imagefont.h"

#include "core/io/image_loader.h"
#include "core/io/resource_saver.h"

String ResourceImporterImageFont::get_importer_name() const {
	return "font_data_image";
}

String ResourceImporterImageFont::get_visible_name() const {
	return "Font Data (Monospace Image Font)";
}

void ResourceImporterImageFont::get_recognized_extensions(List<String> *p_extensions) const {
	if (p_extensions) {
		ImageLoader::get_recognized_extensions(p_extensions);
	}
}

String ResourceImporterImageFont::get_save_extension() const {
	return "fontdata";
}

String ResourceImporterImageFont::get_resource_type() const {
	return "FontData";
}

bool ResourceImporterImageFont::get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) const {
	return true;
}

void ResourceImporterImageFont::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::PACKED_STRING_ARRAY, "character_ranges"), Vector<String>()));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "columns"), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "rows"), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "font_size"), 14));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress"), true));
}

bool ResourceImporterImageFont::_decode_range(const String &p_token, int32_t &r_pos) {
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

Error ResourceImporterImageFont::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	print_verbose("Importing image font from: " + p_source_file);

	int columns = p_options["columns"];
	int rows = p_options["rows"];
	int base_size = p_options["font_size"];
	Vector<String> ranges = p_options["character_ranges"];

	Ref<FontData> font;
	font.instantiate();
	font->set_antialiased(false);
	font->set_multichannel_signed_distance_field(false);
	font->set_fixed_size(base_size);
	font->set_force_autohinter(false);
	font->set_hinting(TextServer::HINTING_NONE);
	font->set_oversampling(1.0f);

	Ref<Image> img;
	img.instantiate();
	Error err = ImageLoader::load_image(p_source_file, img);
	ERR_FAIL_COND_V_MSG(err != OK, ERR_FILE_CANT_READ, TTR("Can't load font texture: ") + "\"" + p_source_file + "\".");
	font->set_texture_image(0, Vector2i(base_size, 0), 0, img);

	int count = columns * rows;
	int chr_width = img->get_width() / columns;
	int chr_height = img->get_height() / rows;
	int pos = 0;

	for (int i = 0; i < ranges.size(); i++) {
		int32_t start, end;
		Vector<String> tokens = ranges[i].split("-");
		if (tokens.size() == 2) {
			if (!_decode_range(tokens[0], start) || !_decode_range(tokens[1], end)) {
				WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
				continue;
			}
		} else if (tokens.size() == 1) {
			if (!_decode_range(tokens[0], start)) {
				WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
				continue;
			}
			end = start;
		} else {
			WARN_PRINT("Invalid range: \"" + ranges[i] + "\"");
			continue;
		}
		for (int32_t idx = start; idx <= end; idx++) {
			int x = pos % columns;
			int y = pos / columns;
			font->set_glyph_advance(0, base_size, idx, Vector2(chr_width, 0));
			font->set_glyph_offset(0, Vector2i(base_size, 0), idx, Vector2(0, -0.5 * chr_height));
			font->set_glyph_size(0, Vector2i(base_size, 0), idx, Vector2(chr_width, chr_height));
			font->set_glyph_uv_rect(0, Vector2i(base_size, 0), idx, Rect2(chr_width * x, chr_height * y, chr_width, chr_height));
			font->set_glyph_texture_idx(0, Vector2i(base_size, 0), idx, 0);
			pos++;
			ERR_FAIL_COND_V_MSG(pos >= count, ERR_CANT_CREATE, "Too many characters in range.");
		}
	}
	font->set_ascent(0, base_size, 0.5 * chr_height);
	font->set_descent(0, base_size, 0.5 * chr_height);

	int flg = ResourceSaver::SaverFlags::FLAG_BUNDLE_RESOURCES | ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;
	if ((bool)p_options["compress"]) {
		flg |= ResourceSaver::SaverFlags::FLAG_COMPRESS;
	}

	print_verbose("Saving to: " + p_save_path + ".fontdata");
	err = ResourceSaver::save(p_save_path + ".fontdata", font, flg);
	ERR_FAIL_COND_V_MSG(err != OK, err, "Cannot save font to file \"" + p_save_path + ".res\".");
	print_verbose("Done saving to: " + p_save_path + ".fontdata");
	return OK;
}

ResourceImporterImageFont::ResourceImporterImageFont() {
}
