/**************************************************************************/
/*  resource_importer_lottie.cpp                                          */
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

#include "resource_importer_lottie.h"
#include "core/io/dir_access.h"
#include "core/io/json.h"
#include "modules/svg/lottie_texture.h"

String ResourceImporterLottie::get_importer_name() const {
	return "lottie_compressed_texture_2d";
}
String ResourceImporterLottie::get_visible_name() const {
	return "Lottie CompressedTexture2D";
}
int ResourceImporterLottie::get_preset_count() const {
	return 1;
}
String ResourceImporterLottie::get_preset_name(int p_idx) const {
	return ResourceImporterTexture::get_preset_name(PRESET_2D);
}
void ResourceImporterLottie::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	Ref<JSON> json;
	Dictionary dict;
	if (!p_path.is_empty()) {
		json.instantiate();
		json->parse(FileAccess::get_file_as_string(p_path), true);
		dict = json->get_data();
	}
	const float p_scale = dict.get("gd_scale", 1.0);
	const float p_frame_begin = dict.get("gd_frame_begin", 0);
	const float p_frame_end = dict.get("gd_frame_end", 0);
	const int p_frame_count = dict.get("gd_frame_count", 0);
	const int p_rows = dict.get("gd_rows", -1);

	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "scale"), p_scale));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "frame_begin"), p_frame_begin));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "frame_end"), p_frame_end));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "frame_count"), p_frame_count));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "rows"), p_rows));

	ResourceImporterTexture::get_import_options(p_path, r_options, p_preset);
}
bool ResourceImporterLottie::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return ResourceImporterTexture::get_option_visibility(p_path, p_option, p_options);
}
void ResourceImporterLottie::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("lottiectex");
}
Error ResourceImporterLottie::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(FileAccess::get_file_as_string(p_source_file), true);
	if (err != OK) {
		String err_text = "Error parsing JSON file at '" + p_source_file + "', on line " + itos(json->get_error_line()) + ": " + json->get_error_message();
		ERR_PRINT(err_text);
		return ERR_INVALID_DATA;
	}
	const float p_scale = p_options["scale"];
	const float p_frame_begin = p_options["frame_begin"];
	const float p_frame_end = p_options["frame_end"];
	const int p_frame_count = p_options["frame_count"];
	const int p_rows = p_options["rows"];
	Ref<LottieTexture2D> lottie;
	lottie.instantiate();
	lottie->update(json, p_frame_begin, p_frame_end, p_frame_count, p_scale, p_rows);

	String tmp_image = p_save_path + ".tmp.png";
	err = lottie->get_image()->save_png(tmp_image);
	if (err == OK) {
		err = ResourceImporterTexture::import(tmp_image, p_save_path, p_options, r_platform_variants, r_gen_files, r_metadata);
		DirAccess::remove_file_or_error(tmp_image.trim_prefix("res://"));
	}
	return err;
}
