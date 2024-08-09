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

String ResourceImporterLottieJSON::get_importer_name() const {
	return "lottie_texture_2d";
}
String ResourceImporterLottieJSON::get_visible_name() const {
	return "LottieTexture2D";
}
int ResourceImporterLottieJSON::get_preset_count() const {
	return 0;
}
bool ResourceImporterLottieJSON::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}
void ResourceImporterLottieJSON::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("json");
}
String ResourceImporterLottieJSON::get_save_extension() const {
	return "lottiejson";
}
String ResourceImporterLottieJSON::get_resource_type() const {
	return "LottieTexture2D";
}
void ResourceImporterLottieJSON::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	String str = FileAccess::get_file_as_string(p_path);
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(str, true);
	if (err != OK) {
		ERR_PRINT("Error parsing JSON file: " + p_path);
		return;
	}
	if (!LottieTexture2D::validate_json(json)) {
		ERR_PRINT("Invalid Lottie file: " + p_path);
		return;
	}
	Ref<LottieTexture2D> lottie = memnew(LottieTexture2D);
	lottie->update(json, 0, 0, 0, 1, 0);
	int total_frame_count = lottie->get_lottie_frame_count();
	int fps = total_frame_count / lottie->get_lottie_duration();
	int default_frame_count = fps <= 30 ? total_frame_count : total_frame_count * 30 / fps;
	int columns = Math::ceil(Math::sqrt((float)default_frame_count));
	int rows = Math::ceil(((float)default_frame_count) / columns);
	Vector2 texture_size = lottie->get_lottie_image_size() * Vector2(columns, rows);
	int default_size_limit = 1024;
	float default_scale = 1.0;
	if (texture_size[texture_size.max_axis_index()] > default_size_limit) {
		default_scale = default_size_limit / texture_size[texture_size.max_axis_index()];
	}

	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "lottie/scale"), default_scale));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "lottie/frame_begin"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "lottie/frame_end"), total_frame_count - 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "lottie/frame_count"), default_frame_count));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "lottie/columns"), 0));
}
Error ResourceImporterLottieJSON::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	// Invalid Lottie file if no import options
	if (p_options.is_empty()) {
		return ERR_INVALID_DATA;
	}
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(FileAccess::get_file_as_string(p_source_file), true);
	if (err != OK) {
		String err_text = "Error parsing JSON file at '" + p_source_file + "', on line " + itos(json->get_error_line()) + ": " + json->get_error_message();
		ERR_PRINT(err_text);
		return ERR_INVALID_DATA;
	}
	const float scale = p_options["lottie/scale"];
	const float frame_begin = p_options["lottie/frame_begin"];
	const float frame_end = p_options["lottie/frame_end"];
	const int frame_count = p_options["lottie/frame_count"];
	const int columns = p_options["lottie/columns"];
	Ref<LottieTexture2D> lottie;
	lottie.instantiate();
	lottie->update(json, frame_begin, frame_end, frame_count, scale, columns);
	err = ResourceSaver::save(lottie, p_save_path + ".lottiejson");
	return err;
}

ResourceImporterLottieJSON::ResourceImporterLottieJSON() {}

ResourceImporterLottieJSON::~ResourceImporterLottieJSON() {}

//////////////////////////////////////////////////////////////

String ResourceImporterLottieCTEX::get_importer_name() const {
	return "lottie_compressed_texture_2d";
}
String ResourceImporterLottieCTEX::get_visible_name() const {
	return "CompressedTexture2D";
}
int ResourceImporterLottieCTEX::get_preset_count() const {
	return 1;
}
String ResourceImporterLottieCTEX::get_preset_name(int p_idx) const {
	return p_idx == 0 ? ResourceImporterTexture::get_preset_name(PRESET_2D) : "";
}
void ResourceImporterLottieCTEX::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	lottie_json_importer->get_import_options(p_path, r_options, p_preset);
	if (r_options->is_empty()) {
		return;
	}
	ResourceImporterTexture::get_import_options(p_path, r_options, p_preset);
}
bool ResourceImporterLottieCTEX::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return ResourceImporterTexture::get_option_visibility(p_path, p_option, p_options);
}
void ResourceImporterLottieCTEX::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("json");
}
Error ResourceImporterLottieCTEX::import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	// Invalid Lottie file if no import options
	if (p_options.is_empty()) {
		return ERR_INVALID_DATA;
	}
	Ref<JSON> json;
	json.instantiate();
	Error err = json->parse(FileAccess::get_file_as_string(p_source_file), true);
	if (err != OK) {
		String err_text = "Error parsing JSON file at '" + p_source_file + "', on line " + itos(json->get_error_line()) + ": " + json->get_error_message();
		ERR_PRINT(err_text);
		return ERR_INVALID_DATA;
	}
	const float scale = p_options["lottie/scale"];
	const float frame_begin = p_options["lottie/frame_begin"];
	const float frame_end = p_options["lottie/frame_end"];
	const int frame_count = p_options["lottie/frame_count"];
	const int columns = p_options["lottie/columns"];
	Ref<LottieTexture2D> lottie;
	lottie.instantiate();
	lottie->update(json, frame_begin, frame_end, frame_count, scale, columns);

	String tmp_image = p_save_path + ".tmp.png";
	err = lottie->get_image()->save_png(tmp_image);
	if (err == OK) {
		err = ResourceImporterTexture::import(tmp_image, p_save_path, p_options, r_platform_variants, r_gen_files, r_metadata);
		DirAccess::remove_file_or_error(tmp_image.trim_prefix("res://"));
	}
	return err;
}
