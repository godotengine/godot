/**************************************************************************/
/*  resource_importer_dds.cpp                                             */
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

#include "resource_importer_dds.h"

#include "../texture_loader_dds.h"

#include "scene/resources/streamed_texture.h"

String ResourceImporterDDS::get_importer_name() const {
	return "dds_texture";
}

String ResourceImporterDDS::get_visible_name() const {
	return "DDS Texture";
}

void ResourceImporterDDS::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("dds");
}

String ResourceImporterDDS::get_save_extension() const {
	// A redirect naming the source, see dds_save_redirect().
	return "ddsref";
}

String ResourceImporterDDS::get_resource_type() const {
	return "Texture";
}

void ResourceImporterDDS::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
}

bool ResourceImporterDDS::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return false;
}

Error ResourceImporterDDS::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	return dds_save_redirect(p_save_path + "." + get_save_extension(), p_source_file);
}

String ResourceImporterDDSStreamed::get_importer_name() const {
	return "dds_streamed_texture";
}

String ResourceImporterDDSStreamed::get_visible_name() const {
	return "DDS Texture (Streamed)";
}

void ResourceImporterDDSStreamed::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("dds");
}

String ResourceImporterDDSStreamed::get_save_extension() const {
	return "stex";
}

String ResourceImporterDDSStreamed::get_resource_type() const {
	return "StreamedTexture2D";
}

void ResourceImporterDDSStreamed::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "streaming/min_lod_override", PROPERTY_HINT_ENUM, "Settings,0,1,2,3,4,5,6,7,8,9,10,11,12,13"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "streaming/max_lod_override", PROPERTY_HINT_ENUM, "Settings,0,1,2,3,4,5,6,7,8,9,10,11,12,13"), 0));
}

bool ResourceImporterDDSStreamed::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	return true;
}

Error ResourceImporterDDSStreamed::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	bool is_2d = false;
	Vector<Ref<Image>> images = dds_load_images(p_source_file, is_2d);
	ERR_FAIL_COND_V_MSG(images.is_empty(), ERR_CANT_OPEN, vformat("Failed to read DDS file: %s.", p_source_file));

	ERR_FAIL_COND_V_MSG(!is_2d, ERR_FILE_UNRECOGNIZED,
			vformat("%s does not hold a 2D texture, which is the only kind that can be streamed. Set its importer back to \"DDS Texture\" in the Import dock.", p_source_file));

	const Ref<Image> &image = images[0];
	ERR_FAIL_COND_V_MSG(image.is_null() || image->is_empty(), ERR_CANT_OPEN, vformat("Failed to read DDS file: %s.", p_source_file));

	ERR_FAIL_COND_V_MSG(!image->has_mipmaps(), ERR_FILE_UNRECOGNIZED,
			vformat("%s has no mipmaps, so it cannot be streamed. Re-export it with a full mipmap chain, or set its importer back to \"DDS Texture\" in the Import dock.", p_source_file));

	const uint32_t streaming_min = p_options.has("streaming/min_lod_override") ? uint32_t(p_options["streaming/min_lod_override"]) : 0;
	const uint32_t streaming_max = p_options.has("streaming/max_lod_override") ? uint32_t(p_options["streaming/max_lod_override"]) : 0;

	const Error err = StreamedTexture2D::save_data(p_save_path + ".stex", image, 0, streaming_min, streaming_max);
	ERR_FAIL_COND_V_MSG(err != OK, err, vformat("Failed to save streamed texture for: %s.", p_source_file));

	if (r_metadata) {
		Dictionary meta;
		meta["vram_texture"] = true;
		*r_metadata = meta;
	}

	return OK;
}
