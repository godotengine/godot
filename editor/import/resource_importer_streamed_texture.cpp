/**************************************************************************/
/*  resource_importer_streamed_texture.cpp                                */
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

#include "resource_importer_streamed_texture.h"

#include "core/config/project_settings.h"
#include "core/io/config_file.h"
#include "core/io/image_loader.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/import/resource_importer_texture_settings.h"
#include "scene/resources/streamed_texture.h"

ResourceImporterStreamedTexture *ResourceImporterStreamedTexture::singleton = nullptr;

void ResourceImporterStreamedTexture::_texture_reimport_roughness(const Ref<StreamedTexture2D> &p_tex, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_channel) {
	ERR_FAIL_COND(p_tex.is_null());

	MutexLock lock(singleton->mutex);
	StringName path = p_tex->get_path();

	if (!singleton->make_flags.has(path)) {
		singleton->make_flags[path] = MakeInfo();
	}

	singleton->make_flags[path].flags |= MAKE_ROUGHNESS_FLAG;
	singleton->make_flags[path].channel_for_roughness = p_channel;
	singleton->make_flags[path].normal_path_for_roughness = p_normal_path;
}

void ResourceImporterStreamedTexture::_texture_reimport_normal(const Ref<StreamedTexture2D> &p_tex) {
	ERR_FAIL_COND(p_tex.is_null());

	MutexLock lock(singleton->mutex);
	StringName path = p_tex->get_path();

	if (!singleton->make_flags.has(path)) {
		singleton->make_flags[path] = MakeInfo();
	}

	singleton->make_flags[path].flags |= MAKE_NORMAL_FLAG;
}

String ResourceImporterStreamedTexture::get_importer_name() const {
	return "streamed_texture_2d";
}
String ResourceImporterStreamedTexture::get_visible_name() const {
	return "Texture2D Streamed";
}
void ResourceImporterStreamedTexture::get_recognized_extensions(List<String> *p_extensions) const {
	ImageLoader::get_recognized_extensions(p_extensions);
	p_extensions->push_back("dds");
}

String ResourceImporterStreamedTexture::get_save_extension() const {
	return "stex";
}
String ResourceImporterStreamedTexture::get_resource_type() const {
	return "StreamedTexture2D";
}

void ResourceImporterStreamedTexture::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/normal_map", PROPERTY_HINT_ENUM, "Detect,Enable,Disabled"), 0));

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "roughness/mode", PROPERTY_HINT_ENUM, "Detect,Disabled,Red,Green,Blue,Alpha,Gray"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "roughness/src_normal", PROPERTY_HINT_FILE, "*.bmp,*.dds,*.exr,*.jpeg,*.jpg,*.hdr,*.png,*.svg,*.tga,*.webp"), ""));

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "streaming/min_resolution", PROPERTY_HINT_ENUM, "System,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "streaming/max_resolution", PROPERTY_HINT_ENUM, "System,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), 0));
}

bool ResourceImporterStreamedTexture::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	const String extension = p_path.get_extension().to_lower();
	if (extension != "dds") {
		return true;
	}

	if (p_option == "streaming/min_resolution" || p_option == "streaming/max_resolution") {
		return true;
	}

	return false;
}

Error ResourceImporterStreamedTexture::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	Ref<Image> image;
	Error err;
	image.instantiate();

	uint32_t save_flags = 0;
	Image::UsedChannels used_channels = Image::USED_CHANNELS_RGBA;
	Image::CompressSource comp_source = Image::COMPRESS_SOURCE_SRGB;

	const String extension = p_source_file.get_extension().to_lower();
	if (extension == "dds") {
		// DDS files are loaded directly without additional processing.
		// They typically contain pre-compressed data with mipmaps already generated,
		// so roughness/normal detection and mipmap generation are skipped.
		Vector<uint8_t> data = FileAccess::get_file_as_bytes(p_source_file);
		if (data.is_empty()) {
			return ERR_CANT_OPEN;
		}

		err = image->load_dds_from_buffer(data);

		// TODO: Detect used channels from the DDS data itself if possible.
	} else {
		err = ImageLoader::load_image(p_source_file, image);

		// Roughness.
		const int roughness = p_options["roughness/mode"];
		const bool detect_roughness = roughness == 0;

		// Normal map.
		const String normal_map = p_options["roughness/src_normal"];
		const int normal = p_options["compress/normal_map"];
		const bool detect_normal = normal == 0;
		const bool force_normal = normal == 1;

		if (detect_normal) {
			save_flags |= StreamedTexture2D::FORMAT_BIT_DETECT_NORMAL;
		}

		if (detect_roughness) {
			save_flags |= StreamedTexture2D::FORMAT_BIT_DETECT_ROUGHNESS;
		}

		if (force_normal) {
			comp_source = Image::COMPRESS_SOURCE_NORMAL;
		}

		// Load the normal image.
		Ref<Image> normal_image;
		Image::RoughnessChannel roughness_channel = Image::ROUGHNESS_CHANNEL_R;

		if (roughness > 1 && FileAccess::exists(normal_map)) {
			normal_image.instantiate();
			if (ImageLoader::load_image(normal_map, normal_image) == OK) {
				roughness_channel = Image::RoughnessChannel(roughness - 2);
			}
		}

		if (!image->has_mipmaps() || force_normal) {
			image->generate_mipmaps(force_normal);
		}

		// Generate roughness mipmaps from normal texture.
		if (image->has_mipmaps() && normal_image.is_valid()) {
			image->generate_mipmap_roughness(roughness_channel, normal_image);
		}
	}

	if (err != OK || image.is_null() || image->is_empty()) {
		return ERR_CANT_OPEN;
	}

	// Detect used channels for optimal compression (after image is fully loaded/processed).
	if (!image->is_compressed()) {
		used_channels = image->detect_used_channels(comp_source);
	}

	Array formats_imported;

	// Streaming resolutions.
	const uint32_t streaming_min = p_options["streaming/min_resolution"];
	const uint32_t streaming_max = p_options["streaming/max_resolution"];

	const bool can_s3tc_bptc = ResourceImporterTextureSettings::should_import_s3tc_bptc();
	const bool can_etc2_astc = ResourceImporterTextureSettings::should_import_etc2_astc();
	ERR_FAIL_COND_V_MSG(!can_s3tc_bptc && !can_etc2_astc, FAILED, "No supported compression formats are enabled in the project settings for streamed textures.");

	const bool high_quality = GLOBAL_GET("rendering/textures/streaming/import_high_quality");

	if (can_s3tc_bptc) {
		formats_imported.push_back("s3tc_bptc");
		Image::CompressMode image_compress_mode = high_quality ? Image::COMPRESS_BPTC : Image::COMPRESS_S3TC;
		String image_compress_format = high_quality ? "bptc" : "s3tc";
		Ref<Image> image_s3tc_bptc = image->duplicate();
		image_s3tc_bptc->compress_from_channels(image_compress_mode, used_channels);
		Error err_s3tc = StreamedTexture2D::_save_data(p_save_path + "." + image_compress_format + ".stex", image_s3tc_bptc, save_flags, streaming_min, streaming_max);
		ERR_FAIL_COND_V_MSG(err_s3tc != OK, err_s3tc, "Failed to save S3TC/BPTC streamed texture.");
		if (err_s3tc == OK) {
			r_platform_variants->push_back(image_compress_format);
		}
	}

	if (can_etc2_astc) {
		formats_imported.push_back("etc2_astc");
		Image::CompressMode image_compress_mode = high_quality ? Image::COMPRESS_ASTC : Image::COMPRESS_ETC2;
		String image_compress_format = high_quality ? "astc" : "etc2";
		Ref<Image> image_etc2_astc = image->duplicate();
		image_etc2_astc->compress_from_channels(image_compress_mode, used_channels);

		Error err_etc2 = StreamedTexture2D::_save_data(p_save_path + "." + image_compress_format + ".stex", image_etc2_astc, save_flags, streaming_min, streaming_max);
		ERR_FAIL_COND_V_MSG(err_etc2 != OK, err_etc2, "Failed to save ETC2/ASTC streamed texture.");
		if (err_etc2 == OK) {
			r_platform_variants->push_back(image_compress_format);
		}
	}

	if (r_metadata) {
		Dictionary meta;
		meta["vram_texture"] = true;

		if (formats_imported.size()) {
			meta["imported_formats"] = formats_imported;
		}

		*r_metadata = meta;
	}

	return OK;
}

ResourceImporterStreamedTexture::ResourceImporterStreamedTexture(bool p_singleton) {
	// This should only be set through the EditorNode.
	if (p_singleton) {
		singleton = this;
	}

	StreamedTexture2D::request_roughness_callback = _texture_reimport_roughness;
	StreamedTexture2D::request_normal_callback = _texture_reimport_normal;
}

void ResourceImporterStreamedTexture::update_imports() {
	if (EditorFileSystem::get_singleton()->is_scanning() || EditorFileSystem::get_singleton()->is_importing()) {
		return; // Don't update when EditorFileSystem is doing something else.
	}

	MutexLock lock(mutex);
	Vector<String> to_reimport;

	if (make_flags.is_empty()) {
		return;
	}

	for (const KeyValue<StringName, MakeInfo> &E : make_flags) {
		Ref<ConfigFile> cf;
		cf.instantiate();
		String src_path = String(E.key) + ".import";

		Error err = cf->load(src_path);
		ERR_CONTINUE(err != OK);

		bool changed = false;

		if (E.value.flags & MAKE_NORMAL_FLAG && int(cf->get_value("params", "compress/normal_map")) == 0) {
			print_line(
					vformat("%s: Texture detected as used as a normal map in 3D. Enabling red-green texture compression to reduce memory usage (blue channel is discarded).",
							String(E.key)));

			cf->set_value("params", "compress/normal_map", 1);
			changed = true;
		}

		if (E.value.flags & MAKE_ROUGHNESS_FLAG && int(cf->get_value("params", "roughness/mode")) == 0) {
			print_line(
					vformat("%s: Texture detected as used as a roughness map in 3D. Enabling roughness limiter based on the detected associated normal map at %s.",
							String(E.key), E.value.normal_path_for_roughness));

			cf->set_value("params", "roughness/mode", E.value.channel_for_roughness + 2);
			cf->set_value("params", "roughness/src_normal", E.value.normal_path_for_roughness);
			changed = true;
		}

		if (changed) {
			cf->save(src_path);
			to_reimport.push_back(E.key);
		}
	}

	make_flags.clear();

	if (!to_reimport.is_empty()) {
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
	}
}
