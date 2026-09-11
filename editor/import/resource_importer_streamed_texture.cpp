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

#include "core/io/config_file.h"
#include "core/io/image_loader.h"
#include "editor/file_system/editor_file_system.h"
#include "editor/import/resource_importer_texture_settings.h"
#include "scene/resources/streamed_texture.h"

ResourceImporterStreamedTexture *ResourceImporterStreamedTexture::singleton = nullptr;

void ResourceImporterStreamedTexture::_texture_reimport_roughness(const Ref<StreamedTexture2D> &p_tex, const String &p_normal_path, RSE::TextureDetectRoughnessChannel p_channel) {
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

void ResourceImporterStreamedTexture::_remap_channels(Ref<Image> &r_image, ChannelRemap p_options[4]) {
	ERR_FAIL_COND(r_image->is_compressed());

	// Currently HDR inverted remapping is not allowed.
	bool attempted_hdr_inverted = false;
	if (r_image->get_format() >= Image::FORMAT_RF && r_image->get_format() <= Image::FORMAT_RGBE9995) {
		// Formats which can hold HDR data cannot be inverted the same way as unsigned normalized ones (1.0 - channel).
		for (int i = 0; i < 4; i++) {
			switch (p_options[i]) {
				case REMAP_INV_R:
					attempted_hdr_inverted = true;
					p_options[i] = REMAP_R;
					break;
				case REMAP_INV_G:
					attempted_hdr_inverted = true;
					p_options[i] = REMAP_G;
					break;
				case REMAP_INV_B:
					attempted_hdr_inverted = true;
					p_options[i] = REMAP_B;
					break;
				case REMAP_INV_A:
					attempted_hdr_inverted = true;
					p_options[i] = REMAP_A;
					break;
				default:
					break;
			}
		}
	}

	if (attempted_hdr_inverted) {
		WARN_PRINT("Attempted to use an inverted channel remap on an HDR image. The remap has been changed to its uninverted equivalent.");
	}

	// Optimization: Set the remap from 'unused' to either 0 or 1 to avoid repeated checks in the conversion loop.
	for (int i = 0; i < 4; i++) {
		if (p_options[i] == REMAP_UNUSED) {
			p_options[i] = i == 3 ? REMAP_1 : REMAP_0;
		}
	}

	// Expand the image's channel count in the event that the current set of channels doesn't allow for the desired remap.
	const Image::Format original_format = r_image->get_format();
	const uint32_t channel_mask = Image::get_format_component_mask(original_format);

	// Whether a channel is supported by the format itself.
	const bool has_channel_r = channel_mask & 0x1;
	const bool has_channel_g = channel_mask & 0x2;
	const bool has_channel_b = channel_mask & 0x4;
	const bool has_channel_a = channel_mask & 0x8;

	// Whether a certain channel needs to be remapped.
	const bool remap_r = p_options[0] != REMAP_R ? !(!has_channel_r && p_options[0] == REMAP_0) : false;
	const bool remap_g = p_options[1] != REMAP_G ? !(!has_channel_g && p_options[1] == REMAP_0) : false;
	const bool remap_b = p_options[2] != REMAP_B ? !(!has_channel_b && p_options[2] == REMAP_0) : false;
	const bool remap_a = p_options[3] != REMAP_A ? !(!has_channel_a && p_options[3] == REMAP_1) : false;

	if (!(remap_r || remap_g || remap_b || remap_a)) {
		// Default color map, do nothing.
		return;
	}

	// Whether a certain channel set is needed, either from the source or the remap.
	const bool needs_rg = remap_g || has_channel_g;
	const bool needs_rgb = remap_b || has_channel_b;
	const bool needs_rgba = remap_a || has_channel_a;

	bool could_not_expand = false;
	switch (original_format) {
		case Image::FORMAT_R8:
		case Image::FORMAT_RG8:
		case Image::FORMAT_RGB8: {
			// Convert to either RGBA8, RGB8 or RG8.
			if (needs_rgba) {
				r_image->convert(Image::FORMAT_RGBA8);
			} else if (needs_rgb) {
				r_image->convert(Image::FORMAT_RGB8);
			} else if (needs_rg) {
				r_image->convert(Image::FORMAT_RG8);
			}
		} break;
		case Image::FORMAT_RH:
		case Image::FORMAT_RGH:
		case Image::FORMAT_RGBH: {
			// Convert to either RGBAH, RGBH or RGH.
			if (needs_rgba) {
				r_image->convert(Image::FORMAT_RGBAH);
			} else if (needs_rgb) {
				r_image->convert(Image::FORMAT_RGBH);
			} else if (needs_rg) {
				r_image->convert(Image::FORMAT_RGH);
			}
		} break;
		case Image::FORMAT_RF:
		case Image::FORMAT_RGF:
		case Image::FORMAT_RGBF: {
			// Convert to either RGBAF, RGBF or RGF.
			if (needs_rgba) {
				r_image->convert(Image::FORMAT_RGBAF);
			} else if (needs_rgb) {
				r_image->convert(Image::FORMAT_RGBF);
			} else if (needs_rg) {
				r_image->convert(Image::FORMAT_RGF);
			}
		} break;
		case Image::FORMAT_L8: {
			const bool uniform_rgb = (p_options[0] == p_options[1] && p_options[1] == p_options[2]) || !(remap_r || remap_g || remap_b);
			if (uniform_rgb) {
				// Uniform RGB.
				if (needs_rgba) {
					r_image->convert(Image::FORMAT_LA8);
				}
			} else {
				// Non-uniform RGB.
				if (needs_rgba) {
					r_image->convert(Image::FORMAT_RGBA8);
				} else {
					r_image->convert(Image::FORMAT_RGB8);
				}
				could_not_expand = true;
			}
		} break;
		case Image::FORMAT_LA8: {
			const bool uniform_rgb = (p_options[0] == p_options[1] && p_options[1] == p_options[2]) || !(remap_r || remap_g || remap_b);
			if (!uniform_rgb) {
				// Non-uniform RGB.
				r_image->convert(Image::FORMAT_RGBA8);
				could_not_expand = true;
			}
		} break;
		case Image::FORMAT_RGB565: {
			if (needs_rgba) {
				// RGB565 doesn't have an alpha expansion, convert to RGBA8.
				r_image->convert(Image::FORMAT_RGBA8);
				could_not_expand = true;
			}
		} break;
		case Image::FORMAT_RGBE9995: {
			if (needs_rgba) {
				// RGB9995 doesn't have an alpha expansion, convert to RGBAH.
				r_image->convert(Image::FORMAT_RGBAH);
				could_not_expand = true;
			}
		} break;

		default: {
		} break;
	}

	if (could_not_expand) {
		WARN_PRINT(vformat("Unable to expand image format %s's channels (the target format does not exist), converting to %s as a fallback.",
				Image::get_format_name(original_format), Image::get_format_name(r_image->get_format())));
	}

	// Remap the channels.
	for (int x = 0; x < r_image->get_width(); x++) {
		for (int y = 0; y < r_image->get_height(); y++) {
			Color src = r_image->get_pixel(x, y);
			Color dst;

			for (int i = 0; i < 4; i++) {
				switch (p_options[i]) {
					case REMAP_R:
						dst[i] = src.r;
						break;
					case REMAP_G:
						dst[i] = src.g;
						break;
					case REMAP_B:
						dst[i] = src.b;
						break;
					case REMAP_A:
						dst[i] = src.a;
						break;

					case REMAP_INV_R:
						dst[i] = 1.0f - src.r;
						break;
					case REMAP_INV_G:
						dst[i] = 1.0f - src.g;
						break;
					case REMAP_INV_B:
						dst[i] = 1.0f - src.b;
						break;
					case REMAP_INV_A:
						dst[i] = 1.0f - src.a;
						break;

					case REMAP_0:
						dst[i] = 0.0f;
						break;
					case REMAP_1:
						dst[i] = 1.0f;
						break;

					default:
						break;
				}
			}

			r_image->set_pixel(x, y, dst);
		}
	}
}

void ResourceImporterStreamedTexture::_clamp_hdr_exposure(Ref<Image> &r_image) {
	// Clamp HDR exposure following Filament's tonemapping formula.
	// This can be used to reduce fireflies in environment maps or reduce the influence
	// of the sun from an HDRI panorama on environment lighting (when a DirectionalLight3D is used instead).
	const int height = r_image->get_height();
	const int width = r_image->get_width();

	// These values are chosen arbitrarily and seem to produce good results with 4,096 samples.
	const float linear = 4096.0;
	const float compressed = 16384.0;

	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			const Color color = r_image->get_pixel(i, j);
			const float luma = color.get_luminance();

			Color clamped_color;
			if (luma <= linear) {
				clamped_color = color;
			} else {
				clamped_color = (color / luma) * ((linear * linear - compressed * luma) / (2 * linear - compressed - luma));
			}

			r_image->set_pixel(i, j, clamped_color);
		}
	}
}

String ResourceImporterStreamedTexture::get_importer_name() const {
	return "streamed_texture_2d";
}
String ResourceImporterStreamedTexture::get_visible_name() const {
	return "Texture2D Streamed";
}
void ResourceImporterStreamedTexture::get_recognized_extensions(List<String> *p_extensions) const {
	ImageLoader::get_recognized_extensions(p_extensions);
}

String ResourceImporterStreamedTexture::get_save_extension() const {
	return "stex";
}
String ResourceImporterStreamedTexture::get_resource_type() const {
	return "StreamedTexture2D";
}

void ResourceImporterStreamedTexture::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress/high_quality"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/hdr_compression", PROPERTY_HINT_ENUM, "Disabled,Opaque Only,Always"), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/normal_map", PROPERTY_HINT_ENUM, "Detect,Enable,Disabled"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/channel_pack", PROPERTY_HINT_ENUM, "sRGB Friendly,Optimized"), 0));

	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "mipmaps/preserve_alpha_test_coverage", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "mipmaps/alpha_test_threshold", PROPERTY_HINT_RANGE, "0.01,0.99,0.01"), 0.5));

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "roughness/mode", PROPERTY_HINT_ENUM, "Detect,Disabled,Red,Green,Blue,Alpha,Gray"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "roughness/src_normal", PROPERTY_HINT_FILE, "*.bmp,*.exr,*.jpeg,*.jpg,*.hdr,*.png,*.svg,*.tga,*.webp"), ""));

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "process/channel_remap/red", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Inverted Red,Inverted Green,Inverted Blue,Inverted Alpha,Unused,Zero,One"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "process/channel_remap/green", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Inverted Red,Inverted Green,Inverted Blue,Inverted Alpha,Unused,Zero,One"), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "process/channel_remap/blue", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Inverted Red,Inverted Green,Inverted Blue,Inverted Alpha,Unused,Zero,One"), 2));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "process/channel_remap/alpha", PROPERTY_HINT_ENUM, "Red,Green,Blue,Alpha,Inverted Red,Inverted Green,Inverted Blue,Inverted Alpha,Unused,Zero,One"), 3));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/hdr_as_srgb"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/hdr_clamp_exposure"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "process/size_limit", PROPERTY_HINT_RANGE, "0,32768,1"), 0));

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "streaming/min_lod_override", PROPERTY_HINT_ENUM, "Settings,0,1,2,3,4,5,6,7,8,9,10,11,12,13"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "streaming/max_lod_override", PROPERTY_HINT_ENUM, "Settings,0,1,2,3,4,5,6,7,8,9,10,11,12,13"), 0));
}

bool ResourceImporterStreamedTexture::get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const {
	if (p_option == "mipmaps/alpha_test_threshold") {
		return p_options["mipmaps/preserve_alpha_test_coverage"];
	}
	return true;
}

Error ResourceImporterStreamedTexture::import(ResourceUID::ID p_source_id, const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	Ref<Image> image;
	Error err;
	image.instantiate();

	uint32_t save_flags = 0;
	Image::UsedChannels used_channels = Image::USED_CHANNELS_RGBA;

	// Compression options.
	const bool high_quality = p_options.has("compress/high_quality") && bool(p_options["compress/high_quality"]);
	const int hdr_compression = p_options.has("compress/hdr_compression") ? int(p_options["compress/hdr_compression"]) : 1;
	const int pack_channels = p_options.has("compress/channel_pack") ? int(p_options["compress/channel_pack"]) : 0;
	const bool srgb_friendly_pack = pack_channels == 0;

	Image::CompressSource comp_source = srgb_friendly_pack ? Image::COMPRESS_SOURCE_SRGB : Image::COMPRESS_SOURCE_GENERIC;

	// Mipmaps
	const bool mipmaps_preserve_alpha_test_coverage = p_options["mipmaps/preserve_alpha_test_coverage"];
	const float mipmaps_alpha_test_threshold = p_options["mipmaps/alpha_test_threshold"];

	// Roughness.
	const int roughness = p_options["roughness/mode"];
	const bool detect_roughness = roughness == 0;

	// Normal map.
	const String normal_map = p_options["roughness/src_normal"];
	const int normal = p_options["compress/normal_map"];
	const bool detect_normal = normal == 0; // Normal is set to Detect
	const bool force_normal = normal == 1; // Normal is set to Enable

	// Processing.
	const int remap_r = p_options["process/channel_remap/red"];
	const int remap_g = p_options["process/channel_remap/green"];
	const int remap_b = p_options["process/channel_remap/blue"];
	const int remap_a = p_options["process/channel_remap/alpha"];

	const bool hdr_as_srgb = p_options["process/hdr_as_srgb"];
	const bool hdr_clamp_exposure = p_options["process/hdr_clamp_exposure"];
	int size_limit = p_options["process/size_limit"];

	bool using_fallback_size_limit = false;
	if (size_limit == 0) {
		using_fallback_size_limit = true;
		// If no size limit is defined, use a fallback size limit to prevent textures from looking incorrect or failing to import.
		// As of June 2024, no GPU can correctly display a texture larger than 32768 pixels on either axis.
		size_limit = 32768;
	}

	// Parse import options.
	int32_t loader_flags = ImageFormatLoader::FLAG_NONE;
	if (hdr_as_srgb) {
		loader_flags |= ImageFormatLoader::FLAG_FORCE_LINEAR;
	}

	if (detect_normal || force_normal) {
		save_flags |= StreamedTexture2D::FORMAT_BIT_DETECT_NORMAL;
	}

	if (detect_roughness) {
		save_flags |= StreamedTexture2D::FORMAT_BIT_DETECT_ROUGHNESS;
	}

	if (force_normal) {
		comp_source = Image::COMPRESS_SOURCE_NORMAL;
	}

	err = ImageLoader::load_image(p_source_file, image, nullptr, loader_flags);

	{
		ChannelRemap remaps[4] = {
			(ChannelRemap)remap_r,
			(ChannelRemap)remap_g,
			(ChannelRemap)remap_b,
			(ChannelRemap)remap_a,
		};

		_remap_channels(image, remaps);
	}

	// Clamp HDR exposure.
	if (hdr_clamp_exposure) {
		_clamp_hdr_exposure(image);
	}

	// Apply the size limit.
	if (size_limit > 0 && (image->get_width() > size_limit || image->get_height() > size_limit)) {
		if (image->get_width() >= image->get_height()) {
			int new_width = size_limit;
			int new_height = image->get_height() * new_width / image->get_width();

			if (using_fallback_size_limit) {
				// Only warn if downsizing occurred when the user did not explicitly request it.
				WARN_PRINT(vformat("%s: Texture was downsized on import as its width (%d pixels) exceeded the importable size limit (%d pixels).", p_source_file, image->get_width(), size_limit));
			}
			image->resize(new_width, new_height, Image::INTERPOLATE_CUBIC);
		} else {
			int new_height = size_limit;
			int new_width = image->get_width() * new_height / image->get_height();

			if (using_fallback_size_limit) {
				// Only warn if downsizing occurred when the user did not explicitly request it.
				WARN_PRINT(vformat("%s: Texture was downsized on import as its height (%d pixels) exceeded the importable size limit (%d pixels).", p_source_file, image->get_height(), size_limit));
			}
			image->resize(new_width, new_height, Image::INTERPOLATE_CUBIC);
		}

		if (normal == 1) {
			image->normalize();
		}
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
		image->generate_mipmaps(force_normal, mipmaps_preserve_alpha_test_coverage, mipmaps_alpha_test_threshold);
	}

	// Generate roughness mipmaps from normal texture.
	if (image->has_mipmaps() && normal_image.is_valid()) {
		image->generate_mipmap_roughness(roughness_channel, normal_image);
	}

	if (err != OK || image.is_null() || image->is_empty()) {
		return ERR_CANT_OPEN;
	}

	// Detect used channels for optimal compression (after image is fully loaded/processed).
	if (!image->is_compressed()) {
		used_channels = image->detect_used_channels(comp_source);
	}

	Array formats_imported;

	// Streaming lod range overrides.
	const uint32_t streaming_min = p_options.has("streaming/min_lod_override") ? uint32_t(p_options["streaming/min_lod_override"]) : 0;
	const uint32_t streaming_max = p_options.has("streaming/max_lod_override") ? uint32_t(p_options["streaming/max_lod_override"]) : 0;

	const bool can_s3tc_bptc = ResourceImporterTextureSettings::should_import_s3tc_bptc();
	const bool can_etc2_astc = ResourceImporterTextureSettings::should_import_etc2_astc();
	ERR_FAIL_COND_V_MSG(!can_s3tc_bptc && !can_etc2_astc, FAILED, "No supported compression formats are enabled in the project settings for streamed textures.");

	// HDR handling.
	const bool is_hdr = (image->get_format() >= Image::FORMAT_RF && image->get_format() <= Image::FORMAT_RGBE9995);
	bool can_compress_hdr = hdr_compression > 0;
	bool force_uncompressed = false;

	if (is_hdr) {
		bool has_alpha = image->detect_alpha() != Image::ALPHA_NONE;
		if (has_alpha) {
			// HDR with alpha is not compressible to BC6H/ASTC-HDR.
			if (hdr_compression == 2) {
				// User selected "Always", so force an alpha-less format.
				if (image->get_format() == Image::FORMAT_RGBAF) {
					image->convert(Image::FORMAT_RGBF);
				} else if (image->get_format() == Image::FORMAT_RGBAH) {
					image->convert(Image::FORMAT_RGBH);
				}
			} else {
				can_compress_hdr = false;
			}
		}

		// Fall back to RGBE9995 uncompressed if HDR compression is disabled.
		if (!can_compress_hdr && image->get_format() != Image::FORMAT_RGBE9995) {
			image->convert(Image::FORMAT_RGBE9995);
			force_uncompressed = true;
		}
	}

	if (force_uncompressed) {
		// Save uncompressed (no platform variants needed).
		Error err_unc = StreamedTexture2D::_save_data(p_save_path + ".stex", image, save_flags, streaming_min, streaming_max);
		ERR_FAIL_COND_V_MSG(err_unc != OK, err_unc, "Failed to save uncompressed HDR streamed texture.");
	} else {
		if (can_s3tc_bptc) {
			formats_imported.push_back("s3tc_bptc");
			Image::CompressMode image_compress_mode;
			String image_compress_format;
			if (high_quality || is_hdr) {
				image_compress_mode = Image::COMPRESS_BPTC;
				image_compress_format = "bptc";
			} else {
				image_compress_mode = Image::COMPRESS_S3TC;
				image_compress_format = "s3tc";
			}
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
			Image::CompressMode image_compress_mode;
			String image_compress_format;
			if (high_quality || is_hdr) {
				image_compress_mode = Image::COMPRESS_ASTC;
				image_compress_format = "astc";
			} else {
				image_compress_mode = Image::COMPRESS_ETC2;
				image_compress_format = "etc2";
			}
			Ref<Image> image_etc2_astc = image->duplicate();
			image_etc2_astc->compress_from_channels(image_compress_mode, used_channels);
			Error err_etc2 = StreamedTexture2D::_save_data(p_save_path + "." + image_compress_format + ".stex", image_etc2_astc, save_flags, streaming_min, streaming_max);
			ERR_FAIL_COND_V_MSG(err_etc2 != OK, err_etc2, "Failed to save ETC2/ASTC streamed texture.");
			if (err_etc2 == OK) {
				r_platform_variants->push_back(image_compress_format);
			}
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
