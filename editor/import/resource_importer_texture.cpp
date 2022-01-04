/*************************************************************************/
/*  resource_importer_texture.cpp                                        */
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

#include "resource_importer_texture.h"

#include "core/io/config_file.h"
#include "core/io/image_loader.h"
#include "core/version.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"

void ResourceImporterTexture::_texture_reimport_roughness(const Ref<StreamTexture2D> &p_tex, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_channel) {
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

void ResourceImporterTexture::_texture_reimport_3d(const Ref<StreamTexture2D> &p_tex) {
	ERR_FAIL_COND(p_tex.is_null());

	MutexLock lock(singleton->mutex);

	StringName path = p_tex->get_path();

	if (!singleton->make_flags.has(path)) {
		singleton->make_flags[path] = MakeInfo();
	}

	singleton->make_flags[path].flags |= MAKE_3D_FLAG;
}

void ResourceImporterTexture::_texture_reimport_normal(const Ref<StreamTexture2D> &p_tex) {
	ERR_FAIL_COND(p_tex.is_null());

	MutexLock lock(singleton->mutex);

	StringName path = p_tex->get_path();

	if (!singleton->make_flags.has(path)) {
		singleton->make_flags[path] = MakeInfo();
	}

	singleton->make_flags[path].flags |= MAKE_NORMAL_FLAG;
}

void ResourceImporterTexture::update_imports() {
	if (EditorFileSystem::get_singleton()->is_scanning() || EditorFileSystem::get_singleton()->is_importing()) {
		return; // do nothing for now
	}

	MutexLock lock(mutex);
	Vector<String> to_reimport;
	{
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
				cf->set_value("params", "compress/normal_map", 1);
				changed = true;
			}

			if (E.value.flags & MAKE_ROUGHNESS_FLAG && int(cf->get_value("params", "roughness/mode")) == 0) {
				cf->set_value("params", "roughness/mode", E.value.channel_for_roughness + 2);
				cf->set_value("params", "roughness/src_normal", E.value.normal_path_for_roughness);
				changed = true;
			}

			if (E.value.flags & MAKE_3D_FLAG && bool(cf->get_value("params", "detect_3d/compress_to"))) {
				int compress_to = cf->get_value("params", "detect_3d/compress_to");
				cf->set_value("params", "detect_3d/compress_to", 0);
				if (compress_to == 1) {
					cf->set_value("params", "compress/mode", COMPRESS_VRAM_COMPRESSED);
				} else if (compress_to == 2) {
					cf->set_value("params", "compress/mode", COMPRESS_BASIS_UNIVERSAL);
				}
				cf->set_value("params", "mipmaps/generate", true);
				changed = true;
			}

			if (changed) {
				cf->save(src_path);
				to_reimport.push_back(E.key);
			}
		}

		make_flags.clear();
	}

	if (to_reimport.size()) {
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
	}
}

String ResourceImporterTexture::get_importer_name() const {
	return "texture";
}

String ResourceImporterTexture::get_visible_name() const {
	return "Texture2D";
}

void ResourceImporterTexture::get_recognized_extensions(List<String> *p_extensions) const {
	ImageLoader::get_recognized_extensions(p_extensions);
}

String ResourceImporterTexture::get_save_extension() const {
	return "stex";
}

String ResourceImporterTexture::get_resource_type() const {
	return "StreamTexture2D";
}

bool ResourceImporterTexture::get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) const {
	if (p_option == "compress/lossy_quality") {
		int compress_mode = int(p_options["compress/mode"]);
		if (compress_mode != COMPRESS_LOSSY && compress_mode != COMPRESS_VRAM_COMPRESSED) {
			return false;
		}
	} else if (p_option == "compress/hdr_mode") {
		int compress_mode = int(p_options["compress/mode"]);
		if (compress_mode < COMPRESS_VRAM_COMPRESSED) {
			return false;
		}
	} else if (p_option == "mipmaps/limit") {
		return p_options["mipmaps/generate"];

	} else if (p_option == "compress/bptc_ldr") {
		int compress_mode = int(p_options["compress/mode"]);
		if (compress_mode < COMPRESS_VRAM_COMPRESSED) {
			return false;
		}
		if (!ProjectSettings::get_singleton()->get("rendering/textures/vram_compression/import_bptc")) {
			return false;
		}
	}

	return true;
}

int ResourceImporterTexture::get_preset_count() const {
	return 3;
}

String ResourceImporterTexture::get_preset_name(int p_idx) const {
	static const char *preset_names[] = {
		"2D/3D (Auto-Detect)",
		"2D",
		"3D",
	};

	return preset_names[p_idx];
}

void ResourceImporterTexture::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "Lossless,Lossy,VRAM Compressed,VRAM Uncompressed,Basis Universal", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), p_preset == PRESET_3D ? 2 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "compress/lossy_quality", PROPERTY_HINT_RANGE, "0,1,0.01"), 0.7));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/hdr_compression", PROPERTY_HINT_ENUM, "Disabled,Opaque Only,Always"), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/bptc_ldr", PROPERTY_HINT_ENUM, "Disabled,Enabled,RGBA Only"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/normal_map", PROPERTY_HINT_ENUM, "Detect,Enable,Disabled"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/channel_pack", PROPERTY_HINT_ENUM, "sRGB Friendly,Optimized"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress/streamed"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "mipmaps/generate"), (p_preset == PRESET_3D ? true : false)));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "mipmaps/limit", PROPERTY_HINT_RANGE, "-1,256"), -1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "roughness/mode", PROPERTY_HINT_ENUM, "Detect,Disabled,Red,Green,Blue,Alpha,Gray"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "roughness/src_normal", PROPERTY_HINT_FILE, "*.bmp,*.dds,*.exr,*.jpeg,*.jpg,*.hdr,*.png,*.svg,*.svgz,*.tga,*.webp"), ""));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/fix_alpha_border"), p_preset != PRESET_3D));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/premult_alpha"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/normal_map_invert_y"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/HDR_as_SRGB"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "process/size_limit", PROPERTY_HINT_RANGE, "0,4096,1"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "detect_3d/compress_to", PROPERTY_HINT_ENUM, "Disabled,VRAM Compressed,Basis Universal"), (p_preset == PRESET_DETECT) ? 1 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "svg/scale", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 1.0));
}

void ResourceImporterTexture::save_to_stex_format(FileAccess *f, const Ref<Image> &p_image, CompressMode p_compress_mode, Image::UsedChannels p_channels, Image::CompressMode p_compress_format, float p_lossy_quality) {
	switch (p_compress_mode) {
		case COMPRESS_LOSSLESS: {
			bool lossless_force_png = ProjectSettings::get_singleton()->get("rendering/textures/lossless_compression/force_png") ||
					!Image::_webp_mem_loader_func; // WebP module disabled.
			bool use_webp = !lossless_force_png && p_image->get_width() <= 16383 && p_image->get_height() <= 16383; // WebP has a size limit
			f->store_32(use_webp ? StreamTexture2D::DATA_FORMAT_WEBP : StreamTexture2D::DATA_FORMAT_PNG);
			f->store_16(p_image->get_width());
			f->store_16(p_image->get_height());
			f->store_32(p_image->get_mipmap_count());
			f->store_32(p_image->get_format());

			for (int i = 0; i < p_image->get_mipmap_count() + 1; i++) {
				Vector<uint8_t> data;
				if (use_webp) {
					data = Image::webp_lossless_packer(p_image->get_image_from_mipmap(i));
				} else {
					data = Image::png_packer(p_image->get_image_from_mipmap(i));
				}
				int data_len = data.size();
				f->store_32(data_len);

				const uint8_t *r = data.ptr();
				f->store_buffer(r, data_len);
			}

		} break;
		case COMPRESS_LOSSY: {
			f->store_32(StreamTexture2D::DATA_FORMAT_WEBP);
			f->store_16(p_image->get_width());
			f->store_16(p_image->get_height());
			f->store_32(p_image->get_mipmap_count());
			f->store_32(p_image->get_format());

			for (int i = 0; i < p_image->get_mipmap_count() + 1; i++) {
				Vector<uint8_t> data = Image::webp_lossy_packer(p_image->get_image_from_mipmap(i), p_lossy_quality);
				int data_len = data.size();
				f->store_32(data_len);

				const uint8_t *r = data.ptr();
				f->store_buffer(r, data_len);
			}
		} break;
		case COMPRESS_VRAM_COMPRESSED: {
			Ref<Image> image = p_image->duplicate();

			image->compress_from_channels(p_compress_format, p_channels, p_lossy_quality);

			f->store_32(StreamTexture2D::DATA_FORMAT_IMAGE);
			f->store_16(image->get_width());
			f->store_16(image->get_height());
			f->store_32(image->get_mipmap_count());
			f->store_32(image->get_format());

			Vector<uint8_t> data = image->get_data();
			int dl = data.size();
			const uint8_t *r = data.ptr();
			f->store_buffer(r, dl);
		} break;
		case COMPRESS_VRAM_UNCOMPRESSED: {
			f->store_32(StreamTexture2D::DATA_FORMAT_IMAGE);
			f->store_16(p_image->get_width());
			f->store_16(p_image->get_height());
			f->store_32(p_image->get_mipmap_count());
			f->store_32(p_image->get_format());

			Vector<uint8_t> data = p_image->get_data();
			int dl = data.size();
			const uint8_t *r = data.ptr();

			f->store_buffer(r, dl);

		} break;
		case COMPRESS_BASIS_UNIVERSAL: {
			f->store_32(StreamTexture2D::DATA_FORMAT_BASIS_UNIVERSAL);
			f->store_16(p_image->get_width());
			f->store_16(p_image->get_height());
			f->store_32(p_image->get_mipmap_count());
			f->store_32(p_image->get_format());

			for (int i = 0; i < p_image->get_mipmap_count() + 1; i++) {
				Vector<uint8_t> data = Image::basis_universal_packer(p_image->get_image_from_mipmap(i), p_channels);
				int data_len = data.size();
				f->store_32(data_len);

				const uint8_t *r = data.ptr();
				f->store_buffer(r, data_len);
			}
		} break;
	}
}

void ResourceImporterTexture::_save_stex(const Ref<Image> &p_image, const String &p_to_path, CompressMode p_compress_mode, float p_lossy_quality, Image::CompressMode p_vram_compression, bool p_mipmaps, bool p_streamable, bool p_detect_3d, bool p_detect_roughness, bool p_detect_normal, bool p_force_normal, bool p_srgb_friendly, bool p_force_po2_for_compressed, uint32_t p_limit_mipmap, const Ref<Image> &p_normal, Image::RoughnessChannel p_roughness_channel) {
	FileAccess *f = FileAccess::open(p_to_path, FileAccess::WRITE);
	ERR_FAIL_NULL(f);
	f->store_8('G');
	f->store_8('S');
	f->store_8('T');
	f->store_8('2'); //godot streamable texture 2D

	//format version
	f->store_32(StreamTexture2D::FORMAT_VERSION);
	//texture may be resized later, so original size must be saved first
	f->store_32(p_image->get_width());
	f->store_32(p_image->get_height());

	uint32_t flags = 0;
	if (p_streamable) {
		flags |= StreamTexture2D::FORMAT_BIT_STREAM;
	}
	if (p_mipmaps) {
		flags |= StreamTexture2D::FORMAT_BIT_HAS_MIPMAPS; //mipmaps bit
	}
	if (p_detect_3d) {
		flags |= StreamTexture2D::FORMAT_BIT_DETECT_3D;
	}
	if (p_detect_roughness) {
		flags |= StreamTexture2D::FORMAT_BIT_DETECT_ROUGNESS;
	}
	if (p_detect_normal) {
		flags |= StreamTexture2D::FORMAT_BIT_DETECT_NORMAL;
	}

	f->store_32(flags);
	f->store_32(p_limit_mipmap);
	//reserved for future use
	f->store_32(0);
	f->store_32(0);
	f->store_32(0);

	/*
	print_line("streamable " + itos(p_streamable));
	print_line("mipmaps " + itos(p_mipmaps));
	print_line("detect_3d " + itos(p_detect_3d));
	print_line("roughness " + itos(p_detect_roughness));
	print_line("normal " + itos(p_detect_normal));
*/

	if ((p_compress_mode == COMPRESS_LOSSLESS || p_compress_mode == COMPRESS_LOSSY) && p_image->get_format() > Image::FORMAT_RGBA8) {
		p_compress_mode = COMPRESS_VRAM_UNCOMPRESSED; //these can't go as lossy
	}

	Ref<Image> image = p_image->duplicate();

	if (((p_compress_mode == COMPRESS_BASIS_UNIVERSAL) || (p_compress_mode == COMPRESS_VRAM_COMPRESSED && p_force_po2_for_compressed)) && p_mipmaps) {
		image->resize_to_po2();
	}

	if (p_mipmaps && (!image->has_mipmaps() || p_force_normal)) {
		image->generate_mipmaps(p_force_normal);
	}

	if (!p_mipmaps) {
		image->clear_mipmaps();
	}

	if (image->has_mipmaps() && p_normal.is_valid()) {
		image->generate_mipmap_roughness(p_roughness_channel, p_normal);
	}

	Image::CompressSource csource = Image::COMPRESS_SOURCE_GENERIC;
	if (p_force_normal) {
		csource = Image::COMPRESS_SOURCE_NORMAL;
	} else if (p_srgb_friendly) {
		csource = Image::COMPRESS_SOURCE_SRGB;
	}

	Image::UsedChannels used_channels = image->detect_used_channels(csource);

	save_to_stex_format(f, image, p_compress_mode, used_channels, p_vram_compression, p_lossy_quality);

	memdelete(f);
}

Error ResourceImporterTexture::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {
	CompressMode compress_mode = CompressMode(int(p_options["compress/mode"]));
	float lossy = p_options["compress/lossy_quality"];
	int pack_channels = p_options["compress/channel_pack"];
	bool mipmaps = p_options["mipmaps/generate"];
	uint32_t mipmap_limit = mipmaps ? uint32_t(p_options["mipmaps/limit"]) : uint32_t(-1);
	bool fix_alpha_border = p_options["process/fix_alpha_border"];
	bool premult_alpha = p_options["process/premult_alpha"];
	bool normal_map_invert_y = p_options["process/normal_map_invert_y"];
	bool stream = p_options["compress/streamed"];
	int size_limit = p_options["process/size_limit"];
	bool hdr_as_srgb = p_options["process/HDR_as_SRGB"];
	int normal = p_options["compress/normal_map"];
	float scale = p_options["svg/scale"];
	int hdr_compression = p_options["compress/hdr_compression"];
	int bptc_ldr = p_options["compress/bptc_ldr"];
	int roughness = p_options["roughness/mode"];
	String normal_map = p_options["roughness/src_normal"];

	Ref<Image> normal_image;
	Image::RoughnessChannel roughness_channel = Image::ROUGHNESS_CHANNEL_R;

	if (mipmaps && roughness > 1 && FileAccess::exists(normal_map)) {
		normal_image.instantiate();
		if (ImageLoader::load_image(normal_map, normal_image) == OK) {
			roughness_channel = Image::RoughnessChannel(roughness - 2);
		}
	}
	Ref<Image> image;
	image.instantiate();
	Error err = ImageLoader::load_image(p_source_file, image, nullptr, hdr_as_srgb, scale);
	if (err != OK) {
		return err;
	}

	Array formats_imported;

	if (size_limit > 0 && (image->get_width() > size_limit || image->get_height() > size_limit)) {
		//limit size
		if (image->get_width() >= image->get_height()) {
			int new_width = size_limit;
			int new_height = image->get_height() * new_width / image->get_width();

			image->resize(new_width, new_height, Image::INTERPOLATE_CUBIC);
		} else {
			int new_height = size_limit;
			int new_width = image->get_width() * new_height / image->get_height();

			image->resize(new_width, new_height, Image::INTERPOLATE_CUBIC);
		}

		if (normal == 1) {
			image->normalize();
		}
	}

	if (fix_alpha_border) {
		image->fix_alpha_edges();
	}

	if (premult_alpha) {
		image->premultiply_alpha();
	}

	if (normal_map_invert_y) {
		// Inverting the green channel can be used to flip a normal map's direction.
		// There's no standard when it comes to normal map Y direction, so this is
		// sometimes needed when using a normal map exported from another program.
		// See <http://wiki.polycount.com/wiki/Normal_Map_Technical_Details#Common_Swizzle_Coordinates>.
		const int height = image->get_height();
		const int width = image->get_width();

		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				const Color color = image->get_pixel(i, j);
				image->set_pixel(i, j, Color(color.r, 1 - color.g, color.b));
			}
		}
	}

	if (compress_mode == COMPRESS_BASIS_UNIVERSAL && image->get_format() >= Image::FORMAT_RF) {
		//basis universal does not support float formats, fall back
		compress_mode = COMPRESS_VRAM_COMPRESSED;
	}

	bool detect_3d = int(p_options["detect_3d/compress_to"]) > 0;
	bool detect_roughness = roughness == 0;
	bool detect_normal = normal == 0;
	bool force_normal = normal == 1;
	bool srgb_friendly_pack = pack_channels == 0;

	if (compress_mode == COMPRESS_VRAM_COMPRESSED) {
		//must import in all formats, in order of priority (so platform choses the best supported one. IE, etc2 over etc).
		//Android, GLES 2.x

		bool ok_on_pc = false;
		bool is_hdr = (image->get_format() >= Image::FORMAT_RF && image->get_format() <= Image::FORMAT_RGBE9995);
		bool is_ldr = (image->get_format() >= Image::FORMAT_L8 && image->get_format() <= Image::FORMAT_RGB565);
		bool can_bptc = ProjectSettings::get_singleton()->get("rendering/textures/vram_compression/import_bptc");
		bool can_s3tc = ProjectSettings::get_singleton()->get("rendering/textures/vram_compression/import_s3tc");

		if (can_bptc) {
			//add to the list anyway
			formats_imported.push_back("bptc");
		}

		bool can_compress_hdr = hdr_compression > 0;
		bool has_alpha = image->detect_alpha() != Image::ALPHA_NONE;

		if (is_hdr && can_compress_hdr) {
			if (has_alpha) {
				//can compress hdr, but hdr with alpha is not compressible
				if (hdr_compression == 2) {
					//but user selected to compress hdr anyway, so force an alpha-less format.
					if (image->get_format() == Image::FORMAT_RGBAF) {
						image->convert(Image::FORMAT_RGBF);
					} else if (image->get_format() == Image::FORMAT_RGBAH) {
						image->convert(Image::FORMAT_RGBH);
					}
				} else {
					can_compress_hdr = false;
				}
			}

			if (can_compress_hdr) {
				if (!can_bptc) {
					//fallback to RGBE99995
					if (image->get_format() != Image::FORMAT_RGBE9995) {
						image->convert(Image::FORMAT_RGBE9995);
					}
				}
			} else {
				can_bptc = false;
			}
		}

		if (is_ldr && can_bptc) {
			if (bptc_ldr == 0 || (bptc_ldr == 1 && !has_alpha)) {
				can_bptc = false;
			}
		}

		if (can_bptc || can_s3tc) {
			_save_stex(image, p_save_path + ".s3tc.stex", compress_mode, lossy, can_bptc ? Image::COMPRESS_BPTC : Image::COMPRESS_S3TC, mipmaps, stream, detect_3d, detect_roughness, detect_normal, force_normal, srgb_friendly_pack, false, mipmap_limit, normal_image, roughness_channel);
			r_platform_variants->push_back("s3tc");
			formats_imported.push_back("s3tc");
			ok_on_pc = true;
		}

		if (ProjectSettings::get_singleton()->get("rendering/textures/vram_compression/import_etc2")) {
			_save_stex(image, p_save_path + ".etc2.stex", compress_mode, lossy, Image::COMPRESS_ETC2, mipmaps, stream, detect_3d, detect_roughness, detect_normal, force_normal, srgb_friendly_pack, true, mipmap_limit, normal_image, roughness_channel);
			r_platform_variants->push_back("etc2");
			formats_imported.push_back("etc2");
		}

		if (ProjectSettings::get_singleton()->get("rendering/textures/vram_compression/import_etc")) {
			_save_stex(image, p_save_path + ".etc.stex", compress_mode, lossy, Image::COMPRESS_ETC, mipmaps, stream, detect_3d, detect_roughness, detect_normal, force_normal, srgb_friendly_pack, true, mipmap_limit, normal_image, roughness_channel);
			r_platform_variants->push_back("etc");
			formats_imported.push_back("etc");
		}

		if (ProjectSettings::get_singleton()->get("rendering/textures/vram_compression/import_pvrtc")) {
			_save_stex(image, p_save_path + ".pvrtc.stex", compress_mode, lossy, Image::COMPRESS_PVRTC1_4, mipmaps, stream, detect_3d, detect_roughness, detect_normal, force_normal, srgb_friendly_pack, true, mipmap_limit, normal_image, roughness_channel);
			r_platform_variants->push_back("pvrtc");
			formats_imported.push_back("pvrtc");
		}

		if (!ok_on_pc) {
			EditorNode::add_io_error("Warning, no suitable PC VRAM compression enabled in Project Settings. This texture will not display correctly on PC.");
		}
	} else {
		//import normally
		_save_stex(image, p_save_path + ".stex", compress_mode, lossy, Image::COMPRESS_S3TC /*this is ignored */, mipmaps, stream, detect_3d, detect_roughness, detect_normal, force_normal, srgb_friendly_pack, false, mipmap_limit, normal_image, roughness_channel);
	}

	if (r_metadata) {
		Dictionary metadata;
		metadata["vram_texture"] = compress_mode == COMPRESS_VRAM_COMPRESSED;
		if (formats_imported.size()) {
			metadata["imported_formats"] = formats_imported;
		}
		*r_metadata = metadata;
	}
	return OK;
}

const char *ResourceImporterTexture::compression_formats[] = {
	"bptc",
	"s3tc",
	"etc",
	"etc2",
	"pvrtc",
	nullptr
};
String ResourceImporterTexture::get_import_settings_string() const {
	String s;

	int index = 0;
	while (compression_formats[index]) {
		String setting_path = "rendering/textures/vram_compression/import_" + String(compression_formats[index]);
		bool test = ProjectSettings::get_singleton()->get(setting_path);
		if (test) {
			s += String(compression_formats[index]);
		}
		index++;
	}

	return s;
}

bool ResourceImporterTexture::are_import_settings_valid(const String &p_path) const {
	//will become invalid if formats are missing to import
	Dictionary metadata = ResourceFormatImporter::get_singleton()->get_resource_metadata(p_path);

	if (!metadata.has("vram_texture")) {
		return false;
	}

	bool vram = metadata["vram_texture"];
	if (!vram) {
		return true; //do not care about non vram
	}

	Vector<String> formats_imported;
	if (metadata.has("imported_formats")) {
		formats_imported = metadata["imported_formats"];
	}

	int index = 0;
	bool valid = true;
	while (compression_formats[index]) {
		String setting_path = "rendering/textures/vram_compression/import_" + String(compression_formats[index]);
		bool test = ProjectSettings::get_singleton()->get(setting_path);
		if (test) {
			if (formats_imported.find(compression_formats[index]) == -1) {
				valid = false;
				break;
			}
		}
		index++;
	}

	return valid;
}

ResourceImporterTexture *ResourceImporterTexture::singleton = nullptr;

ResourceImporterTexture::ResourceImporterTexture() {
	singleton = this;
	StreamTexture2D::request_3d_callback = _texture_reimport_3d;
	StreamTexture2D::request_roughness_callback = _texture_reimport_roughness;
	StreamTexture2D::request_normal_callback = _texture_reimport_normal;
}

ResourceImporterTexture::~ResourceImporterTexture() {
}
