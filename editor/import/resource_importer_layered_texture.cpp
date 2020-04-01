/*************************************************************************/
/*  resource_importer_layered_texture.cpp                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "resource_importer_layered_texture.h"

#include "resource_importer_texture.h"

#include "core/io/config_file.h"
#include "core/io/image_loader.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "scene/resources/texture.h"

#if 0
String ResourceImporterLayeredTexture::get_importer_name() const {

	switch (mode) {
		case MODE_CUBEMAP: {
			return "cubemap_texture";
		} break;
		case MODE_2D_ARRAY: {
			return "2d_array_texture";
		} break;
		case MODE_CUBEMAP_ARRAY: {
			return "cubemap_array_texture";
		} break;
	}

	ERR_FAIL_V("");
}

String ResourceImporterLayeredTexture::get_visible_name() const {

	switch (mode) {
		case MODE_CUBEMAP: {
			return "Cubemap";
		} break;
		case MODE_2D_ARRAY: {
			return "Texture2DArray";
		} break;
		case MODE_CUBEMAP_ARRAY: {
			return "CubemapArray";
		} break;
	}

	ERR_FAIL_V("");
}
void ResourceImporterLayeredTexture::get_recognized_extensions(List<String> *p_extensions) const {

	ImageLoader::get_recognized_extensions(p_extensions);
}
String ResourceImporterLayeredTexture::get_save_extension() const {
	switch (mode) {
		case MODE_CUBEMAP: {
			return "cube";
		} break;
		case MODE_2D_ARRAY: {
			return "tex2darr";
		} break;
		case MODE_CUBEMAP_ARRAY: {
			return "cubearr";
		} break;
	}

	ERR_FAIL_V(String());
}

String ResourceImporterLayeredTexture::get_resource_type() const {

	switch (mode) {
		case MODE_CUBEMAP: {
			return "Cubemap";
		} break;
		case MODE_2D_ARRAY: {
			return "Texture2DArray";
		} break;
		case MODE_CUBEMAP_ARRAY: {
			return "CubemapArray";
		} break;
	}
	ERR_FAIL_V(String());
}

bool ResourceImporterLayeredTexture::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	return true;
}

int ResourceImporterLayeredTexture::get_preset_count() const {
	return 0;
}
String ResourceImporterLayeredTexture::get_preset_name(int p_idx) const {

	return "";
}

void ResourceImporterLayeredTexture::get_import_options(List<ImportOption> *r_options, int p_preset) const {

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "Lossless,Video RAM,Uncompressed", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "compress/no_bptc_if_rgb"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/channel_pack", PROPERTY_HINT_ENUM, "sRGB Friendly,Optimized"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "flags/mipmaps"), true));
	if (mode == MODE_2D_ARRAY) {
		r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slices/horizontal", PROPERTY_HINT_RANGE, "1,256,1"), 8));
	}
	if (mode == MODE_2D_ARRAY || mode == MODE_CUBEMAP_ARRAY) {
		r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "slices/vertical", PROPERTY_HINT_RANGE, "1,256,1"), 8));
	}
}

void ResourceImporterLayeredTexture::_save_tex(const Vector<Ref<Image> > &p_images, const String &p_to_path, int p_compress_mode, Image::CompressMode p_vram_compression, bool p_mipmaps) {

	FileAccess *f = FileAccess::open(p_to_path, FileAccess::WRITE);
	f->store_8('G');
	f->store_8('D');
	switch (mode) {
		case MODE_2D_ARRAY: f->store_8('A'); break;
		case MODE_CUBEMAP: f->store_8('C'); break;
		case MODE_CUBEMAP_ARRAY: f->store_8('X'); break;
	}

	f->store_8('T'); //godot streamable texture

	f->store_32(p_images[0]->get_width());
	f->store_32(p_images[0]->get_height());
	f->store_32(p_images.size()); //depth
	uint32_t flags = 0;
	if (p_mipmaps) {
		flags |= TEXTURE_FLAGS_MIPMAPS;
	}
	f->store_32(flags);
	if (p_compress_mode != COMPRESS_VIDEO_RAM) {
		//vram needs to do a first compression to tell what the format is, for the rest its ok
		f->store_32(p_images[0]->get_format());
		f->store_32(p_compress_mode); // 0 - lossless (PNG), 1 - vram, 2 - uncompressed
	}

	if ((p_compress_mode == COMPRESS_LOSSLESS) && p_images[0]->get_format() > Image::FORMAT_RGBA8) {
		p_compress_mode = COMPRESS_UNCOMPRESSED; //these can't go as lossy
	}

	for (int i = 0; i < p_images.size(); i++) {

		switch (p_compress_mode) {
			case COMPRESS_LOSSLESS: {

				Ref<Image> image = p_images[i]->duplicate();
				if (p_mipmaps) {
					image->generate_mipmaps();
				} else {
					image->clear_mipmaps();
				}

				int mmc = image->get_mipmap_count() + 1;
				f->store_32(mmc);

				for (int j = 0; j < mmc; j++) {

					if (j > 0) {
						image->shrink_x2();
					}

					Vector<uint8_t> data = Image::lossless_packer(image);
					int data_len = data.size();
					f->store_32(data_len);

					const uint8_t* r = data.ptr();
					f->store_buffer(r.ptr(), data_len);
				}

			} break;
			case COMPRESS_VIDEO_RAM: {

				Ref<Image> image = p_images[i]->duplicate();
				image->generate_mipmaps(false);

				Image::CompressSource csource = Image::COMPRESS_SOURCE_LAYERED;
				image->compress(p_vram_compression, csource, 0.7);

				if (i == 0) {
					//hack so we can properly tell the format
					f->store_32(image->get_format());
					f->store_32(p_compress_mode); // 0 - lossless (PNG), 1 - vram, 2 - uncompressed
				}

				Vector<uint8_t> data = image->get_data();
				int dl = data.size();

				const uint8_t* r = data.ptr();
				f->store_buffer(r.ptr(), dl);
			} break;
			case COMPRESS_UNCOMPRESSED: {

				Ref<Image> image = p_images[i]->duplicate();

				if (p_mipmaps) {
					image->generate_mipmaps();
				} else {
					image->clear_mipmaps();
				}

				Vector<uint8_t> data = image->get_data();
				int dl = data.size();

				const uint8_t* r = data.ptr();

				f->store_buffer(r.ptr(), dl);

			} break;
		}
	}

	memdelete(f);
}

Error ResourceImporterLayeredTexture::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) {

	int compress_mode = p_options["compress/mode"];
	int no_bptc_if_rgb = p_options["compress/no_bptc_if_rgb"];
	bool mipmaps = p_options["flags/mipmaps"];
	int channel_pack = p_options["compress/channel_pack"];
	int hslices = (p_options.has("slices/horizontal")) ? int(p_options["slices/horizontal"]) : 0;
	int vslices = (p_options.has("slices/vertical")) ? int(p_options["slices/vertical"]) : 0;

	if (mode == MODE_CUBEMAP) {
		hslices = 3;
		vslices = 2;
	} else if (mode == MODE_CUBEMAP_ARRAY) {
		hslices = 3;
		vslices *= 2; //put cubemaps vertically
	}

	Ref<Image> image;
	image.instance();
	Error err = ImageLoader::load_image(p_source_file, image, nullptr, false, 1.0);
	if (err != OK)
		return err;

	if (compress_mode == COMPRESS_VIDEO_RAM) {
		mipmaps = true;
	}

	Vector<Ref<Image> > slices;

	int slice_w = image->get_width() / hslices;
	int slice_h = image->get_height() / vslices;

	//optimize
	if (compress_mode == COMPRESS_VIDEO_RAM) {
		//if using video ram, optimize
		if (channel_pack == 0) {
			//remove alpha if not needed, so compression is more efficient
			if (image->get_format() == Image::FORMAT_RGBA8 && !image->detect_alpha()) {
				image->convert(Image::FORMAT_RGB8);
			}
		} else {
			image->optimize_channels();
		}
	}

	for (int i = 0; i < vslices; i++) {
		for (int j = 0; j < hslices; j++) {
			int x = slice_w * j;
			int y = slice_h * i;
			Ref<Image> slice = image->get_rect(Rect2(x, y, slice_w, slice_h));
			ERR_CONTINUE(slice.is_null() || slice->empty());
			if (slice->get_width() != slice_w || slice->get_height() != slice_h) {
				slice->resize(slice_w, slice_h);
			}
			slices.push_back(slice);
		}
	}

	String extension = get_save_extension();
	Array formats_imported;

	if (compress_mode == COMPRESS_VIDEO_RAM) {
		//must import in all formats, in order of priority (so platform choses the best supported one. IE, etc2 over etc).
		//Android, GLES 2.x

		bool ok_on_pc = false;
		bool encode_bptc = false;

		if (ProjectSettings::get_singleton()->get("rendering/vram_compression/import_bptc")) {

			encode_bptc = true;

			if (no_bptc_if_rgb) {
				Image::UsedChannels channels = image->detect_used_channels();
				if (channels != Image::USED_CHANNELS_LA && channels != Image::USED_CHANNELS_RGBA) {
					encode_bptc = false;
				}
			}

			formats_imported.push_back("bptc");
		}

		if (encode_bptc) {

			_save_tex(slices, p_save_path + ".bptc." + extension, compress_mode, Image::COMPRESS_BPTC, mipmaps);
			r_platform_variants->push_back("bptc");
			ok_on_pc = true;
		}

		if (ProjectSettings::get_singleton()->get("rendering/vram_compression/import_s3tc")) {

			_save_tex(slices, p_save_path + ".s3tc." + extension, compress_mode, Image::COMPRESS_S3TC, mipmaps);
			r_platform_variants->push_back("s3tc");
			ok_on_pc = true;
			formats_imported.push_back("s3tc");
		}

		if (ProjectSettings::get_singleton()->get("rendering/vram_compression/import_etc2")) {

			_save_tex(slices, p_save_path + ".etc2." + extension, compress_mode, Image::COMPRESS_ETC2, mipmaps);
			r_platform_variants->push_back("etc2");
			formats_imported.push_back("etc2");
		}

		if (ProjectSettings::get_singleton()->get("rendering/vram_compression/import_etc")) {
			_save_tex(slices, p_save_path + ".etc." + extension, compress_mode, Image::COMPRESS_ETC, mipmaps);
			r_platform_variants->push_back("etc");
			formats_imported.push_back("etc");
		}

		if (ProjectSettings::get_singleton()->get("rendering/vram_compression/import_pvrtc")) {

			_save_tex(slices, p_save_path + ".pvrtc." + extension, compress_mode, Image::COMPRESS_PVRTC4, mipmaps);
			r_platform_variants->push_back("pvrtc");
			formats_imported.push_back("pvrtc");
		}

		if (!ok_on_pc) {
			EditorNode::add_io_error("Warning, no suitable PC VRAM compression enabled in Project Settings. This texture will not display correctly on PC.");
		}
	} else {
		//import normally
		_save_tex(slices, p_save_path + "." + extension, compress_mode, Image::COMPRESS_S3TC /*this is ignored */, mipmaps);
	}

	if (r_metadata) {
		Dictionary metadata;
		metadata["vram_texture"] = compress_mode == COMPRESS_VIDEO_RAM;
		if (formats_imported.size()) {
			metadata["imported_formats"] = formats_imported;
		}
		*r_metadata = metadata;
	}

	return OK;
}

const char *ResourceImporterLayeredTexture::compression_formats[] = {
	"bptc",
	"s3tc",
	"etc",
	"etc2",
	"pvrtc",
	nullptr
};
String ResourceImporterLayeredTexture::get_import_settings_string() const {

	String s;

	int index = 0;
	while (compression_formats[index]) {
		String setting_path = "rendering/vram_compression/import_" + String(compression_formats[index]);
		bool test = ProjectSettings::get_singleton()->get(setting_path);
		if (test) {
			s += String(compression_formats[index]);
		}
		index++;
	}

	return s;
}

bool ResourceImporterLayeredTexture::are_import_settings_valid(const String &p_path) const {

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
		String setting_path = "rendering/vram_compression/import_" + String(compression_formats[index]);
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

ResourceImporterLayeredTexture *ResourceImporterLayeredTexture::singleton = nullptr;

ResourceImporterLayeredTexture::ResourceImporterLayeredTexture() {

	singleton = this;
	mode = MODE_CUBEMAP;
}

ResourceImporterLayeredTexture::~ResourceImporterLayeredTexture() {
}
#endif
