/*************************************************************************/
/*  resource_importer_texture.cpp                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "editor/editor_file_system.h"
#include "io/config_file.h"
#include "io/image_loader.h"
#include "scene/resources/texture.h"

void ResourceImporterTexture::_texture_reimport_srgb(const Ref<StreamTexture> &p_tex) {

	singleton->mutex->lock();
	StringName path = p_tex->get_path();

	if (!singleton->make_flags.has(path)) {
		singleton->make_flags[path] = 0;
	}

	singleton->make_flags[path] |= MAKE_SRGB_FLAG;

	print_line("requesting srgb for " + String(path));

	singleton->mutex->unlock();
}

void ResourceImporterTexture::_texture_reimport_3d(const Ref<StreamTexture> &p_tex) {

	singleton->mutex->lock();
	StringName path = p_tex->get_path();

	if (!singleton->make_flags.has(path)) {
		singleton->make_flags[path] = 0;
	}

	singleton->make_flags[path] |= MAKE_3D_FLAG;

	print_line("requesting 3d for " + String(path));

	singleton->mutex->unlock();
}

void ResourceImporterTexture::update_imports() {

	if (EditorFileSystem::get_singleton()->is_scanning() || EditorFileSystem::get_singleton()->is_importing()) {
		return; // do nothing for noe
	}
	mutex->lock();

	if (make_flags.empty()) {
		mutex->unlock();
		return;
	}

	Vector<String> to_reimport;
	for (Map<StringName, int>::Element *E = make_flags.front(); E; E = E->next()) {

		print_line("checking for reimport " + String(E->key()));

		Ref<ConfigFile> cf;
		cf.instance();
		String src_path = String(E->key()) + ".import";

		Error err = cf->load(src_path);
		ERR_CONTINUE(err != OK);

		bool changed = false;
		if (E->get() & MAKE_SRGB_FLAG && int(cf->get_value("params", "flags/srgb")) == 2) {
			cf->set_value("params", "flags/srgb", 1);
			changed = true;
		}

		if (E->get() & MAKE_3D_FLAG && bool(cf->get_value("params", "detect_3d"))) {
			cf->set_value("params", "detect_3d", false);
			cf->set_value("params", "compress/mode", 2);
			cf->set_value("params", "flags/repeat", true);
			cf->set_value("params", "flags/filter", true);
			cf->set_value("params", "flags/mipmaps", true);
			changed = true;
		}

		if (changed) {
			cf->save(src_path);
			to_reimport.push_back(E->key());
		}
	}

	make_flags.clear();

	mutex->unlock();

	if (to_reimport.size()) {
		EditorFileSystem::get_singleton()->reimport_files(to_reimport);
	}
}

String ResourceImporterTexture::get_importer_name() const {

	return "texture";
}

String ResourceImporterTexture::get_visible_name() const {

	return "Texture";
}
void ResourceImporterTexture::get_recognized_extensions(List<String> *p_extensions) const {

	ImageLoader::get_recognized_extensions(p_extensions);
}
String ResourceImporterTexture::get_save_extension() const {
	return "stex";
}

String ResourceImporterTexture::get_resource_type() const {

	return "StreamTexture";
}

bool ResourceImporterTexture::get_option_visibility(const String &p_option, const Map<StringName, Variant> &p_options) const {

	if (p_option == "compress/lossy_quality" && int(p_options["compress/mode"]) != COMPRESS_LOSSY)
		return false;

	return true;
}

int ResourceImporterTexture::get_preset_count() const {
	return 4;
}
String ResourceImporterTexture::get_preset_name(int p_idx) const {

	static const char *preset_names[] = {
		"2D, Detect 3D",
		"2D",
		"2D Pixel",
		"3D"
	};

	return preset_names[p_idx];
}

void ResourceImporterTexture::get_import_options(List<ImportOption> *r_options, int p_preset) const {

	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "compress/mode", PROPERTY_HINT_ENUM, "Lossless,Lossy,Video RAM,Uncompressed", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), p_preset == PRESET_3D ? 2 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL, "compress/lossy_quality", PROPERTY_HINT_RANGE, "0,1,0.01"), 0.7));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "flags/repeat", PROPERTY_HINT_ENUM, "Disabled,Enabled,Mirrored"), p_preset == PRESET_3D ? 1 : 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "flags/filter"), p_preset == PRESET_2D_PIXEL ? false : true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "flags/mipmaps"), p_preset == PRESET_3D ? true : false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "flags/anisotropic"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "flags/srgb", PROPERTY_HINT_ENUM, "Disable,Enable,Detect"), 2));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/fix_alpha_border"), p_preset != PRESET_3D ? true : false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "process/premult_alpha"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "stream"), false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "size_limit", PROPERTY_HINT_RANGE, "0,4096,1"), 0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "detect_3d"), p_preset == PRESET_DETECT));
}

void ResourceImporterTexture::_save_stex(const Image &p_image, const String &p_to_path, int p_compress_mode, float p_lossy_quality, Image::CompressMode p_vram_compression, bool p_mipmaps, int p_texture_flags, bool p_streamable, bool p_detect_3d, bool p_detect_srgb) {

	FileAccess *f = FileAccess::open(p_to_path, FileAccess::WRITE);
	f->store_8('G');
	f->store_8('D');
	f->store_8('S');
	f->store_8('T'); //godot streamable texture

	f->store_32(p_image.get_width());
	f->store_32(p_image.get_height());
	f->store_32(p_texture_flags);

	uint32_t format = 0;

	if (p_streamable)
		format |= StreamTexture::FORMAT_BIT_STREAM;
	if (p_mipmaps || p_compress_mode == COMPRESS_VIDEO_RAM) //VRAM always uses mipmaps
		format |= StreamTexture::FORMAT_BIT_HAS_MIPMAPS; //mipmaps bit
	if (p_detect_3d)
		format |= StreamTexture::FORMAT_BIT_DETECT_3D;
	if (p_detect_srgb)
		format |= StreamTexture::FORMAT_BIT_DETECT_SRGB;

	switch (p_compress_mode) {
		case COMPRESS_LOSSLESS: {

			Image image = p_image;
			if (p_mipmaps) {
				image.generate_mipmaps();
			} else {
				image.clear_mipmaps();
			}

			int mmc = image.get_mipmap_count() + 1;

			format |= StreamTexture::FORMAT_BIT_LOSSLESS;
			f->store_32(format);
			f->store_32(mmc);

			for (int i = 0; i < mmc; i++) {

				if (i > 0) {
					image.shrink_x2();
				}

				PoolVector<uint8_t> data = Image::lossless_packer(image);
				int data_len = data.size();
				f->store_32(data_len);

				PoolVector<uint8_t>::Read r = data.read();
				f->store_buffer(r.ptr(), data_len);
			}

		} break;
		case COMPRESS_LOSSY: {
			Image image = p_image;
			if (p_mipmaps) {
				image.generate_mipmaps();
			} else {
				image.clear_mipmaps();
			}

			int mmc = image.get_mipmap_count() + 1;

			format |= StreamTexture::FORMAT_BIT_LOSSY;
			f->store_32(format);
			f->store_32(mmc);

			for (int i = 0; i < mmc; i++) {

				if (i > 0) {
					image.shrink_x2();
				}

				PoolVector<uint8_t> data = Image::lossy_packer(image, p_lossy_quality);
				int data_len = data.size();
				f->store_32(data_len);

				PoolVector<uint8_t>::Read r = data.read();
				f->store_buffer(r.ptr(), data_len);
			}
		} break;
		case COMPRESS_VIDEO_RAM: {

			Image image = p_image;
			image.generate_mipmaps();
			image.compress(p_vram_compression);

			format |= image.get_format();

			f->store_32(format);

			PoolVector<uint8_t> data = image.get_data();
			int dl = data.size();
			PoolVector<uint8_t>::Read r = data.read();
			f->store_buffer(r.ptr(), dl);

		} break;
		case COMPRESS_UNCOMPRESSED: {

			Image image = p_image;
			if (p_mipmaps) {
				image.generate_mipmaps();
			} else {
				image.clear_mipmaps();
			}

			format |= image.get_format();
			f->store_32(format);

			PoolVector<uint8_t> data = image.get_data();
			int dl = data.size();
			PoolVector<uint8_t>::Read r = data.read();

			f->store_buffer(r.ptr(), dl);

		} break;
	}

	memdelete(f);
}

Error ResourceImporterTexture::import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files) {

	int compress_mode = p_options["compress/mode"];
	float lossy = p_options["compress/lossy_quality"];
	int repeat = p_options["flags/repeat"];
	bool filter = p_options["flags/filter"];
	bool mipmaps = p_options["flags/mipmaps"];
	bool anisotropic = p_options["flags/anisotropic"];
	int srgb = p_options["flags/srgb"];
	bool fix_alpha_border = p_options["process/fix_alpha_border"];
	bool premult_alpha = p_options["process/premult_alpha"];
	bool stream = p_options["stream"];
	int size_limit = p_options["size_limit"];

	Image image;
	Error err = ImageLoader::load_image(p_source_file, &image);
	if (err != OK)
		return err;

	int tex_flags = 0;
	if (repeat > 0)
		tex_flags |= Texture::FLAG_REPEAT;
	if (repeat == 2)
		tex_flags |= Texture::FLAG_MIRRORED_REPEAT;
	if (filter)
		tex_flags |= Texture::FLAG_FILTER;
	if (mipmaps || compress_mode == COMPRESS_VIDEO_RAM)
		tex_flags |= Texture::FLAG_MIPMAPS;
	if (anisotropic)
		tex_flags |= Texture::FLAG_ANISOTROPIC_FILTER;
	if (srgb == 1)
		tex_flags |= Texture::FLAG_CONVERT_TO_LINEAR;

	if (size_limit > 0 && (image.get_width() > size_limit || image.get_height() > size_limit)) {
		//limit size
		if (image.get_width() >= image.get_height()) {
			int new_width = size_limit;
			int new_height = image.get_height() * new_width / image.get_width();

			image.resize(new_width, new_height, Image::INTERPOLATE_CUBIC);
		} else {

			int new_height = size_limit;
			int new_width = image.get_width() * new_height / image.get_height();

			image.resize(new_width, new_height, Image::INTERPOLATE_CUBIC);
		}
	}

	if (fix_alpha_border) {
		image.fix_alpha_edges();
	}

	if (premult_alpha) {
		image.premultiply_alpha();
	}

	bool detect_3d = p_options["detect_3d"];
	bool detect_srgb = srgb == 2;

	if (compress_mode == COMPRESS_VIDEO_RAM) {
		//must import in all formats
		//Android, GLES 2.x
		_save_stex(image, p_save_path + ".etc.stex", compress_mode, lossy, Image::COMPRESS_ETC, mipmaps, tex_flags, stream, detect_3d, detect_srgb);
		r_platform_variants->push_back("etc");
		//_save_stex(image,p_save_path+".etc2.stex",compress_mode,lossy,Image::COMPRESS_ETC2,mipmaps,tex_flags,stream);
		//r_platform_variants->push_back("etc2");
		_save_stex(image, p_save_path + ".s3tc.stex", compress_mode, lossy, Image::COMPRESS_S3TC, mipmaps, tex_flags, stream, detect_3d, detect_srgb);
		r_platform_variants->push_back("s3tc");

	} else {
		//import normally
		_save_stex(image, p_save_path + ".stex", compress_mode, lossy, Image::COMPRESS_16BIT /*this is ignored */, mipmaps, tex_flags, stream, detect_3d, detect_srgb);
	}

	return OK;
}

ResourceImporterTexture *ResourceImporterTexture::singleton = NULL;

ResourceImporterTexture::ResourceImporterTexture() {

	singleton = this;
	StreamTexture::request_3d_callback = _texture_reimport_3d;
	StreamTexture::request_srgb_callback = _texture_reimport_srgb;
	mutex = Mutex::create();
}

ResourceImporterTexture::~ResourceImporterTexture() {

	memdelete(mutex);
}
