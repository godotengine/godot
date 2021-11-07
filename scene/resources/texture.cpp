/*************************************************************************/
/*  texture.cpp                                                          */
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

#include "texture.h"

#include "core/core_string_names.h"
#include "core/io/image_loader.h"
#include "core/math/geometry_2d.h"
#include "core/os/os.h"
#include "mesh.h"
#include "scene/resources/bit_map.h"
#include "servers/camera/camera_feed.h"

Size2 Texture2D::get_size() const {
	return Size2(get_width(), get_height());
}

bool Texture2D::is_pixel_opaque(int p_x, int p_y) const {
	return true;
}

void Texture2D::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, get_size()), get_rid(), false, p_modulate, p_transpose);
}

void Texture2D::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, get_rid(), p_tile, p_modulate, p_transpose);
}

void Texture2D::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, get_rid(), p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

bool Texture2D::get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const {
	r_rect = p_rect;
	r_src_rect = p_src_rect;
	return true;
}

void Texture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_width"), &Texture2D::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &Texture2D::get_height);
	ClassDB::bind_method(D_METHOD("get_size"), &Texture2D::get_size);
	ClassDB::bind_method(D_METHOD("has_alpha"), &Texture2D::has_alpha);
	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "position", "modulate", "transpose"), &Texture2D::draw, DEFVAL(Color(1, 1, 1)), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_rect", "canvas_item", "rect", "tile", "modulate", "transpose"), &Texture2D::draw_rect, DEFVAL(Color(1, 1, 1)), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("draw_rect_region", "canvas_item", "rect", "src_rect", "modulate", "transpose", "clip_uv"), &Texture2D::draw_rect_region, DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_image"), &Texture2D::get_image);

	ADD_GROUP("", "");
}

Texture2D::Texture2D() {
}

/////////////////////

void ImageTexture::reload_from_file() {
	String path = ResourceLoader::path_remap(get_path());
	if (!path.is_resource_file()) {
		return;
	}

	Ref<Image> img;
	img.instantiate();

	if (ImageLoader::load_image(path, img) == OK) {
		create_from_image(img);
	} else {
		Resource::reload_from_file();
		notify_property_list_changed();
		emit_changed();
	}
}

bool ImageTexture::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "image") {
		create_from_image(p_value);
	} else if (p_name == "size") {
		Size2 s = p_value;
		w = s.width;
		h = s.height;
		RenderingServer::get_singleton()->texture_set_size_override(texture, w, h);
	} else {
		return false;
	}

	return true;
}

bool ImageTexture::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "image") {
		r_ret = get_image();
	} else if (p_name == "size") {
		r_ret = Size2(w, h);
	} else {
		return false;
	}

	return true;
}

void ImageTexture::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::OBJECT, "image", PROPERTY_HINT_RESOURCE_TYPE, "Image", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT));
	p_list->push_back(PropertyInfo(Variant::VECTOR2, "size", PROPERTY_HINT_NONE, ""));
}

void ImageTexture::create_from_image(const Ref<Image> &p_image) {
	ERR_FAIL_COND_MSG(p_image.is_null() || p_image->is_empty(), "Invalid image");
	w = p_image->get_width();
	h = p_image->get_height();
	format = p_image->get_format();
	mipmaps = p_image->has_mipmaps();

	if (texture.is_null()) {
		texture = RenderingServer::get_singleton()->texture_2d_create(p_image);
	} else {
		RID new_texture = RenderingServer::get_singleton()->texture_2d_create(p_image);
		RenderingServer::get_singleton()->texture_replace(texture, new_texture);
	}
	notify_property_list_changed();
	emit_changed();

	image_stored = true;
}

Image::Format ImageTexture::get_format() const {
	return format;
}

void ImageTexture::update(const Ref<Image> &p_image) {
	ERR_FAIL_COND_MSG(p_image.is_null(), "Invalid image");
	ERR_FAIL_COND_MSG(texture.is_null(), "Texture is not initialized.");
	ERR_FAIL_COND_MSG(p_image->get_width() != w || p_image->get_height() != h,
			"The new image dimensions must match the texture size.");
	ERR_FAIL_COND_MSG(p_image->get_format() != format,
			"The new image format must match the texture's image format.");
	ERR_FAIL_COND_MSG(mipmaps != p_image->has_mipmaps(),
			"The new image mipmaps configuration must match the texture's image mipmaps configuration");

	RenderingServer::get_singleton()->texture_2d_update(texture, p_image);

	notify_property_list_changed();
	emit_changed();

	alpha_cache.unref();
	image_stored = true;
}

Ref<Image> ImageTexture::get_image() const {
	if (image_stored) {
		return RenderingServer::get_singleton()->texture_2d_get(texture);
	} else {
		return Ref<Image>();
	}
}

int ImageTexture::get_width() const {
	return w;
}

int ImageTexture::get_height() const {
	return h;
}

RID ImageTexture::get_rid() const {
	if (texture.is_null()) {
		//we are in trouble, create something temporary
		texture = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

bool ImageTexture::has_alpha() const {
	return (format == Image::FORMAT_LA8 || format == Image::FORMAT_RGBA8);
}

void ImageTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, Size2(w, h)), texture, false, p_modulate, p_transpose);
}

void ImageTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, texture, p_tile, p_modulate, p_transpose);
}

void ImageTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, texture, p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

bool ImageTexture::is_pixel_opaque(int p_x, int p_y) const {
	if (!alpha_cache.is_valid()) {
		Ref<Image> img = get_image();
		if (img.is_valid()) {
			if (img->is_compressed()) { //must decompress, if compressed
				Ref<Image> decom = img->duplicate();
				decom->decompress();
				img = decom;
			}
			alpha_cache.instantiate();
			alpha_cache->create_from_image_alpha(img);
		}
	}

	if (alpha_cache.is_valid()) {
		int aw = int(alpha_cache->get_size().width);
		int ah = int(alpha_cache->get_size().height);
		if (aw == 0 || ah == 0) {
			return true;
		}

		int x = p_x * aw / w;
		int y = p_y * ah / h;

		x = CLAMP(x, 0, aw);
		y = CLAMP(y, 0, ah);

		return alpha_cache->get_bit(Point2(x, y));
	}

	return true;
}

void ImageTexture::set_size_override(const Size2 &p_size) {
	Size2 s = p_size;
	if (s.x != 0) {
		w = s.x;
	}
	if (s.y != 0) {
		h = s.y;
	}
	RenderingServer::get_singleton()->texture_set_size_override(texture, w, h);
}

void ImageTexture::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

void ImageTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_from_image", "image"), &ImageTexture::create_from_image);
	ClassDB::bind_method(D_METHOD("get_format"), &ImageTexture::get_format);

	ClassDB::bind_method(D_METHOD("update", "image"), &ImageTexture::update);
	ClassDB::bind_method(D_METHOD("set_size_override", "size"), &ImageTexture::set_size_override);
}

ImageTexture::ImageTexture() {}

ImageTexture::~ImageTexture() {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->free(texture);
	}
}

//////////////////////////////////////////

Ref<Image> StreamTexture2D::load_image_from_file(FileAccess *f, int p_size_limit) {
	uint32_t data_format = f->get_32();
	uint32_t w = f->get_16();
	uint32_t h = f->get_16();
	uint32_t mipmaps = f->get_32();
	Image::Format format = Image::Format(f->get_32());

	if (data_format == DATA_FORMAT_PNG || data_format == DATA_FORMAT_WEBP || data_format == DATA_FORMAT_BASIS_UNIVERSAL) {
		//look for a PNG or WEBP file inside

		int sw = w;
		int sh = h;

		//mipmaps need to be read independently, they will be later combined
		Vector<Ref<Image>> mipmap_images;
		uint64_t total_size = 0;

		bool first = true;

		for (uint32_t i = 0; i < mipmaps + 1; i++) {
			uint32_t size = f->get_32();

			if (p_size_limit > 0 && i < (mipmaps - 1) && (sw > p_size_limit || sh > p_size_limit)) {
				//can't load this due to size limit
				sw = MAX(sw >> 1, 1);
				sh = MAX(sh >> 1, 1);
				f->seek(f->get_position() + size);
				continue;
			}

			Vector<uint8_t> pv;
			pv.resize(size);
			{
				uint8_t *wr = pv.ptrw();
				f->get_buffer(wr, size);
			}

			Ref<Image> img;
			if (data_format == DATA_FORMAT_BASIS_UNIVERSAL && Image::basis_universal_unpacker) {
				img = Image::basis_universal_unpacker(pv);
			} else if (data_format == DATA_FORMAT_PNG && Image::png_unpacker) {
				img = Image::png_unpacker(pv);
			} else if (data_format == DATA_FORMAT_WEBP && Image::webp_unpacker) {
				img = Image::webp_unpacker(pv);
			}

			if (img.is_null() || img->is_empty()) {
				ERR_FAIL_COND_V(img.is_null() || img->is_empty(), Ref<Image>());
			}

			if (first) {
				//format will actually be the format of the first image,
				//as it may have changed on compression
				format = img->get_format();
				first = false;
			} else if (img->get_format() != format) {
				img->convert(format); //all needs to be the same format
			}

			total_size += img->get_data().size();

			mipmap_images.push_back(img);

			sw = MAX(sw >> 1, 1);
			sh = MAX(sh >> 1, 1);
		}

		//print_line("mipmap read total: " + itos(mipmap_images.size()));

		Ref<Image> image;
		image.instantiate();

		if (mipmap_images.size() == 1) {
			//only one image (which will most likely be the case anyway for this format)
			image = mipmap_images[0];
			return image;

		} else {
			//rarer use case, but needs to be supported
			Vector<uint8_t> img_data;
			img_data.resize(total_size);

			{
				uint8_t *wr = img_data.ptrw();

				int ofs = 0;
				for (int i = 0; i < mipmap_images.size(); i++) {
					Vector<uint8_t> id = mipmap_images[i]->get_data();
					int len = id.size();
					const uint8_t *r = id.ptr();
					memcpy(&wr[ofs], r, len);
					ofs += len;
				}
			}

			image->create(w, h, true, mipmap_images[0]->get_format(), img_data);
			return image;
		}

	} else if (data_format == DATA_FORMAT_IMAGE) {
		int size = Image::get_image_data_size(w, h, format, mipmaps ? true : false);

		for (uint32_t i = 0; i < mipmaps + 1; i++) {
			int tw, th;
			int ofs = Image::get_image_mipmap_offset_and_dimensions(w, h, format, i, tw, th);

			if (p_size_limit > 0 && i < mipmaps && (p_size_limit > tw || p_size_limit > th)) {
				if (ofs) {
					f->seek(f->get_position() + ofs);
				}
				continue; //oops, size limit enforced, go to next
			}

			Vector<uint8_t> data;
			data.resize(size - ofs);

			{
				uint8_t *wr = data.ptrw();
				f->get_buffer(wr, data.size());
			}

			Ref<Image> image;
			image.instantiate();

			image->create(tw, th, mipmaps - i ? true : false, format, data);

			return image;
		}
	}

	return Ref<Image>();
}

void StreamTexture2D::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

void StreamTexture2D::_requested_3d(void *p_ud) {
	StreamTexture2D *st = (StreamTexture2D *)p_ud;
	Ref<StreamTexture2D> stex(st);
	ERR_FAIL_COND(!request_3d_callback);
	request_3d_callback(stex);
}

void StreamTexture2D::_requested_roughness(void *p_ud, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_roughness_channel) {
	StreamTexture2D *st = (StreamTexture2D *)p_ud;
	Ref<StreamTexture2D> stex(st);
	ERR_FAIL_COND(!request_roughness_callback);
	request_roughness_callback(stex, p_normal_path, p_roughness_channel);
}

void StreamTexture2D::_requested_normal(void *p_ud) {
	StreamTexture2D *st = (StreamTexture2D *)p_ud;
	Ref<StreamTexture2D> stex(st);
	ERR_FAIL_COND(!request_normal_callback);
	request_normal_callback(stex);
}

StreamTexture2D::TextureFormatRequestCallback StreamTexture2D::request_3d_callback = nullptr;
StreamTexture2D::TextureFormatRoughnessRequestCallback StreamTexture2D::request_roughness_callback = nullptr;
StreamTexture2D::TextureFormatRequestCallback StreamTexture2D::request_normal_callback = nullptr;

Image::Format StreamTexture2D::get_format() const {
	return format;
}

Error StreamTexture2D::_load_data(const String &p_path, int &r_width, int &r_height, Ref<Image> &image, bool &r_request_3d, bool &r_request_normal, bool &r_request_roughness, int &mipmap_limit, int p_size_limit) {
	alpha_cache.unref();

	ERR_FAIL_COND_V(image.is_null(), ERR_INVALID_PARAMETER);

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!f, ERR_CANT_OPEN, vformat("Unable to open file: %s.", p_path));

	uint8_t header[4];
	f->get_buffer(header, 4);
	if (header[0] != 'G' || header[1] != 'S' || header[2] != 'T' || header[3] != '2') {
		memdelete(f);
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Stream texture file is corrupt (Bad header).");
	}

	uint32_t version = f->get_32();

	if (version > FORMAT_VERSION) {
		memdelete(f);
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Stream texture file is too new.");
	}
	r_width = f->get_32();
	r_height = f->get_32();
	uint32_t df = f->get_32(); //data format

	//skip reserved
	mipmap_limit = int(f->get_32());
	//reserved
	f->get_32();
	f->get_32();
	f->get_32();

#ifdef TOOLS_ENABLED

	r_request_3d = request_3d_callback && df & FORMAT_BIT_DETECT_3D;
	r_request_roughness = request_roughness_callback && df & FORMAT_BIT_DETECT_ROUGNESS;
	r_request_normal = request_normal_callback && df & FORMAT_BIT_DETECT_NORMAL;

#else

	r_request_3d = false;
	r_request_roughness = false;
	r_request_normal = false;

#endif
	if (!(df & FORMAT_BIT_STREAM)) {
		p_size_limit = 0;
	}

	image = load_image_from_file(f, p_size_limit);

	memdelete(f);

	if (image.is_null() || image->is_empty()) {
		return ERR_CANT_OPEN;
	}

	return OK;
}

Error StreamTexture2D::load(const String &p_path) {
	int lw, lh;
	Ref<Image> image;
	image.instantiate();

	bool request_3d;
	bool request_normal;
	bool request_roughness;
	int mipmap_limit;

	Error err = _load_data(p_path, lw, lh, image, request_3d, request_normal, request_roughness, mipmap_limit);
	if (err) {
		return err;
	}

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_create(image);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_2d_create(image);
	}
	if (lw || lh) {
		RS::get_singleton()->texture_set_size_override(texture, lw, lh);
	}

	w = lw;
	h = lh;
	path_to_file = p_path;
	format = image->get_format();

	if (get_path() == String()) {
		//temporarily set path if no path set for resource, helps find errors
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

#ifdef TOOLS_ENABLED

	if (request_3d) {
		//print_line("request detect 3D at " + p_path);
		RS::get_singleton()->texture_set_detect_3d_callback(texture, _requested_3d, this);
	} else {
		//print_line("not requesting detect 3D at " + p_path);
		RS::get_singleton()->texture_set_detect_3d_callback(texture, nullptr, nullptr);
	}

	if (request_roughness) {
		//print_line("request detect srgb at " + p_path);
		RS::get_singleton()->texture_set_detect_roughness_callback(texture, _requested_roughness, this);
	} else {
		//print_line("not requesting detect srgb at " + p_path);
		RS::get_singleton()->texture_set_detect_roughness_callback(texture, nullptr, nullptr);
	}

	if (request_normal) {
		//print_line("request detect srgb at " + p_path);
		RS::get_singleton()->texture_set_detect_normal_callback(texture, _requested_normal, this);
	} else {
		//print_line("not requesting detect normal at " + p_path);
		RS::get_singleton()->texture_set_detect_normal_callback(texture, nullptr, nullptr);
	}

#endif
	notify_property_list_changed();
	emit_changed();
	return OK;
}

String StreamTexture2D::get_load_path() const {
	return path_to_file;
}

int StreamTexture2D::get_width() const {
	return w;
}

int StreamTexture2D::get_height() const {
	return h;
}

RID StreamTexture2D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

void StreamTexture2D::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, Size2(w, h)), texture, false, p_modulate, p_transpose);
}

void StreamTexture2D::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, texture, p_tile, p_modulate, p_transpose);
}

void StreamTexture2D::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, texture, p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

bool StreamTexture2D::has_alpha() const {
	return false;
}

Ref<Image> StreamTexture2D::get_image() const {
	if (texture.is_valid()) {
		return RS::get_singleton()->texture_2d_get(texture);
	} else {
		return Ref<Image>();
	}
}

bool StreamTexture2D::is_pixel_opaque(int p_x, int p_y) const {
	if (!alpha_cache.is_valid()) {
		Ref<Image> img = get_image();
		if (img.is_valid()) {
			if (img->is_compressed()) { //must decompress, if compressed
				Ref<Image> decom = img->duplicate();
				decom->decompress();
				img = decom;
			}

			alpha_cache.instantiate();
			alpha_cache->create_from_image_alpha(img);
		}
	}

	if (alpha_cache.is_valid()) {
		int aw = int(alpha_cache->get_size().width);
		int ah = int(alpha_cache->get_size().height);
		if (aw == 0 || ah == 0) {
			return true;
		}

		int x = p_x * aw / w;
		int y = p_y * ah / h;

		x = CLAMP(x, 0, aw);
		y = CLAMP(y, 0, ah);

		return alpha_cache->get_bit(Point2(x, y));
	}

	return true;
}

void StreamTexture2D::reload_from_file() {
	String path = get_path();
	if (!path.is_resource_file()) {
		return;
	}

	path = ResourceLoader::path_remap(path); //remap for translation
	path = ResourceLoader::import_remap(path); //remap for import
	if (!path.is_resource_file()) {
		return;
	}

	load(path);
}

void StreamTexture2D::_validate_property(PropertyInfo &property) const {
}

void StreamTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load", "path"), &StreamTexture2D::load);
	ClassDB::bind_method(D_METHOD("get_load_path"), &StreamTexture2D::get_load_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "load_path", PROPERTY_HINT_FILE, "*.stex"), "load", "get_load_path");
}

StreamTexture2D::StreamTexture2D() {}

StreamTexture2D::~StreamTexture2D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

RES ResourceFormatLoaderStreamTexture2D::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<StreamTexture2D> st;
	st.instantiate();
	Error err = st->load(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		return RES();
	}

	return st;
}

void ResourceFormatLoaderStreamTexture2D::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("stex");
}

bool ResourceFormatLoaderStreamTexture2D::handles_type(const String &p_type) const {
	return p_type == "StreamTexture2D";
}

String ResourceFormatLoaderStreamTexture2D::get_resource_type(const String &p_path) const {
	if (p_path.get_extension().to_lower() == "stex") {
		return "StreamTexture2D";
	}
	return "";
}

////////////////////////////////////

TypedArray<Image> Texture3D::_get_data() const {
	Vector<Ref<Image>> data = get_data();

	TypedArray<Image> ret;
	ret.resize(data.size());
	for (int i = 0; i < data.size(); i++) {
		ret[i] = data[i];
	}
	return ret;
}

void Texture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_format"), &Texture3D::get_format);
	ClassDB::bind_method(D_METHOD("get_width"), &Texture3D::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &Texture3D::get_height);
	ClassDB::bind_method(D_METHOD("get_depth"), &Texture3D::get_depth);
	ClassDB::bind_method(D_METHOD("has_mipmaps"), &Texture3D::has_mipmaps);
	ClassDB::bind_method(D_METHOD("get_data"), &Texture3D::_get_data);
}
//////////////////////////////////////////

Image::Format ImageTexture3D::get_format() const {
	return format;
}
int ImageTexture3D::get_width() const {
	return width;
}
int ImageTexture3D::get_height() const {
	return height;
}
int ImageTexture3D::get_depth() const {
	return depth;
}
bool ImageTexture3D::has_mipmaps() const {
	return mipmaps;
}

Error ImageTexture3D::_create(Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const TypedArray<Image> &p_data) {
	Vector<Ref<Image>> images;
	images.resize(p_data.size());
	for (int i = 0; i < images.size(); i++) {
		images.write[i] = p_data[i];
	}
	return create(p_format, p_width, p_height, p_depth, p_mipmaps, images);
}

void ImageTexture3D::_update(const TypedArray<Image> &p_data) {
	Vector<Ref<Image>> images;
	images.resize(p_data.size());
	for (int i = 0; i < images.size(); i++) {
		images.write[i] = p_data[i];
	}
	return update(images);
}

Error ImageTexture3D::create(Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data) {
	RID tex = RenderingServer::get_singleton()->texture_3d_create(p_format, p_width, p_height, p_depth, p_mipmaps, p_data);
	ERR_FAIL_COND_V(tex.is_null(), ERR_CANT_CREATE);

	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_replace(texture, tex);
	}

	return OK;
}

void ImageTexture3D::update(const Vector<Ref<Image>> &p_data) {
	ERR_FAIL_COND(!texture.is_valid());
	RenderingServer::get_singleton()->texture_3d_update(texture, p_data);
}

Vector<Ref<Image>> ImageTexture3D::get_data() const {
	ERR_FAIL_COND_V(!texture.is_valid(), Vector<Ref<Image>>());
	return RS::get_singleton()->texture_3d_get(texture);
}

RID ImageTexture3D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_3d_placeholder_create();
	}
	return texture;
}
void ImageTexture3D::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

void ImageTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create", "format", "width", "height", "depth", "use_mipmaps", "data"), &ImageTexture3D::_create);
	ClassDB::bind_method(D_METHOD("update", "data"), &ImageTexture3D::_update);
}

ImageTexture3D::ImageTexture3D() {
}

ImageTexture3D::~ImageTexture3D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

////////////////////////////////////////////

void StreamTexture3D::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

Image::Format StreamTexture3D::get_format() const {
	return format;
}

Error StreamTexture3D::_load_data(const String &p_path, Vector<Ref<Image>> &r_data, Image::Format &r_format, int &r_width, int &r_height, int &r_depth, bool &r_mipmaps) {
	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!f, ERR_CANT_OPEN, vformat("Unable to open file: %s.", p_path));

	uint8_t header[4];
	f->get_buffer(header, 4);
	ERR_FAIL_COND_V(header[0] != 'G' || header[1] != 'S' || header[2] != 'T' || header[3] != 'L', ERR_FILE_UNRECOGNIZED);

	//stored as stream textures (used for lossless and lossy compression)
	uint32_t version = f->get_32();

	if (version > FORMAT_VERSION) {
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Stream texture file is too new.");
	}

	r_depth = f->get_32(); //depth
	f->get_32(); //ignored (mode)
	f->get_32(); // ignored (data format)

	f->get_32(); //ignored
	int mipmaps = f->get_32();
	f->get_32(); //ignored
	f->get_32(); //ignored

	r_mipmaps = mipmaps != 0;

	r_data.clear();

	for (int i = 0; i < (r_depth + mipmaps); i++) {
		Ref<Image> image = StreamTexture2D::load_image_from_file(f, 0);
		ERR_FAIL_COND_V(image.is_null() || image->is_empty(), ERR_CANT_OPEN);
		if (i == 0) {
			r_format = image->get_format();
			r_width = image->get_width();
			r_height = image->get_height();
		}
		r_data.push_back(image);
	}

	return OK;
}

Error StreamTexture3D::load(const String &p_path) {
	Vector<Ref<Image>> data;

	int tw, th, td;
	Image::Format tfmt;
	bool tmm;

	Error err = _load_data(p_path, data, tfmt, tw, th, td, tmm);
	if (err) {
		return err;
	}

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_3d_create(tfmt, tw, th, td, tmm, data);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_3d_create(tfmt, tw, th, td, tmm, data);
	}

	w = tw;
	h = th;
	d = td;
	mipmaps = tmm;
	format = tfmt;

	path_to_file = p_path;

	if (get_path() == String()) {
		//temporarily set path if no path set for resource, helps find errors
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	notify_property_list_changed();
	emit_changed();
	return OK;
}

String StreamTexture3D::get_load_path() const {
	return path_to_file;
}

int StreamTexture3D::get_width() const {
	return w;
}

int StreamTexture3D::get_height() const {
	return h;
}

int StreamTexture3D::get_depth() const {
	return d;
}

bool StreamTexture3D::has_mipmaps() const {
	return mipmaps;
}

RID StreamTexture3D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_3d_placeholder_create();
	}
	return texture;
}

Vector<Ref<Image>> StreamTexture3D::get_data() const {
	if (texture.is_valid()) {
		return RS::get_singleton()->texture_3d_get(texture);
	} else {
		return Vector<Ref<Image>>();
	}
}

void StreamTexture3D::reload_from_file() {
	String path = get_path();
	if (!path.is_resource_file()) {
		return;
	}

	path = ResourceLoader::path_remap(path); //remap for translation
	path = ResourceLoader::import_remap(path); //remap for import
	if (!path.is_resource_file()) {
		return;
	}

	load(path);
}

void StreamTexture3D::_validate_property(PropertyInfo &property) const {
}

void StreamTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load", "path"), &StreamTexture3D::load);
	ClassDB::bind_method(D_METHOD("get_load_path"), &StreamTexture3D::get_load_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "load_path", PROPERTY_HINT_FILE, "*.stex"), "load", "get_load_path");
}

StreamTexture3D::StreamTexture3D() {}

StreamTexture3D::~StreamTexture3D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

/////////////////////////////

RES ResourceFormatLoaderStreamTexture3D::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<StreamTexture3D> st;
	st.instantiate();
	Error err = st->load(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		return RES();
	}

	return st;
}

void ResourceFormatLoaderStreamTexture3D::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("stex3d");
}

bool ResourceFormatLoaderStreamTexture3D::handles_type(const String &p_type) const {
	return p_type == "StreamTexture3D";
}

String ResourceFormatLoaderStreamTexture3D::get_resource_type(const String &p_path) const {
	if (p_path.get_extension().to_lower() == "stex3d") {
		return "StreamTexture3D";
	}
	return "";
}

////////////////////////////////////////////

int AtlasTexture::get_width() const {
	if (region.size.width == 0) {
		if (atlas.is_valid()) {
			return atlas->get_width();
		}
		return 1;
	} else {
		return region.size.width + margin.size.width;
	}
}

int AtlasTexture::get_height() const {
	if (region.size.height == 0) {
		if (atlas.is_valid()) {
			return atlas->get_height();
		}
		return 1;
	} else {
		return region.size.height + margin.size.height;
	}
}

RID AtlasTexture::get_rid() const {
	if (atlas.is_valid()) {
		return atlas->get_rid();
	}

	return RID();
}

bool AtlasTexture::has_alpha() const {
	if (atlas.is_valid()) {
		return atlas->has_alpha();
	}

	return false;
}

void AtlasTexture::set_atlas(const Ref<Texture2D> &p_atlas) {
	ERR_FAIL_COND(p_atlas == this);
	if (atlas == p_atlas) {
		return;
	}
	atlas = p_atlas;
	emit_changed();
}

Ref<Texture2D> AtlasTexture::get_atlas() const {
	return atlas;
}

void AtlasTexture::set_region(const Rect2 &p_region) {
	if (region == p_region) {
		return;
	}
	region = p_region;
	emit_changed();
}

Rect2 AtlasTexture::get_region() const {
	return region;
}

void AtlasTexture::set_margin(const Rect2 &p_margin) {
	if (margin == p_margin) {
		return;
	}
	margin = p_margin;
	emit_changed();
}

Rect2 AtlasTexture::get_margin() const {
	return margin;
}

void AtlasTexture::set_filter_clip(const bool p_enable) {
	filter_clip = p_enable;
	emit_changed();
}

bool AtlasTexture::has_filter_clip() const {
	return filter_clip;
}

void AtlasTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_atlas", "atlas"), &AtlasTexture::set_atlas);
	ClassDB::bind_method(D_METHOD("get_atlas"), &AtlasTexture::get_atlas);

	ClassDB::bind_method(D_METHOD("set_region", "region"), &AtlasTexture::set_region);
	ClassDB::bind_method(D_METHOD("get_region"), &AtlasTexture::get_region);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &AtlasTexture::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &AtlasTexture::get_margin);

	ClassDB::bind_method(D_METHOD("set_filter_clip", "enable"), &AtlasTexture::set_filter_clip);
	ClassDB::bind_method(D_METHOD("has_filter_clip"), &AtlasTexture::has_filter_clip);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "atlas", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_atlas", "get_atlas");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "region"), "set_region", "get_region");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "margin"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_clip"), "set_filter_clip", "has_filter_clip");
}

void AtlasTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if (!atlas.is_valid()) {
		return;
	}

	Rect2 rc = region;

	if (rc.size.width == 0) {
		rc.size.width = atlas->get_width();
	}

	if (rc.size.height == 0) {
		rc.size.height = atlas->get_height();
	}

	RS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, Rect2(p_pos + margin.position, rc.size), atlas->get_rid(), rc, p_modulate, p_transpose, filter_clip);
}

void AtlasTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if (!atlas.is_valid()) {
		return;
	}

	Rect2 rc = region;

	if (rc.size.width == 0) {
		rc.size.width = atlas->get_width();
	}

	if (rc.size.height == 0) {
		rc.size.height = atlas->get_height();
	}

	Vector2 scale = p_rect.size / (region.size + margin.size);
	Rect2 dr(p_rect.position + margin.position * scale, rc.size * scale);

	RS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, dr, atlas->get_rid(), rc, p_modulate, p_transpose, filter_clip);
}

void AtlasTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	//this might not necessarily work well if using a rect, needs to be fixed properly
	if (!atlas.is_valid()) {
		return;
	}

	Rect2 dr;
	Rect2 src_c;
	get_rect_region(p_rect, p_src_rect, dr, src_c);

	RS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, dr, atlas->get_rid(), src_c, p_modulate, p_transpose, filter_clip);
}

bool AtlasTexture::get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const {
	if (!atlas.is_valid()) {
		return false;
	}

	Rect2 rc = region;

	Rect2 src = p_src_rect;
	if (src.size == Size2()) {
		src.size = rc.size;
	}
	Vector2 scale = p_rect.size / src.size;

	src.position += (rc.position - margin.position);
	Rect2 src_c = rc.intersection(src);
	if (src_c.size == Size2()) {
		return false;
	}
	Vector2 ofs = (src_c.position - src.position);

	if (scale.x < 0) {
		float mx = (margin.size.width - margin.position.x);
		mx -= margin.position.x;
		ofs.x = -(ofs.x + mx);
	}
	if (scale.y < 0) {
		float my = margin.size.height - margin.position.y;
		my -= margin.position.y;
		ofs.y = -(ofs.y + my);
	}
	Rect2 dr(p_rect.position + ofs * scale, src_c.size * scale);

	r_rect = dr;
	r_src_rect = src_c;
	return true;
}

bool AtlasTexture::is_pixel_opaque(int p_x, int p_y) const {
	if (!atlas.is_valid()) {
		return true;
	}

	int x = p_x + region.position.x - margin.position.x;
	int y = p_y + region.position.y - margin.position.y;

	// margin edge may outside of atlas
	if (x < 0 || x >= atlas->get_width()) {
		return false;
	}
	if (y < 0 || y >= atlas->get_height()) {
		return false;
	}

	return atlas->is_pixel_opaque(x, y);
}

Ref<Image> AtlasTexture::get_image() const {
	if (!atlas.is_valid() || !atlas->get_image().is_valid()) {
		return Ref<Image>();
	}

	return atlas->get_image()->get_rect(region);
}

AtlasTexture::AtlasTexture() {}

/////////////////////////////////////////

int MeshTexture::get_width() const {
	return size.width;
}

int MeshTexture::get_height() const {
	return size.height;
}

RID MeshTexture::get_rid() const {
	return RID();
}

bool MeshTexture::has_alpha() const {
	return false;
}

void MeshTexture::set_mesh(const Ref<Mesh> &p_mesh) {
	mesh = p_mesh;
}

Ref<Mesh> MeshTexture::get_mesh() const {
	return mesh;
}

void MeshTexture::set_image_size(const Size2 &p_size) {
	size = p_size;
}

Size2 MeshTexture::get_image_size() const {
	return size;
}

void MeshTexture::set_base_texture(const Ref<Texture2D> &p_texture) {
	base_texture = p_texture;
}

Ref<Texture2D> MeshTexture::get_base_texture() const {
	return base_texture;
}

void MeshTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if (mesh.is_null() || base_texture.is_null()) {
		return;
	}
	Transform2D xform;
	xform.set_origin(p_pos);
	if (p_transpose) {
		SWAP(xform.elements[0][1], xform.elements[1][0]);
		SWAP(xform.elements[0][0], xform.elements[1][1]);
	}
	RenderingServer::get_singleton()->canvas_item_add_mesh(p_canvas_item, mesh->get_rid(), xform, p_modulate, base_texture->get_rid());
}

void MeshTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if (mesh.is_null() || base_texture.is_null()) {
		return;
	}
	Transform2D xform;
	Vector2 origin = p_rect.position;
	if (p_rect.size.x < 0) {
		origin.x += size.x;
	}
	if (p_rect.size.y < 0) {
		origin.y += size.y;
	}
	xform.set_origin(origin);
	xform.set_scale(p_rect.size / size);

	if (p_transpose) {
		SWAP(xform.elements[0][1], xform.elements[1][0]);
		SWAP(xform.elements[0][0], xform.elements[1][1]);
	}
	RenderingServer::get_singleton()->canvas_item_add_mesh(p_canvas_item, mesh->get_rid(), xform, p_modulate, base_texture->get_rid());
}

void MeshTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if (mesh.is_null() || base_texture.is_null()) {
		return;
	}
	Transform2D xform;
	Vector2 origin = p_rect.position;
	if (p_rect.size.x < 0) {
		origin.x += size.x;
	}
	if (p_rect.size.y < 0) {
		origin.y += size.y;
	}
	xform.set_origin(origin);
	xform.set_scale(p_rect.size / size);

	if (p_transpose) {
		SWAP(xform.elements[0][1], xform.elements[1][0]);
		SWAP(xform.elements[0][0], xform.elements[1][1]);
	}
	RenderingServer::get_singleton()->canvas_item_add_mesh(p_canvas_item, mesh->get_rid(), xform, p_modulate, base_texture->get_rid());
}

bool MeshTexture::get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const {
	r_rect = p_rect;
	r_src_rect = p_src_rect;
	return true;
}

bool MeshTexture::is_pixel_opaque(int p_x, int p_y) const {
	return true;
}

void MeshTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MeshTexture::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MeshTexture::get_mesh);
	ClassDB::bind_method(D_METHOD("set_image_size", "size"), &MeshTexture::set_image_size);
	ClassDB::bind_method(D_METHOD("get_image_size"), &MeshTexture::get_image_size);
	ClassDB::bind_method(D_METHOD("set_base_texture", "texture"), &MeshTexture::set_base_texture);
	ClassDB::bind_method(D_METHOD("get_base_texture"), &MeshTexture::get_base_texture);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "base_texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_base_texture", "get_base_texture");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "image_size", PROPERTY_HINT_RANGE, "0,16384,1"), "set_image_size", "get_image_size");
}

MeshTexture::MeshTexture() {
}

//////////////////////////////////////////

void CurveTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &CurveTexture::set_width);

	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &CurveTexture::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &CurveTexture::get_curve);

	ClassDB::bind_method(D_METHOD("set_texture_mode", "texture_mode"), &CurveTexture::set_texture_mode);
	ClassDB::bind_method(D_METHOD("get_texture_mode"), &CurveTexture::get_texture_mode);

	ClassDB::bind_method(D_METHOD("_update"), &CurveTexture::_update);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,4096"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "texture_mode", PROPERTY_HINT_ENUM, "RGB,Red"), "set_texture_mode", "get_texture_mode");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve", "get_curve");

	BIND_ENUM_CONSTANT(TEXTURE_MODE_RGB);
	BIND_ENUM_CONSTANT(TEXTURE_MODE_RED);
}

void CurveTexture::set_width(int p_width) {
	ERR_FAIL_COND(p_width < 32 || p_width > 4096);

	if (_width == p_width) {
		return;
	}

	_width = p_width;
	_update();
}

int CurveTexture::get_width() const {
	return _width;
}

void CurveTexture::ensure_default_setup(float p_min, float p_max) {
	if (_curve.is_null()) {
		Ref<Curve> curve = Ref<Curve>(memnew(Curve));
		curve->add_point(Vector2(0, 1));
		curve->add_point(Vector2(1, 1));
		curve->set_min_value(p_min);
		curve->set_max_value(p_max);
		set_curve(curve);
		// Min and max is 0..1 by default
	}
}

void CurveTexture::set_curve(Ref<Curve> p_curve) {
	if (_curve != p_curve) {
		if (_curve.is_valid()) {
			_curve->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveTexture::_update));
		}
		_curve = p_curve;
		if (_curve.is_valid()) {
			_curve->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveTexture::_update));
		}
		_update();
	}
}

void CurveTexture::_update() {
	Vector<uint8_t> data;
	data.resize(_width * sizeof(float) * (texture_mode == TEXTURE_MODE_RGB ? 3 : 1));

	// The array is locked in that scope
	{
		uint8_t *wd8 = data.ptrw();
		float *wd = (float *)wd8;

		if (_curve.is_valid()) {
			Curve &curve = **_curve;
			for (int i = 0; i < _width; ++i) {
				float t = i / static_cast<float>(_width);
				if (texture_mode == TEXTURE_MODE_RGB) {
					wd[i * 3 + 0] = curve.interpolate_baked(t);
					wd[i * 3 + 1] = wd[i * 3 + 0];
					wd[i * 3 + 2] = wd[i * 3 + 0];
				} else {
					wd[i] = curve.interpolate_baked(t);
				}
			}

		} else {
			for (int i = 0; i < _width; ++i) {
				if (texture_mode == TEXTURE_MODE_RGB) {
					wd[i * 3 + 0] = 0;
					wd[i * 3 + 1] = 0;
					wd[i * 3 + 2] = 0;
				} else {
					wd[i] = 0;
				}
			}
		}
	}

	Ref<Image> image = memnew(Image(_width, 1, false, texture_mode == TEXTURE_MODE_RGB ? Image::FORMAT_RGBF : Image::FORMAT_RF, data));

	if (_texture.is_valid()) {
		if (_current_texture_mode != texture_mode || _current_width != _width) {
			RID new_texture = RS::get_singleton()->texture_2d_create(image);
			RS::get_singleton()->texture_replace(_texture, new_texture);
		} else {
			RS::get_singleton()->texture_2d_update(_texture, image);
		}
	} else {
		_texture = RS::get_singleton()->texture_2d_create(image);
	}
	_current_texture_mode = texture_mode;
	_current_width = _width;

	emit_changed();
}

Ref<Curve> CurveTexture::get_curve() const {
	return _curve;
}

void CurveTexture::set_texture_mode(TextureMode p_mode) {
	ERR_FAIL_COND(p_mode < TEXTURE_MODE_RGB || p_mode > TEXTURE_MODE_RED);
	if (texture_mode == p_mode) {
		return;
	}
	texture_mode = p_mode;
	_update();
}
CurveTexture::TextureMode CurveTexture::get_texture_mode() const {
	return texture_mode;
}

RID CurveTexture::get_rid() const {
	if (!_texture.is_valid()) {
		_texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return _texture;
}

CurveTexture::CurveTexture() {}

CurveTexture::~CurveTexture() {
	if (_texture.is_valid()) {
		RS::get_singleton()->free(_texture);
	}
}

//////////////////

void CurveXYZTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &CurveXYZTexture::set_width);

	ClassDB::bind_method(D_METHOD("set_curve_x", "curve"), &CurveXYZTexture::set_curve_x);
	ClassDB::bind_method(D_METHOD("get_curve_x"), &CurveXYZTexture::get_curve_x);

	ClassDB::bind_method(D_METHOD("set_curve_y", "curve"), &CurveXYZTexture::set_curve_y);
	ClassDB::bind_method(D_METHOD("get_curve_y"), &CurveXYZTexture::get_curve_y);

	ClassDB::bind_method(D_METHOD("set_curve_z", "curve"), &CurveXYZTexture::set_curve_z);
	ClassDB::bind_method(D_METHOD("get_curve_z"), &CurveXYZTexture::get_curve_z);

	ClassDB::bind_method(D_METHOD("_update"), &CurveXYZTexture::_update);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,4096"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve_x", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve_x", "get_curve_x");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve_y", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve_y", "get_curve_y");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve_z", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve_z", "get_curve_z");
}

void CurveXYZTexture::set_width(int p_width) {
	ERR_FAIL_COND(p_width < 32 || p_width > 4096);

	if (_width == p_width) {
		return;
	}

	_width = p_width;
	_update();
}

int CurveXYZTexture::get_width() const {
	return _width;
}

void CurveXYZTexture::ensure_default_setup(float p_min, float p_max) {
	if (_curve_x.is_null()) {
		Ref<Curve> curve = Ref<Curve>(memnew(Curve));
		curve->add_point(Vector2(0, 1));
		curve->add_point(Vector2(1, 1));
		curve->set_min_value(p_min);
		curve->set_max_value(p_max);
		set_curve_x(curve);
	}

	if (_curve_y.is_null()) {
		Ref<Curve> curve = Ref<Curve>(memnew(Curve));
		curve->add_point(Vector2(0, 1));
		curve->add_point(Vector2(1, 1));
		curve->set_min_value(p_min);
		curve->set_max_value(p_max);
		set_curve_y(curve);
	}

	if (_curve_z.is_null()) {
		Ref<Curve> curve = Ref<Curve>(memnew(Curve));
		curve->add_point(Vector2(0, 1));
		curve->add_point(Vector2(1, 1));
		curve->set_min_value(p_min);
		curve->set_max_value(p_max);
		set_curve_z(curve);
	}
}

void CurveXYZTexture::set_curve_x(Ref<Curve> p_curve) {
	if (_curve_x != p_curve) {
		if (_curve_x.is_valid()) {
			_curve_x->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveXYZTexture::_update));
		}
		_curve_x = p_curve;
		if (_curve_x.is_valid()) {
			_curve_x->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveXYZTexture::_update), varray(), CONNECT_REFERENCE_COUNTED);
		}
		_update();
	}
}

void CurveXYZTexture::set_curve_y(Ref<Curve> p_curve) {
	if (_curve_y != p_curve) {
		if (_curve_y.is_valid()) {
			_curve_y->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveXYZTexture::_update));
		}
		_curve_y = p_curve;
		if (_curve_y.is_valid()) {
			_curve_y->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveXYZTexture::_update), varray(), CONNECT_REFERENCE_COUNTED);
		}
		_update();
	}
}

void CurveXYZTexture::set_curve_z(Ref<Curve> p_curve) {
	if (_curve_z != p_curve) {
		if (_curve_z.is_valid()) {
			_curve_z->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveXYZTexture::_update));
		}
		_curve_z = p_curve;
		if (_curve_z.is_valid()) {
			_curve_z->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &CurveXYZTexture::_update), varray(), CONNECT_REFERENCE_COUNTED);
		}
		_update();
	}
}

void CurveXYZTexture::_update() {
	Vector<uint8_t> data;
	data.resize(_width * sizeof(float) * 3);

	// The array is locked in that scope
	{
		uint8_t *wd8 = data.ptrw();
		float *wd = (float *)wd8;

		if (_curve_x.is_valid()) {
			Curve &curve_x = **_curve_x;
			for (int i = 0; i < _width; ++i) {
				float t = i / static_cast<float>(_width);
				wd[i * 3 + 0] = curve_x.interpolate_baked(t);
			}

		} else {
			for (int i = 0; i < _width; ++i) {
				wd[i * 3 + 0] = 0;
			}
		}

		if (_curve_y.is_valid()) {
			Curve &curve_y = **_curve_y;
			for (int i = 0; i < _width; ++i) {
				float t = i / static_cast<float>(_width);
				wd[i * 3 + 1] = curve_y.interpolate_baked(t);
			}

		} else {
			for (int i = 0; i < _width; ++i) {
				wd[i * 3 + 1] = 0;
			}
		}

		if (_curve_z.is_valid()) {
			Curve &curve_z = **_curve_z;
			for (int i = 0; i < _width; ++i) {
				float t = i / static_cast<float>(_width);
				wd[i * 3 + 2] = curve_z.interpolate_baked(t);
			}

		} else {
			for (int i = 0; i < _width; ++i) {
				wd[i * 3 + 2] = 0;
			}
		}
	}

	Ref<Image> image = memnew(Image(_width, 1, false, Image::FORMAT_RGBF, data));

	if (_texture.is_valid()) {
		if (_current_width != _width) {
			RID new_texture = RS::get_singleton()->texture_2d_create(image);
			RS::get_singleton()->texture_replace(_texture, new_texture);
		} else {
			RS::get_singleton()->texture_2d_update(_texture, image);
		}
	} else {
		_texture = RS::get_singleton()->texture_2d_create(image);
	}
	_current_width = _width;

	emit_changed();
}

Ref<Curve> CurveXYZTexture::get_curve_x() const {
	return _curve_x;
}

Ref<Curve> CurveXYZTexture::get_curve_y() const {
	return _curve_y;
}

Ref<Curve> CurveXYZTexture::get_curve_z() const {
	return _curve_z;
}

RID CurveXYZTexture::get_rid() const {
	if (!_texture.is_valid()) {
		_texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return _texture;
}

CurveXYZTexture::CurveXYZTexture() {}

CurveXYZTexture::~CurveXYZTexture() {
	if (_texture.is_valid()) {
		RS::get_singleton()->free(_texture);
	}
}

//////////////////

GradientTexture1D::GradientTexture1D() {
	_queue_update();
}

GradientTexture1D::~GradientTexture1D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

void GradientTexture1D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gradient", "gradient"), &GradientTexture1D::set_gradient);
	ClassDB::bind_method(D_METHOD("get_gradient"), &GradientTexture1D::get_gradient);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &GradientTexture1D::set_width);
	// The `get_width()` method is already exposed by the parent class Texture2D.

	ClassDB::bind_method(D_METHOD("set_use_hdr", "enabled"), &GradientTexture1D::set_use_hdr);
	ClassDB::bind_method(D_METHOD("is_using_hdr"), &GradientTexture1D::is_using_hdr);

	ClassDB::bind_method(D_METHOD("_update"), &GradientTexture1D::_update);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gradient", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_gradient", "get_gradient");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,4096"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_hdr"), "set_use_hdr", "is_using_hdr");
}

void GradientTexture1D::set_gradient(Ref<Gradient> p_gradient) {
	if (p_gradient == gradient) {
		return;
	}
	if (gradient.is_valid()) {
		gradient->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &GradientTexture1D::_update));
	}
	gradient = p_gradient;
	if (gradient.is_valid()) {
		gradient->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &GradientTexture1D::_update));
	}
	_update();
	emit_changed();
}

Ref<Gradient> GradientTexture1D::get_gradient() const {
	return gradient;
}

void GradientTexture1D::_queue_update() {
	if (update_pending) {
		return;
	}

	update_pending = true;
	call_deferred(SNAME("_update"));
}

void GradientTexture1D::_update() {
	update_pending = false;

	if (gradient.is_null()) {
		return;
	}

	if (use_hdr) {
		// High dynamic range.
		Ref<Image> image = memnew(Image(width, 1, false, Image::FORMAT_RGBAF));
		Gradient &g = **gradient;
		// `create()` isn't available for non-uint8_t data, so fill in the data manually.
		for (int i = 0; i < width; i++) {
			float ofs = float(i) / (width - 1);
			image->set_pixel(i, 0, g.get_color_at_offset(ofs));
		}

		if (texture.is_valid()) {
			RID new_texture = RS::get_singleton()->texture_2d_create(image);
			RS::get_singleton()->texture_replace(texture, new_texture);
		} else {
			texture = RS::get_singleton()->texture_2d_create(image);
		}
	} else {
		// Low dynamic range. "Overbright" colors will be clamped.
		Vector<uint8_t> data;
		data.resize(width * 4);
		{
			uint8_t *wd8 = data.ptrw();
			Gradient &g = **gradient;

			for (int i = 0; i < width; i++) {
				float ofs = float(i) / (width - 1);
				Color color = g.get_color_at_offset(ofs);

				wd8[i * 4 + 0] = uint8_t(CLAMP(color.r * 255.0, 0, 255));
				wd8[i * 4 + 1] = uint8_t(CLAMP(color.g * 255.0, 0, 255));
				wd8[i * 4 + 2] = uint8_t(CLAMP(color.b * 255.0, 0, 255));
				wd8[i * 4 + 3] = uint8_t(CLAMP(color.a * 255.0, 0, 255));
			}
		}

		Ref<Image> image = memnew(Image(width, 1, false, Image::FORMAT_RGBA8, data));

		if (texture.is_valid()) {
			RID new_texture = RS::get_singleton()->texture_2d_create(image);
			RS::get_singleton()->texture_replace(texture, new_texture);
		} else {
			texture = RS::get_singleton()->texture_2d_create(image);
		}
	}

	emit_changed();
}

void GradientTexture1D::set_width(int p_width) {
	ERR_FAIL_COND(p_width <= 0);
	width = p_width;
	_queue_update();
}

int GradientTexture1D::get_width() const {
	return width;
}

void GradientTexture1D::set_use_hdr(bool p_enabled) {
	if (p_enabled == use_hdr) {
		return;
	}

	use_hdr = p_enabled;
	_queue_update();
}

bool GradientTexture1D::is_using_hdr() const {
	return use_hdr;
}

Ref<Image> GradientTexture1D::get_image() const {
	if (!texture.is_valid()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(texture);
}

GradientTexture2D::GradientTexture2D() {
	_queue_update();
}

GradientTexture2D::~GradientTexture2D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

void GradientTexture2D::set_gradient(Ref<Gradient> p_gradient) {
	if (gradient == p_gradient) {
		return;
	}
	if (gradient.is_valid()) {
		gradient->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &GradientTexture2D::_queue_update));
	}
	gradient = p_gradient;
	if (gradient.is_valid()) {
		gradient->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &GradientTexture2D::_queue_update));
	}
	_queue_update();
}

Ref<Gradient> GradientTexture2D::get_gradient() const {
	return gradient;
}

void GradientTexture2D::_queue_update() {
	if (update_pending) {
		return;
	}
	update_pending = true;
	call_deferred("_update");
}

void GradientTexture2D::_update() {
	update_pending = false;

	if (gradient.is_null()) {
		return;
	}
	Ref<Image> image;
	image.instantiate();

	if (gradient->get_points_count() <= 1) { // No need to interpolate.
		image->create(width, height, false, (use_hdr) ? Image::FORMAT_RGBAF : Image::FORMAT_RGBA8);
		image->fill((gradient->get_points_count() == 1) ? gradient->get_color(0) : Color(0, 0, 0, 1));
	} else {
		if (use_hdr) {
			image->create(width, height, false, Image::FORMAT_RGBAF);
			Gradient &g = **gradient;
			// `create()` isn't available for non-uint8_t data, so fill in the data manually.
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					float ofs = _get_gradient_offset_at(x, y);
					image->set_pixel(x, y, g.get_color_at_offset(ofs));
				}
			}
		} else {
			Vector<uint8_t> data;
			data.resize(width * height * 4);
			{
				uint8_t *wd8 = data.ptrw();
				Gradient &g = **gradient;
				for (int y = 0; y < height; y++) {
					for (int x = 0; x < width; x++) {
						float ofs = _get_gradient_offset_at(x, y);
						const Color &c = g.get_color_at_offset(ofs);

						wd8[(x + (y * width)) * 4 + 0] = uint8_t(CLAMP(c.r * 255.0, 0, 255));
						wd8[(x + (y * width)) * 4 + 1] = uint8_t(CLAMP(c.g * 255.0, 0, 255));
						wd8[(x + (y * width)) * 4 + 2] = uint8_t(CLAMP(c.b * 255.0, 0, 255));
						wd8[(x + (y * width)) * 4 + 3] = uint8_t(CLAMP(c.a * 255.0, 0, 255));
					}
				}
			}
			image->create(width, height, false, Image::FORMAT_RGBA8, data);
		}
	}

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_create(image);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_2d_create(image);
	}
	emit_changed();
}

float GradientTexture2D::_get_gradient_offset_at(int x, int y) const {
	if (fill_to == fill_from) {
		return 0;
	}
	float ofs = 0;
	Vector2 pos;
	if (width > 1) {
		pos.x = static_cast<float>(x) / (width - 1);
	}
	if (height > 1) {
		pos.y = static_cast<float>(y) / (height - 1);
	}
	if (fill == Fill::FILL_LINEAR) {
		Vector2 segment[2];
		segment[0] = fill_from;
		segment[1] = fill_to;
		Vector2 closest = Geometry2D::get_closest_point_to_segment_uncapped(pos, &segment[0]);
		ofs = (closest - fill_from).length() / (fill_to - fill_from).length();
		if ((closest - fill_from).dot(fill_to - fill_from) < 0) {
			ofs *= -1;
		}
	} else if (fill == Fill::FILL_RADIAL) {
		ofs = (pos - fill_from).length() / (fill_to - fill_from).length();
	}
	if (repeat == Repeat::REPEAT_NONE) {
		ofs = CLAMP(ofs, 0.0, 1.0);
	} else if (repeat == Repeat::REPEAT) {
		ofs = Math::fmod(ofs, 1.0f);
		if (ofs < 0) {
			ofs = 1 + ofs;
		}
	} else if (repeat == Repeat::REPEAT_MIRROR) {
		ofs = Math::abs(ofs);
		ofs = Math::fmod(ofs, 2.0f);
		if (ofs > 1.0) {
			ofs = 2.0 - ofs;
		}
	}
	return ofs;
}

void GradientTexture2D::set_width(int p_width) {
	width = p_width;
	_queue_update();
}

int GradientTexture2D::get_width() const {
	return width;
}

void GradientTexture2D::set_height(int p_height) {
	height = p_height;
	_queue_update();
}
int GradientTexture2D::get_height() const {
	return height;
}

void GradientTexture2D::set_use_hdr(bool p_enabled) {
	if (p_enabled == use_hdr) {
		return;
	}

	use_hdr = p_enabled;
	_queue_update();
}

bool GradientTexture2D::is_using_hdr() const {
	return use_hdr;
}

void GradientTexture2D::set_fill_from(Vector2 p_fill_from) {
	fill_from = p_fill_from;
	_queue_update();
}

Vector2 GradientTexture2D::get_fill_from() const {
	return fill_from;
}

void GradientTexture2D::set_fill_to(Vector2 p_fill_to) {
	fill_to = p_fill_to;
	_queue_update();
}

Vector2 GradientTexture2D::get_fill_to() const {
	return fill_to;
}

void GradientTexture2D::set_fill(Fill p_fill) {
	fill = p_fill;
	_queue_update();
}

GradientTexture2D::Fill GradientTexture2D::get_fill() const {
	return fill;
}

void GradientTexture2D::set_repeat(Repeat p_repeat) {
	repeat = p_repeat;
	_queue_update();
}

GradientTexture2D::Repeat GradientTexture2D::get_repeat() const {
	return repeat;
}

RID GradientTexture2D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

Ref<Image> GradientTexture2D::get_image() const {
	if (!texture.is_valid()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(texture);
}

void GradientTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gradient", "gradient"), &GradientTexture2D::set_gradient);
	ClassDB::bind_method(D_METHOD("get_gradient"), &GradientTexture2D::get_gradient);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &GradientTexture2D::set_width);
	ClassDB::bind_method(D_METHOD("set_height", "height"), &GradientTexture2D::set_height);

	ClassDB::bind_method(D_METHOD("set_use_hdr", "enabled"), &GradientTexture2D::set_use_hdr);
	ClassDB::bind_method(D_METHOD("is_using_hdr"), &GradientTexture2D::is_using_hdr);

	ClassDB::bind_method(D_METHOD("set_fill", "fill"), &GradientTexture2D::set_fill);
	ClassDB::bind_method(D_METHOD("get_fill"), &GradientTexture2D::get_fill);
	ClassDB::bind_method(D_METHOD("set_fill_from", "fill_from"), &GradientTexture2D::set_fill_from);
	ClassDB::bind_method(D_METHOD("get_fill_from"), &GradientTexture2D::get_fill_from);
	ClassDB::bind_method(D_METHOD("set_fill_to", "fill_to"), &GradientTexture2D::set_fill_to);
	ClassDB::bind_method(D_METHOD("get_fill_to"), &GradientTexture2D::get_fill_to);

	ClassDB::bind_method(D_METHOD("set_repeat", "repeat"), &GradientTexture2D::set_repeat);
	ClassDB::bind_method(D_METHOD("get_repeat"), &GradientTexture2D::get_repeat);

	ClassDB::bind_method(D_METHOD("_update"), &GradientTexture2D::_update);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gradient", PROPERTY_HINT_RESOURCE_TYPE, "Gradient", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_EDITOR_INSTANTIATE_OBJECT), "set_gradient", "get_gradient");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,1,or_greater"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "height", PROPERTY_HINT_RANGE, "1,2048,1,or_greater"), "set_height", "get_height");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_hdr"), "set_use_hdr", "is_using_hdr");

	ADD_GROUP("Fill", "fill_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "fill", PROPERTY_HINT_ENUM, "Linear,Radial"), "set_fill", "get_fill");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "fill_from"), "set_fill_from", "get_fill_from");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "fill_to"), "set_fill_to", "get_fill_to");

	ADD_GROUP("Repeat", "repeat_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "repeat", PROPERTY_HINT_ENUM, "No Repeat,Repeat,Mirror Repeat"), "set_repeat", "get_repeat");

	BIND_ENUM_CONSTANT(FILL_LINEAR);
	BIND_ENUM_CONSTANT(FILL_RADIAL);

	BIND_ENUM_CONSTANT(REPEAT_NONE);
	BIND_ENUM_CONSTANT(REPEAT);
	BIND_ENUM_CONSTANT(REPEAT_MIRROR);
}

//////////////////////////////////////

void ProxyTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_base", "base"), &ProxyTexture::set_base);
	ClassDB::bind_method(D_METHOD("get_base"), &ProxyTexture::get_base);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "base", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D"), "set_base", "get_base");
}

void ProxyTexture::set_base(const Ref<Texture2D> &p_texture) {
	ERR_FAIL_COND(p_texture == this);

	base = p_texture;
	if (base.is_valid()) {
		if (proxy_ph.is_valid()) {
			RS::get_singleton()->texture_proxy_update(proxy, base->get_rid());
			RS::get_singleton()->free(proxy_ph);
			proxy_ph = RID();
		} else if (proxy.is_valid()) {
			RS::get_singleton()->texture_proxy_update(proxy, base->get_rid());
		} else {
			proxy = RS::get_singleton()->texture_proxy_create(base->get_rid());
		}
	}
}

Ref<Texture2D> ProxyTexture::get_base() const {
	return base;
}

int ProxyTexture::get_width() const {
	if (base.is_valid()) {
		return base->get_width();
	}
	return 1;
}

int ProxyTexture::get_height() const {
	if (base.is_valid()) {
		return base->get_height();
	}
	return 1;
}

RID ProxyTexture::get_rid() const {
	if (proxy.is_null()) {
		proxy_ph = RS::get_singleton()->texture_2d_placeholder_create();
		proxy = RS::get_singleton()->texture_proxy_create(proxy_ph);
	}
	return proxy;
}

bool ProxyTexture::has_alpha() const {
	if (base.is_valid()) {
		return base->has_alpha();
	}
	return false;
}

ProxyTexture::ProxyTexture() {
	//proxy = RS::get_singleton()->texture_create();
}

ProxyTexture::~ProxyTexture() {
	if (proxy_ph.is_valid()) {
		RS::get_singleton()->free(proxy_ph);
	}
	if (proxy.is_valid()) {
		RS::get_singleton()->free(proxy);
	}
}

//////////////////////////////////////////////

void AnimatedTexture::_update_proxy() {
	RWLockRead r(rw_lock);

	float delta;
	if (prev_ticks == 0) {
		delta = 0;
		prev_ticks = OS::get_singleton()->get_ticks_usec();
	} else {
		uint64_t ticks = OS::get_singleton()->get_ticks_usec();
		delta = float(double(ticks - prev_ticks) / 1000000.0);
		prev_ticks = ticks;
	}

	time += delta;

	float limit;

	if (fps == 0) {
		limit = 0;
	} else {
		limit = 1.0 / fps;
	}

	int iter_max = frame_count;
	while (iter_max && !pause) {
		float frame_limit = limit + frames[current_frame].delay_sec;

		if (time > frame_limit) {
			current_frame++;
			if (current_frame >= frame_count) {
				if (oneshot) {
					current_frame = frame_count - 1;
				} else {
					current_frame = 0;
				}
			}
			time -= frame_limit;

		} else {
			break;
		}
		iter_max--;
	}

	if (frames[current_frame].texture.is_valid()) {
		RenderingServer::get_singleton()->texture_proxy_update(proxy, frames[current_frame].texture->get_rid());
	}
}

void AnimatedTexture::set_frames(int p_frames) {
	ERR_FAIL_COND(p_frames < 1 || p_frames > MAX_FRAMES);

	RWLockWrite r(rw_lock);

	frame_count = p_frames;
}

int AnimatedTexture::get_frames() const {
	return frame_count;
}

void AnimatedTexture::set_current_frame(int p_frame) {
	ERR_FAIL_COND(p_frame < 0 || p_frame >= frame_count);

	RWLockWrite r(rw_lock);

	current_frame = p_frame;
}

int AnimatedTexture::get_current_frame() const {
	return current_frame;
}

void AnimatedTexture::set_pause(bool p_pause) {
	RWLockWrite r(rw_lock);
	pause = p_pause;
}

bool AnimatedTexture::get_pause() const {
	return pause;
}

void AnimatedTexture::set_oneshot(bool p_oneshot) {
	RWLockWrite r(rw_lock);
	oneshot = p_oneshot;
}

bool AnimatedTexture::get_oneshot() const {
	return oneshot;
}

void AnimatedTexture::set_frame_texture(int p_frame, const Ref<Texture2D> &p_texture) {
	ERR_FAIL_COND(p_texture == this);
	ERR_FAIL_INDEX(p_frame, MAX_FRAMES);

	RWLockWrite w(rw_lock);

	frames[p_frame].texture = p_texture;
}

Ref<Texture2D> AnimatedTexture::get_frame_texture(int p_frame) const {
	ERR_FAIL_INDEX_V(p_frame, MAX_FRAMES, Ref<Texture2D>());

	RWLockRead r(rw_lock);

	return frames[p_frame].texture;
}

void AnimatedTexture::set_frame_delay(int p_frame, float p_delay_sec) {
	ERR_FAIL_INDEX(p_frame, MAX_FRAMES);

	RWLockRead r(rw_lock);

	frames[p_frame].delay_sec = p_delay_sec;
}

float AnimatedTexture::get_frame_delay(int p_frame) const {
	ERR_FAIL_INDEX_V(p_frame, MAX_FRAMES, 0);

	RWLockRead r(rw_lock);

	return frames[p_frame].delay_sec;
}

void AnimatedTexture::set_fps(float p_fps) {
	ERR_FAIL_COND(p_fps < 0 || p_fps >= 1000);

	fps = p_fps;
}

float AnimatedTexture::get_fps() const {
	return fps;
}

int AnimatedTexture::get_width() const {
	RWLockRead r(rw_lock);

	if (!frames[current_frame].texture.is_valid()) {
		return 1;
	}

	return frames[current_frame].texture->get_width();
}

int AnimatedTexture::get_height() const {
	RWLockRead r(rw_lock);

	if (!frames[current_frame].texture.is_valid()) {
		return 1;
	}

	return frames[current_frame].texture->get_height();
}

RID AnimatedTexture::get_rid() const {
	return proxy;
}

bool AnimatedTexture::has_alpha() const {
	RWLockRead r(rw_lock);

	if (!frames[current_frame].texture.is_valid()) {
		return false;
	}

	return frames[current_frame].texture->has_alpha();
}

Ref<Image> AnimatedTexture::get_image() const {
	RWLockRead r(rw_lock);

	if (!frames[current_frame].texture.is_valid()) {
		return Ref<Image>();
	}

	return frames[current_frame].texture->get_image();
}

bool AnimatedTexture::is_pixel_opaque(int p_x, int p_y) const {
	RWLockRead r(rw_lock);

	if (frames[current_frame].texture.is_valid()) {
		return frames[current_frame].texture->is_pixel_opaque(p_x, p_y);
	}
	return true;
}

void AnimatedTexture::_validate_property(PropertyInfo &property) const {
	String prop = property.name;
	if (prop.begins_with("frame_")) {
		int frame = prop.get_slicec('/', 0).get_slicec('_', 1).to_int();
		if (frame >= frame_count) {
			property.usage = PROPERTY_USAGE_NONE;
		}
	}
}

void AnimatedTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_frames", "frames"), &AnimatedTexture::set_frames);
	ClassDB::bind_method(D_METHOD("get_frames"), &AnimatedTexture::get_frames);

	ClassDB::bind_method(D_METHOD("set_current_frame", "frame"), &AnimatedTexture::set_current_frame);
	ClassDB::bind_method(D_METHOD("get_current_frame"), &AnimatedTexture::get_current_frame);

	ClassDB::bind_method(D_METHOD("set_pause", "pause"), &AnimatedTexture::set_pause);
	ClassDB::bind_method(D_METHOD("get_pause"), &AnimatedTexture::get_pause);

	ClassDB::bind_method(D_METHOD("set_oneshot", "oneshot"), &AnimatedTexture::set_oneshot);
	ClassDB::bind_method(D_METHOD("get_oneshot"), &AnimatedTexture::get_oneshot);

	ClassDB::bind_method(D_METHOD("set_fps", "fps"), &AnimatedTexture::set_fps);
	ClassDB::bind_method(D_METHOD("get_fps"), &AnimatedTexture::get_fps);

	ClassDB::bind_method(D_METHOD("set_frame_texture", "frame", "texture"), &AnimatedTexture::set_frame_texture);
	ClassDB::bind_method(D_METHOD("get_frame_texture", "frame"), &AnimatedTexture::get_frame_texture);

	ClassDB::bind_method(D_METHOD("set_frame_delay", "frame", "delay"), &AnimatedTexture::set_frame_delay);
	ClassDB::bind_method(D_METHOD("get_frame_delay", "frame"), &AnimatedTexture::get_frame_delay);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "frames", PROPERTY_HINT_RANGE, "1," + itos(MAX_FRAMES), PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), "set_frames", "get_frames");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_frame", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_current_frame", "get_current_frame");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "pause"), "set_pause", "get_pause");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "oneshot"), "set_oneshot", "get_oneshot");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "fps", PROPERTY_HINT_RANGE, "0,1024,0.1"), "set_fps", "get_fps");

	for (int i = 0; i < MAX_FRAMES; i++) {
		ADD_PROPERTYI(PropertyInfo(Variant::OBJECT, "frame_" + itos(i) + "/texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture2D", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_frame_texture", "get_frame_texture", i);
		ADD_PROPERTYI(PropertyInfo(Variant::FLOAT, "frame_" + itos(i) + "/delay_sec", PROPERTY_HINT_RANGE, "0.0,16.0,0.01", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_INTERNAL), "set_frame_delay", "get_frame_delay", i);
	}

	BIND_CONSTANT(MAX_FRAMES);
}

AnimatedTexture::AnimatedTexture() {
	//proxy = RS::get_singleton()->texture_create();
	proxy_ph = RS::get_singleton()->texture_2d_placeholder_create();
	proxy = RS::get_singleton()->texture_proxy_create(proxy_ph);

	RenderingServer::get_singleton()->texture_set_force_redraw_if_visible(proxy, true);
	RenderingServer::get_singleton()->connect("frame_pre_draw", callable_mp(this, &AnimatedTexture::_update_proxy));
}

AnimatedTexture::~AnimatedTexture() {
	RS::get_singleton()->free(proxy);
	RS::get_singleton()->free(proxy_ph);
}

///////////////////////////////

void TextureLayered::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_format"), &TextureLayered::get_format);
	ClassDB::bind_method(D_METHOD("get_layered_type"), &TextureLayered::get_layered_type);
	ClassDB::bind_method(D_METHOD("get_width"), &TextureLayered::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &TextureLayered::get_height);
	ClassDB::bind_method(D_METHOD("get_layers"), &TextureLayered::get_layers);
	ClassDB::bind_method(D_METHOD("has_mipmaps"), &TextureLayered::has_mipmaps);
	ClassDB::bind_method(D_METHOD("get_layer_data", "layer"), &TextureLayered::get_layer_data);

	BIND_ENUM_CONSTANT(LAYERED_TYPE_2D_ARRAY);
	BIND_ENUM_CONSTANT(LAYERED_TYPE_CUBEMAP);
	BIND_ENUM_CONSTANT(LAYERED_TYPE_CUBEMAP_ARRAY);
}

///////////////////////////////
Image::Format ImageTextureLayered::get_format() const {
	return format;
}

int ImageTextureLayered::get_width() const {
	return width;
}

int ImageTextureLayered::get_height() const {
	return height;
}

int ImageTextureLayered::get_layers() const {
	return layers;
}

bool ImageTextureLayered::has_mipmaps() const {
	return mipmaps;
}

ImageTextureLayered::LayeredType ImageTextureLayered::get_layered_type() const {
	return layered_type;
}

Error ImageTextureLayered::_create_from_images(const Array &p_images) {
	Vector<Ref<Image>> images;
	for (int i = 0; i < p_images.size(); i++) {
		Ref<Image> img = p_images[i];
		ERR_FAIL_COND_V(img.is_null(), ERR_INVALID_PARAMETER);
		images.push_back(img);
	}

	return create_from_images(images);
}

Array ImageTextureLayered::_get_images() const {
	Array images;
	for (int i = 0; i < layers; i++) {
		images.push_back(get_layer_data(i));
	}
	return images;
}

Error ImageTextureLayered::create_from_images(Vector<Ref<Image>> p_images) {
	int new_layers = p_images.size();
	ERR_FAIL_COND_V(new_layers == 0, ERR_INVALID_PARAMETER);
	if (layered_type == LAYERED_TYPE_CUBEMAP) {
		ERR_FAIL_COND_V_MSG(new_layers != 6, ERR_INVALID_PARAMETER,
				"Cubemaps require exactly 6 layers");
	} else if (layered_type == LAYERED_TYPE_CUBEMAP_ARRAY) {
		ERR_FAIL_COND_V_MSG((new_layers % 6) != 0, ERR_INVALID_PARAMETER,
				"Cubemap array layers must be a multiple of 6");
	}

	ERR_FAIL_COND_V(p_images[0].is_null() || p_images[0]->is_empty(), ERR_INVALID_PARAMETER);

	Image::Format new_format = p_images[0]->get_format();
	int new_width = p_images[0]->get_width();
	int new_height = p_images[0]->get_height();
	bool new_mipmaps = p_images[0]->has_mipmaps();

	for (int i = 1; i < p_images.size(); i++) {
		ERR_FAIL_COND_V_MSG(p_images[i]->get_format() != new_format, ERR_INVALID_PARAMETER,
				"All images must share the same format");
		ERR_FAIL_COND_V_MSG(p_images[i]->get_width() != new_width || p_images[i]->get_height() != new_height, ERR_INVALID_PARAMETER,
				"All images must share the same dimensions");
		ERR_FAIL_COND_V_MSG(p_images[i]->has_mipmaps() != new_mipmaps, ERR_INVALID_PARAMETER,
				"All images must share the usage of mipmaps");
	}

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_layered_create(p_images, RS::TextureLayeredType(layered_type));
		ERR_FAIL_COND_V(!new_texture.is_valid(), ERR_CANT_CREATE);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_2d_layered_create(p_images, RS::TextureLayeredType(layered_type));
		ERR_FAIL_COND_V(!texture.is_valid(), ERR_CANT_CREATE);
	}

	format = new_format;
	width = new_width;
	height = new_height;
	layers = new_layers;
	mipmaps = new_mipmaps;
	return OK;
}

void ImageTextureLayered::update_layer(const Ref<Image> &p_image, int p_layer) {
	ERR_FAIL_COND(texture.is_valid());
	ERR_FAIL_COND(p_image.is_null());
	ERR_FAIL_COND(p_image->get_format() != format);
	ERR_FAIL_COND(p_image->get_width() != width || p_image->get_height() != height);
	ERR_FAIL_INDEX(p_layer, layers);
	ERR_FAIL_COND(p_image->has_mipmaps() != mipmaps);
	RS::get_singleton()->texture_2d_update(texture, p_image, p_layer);
}

Ref<Image> ImageTextureLayered::get_layer_data(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, layers, Ref<Image>());
	return RS::get_singleton()->texture_2d_layer_get(texture, p_layer);
}

RID ImageTextureLayered::get_rid() const {
	if (texture.is_null()) {
		texture = RS::get_singleton()->texture_2d_layered_placeholder_create(RS::TextureLayeredType(layered_type));
	}
	return texture;
}

void ImageTextureLayered::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RS::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

void ImageTextureLayered::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_from_images", "images"), &ImageTextureLayered::_create_from_images);
	ClassDB::bind_method(D_METHOD("update_layer", "image", "layer"), &ImageTextureLayered::update_layer);

	ClassDB::bind_method(D_METHOD("_get_images"), &ImageTextureLayered::_get_images);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "_images", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_INTERNAL), "create_from_images", "_get_images");
}

ImageTextureLayered::ImageTextureLayered(LayeredType p_layered_type) {
	layered_type = p_layered_type;
}

ImageTextureLayered::~ImageTextureLayered() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

///////////////////////////////////////////

void StreamTextureLayered::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

Image::Format StreamTextureLayered::get_format() const {
	return format;
}

Error StreamTextureLayered::_load_data(const String &p_path, Vector<Ref<Image>> &images, int &mipmap_limit, int p_size_limit) {
	ERR_FAIL_COND_V(images.size() != 0, ERR_INVALID_PARAMETER);

	FileAccessRef f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(!f, ERR_CANT_OPEN, vformat("Unable to open file: %s.", p_path));

	uint8_t header[4];
	f->get_buffer(header, 4);
	if (header[0] != 'G' || header[1] != 'S' || header[2] != 'T' || header[3] != 'L') {
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Stream texture layered file is corrupt (Bad header).");
	}

	uint32_t version = f->get_32();

	if (version > FORMAT_VERSION) {
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Stream texture file is too new.");
	}

	uint32_t layer_count = f->get_32(); //layer count
	uint32_t type = f->get_32(); //layer count
	ERR_FAIL_COND_V(type != layered_type, ERR_INVALID_DATA);

	uint32_t df = f->get_32(); //data format
	mipmap_limit = int(f->get_32());
	//reserved
	f->get_32();
	f->get_32();
	f->get_32();

	if (!(df & FORMAT_BIT_STREAM)) {
		p_size_limit = 0;
	}

	images.resize(layer_count);

	for (uint32_t i = 0; i < layer_count; i++) {
		Ref<Image> image = StreamTexture2D::load_image_from_file(f, p_size_limit);
		ERR_FAIL_COND_V(image.is_null() || image->is_empty(), ERR_CANT_OPEN);
		images.write[i] = image;
	}

	return OK;
}

Error StreamTextureLayered::load(const String &p_path) {
	Vector<Ref<Image>> images;

	int mipmap_limit;

	Error err = _load_data(p_path, images, mipmap_limit);
	if (err) {
		return err;
	}

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_layered_create(images, RS::TextureLayeredType(layered_type));
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_2d_layered_create(images, RS::TextureLayeredType(layered_type));
	}

	w = images[0]->get_width();
	h = images[0]->get_height();
	mipmaps = images[0]->has_mipmaps();
	format = images[0]->get_format();
	layers = images.size();

	path_to_file = p_path;

	if (get_path() == String()) {
		//temporarily set path if no path set for resource, helps find errors
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	notify_property_list_changed();
	emit_changed();
	return OK;
}

String StreamTextureLayered::get_load_path() const {
	return path_to_file;
}

int StreamTextureLayered::get_width() const {
	return w;
}

int StreamTextureLayered::get_height() const {
	return h;
}

int StreamTextureLayered::get_layers() const {
	return layers;
}

bool StreamTextureLayered::has_mipmaps() const {
	return mipmaps;
}

TextureLayered::LayeredType StreamTextureLayered::get_layered_type() const {
	return layered_type;
}

RID StreamTextureLayered::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_2d_layered_placeholder_create(RS::TextureLayeredType(layered_type));
	}
	return texture;
}

Ref<Image> StreamTextureLayered::get_layer_data(int p_layer) const {
	if (texture.is_valid()) {
		return RS::get_singleton()->texture_2d_layer_get(texture, p_layer);
	} else {
		return Ref<Image>();
	}
}

void StreamTextureLayered::reload_from_file() {
	String path = get_path();
	if (!path.is_resource_file()) {
		return;
	}

	path = ResourceLoader::path_remap(path); //remap for translation
	path = ResourceLoader::import_remap(path); //remap for import
	if (!path.is_resource_file()) {
		return;
	}

	load(path);
}

void StreamTextureLayered::_validate_property(PropertyInfo &property) const {
}

void StreamTextureLayered::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load", "path"), &StreamTextureLayered::load);
	ClassDB::bind_method(D_METHOD("get_load_path"), &StreamTextureLayered::get_load_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "load_path", PROPERTY_HINT_FILE, "*.stex"), "load", "get_load_path");
}

StreamTextureLayered::StreamTextureLayered(LayeredType p_type) {
	layered_type = p_type;
}

StreamTextureLayered::~StreamTextureLayered() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

/////////////////////////////////////////////////

RES ResourceFormatLoaderStreamTextureLayered::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<StreamTextureLayered> st;
	if (p_path.get_extension().to_lower() == "stexarray") {
		Ref<StreamTexture2DArray> s;
		s.instantiate();
		st = s;
	} else if (p_path.get_extension().to_lower() == "scube") {
		Ref<StreamCubemap> s;
		s.instantiate();
		st = s;
	} else if (p_path.get_extension().to_lower() == "scubearray") {
		Ref<StreamCubemapArray> s;
		s.instantiate();
		st = s;
	} else {
		if (r_error) {
			*r_error = ERR_FILE_UNRECOGNIZED;
		}
		return RES();
	}
	Error err = st->load(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		return RES();
	}

	return st;
}

void ResourceFormatLoaderStreamTextureLayered::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("stexarray");
	p_extensions->push_back("scube");
	p_extensions->push_back("scubearray");
}

bool ResourceFormatLoaderStreamTextureLayered::handles_type(const String &p_type) const {
	return p_type == "StreamTexture2DArray" || p_type == "StreamCubemap" || p_type == "StreamCubemapArray";
}

String ResourceFormatLoaderStreamTextureLayered::get_resource_type(const String &p_path) const {
	if (p_path.get_extension().to_lower() == "stexarray") {
		return "StreamTexture2DArray";
	}
	if (p_path.get_extension().to_lower() == "scube") {
		return "StreamCubemap";
	}
	if (p_path.get_extension().to_lower() == "scubearray") {
		return "StreamCubemapArray";
	}
	return "";
}

///////////////////////////////

void CameraTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_camera_feed_id", "feed_id"), &CameraTexture::set_camera_feed_id);
	ClassDB::bind_method(D_METHOD("get_camera_feed_id"), &CameraTexture::get_camera_feed_id);

	ClassDB::bind_method(D_METHOD("set_which_feed", "which_feed"), &CameraTexture::set_which_feed);
	ClassDB::bind_method(D_METHOD("get_which_feed"), &CameraTexture::get_which_feed);

	ClassDB::bind_method(D_METHOD("set_camera_active", "active"), &CameraTexture::set_camera_active);
	ClassDB::bind_method(D_METHOD("get_camera_active"), &CameraTexture::get_camera_active);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "camera_feed_id"), "set_camera_feed_id", "get_camera_feed_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "which_feed"), "set_which_feed", "get_which_feed");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "camera_is_active"), "set_camera_active", "get_camera_active");
}

int CameraTexture::get_width() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->get_base_width();
	} else {
		return 0;
	}
}

int CameraTexture::get_height() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->get_base_height();
	} else {
		return 0;
	}
}

bool CameraTexture::has_alpha() const {
	return false;
}

RID CameraTexture::get_rid() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->get_texture(which_feed);
	} else {
		if (_texture.is_null()) {
			_texture = RenderingServer::get_singleton()->texture_2d_placeholder_create();
		}
		return _texture;
	}
}

void CameraTexture::set_flags(uint32_t p_flags) {
	// not supported
}

uint32_t CameraTexture::get_flags() const {
	// not supported
	return 0;
}

Ref<Image> CameraTexture::get_image() const {
	// not (yet) supported
	return Ref<Image>();
}

void CameraTexture::set_camera_feed_id(int p_new_id) {
	camera_feed_id = p_new_id;
	notify_property_list_changed();
}

int CameraTexture::get_camera_feed_id() const {
	return camera_feed_id;
}

void CameraTexture::set_which_feed(CameraServer::FeedImage p_which) {
	which_feed = p_which;
	notify_property_list_changed();
}

CameraServer::FeedImage CameraTexture::get_which_feed() const {
	return which_feed;
}

void CameraTexture::set_camera_active(bool p_active) {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		feed->set_active(p_active);
		notify_property_list_changed();
	}
}

bool CameraTexture::get_camera_active() const {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		return feed->is_active();
	} else {
		return false;
	}
}

CameraTexture::CameraTexture() {}

CameraTexture::~CameraTexture() {
	if (_texture.is_valid()) {
		RenderingServer::get_singleton()->free(_texture);
	}
}
