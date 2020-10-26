/*************************************************************************/
/*  texture.cpp                                                          */
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

#include "texture.h"

#include "core/core_string_names.h"
#include "core/io/image_loader.h"
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
	ClassDB::bind_method(D_METHOD("get_data"), &Texture2D::get_data);

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
	img.instance();

	if (ImageLoader::load_image(path, img) == OK) {
		create_from_image(img);
	} else {
		Resource::reload_from_file();
		_change_notify();
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
		r_ret = get_data();
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

void ImageTexture::_reload_hook(const RID &p_hook) {
	String path = get_path();
	if (!path.is_resource_file()) {
		return;
	}

	Ref<Image> img;
	img.instance();
	Error err = ImageLoader::load_image(path, img);

	ERR_FAIL_COND_MSG(err != OK, "Cannot load image from path '" + path + "'.");

	RID new_texture = RenderingServer::get_singleton()->texture_2d_create(img);
	RenderingServer::get_singleton()->texture_replace(texture, new_texture);

	_change_notify();
	emit_changed();
}

void ImageTexture::create_from_image(const Ref<Image> &p_image) {
	ERR_FAIL_COND(p_image.is_null());
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
	_change_notify();
	emit_changed();

	image_stored = true;
}

Image::Format ImageTexture::get_format() const {
	return format;
}

void ImageTexture::update(const Ref<Image> &p_image, bool p_immediate) {
	ERR_FAIL_COND(p_image.is_null());
	ERR_FAIL_COND(texture.is_null());
	ERR_FAIL_COND(p_image->get_width() != w || p_image->get_height() != h);
	ERR_FAIL_COND(p_image->get_format() != format);
	ERR_FAIL_COND(mipmaps != p_image->has_mipmaps());

	if (p_immediate) {
		RenderingServer::get_singleton()->texture_2d_update_immediate(texture, p_image);
	} else {
		RenderingServer::get_singleton()->texture_2d_update(texture, p_image);
	}

	_change_notify();
	emit_changed();

	alpha_cache.unref();
	image_stored = true;
}

void ImageTexture::_resource_path_changed() {
	String path = get_path();
}

Ref<Image> ImageTexture::get_data() const {
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
		Ref<Image> img = get_data();
		if (img.is_valid()) {
			if (img->is_compressed()) { //must decompress, if compressed
				Ref<Image> decom = img->duplicate();
				decom->decompress();
				img = decom;
			}
			alpha_cache.instance();
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

	ClassDB::bind_method(D_METHOD("update", "image", "immediate"), &ImageTexture::update, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("set_size_override", "size"), &ImageTexture::set_size_override);
	ClassDB::bind_method(D_METHOD("_reload_hook", "rid"), &ImageTexture::_reload_hook);
}

ImageTexture::ImageTexture() {
	w = h = 0;
	image_stored = false;
	mipmaps = false;
	format = Image::FORMAT_L8;
}

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

	if (data_format == DATA_FORMAT_LOSSLESS || data_format == DATA_FORMAT_LOSSY || data_format == DATA_FORMAT_BASIS_UNIVERSAL) {
		//look for a PNG or WEBP file inside

		int sw = w;
		int sh = h;

		//mipmaps need to be read independently, they will be later combined
		Vector<Ref<Image>> mipmap_images;
		int total_size = 0;

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
			if (data_format == DATA_FORMAT_BASIS_UNIVERSAL) {
				img = Image::basis_universal_unpacker(pv);
			} else if (data_format == DATA_FORMAT_LOSSLESS) {
				img = Image::lossless_unpacker(pv);
			} else {
				img = Image::lossy_unpacker(pv);
			}

			if (img.is_null() || img->empty()) {
				ERR_FAIL_COND_V(img.is_null() || img->empty(), Ref<Image>());
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
		image.instance();

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
					copymem(&wr[ofs], r, len);
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
			image.instance();

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

Error StreamTexture2D::_load_data(const String &p_path, int &tw, int &th, int &tw_custom, int &th_custom, Ref<Image> &image, bool &r_request_3d, bool &r_request_normal, bool &r_request_roughness, int &mipmap_limit, int p_size_limit) {
	alpha_cache.unref();

	ERR_FAIL_COND_V(image.is_null(), ERR_INVALID_PARAMETER);

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);

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
	tw_custom = f->get_32();
	th_custom = f->get_32();
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

	if (image.is_null() || image->empty()) {
		return ERR_CANT_OPEN;
	}

	return OK;
}

Error StreamTexture2D::load(const String &p_path) {
	int lw, lh, lwc, lhc;
	Ref<Image> image;
	image.instance();

	bool request_3d;
	bool request_normal;
	bool request_roughness;
	int mipmap_limit;

	Error err = _load_data(p_path, lw, lh, lwc, lhc, image, request_3d, request_normal, request_roughness, mipmap_limit);
	if (err) {
		return err;
	}

	if (texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_create(image);
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = RS::get_singleton()->texture_2d_create(image);
	}
	if (lwc || lhc) {
		RS::get_singleton()->texture_set_size_override(texture, lwc, lhc);
	}

	w = lwc ? lwc : lw;
	h = lhc ? lhc : lh;
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
	_change_notify();
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

Ref<Image> StreamTexture2D::get_data() const {
	if (texture.is_valid()) {
		return RS::get_singleton()->texture_2d_get(texture);
	} else {
		return Ref<Image>();
	}
}

bool StreamTexture2D::is_pixel_opaque(int p_x, int p_y) const {
	if (!alpha_cache.is_valid()) {
		Ref<Image> img = get_data();
		if (img.is_valid()) {
			if (img->is_compressed()) { //must decompress, if compressed
				Ref<Image> decom = img->duplicate();
				decom->decompress();
				img = decom;
			}

			alpha_cache.instance();
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

StreamTexture2D::StreamTexture2D() {
	format = Image::FORMAT_MAX;
	w = 0;
	h = 0;
}

StreamTexture2D::~StreamTexture2D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

RES ResourceFormatLoaderStreamTexture2D::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, bool p_no_cache) {
	Ref<StreamTexture2D> st;
	st.instance();
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
	ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);

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
		ERR_FAIL_COND_V(image.is_null() || image->empty(), ERR_CANT_OPEN);
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

	_change_notify();
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

StreamTexture3D::StreamTexture3D() {
	format = Image::FORMAT_MAX;
	w = 0;
	h = 0;
	d = 0;
	mipmaps = false;
}

StreamTexture3D::~StreamTexture3D() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

/////////////////////////////

RES ResourceFormatLoaderStreamTexture3D::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, bool p_no_cache) {
	Ref<StreamTexture3D> st;
	st.instance();
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
	_change_notify("atlas");
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
	_change_notify("region");
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
	_change_notify("margin");
}

Rect2 AtlasTexture::get_margin() const {
	return margin;
}

void AtlasTexture::set_filter_clip(const bool p_enable) {
	filter_clip = p_enable;
	emit_changed();
	_change_notify("filter_clip");
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
	Rect2 src_c = rc.clip(src);
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

AtlasTexture::AtlasTexture() {
	filter_clip = false;
}

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

int LargeTexture::get_width() const {
	return size.width;
}

int LargeTexture::get_height() const {
	return size.height;
}

RID LargeTexture::get_rid() const {
	return RID();
}

bool LargeTexture::has_alpha() const {
	for (int i = 0; i < pieces.size(); i++) {
		if (pieces[i].texture->has_alpha()) {
			return true;
		}
	}

	return false;
}

int LargeTexture::add_piece(const Point2 &p_offset, const Ref<Texture2D> &p_texture) {
	ERR_FAIL_COND_V(p_texture.is_null(), -1);
	Piece p;
	p.offset = p_offset;
	p.texture = p_texture;
	pieces.push_back(p);

	return pieces.size() - 1;
}

void LargeTexture::set_piece_offset(int p_idx, const Point2 &p_offset) {
	ERR_FAIL_INDEX(p_idx, pieces.size());
	pieces.write[p_idx].offset = p_offset;
};

void LargeTexture::set_piece_texture(int p_idx, const Ref<Texture2D> &p_texture) {
	ERR_FAIL_COND(p_texture == this);
	ERR_FAIL_COND(p_texture.is_null());
	ERR_FAIL_INDEX(p_idx, pieces.size());
	pieces.write[p_idx].texture = p_texture;
};

void LargeTexture::set_size(const Size2 &p_size) {
	size = p_size;
}

void LargeTexture::clear() {
	pieces.clear();
	size = Size2i();
}

Array LargeTexture::_get_data() const {
	Array arr;
	for (int i = 0; i < pieces.size(); i++) {
		arr.push_back(pieces[i].offset);
		arr.push_back(pieces[i].texture);
	}
	arr.push_back(Size2(size));
	return arr;
}

void LargeTexture::_set_data(const Array &p_array) {
	ERR_FAIL_COND(p_array.size() < 1);
	ERR_FAIL_COND(!(p_array.size() & 1));
	clear();
	for (int i = 0; i < p_array.size() - 1; i += 2) {
		add_piece(p_array[i], p_array[i + 1]);
	}
	size = Size2(p_array[p_array.size() - 1]);
}

int LargeTexture::get_piece_count() const {
	return pieces.size();
}

Vector2 LargeTexture::get_piece_offset(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, pieces.size(), Vector2());
	return pieces[p_idx].offset;
}

Ref<Texture2D> LargeTexture::get_piece_texture(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, pieces.size(), Ref<Texture2D>());
	return pieces[p_idx].texture;
}

Ref<Image> LargeTexture::to_image() const {
	Ref<Image> img = memnew(Image(this->get_width(), this->get_height(), false, Image::FORMAT_RGBA8));
	for (int i = 0; i < pieces.size(); i++) {
		Ref<Image> src_img = pieces[i].texture->get_data();
		img->blit_rect(src_img, Rect2(0, 0, src_img->get_width(), src_img->get_height()), pieces[i].offset);
	}

	return img;
}

void LargeTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_piece", "ofs", "texture"), &LargeTexture::add_piece);
	ClassDB::bind_method(D_METHOD("set_piece_offset", "idx", "ofs"), &LargeTexture::set_piece_offset);
	ClassDB::bind_method(D_METHOD("set_piece_texture", "idx", "texture"), &LargeTexture::set_piece_texture);
	ClassDB::bind_method(D_METHOD("set_size", "size"), &LargeTexture::set_size);
	ClassDB::bind_method(D_METHOD("clear"), &LargeTexture::clear);

	ClassDB::bind_method(D_METHOD("get_piece_count"), &LargeTexture::get_piece_count);
	ClassDB::bind_method(D_METHOD("get_piece_offset", "idx"), &LargeTexture::get_piece_offset);
	ClassDB::bind_method(D_METHOD("get_piece_texture", "idx"), &LargeTexture::get_piece_texture);

	ClassDB::bind_method(D_METHOD("_set_data", "data"), &LargeTexture::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &LargeTexture::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_INTERNAL), "_set_data", "_get_data");
}

void LargeTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	for (int i = 0; i < pieces.size(); i++) {
		// TODO
		pieces[i].texture->draw(p_canvas_item, pieces[i].offset + p_pos, p_modulate, p_transpose);
	}
}

void LargeTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	//tiling not supported for this
	if (size.x == 0 || size.y == 0) {
		return;
	}

	Size2 scale = p_rect.size / size;

	for (int i = 0; i < pieces.size(); i++) {
		// TODO
		pieces[i].texture->draw_rect(p_canvas_item, Rect2(pieces[i].offset * scale + p_rect.position, pieces[i].texture->get_size() * scale), false, p_modulate, p_transpose);
	}
}

void LargeTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	//tiling not supported for this
	if (p_src_rect.size.x == 0 || p_src_rect.size.y == 0) {
		return;
	}

	Size2 scale = p_rect.size / p_src_rect.size;

	for (int i = 0; i < pieces.size(); i++) {
		// TODO
		Rect2 rect(pieces[i].offset, pieces[i].texture->get_size());
		if (!p_src_rect.intersects(rect)) {
			continue;
		}
		Rect2 local = p_src_rect.clip(rect);
		Rect2 target = local;
		target.size *= scale;
		target.position = p_rect.position + (p_src_rect.position + rect.position) * scale;
		local.position -= rect.position;
		pieces[i].texture->draw_rect_region(p_canvas_item, target, local, p_modulate, p_transpose, false);
	}
}

bool LargeTexture::is_pixel_opaque(int p_x, int p_y) const {
	for (int i = 0; i < pieces.size(); i++) {
		// TODO
		if (!pieces[i].texture.is_valid()) {
			continue;
		}

		Rect2 rect(pieces[i].offset, pieces[i].texture->get_size());
		if (rect.has_point(Point2(p_x, p_y))) {
			return pieces[i].texture->is_pixel_opaque(p_x - rect.position.x, p_y - rect.position.y);
		}
	}

	return true;
}

LargeTexture::LargeTexture() {
}

///////////////////

void CurveTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_width", "width"), &CurveTexture::set_width);

	ClassDB::bind_method(D_METHOD("set_curve", "curve"), &CurveTexture::set_curve);
	ClassDB::bind_method(D_METHOD("get_curve"), &CurveTexture::get_curve);

	ClassDB::bind_method(D_METHOD("_update"), &CurveTexture::_update);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "32,4096"), "set_width", "get_width");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "curve", PROPERTY_HINT_RESOURCE_TYPE, "Curve"), "set_curve", "get_curve");
}

void CurveTexture::set_width(int p_width) {
	ERR_FAIL_COND(p_width < 32 || p_width > 4096);
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
	data.resize(_width * sizeof(float));

	// The array is locked in that scope
	{
		uint8_t *wd8 = data.ptrw();
		float *wd = (float *)wd8;

		if (_curve.is_valid()) {
			Curve &curve = **_curve;
			for (int i = 0; i < _width; ++i) {
				float t = i / static_cast<float>(_width);
				wd[i] = curve.interpolate_baked(t);
			}

		} else {
			for (int i = 0; i < _width; ++i) {
				wd[i] = 0;
			}
		}
	}

	Ref<Image> image = memnew(Image(_width, 1, false, Image::FORMAT_RF, data));

	if (_texture.is_valid()) {
		RID new_texture = RS::get_singleton()->texture_2d_create(image);
		RS::get_singleton()->texture_replace(_texture, new_texture);
	} else {
		_texture = RS::get_singleton()->texture_2d_create(image);
	}

	emit_changed();
}

Ref<Curve> CurveTexture::get_curve() const {
	return _curve;
}

RID CurveTexture::get_rid() const {
	if (!_texture.is_valid()) {
		_texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return _texture;
}

CurveTexture::CurveTexture() {
	_width = 2048;
}

CurveTexture::~CurveTexture() {
	if (_texture.is_valid()) {
		RS::get_singleton()->free(_texture);
	}
}

//////////////////

GradientTexture::GradientTexture() {
	update_pending = false;
	width = 2048;

	_queue_update();
}

GradientTexture::~GradientTexture() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

void GradientTexture::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_gradient", "gradient"), &GradientTexture::set_gradient);
	ClassDB::bind_method(D_METHOD("get_gradient"), &GradientTexture::get_gradient);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &GradientTexture::set_width);

	ClassDB::bind_method(D_METHOD("_update"), &GradientTexture::_update);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gradient", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_gradient", "get_gradient");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "width", PROPERTY_HINT_RANGE, "1,2048,1,or_greater"), "set_width", "get_width");
}

void GradientTexture::set_gradient(Ref<Gradient> p_gradient) {
	if (p_gradient == gradient) {
		return;
	}
	if (gradient.is_valid()) {
		gradient->disconnect(CoreStringNames::get_singleton()->changed, callable_mp(this, &GradientTexture::_update));
	}
	gradient = p_gradient;
	if (gradient.is_valid()) {
		gradient->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &GradientTexture::_update));
	}
	_update();
	emit_changed();
}

Ref<Gradient> GradientTexture::get_gradient() const {
	return gradient;
}

void GradientTexture::_queue_update() {
	if (update_pending) {
		return;
	}

	update_pending = true;
	call_deferred("_update");
}

void GradientTexture::_update() {
	update_pending = false;

	if (gradient.is_null()) {
		return;
	}

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

	emit_changed();
}

void GradientTexture::set_width(int p_width) {
	width = p_width;
	_queue_update();
}

int GradientTexture::get_width() const {
	return width;
}

Ref<Image> GradientTexture::get_data() const {
	if (!texture.is_valid()) {
		return Ref<Image>();
	}
	return RenderingServer::get_singleton()->texture_2d_get(texture);
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
			_change_notify("current_frame");
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

Ref<Image> AnimatedTexture::get_data() const {
	RWLockRead r(rw_lock);

	if (!frames[current_frame].texture.is_valid()) {
		return Ref<Image>();
	}

	return frames[current_frame].texture->get_data();
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
			property.usage = 0;
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
	ADD_PROPERTY(PropertyInfo(Variant::INT, "current_frame", PROPERTY_HINT_NONE, "", 0), "set_current_frame", "get_current_frame");
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
	time = 0;
	frame_count = 1;
	fps = 4;
	prev_ticks = 0;
	current_frame = 0;
	pause = false;
	oneshot = false;
	RenderingServer::get_singleton()->connect("frame_pre_draw", callable_mp(this, &AnimatedTexture::_update_proxy));

#ifndef NO_THREADS
	rw_lock = RWLock::create();
#else
	rw_lock = nullptr;
#endif
}

AnimatedTexture::~AnimatedTexture() {
	RS::get_singleton()->free(proxy);
	RS::get_singleton()->free(proxy_ph);
	if (rw_lock) {
		memdelete(rw_lock);
	}
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

	ERR_FAIL_COND_V(p_images[0].is_null() || p_images[0]->empty(), ERR_INVALID_PARAMETER);

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
	format = Image::FORMAT_MAX;

	width = 0;
	height = 0;
	layers = 0;
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
	ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);

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
		ERR_FAIL_COND_V(image.is_null() || image->empty(), ERR_CANT_OPEN);
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

	_change_notify();
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
	format = Image::FORMAT_MAX;
	w = 0;
	h = 0;
	layers = 0;
	mipmaps = false;
}

StreamTextureLayered::~StreamTextureLayered() {
	if (texture.is_valid()) {
		RS::get_singleton()->free(texture);
	}
}

/////////////////////////////////////////////////

RES ResourceFormatLoaderStreamTextureLayered::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, bool p_no_cache) {
	Ref<StreamTextureLayered> st;
	if (p_path.get_extension().to_lower() == "stexarray") {
		Ref<StreamTexture2DArray> s;
		s.instance();
		st = s;
	} else if (p_path.get_extension().to_lower() == "scube") {
		Ref<StreamCubemap> s;
		s.instance();
		st = s;
	} else if (p_path.get_extension().to_lower() == "scubearray") {
		Ref<StreamCubemapArray> s;
		s.instance();
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
		return RID();
	}
}

void CameraTexture::set_flags(uint32_t p_flags) {
	// not supported
}

uint32_t CameraTexture::get_flags() const {
	// not supported
	return 0;
}

Ref<Image> CameraTexture::get_data() const {
	// not (yet) supported
	return Ref<Image>();
}

void CameraTexture::set_camera_feed_id(int p_new_id) {
	camera_feed_id = p_new_id;
	_change_notify();
}

int CameraTexture::get_camera_feed_id() const {
	return camera_feed_id;
}

void CameraTexture::set_which_feed(CameraServer::FeedImage p_which) {
	which_feed = p_which;
	_change_notify();
}

CameraServer::FeedImage CameraTexture::get_which_feed() const {
	return which_feed;
}

void CameraTexture::set_camera_active(bool p_active) {
	Ref<CameraFeed> feed = CameraServer::get_singleton()->get_feed_by_id(camera_feed_id);
	if (feed.is_valid()) {
		feed->set_active(p_active);
		_change_notify();
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

CameraTexture::CameraTexture() {
	camera_feed_id = 0;
	which_feed = CameraServer::FEED_RGBA_IMAGE;
}

CameraTexture::~CameraTexture() {
	// nothing to do here yet
}
