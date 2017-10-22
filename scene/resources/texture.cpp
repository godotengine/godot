/*************************************************************************/
/*  texture.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "texture.h"
#include "core/method_bind_ext.gen.inc"
#include "core/os/os.h"
#include "core_string_names.h"
#include "io/image_loader.h"

Size2 Texture::get_size() const {

	return Size2(get_width(), get_height());
}

void Texture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, get_size()), get_rid(), false, p_modulate, p_transpose, normal_rid);
}
void Texture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, get_rid(), p_tile, p_modulate, p_transpose, normal_rid);
}
void Texture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map, bool p_clip_uv) const {

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, get_rid(), p_src_rect, p_modulate, p_transpose, normal_rid, p_clip_uv);
}

bool Texture::get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const {

	r_rect = p_rect;
	r_src_rect = p_src_rect;

	return true;
}

void Texture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_width"), &Texture::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &Texture::get_height);
	ClassDB::bind_method(D_METHOD("get_size"), &Texture::get_size);
	ClassDB::bind_method(D_METHOD("has_alpha"), &Texture::has_alpha);
	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &Texture::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &Texture::get_flags);
	ClassDB::bind_method(D_METHOD("draw", "canvas_item", "position", "modulate", "transpose", "normal_map"), &Texture::draw, DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("draw_rect", "canvas_item", "rect", "tile", "modulate", "transpose", "normal_map"), &Texture::draw_rect, DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("draw_rect_region", "canvas_item", "rect", "src_rect", "modulate", "transpose", "normal_map", "clip_uv"), &Texture::draw_rect_region, DEFVAL(Color(1, 1, 1)), DEFVAL(false), DEFVAL(Variant()), DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_data"), &Texture::get_data);

	BIND_ENUM_CONSTANT(FLAG_MIPMAPS);
	BIND_ENUM_CONSTANT(FLAG_REPEAT);
	BIND_ENUM_CONSTANT(FLAG_FILTER);
	BIND_ENUM_CONSTANT(FLAGS_DEFAULT);
	BIND_ENUM_CONSTANT(FLAG_ANISOTROPIC_FILTER);
	BIND_ENUM_CONSTANT(FLAG_CONVERT_TO_LINEAR);
	BIND_ENUM_CONSTANT(FLAG_MIRRORED_REPEAT);
	BIND_ENUM_CONSTANT(FLAG_VIDEO_SURFACE);
}

Texture::Texture() {
}

/////////////////////

void ImageTexture::reload_from_file() {

	String path = ResourceLoader::path_remap(get_path());
	if (!path.is_resource_file())
		return;

	uint32_t flags = get_flags();
	Ref<Image> img;
	img.instance();

	Error err = ImageLoader::load_image(path, img);
	ERR_FAIL_COND(err != OK);

	create_from_image(img, flags);
}

bool ImageTexture::_set(const StringName &p_name, const Variant &p_value) {

	if (p_name == "image")
		create_from_image(p_value, flags);
	else if (p_name == "flags")
		if (w * h == 0)
			flags = p_value;
		else
			set_flags(p_value);
	else if (p_name == "size") {
		Size2 s = p_value;
		w = s.width;
		h = s.height;
		VisualServer::get_singleton()->texture_set_size_override(texture, w, h);
	} else if (p_name == "storage") {
		storage = Storage(p_value.operator int());
	} else if (p_name == "lossy_quality") {
		lossy_storage_quality = p_value;
	} else if (p_name == "_data") {
		_set_data(p_value);
	} else
		return false;

	return true;
}

bool ImageTexture::_get(const StringName &p_name, Variant &r_ret) const {

	if (p_name == "image_data") {

	} else if (p_name == "image")
		r_ret = get_data();
	else if (p_name == "flags")
		r_ret = flags;
	else if (p_name == "size")
		r_ret = Size2(w, h);
	else if (p_name == "storage")
		r_ret = storage;
	else if (p_name == "lossy_quality")
		r_ret = lossy_storage_quality;
	else
		return false;

	return true;
}

void ImageTexture::_get_property_list(List<PropertyInfo> *p_list) const {

	PropertyHint img_hint = PROPERTY_HINT_NONE;
	if (storage == STORAGE_COMPRESS_LOSSY) {
		img_hint = PROPERTY_HINT_IMAGE_COMPRESS_LOSSY;
	} else if (storage == STORAGE_COMPRESS_LOSSLESS) {
		img_hint = PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS;
	}

	p_list->push_back(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Mipmaps,Repeat,Filter,Anisotropic,sRGB,Mirrored Repeat"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "image", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
	p_list->push_back(PropertyInfo(Variant::VECTOR2, "size", PROPERTY_HINT_NONE, ""));
	p_list->push_back(PropertyInfo(Variant::INT, "storage", PROPERTY_HINT_ENUM, "Uncompressed,Compress Lossy,Compress Lossless"));
	p_list->push_back(PropertyInfo(Variant::REAL, "lossy_quality", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"));
}

void ImageTexture::_reload_hook(const RID &p_hook) {

	String path = get_path();
	if (!path.is_resource_file())
		return;

	Ref<Image> img;
	img.instance();
	Error err = ImageLoader::load_image(path, img);

	ERR_FAIL_COND(err != OK);

	VisualServer::get_singleton()->texture_set_data(texture, img);

	_change_notify();
}

void ImageTexture::create(int p_width, int p_height, Image::Format p_format, uint32_t p_flags) {

	flags = p_flags;
	VisualServer::get_singleton()->texture_allocate(texture, p_width, p_height, p_format, p_flags);
	format = p_format;
	w = p_width;
	h = p_height;
}
void ImageTexture::create_from_image(const Ref<Image> &p_image, uint32_t p_flags) {

	ERR_FAIL_COND(p_image.is_null());
	flags = p_flags;
	w = p_image->get_width();
	h = p_image->get_height();
	format = p_image->get_format();

	VisualServer::get_singleton()->texture_allocate(texture, p_image->get_width(), p_image->get_height(), p_image->get_format(), p_flags);
	VisualServer::get_singleton()->texture_set_data(texture, p_image);
	_change_notify();
}

void ImageTexture::set_flags(uint32_t p_flags) {

	/*	uint32_t cube = flags & FLAG_CUBEMAP;
	if (flags == p_flags&cube)
		return;

	flags=p_flags|cube;	*/
	flags = p_flags;
	VisualServer::get_singleton()->texture_set_flags(texture, p_flags);
}

uint32_t ImageTexture::get_flags() const {

	return ImageTexture::flags;
}

Image::Format ImageTexture::get_format() const {

	return format;
}

void ImageTexture::load(const String &p_path) {

	Ref<Image> img;
	img.instance();
	img->load(p_path);
	create_from_image(img);
}

void ImageTexture::set_data(const Ref<Image> &p_image) {

	VisualServer::get_singleton()->texture_set_data(texture, p_image);

	_change_notify();
}

void ImageTexture::_resource_path_changed() {

	String path = get_path();
}

Ref<Image> ImageTexture::get_data() const {

	return VisualServer::get_singleton()->texture_get_data(texture);
}

int ImageTexture::get_width() const {

	return w;
}

int ImageTexture::get_height() const {

	return h;
}

RID ImageTexture::get_rid() const {

	return texture;
}

bool ImageTexture::has_alpha() const {

	return (format == Image::FORMAT_LA8 || format == Image::FORMAT_RGBA8);
}

void ImageTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	if ((w | h) == 0)
		return;
	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, Size2(w, h)), texture, false, p_modulate, p_transpose, normal_rid);
}
void ImageTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	if ((w | h) == 0)
		return;
	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, texture, p_tile, p_modulate, p_transpose, normal_rid);
}
void ImageTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map, bool p_clip_uv) const {

	if ((w | h) == 0)
		return;
	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, texture, p_src_rect, p_modulate, p_transpose, normal_rid, p_clip_uv);
}

void ImageTexture::set_size_override(const Size2 &p_size) {

	Size2 s = p_size;
	if (s.x != 0)
		w = s.x;
	if (s.y != 0)
		h = s.y;
	VisualServer::get_singleton()->texture_set_size_override(texture, w, h);
}

void ImageTexture::set_path(const String &p_path, bool p_take_over) {

	if (texture.is_valid()) {
		VisualServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

void ImageTexture::set_storage(Storage p_storage) {

	storage = p_storage;
}

ImageTexture::Storage ImageTexture::get_storage() const {

	return storage;
}

void ImageTexture::set_lossy_storage_quality(float p_lossy_storage_quality) {

	lossy_storage_quality = p_lossy_storage_quality;
}

float ImageTexture::get_lossy_storage_quality() const {

	return lossy_storage_quality;
}

void ImageTexture::_set_data(Dictionary p_data) {

	Ref<Image> img = p_data["image"];
	ERR_FAIL_COND(!img.is_valid());
	uint32_t flags = p_data["flags"];

	create_from_image(img, flags);

	set_storage(Storage(p_data["storage"].operator int()));
	set_lossy_storage_quality(p_data["lossy_quality"]);

	set_size_override(p_data["size"]);
};

void ImageTexture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("create", "width", "height", "format", "flags"), &ImageTexture::create, DEFVAL(FLAGS_DEFAULT));
	ClassDB::bind_method(D_METHOD("create_from_image", "image", "flags"), &ImageTexture::create_from_image, DEFVAL(FLAGS_DEFAULT));
	ClassDB::bind_method(D_METHOD("get_format"), &ImageTexture::get_format);
	ClassDB::bind_method(D_METHOD("load", "path"), &ImageTexture::load);
	ClassDB::bind_method(D_METHOD("set_data", "image"), &ImageTexture::set_data);
	ClassDB::bind_method(D_METHOD("set_storage", "mode"), &ImageTexture::set_storage);
	ClassDB::bind_method(D_METHOD("get_storage"), &ImageTexture::get_storage);
	ClassDB::bind_method(D_METHOD("set_lossy_storage_quality", "quality"), &ImageTexture::set_lossy_storage_quality);
	ClassDB::bind_method(D_METHOD("get_lossy_storage_quality"), &ImageTexture::get_lossy_storage_quality);

	ClassDB::bind_method(D_METHOD("set_size_override", "size"), &ImageTexture::set_size_override);
	ClassDB::bind_method(D_METHOD("_reload_hook", "rid"), &ImageTexture::_reload_hook);

	BIND_ENUM_CONSTANT(STORAGE_RAW);
	BIND_ENUM_CONSTANT(STORAGE_COMPRESS_LOSSY);
	BIND_ENUM_CONSTANT(STORAGE_COMPRESS_LOSSLESS);
}

ImageTexture::ImageTexture() {

	w = h = 0;
	flags = FLAGS_DEFAULT;
	texture = VisualServer::get_singleton()->texture_create();
	storage = STORAGE_RAW;
	lossy_storage_quality = 0.7;
}

ImageTexture::~ImageTexture() {

	VisualServer::get_singleton()->free(texture);
}

//////////////////////////////////////////

void StreamTexture::_requested_3d(void *p_ud) {

	StreamTexture *st = (StreamTexture *)p_ud;
	Ref<StreamTexture> stex(st);
	ERR_FAIL_COND(!request_3d_callback);
	request_3d_callback(stex);
}

void StreamTexture::_requested_srgb(void *p_ud) {

	StreamTexture *st = (StreamTexture *)p_ud;
	Ref<StreamTexture> stex(st);
	ERR_FAIL_COND(!request_srgb_callback);
	request_srgb_callback(stex);
}

void StreamTexture::_requested_normal(void *p_ud) {

	StreamTexture *st = (StreamTexture *)p_ud;
	Ref<StreamTexture> stex(st);
	ERR_FAIL_COND(!request_normal_callback);
	request_normal_callback(stex);
}

StreamTexture::TextureFormatRequestCallback StreamTexture::request_3d_callback = NULL;
StreamTexture::TextureFormatRequestCallback StreamTexture::request_srgb_callback = NULL;
StreamTexture::TextureFormatRequestCallback StreamTexture::request_normal_callback = NULL;

uint32_t StreamTexture::get_flags() const {

	return flags;
}
Image::Format StreamTexture::get_format() const {

	return format;
}

Error StreamTexture::_load_data(const String &p_path, int &tw, int &th, int &flags, Ref<Image> &image, int p_size_limit) {

	ERR_FAIL_COND_V(image.is_null(), ERR_INVALID_PARAMETER);

	FileAccess *f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V(!f, ERR_CANT_OPEN);

	uint8_t header[4];
	f->get_buffer(header, 4);
	if (header[0] != 'G' || header[1] != 'D' || header[2] != 'S' || header[3] != 'T') {
		memdelete(f);
		ERR_FAIL_COND_V(header[0] != 'G' || header[1] != 'D' || header[2] != 'S' || header[3] != 'T', ERR_FILE_CORRUPT);
	}

	tw = f->get_32();
	th = f->get_32();
	flags = f->get_32(); //texture flags!
	uint32_t df = f->get_32(); //data format

/*
	print_line("width: " + itos(tw));
	print_line("height: " + itos(th));
	print_line("flags: " + itos(flags));
	print_line("df: " + itos(df));
	*/
#ifdef TOOLS_ENABLED

	if (request_3d_callback && df & FORMAT_BIT_DETECT_3D) {
		//print_line("request detect 3D at " + p_path);
		VS::get_singleton()->texture_set_detect_3d_callback(texture, _requested_3d, this);
	} else {
		//print_line("not requesting detect 3D at " + p_path);
		VS::get_singleton()->texture_set_detect_3d_callback(texture, NULL, NULL);
	}

	if (request_srgb_callback && df & FORMAT_BIT_DETECT_SRGB) {
		//print_line("request detect srgb at " + p_path);
		VS::get_singleton()->texture_set_detect_srgb_callback(texture, _requested_srgb, this);
	} else {
		//print_line("not requesting detect srgb at " + p_path);
		VS::get_singleton()->texture_set_detect_srgb_callback(texture, NULL, NULL);
	}

	if (request_srgb_callback && df & FORMAT_BIT_DETECT_NORMAL) {
		//print_line("request detect srgb at " + p_path);
		VS::get_singleton()->texture_set_detect_normal_callback(texture, _requested_normal, this);
	} else {
		//print_line("not requesting detect normal at " + p_path);
		VS::get_singleton()->texture_set_detect_normal_callback(texture, NULL, NULL);
	}
#endif
	if (!(df & FORMAT_BIT_STREAM)) {
		p_size_limit = 0;
	}

	if (df & FORMAT_BIT_LOSSLESS || df & FORMAT_BIT_LOSSY) {
		//look for a PNG or WEBP file inside

		int sw = tw;
		int sh = th;

		uint32_t mipmaps = f->get_32();
		uint32_t size = f->get_32();

		//print_line("mipmaps: " + itos(mipmaps));

		while (mipmaps > 1 && p_size_limit > 0 && (sw > p_size_limit || sh > p_size_limit)) {

			f->seek(f->get_position() + size);
			mipmaps = f->get_32();
			size = f->get_32();

			sw = MAX(sw >> 1, 1);
			sh = MAX(sh >> 1, 1);
			mipmaps--;
		}

		//mipmaps need to be read independently, they will be later combined
		Vector<Ref<Image> > mipmap_images;
		int total_size = 0;

		for (uint32_t i = 0; i < mipmaps; i++) {

			if (i) {
				size = f->get_32();
			}

			PoolVector<uint8_t> pv;
			pv.resize(size);
			{
				PoolVector<uint8_t>::Write w = pv.write();
				f->get_buffer(w.ptr(), size);
			}

			Ref<Image> img;
			if (df & FORMAT_BIT_LOSSLESS) {
				img = Image::lossless_unpacker(pv);
			} else {
				img = Image::lossy_unpacker(pv);
			}

			if (img.is_null() || img->empty()) {
				memdelete(f);
				ERR_FAIL_COND_V(img.is_null() || img->empty(), ERR_FILE_CORRUPT);
			}

			total_size += img->get_data().size();

			mipmap_images.push_back(img);
		}

		//print_line("mipmap read total: " + itos(mipmap_images.size()));

		memdelete(f); //no longer needed

		if (mipmap_images.size() == 1) {

			image = mipmap_images[0];
			return OK;

		} else {
			PoolVector<uint8_t> img_data;
			img_data.resize(total_size);

			{
				PoolVector<uint8_t>::Write w = img_data.write();

				int ofs = 0;
				for (int i = 0; i < mipmap_images.size(); i++) {

					PoolVector<uint8_t> id = mipmap_images[i]->get_data();
					int len = id.size();
					PoolVector<uint8_t>::Read r = id.read();
					copymem(&w[ofs], r.ptr(), len);
					ofs += len;
				}
			}

			image->create(sw, sh, true, mipmap_images[0]->get_format(), img_data);
			return OK;
		}

	} else {

		//look for regular format
		Image::Format format = (Image::Format)(df & FORMAT_MASK_IMAGE_FORMAT);
		bool mipmaps = df & FORMAT_BIT_HAS_MIPMAPS;

		if (!mipmaps) {
			int size = Image::get_image_data_size(tw, th, format, 0);

			PoolVector<uint8_t> img_data;
			img_data.resize(size);

			{
				PoolVector<uint8_t>::Write w = img_data.write();
				f->get_buffer(w.ptr(), size);
			}

			memdelete(f);

			image->create(tw, th, false, format, img_data);
			return OK;
		} else {

			int sw = tw;
			int sh = th;

			int mipmaps = Image::get_image_required_mipmaps(tw, th, format);
			int total_size = Image::get_image_data_size(tw, th, format, mipmaps);
			int idx = 0;
			int ofs = 0;

			while (mipmaps > 1 && p_size_limit > 0 && (sw > p_size_limit || sh > p_size_limit)) {

				sw = MAX(sw >> 1, 1);
				sh = MAX(sh >> 1, 1);
				mipmaps--;
				idx++;
			}

			if (idx > 0) {
				ofs = Image::get_image_data_size(tw, th, format, idx - 1);
			}

			if (total_size - ofs <= 0) {
				memdelete(f);
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}

			f->seek(f->get_position() + ofs);

			PoolVector<uint8_t> img_data;
			img_data.resize(total_size - ofs);

			{
				PoolVector<uint8_t>::Write w = img_data.write();
				int bytes = f->get_buffer(w.ptr(), total_size - ofs);
				//print_line("requested read: " + itos(total_size - ofs) + " but got: " + itos(bytes));

				memdelete(f);

				if (bytes != total_size - ofs) {
					ERR_FAIL_V(ERR_FILE_CORRUPT);
				}
			}

			image->create(sw, sh, true, format, img_data);

			return OK;
		}
	}

	return ERR_BUG; //unreachable
}

Error StreamTexture::load(const String &p_path) {

	int lw, lh, lflags;
	Ref<Image> image;
	image.instance();
	Error err = _load_data(p_path, lw, lh, lflags, image);
	if (err)
		return err;

	VS::get_singleton()->texture_allocate(texture, image->get_width(), image->get_height(), image->get_format(), lflags);
	VS::get_singleton()->texture_set_data(texture, image);

	w = lw;
	h = lh;
	flags = lflags;
	path_to_file = p_path;
	format = image->get_format();

	return OK;
}
String StreamTexture::get_load_path() const {

	return path_to_file;
}

int StreamTexture::get_width() const {

	return w;
}
int StreamTexture::get_height() const {

	return h;
}
RID StreamTexture::get_rid() const {

	return texture;
}

void StreamTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	if ((w | h) == 0)
		return;
	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, Size2(w, h)), texture, false, p_modulate, p_transpose, normal_rid);
}
void StreamTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	if ((w | h) == 0)
		return;
	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, texture, p_tile, p_modulate, p_transpose, normal_rid);
}
void StreamTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map, bool p_clip_uv) const {

	if ((w | h) == 0)
		return;
	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, texture, p_src_rect, p_modulate, p_transpose, normal_rid, p_clip_uv);
}

bool StreamTexture::has_alpha() const {

	return false;
}

Ref<Image> StreamTexture::get_data() const {

	return VS::get_singleton()->texture_get_data(texture);
}

void StreamTexture::set_flags(uint32_t p_flags) {
}

void StreamTexture::reload_from_file() {

	String path = get_path();
	if (!path.is_resource_file())
		return;

	path = ResourceLoader::path_remap(path); //remap for translation
	path = ResourceLoader::import_remap(path); //remap for import
	if (!path.is_resource_file())
		return;

	load(path);
}

void StreamTexture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("load", "path"), &StreamTexture::load);
	ClassDB::bind_method(D_METHOD("get_load_path"), &StreamTexture::get_load_path);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "load_path", PROPERTY_HINT_FILE, "*.stex"), "load", "get_load_path");
}

StreamTexture::StreamTexture() {

	format = Image::FORMAT_MAX;
	flags = 0;
	w = 0;
	h = 0;

	texture = VS::get_singleton()->texture_create();
}

StreamTexture::~StreamTexture() {

	VS::get_singleton()->free(texture);
}

RES ResourceFormatLoaderStreamTexture::load(const String &p_path, const String &p_original_path, Error *r_error) {

	Ref<StreamTexture> st;
	st.instance();
	Error err = st->load(p_path);
	if (r_error)
		*r_error = err;
	if (err != OK)
		return RES();

	return st;
}

void ResourceFormatLoaderStreamTexture::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("stex");
}
bool ResourceFormatLoaderStreamTexture::handles_type(const String &p_type) const {
	return p_type == "StreamTexture";
}
String ResourceFormatLoaderStreamTexture::get_resource_type(const String &p_path) const {

	if (p_path.get_extension().to_lower() == "stex")
		return "StreamTexture";
	return "";
}

//////////////////////////////////////////

int AtlasTexture::get_width() const {

	if (region.size.width == 0) {
		if (atlas.is_valid())
			return atlas->get_width();
		return 1;
	} else {
		return region.size.width + margin.size.width;
	}
}
int AtlasTexture::get_height() const {

	if (region.size.height == 0) {
		if (atlas.is_valid())
			return atlas->get_height();
		return 1;
	} else {
		return region.size.height + margin.size.height;
	}
}
RID AtlasTexture::get_rid() const {

	if (atlas.is_valid())
		return atlas->get_rid();

	return RID();
}

bool AtlasTexture::has_alpha() const {

	if (atlas.is_valid())
		return atlas->has_alpha();

	return false;
}

void AtlasTexture::set_flags(uint32_t p_flags) {

	if (atlas.is_valid())
		atlas->set_flags(p_flags);
}

uint32_t AtlasTexture::get_flags() const {

	if (atlas.is_valid())
		return atlas->get_flags();

	return 0;
}

void AtlasTexture::set_atlas(const Ref<Texture> &p_atlas) {

	if (atlas == p_atlas)
		return;
	atlas = p_atlas;
	emit_changed();
	_change_notify("atlas");
}
Ref<Texture> AtlasTexture::get_atlas() const {

	return atlas;
}

void AtlasTexture::set_region(const Rect2 &p_region) {

	if (region == p_region)
		return;
	region = p_region;
	emit_changed();
	_change_notify("region");
}

Rect2 AtlasTexture::get_region() const {

	return region;
}

void AtlasTexture::set_margin(const Rect2 &p_margin) {

	if (margin == p_margin)
		return;
	margin = p_margin;
	emit_changed();
	_change_notify("margin");
}

Rect2 AtlasTexture::get_margin() const {

	return margin;
}

void AtlasTexture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_atlas", "atlas"), &AtlasTexture::set_atlas);
	ClassDB::bind_method(D_METHOD("get_atlas"), &AtlasTexture::get_atlas);

	ClassDB::bind_method(D_METHOD("set_region", "region"), &AtlasTexture::set_region);
	ClassDB::bind_method(D_METHOD("get_region"), &AtlasTexture::get_region);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &AtlasTexture::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &AtlasTexture::get_margin);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "atlas", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), "set_atlas", "get_atlas");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "region"), "set_region", "get_region");
	ADD_PROPERTY(PropertyInfo(Variant::RECT2, "margin"), "set_margin", "get_margin");
}

void AtlasTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	Rect2 rc = region;

	if (!atlas.is_valid())
		return;

	if (rc.size.width == 0) {
		rc.size.width = atlas->get_width();
	}

	if (rc.size.height == 0) {
		rc.size.height = atlas->get_height();
	}

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, Rect2(p_pos + margin.position, rc.size), atlas->get_rid(), rc, p_modulate, p_transpose, normal_rid);
}

void AtlasTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	Rect2 rc = region;

	if (!atlas.is_valid())
		return;

	if (rc.size.width == 0) {
		rc.size.width = atlas->get_width();
	}

	if (rc.size.height == 0) {
		rc.size.height = atlas->get_height();
	}

	Vector2 scale = p_rect.size / (region.size + margin.size);
	Rect2 dr(p_rect.position + margin.position * scale, rc.size * scale);

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, dr, atlas->get_rid(), rc, p_modulate, p_transpose, normal_rid);
}
void AtlasTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map, bool p_clip_uv) const {

	//this might not necessarily work well if using a rect, needs to be fixed properly
	Rect2 rc = region;

	if (!atlas.is_valid())
		return;

	Rect2 src = p_src_rect;
	src.position += (rc.position - margin.position);
	Rect2 src_c = rc.clip(src);
	if (src_c.size == Size2())
		return;
	Vector2 ofs = (src_c.position - src.position);

	Vector2 scale = p_rect.size / p_src_rect.size;
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

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	VS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, dr, atlas->get_rid(), src_c, p_modulate, p_transpose, normal_rid, p_clip_uv);
}

bool AtlasTexture::get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const {

	Rect2 rc = region;

	if (!atlas.is_valid())
		return false;

	Rect2 src = p_src_rect;
	src.position += (rc.position - margin.position);
	Rect2 src_c = rc.clip(src);
	if (src_c.size == Size2())
		return false;
	Vector2 ofs = (src_c.position - src.position);

	Vector2 scale = p_rect.size / p_src_rect.size;
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

AtlasTexture::AtlasTexture() {
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
		if (pieces[i].texture->has_alpha())
			return true;
	}

	return false;
}

void LargeTexture::set_flags(uint32_t p_flags) {

	for (int i = 0; i < pieces.size(); i++) {
		pieces[i].texture->set_flags(p_flags);
	}
}

uint32_t LargeTexture::get_flags() const {

	if (pieces.size())
		return pieces[0].texture->get_flags();

	return 0;
}

int LargeTexture::add_piece(const Point2 &p_offset, const Ref<Texture> &p_texture) {

	ERR_FAIL_COND_V(p_texture.is_null(), -1);
	Piece p;
	p.offset = p_offset;
	p.texture = p_texture;
	pieces.push_back(p);

	return pieces.size() - 1;
}

void LargeTexture::set_piece_offset(int p_idx, const Point2 &p_offset) {

	ERR_FAIL_INDEX(p_idx, pieces.size());
	pieces[p_idx].offset = p_offset;
};

void LargeTexture::set_piece_texture(int p_idx, const Ref<Texture> &p_texture) {

	ERR_FAIL_INDEX(p_idx, pieces.size());
	pieces[p_idx].texture = p_texture;
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
Ref<Texture> LargeTexture::get_piece_texture(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, pieces.size(), Ref<Texture>());
	return pieces[p_idx].texture;
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

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "_data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_data", "_get_data");
}

void LargeTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	for (int i = 0; i < pieces.size(); i++) {

		// TODO
		pieces[i].texture->draw(p_canvas_item, pieces[i].offset + p_pos, p_modulate, p_transpose, p_normal_map);
	}
}

void LargeTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map) const {

	//tiling not supported for this
	if (size.x == 0 || size.y == 0)
		return;

	Size2 scale = p_rect.size / size;

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	for (int i = 0; i < pieces.size(); i++) {

		// TODO
		pieces[i].texture->draw_rect(p_canvas_item, Rect2(pieces[i].offset * scale + p_rect.position, pieces[i].texture->get_size() * scale), false, p_modulate, p_transpose, p_normal_map);
	}
}
void LargeTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, const Ref<Texture> &p_normal_map, bool p_clip_uv) const {

	//tiling not supported for this
	if (p_src_rect.size.x == 0 || p_src_rect.size.y == 0)
		return;

	Size2 scale = p_rect.size / p_src_rect.size;

	RID normal_rid = p_normal_map.is_valid() ? p_normal_map->get_rid() : RID();
	for (int i = 0; i < pieces.size(); i++) {

		// TODO
		Rect2 rect(pieces[i].offset, pieces[i].texture->get_size());
		if (!p_src_rect.intersects(rect))
			continue;
		Rect2 local = p_src_rect.clip(rect);
		Rect2 target = local;
		target.size *= scale;
		target.position = p_rect.position + (p_src_rect.position + rect.position) * scale;
		local.position -= rect.position;
		pieces[i].texture->draw_rect_region(p_canvas_item, target, local, p_modulate, p_transpose, p_normal_map, false);
	}
}

LargeTexture::LargeTexture() {
}

///////////////////////////////////////////////

void CubeMap::set_flags(uint32_t p_flags) {

	flags = p_flags;
	if (_is_valid())
		VS::get_singleton()->texture_set_flags(cubemap, flags | VS::TEXTURE_FLAG_CUBEMAP);
}

uint32_t CubeMap::get_flags() const {

	return flags;
}

void CubeMap::set_side(Side p_side, const Ref<Image> &p_image) {

	ERR_FAIL_COND(p_image->empty());
	ERR_FAIL_INDEX(p_side, 6);
	if (!_is_valid()) {
		format = p_image->get_format();
		w = p_image->get_width();
		h = p_image->get_height();
		VS::get_singleton()->texture_allocate(cubemap, w, h, p_image->get_format(), flags | VS::TEXTURE_FLAG_CUBEMAP);
	}

	VS::get_singleton()->texture_set_data(cubemap, p_image, VS::CubeMapSide(p_side));
	valid[p_side] = true;
}

Ref<Image> CubeMap::get_side(Side p_side) const {

	if (!valid[p_side])
		return Ref<Image>();
	return VS::get_singleton()->texture_get_data(cubemap, VS::CubeMapSide(p_side));
}

Image::Format CubeMap::get_format() const {

	return format;
}
int CubeMap::get_width() const {

	return w;
}
int CubeMap::get_height() const {

	return h;
}

RID CubeMap::get_rid() const {

	return cubemap;
}

void CubeMap::set_storage(Storage p_storage) {

	storage = p_storage;
}

CubeMap::Storage CubeMap::get_storage() const {

	return storage;
}

void CubeMap::set_lossy_storage_quality(float p_lossy_storage_quality) {

	lossy_storage_quality = p_lossy_storage_quality;
}

float CubeMap::get_lossy_storage_quality() const {

	return lossy_storage_quality;
}

void CubeMap::set_path(const String &p_path, bool p_take_over) {

	if (cubemap.is_valid()) {
		VisualServer::get_singleton()->texture_set_path(cubemap, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

bool CubeMap::_set(const StringName &p_name, const Variant &p_value) {

	if (p_name == "side/left") {
		set_side(SIDE_LEFT, p_value);
	} else if (p_name == "side/right") {
		set_side(SIDE_RIGHT, p_value);
	} else if (p_name == "side/bottom") {
		set_side(SIDE_BOTTOM, p_value);
	} else if (p_name == "side/top") {
		set_side(SIDE_TOP, p_value);
	} else if (p_name == "side/front") {
		set_side(SIDE_FRONT, p_value);
	} else if (p_name == "side/back") {
		set_side(SIDE_BACK, p_value);
	} else if (p_name == "flags") {
		set_flags(p_value);
	} else if (p_name == "storage") {
		storage = Storage(p_value.operator int());
	} else if (p_name == "lossy_quality") {
		lossy_storage_quality = p_value;
	} else
		return false;

	return true;
}

bool CubeMap::_get(const StringName &p_name, Variant &r_ret) const {

	if (p_name == "side/left") {
		r_ret = get_side(SIDE_LEFT);
	} else if (p_name == "side/right") {
		r_ret = get_side(SIDE_RIGHT);
	} else if (p_name == "side/bottom") {
		r_ret = get_side(SIDE_BOTTOM);
	} else if (p_name == "side/top") {
		r_ret = get_side(SIDE_TOP);
	} else if (p_name == "side/front") {
		r_ret = get_side(SIDE_FRONT);
	} else if (p_name == "side/back") {
		r_ret = get_side(SIDE_BACK);
	} else if (p_name == "flags") {
		r_ret = flags;
	} else if (p_name == "storage") {
		r_ret = storage;
	} else if (p_name == "lossy_quality") {
		r_ret = lossy_storage_quality;
	} else
		return false;

	return true;
}

void CubeMap::_get_property_list(List<PropertyInfo> *p_list) const {

	PropertyHint img_hint = PROPERTY_HINT_NONE;
	if (storage == STORAGE_COMPRESS_LOSSY) {
		img_hint = PROPERTY_HINT_IMAGE_COMPRESS_LOSSY;
	} else if (storage == STORAGE_COMPRESS_LOSSLESS) {
		img_hint = PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS;
	}

	p_list->push_back(PropertyInfo(Variant::INT, "flags", PROPERTY_HINT_FLAGS, "Mipmaps,Repeat,Filter"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "side/left", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "side/right", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "side/bottom", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "side/top", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "side/front", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
	p_list->push_back(PropertyInfo(Variant::OBJECT, "side/back", PROPERTY_HINT_RESOURCE_TYPE, "Image"));
}

void CubeMap::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_width"), &CubeMap::get_width);
	ClassDB::bind_method(D_METHOD("get_height"), &CubeMap::get_height);
	ClassDB::bind_method(D_METHOD("set_flags", "flags"), &CubeMap::set_flags);
	ClassDB::bind_method(D_METHOD("get_flags"), &CubeMap::get_flags);
	ClassDB::bind_method(D_METHOD("set_side", "side", "image"), &CubeMap::set_side);
	ClassDB::bind_method(D_METHOD("get_side", "side"), &CubeMap::get_side);
	ClassDB::bind_method(D_METHOD("set_storage", "mode"), &CubeMap::set_storage);
	ClassDB::bind_method(D_METHOD("get_storage"), &CubeMap::get_storage);
	ClassDB::bind_method(D_METHOD("set_lossy_storage_quality", "quality"), &CubeMap::set_lossy_storage_quality);
	ClassDB::bind_method(D_METHOD("get_lossy_storage_quality"), &CubeMap::get_lossy_storage_quality);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "storage_mode", PROPERTY_HINT_ENUM, "Raw,Lossy Compressed,Lossless Compressed"), "set_storage", "get_storage");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "lossy_storage_quality"), "set_lossy_storage_quality", "get_lossy_storage_quality");

	BIND_ENUM_CONSTANT(STORAGE_RAW);
	BIND_ENUM_CONSTANT(STORAGE_COMPRESS_LOSSY);
	BIND_ENUM_CONSTANT(STORAGE_COMPRESS_LOSSLESS);

	BIND_ENUM_CONSTANT(SIDE_LEFT);
	BIND_ENUM_CONSTANT(SIDE_RIGHT);
	BIND_ENUM_CONSTANT(SIDE_BOTTOM);
	BIND_ENUM_CONSTANT(SIDE_TOP);
	BIND_ENUM_CONSTANT(SIDE_FRONT);
	BIND_ENUM_CONSTANT(SIDE_BACK);

	BIND_ENUM_CONSTANT(FLAG_MIPMAPS);
	BIND_ENUM_CONSTANT(FLAG_REPEAT);
	BIND_ENUM_CONSTANT(FLAG_FILTER);
	BIND_ENUM_CONSTANT(FLAGS_DEFAULT);
}

CubeMap::CubeMap() {

	w = h = 0;
	flags = FLAGS_DEFAULT;
	for (int i = 0; i < 6; i++)
		valid[i] = false;
	cubemap = VisualServer::get_singleton()->texture_create();
	storage = STORAGE_RAW;
	lossy_storage_quality = 0.7;
}

CubeMap::~CubeMap() {

	VisualServer::get_singleton()->free(cubemap);
}

/*	BIND_ENUM(CubeMapSize);
	BIND_ENUM_CONSTANT( FLAG_CUBEMAP );
	BIND_ENUM_CONSTANT( CUBEMAP_LEFT );
	BIND_ENUM_CONSTANT( CUBEMAP_RIGHT );
	BIND_ENUM_CONSTANT( CUBEMAP_BOTTOM );
	BIND_ENUM_CONSTANT( CUBEMAP_TOP );
	BIND_ENUM_CONSTANT( CUBEMAP_FRONT );
	BIND_ENUM_CONSTANT( CUBEMAP_BACK );
*/
///////////////////////////

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
			_curve->disconnect(CoreStringNames::get_singleton()->changed, this, "_update");
		}
		_curve = p_curve;
		if (_curve.is_valid()) {
			_curve->connect(CoreStringNames::get_singleton()->changed, this, "_update");
		}
		_update();
	}
}

void CurveTexture::_update() {

	PoolVector<uint8_t> data;
	data.resize(_width * sizeof(float));

	// The array is locked in that scope
	{
		PoolVector<uint8_t>::Write wd8 = data.write();
		float *wd = (float *)wd8.ptr();

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

	VS::get_singleton()->texture_allocate(_texture, _width, 1, Image::FORMAT_RF, VS::TEXTURE_FLAG_FILTER);
	VS::get_singleton()->texture_set_data(_texture, image);

	emit_changed();
}

Ref<Curve> CurveTexture::get_curve() const {

	return _curve;
}

RID CurveTexture::get_rid() const {

	return _texture;
}

CurveTexture::CurveTexture() {
	_width = 2048;
	_texture = VS::get_singleton()->texture_create();
}
CurveTexture::~CurveTexture() {
	VS::get_singleton()->free(_texture);
}
//////////////////

//setter and getter names for property serialization
#define COLOR_RAMP_GET_OFFSETS "get_offsets"
#define COLOR_RAMP_GET_COLORS "get_colors"
#define COLOR_RAMP_SET_OFFSETS "set_offsets"
#define COLOR_RAMP_SET_COLORS "set_colors"

GradientTexture::GradientTexture() {
	update_pending = false;
	width = 2048;

	texture = VS::get_singleton()->texture_create();
	_queue_update();
}

GradientTexture::~GradientTexture() {
	VS::get_singleton()->free(texture);
}

void GradientTexture::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_gradient", "gradient"), &GradientTexture::set_gradient);
	ClassDB::bind_method(D_METHOD("get_gradient"), &GradientTexture::get_gradient);

	ClassDB::bind_method(D_METHOD("set_width", "width"), &GradientTexture::set_width);

	ClassDB::bind_method(D_METHOD("_update"), &GradientTexture::_update);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "gradient", PROPERTY_HINT_RESOURCE_TYPE, "Gradient"), "set_gradient", "get_gradient");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "width"), "set_width", "get_width");
}

void GradientTexture::set_gradient(Ref<Gradient> p_gradient) {
	if (p_gradient == gradient)
		return;
	if (gradient.is_valid()) {
		gradient->disconnect(CoreStringNames::get_singleton()->changed, this, "_update");
	}
	gradient = p_gradient;
	if (gradient.is_valid()) {
		gradient->connect(CoreStringNames::get_singleton()->changed, this, "_update");
	}
	_update();
	emit_changed();
}

Ref<Gradient> GradientTexture::get_gradient() const {
	return gradient;
}

void GradientTexture::_queue_update() {

	if (update_pending)
		return;

	call_deferred("_update");
}

void GradientTexture::_update() {

	if (gradient.is_null())
		return;

	update_pending = false;

	PoolVector<uint8_t> data;
	data.resize(width * 4);
	{
		PoolVector<uint8_t>::Write wd8 = data.write();
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

	VS::get_singleton()->texture_allocate(texture, width, 1, Image::FORMAT_RGBA8, VS::TEXTURE_FLAG_FILTER);
	VS::get_singleton()->texture_set_data(texture, image);

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
	return VisualServer::get_singleton()->texture_get_data(texture);
}
