/**************************************************************************/
/*  image_texture.cpp                                                     */
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

#include "image_texture.h"

#include "core/io/image_loader.h"
#include "scene/main/canvas_item.h"
#include "scene/main/viewport.h"
#include "scene/resources/bit_map.h"
#include "scene/resources/placeholder_textures.h"

#include "modules/modules_enabled.gen.h" // For svg.
#ifdef MODULE_SVG_ENABLED
#include "modules/svg/image_loader_svg.h"
#endif

Mutex ImageTexture::mutex;
HashMap<double, ImageTexture::ScalingLevel> ImageTexture::scaling_levels;

void ImageTexture::reference_scaling_level(double p_scale) {
	uint32_t oversampling = CLAMP(p_scale, 0.1, 100.0) * 64;
	if (oversampling == 64) {
		return;
	}
	double scale = double(oversampling) / 64.0;

	MutexLock lock(mutex);
	ScalingLevel *sl = scaling_levels.getptr(scale);
	if (sl) {
		sl->refcount++;
	} else {
		ScalingLevel new_sl;
		scaling_levels.insert(scale, new_sl);
	}
}

void ImageTexture::unreference_scaling_level(double p_scale) {
	uint32_t oversampling = CLAMP(p_scale, 0.1, 100.0) * 64;
	if (oversampling == 64) {
		return;
	}
	double scale = double(oversampling) / 64.0;

	MutexLock lock(mutex);
	ScalingLevel *sl = scaling_levels.getptr(scale);
	if (sl) {
		sl->refcount--;
		if (sl->refcount == 0) {
			for (ImageTexture *tx : sl->textures) {
				tx->_remove_scale(scale);
			}
			sl->textures.clear();
			scaling_levels.erase(scale);
		}
	}
}

void ImageTexture::reload_from_file() {
	String path = ResourceLoader::path_remap(get_path());
	if (!path.is_resource_file()) {
		return;
	}

	if (path.get_extension().to_lower() == "svg") {
		Error err = OK;
		String src = FileAccess::get_file_as_string(path, &err);
		if (err == OK) {
			set_source(src);
		} else {
			Resource::reload_from_file();
			notify_property_list_changed();
			emit_changed();
		}
	} else {
		Ref<Image> img;
		img.instantiate();

		if (ImageLoader::load_image(path, img) == OK) {
			set_image(img);
		} else {
			Resource::reload_from_file();
			notify_property_list_changed();
			emit_changed();
		}
	}
}

Ref<ImageTexture> ImageTexture::create_from_image(const Ref<Image> &p_image) {
	ERR_FAIL_COND_V_MSG(p_image.is_null(), Ref<ImageTexture>(), "Invalid image: null");
	ERR_FAIL_COND_V_MSG(p_image->is_empty(), Ref<ImageTexture>(), "Invalid image: image is empty");

	Ref<ImageTexture> image_texture;
	image_texture.instantiate();
	image_texture->set_image(p_image);
	return image_texture;
}

Ref<ImageTexture> ImageTexture::create_from_string(const String &p_source, float p_scale, float p_saturation, const Dictionary &p_color_map) {
	Ref<ImageTexture> image_texture;
	image_texture.instantiate();
	image_texture->set_source(p_source);
	image_texture->set_base_scale(p_scale);
	image_texture->set_saturation(p_saturation);
	image_texture->set_color_map(p_color_map);
	return image_texture;
}

void ImageTexture::set_image(const Ref<Image> &p_image) {
	ERR_FAIL_COND_MSG(p_image.is_null() || p_image->is_empty(), "Invalid image");
	size.x = p_image->get_width();
	size.y = p_image->get_height();
	format = p_image->get_format();
	mipmaps = p_image->has_mipmaps();
	source = String();

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
	_ensure_scale(1.0);
	return format;
}

void ImageTexture::_clear() {
	for (KeyValue<double, RID> &tx : texture_cache) {
		if (tx.value.is_valid()) {
			RenderingServer::get_singleton()->free(tx.value);
		}
	}
	texture_cache.clear();
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->free(texture);
	}
	texture = RID();
	alpha_cache.unref();
}

void ImageTexture::_update_texture() {
	_clear();
	emit_changed();
}

void ImageTexture::_remove_scale(double p_scale) {
	if (Math::is_equal_approx(p_scale, 1.0)) {
		return;
	}

	RID *rid = texture_cache.getptr(p_scale);
	if (rid) {
		if (rid->is_valid()) {
			RenderingServer::get_singleton()->free(*rid);
		}
		texture_cache.erase(p_scale);
	}
}

RID ImageTexture::_ensure_scale(double p_scale) const {
	if (source.is_empty()) {
		return texture;
	}
	uint32_t oversampling = CLAMP(p_scale, 0.1, 100.0) * 64;
	if (oversampling == 64) {
		if (texture.is_null()) {
			texture = _load_at_scale(p_scale, true);
		}
		return texture;
	}
	double scale = double(oversampling) / 64.0;

	RID *rid = texture_cache.getptr(scale);
	if (rid) {
		return *rid;
	}

	MutexLock lock(mutex);
	ScalingLevel *sl = scaling_levels.getptr(scale);
	ERR_FAIL_NULL_V_MSG(sl, RID(), "Invalid scaling level");
	sl->textures.insert(const_cast<ImageTexture *>(this));

	RID new_rid = _load_at_scale(scale, false);
	texture_cache[scale] = new_rid;
	return new_rid;
}

RID ImageTexture::_load_at_scale(double p_scale, bool p_set_size) const {
	Ref<Image> img;
	img.instantiate();
#ifdef MODULE_SVG_ENABLED
	const bool upsample = !Math::is_equal_approx(Math::round(p_scale * base_scale), p_scale * base_scale);

	Error err = ImageLoaderSVG::create_image_from_string(img, source, p_scale * base_scale, upsample, cmap);
	if (err != OK) {
		return RID();
	}
#else
	img = Image::create_empty(Math::round(16 * p_scale * base_scale), Math::round(16 * p_scale * base_scale), false, Image::FORMAT_RGBA8);
#endif
	if (saturation != 1.0) {
		img->adjust_bcs(1.0, 1.0, saturation);
	}

	Size2 current_size = size;
	if (p_set_size) {
		size.x = img->get_width();
		//base_size.x = img->get_width();
		if (size_override.x != 0) {
			size.x = size_override.x;
		}
		size.y = img->get_height();
		//base_size.y = img->get_height();
		if (size_override.y != 0) {
			size.y = size_override.y;
		}
		format = img->get_format();
		mipmaps = img->has_mipmaps();
		current_size = size;
	}
	if (current_size.is_zero_approx()) {
		current_size.x = img->get_width();
		current_size.y = img->get_height();
	}

	RID rid = RenderingServer::get_singleton()->texture_2d_create(img);
	RenderingServer::get_singleton()->texture_set_size_override(rid, current_size.x, current_size.y);
	image_stored = true;
	return rid;
}

void ImageTexture::update(const Ref<Image> &p_image) {
	ERR_FAIL_COND_MSG(p_image.is_null(), "Invalid image");
	ERR_FAIL_COND_MSG(texture.is_null(), "Texture is not initialized.");
	ERR_FAIL_COND_MSG(p_image->get_width() != size.x || p_image->get_height() != size.y,
			"The new image dimensions must match the texture size.");
	ERR_FAIL_COND_MSG(p_image->get_format() != format,
			"The new image format must match the texture's image format.");
	ERR_FAIL_COND_MSG(mipmaps != p_image->has_mipmaps(),
			"The new image mipmaps configuration must match the texture's image mipmaps configuration");

	RS::get_singleton()->texture_2d_update(texture, p_image);

	notify_property_list_changed();
	emit_changed();

	for (KeyValue<double, RID> &tx : texture_cache) {
		if (tx.value.is_valid()) {
			RenderingServer::get_singleton()->free(tx.value);
		}
	}
	texture_cache.clear();
	alpha_cache.unref();
	source = String();
	image_stored = true;
}

Ref<Image> ImageTexture::get_image() const {
	_ensure_scale(1.0);
	if (image_stored) {
		return RenderingServer::get_singleton()->texture_2d_get(texture);
	} else {
		return Ref<Image>();
	}
}

void ImageTexture::set_source(const String &p_source) {
	if (source == p_source) {
		return;
	}
	source = p_source;
	_update_texture();
}

String ImageTexture::get_source() const {
	return source;
}

void ImageTexture::set_base_scale(float p_scale) {
	if (base_scale == p_scale) {
		return;
	}
	ERR_FAIL_COND(p_scale <= 0.0);

	base_scale = p_scale;
	_update_texture();
}

float ImageTexture::get_base_scale() const {
	return base_scale;
}

void ImageTexture::set_saturation(float p_saturation) {
	if (saturation == p_saturation) {
		return;
	}

	saturation = p_saturation;
	_update_texture();
}

float ImageTexture::get_saturation() const {
	return saturation;
}

void ImageTexture::set_color_map(const Dictionary &p_color_map) {
	if (color_map == p_color_map) {
		return;
	}
	color_map = p_color_map;
	cmap.clear();
	for (const Variant *E = color_map.next(); E; E = color_map.next(E)) {
		cmap[*E] = color_map[*E];
	}
	_update_texture();
}

Dictionary ImageTexture::get_color_map() const {
	return color_map;
}

int ImageTexture::get_width() const {
	_ensure_scale(1.0);
	return size.x;
}

int ImageTexture::get_height() const {
	_ensure_scale(1.0);
	return size.y;
}

RID ImageTexture::get_rid() const {
	_ensure_scale(1.0);
	if (texture.is_null()) {
		// We are in trouble, create something temporary.
		texture = RenderingServer::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

RID ImageTexture::get_scaled_rid() const {
	if (source.is_empty()) {
		return get_rid();
	}

	double scale = 1.0;
	CanvasItem *ci = CanvasItem::get_current_item_drawn();
	if (ci) {
		Viewport *vp = ci->get_viewport();
		if (vp) {
			scale = vp->get_oversampling();
		}
	}
	return _ensure_scale(scale);
}

bool ImageTexture::has_alpha() const {
	return (format == Image::FORMAT_LA8 || format == Image::FORMAT_RGBA8);
}

void ImageTexture::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if (size.x == 0 || size.y == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, size), get_scaled_rid(), false, p_modulate, p_transpose);
}

void ImageTexture::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if (size.x == 0 || size.y == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, get_scaled_rid(), p_tile, p_modulate, p_transpose);
}

void ImageTexture::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if (size.x == 0 || size.y == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, get_scaled_rid(), p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

bool ImageTexture::is_pixel_opaque(int p_x, int p_y) const {
	if (alpha_cache.is_null()) {
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

		int x = p_x * aw / size.x;
		int y = p_y * ah / size.y;

		x = CLAMP(x, 0, aw - 1);
		y = CLAMP(y, 0, ah - 1);

		return alpha_cache->get_bit(x, y);
	}

	return true;
}

void ImageTexture::set_size_override(const Size2i &p_size) {
	Size2i s = p_size;
	if (s.x != 0) {
		size.x = s.x;
	}
	if (s.y != 0) {
		size.y = s.y;
	}
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_size_override(texture, size.x, size.y);
	}
	for (KeyValue<double, RID> &tx : texture_cache) {
		if (tx.value.is_valid()) {
			RenderingServer::get_singleton()->texture_set_size_override(tx.value, size.x, size.y);
		}
	}
}

void ImageTexture::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

bool ImageTexture::_set(const StringName &p_name, const Variant &p_value) {
	if (p_name == "image") {
		set_image(p_value);
		return true;
	} else if (p_name == "_source") {
		set_source(p_value);
		return true;
	} else if (p_name == "base_scale") {
		set_base_scale(p_value);
		return true;
	} else if (p_name == "saturation") {
		set_saturation(p_value);
		return true;
	} else if (p_name == "color_map") {
		set_color_map(p_value);
		return true;
	}
	return false;
}

bool ImageTexture::_get(const StringName &p_name, Variant &r_ret) const {
	if (p_name == "image") {
		r_ret = get_image();
		return true;
	} else if (p_name == "_source") {
		r_ret = get_source();
		return true;
	} else if (p_name == "base_scale") {
		r_ret = get_base_scale();
		return true;
	} else if (p_name == "saturation") {
		r_ret = get_saturation();
		return true;
	} else if (p_name == "color_map") {
		r_ret = get_color_map();
		return true;
	}
	return false;
}

void ImageTexture::_get_property_list(List<PropertyInfo> *p_list) const {
	p_list->push_back(PropertyInfo(Variant::OBJECT, PNAME("image"), PROPERTY_HINT_RESOURCE_TYPE, "Image", PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT));
	p_list->push_back(PropertyInfo(Variant::STRING, "_source", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_STORAGE));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "base_scale", PROPERTY_HINT_RANGE, "0.01,10.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::FLOAT, "saturation", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"));
	p_list->push_back(PropertyInfo(Variant::DICTIONARY, "color_map", PROPERTY_HINT_DICTIONARY_TYPE, "Color;Color"));
}

void ImageTexture::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "base_scale" || p_property.name == "saturation" || p_property.name == "color_map") {
		if (source.is_empty()) {
			p_property.usage = PROPERTY_USAGE_NO_EDITOR;
			return;
		}
	}
}

void ImageTexture::_bind_methods() {
	ClassDB::bind_static_method("ImageTexture", D_METHOD("create_from_image", "image"), &ImageTexture::create_from_image);
	ClassDB::bind_static_method("ImageTexture", D_METHOD("create_from_string", "source", "scale", "saturation", "color_map"), &ImageTexture::create_from_string, DEFVAL(1.0), DEFVAL(1.0), DEFVAL(Dictionary()));

	ClassDB::bind_method(D_METHOD("set_source", "source"), &ImageTexture::set_source);
	ClassDB::bind_method(D_METHOD("get_source"), &ImageTexture::get_source);
	ClassDB::bind_method(D_METHOD("set_base_scale", "base_scale"), &ImageTexture::set_base_scale);
	ClassDB::bind_method(D_METHOD("get_base_scale"), &ImageTexture::get_base_scale);
	ClassDB::bind_method(D_METHOD("set_saturation", "saturation"), &ImageTexture::set_saturation);
	ClassDB::bind_method(D_METHOD("get_saturation"), &ImageTexture::get_saturation);
	ClassDB::bind_method(D_METHOD("set_color_map", "color_map"), &ImageTexture::set_color_map);
	ClassDB::bind_method(D_METHOD("get_color_map"), &ImageTexture::get_color_map);

	ClassDB::bind_method(D_METHOD("get_scaled_rid"), &ImageTexture::get_scaled_rid);

	ClassDB::bind_method(D_METHOD("get_format"), &ImageTexture::get_format);

	ClassDB::bind_method(D_METHOD("set_image", "image"), &ImageTexture::set_image);
	ClassDB::bind_method(D_METHOD("update", "image"), &ImageTexture::update);
	ClassDB::bind_method(D_METHOD("set_size_override", "size"), &ImageTexture::set_size_override);
}

ImageTexture::~ImageTexture() {
	_clear();

	MutexLock lock(mutex);
	for (KeyValue<double, ScalingLevel> &sl : scaling_levels) {
		sl.value.textures.erase(this);
	}
}

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

Error ImageTextureLayered::_create_from_images(const TypedArray<Image> &p_images) {
	Vector<Ref<Image>> images;
	for (int i = 0; i < p_images.size(); i++) {
		Ref<Image> img = p_images[i];
		ERR_FAIL_COND_V(img.is_null(), ERR_INVALID_PARAMETER);
		images.push_back(img);
	}

	return create_from_images(images);
}

TypedArray<Image> ImageTextureLayered::_get_images() const {
	TypedArray<Image> images;
	for (int i = 0; i < layers; i++) {
		images.push_back(get_layer_data(i));
	}
	return images;
}

void ImageTextureLayered::_set_images(const TypedArray<Image> &p_images) {
	ERR_FAIL_COND(_create_from_images(p_images) != OK);
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
	ERR_FAIL_COND_MSG(texture.is_null(), "Texture is not initialized.");
	ERR_FAIL_COND_MSG(p_image.is_null(), "Invalid image.");
	ERR_FAIL_COND_MSG(p_image->get_format() != format, "Image format must match texture's image format.");
	ERR_FAIL_COND_MSG(p_image->get_width() != width || p_image->get_height() != height, "Image size must match texture's image size.");
	ERR_FAIL_COND_MSG(p_image->has_mipmaps() != mipmaps, "Image mipmap configuration must match texture's image mipmap configuration.");
	ERR_FAIL_INDEX_MSG(p_layer, layers, "Layer index is out of bounds.");
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
	ClassDB::bind_method(D_METHOD("_set_images", "images"), &ImageTextureLayered::_set_images);

	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "_images", PROPERTY_HINT_ARRAY_TYPE, "Image", PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT), "_set_images", "_get_images");
}

ImageTextureLayered::ImageTextureLayered(LayeredType p_layered_type) {
	layered_type = p_layered_type;
}

ImageTextureLayered::~ImageTextureLayered() {
	if (texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RS::get_singleton()->free(texture);
	}
}

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
	} else {
		texture = tex;
	}

	format = p_format;
	width = p_width;
	height = p_height;
	depth = p_depth;
	mipmaps = p_mipmaps;

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

TypedArray<Image> ImageTexture3D::_get_images() const {
	TypedArray<Image> images;
	if (texture.is_valid()) {
		Vector<Ref<Image>> raw_images = get_data();
		ERR_FAIL_COND_V(raw_images.is_empty(), TypedArray<Image>());

		for (int i = 0; i < raw_images.size(); i++) {
			images.push_back(raw_images[i]);
		}
	}
	return images;
}

void ImageTexture3D::_set_images(const TypedArray<Image> &p_images) {
	int new_layers = p_images.size();
	ERR_FAIL_COND(new_layers == 0);
	Ref<Image> img_base = p_images[0];
	ERR_FAIL_COND(img_base.is_null());

	Image::Format new_format = img_base->get_format();
	int new_width = img_base->get_width();
	int new_height = img_base->get_height();
	int new_depth = 0;
	bool new_mipmaps = false;

	for (int i = 1; i < p_images.size(); i++) {
		Ref<Image> img = p_images[i];
		ERR_FAIL_COND(img.is_null());
		ERR_FAIL_COND_MSG(img->get_format() != new_format, "All images must share the same format.");

		if (img->get_width() != new_width || img->get_height() != new_height) {
			new_mipmaps = true;
			if (new_depth == 0) {
				new_depth = i;
			}
		}
	}

	if (new_depth == 0) {
		new_depth = p_images.size();
	}

	Error err = _create(new_format, new_width, new_height, new_depth, new_mipmaps, p_images);
	ERR_FAIL_COND(err != OK);
}

void ImageTexture3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create", "format", "width", "height", "depth", "use_mipmaps", "data"), &ImageTexture3D::_create);
	ClassDB::bind_method(D_METHOD("update", "data"), &ImageTexture3D::_update);
	ClassDB::bind_method(D_METHOD("_get_images"), &ImageTexture3D::_get_images);
	ClassDB::bind_method(D_METHOD("_set_images", "images"), &ImageTexture3D::_set_images);
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "_images", PROPERTY_HINT_ARRAY_TYPE, "Image", PROPERTY_USAGE_INTERNAL | PROPERTY_USAGE_STORAGE | PROPERTY_USAGE_RESOURCE_NOT_PERSISTENT), "_set_images", "_get_images");
}

ImageTexture3D::ImageTexture3D() {
}

ImageTexture3D::~ImageTexture3D() {
	if (texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RS::get_singleton()->free(texture);
	}
}

void Texture2DArray::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_placeholder"), &Texture2DArray::create_placeholder);
}

Ref<Resource> Texture2DArray::create_placeholder() const {
	Ref<PlaceholderTexture2DArray> placeholder;
	placeholder.instantiate();
	placeholder->set_size(Size2i(get_width(), get_height()));
	placeholder->set_layers(get_layers());
	return placeholder;
}

void Cubemap::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_placeholder"), &Cubemap::create_placeholder);
}

Ref<Resource> Cubemap::create_placeholder() const {
	Ref<PlaceholderCubemap> placeholder;
	placeholder.instantiate();
	placeholder->set_size(Size2i(get_width(), get_height()));
	placeholder->set_layers(get_layers());
	return placeholder;
}

void CubemapArray::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_placeholder"), &CubemapArray::create_placeholder);
}

Ref<Resource> CubemapArray::create_placeholder() const {
	Ref<PlaceholderCubemapArray> placeholder;
	placeholder.instantiate();
	placeholder->set_size(Size2i(get_width(), get_height()));
	placeholder->set_layers(get_layers());
	return placeholder;
}
