/**************************************************************************/
/*  streamed_texture.cpp                                                  */
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

#include "streamed_texture.h"

#include "core/config/project_settings.h"
#include "core/io/file_access.h"
#include "core/io/image.h"
#include "scene/resources/bit_map.h"
#include "scene/resources/compressed_texture.h"

#include "modules/modules_enabled.gen.h"

#ifdef MODULE_TEXTURE_STREAMING_ENABLED
#include "modules/texture_streaming/texture_streaming.h"
#endif

StreamedTexture2D::TextureFormatRoughnessRequestCallback StreamedTexture2D::request_roughness_callback = nullptr;
StreamedTexture2D::TextureFormatRequestCallback StreamedTexture2D::request_normal_callback = nullptr;

void StreamedTexture2D::_requested_roughness(void *p_ud, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_roughness_channel) {
	StreamedTexture2D *ct = (StreamedTexture2D *)p_ud;
	Ref<StreamedTexture2D> ctex(ct);
	ERR_FAIL_NULL(request_roughness_callback);
	request_roughness_callback(ctex, p_normal_path, p_roughness_channel);
}

void StreamedTexture2D::_requested_normal(void *p_ud) {
	StreamedTexture2D *ct = (StreamedTexture2D *)p_ud;
	Ref<StreamedTexture2D> ctex(ct);
	ERR_FAIL_NULL(request_normal_callback);
	request_normal_callback(ctex);
}

void StreamedTexture2D::texture_reload(uint32_t p_resolution) {
	ERR_FAIL_COND(texture.is_null());
	ERR_FAIL_COND(path_to_file.is_empty());

	if (_current_resolution == p_resolution) {
		return;
	}

	// Limit to minimum 4 to avoid too small textures.
	_current_resolution = MAX(4u, p_resolution);

	{
		StreamedTexture2DLoadData load_data;
		load_data.image.instantiate();
		Error err = _load_data(path_to_file, _current_resolution, load_data);
		ERR_FAIL_COND(err != OK);

		RID new_texture = RS::get_singleton()->texture_2d_create(load_data.image);
		RenderingServer::get_singleton()->texture_set_path(new_texture, path_to_file);

		if (texture.is_valid()) {
			RS::get_singleton()->texture_replace(texture, new_texture);
		} else {
			texture = new_texture;
		}
	}
}

Error StreamedTexture2D::_load_data(const String &p_path, const uint32_t p_max_resolution, StreamedTexture2DLoadData &p_load_data) {
	ERR_FAIL_COND_V(p_load_data.image.is_null(), ERR_INVALID_PARAMETER);

	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::READ);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_OPEN, vformat("Unable to open file: %s.", p_path));

	// Check for Godot Streamed Texture Format header (ie, GSTF).
	uint8_t header[4];
	f->get_buffer(header, 4);
	if (header[0] != 'G' || header[1] != 'S' || header[2] != 'T' || header[3] != 'F') {
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Streamed texture file is corrupt (Bad header).");
	}

	// Version.
	uint32_t version = f->get_32();

	constexpr uint32_t FORMAT_VERSION = 1;
	if (version > FORMAT_VERSION) {
		ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Streamed texture file is too new.");
	}

	// Read width, height.
	p_load_data.width = f->get_32();
	p_load_data.height = f->get_32();
	ERR_FAIL_COND_V_MSG(p_load_data.width == 0 || p_load_data.height == 0, ERR_FILE_CORRUPT, "Streamed texture has invalid dimensions.");
	ERR_FAIL_COND_V_MSG(p_load_data.width > 16384 || p_load_data.height > 16384, ERR_FILE_CORRUPT, "Streamed texture dimensions are too large.");

	// Read data size.
	uint32_t data_size = f->get_32();

	// Read image format.
	uint32_t format_value = f->get_32();
	ERR_FAIL_COND_V_MSG(format_value >= Image::FORMAT_MAX, ERR_FILE_CORRUPT, "Streamed texture has invalid image format.");
	p_load_data.format = (Image::Format)format_value;

	// Read mipmap count.
	uint32_t num_mipmaps = f->get_32();
	ERR_FAIL_COND_V_MSG(num_mipmaps > 16, ERR_FILE_CORRUPT, "Streamed texture has too many mipmaps.");

	// Read the offset and size for each mip level.
	LocalVector<Pair<uint32_t, uint32_t>> mipmap_offset_size;
	mipmap_offset_size.resize(num_mipmaps + 1);
	for (uint32_t i = 0; i <= num_mipmaps; i++) {
		mipmap_offset_size[i].first = f->get_32();
		mipmap_offset_size[i].second = f->get_32();
	}

	// Read flags (request_normal, request_roughness).
	uint32_t flags = f->get_32();
#ifdef TOOLS_ENABLED
	p_load_data.request_normal = request_normal_callback && (flags & FORMAT_BIT_DETECT_NORMAL);
	p_load_data.request_roughness = request_roughness_callback && (flags & FORMAT_BIT_DETECT_ROUGHNESS);
#else
	(void)flags; // Flags must be read to advance file position, but are unused in non-tools builds.
	p_load_data.request_normal = false;
	p_load_data.request_roughness = false;
#endif

	// Read streaming min size.
	p_load_data.streaming_min = f->get_32();

	// Read streaming max size.
	p_load_data.streaming_max = f->get_32();

	// Reserved for future use.
	f->get_32();

	// Validate that the file contains enough data.
	uint64_t header_size = f->get_position();
	uint64_t file_length = f->get_length();
	ERR_FAIL_COND_V_MSG(file_length < header_size + data_size, ERR_FILE_CORRUPT, "Streamed texture file is truncated (file too small for declared data size).");

	// Determine which mipmap level to load based on max resolution.
	uint32_t load_mip = 0;
	if (p_max_resolution > 0 && p_max_resolution < p_load_data.width && p_max_resolution < p_load_data.height) {
		// Find the mip level that corresponds to the requested resolution.
		load_mip = uint32_t(log2(MAX(p_load_data.width, p_load_data.height) / p_max_resolution));
		load_mip = MIN(load_mip, num_mipmaps);
	}

	uint32_t load_offset = mipmap_offset_size[load_mip].first;
	ERR_FAIL_COND_V_MSG(load_offset > data_size, ERR_FILE_CORRUPT, "Streamed texture has invalid mipmap offset.");
	uint32_t load_size = data_size - load_offset;

	// Calculate the dimensions at this mip level.
	uint32_t mip_width = MAX(1u, p_load_data.width >> load_mip);
	uint32_t mip_height = MAX(1u, p_load_data.height >> load_mip);
	uint32_t mip_count = num_mipmaps - load_mip;

	Vector<uint8_t> data;
	data.resize(load_size);
	f->seek(f->get_position() + load_offset);
	uint64_t bytes_read = f->get_buffer(data.ptrw(), data.size());
	ERR_FAIL_COND_V_MSG(bytes_read != load_size, ERR_FILE_CORRUPT, "Streamed texture file is truncated.");

	p_load_data.image->set_data(mip_width, mip_height, mip_count > 0, p_load_data.format, data);

	return OK;
}

Error StreamedTexture2D::_save_data(const String &p_path, const Ref<Image> &p_image, uint32_t p_flags, uint32_t p_streaming_min, uint32_t p_streaming_max) {
	ERR_FAIL_COND_V(p_image.is_null(), ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V(p_image->is_empty(), ERR_INVALID_PARAMETER);

	Ref<FileAccess> f = FileAccess::open(p_path, FileAccess::WRITE);
	ERR_FAIL_COND_V_MSG(f.is_null(), ERR_CANT_CREATE, vformat("Unable to create file: %s.", p_path));

	// Write Godot Streamed Texture Format header (ie, GSTF).
	f->store_8('G');
	f->store_8('S');
	f->store_8('T');
	f->store_8('F');

	// Version.
	constexpr uint32_t FORMAT_VERSION = 1;
	f->store_32(FORMAT_VERSION);

	// Write width, height.
	f->store_32(p_image->get_width());
	f->store_32(p_image->get_height());

	// Get image data.
	Vector<uint8_t> data = p_image->get_data();

	// Write data size.
	f->store_32(data.size());

	// Write image format.
	f->store_32((uint32_t)p_image->get_format());

	// Write mipmap count.
	uint32_t num_mipmaps = p_image->get_mipmap_count();
	f->store_32(num_mipmaps);

	// Write the offset and size for each mip level.
	for (uint32_t i = 0; i <= num_mipmaps; i++) {
		int64_t mip_offset, mip_size;
		p_image->get_mipmap_offset_and_size(i, mip_offset, mip_size);
		f->store_32((uint32_t)mip_offset);
		f->store_32((uint32_t)mip_size);
	}

	// Write flags (request_normal, request_roughness).
	f->store_32(p_flags);

	// Write streaming min size.
	f->store_32(p_streaming_min);

	// Write streaming max size.
	f->store_32(p_streaming_max);

	// Reserved for future use.
	f->store_32(0);

	// Write image data.
	f->store_buffer(data.ptr(), data.size());

	return OK;
}

Error StreamedTexture2D::_load_internal(const String &p_path, bool p_load_settings) {
	alpha_cache.unref();

	StreamedTexture2DLoadData load_data;
	load_data.image.instantiate();
	Error err = _load_data(p_path, _current_resolution, load_data);
	ERR_FAIL_COND_V(err != OK, err);

	RID new_texture = RS::get_singleton()->texture_2d_create(load_data.image);
	RenderingServer::get_singleton()->texture_set_path(new_texture, p_path);
	path_to_file = p_path;

	if (texture.is_valid()) {
		RS::get_singleton()->texture_replace(texture, new_texture);
	} else {
		texture = new_texture;
	}

	w = load_data.width;
	h = load_data.height;
	format = load_data.format;
#ifdef MODULE_TEXTURE_STREAMING_ENABLED
	if (use_streaming) {
		if (p_load_settings) {
			mipmap_streaming_min = load_data.streaming_min;
			mipmap_streaming_max = load_data.streaming_max;
		}
		const uint32_t streaming_min = mipmap_streaming_min > 0 ? 1 << (mipmap_streaming_min - 1) : 0;
		const uint32_t streaming_max = mipmap_streaming_max > 0 ? 1 << (mipmap_streaming_max - 1) : 0;

		if (streaming_state.is_null()) {
			streaming_state = TextureStreaming::get_singleton()->texture_configure_streaming(
					texture,
					RS::get_singleton()->texture_get_format(texture),
					w,
					h,
					streaming_min,
					streaming_max,
					callable_mp(this, &StreamedTexture2D::texture_reload));

			RS::get_singleton()->texture_2d_attach_streaming_state(texture, streaming_state);
		} else {
			TextureStreaming::get_singleton()->texture_update(streaming_state, w, h, streaming_min, streaming_max);
		}
	}
#endif

	if (p_load_settings) {
#ifdef TOOLS_ENABLED
		if (load_data.request_roughness) {
			RS::get_singleton()->texture_set_detect_roughness_callback(texture, _requested_roughness, this);
		} else {
			RS::get_singleton()->texture_set_detect_roughness_callback(texture, nullptr, nullptr);
		}

		if (load_data.request_normal) {
			RS::get_singleton()->texture_set_detect_normal_callback(texture, _requested_normal, this);
		} else {
			RS::get_singleton()->texture_set_detect_normal_callback(texture, nullptr, nullptr);
		}
#endif

		notify_property_list_changed();
		emit_changed();
	}

	return OK;
}

Error StreamedTexture2D::load(const String &p_path) {
	return _load_internal(p_path, true);
}

void StreamedTexture2D::update_texture() {
	_load_internal(path_to_file, false);

#ifdef MODULE_TEXTURE_STREAMING_ENABLED
	if (streaming_state.is_valid()) {
		const uint32_t streaming_min = mipmap_streaming_min > 0 ? 1 << (mipmap_streaming_min - 1) : 0;
		const uint32_t streaming_max = mipmap_streaming_max > 0 ? 1 << (mipmap_streaming_max - 1) : 0;
		TextureStreaming::get_singleton()->texture_update(streaming_state, w, h, streaming_min, streaming_max);
	}
#endif
}

void StreamedTexture2D::reload_from_file() {
	String path = path_to_file;
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

void StreamedTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mipmap_streaming_min", "min"), &StreamedTexture2D::set_mipmap_streaming_min);
	ClassDB::bind_method(D_METHOD("get_mipmap_streaming_min"), &StreamedTexture2D::get_mipmap_streaming_min);

	ClassDB::bind_method(D_METHOD("set_mipmap_streaming_max", "max"), &StreamedTexture2D::set_mipmap_streaming_max);
	ClassDB::bind_method(D_METHOD("get_mipmap_streaming_max"), &StreamedTexture2D::get_mipmap_streaming_max);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "min_resolution", PROPERTY_HINT_ENUM, "Default,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), "set_mipmap_streaming_min", "get_mipmap_streaming_min");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_resolution", PROPERTY_HINT_ENUM, "Default,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192"), "set_mipmap_streaming_max", "get_mipmap_streaming_max");
}

void StreamedTexture2D::set_mipmap_streaming_min(int p_min) {
	mipmap_streaming_min = p_min;
	update_texture();
}

int StreamedTexture2D::get_mipmap_streaming_min() const {
	return mipmap_streaming_min;
}

void StreamedTexture2D::set_mipmap_streaming_max(int p_max) {
	mipmap_streaming_max = p_max;
	update_texture();
}

int StreamedTexture2D::get_mipmap_streaming_max() const {
	return mipmap_streaming_max;
}

Image::Format StreamedTexture2D::get_format() const {
	return format;
}

String StreamedTexture2D::get_load_path() const {
	return path_to_file;
}

int StreamedTexture2D::get_width() const {
	return w;
}

int StreamedTexture2D::get_height() const {
	return h;
}

RID StreamedTexture2D::get_rid() const {
	if (!texture.is_valid()) {
		texture = RS::get_singleton()->texture_2d_placeholder_create();
	}
	return texture;
}

void StreamedTexture2D::set_path(const String &p_path, bool p_take_over) {
	if (texture.is_valid()) {
		RenderingServer::get_singleton()->texture_set_path(texture, p_path);
	}

	Resource::set_path(p_path, p_take_over);
}

void StreamedTexture2D::draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, Rect2(p_pos, Size2(w, h)), texture, false, p_modulate, p_transpose);
}

void StreamedTexture2D::draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile, const Color &p_modulate, bool p_transpose) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item, p_rect, texture, p_tile, p_modulate, p_transpose);
}

void StreamedTexture2D::draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate, bool p_transpose, bool p_clip_uv) const {
	if ((w | h) == 0) {
		return;
	}
	RenderingServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item, p_rect, texture, p_src_rect, p_modulate, p_transpose, p_clip_uv);
}

bool StreamedTexture2D::has_alpha() const {
	return false;
}

Ref<Image> StreamedTexture2D::get_image() const {
	return _load_image(path_to_file, 0);
}

bool StreamedTexture2D::is_pixel_opaque(int p_x, int p_y) const {
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

		int x = p_x * aw / w;
		int y = p_y * ah / h;

		x = CLAMP(x, 0, aw - 1);
		y = CLAMP(y, 0, ah - 1);

		return alpha_cache->get_bit(x, y);
	}

	return true;
}

StreamedTexture2D::StreamedTexture2D() {
	mipmap_streaming_min = 0;
	mipmap_streaming_max = 0;

	const bool streaming_enabled = GLOBAL_GET("rendering/textures/streaming/enabled");
	const String rendering_method = OS::get_singleton()->get_current_rendering_method();
	use_streaming = streaming_enabled && rendering_method != "gl_compatibility";
	if (!use_streaming) {
		_current_resolution = 0; // force full resolution
	} else {
		_current_resolution = 1u << uint32_t(GLOBAL_GET("rendering/textures/streaming/initial_size"));
	}
}

StreamedTexture2D::~StreamedTexture2D() {
#ifdef MODULE_TEXTURE_STREAMING_ENABLED
	if (streaming_state.is_valid()) {
		TextureStreaming::get_singleton()->texture_remove(streaming_state);
		streaming_state = RID();
	}
#endif

	if (texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RS::get_singleton()->free_rid(texture);
	}
}

Ref<Resource> ResourceFormatLoaderStreamedTexture2D::load(const String &p_path, const String &p_original_path, Error *r_error, bool p_use_sub_threads, float *r_progress, CacheMode p_cache_mode) {
	Ref<StreamedTexture2D> st;
	st.instantiate();

	Error err = st->load(p_path);
	if (r_error) {
		*r_error = err;
	}
	if (err != OK) {
		return Ref<Resource>();
	}

	return st;
}

void ResourceFormatLoaderStreamedTexture2D::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("stex");
}

bool ResourceFormatLoaderStreamedTexture2D::handles_type(const String &p_type) const {
	return p_type == "StreamedTexture2D";
}

String ResourceFormatLoaderStreamedTexture2D::get_resource_type(const String &p_path) const {
	if (p_path.get_extension().to_lower() == "stex") {
		return "StreamedTexture2D";
	}
	return "";
}
