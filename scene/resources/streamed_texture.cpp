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
#include <cstdio>

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

void StreamedTexture2D::texture_reload(uint8_t p_mip_level) {
	ERR_FAIL_COND(texture.is_null());
	ERR_FAIL_COND(path_to_file.is_empty());

	// Clamp cached current mip to what is actually available in file.
	uint32_t current_mip = MIN((uint32_t)_current_mip.load(), num_mipmaps);
	if (current_mip != _current_mip.load()) {
		_current_mip.store((uint8_t)current_mip);
	}

	uint32_t new_mip = MIN((uint32_t)p_mip_level, num_mipmaps);

	// If the requested mip level is the same as the current, do nothing.
	if (current_mip == new_mip) {
		return;
	}

	_current_mip.store((uint8_t)new_mip); // Atomic store

	// Calculate the new dimensions.
	uint32_t new_width = MAX(1u, (uint32_t)w >> new_mip);
	uint32_t new_height = MAX(1u, (uint32_t)h >> new_mip);
	uint32_t new_mipmap_count = Image::get_image_required_mipmaps(new_width, new_height, format) + 1;

	// Validate that calculated mipmap count doesn't exceed what's in the file.
	uint32_t available_mips = num_mipmaps - new_mip + 1;
	if (new_mipmap_count > available_mips) {
		ERR_PRINT(vformat("Mipmap count mismatch: calculated %d but only %d available. Falling back to full reload.", new_mipmap_count, available_mips));
		_load_internal(path_to_file, false);
		return;
	}

	Vector<uint8_t> new_mip_data;
	uint32_t copy_mips_count;

	if (new_mip >= current_mip) {
		// Decreasing resolution (or same) - just copy existing mips, no new data needed.
		copy_mips_count = new_mipmap_count;
	} else {
		// Increasing resolution - need to load additional mip data.
		ERR_FAIL_COND_MSG(current_mip <= new_mip, "Logic error: current_mip should be greater than new_mip in this branch.");
		uint32_t mips_to_add = current_mip - new_mip;
		copy_mips_count = new_mipmap_count - mips_to_add;
		DEV_ASSERT(current_mip > new_mip);
		Error err = _load_mip_range(path_to_file, new_mip, mips_to_add, new_mip_data);
		if (err != OK) {
			ERR_PRINT(vformat("Failed to load mip range [%d, %d) %s. Falling back to full reload.", new_mip, current_mip, path_to_file));
			_load_internal(path_to_file, false);
			return;
		}

		// Validate that we received the expected amount of data.
		if (new_mip_data.is_empty()) {
			ERR_PRINT(vformat("Loaded mip data is empty for %s. Falling back to full reload.", path_to_file));
			_load_internal(path_to_file, false);
			return;
		}
	}

	// Create new texture from existing texture.
	RID new_texture = RS::get_singleton()->texture_2d_create_from_texture(
			texture,
			new_width,
			new_height,
			copy_mips_count,
			new_mip_data);

	if (new_texture.is_null()) {
		ERR_PRINT("Failed to create texture from existing texture. Falling back to full reload.");
		_load_internal(path_to_file, false);
		return;
	}

	RenderingServer::get_singleton()->texture_set_path(new_texture, path_to_file);
	RS::get_singleton()->texture_replace(texture, new_texture);

#ifdef MODULE_TEXTURE_STREAMING_ENABLED
	// Re-attach streaming state after texture replacement if streaming is active
	if (use_streaming && streaming_state.is_valid()) {
		RS::get_singleton()->texture_2d_attach_streaming_state(texture, streaming_state);
	}
#endif
}

Error StreamedTexture2D::_parse_header(const String &p_path, StreamedTextureFileHeader &r_header, Ref<FileAccess> &r_file) {
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
	r_header.width = f->get_32();
	r_header.height = f->get_32();

	// Read data size.
	r_header.data_size = f->get_32();

	// Read image format.
	uint32_t format_value = f->get_32();
	ERR_FAIL_COND_V_MSG(format_value >= Image::FORMAT_MAX, ERR_FILE_CORRUPT, "Streamed texture has invalid image format.");
	r_header.format = (Image::Format)format_value;

	// Read mipmap count.
	r_header.num_mipmaps = f->get_32();
	ERR_FAIL_COND_V_MSG(r_header.num_mipmaps > 16, ERR_FILE_CORRUPT, "Streamed texture has too many mipmaps.");

	// Read the offset and size for each mip level.
	r_header.mipmap_offset_size.resize(r_header.num_mipmaps + 1);
	for (uint32_t i = 0; i <= r_header.num_mipmaps; i++) {
		r_header.mipmap_offset_size[i].first = f->get_32();
		r_header.mipmap_offset_size[i].second = f->get_32();
	}

	// Read flags.
	r_header.flags = f->get_32();

	// Read streaming settings.
	r_header.streaming_min = f->get_32();
	r_header.streaming_max = f->get_32();

	// Reserved for future use.
	f->get_32();

	r_file = f;
	return OK;
}

Error StreamedTexture2D::_load_mips_from_file(Ref<FileAccess> p_file, const StreamedTextureFileHeader &p_header, uint32_t p_start_mip, uint32_t p_mip_count, Vector<uint8_t> &r_mip_data) {
	ERR_FAIL_COND_V_MSG(
			p_start_mip + p_mip_count > p_header.num_mipmaps + 1,
			ERR_INVALID_PARAMETER,
			vformat("Requested mip range [%d, %d) exceeds available mip range [0, %d).", int(p_start_mip), int(p_start_mip + p_mip_count), int(p_header.num_mipmaps + 1)));

	uint64_t data_start = p_file->get_position();

	// Calculate total size needed for all requested mips.
	uint32_t total_size = 0;
	for (uint32_t i = 0; i < p_mip_count; i++) {
		total_size += p_header.mipmap_offset_size[p_start_mip + i].second;
	}

	// Load all mip data into a single buffer.
	r_mip_data.resize(total_size);
	uint32_t write_offset = 0;
	for (uint32_t i = 0; i < p_mip_count; i++) {
		uint32_t mip_index = p_start_mip + i;
		uint32_t mip_offset = p_header.mipmap_offset_size[mip_index].first;
		uint32_t mip_size = p_header.mipmap_offset_size[mip_index].second;

		p_file->seek(data_start + mip_offset);
		uint64_t bytes_read = p_file->get_buffer(r_mip_data.ptrw() + write_offset, mip_size);
		ERR_FAIL_COND_V_MSG(bytes_read != mip_size, ERR_FILE_CORRUPT, "Streamed texture file is truncated.");
		write_offset += mip_size;
	}

	return OK;
}

Error StreamedTexture2D::_load_data(const String &p_path, const uint8_t p_mip_level, StreamedTexture2DLoadData &p_load_data) {
	ERR_FAIL_COND_V(p_load_data.image.is_null(), ERR_INVALID_PARAMETER);

	StreamedTextureFileHeader header;
	Ref<FileAccess> f;
	Error err = _parse_header(p_path, header, f);
	ERR_FAIL_COND_V(err != OK, err);

	// Validate dimensions.
	ERR_FAIL_COND_V_MSG(header.width == 0 || header.height == 0, ERR_FILE_CORRUPT, "Streamed texture has invalid dimensions.");
	ERR_FAIL_COND_V_MSG(header.width > 16384 || header.height > 16384, ERR_FILE_CORRUPT, "Streamed texture dimensions are too large.");

	// Populate load data from header.
	p_load_data.width = header.width;
	p_load_data.height = header.height;
	p_load_data.format = header.format;
	p_load_data.mipmap_count = header.num_mipmaps;
	p_load_data.streaming_min = header.streaming_min;
	p_load_data.streaming_max = header.streaming_max;

#ifdef TOOLS_ENABLED
	p_load_data.request_normal = request_normal_callback && (header.flags & FORMAT_BIT_DETECT_NORMAL);
	p_load_data.request_roughness = request_roughness_callback && (header.flags & FORMAT_BIT_DETECT_ROUGHNESS);
#else
	p_load_data.request_normal = false;
	p_load_data.request_roughness = false;
#endif

	// Validate that the file contains enough data.
	uint64_t header_end = f->get_position();
	uint64_t file_length = f->get_length();
	ERR_FAIL_COND_V_MSG(file_length < header_end + header.data_size, ERR_FILE_CORRUPT, "Streamed texture file is truncated (file too small for declared data size).");

	// Clamp the mip level to valid range.
	uint32_t load_mip = MIN((uint32_t)p_mip_level, header.num_mipmaps);
	uint32_t mip_count = header.num_mipmaps - load_mip;

	// Load mip data from file.
	Vector<uint8_t> data;
	err = _load_mips_from_file(f, header, load_mip, mip_count + 1, data);
	ERR_FAIL_COND_V(err != OK, err);

	// Calculate the dimensions at this mip level.
	uint32_t mip_width = header.width >> load_mip;
	uint32_t mip_height = header.height >> load_mip;

	p_load_data.image->set_data(mip_width, mip_height, mip_count > 0, header.format, data);

	return OK;
}

Error StreamedTexture2D::_load_mip_range(const String &p_path, uint32_t p_start_mip, uint32_t p_mip_count, Vector<uint8_t> &r_mip_data) {
	StreamedTextureFileHeader header;
	Ref<FileAccess> f;
	Error err = _parse_header(p_path, header, f);
	ERR_FAIL_COND_V(err != OK, err);

	return _load_mips_from_file(f, header, p_start_mip, p_mip_count, r_mip_data);
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
	ERR_FAIL_COND_V(p_path.is_empty(), ERR_FILE_BAD_PATH);
	alpha_cache.unref();

	StreamedTexture2DLoadData load_data;
	load_data.image.instantiate();
	Error err = _load_data(p_path, _current_mip.load(), load_data);
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
	// Use the mip count declared in file (not a computed full chain), as some files may not include all mips.
	num_mipmaps = load_data.mipmap_count;
	_current_mip.store(uint8_t(MIN((uint32_t)_current_mip.load(), num_mipmaps)));
#ifdef MODULE_TEXTURE_STREAMING_ENABLED
	if (use_streaming) {
		if (p_load_settings) {
			// streaming_min in file = best quality limit (min mip), streaming_max = worst quality limit (max mip)
			min_lod_override = load_data.streaming_min;
			max_lod_override = load_data.streaming_max;
		}

		if (streaming_state.is_null()) {
			streaming_state = TextureStreaming::get_singleton()->texture_configure_streaming(
					texture,
					format,
					w,
					h,
					min_lod_override,
					max_lod_override,
					callable_mp(this, &StreamedTexture2D::texture_reload));
		} else {
			TextureStreaming::get_singleton()->texture_update(streaming_state, w, h, min_lod_override, max_lod_override);
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

// Called when LOD settings change or after loading.
// Part of the reason this is done is to work around a limitation in the current texture
// system in godot.  There are no notifications when a texture is changed.  So the only way
// for the editor to update the displayed texture after changing LOD settings is to "change"
// the texture so just do that.
void StreamedTexture2D::update_texture() {
	_load_internal(path_to_file, false);
#ifdef MODULE_TEXTURE_STREAMING_ENABLED
	if (streaming_state.is_valid()) {
		TextureStreaming::get_singleton()->texture_update(streaming_state, w, h, min_lod_override, max_lod_override);
	}
#endif
}

void StreamedTexture2D::reload_from_file() {
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

void StreamedTexture2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("load", "path"), &StreamedTexture2D::load);
	ClassDB::bind_method(D_METHOD("get_load_path"), &StreamedTexture2D::get_load_path);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "load_path", PROPERTY_HINT_FILE, "*.stex"), "load", "get_load_path");

	ClassDB::bind_method(D_METHOD("set_max_lod_override", "max_lod"), &StreamedTexture2D::set_max_lod_override);
	ClassDB::bind_method(D_METHOD("get_max_lod_override"), &StreamedTexture2D::get_max_lod_override);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_lod_override", PROPERTY_HINT_ENUM, "Setting Default,0,1,2,3,4,5,6,7,8,9,10,11,12,13"), "set_max_lod_override", "get_max_lod_override");

	ClassDB::bind_method(D_METHOD("set_min_lod_override", "min_lod"), &StreamedTexture2D::set_min_lod_override);
	ClassDB::bind_method(D_METHOD("get_min_lod_override"), &StreamedTexture2D::get_min_lod_override);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "min_lod_override", PROPERTY_HINT_ENUM, "Setting Default,0,1,2,3,4,5,6,7,8,9,10,11,12,13"), "set_min_lod_override", "get_min_lod_override");
}

void StreamedTexture2D::set_max_lod_override(int p_max_lod) {
	max_lod_override = p_max_lod;
	update_texture();
}

int StreamedTexture2D::get_max_lod_override() const {
	return max_lod_override;
}

void StreamedTexture2D::set_min_lod_override(int p_min_lod) {
	min_lod_override = p_min_lod;
	update_texture();
}

int StreamedTexture2D::get_min_lod_override() const {
	return min_lod_override;
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
	max_lod_override = 0;
	min_lod_override = 0;

	const bool streaming_enabled = GLOBAL_GET("rendering/textures/streaming/enabled");
	const String rendering_method = OS::get_singleton()->get_current_rendering_method();
	use_streaming = streaming_enabled && rendering_method != "gl_compatibility";
	if (!use_streaming) {
		_current_mip.store(0); // force full resolution (mip 0)
	} else {
		// Initial mip level from settings (higher mip = lower quality initially)
		_current_mip.store(uint8_t(GLOBAL_GET("rendering/textures/streaming/max_lod")));
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
