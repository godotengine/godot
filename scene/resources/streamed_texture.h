/**************************************************************************/
/*  streamed_texture.h                                                    */
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

#pragma once

#include "core/io/resource_loader.h"
#include "scene/resources/texture.h"
#include "servers/rendering/rendering_server.h"

class BitMap;
class ResourceImporterStreamedTexture;

class StreamedTexture2D : public Texture2D {
	GDCLASS(StreamedTexture2D, Texture2D);

	struct StreamedTexture2DLoadData {
		uint32_t width = 0;
		uint32_t height = 0;
		Image::Format format = Image::FORMAT_DXT1;
		Ref<Image> image;
		bool request_normal = false;
		bool request_roughness = false;
		uint32_t streaming_min = 0;
		uint32_t streaming_max = 0;
	};

	enum FormatBits {
		FORMAT_BIT_DETECT_NORMAL = 1 << 0,
		FORMAT_BIT_DETECT_ROUGHNESS = 1 << 1,
	};

	static Error _load_data(const String &p_path, const uint32_t p_max_resolution, struct StreamedTexture2DLoadData &p_load_data);
	static Error _save_data(const String &p_path, const Ref<Image> &p_image, uint32_t p_flags, uint32_t p_streaming_min, uint32_t p_streaming_max);

	Ref<Image> _load_image(const String &p_path, const uint32_t p_max_resolution) const {
		StreamedTexture2DLoadData load_data;
		load_data.image = Ref<Image>();
		load_data.image.instantiate();

		Error err = _load_data(p_path, p_max_resolution, load_data);
		if (err != OK) {
			return Ref<Image>();
		}

		return load_data.image;
	}

	void _save_image(const String &p_path, const Ref<Image> &p_image) const {
		_save_data(p_path, p_image, 0, 0, 0);
	}

	void update_texture();
	void texture_reload(uint32_t p_resolution);
	uint32_t _current_resolution = 0;
	bool use_streaming = false;

	RID streaming_state;

	int mipmap_streaming_min = 0;
	int mipmap_streaming_max = 0;

	Error _load_internal(const String &p_path, bool p_load_settings = true);

	String path_to_file;
	mutable RID texture;
	Image::Format format = Image::FORMAT_L8;
	int w = 0;
	int h = 0;
	mutable Ref<BitMap> alpha_cache;

	typedef void (*TextureFormatRequestCallback)(const Ref<StreamedTexture2D> &);
	typedef void (*TextureFormatRoughnessRequestCallback)(const Ref<StreamedTexture2D> &, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_roughness_channel);

	static void _requested_roughness(void *p_ud, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_roughness_channel);
	static void _requested_normal(void *p_ud);

protected:
	static void _bind_methods();
	virtual void reload_from_file() override;

public:
	static TextureFormatRoughnessRequestCallback request_roughness_callback;
	static TextureFormatRequestCallback request_normal_callback;

	void set_mipmap_streaming_min(int p_min);
	int get_mipmap_streaming_min() const;

	void set_mipmap_streaming_max(int p_max);
	int get_mipmap_streaming_max() const;

	Error load(const String &p_path);

	Image::Format get_format() const;
	String get_load_path() const;

	int get_width() const override;
	int get_height() const override;
	virtual RID get_rid() const override;

	virtual void set_path(const String &p_path, bool p_take_over) override;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const override;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const override;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, bool p_clip_uv = true) const override;

	virtual bool has_alpha() const override;
	bool is_pixel_opaque(int p_x, int p_y) const override;

	virtual Ref<Image> get_image() const override;

	StreamedTexture2D();
	~StreamedTexture2D();

	friend class ResourceImporterStreamedTexture; // For saving to stex format
};

class ResourceFormatLoaderStreamedTexture2D : public ResourceFormatLoader {
	GDSOFTCLASS(ResourceFormatLoaderStreamedTexture2D, ResourceFormatLoader);

public:
	virtual Ref<Resource> load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr, CacheMode p_cache_mode = CACHE_MODE_REUSE) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual bool handles_type(const String &p_type) const override;
	virtual String get_resource_type(const String &p_path) const override;
};
