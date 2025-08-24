/**************************************************************************/
/*  image_texture.h                                                       */
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

#include "scene/resources/texture.h"

class BitMap;

class ImageTexture : public Texture2D {
	GDCLASS(ImageTexture, Texture2D);
	RES_BASE_EXTENSION("tex");

	// SVG source only.
	String source;
	float base_scale = 1.0;
	float saturation = 1.0;
	Dictionary color_map;
	mutable HashMap<double, RID> texture_cache;
	mutable HashMap<Color, Color> cmap;

	struct ScalingLevel {
		HashSet<ImageTexture *> textures;
		int32_t refcount = 1;
	};
	static Mutex mutex;
	static HashMap<double, ScalingLevel> scaling_levels;

	// Common.
	mutable RID texture;
	mutable Image::Format format = Image::FORMAT_L8;
	mutable bool mipmaps = false;

	mutable Size2 size;
	Size2 size_override;
	mutable Ref<BitMap> alpha_cache;
	mutable bool image_stored = false;

	void _remove_scale(double p_scale);
	RID _ensure_scale(double p_scale) const;
	RID _load_at_scale(double p_scale, bool p_set_size) const;
	void _update_texture();
	void _clear();

protected:
	virtual void reload_from_file() override;

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _validate_property(PropertyInfo &p_property) const;

	static void _bind_methods();

public:
	static Ref<ImageTexture> create_from_image(const Ref<Image> &p_image);
	static Ref<ImageTexture> create_from_string(const String &p_source, float p_scale = 1.0, float p_saturation = 1.0, const Dictionary &p_color_map = Dictionary());

	Image::Format get_format() const;

	void set_source(const String &p_source);
	String get_source() const;

	void set_base_scale(float p_scale);
	float get_base_scale() const;

	void set_color_map(const Dictionary &p_color_map);
	Dictionary get_color_map() const;

	void set_saturation(float p_saturation);
	float get_saturation() const;

	void update(const Ref<Image> &p_image);
	void set_image(const Ref<Image> &p_image);
	Ref<Image> get_image() const override;

	int get_width() const override;
	int get_height() const override;

	virtual RID get_rid() const override;
	virtual RID get_scaled_rid() const override;

	bool has_alpha() const override;
	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const override;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const override;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, bool p_clip_uv = true) const override;

	bool is_pixel_opaque(int p_x, int p_y) const override;

	void set_size_override(const Size2i &p_size);

	virtual void set_path(const String &p_path, bool p_take_over = false) override;

	static void reference_scaling_level(double p_scale);
	static void unreference_scaling_level(double p_scale);

	~ImageTexture();
};

class ImageTextureLayered : public TextureLayered {
	GDCLASS(ImageTextureLayered, TextureLayered);

	LayeredType layered_type;

	mutable RID texture;
	Image::Format format = Image::FORMAT_L8;

	int width = 0;
	int height = 0;
	int layers = 0;
	bool mipmaps = false;

	Error _create_from_images(const TypedArray<Image> &p_images);

	TypedArray<Image> _get_images() const;
	void _set_images(const TypedArray<Image> &p_images);

protected:
	static void _bind_methods();

public:
	virtual Image::Format get_format() const override;
	virtual int get_width() const override;
	virtual int get_height() const override;
	virtual int get_layers() const override;
	virtual bool has_mipmaps() const override;
	virtual LayeredType get_layered_type() const override;

	Error create_from_images(Vector<Ref<Image>> p_images);
	void update_layer(const Ref<Image> &p_image, int p_layer);
	virtual Ref<Image> get_layer_data(int p_layer) const override;

	virtual RID get_rid() const override;
	virtual void set_path(const String &p_path, bool p_take_over = false) override;

	ImageTextureLayered(LayeredType p_layered_type);
	~ImageTextureLayered();
};

class ImageTexture3D : public Texture3D {
	GDCLASS(ImageTexture3D, Texture3D);

	mutable RID texture;

	Image::Format format = Image::FORMAT_L8;
	int width = 1;
	int height = 1;
	int depth = 1;
	bool mipmaps = false;

	TypedArray<Image> _get_images() const;
	void _set_images(const TypedArray<Image> &p_images);

protected:
	static void _bind_methods();

	Error _create(Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const TypedArray<Image> &p_data);
	void _update(const TypedArray<Image> &p_data);

public:
	virtual Image::Format get_format() const override;
	virtual int get_width() const override;
	virtual int get_height() const override;
	virtual int get_depth() const override;
	virtual bool has_mipmaps() const override;

	Error create(Image::Format p_format, int p_width, int p_height, int p_depth, bool p_mipmaps, const Vector<Ref<Image>> &p_data);
	void update(const Vector<Ref<Image>> &p_data);
	virtual Vector<Ref<Image>> get_data() const override;

	virtual RID get_rid() const override;
	virtual void set_path(const String &p_path, bool p_take_over = false) override;

	ImageTexture3D();
	~ImageTexture3D();
};

class Texture2DArray : public ImageTextureLayered {
	GDCLASS(Texture2DArray, ImageTextureLayered)

protected:
	static void _bind_methods();

public:
	Texture2DArray() :
			ImageTextureLayered(LAYERED_TYPE_2D_ARRAY) {}

	virtual Ref<Resource> create_placeholder() const;
};

class Cubemap : public ImageTextureLayered {
	GDCLASS(Cubemap, ImageTextureLayered);

protected:
	static void _bind_methods();

public:
	Cubemap() :
			ImageTextureLayered(LAYERED_TYPE_CUBEMAP) {}

	virtual Ref<Resource> create_placeholder() const;
};

class CubemapArray : public ImageTextureLayered {
	GDCLASS(CubemapArray, ImageTextureLayered);

protected:
	static void _bind_methods();

public:
	CubemapArray() :
			ImageTextureLayered(LAYERED_TYPE_CUBEMAP_ARRAY) {}

	virtual Ref<Resource> create_placeholder() const;
};
