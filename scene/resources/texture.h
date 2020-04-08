/*************************************************************************/
/*  texture.h                                                            */
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

#ifndef TEXTURE_H
#define TEXTURE_H

#include "core/io/resource_loader.h"
#include "core/math/rect2.h"
#include "core/os/file_access.h"
#include "core/os/mutex.h"
#include "core/os/rw_lock.h"
#include "core/os/thread_safe.h"
#include "core/resource.h"
#include "scene/resources/curve.h"
#include "scene/resources/gradient.h"
#include "servers/camera_server.h"
#include "servers/rendering_server.h"

class Texture : public Resource {
	GDCLASS(Texture, Resource);

public:
	Texture() {}
};

class Texture2D : public Texture {

	GDCLASS(Texture2D, Texture);
	OBJ_SAVE_TYPE(Texture2D); // Saves derived classes with common type so they can be interchanged.

protected:
	static void _bind_methods();

public:
	virtual int get_width() const = 0;
	virtual int get_height() const = 0;
	virtual Size2 get_size() const;
	virtual RID get_rid() const = 0;

	virtual bool is_pixel_opaque(int p_x, int p_y) const;

	virtual bool has_alpha() const = 0;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, bool p_clip_uv = true) const;
	virtual bool get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const;

	virtual Ref<Image> get_data() const { return Ref<Image>(); }

	Texture2D();
};

class BitMap;

class ImageTexture : public Texture2D {

	GDCLASS(ImageTexture, Texture2D);
	RES_BASE_EXTENSION("tex");

	mutable RID texture;
	Image::Format format;
	bool mipmaps;
	int w, h;
	Size2 size_override;
	mutable Ref<BitMap> alpha_cache;
	bool image_stored;

protected:
	virtual void reload_from_file();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _reload_hook(const RID &p_hook);
	virtual void _resource_path_changed();
	static void _bind_methods();

public:
	void create_from_image(const Ref<Image> &p_image);

	Image::Format get_format() const;

	void update(const Ref<Image> &p_image, bool p_immediate = false);
	Ref<Image> get_data() const;

	int get_width() const;
	int get_height() const;

	virtual RID get_rid() const;

	bool has_alpha() const;
	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, bool p_clip_uv = true) const;

	bool is_pixel_opaque(int p_x, int p_y) const;

	void set_size_override(const Size2 &p_size);

	virtual void set_path(const String &p_path, bool p_take_over = false);

	ImageTexture();
	~ImageTexture();
};

class StreamTexture : public Texture2D {

	GDCLASS(StreamTexture, Texture2D);

public:
	enum DataFormat {
		DATA_FORMAT_IMAGE,
		DATA_FORMAT_LOSSLESS,
		DATA_FORMAT_LOSSY,
		DATA_FORMAT_BASIS_UNIVERSAL,
	};

	enum {
		FORMAT_VERSION = 1
	};

	enum FormatBits {
		FORMAT_MASK_IMAGE_FORMAT = (1 << 20) - 1,
		FORMAT_BIT_LOSSLESS = 1 << 20,
		FORMAT_BIT_LOSSY = 1 << 21,
		FORMAT_BIT_STREAM = 1 << 22,
		FORMAT_BIT_HAS_MIPMAPS = 1 << 23,
		FORMAT_BIT_DETECT_3D = 1 << 24,
		//FORMAT_BIT_DETECT_SRGB = 1 << 25,
		FORMAT_BIT_DETECT_NORMAL = 1 << 26,
		FORMAT_BIT_DETECT_ROUGNESS = 1 << 27,
	};

private:
	Error _load_data(const String &p_path, int &tw, int &th, int &tw_custom, int &th_custom, Ref<Image> &image, bool &r_request_3d, bool &r_request_normal, bool &r_request_roughness, int &mipmap_limit, int p_size_limit = 0);
	String path_to_file;
	mutable RID texture;
	Image::Format format;
	int w, h;
	mutable Ref<BitMap> alpha_cache;

	virtual void reload_from_file();

	static void _requested_3d(void *p_ud);
	static void _requested_roughness(void *p_ud, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_roughness_channel);
	static void _requested_normal(void *p_ud);

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;

public:
	static Ref<Image> load_image_from_file(FileAccess *p_file, int p_size_limit);

	typedef void (*TextureFormatRequestCallback)(const Ref<StreamTexture> &);
	typedef void (*TextureFormatRoughnessRequestCallback)(const Ref<StreamTexture> &, const String &p_normal_path, RS::TextureDetectRoughnessChannel p_roughness_channel);

	static TextureFormatRequestCallback request_3d_callback;
	static TextureFormatRoughnessRequestCallback request_roughness_callback;
	static TextureFormatRequestCallback request_normal_callback;

	Image::Format get_format() const;
	Error load(const String &p_path);
	String get_load_path() const;

	int get_width() const;
	int get_height() const;
	virtual RID get_rid() const;

	virtual void set_path(const String &p_path, bool p_take_over);

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, bool p_clip_uv = true) const;

	virtual bool has_alpha() const;
	bool is_pixel_opaque(int p_x, int p_y) const;

	virtual Ref<Image> get_data() const;

	StreamTexture();
	~StreamTexture();
};

class ResourceFormatLoaderStreamTexture : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

class AtlasTexture : public Texture2D {

	GDCLASS(AtlasTexture, Texture2D);
	RES_BASE_EXTENSION("atlastex");

protected:
	Ref<Texture2D> atlas;
	Rect2 region;
	Rect2 margin;
	bool filter_clip;

	static void _bind_methods();

public:
	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	void set_atlas(const Ref<Texture2D> &p_atlas);
	Ref<Texture2D> get_atlas() const;

	void set_region(const Rect2 &p_region);
	Rect2 get_region() const;

	void set_margin(const Rect2 &p_margin);
	Rect2 get_margin() const;

	void set_filter_clip(const bool p_enable);
	bool has_filter_clip() const;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, bool p_clip_uv = true) const;
	virtual bool get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const;

	bool is_pixel_opaque(int p_x, int p_y) const;

	AtlasTexture();
};

class Mesh;

class MeshTexture : public Texture2D {

	GDCLASS(MeshTexture, Texture2D);
	RES_BASE_EXTENSION("meshtex");

	Ref<Texture2D> base_texture;
	Ref<Mesh> mesh;
	Size2i size;

protected:
	static void _bind_methods();

public:
	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	void set_mesh(const Ref<Mesh> &p_mesh);
	Ref<Mesh> get_mesh() const;

	void set_image_size(const Size2 &p_size);
	Size2 get_image_size() const;

	void set_base_texture(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_base_texture() const;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, bool p_clip_uv = true) const;
	virtual bool get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const;

	bool is_pixel_opaque(int p_x, int p_y) const;

	MeshTexture();
};

class LargeTexture : public Texture2D {

	GDCLASS(LargeTexture, Texture2D);
	RES_BASE_EXTENSION("largetex");

protected:
	struct Piece {

		Point2 offset;
		Ref<Texture2D> texture;
	};

	Vector<Piece> pieces;
	Size2i size;

	Array _get_data() const;
	void _set_data(const Array &p_array);
	static void _bind_methods();

public:
	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	int add_piece(const Point2 &p_offset, const Ref<Texture2D> &p_texture);
	void set_piece_offset(int p_idx, const Point2 &p_offset);
	void set_piece_texture(int p_idx, const Ref<Texture2D> &p_texture);

	void set_size(const Size2 &p_size);
	void clear();

	int get_piece_count() const;
	Vector2 get_piece_offset(int p_idx) const;
	Ref<Texture2D> get_piece_texture(int p_idx) const;
	Ref<Image> to_image() const;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, const Ref<Texture2D> &p_normal_map = Ref<Texture2D>(), const Ref<Texture2D> &p_specular_map = Ref<Texture2D>(), const Color &p_specular_color_shininess = Color(1, 1, 1, 1), RS::CanvasItemTextureFilter p_texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT, RS::CanvasItemTextureRepeat p_texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT, bool p_clip_uv = true) const;

	bool is_pixel_opaque(int p_x, int p_y) const;

	LargeTexture();
};

class TextureLayered : public Texture {

	GDCLASS(TextureLayered, Texture);

	RS::TextureLayeredType layered_type;

	mutable RID texture;
	Image::Format format;

	int width;
	int height;
	int layers;
	bool mipmaps;

	Error _create_from_images(const Array &p_images);

	Array _get_images() const;

protected:
	static void _bind_methods();

public:
	Image::Format get_format() const;
	uint32_t get_width() const;
	uint32_t get_height() const;
	uint32_t get_layers() const;
	bool has_mipmaps() const;

	Error create_from_images(Vector<Ref<Image>> p_images);
	void update_layer(const Ref<Image> &p_image, int p_layer);
	Ref<Image> get_layer_data(int p_layer) const;

	virtual RID get_rid() const;
	virtual void set_path(const String &p_path, bool p_take_over = false);

	TextureLayered(RS::TextureLayeredType p_layered_type);
	~TextureLayered();
};

class Texture2DArray : public TextureLayered {

	GDCLASS(Texture2DArray, TextureLayered)
public:
	Texture2DArray() :
			TextureLayered(RS::TEXTURE_LAYERED_2D_ARRAY) {}
};

class Cubemap : public TextureLayered {

	GDCLASS(Cubemap, TextureLayered);

public:
	Cubemap() :
			TextureLayered(RS::TEXTURE_LAYERED_CUBEMAP) {}
};

class CubemapArray : public TextureLayered {

	GDCLASS(CubemapArray, TextureLayered);

public:
	CubemapArray() :
			TextureLayered(RS::TEXTURE_LAYERED_CUBEMAP_ARRAY) {}
};

class ResourceFormatLoaderTextureLayered : public ResourceFormatLoader {
public:
	enum Compression {
		COMPRESSION_LOSSLESS,
		COMPRESSION_VRAM,
		COMPRESSION_UNCOMPRESSED
	};

	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = nullptr, bool p_use_sub_threads = false, float *r_progress = nullptr);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

class CurveTexture : public Texture2D {

	GDCLASS(CurveTexture, Texture2D);
	RES_BASE_EXTENSION("curvetex")

private:
	mutable RID _texture;
	Ref<Curve> _curve;
	int _width;

	void _update();

protected:
	static void _bind_methods();

public:
	void set_width(int p_width);
	int get_width() const;

	void ensure_default_setup(float p_min = 0, float p_max = 1);

	void set_curve(Ref<Curve> p_curve);
	Ref<Curve> get_curve() const;

	virtual RID get_rid() const;

	virtual int get_height() const { return 1; }
	virtual bool has_alpha() const { return false; }

	CurveTexture();
	~CurveTexture();
};
/*
	enum CubeMapSide {

		CUBEMAP_LEFT,
		CUBEMAP_RIGHT,
		CUBEMAP_BOTTOM,
		CUBEMAP_TOP,
		CUBEMAP_FRONT,
		CUBEMAP_BACK,
	};

*/
//VARIANT_ENUM_CAST( Texture::CubeMapSide );

class GradientTexture : public Texture2D {
	GDCLASS(GradientTexture, Texture2D);

public:
	struct Point {

		float offset;
		Color color;
		bool operator<(const Point &p_ponit) const {
			return offset < p_ponit.offset;
		}
	};

private:
	Ref<Gradient> gradient;
	bool update_pending;
	RID texture;
	int width;

	void _queue_update();
	void _update();

protected:
	static void _bind_methods();

public:
	void set_gradient(Ref<Gradient> p_gradient);
	Ref<Gradient> get_gradient() const;

	void set_width(int p_width);
	int get_width() const;

	virtual RID get_rid() const { return texture; }
	virtual int get_height() const { return 1; }
	virtual bool has_alpha() const { return true; }

	virtual Ref<Image> get_data() const;

	GradientTexture();
	virtual ~GradientTexture();
};

class ProxyTexture : public Texture2D {
	GDCLASS(ProxyTexture, Texture2D);

private:
	mutable RID proxy_ph;
	mutable RID proxy;
	Ref<Texture2D> base;

protected:
	static void _bind_methods();

public:
	void set_base(const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_base() const;

	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	ProxyTexture();
	~ProxyTexture();
};

class AnimatedTexture : public Texture2D {
	GDCLASS(AnimatedTexture, Texture2D);

	//use readers writers lock for this, since its far more times read than written to
	RWLock *rw_lock;

private:
	enum {
		MAX_FRAMES = 256
	};

	RID proxy_ph;
	RID proxy;

	struct Frame {

		Ref<Texture2D> texture;
		float delay_sec;

		Frame() {
			delay_sec = 0;
		}
	};

	Frame frames[MAX_FRAMES];
	int frame_count;
	int current_frame;

	float fps;

	float time;

	uint64_t prev_ticks;

	void _update_proxy();

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;

public:
	void set_frames(int p_frames);
	int get_frames() const;

	void set_frame_texture(int p_frame, const Ref<Texture2D> &p_texture);
	Ref<Texture2D> get_frame_texture(int p_frame) const;

	void set_frame_delay(int p_frame, float p_delay_sec);
	float get_frame_delay(int p_frame) const;

	void set_fps(float p_fps);
	float get_fps() const;

	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	virtual Ref<Image> get_data() const;

	bool is_pixel_opaque(int p_x, int p_y) const;

	AnimatedTexture();
	~AnimatedTexture();
};

class CameraTexture : public Texture2D {
	GDCLASS(CameraTexture, Texture2D);

private:
	int camera_feed_id;
	CameraServer::FeedImage which_feed;

protected:
	static void _bind_methods();

public:
	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;
	virtual bool has_alpha() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	virtual Ref<Image> get_data() const;

	void set_camera_feed_id(int p_new_id);
	int get_camera_feed_id() const;

	void set_which_feed(CameraServer::FeedImage p_which);
	CameraServer::FeedImage get_which_feed() const;

	void set_camera_active(bool p_active);
	bool get_camera_active() const;

	CameraTexture();
	~CameraTexture();
};

#endif
