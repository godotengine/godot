/*************************************************************************/
/*  texture.h                                                            */
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
#ifndef TEXTURE_H
#define TEXTURE_H

#include "io/resource_loader.h"
#include "math_2d.h"
#include "resource.h"
#include "servers/visual_server.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class Texture : public Resource {

	GDCLASS(Texture, Resource);
	OBJ_SAVE_TYPE(Texture); //children are all saved as Texture, so they can be exchanged
protected:
	static void _bind_methods();

public:
	enum Flags {
		FLAG_MIPMAPS = VisualServer::TEXTURE_FLAG_MIPMAPS,
		FLAG_REPEAT = VisualServer::TEXTURE_FLAG_REPEAT,
		FLAG_FILTER = VisualServer::TEXTURE_FLAG_FILTER,
		FLAG_ANISOTROPIC_FILTER = VisualServer::TEXTURE_FLAG_ANISOTROPIC_FILTER,
		FLAG_CONVERT_TO_LINEAR = VisualServer::TEXTURE_FLAG_CONVERT_TO_LINEAR,
		FLAG_VIDEO_SURFACE = VisualServer::TEXTURE_FLAG_USED_FOR_STREAMING,
		FLAGS_DEFAULT = FLAG_MIPMAPS | FLAG_REPEAT | FLAG_FILTER,
		FLAG_MIRRORED_REPEAT = VisualServer::TEXTURE_FLAG_MIRRORED_REPEAT
	};

	virtual int get_width() const = 0;
	virtual int get_height() const = 0;
	virtual Size2 get_size() const;
	virtual RID get_rid() const = 0;

	virtual bool has_alpha() const = 0;

	virtual void set_flags(uint32_t p_flags) = 0;
	virtual uint32_t get_flags() const = 0;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual bool get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const;

	virtual Image get_data() const { return Image(); }

	Texture();
};

VARIANT_ENUM_CAST(Texture::Flags);

class ImageTexture : public Texture {

	GDCLASS(ImageTexture, Texture);
	RES_BASE_EXTENSION("tex");

public:
	enum Storage {
		STORAGE_RAW,
		STORAGE_COMPRESS_LOSSY,
		STORAGE_COMPRESS_LOSSLESS
	};

private:
	RID texture;
	Image::Format format;
	uint32_t flags;
	int w, h;
	Storage storage;
	Size2 size_override;
	float lossy_storage_quality;

protected:
	virtual void reload_from_file();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _reload_hook(const RID &p_hook);
	virtual void _resource_path_changed();
	static void _bind_methods();

	void _set_data(Dictionary p_data);

public:
	void create(int p_width, int p_height, Image::Format p_format, uint32_t p_flags = FLAGS_DEFAULT);
	void create_from_image(const Image &p_image, uint32_t p_flags = FLAGS_DEFAULT);

	void set_flags(uint32_t p_flags);
	uint32_t get_flags() const;
	Image::Format get_format() const;
	void load(const String &p_path);
	void set_data(const Image &p_image);
	Image get_data() const;

	int get_width() const;
	int get_height() const;

	virtual RID get_rid() const;

	bool has_alpha() const;
	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	void set_storage(Storage p_storage);
	Storage get_storage() const;

	void set_lossy_storage_quality(float p_lossy_storage_quality);
	float get_lossy_storage_quality() const;

	void fix_alpha_edges();
	void premultiply_alpha();
	void normal_to_xy();
	void shrink_x2_and_keep_size();

	void set_size_override(const Size2 &p_size);

	virtual void set_path(const String &p_path, bool p_take_over = false);

	ImageTexture();
	~ImageTexture();
};

class StreamTexture : public Texture {

	GDCLASS(StreamTexture, Texture);

public:
	enum DataFormat {
		DATA_FORMAT_IMAGE,
		DATA_FORMAT_LOSSLESS,
		DATA_FORMAT_LOSSY
	};

	enum FormatBits {
		FORMAT_MASK_IMAGE_FORMAT = (1 << 20) - 1,
		FORMAT_BIT_LOSSLESS = 1 << 20,
		FORMAT_BIT_LOSSY = 1 << 21,
		FORMAT_BIT_STREAM = 1 << 22,
		FORMAT_BIT_HAS_MIPMAPS = 1 << 23,
		FORMAT_BIT_DETECT_3D = 1 << 24,
		FORMAT_BIT_DETECT_SRGB = 1 << 25,
	};

private:
	Error _load_data(const String &p_path, int &tw, int &th, int &flags, Image &image, int p_size_limit = 0);
	String path_to_file;
	RID texture;
	Image::Format format;
	uint32_t flags;
	int w, h;

	virtual void reload_from_file();

	static void _requested_3d(void *p_ud);
	static void _requested_srgb(void *p_ud);

protected:
	static void _bind_methods();

public:
	typedef void (*TextureFormatRequestCallback)(const Ref<StreamTexture> &);

	static TextureFormatRequestCallback request_3d_callback;
	static TextureFormatRequestCallback request_srgb_callback;

	uint32_t get_flags() const;
	Image::Format get_format() const;
	Error load(const String &p_path);
	String get_load_path() const;

	int get_width() const;
	int get_height() const;
	virtual RID get_rid() const;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;

	virtual bool has_alpha() const;
	virtual void set_flags(uint32_t p_flags);

	virtual Image get_data() const;

	StreamTexture();
	~StreamTexture();
};

class ResourceFormatLoaderStreamTexture : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

VARIANT_ENUM_CAST(ImageTexture::Storage);

class AtlasTexture : public Texture {

	GDCLASS(AtlasTexture, Texture);
	RES_BASE_EXTENSION("atex");

protected:
	Ref<Texture> atlas;
	Rect2 region;
	Rect2 margin;

	static void _bind_methods();

public:
	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	void set_atlas(const Ref<Texture> &p_atlas);
	Ref<Texture> get_atlas() const;

	void set_region(const Rect2 &p_region);
	Rect2 get_region() const;

	void set_margin(const Rect2 &p_margin);
	Rect2 get_margin() const;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual bool get_rect_region(const Rect2 &p_rect, const Rect2 &p_src_rect, Rect2 &r_rect, Rect2 &r_src_rect) const;

	AtlasTexture();
};

class LargeTexture : public Texture {

	GDCLASS(LargeTexture, Texture);
	RES_BASE_EXTENSION("ltex");

protected:
	struct Piece {

		Point2 offset;
		Ref<Texture> texture;
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

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	int add_piece(const Point2 &p_offset, const Ref<Texture> &p_texture);
	void set_piece_offset(int p_idx, const Point2 &p_offset);
	void set_piece_texture(int p_idx, const Ref<Texture> &p_texture);

	void set_size(const Size2 &p_size);
	void clear();

	int get_piece_count() const;
	Vector2 get_piece_offset(int p_idx) const;
	Ref<Texture> get_piece_texture(int p_idx) const;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const;

	LargeTexture();
};

class CubeMap : public Resource {

	GDCLASS(CubeMap, Resource);
	RES_BASE_EXTENSION("cbm");

public:
	enum Storage {
		STORAGE_RAW,
		STORAGE_COMPRESS_LOSSY,
		STORAGE_COMPRESS_LOSSLESS
	};

	enum Side {

		SIDE_LEFT,
		SIDE_RIGHT,
		SIDE_BOTTOM,
		SIDE_TOP,
		SIDE_FRONT,
		SIDE_BACK
	};

	enum Flags {
		FLAG_MIPMAPS = VisualServer::TEXTURE_FLAG_MIPMAPS,
		FLAG_REPEAT = VisualServer::TEXTURE_FLAG_REPEAT,
		FLAG_FILTER = VisualServer::TEXTURE_FLAG_FILTER,
		FLAGS_DEFAULT = FLAG_MIPMAPS | FLAG_REPEAT | FLAG_FILTER,
	};

private:
	bool valid[6];
	RID cubemap;
	Image::Format format;
	uint32_t flags;
	int w, h;
	Storage storage;
	Size2 size_override;
	float lossy_storage_quality;

	_FORCE_INLINE_ bool _is_valid() const {
		for (int i = 0; i < 6; i++) {
			if (valid[i]) return true;
		}
		return false;
	}

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	void set_flags(uint32_t p_flags);
	uint32_t get_flags() const;
	void set_side(Side p_side, const Image &p_image);
	Image get_side(Side p_side) const;

	Image::Format get_format() const;
	int get_width() const;
	int get_height() const;

	virtual RID get_rid() const;

	void set_storage(Storage p_storage);
	Storage get_storage() const;

	void set_lossy_storage_quality(float p_lossy_storage_quality);
	float get_lossy_storage_quality() const;

	virtual void set_path(const String &p_path, bool p_take_over = false);

	CubeMap();
	~CubeMap();
};

VARIANT_ENUM_CAST(CubeMap::Flags);
VARIANT_ENUM_CAST(CubeMap::Side);
VARIANT_ENUM_CAST(CubeMap::Storage);

class CurveTexture : public Texture {

	GDCLASS(CurveTexture, Texture);
	RES_BASE_EXTENSION("cvtex");

private:
	RID texture;
	PoolVector<Vector2> points;
	float min, max;
	int width;

protected:
	static void _bind_methods();

public:
	void set_max(float p_max);
	float get_max() const;

	void set_min(float p_min);
	float get_min() const;

	void set_width(int p_width);
	int get_width() const;

	void set_points(const PoolVector<Vector2> &p_points);
	PoolVector<Vector2> get_points() const;

	virtual RID get_rid() const;

	virtual int get_height() const { return 1; }
	virtual bool has_alpha() const { return false; }

	virtual void set_flags(uint32_t p_flags) {}
	virtual uint32_t get_flags() const { return FLAG_FILTER; }

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

class GradientTexture : public Texture {
	GDCLASS(GradientTexture, Texture);

public:
	struct Point {

		float offset;
		Color color;
		bool operator<(const Point &p_ponit) const {
			return offset < p_ponit.offset;
		}
	};

private:
	Vector<Point> points;
	bool is_sorted;
	bool update_pending;
	RID texture;
	int width;

	void _queue_update();
	void _update();

protected:
	static void _bind_methods();

public:
	void add_point(float p_offset, const Color &p_color);
	void remove_point(int p_index);

	void set_points(Vector<Point> &points);
	Vector<Point> &get_points();

	void set_offset(int pos, const float offset);
	float get_offset(int pos) const;

	void set_color(int pos, const Color &color);
	Color get_color(int pos) const;

	void set_offsets(const Vector<float> &offsets);
	Vector<float> get_offsets() const;

	void set_colors(const Vector<Color> &colors);
	Vector<Color> get_colors() const;

	void set_width(int p_width);
	int get_width() const;

	virtual RID get_rid() const { return texture; }
	virtual int get_height() const { return 1; }
	virtual bool has_alpha() const { return true; }

	virtual void set_flags(uint32_t p_flags) {}
	virtual uint32_t get_flags() const { return FLAG_FILTER; }

	_FORCE_INLINE_ Color get_color_at_offset(float p_offset) {

		if (points.empty())
			return Color(0, 0, 0, 1);

		if (!is_sorted) {
			points.sort();
			is_sorted = true;
		}

		//binary search
		int low = 0;
		int high = points.size() - 1;
		int middle;

		while (low <= high) {
			middle = (low + high) / 2;
			Point &point = points[middle];
			if (point.offset > p_offset) {
				high = middle - 1; //search low end of array
			} else if (point.offset < p_offset) {
				low = middle + 1; //search high end of array
			} else {
				return point.color;
			}
		}

		//return interpolated value
		if (points[middle].offset > p_offset) {
			middle--;
		}
		int first = middle;
		int second = middle + 1;
		if (second >= points.size())
			return points[points.size() - 1].color;
		if (first < 0)
			return points[0].color;
		Point &pointFirst = points[first];
		Point &pointSecond = points[second];
		return pointFirst.color.linear_interpolate(pointSecond.color, (p_offset - pointFirst.offset) / (pointSecond.offset - pointFirst.offset));
	}

	int get_points_count() const;

	GradientTexture();
	virtual ~GradientTexture();
};

#endif
