/**************************************************************************/
/*  drawable_texture_2d.h                                                 */
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

#include "scene/resources/atlas_texture.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/material.h"

class DrawableTexture2D : public Texture2D {
	GDCLASS(DrawableTexture2D, Texture2D);
	RES_BASE_EXTENSION("tex");

public:
	enum DrawableFormat {
		DRAWABLE_FORMAT_RGBA8,
		DRAWABLE_FORMAT_RGBA8_SRGB,
		DRAWABLE_FORMAT_RGBAH,
		DRAWABLE_FORMAT_RGBAF,
	};

private:
	mutable RID texture;
	int width = 64;
	int height = 64;
	bool mipmaps = false;
	DrawableFormat format = DRAWABLE_FORMAT_RGBA8;

	Color base_color = Color(1, 1, 1, 1);

	RID default_material;

	void _initialize();

protected:
	static void _bind_methods();

public:
	void set_width(int p_width);
	int get_width() const override;
	void set_height(int p_height);
	int get_height() const override;

	void set_format(DrawableFormat p_format);
	DrawableFormat get_format() const;
	void set_use_mipmaps(bool p_mipmaps);
	bool get_use_mipmaps() const;

	virtual RID get_rid() const override;

	virtual void draw(RID p_canvas_item, const Point2 &p_pos, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const override;
	virtual void draw_rect(RID p_canvas_item, const Rect2 &p_rect, bool p_tile = false, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false) const override;
	virtual void draw_rect_region(RID p_canvas_item, const Rect2 &p_rect, const Rect2 &p_src_rect, const Color &p_modulate = Color(1, 1, 1), bool p_transpose = false, bool p_clip_uv = true) const override;

	void setup(int p_width, int p_height, DrawableFormat p_format, const Color &p_modulate = Color(1, 1, 1, 1), bool p_use_mipmaps = false);

	void blit_rect(const Rect2i p_rect, const Ref<Texture2D> &p_source, const Color &p_modulate = Color(1, 1, 1, 1), int p_mipmap = 0, const Ref<Material> &p_material = Ref<Material>());
	void blit_rect_multi(const Rect2i p_rect, const TypedArray<Texture2D> &p_sources, const TypedArray<DrawableTexture2D> &p_extra_targets, const Color &p_modulate = Color(1, 1, 1, 1), int p_mipmap = 0, const Ref<Material> &p_material = Ref<Material>());

	virtual Ref<Image> get_image() const override;

	void generate_mipmaps();

	DrawableTexture2D();
	~DrawableTexture2D();
};

VARIANT_ENUM_CAST(DrawableTexture2D::DrawableFormat)
