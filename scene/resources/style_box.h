/*************************************************************************/
/*  style_box.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef STYLE_BOX_H
#define STYLE_BOX_H

#include "core/io/resource.h"
#include "scene/resources/texture.h"
#include "servers/rendering_server.h"

class CanvasItem;

class StyleBox : public Resource {
	GDCLASS(StyleBox, Resource);
	RES_BASE_EXTENSION("stylebox");
	OBJ_SAVE_TYPE(StyleBox);
	float margin[4];

protected:
	virtual float get_style_margin(Side p_side) const = 0;
	static void _bind_methods();

public:
	virtual bool test_mask(const Point2 &p_point, const Rect2 &p_rect) const;

	void set_default_margin(Side p_side, float p_value);
	float get_default_margin(Side p_side) const;
	float get_margin(Side p_side) const;
	virtual Size2 get_center_size() const;

	virtual Rect2 get_draw_rect(const Rect2 &p_rect) const;
	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const = 0;

	CanvasItem *get_current_item_drawn() const;

	Size2 get_minimum_size() const;
	Point2 get_offset() const;

	StyleBox();
};

class StyleBoxEmpty : public StyleBox {
	GDCLASS(StyleBoxEmpty, StyleBox);
	virtual float get_style_margin(Side p_side) const override { return 0; }

public:
	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const override {}
	StyleBoxEmpty() {}
};

class StyleBoxTexture : public StyleBox {
	GDCLASS(StyleBoxTexture, StyleBox);

public:
	enum AxisStretchMode {
		AXIS_STRETCH_MODE_STRETCH,
		AXIS_STRETCH_MODE_TILE,
		AXIS_STRETCH_MODE_TILE_FIT,
	};

private:
	float expand_margin[4] = {};
	float margin[4] = {};
	Rect2 region_rect;
	Ref<Texture2D> texture;
	bool draw_center = true;
	Color modulate = Color(1, 1, 1, 1);
	AxisStretchMode axis_h = AXIS_STRETCH_MODE_STRETCH;
	AxisStretchMode axis_v = AXIS_STRETCH_MODE_STRETCH;

protected:
	virtual float get_style_margin(Side p_side) const override;
	static void _bind_methods();

public:
	void set_expand_margin_size(Side p_expand_side, float p_size);
	void set_expand_margin_size_all(float p_expand_margin_size);
	void set_expand_margin_size_individual(float p_left, float p_top, float p_right, float p_bottom);
	float get_expand_margin_size(Side p_expand_side) const;

	void set_margin_size(Side p_side, float p_size);
	float get_margin_size(Side p_side) const;

	void set_region_rect(const Rect2 &p_region_rect);
	Rect2 get_region_rect() const;

	void set_texture(Ref<Texture2D> p_texture);
	Ref<Texture2D> get_texture() const;

	void set_draw_center(bool p_enabled);
	bool is_draw_center_enabled() const;
	virtual Size2 get_center_size() const override;

	void set_h_axis_stretch_mode(AxisStretchMode p_mode);
	AxisStretchMode get_h_axis_stretch_mode() const;

	void set_v_axis_stretch_mode(AxisStretchMode p_mode);
	AxisStretchMode get_v_axis_stretch_mode() const;

	void set_modulate(const Color &p_modulate);
	Color get_modulate() const;

	virtual Rect2 get_draw_rect(const Rect2 &p_rect) const override;
	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const override;

	StyleBoxTexture();
	~StyleBoxTexture();
};

VARIANT_ENUM_CAST(StyleBoxTexture::AxisStretchMode)

class StyleBoxFlat : public StyleBox {
	GDCLASS(StyleBoxFlat, StyleBox);

	Color bg_color = Color(0.6, 0.6, 0.6);
	Color shadow_color = Color(0, 0, 0, 0.6);
	Color border_color = Color(0.8, 0.8, 0.8);

	int border_width[4] = {};
	int expand_margin[4] = {};
	int corner_radius[4] = {};

	bool draw_center = true;
	bool blend_border = false;
	bool anti_aliased = true;

	int corner_detail = 8;
	int shadow_size = 0;
	Point2 shadow_offset;
	int aa_size = 1;

protected:
	virtual float get_style_margin(Side p_side) const override;
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const override;

public:
	//Color
	void set_bg_color(const Color &p_color);
	Color get_bg_color() const;

	//Border Color
	void set_border_color(const Color &p_color);
	Color get_border_color() const;

	//BORDER
	//width
	void set_border_width_all(int p_size);
	int get_border_width_min() const;

	void set_border_width(Side p_side, int p_width);
	int get_border_width(Side p_side) const;

	//blend
	void set_border_blend(bool p_blend);
	bool get_border_blend() const;

	//CORNER
	void set_corner_radius_all(int radius);
	void set_corner_radius_individual(const int radius_top_left, const int radius_top_right, const int radius_bottom_right, const int radius_bottom_left);

	void set_corner_radius(Corner p_corner, const int radius);
	int get_corner_radius(Corner p_corner) const;

	void set_corner_detail(const int &p_corner_detail);
	int get_corner_detail() const;

	//EXPANDS
	void set_expand_margin_size(Side p_expand_side, float p_size);
	void set_expand_margin_size_all(float p_expand_margin_size);
	void set_expand_margin_size_individual(float p_left, float p_top, float p_right, float p_bottom);
	float get_expand_margin_size(Side p_expand_side) const;

	//DRAW CENTER
	void set_draw_center(bool p_enabled);
	bool is_draw_center_enabled() const;

	//SHADOW
	void set_shadow_color(const Color &p_color);
	Color get_shadow_color() const;

	void set_shadow_size(const int &p_size);
	int get_shadow_size() const;

	void set_shadow_offset(const Point2 &p_offset);
	Point2 get_shadow_offset() const;

	//ANTI_ALIASING
	void set_anti_aliased(const bool &p_anti_aliased);
	bool is_anti_aliased() const;
	//tempAA
	void set_aa_size(const int &p_aa_size);
	int get_aa_size() const;

	virtual Size2 get_center_size() const override;

	virtual Rect2 get_draw_rect(const Rect2 &p_rect) const override;
	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const override;

	StyleBoxFlat();
	~StyleBoxFlat();
};

// just used to draw lines.
class StyleBoxLine : public StyleBox {
	GDCLASS(StyleBoxLine, StyleBox);
	Color color;
	int thickness = 1;
	bool vertical = false;
	float grow_begin = 1.0;
	float grow_end = 1.0;

protected:
	virtual float get_style_margin(Side p_side) const override;
	static void _bind_methods();

public:
	void set_color(const Color &p_color);
	Color get_color() const;

	void set_thickness(int p_thickness);
	int get_thickness() const;

	void set_vertical(bool p_vertical);
	bool is_vertical() const;

	void set_grow_begin(float p_grow);
	float get_grow_begin() const;

	void set_grow_end(float p_grow);
	float get_grow_end() const;

	virtual Size2 get_center_size() const override;

	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const override;

	StyleBoxLine();
	~StyleBoxLine();
};

#endif
