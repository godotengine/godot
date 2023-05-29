/**************************************************************************/
/*  style_box.h                                                           */
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

#ifndef STYLE_BOX_H
#define STYLE_BOX_H

#include "core/io/resource.h"
#include "core/typedefs.h"
#include "scene/resources/texture.h"
#include "servers/rendering_server.h"

class CanvasItem;

class StyleBox : public Resource {
	GDCLASS(StyleBox, Resource);
	RES_BASE_EXTENSION("stylebox");
	OBJ_SAVE_TYPE(StyleBox);
	float content_margin[4];

protected:
	virtual float get_style_margin(Side p_side) const { return 0; }
	static void _bind_methods();

	GDVIRTUAL2C(_draw, RID, Rect2)
	GDVIRTUAL1RC(Rect2, _get_draw_rect, Rect2)
	GDVIRTUAL0RC(Size2, _get_minimum_size)
	GDVIRTUAL2RC(bool, _test_mask, Point2, Rect2)

public:
	virtual Size2 get_minimum_size() const;

	void set_content_margin(Side p_side, float p_value);
	void set_content_margin_all(float p_value);
	void set_content_margin_individual(float p_left, float p_top, float p_right, float p_bottom);
	float get_content_margin(Side p_side) const;

	float get_margin(Side p_side) const;
	Point2 get_offset() const;

	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const;
	virtual Rect2 get_draw_rect(const Rect2 &p_rect) const;

	CanvasItem *get_current_item_drawn() const;

	virtual bool test_mask(const Point2 &p_point, const Rect2 &p_rect) const;

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
	float texture_margin[4] = {};
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
	void set_expand_margin(Side p_expand_side, float p_size);
	void set_expand_margin_all(float p_expand_margin_size);
	void set_expand_margin_individual(float p_left, float p_top, float p_right, float p_bottom);
	float get_expand_margin(Side p_expand_side) const;

	void set_texture_margin(Side p_side, float p_size);
	void set_texture_margin_all(float p_size);
	void set_texture_margin_individual(float p_left, float p_top, float p_right, float p_bottom);
	float get_texture_margin(Side p_side) const;

	void set_region_rect(const Rect2 &p_region_rect);
	Rect2 get_region_rect() const;

	void set_texture(Ref<Texture2D> p_texture);
	Ref<Texture2D> get_texture() const;

	void set_draw_center(bool p_enabled);
	bool is_draw_center_enabled() const;

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

public:
	enum BorderColorJoinStyle {
		BORDER_COLOR_JOIN_STYLE_GRADIENT,
		BORDER_COLOR_JOIN_STYLE_SHARP,
	};
	enum BorderSideColoringStyle {
		BORDER_SIDE_COLORING_STYLE_SOLID,
		BORDER_SIDE_COLORING_STYLE_SIMPLE_GRADIENT,
	};
	enum BorderColoringStyle {
		BORDER_COLORING_STYLE_SOLID,
		BORDER_COLORING_STYLE_SIMPLE_GRADIENT,
		BORDER_COLORING_STYLE_SIMPLE_2D_GRADIENT,
		BORDER_COLORING_STYLE_OUTSET,
		BORDER_COLORING_STYLE_INSET,
	};
	enum FillStyle {
		FILL_STYLE_SOLID,
		FILL_STYLE_SIMPLE_GRADIENT,
	};
	enum SimpleGradientOrientation {
		SIMPLE_GRADIENT_ORIENTATION_LEFT_TO_RIGHT,
		SIMPLE_GRADIENT_ORIENTATION_TOP_TO_BOTTOM,
		SIMPLE_GRADIENT_ORIENTATION_RIGHT_TO_LEFT,
		SIMPLE_GRADIENT_ORIENTATION_BOTTOM_TO_TOP,
	};
	enum SimpleGradientColoringAlgorithm {
		GRADIENT_ALGO_INTERPOLATE_CORNER_LIMITS,
		GRADIENT_ALGO_FULL_CORNER_LIMITS,
	};

private:
	Color bg_color = Color(0.6, 0.6, 0.6);
	Color bg_secondary_color = Color(0.4, 0.4, 0.4);
	Color shadow_color = Color(0, 0, 0, 0.6);
	Color border_color = Color(0.8, 0.8, 0.8);
	Color border_secondary_color = Color(0.7, 0.7, 0.7);
	Color border_third_color = Color(0.6, 0.6, 0.6);
	Color border_fourth_color = Color(0.5, 0.5, 0.5);
	Color border_bevel_light_color = Color(1, 1, 1);
	Color border_bevel_dark_color = Color(0, 0, 0);
	Color border_side_colors[4] = { Color(1, 1, 1), Color(1, 1, 1), Color(1, 1, 1), Color(1, 1, 1) };
	Color border_side_secondary_colors[4] = { Color(1, 1, 1), Color(1, 1, 1), Color(1, 1, 1), Color(1, 1, 1) };

	real_t border_width[4] = {};
	real_t expand_margin[4] = {};
	real_t corner_radius[4] = {};

	bool draw_center = true;
	bool blend_border = false;
	BorderColorJoinStyle color_join_style = BORDER_COLOR_JOIN_STYLE_GRADIENT;
	BorderColoringStyle border_coloring_style = BORDER_COLORING_STYLE_SOLID;
	SimpleGradientOrientation border_gradient_orientation = SIMPLE_GRADIENT_ORIENTATION_LEFT_TO_RIGHT;
	SimpleGradientColoringAlgorithm border_gradient_algo = GRADIENT_ALGO_INTERPOLATE_CORNER_LIMITS;
	bool border_side_colors_defined[4] = { false, false, false, false };
	BorderSideColoringStyle border_side_coloring_styles[4] = { BORDER_SIDE_COLORING_STYLE_SOLID, BORDER_SIDE_COLORING_STYLE_SOLID, BORDER_SIDE_COLORING_STYLE_SOLID, BORDER_SIDE_COLORING_STYLE_SOLID };

	FillStyle center_fill_style = FILL_STYLE_SOLID;
	SimpleGradientOrientation center_gradient_orientation = SIMPLE_GRADIENT_ORIENTATION_LEFT_TO_RIGHT;

	real_t bevel_lighting_intensity = 0.3;
	real_t bevel_darkening_intensity = 0.3;
	real_t bevel_lighting_angle = 135.0;
	real_t bevel_max_intensity_angle_ratio = 0.5;
	real_t corner_color_join_angles[4] = { 0.5, 0.5, 0.5, 0.5 };

	Vector2 skew;
	bool anti_aliased = true;

	int corner_detail = 8;
	int shadow_size = 0;
	Point2 shadow_offset;
	real_t aa_size = 1;

protected:
	virtual float get_style_margin(Side p_side) const override;
	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

public:
	void set_bg_color(const Color &p_color);
	Color get_bg_color() const;

	void set_bg_secondary_color(const Color &p_color);
	Color get_bg_secondary_color() const;

	void set_bg_simple_gradient_orientation(SimpleGradientOrientation p_orientation);
	SimpleGradientOrientation get_bg_simple_gradient_orientation() const;

	void set_border_color(const Color &p_color);
	Color get_border_color() const;

	void set_border_secondary_color(const Color &p_color);
	Color get_border_secondary_color() const;

	void set_border_third_color(const Color &p_color);
	Color get_border_third_color() const;

	void set_border_fourth_color(const Color &p_color);
	Color get_border_fourth_color() const;

	void set_border_simple_gradient_orientation(SimpleGradientOrientation p_orientation);
	SimpleGradientOrientation get_border_simple_gradient_orientation() const;

	void set_border_simple_gradient_algorithm(SimpleGradientColoringAlgorithm p_algo);
	SimpleGradientColoringAlgorithm get_border_simple_gradient_algorithm() const;

	void set_border_bevel_lighting_color(const Color &p_color);
	Color get_border_bevel_lighting_color() const;

	void set_border_bevel_darkening_color(const Color &p_color);
	Color get_border_bevel_darkening_color() const;

	void set_border_bevel_lighting_intensity(float p_intensity);
	float get_border_bevel_lighting_intensity() const;

	void set_border_bevel_darkening_intensity(float p_intensity);
	float get_border_bevel_darkening_intensity() const;

	void set_border_bevel_lighting_angle(float p_angle);
	float get_border_bevel_lighting_angle() const;

	void set_border_bevel_max_intensity_angle_ratio(float p_ratio);
	float get_border_bevel_max_intensity_angle_ratio() const;

	void set_border_side_color(Side p_expand_side, const Color &p_color);
	void set_border_side_color_all(const Color &p_color);
	void set_border_side_color_individual(const Color &p_left, const Color &p_top, const Color &p_right, const Color &p_bottom);
	Color get_border_side_color(Side p_expand_side) const;

	void set_border_side_secondary_color(Side p_expand_side, const Color &p_color);
	void set_border_side_secondary_color_all(const Color &p_color);
	void set_border_side_secondary_color_individual(const Color &p_left, const Color &p_top, const Color &p_right, const Color &p_bottom);
	Color get_border_side_secondary_color(Side p_expand_side) const;

	void set_border_side_color_defined(Side p_side, bool p_defined);
	bool get_border_side_color_defined(Side p_side) const;

	void set_border_side_coloring_style(Side p_side, BorderSideColoringStyle p_coloring_style);
	BorderSideColoringStyle get_border_side_coloring_style(Side p_side) const;

	void set_corner_color_join_angle(Corner p_corner, float p_angle);
	float get_corner_color_join_angle(Corner p_corner) const;

	void set_border_width_all(int p_size);
	int get_border_width_min() const;

	void set_border_width(Side p_side, int p_width);
	int get_border_width(Side p_side) const;

	void set_border_blend(bool p_blend);
	bool get_border_blend() const;

	void set_border_coloring_style(BorderColoringStyle p_coloring_style);
	BorderColoringStyle get_border_coloring_style() const;

	void set_border_color_join_style(BorderColorJoinStyle p_join_style);
	BorderColorJoinStyle get_border_color_join_style() const;

	void set_corner_radius_all(int radius);
	void set_corner_radius_individual(const int radius_top_left, const int radius_top_right, const int radius_bottom_right, const int radius_bottom_left);

	void set_corner_radius(Corner p_corner, const int radius);
	int get_corner_radius(Corner p_corner) const;

	void set_corner_detail(const int &p_corner_detail);
	int get_corner_detail() const;

	void set_center_fill_style(FillStyle p_fill_style);
	FillStyle get_center_fill_style() const;

	void set_expand_margin(Side p_expand_side, float p_size);
	void set_expand_margin_all(float p_expand_margin_size);
	void set_expand_margin_individual(float p_left, float p_top, float p_right, float p_bottom);
	float get_expand_margin(Side p_expand_side) const;

	void set_draw_center(bool p_enabled);
	bool is_draw_center_enabled() const;

	void set_skew(Vector2 p_skew);
	Vector2 get_skew() const;

	void set_shadow_color(const Color &p_color);
	Color get_shadow_color() const;

	void set_shadow_size(const int &p_size);
	int get_shadow_size() const;

	void set_shadow_offset(const Point2 &p_offset);
	Point2 get_shadow_offset() const;

	void set_anti_aliased(const bool &p_anti_aliased);
	bool is_anti_aliased() const;
	void set_aa_size(const real_t p_aa_size);
	real_t get_aa_size() const;

	virtual Rect2 get_draw_rect(const Rect2 &p_rect) const override;
	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const override;

	StyleBoxFlat();
	~StyleBoxFlat();
};

VARIANT_ENUM_CAST(StyleBoxFlat::BorderColorJoinStyle)
VARIANT_ENUM_CAST(StyleBoxFlat::BorderSideColoringStyle)
VARIANT_ENUM_CAST(StyleBoxFlat::BorderColoringStyle)
VARIANT_ENUM_CAST(StyleBoxFlat::FillStyle)
VARIANT_ENUM_CAST(StyleBoxFlat::SimpleGradientOrientation)
VARIANT_ENUM_CAST(StyleBoxFlat::SimpleGradientColoringAlgorithm)

// Just used to draw lines.
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

	virtual void draw(RID p_canvas_item, const Rect2 &p_rect) const override;

	StyleBoxLine();
	~StyleBoxLine();
};

#endif // STYLE_BOX_H
