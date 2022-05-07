/*************************************************************************/
/*  texture_progress.h                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef TEXTURE_PROGRESS_H
#define TEXTURE_PROGRESS_H

#include "scene/gui/range.h"

class TextureProgress : public Range {
	GDCLASS(TextureProgress, Range);

	Ref<Texture> under;
	Ref<Texture> progress;
	Ref<Texture> over;

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	enum FillMode {
		FILL_LEFT_TO_RIGHT = 0,
		FILL_RIGHT_TO_LEFT,
		FILL_TOP_TO_BOTTOM,
		FILL_BOTTOM_TO_TOP,
		FILL_CLOCKWISE,
		FILL_COUNTER_CLOCKWISE,
		FILL_BILINEAR_LEFT_AND_RIGHT,
		FILL_BILINEAR_TOP_AND_BOTTOM,
		FILL_CLOCKWISE_AND_COUNTER_CLOCKWISE,
		FILL_MODE_MAX,
	};

	void set_fill_mode(int p_fill);
	int get_fill_mode();

	void set_progress_offset(Point2 p_offset);
	Point2 get_progress_offset() const;

	void set_radial_initial_angle(float p_angle);
	float get_radial_initial_angle();

	void set_fill_degrees(float p_angle);
	float get_fill_degrees();

	void set_radial_center_offset(const Point2 &p_off);
	Point2 get_radial_center_offset();

	void set_under_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_under_texture() const;

	void set_progress_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_progress_texture() const;

	void set_over_texture(const Ref<Texture> &p_texture);
	Ref<Texture> get_over_texture() const;

	void set_stretch_margin(Margin p_margin, int p_size);
	int get_stretch_margin(Margin p_margin) const;

	void set_nine_patch_stretch(bool p_stretch);
	bool get_nine_patch_stretch() const;

	void set_tint_under(const Color &p_tint);
	Color get_tint_under() const;

	void set_tint_progress(const Color &p_tint);
	Color get_tint_progress() const;

	void set_tint_over(const Color &p_tint);
	Color get_tint_over() const;

	Size2 get_minimum_size() const;

	TextureProgress();

private:
	FillMode mode;
	Point2 progress_offset;
	float rad_init_angle;
	float rad_max_degrees;
	Point2 rad_center_off;
	bool nine_patch_stretch;
	int stretch_margin[4];
	Color tint_under, tint_progress, tint_over;

	Point2 unit_val_to_uv(float val);
	Point2 get_relative_center();
	void draw_nine_patch_stretched(const Ref<Texture> &p_texture, FillMode p_mode, double p_ratio, const Color &p_modulate);
};

VARIANT_ENUM_CAST(TextureProgress::FillMode);

#endif // TEXTURE_PROGRESS_H
