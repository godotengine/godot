/*************************************************************************/
/*  texture_progress.h                                                   */
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
		FILL_COUNTER_CLOCKWISE
	};

	void set_fill_mode(int p_fill);
	int get_fill_mode();

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

	Size2 get_minimum_size() const;

	TextureProgress();

private:
	FillMode mode;
	float rad_init_angle;
	float rad_max_degrees;
	Point2 rad_center_off;

	Point2 unit_val_to_uv(float val);
	Point2 get_relative_center();
};

#endif // TEXTURE_PROGRESS_H
