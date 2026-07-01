/**************************************************************************/
/*  trail_2d.h                                                            */
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

#include "scene/2d/line_2d.h"
#include "scene/2d/node_2d.h"
#include "scene/resources/gradient.h"

class Trail2D : public Node2D {
	GDCLASS(Trail2D, Node2D);

public:
	Trail2D();

	void set_emitting(bool p_emitting);
	bool is_emitting() const;

	void set_lifetime(double p_lifetime);
	double get_lifetime() const;

	void set_segment_length(float p_segment_length);
	float get_segment_length() const;

	void set_width(float p_width);
	float get_width() const;

	void set_curve(const Ref<Curve> &curve);
	Ref<Curve> get_curve() const;

	void set_default_color(Color color);
	Color get_default_color() const;

	void set_gradient(const Ref<Gradient> &gradient);
	Ref<Gradient> get_gradient() const;

	void set_texture(const Ref<Texture2D> &texture);
	Ref<Texture2D> get_texture() const;

	void set_texture_mode(const Line2D::LineTextureMode mode);
	Line2D::LineTextureMode get_texture_mode() const;

	void set_joint_mode(Line2D::LineJointMode mode);
	Line2D::LineJointMode get_joint_mode() const;

	void set_begin_cap_mode(Line2D::LineCapMode mode);
	Line2D::LineCapMode get_begin_cap_mode() const;

	void set_end_cap_mode(Line2D::LineCapMode mode);
	Line2D::LineCapMode get_end_cap_mode() const;

	void set_sharp_limit(float limit);
	float get_sharp_limit() const;

	void set_round_precision(int precision);
	int get_round_precision() const;

protected:
	void _notification(int p_what);
	void _update_internal();
	void _draw();

	static void _bind_methods();

private:
	void _gradient_changed();
	void _curve_changed();

private:
	bool emitting = false;
	bool active = false;

	struct TrailPoint {
		Vector2 position;
		double time;
	};

	double lifetime = 1.0;
	float segment_length = 30.0f;
	Vector<TrailPoint> _points;
	Line2D::LineJointMode _joint_mode = Line2D::LINE_JOINT_SHARP;
	Line2D::LineCapMode _begin_cap_mode = Line2D::LINE_CAP_NONE;
	Line2D::LineCapMode _end_cap_mode = Line2D::LINE_CAP_NONE;
	float _width = 10.0;
	Ref<Curve> _curve;
	Color _default_color = Color(1, 1, 1);
	Ref<Gradient> _gradient;
	Ref<Texture2D> _texture;
	Line2D::LineTextureMode _texture_mode = Line2D::LINE_TEXTURE_NONE;
	float _sharp_limit = 2.f;
	int _round_precision = 8;

	Point2 _last_position;
	float _end_age_diff = 0.0f;
};
