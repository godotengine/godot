/*************************************************************************/
/*  path_2d.h                                                            */
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

#ifndef PATH_2D_H
#define PATH_2D_H

#include "scene/2d/node_2d.h"
#include "scene/resources/curve.h"

class Path2D : public Node2D {
	GDCLASS(Path2D, Node2D);

	Ref<Curve2D> curve;
	Vector<Vector2> _cached_draw_pts;

	void _curve_changed();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
#ifdef TOOLS_ENABLED
	virtual Rect2 _edit_get_rect() const override;
	virtual bool _edit_use_rect() const override;
	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const override;
#endif

	void set_curve(const Ref<Curve2D> &p_curve);
	Ref<Curve2D> get_curve() const;

	Path2D() {}
};

class PathFollow2D : public Node2D {
	GDCLASS(PathFollow2D, Node2D);

public:
private:
	Path2D *path = nullptr;
	real_t offset = 0.0;
	real_t h_offset = 0.0;
	real_t v_offset = 0.0;
	real_t lookahead = 4.0;
	bool cubic = true;
	bool loop = true;
	bool rotates = true;

	void _update_transform();

protected:
	virtual void _validate_property(PropertyInfo &property) const override;

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_offset(real_t p_offset);
	real_t get_offset() const;

	void set_h_offset(real_t p_h_offset);
	real_t get_h_offset() const;

	void set_v_offset(real_t p_v_offset);
	real_t get_v_offset() const;

	void set_unit_offset(real_t p_unit_offset);
	real_t get_unit_offset() const;

	void set_lookahead(real_t p_lookahead);
	real_t get_lookahead() const;

	void set_loop(bool p_loop);
	bool has_loop() const;

	void set_rotates(bool p_rotates);
	bool is_rotating() const;

	void set_cubic_interpolation(bool p_enable);
	bool get_cubic_interpolation() const;

	TypedArray<String> get_configuration_warnings() const override;

	PathFollow2D() {}
};

#endif // PATH_2D_H
