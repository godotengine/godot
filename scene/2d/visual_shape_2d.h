/**************************************************************************/
/*  visual_shape_2d.h                                                     */
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

#ifndef VISUAL_SHAPE_2D_H
#define VISUAL_SHAPE_2D_H

#include "scene/2d/node_2d.h"

class VisualShape2D : public Node2D {
	GDCLASS(VisualShape2D, Node2D);

public:
	enum ShapeType {
		SHAPE_RECTANGLE,
		SHAPE_CIRCLE,
		SHAPE_EQUILATERAL_TRIANGLE,
		SHAPE_RIGHT_TRIANGLE,
		SHAPE_CAPSULE,
	};

private:
	ShapeType shape_type = SHAPE_RECTANGLE;
	Color color = Color(1, 1, 1, 1);
	Size2 size = Size2(128, 128);
	Point2 offset;

	bool antialiased = false;
	float outline_width = 0;
	int resolution = 64;

	PackedVector2Array _get_shape_points() const;

protected:
	void _notification(int p_what);

	static void _bind_methods();
	void _validate_property(PropertyInfo &p_property) const;

public:
#ifdef TOOLS_ENABLED
	virtual Dictionary _edit_get_state() const override;
	virtual void _edit_set_state(const Dictionary &p_state) override;

	virtual void _edit_set_pivot(const Point2 &p_pivot) override;
	virtual Point2 _edit_get_pivot() const override;
	virtual bool _edit_use_pivot() const override;
	virtual void _edit_set_rect(const Rect2 &p_edit_rect) override;

#endif // TOOLS_ENABLED

#ifdef DEBUG_ENABLED
	virtual bool _edit_is_selected_on_click(const Point2 &p_point, double p_tolerance) const override;

	virtual Rect2 _edit_get_rect() const override;
	virtual bool _edit_use_rect() const override;
#endif // DEBUG_ENABLED

	PackedVector2Array get_points() const;
	PackedVector2Array get_uvs() const;
	Rect2 get_rect() const;

	void set_shape_type(ShapeType p_shape_type);
	ShapeType get_shape_type() const;
	void set_color(const Color &p_color);
	Color get_color() const;
	void set_size(const Size2 &p_size);
	Size2 get_size() const;
	void set_offset(const Point2 &p_offset);
	Point2 get_offset() const;

	void set_antialiased(bool p_antialiased);
	bool is_antialiased() const;
	void set_outline_width(float p_outline_width);
	float get_outline_width() const;
	void set_resolution(int p_resolution);
	int get_resolution() const;
};

VARIANT_ENUM_CAST(VisualShape2D::ShapeType)

#endif // VISUAL_SHAPE_2D_H
