/**************************************************************************/
/*  transform_container.h                                                 */
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

#include "scene/gui/container.h"

class TransformContainer : public Container {
	GDCLASS(TransformContainer, Container);

	Vector2 transform_offset;
	bool transform_offset_relative_to_size = false;
	Vector2 transform_scale = Vector2(1, 1);
	real_t transform_rotation = 0;
	Vector2 transform_pivot;
	bool transform_pivot_relative_to_size = true;

protected:
	void _notification(int p_what);
	static void _bind_methods();

	virtual Size2 get_minimum_size() const override;
	virtual Vector<int> get_allowed_size_flags_horizontal() const override;
	virtual Vector<int> get_allowed_size_flags_vertical() const override;

public:
	void set_transform_offset(const Vector2 &p_offset);
	Vector2 get_transform_offset() const { return transform_offset; }

	void set_transform_offset_relative_to_size(bool p_relative);
	bool is_transform_offset_relative_to_size() const { return transform_offset_relative_to_size; }

	void set_transform_scale(const Vector2 &p_scale);
	Vector2 get_transform_scale() const { return transform_scale; }

	void set_transform_rotation(real_t p_rotation);
	real_t get_transform_rotation() const { return transform_rotation; }

	void set_transform_rotation_degrees(real_t p_degrees) { set_transform_rotation(Math::deg_to_rad(p_degrees)); }
	real_t get_transform_rotation_degrees() const { return Math::rad_to_deg(transform_rotation); }

	void set_transform_pivot(const Vector2 &p_pivot);
	Vector2 get_transform_pivot() const { return transform_pivot; }

	void set_transform_pivot_relative_to_size(bool p_relative);
	bool is_transform_pivot_relative_to_size() const { return transform_pivot_relative_to_size; }
};
