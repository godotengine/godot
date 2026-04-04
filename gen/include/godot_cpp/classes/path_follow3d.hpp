/**************************************************************************/
/*  path_follow3d.hpp                                                     */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/variant/transform3d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class PathFollow3D : public Node3D {
	GDEXTENSION_CLASS(PathFollow3D, Node3D)

public:
	enum RotationMode {
		ROTATION_NONE = 0,
		ROTATION_Y = 1,
		ROTATION_XY = 2,
		ROTATION_XYZ = 3,
		ROTATION_ORIENTED = 4,
	};

	void set_progress(float p_progress);
	float get_progress() const;
	void set_h_offset(float p_h_offset);
	float get_h_offset() const;
	void set_v_offset(float p_v_offset);
	float get_v_offset() const;
	void set_progress_ratio(float p_ratio);
	float get_progress_ratio() const;
	void set_rotation_mode(PathFollow3D::RotationMode p_rotation_mode);
	PathFollow3D::RotationMode get_rotation_mode() const;
	void set_cubic_interpolation(bool p_enabled);
	bool get_cubic_interpolation() const;
	void set_use_model_front(bool p_enabled);
	bool is_using_model_front() const;
	void set_loop(bool p_loop);
	bool has_loop() const;
	void set_tilt_enabled(bool p_enabled);
	bool is_tilt_enabled() const;
	static Transform3D correct_posture(const Transform3D &p_transform, PathFollow3D::RotationMode p_rotation_mode);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(PathFollow3D::RotationMode);

