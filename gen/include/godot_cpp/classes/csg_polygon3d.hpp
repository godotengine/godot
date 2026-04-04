/**************************************************************************/
/*  csg_polygon3d.hpp                                                     */
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

#include <godot_cpp/classes/csg_primitive3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Material;

class CSGPolygon3D : public CSGPrimitive3D {
	GDEXTENSION_CLASS(CSGPolygon3D, CSGPrimitive3D)

public:
	enum Mode {
		MODE_DEPTH = 0,
		MODE_SPIN = 1,
		MODE_PATH = 2,
	};

	enum PathRotation {
		PATH_ROTATION_POLYGON = 0,
		PATH_ROTATION_PATH = 1,
		PATH_ROTATION_PATH_FOLLOW = 2,
	};

	enum PathIntervalType {
		PATH_INTERVAL_DISTANCE = 0,
		PATH_INTERVAL_SUBDIVIDE = 1,
	};

	void set_polygon(const PackedVector2Array &p_polygon);
	PackedVector2Array get_polygon() const;
	void set_mode(CSGPolygon3D::Mode p_mode);
	CSGPolygon3D::Mode get_mode() const;
	void set_depth(float p_depth);
	float get_depth() const;
	void set_spin_degrees(float p_degrees);
	float get_spin_degrees() const;
	void set_spin_sides(int32_t p_spin_sides);
	int32_t get_spin_sides() const;
	void set_path_node(const NodePath &p_path);
	NodePath get_path_node() const;
	void set_path_interval_type(CSGPolygon3D::PathIntervalType p_interval_type);
	CSGPolygon3D::PathIntervalType get_path_interval_type() const;
	void set_path_interval(float p_interval);
	float get_path_interval() const;
	void set_path_simplify_angle(float p_degrees);
	float get_path_simplify_angle() const;
	void set_path_rotation(CSGPolygon3D::PathRotation p_path_rotation);
	CSGPolygon3D::PathRotation get_path_rotation() const;
	void set_path_rotation_accurate(bool p_enable);
	bool get_path_rotation_accurate() const;
	void set_path_local(bool p_enable);
	bool is_path_local() const;
	void set_path_continuous_u(bool p_enable);
	bool is_path_continuous_u() const;
	void set_path_u_distance(float p_distance);
	float get_path_u_distance() const;
	void set_path_joined(bool p_enable);
	bool is_path_joined() const;
	void set_material(const Ref<Material> &p_material);
	Ref<Material> get_material() const;
	void set_smooth_faces(bool p_smooth_faces);
	bool get_smooth_faces() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		CSGPrimitive3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CSGPolygon3D::Mode);
VARIANT_ENUM_CAST(CSGPolygon3D::PathRotation);
VARIANT_ENUM_CAST(CSGPolygon3D::PathIntervalType);

