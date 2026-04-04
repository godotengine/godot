/**************************************************************************/
/*  csg_shape3d.hpp                                                       */
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

#include <godot_cpp/classes/geometry_instance3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class ArrayMesh;
class ConcavePolygonShape3D;

class CSGShape3D : public GeometryInstance3D {
	GDEXTENSION_CLASS(CSGShape3D, GeometryInstance3D)

public:
	enum Operation {
		OPERATION_UNION = 0,
		OPERATION_INTERSECTION = 1,
		OPERATION_SUBTRACTION = 2,
	};

	bool is_root_shape() const;
	void set_operation(CSGShape3D::Operation p_operation);
	CSGShape3D::Operation get_operation() const;
	void set_snap(float p_snap);
	float get_snap() const;
	void set_use_collision(bool p_operation);
	bool is_using_collision() const;
	void set_collision_layer(uint32_t p_layer);
	uint32_t get_collision_layer() const;
	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;
	void set_collision_mask_value(int32_t p_layer_number, bool p_value);
	bool get_collision_mask_value(int32_t p_layer_number) const;
	void set_collision_layer_value(int32_t p_layer_number, bool p_value);
	bool get_collision_layer_value(int32_t p_layer_number) const;
	void set_collision_priority(float p_priority);
	float get_collision_priority() const;
	Ref<ConcavePolygonShape3D> bake_collision_shape();
	void set_calculate_tangents(bool p_enabled);
	bool is_calculating_tangents() const;
	Array get_meshes() const;
	Ref<ArrayMesh> bake_static_mesh();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		GeometryInstance3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(CSGShape3D::Operation);

