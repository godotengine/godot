/**************************************************************************/
/*  navigation_obstacle3d.hpp                                             */
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
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class NavigationObstacle3D : public Node3D {
	GDEXTENSION_CLASS(NavigationObstacle3D, Node3D)

public:
	RID get_rid() const;
	void set_avoidance_enabled(bool p_enabled);
	bool get_avoidance_enabled() const;
	void set_navigation_map(const RID &p_navigation_map);
	RID get_navigation_map() const;
	void set_radius(float p_radius);
	float get_radius() const;
	void set_height(float p_height);
	float get_height() const;
	void set_velocity(const Vector3 &p_velocity);
	Vector3 get_velocity() const;
	void set_vertices(const PackedVector3Array &p_vertices);
	PackedVector3Array get_vertices() const;
	void set_avoidance_layers(uint32_t p_layers);
	uint32_t get_avoidance_layers() const;
	void set_avoidance_layer_value(int32_t p_layer_number, bool p_value);
	bool get_avoidance_layer_value(int32_t p_layer_number) const;
	void set_use_3d_avoidance(bool p_enabled);
	bool get_use_3d_avoidance() const;
	void set_affect_navigation_mesh(bool p_enabled);
	bool get_affect_navigation_mesh() const;
	void set_carve_navigation_mesh(bool p_enabled);
	bool get_carve_navigation_mesh() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

