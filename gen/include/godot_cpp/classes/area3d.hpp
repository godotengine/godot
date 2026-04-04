/**************************************************************************/
/*  area3d.hpp                                                            */
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

#include <godot_cpp/classes/collision_object3d.hpp>
#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Node;
class Node3D;

class Area3D : public CollisionObject3D {
	GDEXTENSION_CLASS(Area3D, CollisionObject3D)

public:
	enum SpaceOverride {
		SPACE_OVERRIDE_DISABLED = 0,
		SPACE_OVERRIDE_COMBINE = 1,
		SPACE_OVERRIDE_COMBINE_REPLACE = 2,
		SPACE_OVERRIDE_REPLACE = 3,
		SPACE_OVERRIDE_REPLACE_COMBINE = 4,
	};

	void set_gravity_space_override_mode(Area3D::SpaceOverride p_space_override_mode);
	Area3D::SpaceOverride get_gravity_space_override_mode() const;
	void set_gravity_is_point(bool p_enable);
	bool is_gravity_a_point() const;
	void set_gravity_point_unit_distance(float p_distance_scale);
	float get_gravity_point_unit_distance() const;
	void set_gravity_point_center(const Vector3 &p_center);
	Vector3 get_gravity_point_center() const;
	void set_gravity_direction(const Vector3 &p_direction);
	Vector3 get_gravity_direction() const;
	void set_gravity(float p_gravity);
	float get_gravity() const;
	void set_linear_damp_space_override_mode(Area3D::SpaceOverride p_space_override_mode);
	Area3D::SpaceOverride get_linear_damp_space_override_mode() const;
	void set_angular_damp_space_override_mode(Area3D::SpaceOverride p_space_override_mode);
	Area3D::SpaceOverride get_angular_damp_space_override_mode() const;
	void set_angular_damp(float p_angular_damp);
	float get_angular_damp() const;
	void set_linear_damp(float p_linear_damp);
	float get_linear_damp() const;
	void set_priority(int32_t p_priority);
	int32_t get_priority() const;
	void set_wind_force_magnitude(float p_wind_force_magnitude);
	float get_wind_force_magnitude() const;
	void set_wind_attenuation_factor(float p_wind_attenuation_factor);
	float get_wind_attenuation_factor() const;
	void set_wind_source_path(const NodePath &p_wind_source_path);
	NodePath get_wind_source_path() const;
	void set_monitorable(bool p_enable);
	bool is_monitorable() const;
	void set_monitoring(bool p_enable);
	bool is_monitoring() const;
	TypedArray<Node3D> get_overlapping_bodies() const;
	TypedArray<Area3D> get_overlapping_areas() const;
	bool has_overlapping_bodies() const;
	bool has_overlapping_areas() const;
	bool overlaps_body(Node *p_body) const;
	bool overlaps_area(Node *p_area) const;
	void set_audio_bus_override(bool p_enable);
	bool is_overriding_audio_bus() const;
	void set_audio_bus_name(const StringName &p_name);
	StringName get_audio_bus_name() const;
	void set_use_reverb_bus(bool p_enable);
	bool is_using_reverb_bus() const;
	void set_reverb_bus_name(const StringName &p_name);
	StringName get_reverb_bus_name() const;
	void set_reverb_amount(float p_amount);
	float get_reverb_amount() const;
	void set_reverb_uniformity(float p_amount);
	float get_reverb_uniformity() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		CollisionObject3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Area3D::SpaceOverride);

