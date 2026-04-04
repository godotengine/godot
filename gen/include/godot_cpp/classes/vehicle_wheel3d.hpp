/**************************************************************************/
/*  vehicle_wheel3d.hpp                                                   */
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
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class VehicleWheel3D : public Node3D {
	GDEXTENSION_CLASS(VehicleWheel3D, Node3D)

public:
	void set_radius(float p_length);
	float get_radius() const;
	void set_suspension_rest_length(float p_length);
	float get_suspension_rest_length() const;
	void set_suspension_travel(float p_length);
	float get_suspension_travel() const;
	void set_suspension_stiffness(float p_length);
	float get_suspension_stiffness() const;
	void set_suspension_max_force(float p_length);
	float get_suspension_max_force() const;
	void set_damping_compression(float p_length);
	float get_damping_compression() const;
	void set_damping_relaxation(float p_length);
	float get_damping_relaxation() const;
	void set_use_as_traction(bool p_enable);
	bool is_used_as_traction() const;
	void set_use_as_steering(bool p_enable);
	bool is_used_as_steering() const;
	void set_friction_slip(float p_length);
	float get_friction_slip() const;
	bool is_in_contact() const;
	Node3D *get_contact_body() const;
	Vector3 get_contact_point() const;
	Vector3 get_contact_normal() const;
	void set_roll_influence(float p_roll_influence);
	float get_roll_influence() const;
	float get_skidinfo() const;
	float get_rpm() const;
	void set_engine_force(float p_engine_force);
	float get_engine_force() const;
	void set_brake(float p_brake);
	float get_brake() const;
	void set_steering(float p_steering);
	float get_steering() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

