/**************************************************************************/
/*  gltf_physics_body.hpp                                                 */
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

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/resource.hpp>
#include <godot_cpp/variant/basis.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CollisionObject3D;

class GLTFPhysicsBody : public Resource {
	GDEXTENSION_CLASS(GLTFPhysicsBody, Resource)

public:
	static Ref<GLTFPhysicsBody> from_node(CollisionObject3D *p_body_node);
	CollisionObject3D *to_node() const;
	static Ref<GLTFPhysicsBody> from_dictionary(const Dictionary &p_dictionary);
	Dictionary to_dictionary() const;
	String get_body_type() const;
	void set_body_type(const String &p_body_type);
	float get_mass() const;
	void set_mass(float p_mass);
	Vector3 get_linear_velocity() const;
	void set_linear_velocity(const Vector3 &p_linear_velocity);
	Vector3 get_angular_velocity() const;
	void set_angular_velocity(const Vector3 &p_angular_velocity);
	Vector3 get_center_of_mass() const;
	void set_center_of_mass(const Vector3 &p_center_of_mass);
	Vector3 get_inertia_diagonal() const;
	void set_inertia_diagonal(const Vector3 &p_inertia_diagonal);
	Quaternion get_inertia_orientation() const;
	void set_inertia_orientation(const Quaternion &p_inertia_orientation);
	Basis get_inertia_tensor() const;
	void set_inertia_tensor(const Basis &p_inertia_tensor);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Resource::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

