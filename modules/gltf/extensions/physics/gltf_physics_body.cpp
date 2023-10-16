/**************************************************************************/
/*  gltf_physics_body.cpp                                                 */
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

#include "gltf_physics_body.h"

#include "scene/3d/area_3d.h"
#include "scene/3d/vehicle_body_3d.h"

void GLTFPhysicsBody::_bind_methods() {
	ClassDB::bind_static_method("GLTFPhysicsBody", D_METHOD("from_node", "body_node"), &GLTFPhysicsBody::from_node);
	ClassDB::bind_method(D_METHOD("to_node"), &GLTFPhysicsBody::to_node);

	ClassDB::bind_static_method("GLTFPhysicsBody", D_METHOD("from_dictionary", "dictionary"), &GLTFPhysicsBody::from_dictionary);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GLTFPhysicsBody::to_dictionary);

	ClassDB::bind_method(D_METHOD("get_body_type"), &GLTFPhysicsBody::get_body_type);
	ClassDB::bind_method(D_METHOD("set_body_type", "body_type"), &GLTFPhysicsBody::set_body_type);
	ClassDB::bind_method(D_METHOD("get_mass"), &GLTFPhysicsBody::get_mass);
	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &GLTFPhysicsBody::set_mass);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &GLTFPhysicsBody::get_linear_velocity);
	ClassDB::bind_method(D_METHOD("set_linear_velocity", "linear_velocity"), &GLTFPhysicsBody::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &GLTFPhysicsBody::get_angular_velocity);
	ClassDB::bind_method(D_METHOD("set_angular_velocity", "angular_velocity"), &GLTFPhysicsBody::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_center_of_mass"), &GLTFPhysicsBody::get_center_of_mass);
	ClassDB::bind_method(D_METHOD("set_center_of_mass", "center_of_mass"), &GLTFPhysicsBody::set_center_of_mass);
	ClassDB::bind_method(D_METHOD("get_inertia_tensor"), &GLTFPhysicsBody::get_inertia_tensor);
	ClassDB::bind_method(D_METHOD("set_inertia_tensor", "inertia_tensor"), &GLTFPhysicsBody::set_inertia_tensor);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "body_type"), "set_body_type", "get_body_type");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_of_mass"), "set_center_of_mass", "get_center_of_mass");
	ADD_PROPERTY(PropertyInfo(Variant::BASIS, "inertia_tensor"), "set_inertia_tensor", "get_inertia_tensor");
}

String GLTFPhysicsBody::get_body_type() const {
	return body_type;
}

void GLTFPhysicsBody::set_body_type(String p_body_type) {
	body_type = p_body_type;
}

real_t GLTFPhysicsBody::get_mass() const {
	return mass;
}

void GLTFPhysicsBody::set_mass(real_t p_mass) {
	mass = p_mass;
}

Vector3 GLTFPhysicsBody::get_linear_velocity() const {
	return linear_velocity;
}

void GLTFPhysicsBody::set_linear_velocity(Vector3 p_linear_velocity) {
	linear_velocity = p_linear_velocity;
}

Vector3 GLTFPhysicsBody::get_angular_velocity() const {
	return angular_velocity;
}

void GLTFPhysicsBody::set_angular_velocity(Vector3 p_angular_velocity) {
	angular_velocity = p_angular_velocity;
}

Vector3 GLTFPhysicsBody::get_center_of_mass() const {
	return center_of_mass;
}

void GLTFPhysicsBody::set_center_of_mass(const Vector3 &p_center_of_mass) {
	center_of_mass = p_center_of_mass;
}

Basis GLTFPhysicsBody::get_inertia_tensor() const {
	return inertia_tensor;
}

void GLTFPhysicsBody::set_inertia_tensor(Basis p_inertia_tensor) {
	inertia_tensor = p_inertia_tensor;
}

Ref<GLTFPhysicsBody> GLTFPhysicsBody::from_node(const CollisionObject3D *p_body_node) {
	Ref<GLTFPhysicsBody> physics_body;
	physics_body.instantiate();
	ERR_FAIL_NULL_V_MSG(p_body_node, physics_body, "Tried to create a GLTFPhysicsBody from a CollisionObject3D node, but the given node was null.");
	if (cast_to<CharacterBody3D>(p_body_node)) {
		physics_body->body_type = "character";
	} else if (cast_to<AnimatableBody3D>(p_body_node)) {
		physics_body->body_type = "kinematic";
	} else if (cast_to<RigidBody3D>(p_body_node)) {
		const RigidBody3D *body = cast_to<const RigidBody3D>(p_body_node);
		physics_body->mass = body->get_mass();
		physics_body->linear_velocity = body->get_linear_velocity();
		physics_body->angular_velocity = body->get_angular_velocity();
		physics_body->center_of_mass = body->get_center_of_mass();
		Vector3 inertia_diagonal = body->get_inertia();
		physics_body->inertia_tensor = Basis(inertia_diagonal.x, 0, 0, 0, inertia_diagonal.y, 0, 0, 0, inertia_diagonal.z);
		if (body->get_center_of_mass() != Vector3()) {
			WARN_PRINT("GLTFPhysicsBody: This rigid body has a center of mass offset from the origin, which will be ignored when exporting to GLTF.");
		}
		if (cast_to<VehicleBody3D>(p_body_node)) {
			physics_body->body_type = "vehicle";
		} else {
			physics_body->body_type = "rigid";
		}
	} else if (cast_to<StaticBody3D>(p_body_node)) {
		physics_body->body_type = "static";
	} else if (cast_to<Area3D>(p_body_node)) {
		physics_body->body_type = "trigger";
	}
	return physics_body;
}

CollisionObject3D *GLTFPhysicsBody::to_node() const {
	if (body_type == "character") {
		CharacterBody3D *body = memnew(CharacterBody3D);
		return body;
	}
	if (body_type == "kinematic") {
		AnimatableBody3D *body = memnew(AnimatableBody3D);
		return body;
	}
	if (body_type == "vehicle") {
		VehicleBody3D *body = memnew(VehicleBody3D);
		body->set_mass(mass);
		body->set_linear_velocity(linear_velocity);
		body->set_angular_velocity(angular_velocity);
		body->set_inertia(inertia_tensor.get_main_diagonal());
		body->set_center_of_mass_mode(RigidBody3D::CENTER_OF_MASS_MODE_CUSTOM);
		body->set_center_of_mass(center_of_mass);
		return body;
	}
	if (body_type == "rigid") {
		RigidBody3D *body = memnew(RigidBody3D);
		body->set_mass(mass);
		body->set_linear_velocity(linear_velocity);
		body->set_angular_velocity(angular_velocity);
		body->set_inertia(inertia_tensor.get_main_diagonal());
		body->set_center_of_mass_mode(RigidBody3D::CENTER_OF_MASS_MODE_CUSTOM);
		body->set_center_of_mass(center_of_mass);
		return body;
	}
	if (body_type == "static") {
		StaticBody3D *body = memnew(StaticBody3D);
		return body;
	}
	if (body_type == "trigger") {
		Area3D *body = memnew(Area3D);
		return body;
	}
	ERR_FAIL_V_MSG(nullptr, "Error converting GLTFPhysicsBody to a node: Body type '" + body_type + "' is unknown.");
}

Ref<GLTFPhysicsBody> GLTFPhysicsBody::from_dictionary(const Dictionary p_dictionary) {
	Ref<GLTFPhysicsBody> physics_body;
	physics_body.instantiate();
	ERR_FAIL_COND_V_MSG(!p_dictionary.has("type"), physics_body, "Failed to parse GLTF physics body, missing required field 'type'.");
	const String &body_type = p_dictionary["type"];
	physics_body->body_type = body_type;

	if (p_dictionary.has("mass")) {
		physics_body->mass = p_dictionary["mass"];
	}
	if (p_dictionary.has("linearVelocity")) {
		const Array &arr = p_dictionary["linearVelocity"];
		if (arr.size() == 3) {
			physics_body->set_linear_velocity(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing GLTF physics body: The linear velocity vector must have exactly 3 numbers.");
		}
	}
	if (p_dictionary.has("angularVelocity")) {
		const Array &arr = p_dictionary["angularVelocity"];
		if (arr.size() == 3) {
			physics_body->set_angular_velocity(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing GLTF physics body: The angular velocity vector must have exactly 3 numbers.");
		}
	}
	if (p_dictionary.has("centerOfMass")) {
		const Array &arr = p_dictionary["centerOfMass"];
		if (arr.size() == 3) {
			physics_body->set_center_of_mass(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing GLTF physics body: The center of mass vector must have exactly 3 numbers.");
		}
	}
	if (p_dictionary.has("inertiaTensor")) {
		const Array &arr = p_dictionary["inertiaTensor"];
		if (arr.size() == 9) {
			// Only use the diagonal elements of the inertia tensor matrix (principal axes).
			physics_body->set_inertia_tensor(Basis(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7], arr[8]));
		} else {
			ERR_PRINT("Error parsing GLTF physics body: The inertia tensor must be a 3x3 matrix (9 number array).");
		}
	}
	if (body_type != "character" && body_type != "kinematic" && body_type != "rigid" && body_type != "static" && body_type != "trigger" && body_type != "vehicle") {
		ERR_PRINT("Error parsing GLTF physics body: Body type '" + body_type + "' is unknown.");
	}
	return physics_body;
}

Dictionary GLTFPhysicsBody::to_dictionary() const {
	Dictionary d;
	d["type"] = body_type;
	if (mass != 1.0) {
		d["mass"] = mass;
	}
	if (linear_velocity != Vector3()) {
		Array velocity_array;
		velocity_array.resize(3);
		velocity_array[0] = linear_velocity.x;
		velocity_array[1] = linear_velocity.y;
		velocity_array[2] = linear_velocity.z;
		d["linearVelocity"] = velocity_array;
	}
	if (angular_velocity != Vector3()) {
		Array velocity_array;
		velocity_array.resize(3);
		velocity_array[0] = angular_velocity.x;
		velocity_array[1] = angular_velocity.y;
		velocity_array[2] = angular_velocity.z;
		d["angularVelocity"] = velocity_array;
	}
	if (center_of_mass != Vector3()) {
		Array center_of_mass_array;
		center_of_mass_array.resize(3);
		center_of_mass_array[0] = center_of_mass.x;
		center_of_mass_array[1] = center_of_mass.y;
		center_of_mass_array[2] = center_of_mass.z;
		d["centerOfMass"] = center_of_mass_array;
	}
	if (inertia_tensor != Basis(0, 0, 0, 0, 0, 0, 0, 0, 0)) {
		Array inertia_array;
		inertia_array.resize(9);
		inertia_array.fill(0.0);
		inertia_array[0] = inertia_tensor[0][0];
		inertia_array[1] = inertia_tensor[0][1];
		inertia_array[2] = inertia_tensor[0][2];
		inertia_array[3] = inertia_tensor[1][0];
		inertia_array[4] = inertia_tensor[1][1];
		inertia_array[5] = inertia_tensor[1][2];
		inertia_array[6] = inertia_tensor[2][0];
		inertia_array[7] = inertia_tensor[2][1];
		inertia_array[8] = inertia_tensor[2][2];
		d["inertiaTensor"] = inertia_array;
	}
	return d;
}
