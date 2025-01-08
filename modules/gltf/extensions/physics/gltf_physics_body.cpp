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

#include "scene/3d/physics/animatable_body_3d.h"
#include "scene/3d/physics/area_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "scene/3d/physics/static_body_3d.h"
#include "scene/3d/physics/vehicle_body_3d.h"

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
	ClassDB::bind_method(D_METHOD("get_inertia_diagonal"), &GLTFPhysicsBody::get_inertia_diagonal);
	ClassDB::bind_method(D_METHOD("set_inertia_diagonal", "inertia_diagonal"), &GLTFPhysicsBody::set_inertia_diagonal);
	ClassDB::bind_method(D_METHOD("get_inertia_orientation"), &GLTFPhysicsBody::get_inertia_orientation);
	ClassDB::bind_method(D_METHOD("set_inertia_orientation", "inertia_orientation"), &GLTFPhysicsBody::set_inertia_orientation);
#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("get_inertia_tensor"), &GLTFPhysicsBody::get_inertia_tensor);
	ClassDB::bind_method(D_METHOD("set_inertia_tensor", "inertia_tensor"), &GLTFPhysicsBody::set_inertia_tensor);
#endif // DISABLE_DEPRECATED

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "body_type"), "set_body_type", "get_body_type");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "mass"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_of_mass"), "set_center_of_mass", "get_center_of_mass");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "inertia_diagonal"), "set_inertia_diagonal", "get_inertia_diagonal");
	ADD_PROPERTY(PropertyInfo(Variant::QUATERNION, "inertia_orientation"), "set_inertia_orientation", "get_inertia_orientation");
#ifndef DISABLE_DEPRECATED
	ADD_PROPERTY(PropertyInfo(Variant::BASIS, "inertia_tensor"), "set_inertia_tensor", "get_inertia_tensor");
#endif // DISABLE_DEPRECATED
}

String GLTFPhysicsBody::get_body_type() const {
	switch (body_type) {
		case PhysicsBodyType::STATIC:
			return "static";
		case PhysicsBodyType::ANIMATABLE:
			return "animatable";
		case PhysicsBodyType::CHARACTER:
			return "character";
		case PhysicsBodyType::RIGID:
			return "rigid";
		case PhysicsBodyType::VEHICLE:
			return "vehicle";
		case PhysicsBodyType::TRIGGER:
			return "trigger";
	}
	// Unreachable, the switch cases handle all values the enum can take.
	// Omitting this works on Clang but not GCC or MSVC. If reached, it's UB.
	return "rigid";
}

void GLTFPhysicsBody::set_body_type(String p_body_type) {
	if (p_body_type == "static") {
		body_type = PhysicsBodyType::STATIC;
	} else if (p_body_type == "animatable") {
		body_type = PhysicsBodyType::ANIMATABLE;
	} else if (p_body_type == "character") {
		body_type = PhysicsBodyType::CHARACTER;
	} else if (p_body_type == "rigid") {
		body_type = PhysicsBodyType::RIGID;
	} else if (p_body_type == "vehicle") {
		body_type = PhysicsBodyType::VEHICLE;
	} else if (p_body_type == "trigger") {
		body_type = PhysicsBodyType::TRIGGER;
	} else {
		ERR_PRINT("Error setting glTF physics body type: The body type must be one of \"static\", \"animatable\", \"character\", \"rigid\", \"vehicle\", or \"trigger\".");
	}
}

GLTFPhysicsBody::PhysicsBodyType GLTFPhysicsBody::get_physics_body_type() const {
	return body_type;
}

void GLTFPhysicsBody::set_physics_body_type(PhysicsBodyType p_body_type) {
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

Vector3 GLTFPhysicsBody::get_inertia_diagonal() const {
	return inertia_diagonal;
}

void GLTFPhysicsBody::set_inertia_diagonal(const Vector3 &p_inertia_diagonal) {
	inertia_diagonal = p_inertia_diagonal;
}

Quaternion GLTFPhysicsBody::get_inertia_orientation() const {
	return inertia_orientation;
}

void GLTFPhysicsBody::set_inertia_orientation(const Quaternion &p_inertia_orientation) {
	inertia_orientation = p_inertia_orientation;
}

#ifndef DISABLE_DEPRECATED
Basis GLTFPhysicsBody::get_inertia_tensor() const {
	return Basis::from_scale(inertia_diagonal);
}

void GLTFPhysicsBody::set_inertia_tensor(Basis p_inertia_tensor) {
	inertia_diagonal = p_inertia_tensor.get_main_diagonal();
}
#endif // DISABLE_DEPRECATED

Ref<GLTFPhysicsBody> GLTFPhysicsBody::from_node(const CollisionObject3D *p_body_node) {
	Ref<GLTFPhysicsBody> physics_body;
	physics_body.instantiate();
	ERR_FAIL_NULL_V_MSG(p_body_node, physics_body, "Tried to create a GLTFPhysicsBody from a CollisionObject3D node, but the given node was null.");
	if (cast_to<CharacterBody3D>(p_body_node)) {
		physics_body->body_type = PhysicsBodyType::CHARACTER;
	} else if (cast_to<AnimatableBody3D>(p_body_node)) {
		physics_body->body_type = PhysicsBodyType::ANIMATABLE;
	} else if (cast_to<RigidBody3D>(p_body_node)) {
		const RigidBody3D *body = cast_to<const RigidBody3D>(p_body_node);
		physics_body->mass = body->get_mass();
		physics_body->linear_velocity = body->get_linear_velocity();
		physics_body->angular_velocity = body->get_angular_velocity();
		physics_body->center_of_mass = body->get_center_of_mass();
		physics_body->inertia_diagonal = body->get_inertia();
		if (cast_to<VehicleBody3D>(p_body_node)) {
			physics_body->body_type = PhysicsBodyType::VEHICLE;
		} else {
			physics_body->body_type = PhysicsBodyType::RIGID;
		}
	} else if (cast_to<StaticBody3D>(p_body_node)) {
		physics_body->body_type = PhysicsBodyType::STATIC;
	} else if (cast_to<Area3D>(p_body_node)) {
		physics_body->body_type = PhysicsBodyType::TRIGGER;
	}
	return physics_body;
}

CollisionObject3D *GLTFPhysicsBody::to_node() const {
	switch (body_type) {
		case PhysicsBodyType::CHARACTER: {
			CharacterBody3D *body = memnew(CharacterBody3D);
			return body;
		}
		case PhysicsBodyType::ANIMATABLE: {
			AnimatableBody3D *body = memnew(AnimatableBody3D);
			return body;
		}
		case PhysicsBodyType::VEHICLE: {
			VehicleBody3D *body = memnew(VehicleBody3D);
			body->set_mass(mass);
			body->set_linear_velocity(linear_velocity);
			body->set_angular_velocity(angular_velocity);
			body->set_inertia(inertia_diagonal);
			body->set_center_of_mass_mode(RigidBody3D::CENTER_OF_MASS_MODE_CUSTOM);
			body->set_center_of_mass(center_of_mass);
			return body;
		}
		case PhysicsBodyType::RIGID: {
			RigidBody3D *body = memnew(RigidBody3D);
			body->set_mass(mass);
			body->set_linear_velocity(linear_velocity);
			body->set_angular_velocity(angular_velocity);
			body->set_inertia(inertia_diagonal);
			body->set_center_of_mass_mode(RigidBody3D::CENTER_OF_MASS_MODE_CUSTOM);
			body->set_center_of_mass(center_of_mass);
			return body;
		}
		case PhysicsBodyType::STATIC: {
			StaticBody3D *body = memnew(StaticBody3D);
			return body;
		}
		case PhysicsBodyType::TRIGGER: {
			Area3D *body = memnew(Area3D);
			return body;
		}
	}
	// Unreachable, the switch cases handle all values the enum can take.
	// Omitting this works on Clang but not GCC or MSVC. If reached, it's UB.
	return nullptr;
}

Ref<GLTFPhysicsBody> GLTFPhysicsBody::from_dictionary(const Dictionary p_dictionary) {
	Ref<GLTFPhysicsBody> physics_body;
	physics_body.instantiate();
	Dictionary motion;
	if (p_dictionary.has("motion")) {
		motion = p_dictionary["motion"];
#ifndef DISABLE_DEPRECATED
	} else {
		motion = p_dictionary;
#endif // DISABLE_DEPRECATED
	}
	if (motion.has("type")) {
		// Read the body type. This representation sits between glTF's and Godot's physics nodes.
		// While we may only read "static", "kinematic", or "dynamic" from a valid glTF file, we
		// want to allow another extension to override this to another Godot node type mid-import.
		// For example, a vehicle extension may want to override the body type to "vehicle"
		// so Godot generates a VehicleBody3D node. Therefore we distinguish by importing
		// "dynamic" as "rigid", and "kinematic" as "animatable", in the GLTFPhysicsBody code.
		String body_type_string = motion["type"];
		if (body_type_string == "static") {
			physics_body->body_type = PhysicsBodyType::STATIC;
		} else if (body_type_string == "kinematic") {
			physics_body->body_type = PhysicsBodyType::ANIMATABLE;
		} else if (body_type_string == "dynamic") {
			physics_body->body_type = PhysicsBodyType::RIGID;
#ifndef DISABLE_DEPRECATED
		} else if (body_type_string == "character") {
			physics_body->body_type = PhysicsBodyType::CHARACTER;
		} else if (body_type_string == "rigid") {
			physics_body->body_type = PhysicsBodyType::RIGID;
		} else if (body_type_string == "vehicle") {
			physics_body->body_type = PhysicsBodyType::VEHICLE;
		} else if (body_type_string == "trigger") {
			physics_body->body_type = PhysicsBodyType::TRIGGER;
#endif // DISABLE_DEPRECATED
		} else {
			ERR_PRINT("Error parsing glTF physics body: The body type in the glTF file \"" + body_type_string + "\" was not recognized.");
		}
	}
	if (motion.has("mass")) {
		physics_body->mass = motion["mass"];
	}
	if (motion.has("linearVelocity")) {
		const Array &arr = motion["linearVelocity"];
		if (arr.size() == 3) {
			physics_body->set_linear_velocity(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing glTF physics body: The linear velocity vector must have exactly 3 numbers.");
		}
	}
	if (motion.has("angularVelocity")) {
		const Array &arr = motion["angularVelocity"];
		if (arr.size() == 3) {
			physics_body->set_angular_velocity(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing glTF physics body: The angular velocity vector must have exactly 3 numbers.");
		}
	}
	if (motion.has("centerOfMass")) {
		const Array &arr = motion["centerOfMass"];
		if (arr.size() == 3) {
			physics_body->set_center_of_mass(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing glTF physics body: The center of mass vector must have exactly 3 numbers.");
		}
	}
	if (motion.has("inertiaDiagonal")) {
		const Array &arr = motion["inertiaDiagonal"];
		if (arr.size() == 3) {
			physics_body->set_inertia_diagonal(Vector3(arr[0], arr[1], arr[2]));
		} else {
			ERR_PRINT("Error parsing glTF physics body: The inertia diagonal vector must have exactly 3 numbers.");
		}
	}
	if (motion.has("inertiaOrientation")) {
		const Array &arr = motion["inertiaOrientation"];
		if (arr.size() == 4) {
			physics_body->set_inertia_orientation(Quaternion(arr[0], arr[1], arr[2], arr[3]));
		} else {
			ERR_PRINT("Error parsing glTF physics body: The inertia orientation quaternion must have exactly 4 numbers.");
		}
	}
	return physics_body;
}

Dictionary GLTFPhysicsBody::to_dictionary() const {
	Dictionary ret;
	if (body_type == PhysicsBodyType::TRIGGER) {
		// The equivalent of a Godot Area3D node in glTF is a node that
		// defines that it is a trigger, but does not have a shape.
		Dictionary trigger;
		ret["trigger"] = trigger;
		return ret;
	}
	// All non-trigger body types are defined using the motion property.
	Dictionary motion;
	// When stored in memory, the body type can correspond to a Godot
	// node type. However, when exporting to glTF, we need to squash
	// this down to one of "static", "kinematic", or "dynamic".
	if (body_type == PhysicsBodyType::STATIC) {
		motion["type"] = "static";
	} else if (body_type == PhysicsBodyType::ANIMATABLE || body_type == PhysicsBodyType::CHARACTER) {
		motion["type"] = "kinematic";
	} else {
		motion["type"] = "dynamic";
	}
	if (mass != 1.0) {
		motion["mass"] = mass;
	}
	if (linear_velocity != Vector3()) {
		Array velocity_array;
		velocity_array.resize(3);
		velocity_array[0] = linear_velocity.x;
		velocity_array[1] = linear_velocity.y;
		velocity_array[2] = linear_velocity.z;
		motion["linearVelocity"] = velocity_array;
	}
	if (angular_velocity != Vector3()) {
		Array velocity_array;
		velocity_array.resize(3);
		velocity_array[0] = angular_velocity.x;
		velocity_array[1] = angular_velocity.y;
		velocity_array[2] = angular_velocity.z;
		motion["angularVelocity"] = velocity_array;
	}
	if (center_of_mass != Vector3()) {
		Array center_of_mass_array;
		center_of_mass_array.resize(3);
		center_of_mass_array[0] = center_of_mass.x;
		center_of_mass_array[1] = center_of_mass.y;
		center_of_mass_array[2] = center_of_mass.z;
		motion["centerOfMass"] = center_of_mass_array;
	}
	if (inertia_diagonal != Vector3()) {
		Array inertia_array;
		inertia_array.resize(3);
		inertia_array[0] = inertia_diagonal[0];
		inertia_array[1] = inertia_diagonal[1];
		inertia_array[2] = inertia_diagonal[2];
		motion["inertiaDiagonal"] = inertia_array;
	}
	if (inertia_orientation != Quaternion()) {
		Array inertia_array;
		inertia_array.resize(4);
		inertia_array[0] = inertia_orientation[0];
		inertia_array[1] = inertia_orientation[1];
		inertia_array[2] = inertia_orientation[2];
		inertia_array[3] = inertia_orientation[3];
		motion["inertiaDiagonal"] = inertia_array;
	}
	ret["motion"] = motion;
	return ret;
}
