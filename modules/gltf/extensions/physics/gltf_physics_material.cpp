/**************************************************************************/
/*  gltf_physics_material.cpp                                             */
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

#include "gltf_physics_material.h"

void GLTFPhysicsMaterial::_bind_methods() {
	ClassDB::bind_static_method("GLTFPhysicsMaterial", D_METHOD("from_resource", "material"), &GLTFPhysicsMaterial::from_resource);
	ClassDB::bind_method(D_METHOD("to_resource"), &GLTFPhysicsMaterial::to_resource);

	ClassDB::bind_static_method("GLTFPhysicsMaterial", D_METHOD("from_dictionary", "dictionary"), &GLTFPhysicsMaterial::from_dictionary);
	ClassDB::bind_method(D_METHOD("to_dictionary"), &GLTFPhysicsMaterial::to_dictionary);

	ClassDB::bind_method(D_METHOD("get_static_friction"), &GLTFPhysicsMaterial::get_static_friction);
	ClassDB::bind_method(D_METHOD("set_static_friction", "static_friction"), &GLTFPhysicsMaterial::set_static_friction);
	ClassDB::bind_method(D_METHOD("get_dynamic_friction"), &GLTFPhysicsMaterial::get_dynamic_friction);
	ClassDB::bind_method(D_METHOD("set_dynamic_friction", "dynamic_friction"), &GLTFPhysicsMaterial::set_dynamic_friction);
	ClassDB::bind_method(D_METHOD("get_restitution"), &GLTFPhysicsMaterial::get_restitution);
	ClassDB::bind_method(D_METHOD("set_restitution", "restitution"), &GLTFPhysicsMaterial::set_restitution);
	ClassDB::bind_method(D_METHOD("get_friction_combine"), &GLTFPhysicsMaterial::get_friction_combine);
	ClassDB::bind_method(D_METHOD("set_friction_combine", "friction_combine"), &GLTFPhysicsMaterial::set_friction_combine);
	ClassDB::bind_method(D_METHOD("get_restitution_combine"), &GLTFPhysicsMaterial::get_restitution_combine);
	ClassDB::bind_method(D_METHOD("set_restitution_combine", "restitution_combine"), &GLTFPhysicsMaterial::set_restitution_combine);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "static_friction"), "set_static_friction", "get_static_friction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "dynamic_friction"), "set_dynamic_friction", "get_dynamic_friction");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "restitution"), "set_restitution", "get_restitution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "friction_combine"), "set_friction_combine", "get_friction_combine");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "restitution_combine"), "set_restitution_combine", "get_restitution_combine");

	BIND_ENUM_CONSTANT(COMBINE_AVERAGE);
	BIND_ENUM_CONSTANT(COMBINE_MINIMUM);
	BIND_ENUM_CONSTANT(COMBINE_MAXIMUM);
	BIND_ENUM_CONSTANT(COMBINE_MULTIPLY);
}

real_t GLTFPhysicsMaterial::get_static_friction() const {
	return static_friction;
}

void GLTFPhysicsMaterial::set_static_friction(real_t p_static_friction) {
	static_friction = p_static_friction;
}

real_t GLTFPhysicsMaterial::get_dynamic_friction() const {
	return dynamic_friction;
}

void GLTFPhysicsMaterial::set_dynamic_friction(real_t p_dynamic_friction) {
	dynamic_friction = p_dynamic_friction;
}

real_t GLTFPhysicsMaterial::get_restitution() const {
	return restitution;
}

void GLTFPhysicsMaterial::set_restitution(real_t p_restitution) {
	restitution = p_restitution;
}

GLTFPhysicsMaterial::CombineMode GLTFPhysicsMaterial::get_friction_combine() const {
	return friction_combine;
}

void GLTFPhysicsMaterial::set_friction_combine(CombineMode p_friction_combine) {
	friction_combine = p_friction_combine;
}

GLTFPhysicsMaterial::CombineMode GLTFPhysicsMaterial::get_restitution_combine() const {
	return restitution_combine;
}

void GLTFPhysicsMaterial::set_restitution_combine(CombineMode p_restitution_combine) {
	restitution_combine = p_restitution_combine;
}

Ref<GLTFPhysicsMaterial> GLTFPhysicsMaterial::from_resource(const Ref<PhysicsMaterial> &p_material) {
	Ref<GLTFPhysicsMaterial> gltf_material;
	gltf_material.instantiate();
	
	if (p_material.is_null()) {
		return gltf_material;
	}

	// Godot's PhysicsMaterial has:
	// - friction (real_t) with rough boolean flag
	// - bounce (real_t) with absorbent boolean flag
	// We map these to glTF's static/dynamic friction and restitution.
	
	real_t godot_friction = p_material->get_friction();
	gltf_material->set_static_friction(godot_friction);
	gltf_material->set_dynamic_friction(godot_friction);
	
	real_t godot_bounce = p_material->get_bounce();
	gltf_material->set_restitution(godot_bounce);
	
	// Godot doesn't have combine modes, so we use the default "average"
	gltf_material->set_friction_combine(COMBINE_AVERAGE);
	gltf_material->set_restitution_combine(COMBINE_AVERAGE);
	
	return gltf_material;
}

Ref<PhysicsMaterial> GLTFPhysicsMaterial::to_resource() const {
	Ref<PhysicsMaterial> material;
	material.instantiate();
	
	// Use dynamic friction as it's more commonly used
	material->set_friction(dynamic_friction);
	material->set_rough(false);
	
	material->set_bounce(restitution);
	material->set_absorbent(false);
	
	return material;
}

Ref<GLTFPhysicsMaterial> GLTFPhysicsMaterial::from_dictionary(const Dictionary &p_dictionary) {
	Ref<GLTFPhysicsMaterial> gltf_material;
	gltf_material.instantiate();
	
	if (p_dictionary.has("staticFriction")) {
		gltf_material->set_static_friction(p_dictionary["staticFriction"]);
	}
	if (p_dictionary.has("dynamicFriction")) {
		gltf_material->set_dynamic_friction(p_dictionary["dynamicFriction"]);
	}
	if (p_dictionary.has("restitution")) {
		gltf_material->set_restitution(p_dictionary["restitution"]);
	}
	if (p_dictionary.has("frictionCombine")) {
		String combine_str = p_dictionary["frictionCombine"];
		if (combine_str == "average") {
			gltf_material->set_friction_combine(COMBINE_AVERAGE);
		} else if (combine_str == "minimum") {
			gltf_material->set_friction_combine(COMBINE_MINIMUM);
		} else if (combine_str == "maximum") {
			gltf_material->set_friction_combine(COMBINE_MAXIMUM);
		} else if (combine_str == "multiply") {
			gltf_material->set_friction_combine(COMBINE_MULTIPLY);
		}
	}
	if (p_dictionary.has("restitutionCombine")) {
		String combine_str = p_dictionary["restitutionCombine"];
		if (combine_str == "average") {
			gltf_material->set_restitution_combine(COMBINE_AVERAGE);
		} else if (combine_str == "minimum") {
			gltf_material->set_restitution_combine(COMBINE_MINIMUM);
		} else if (combine_str == "maximum") {
			gltf_material->set_restitution_combine(COMBINE_MAXIMUM);
		} else if (combine_str == "multiply") {
			gltf_material->set_restitution_combine(COMBINE_MULTIPLY);
		}
	}
	
	return gltf_material;
}

Dictionary GLTFPhysicsMaterial::to_dictionary() const {
	Dictionary dict;
	
	dict["staticFriction"] = static_friction;
	dict["dynamicFriction"] = dynamic_friction;
	dict["restitution"] = restitution;
	
	String friction_combine_str;
	switch (friction_combine) {
		case COMBINE_AVERAGE:
			friction_combine_str = "average";
			break;
		case COMBINE_MINIMUM:
			friction_combine_str = "minimum";
			break;
		case COMBINE_MAXIMUM:
			friction_combine_str = "maximum";
			break;
		case COMBINE_MULTIPLY:
			friction_combine_str = "multiply";
			break;
	}
	dict["frictionCombine"] = friction_combine_str;
	
	String restitution_combine_str;
	switch (restitution_combine) {
		case COMBINE_AVERAGE:
			restitution_combine_str = "average";
			break;
		case COMBINE_MINIMUM:
			restitution_combine_str = "minimum";
			break;
		case COMBINE_MAXIMUM:
			restitution_combine_str = "maximum";
			break;
		case COMBINE_MULTIPLY:
			restitution_combine_str = "multiply";
			break;
	}
	dict["restitutionCombine"] = restitution_combine_str;
	
	return dict;
}

bool GLTFPhysicsMaterial::operator==(const GLTFPhysicsMaterial &p_other) const {
	return static_friction == p_other.static_friction &&
			dynamic_friction == p_other.dynamic_friction &&
			restitution == p_other.restitution &&
			friction_combine == p_other.friction_combine &&
			restitution_combine == p_other.restitution_combine;
}

bool GLTFPhysicsMaterial::operator!=(const GLTFPhysicsMaterial &p_other) const {
	return !(*this == p_other);
}
