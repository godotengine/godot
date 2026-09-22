/**************************************************************************/
/*  physics_ray_query_parameters_3d.cpp                                   */
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

#include "physics_ray_query_parameters_3d.h"

#include "core/object/class_db.h"
#include "core/variant/typed_array.h"

void PhysicsRayQueryParameters3D::set_exclude(const TypedArray<RID> &p_exclude) {
	parameters.exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++) {
		parameters.exclude.insert(p_exclude[i]);
	}
}

TypedArray<RID> PhysicsRayQueryParameters3D::get_exclude() const {
	TypedArray<RID> ret;
	ret.resize(parameters.exclude.size());
	int idx = 0;
	for (const RID &E : parameters.exclude) {
		ret[idx++] = E;
	}
	return ret;
}

Ref<PhysicsRayQueryParameters3D> PhysicsRayQueryParameters3D::create(Vector3 p_from, Vector3 p_to, uint32_t p_mask, const TypedArray<RID> &p_exclude) {
	Ref<PhysicsRayQueryParameters3D> params;
	params.instantiate();
	params->set_from(p_from);
	params->set_to(p_to);
	params->set_collision_mask(p_mask);
	params->set_exclude(p_exclude);
	return params;
}

void PhysicsRayQueryParameters3D::_bind_methods() {
	ClassDB::bind_static_method("PhysicsRayQueryParameters3D", D_METHOD("create", "from", "to", "collision_mask", "exclude"), &PhysicsRayQueryParameters3D::create, DEFVAL(UINT32_MAX), DEFVAL(TypedArray<RID>()));

	ClassDB::bind_method(D_METHOD("set_from", "from"), &PhysicsRayQueryParameters3D::set_from);
	ClassDB::bind_method(D_METHOD("get_from"), &PhysicsRayQueryParameters3D::get_from);

	ClassDB::bind_method(D_METHOD("set_to", "to"), &PhysicsRayQueryParameters3D::set_to);
	ClassDB::bind_method(D_METHOD("get_to"), &PhysicsRayQueryParameters3D::get_to);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &PhysicsRayQueryParameters3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PhysicsRayQueryParameters3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &PhysicsRayQueryParameters3D::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &PhysicsRayQueryParameters3D::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &PhysicsRayQueryParameters3D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &PhysicsRayQueryParameters3D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &PhysicsRayQueryParameters3D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &PhysicsRayQueryParameters3D::is_collide_with_areas_enabled);

	ClassDB::bind_method(D_METHOD("set_hit_from_inside", "enable"), &PhysicsRayQueryParameters3D::set_hit_from_inside);
	ClassDB::bind_method(D_METHOD("is_hit_from_inside_enabled"), &PhysicsRayQueryParameters3D::is_hit_from_inside_enabled);

	ClassDB::bind_method(D_METHOD("set_hit_back_faces", "enable"), &PhysicsRayQueryParameters3D::set_hit_back_faces);
	ClassDB::bind_method(D_METHOD("is_hit_back_faces_enabled"), &PhysicsRayQueryParameters3D::is_hit_back_faces_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "from"), "set_from", "get_from");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "to"), "set_to", "get_to");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_ARRAY_TYPE, "RID"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hit_from_inside"), "set_hit_from_inside", "is_hit_from_inside_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hit_back_faces"), "set_hit_back_faces", "is_hit_back_faces_enabled");
}
