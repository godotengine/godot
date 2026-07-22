/**************************************************************************/
/*  physics_shape_query_parameters_3d.cpp                                 */
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

#include "physics_shape_query_parameters_3d.h"

#include "core/io/resource.h"
#include "core/object/class_db.h"
#include "core/variant/typed_array.h"

Ref<Resource> PhysicsShapeQueryParameters3D::get_shape() const {
	return shape_ref;
}

void PhysicsShapeQueryParameters3D::set_shape(const Ref<Resource> &p_shape_ref) {
	ERR_FAIL_COND(p_shape_ref.is_null());
	shape_ref = p_shape_ref;
	parameters.shape_rid = p_shape_ref->get_rid();
}

void PhysicsShapeQueryParameters3D::set_shape_rid(const RID &p_shape) {
	if (parameters.shape_rid != p_shape) {
		shape_ref = Ref<Resource>();
		parameters.shape_rid = p_shape;
	}
}

void PhysicsShapeQueryParameters3D::set_exclude(const TypedArray<RID> &p_exclude) {
	parameters.exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++) {
		parameters.exclude.insert(p_exclude[i]);
	}
}

TypedArray<RID> PhysicsShapeQueryParameters3D::get_exclude() const {
	TypedArray<RID> ret;
	ret.resize(parameters.exclude.size());
	int idx = 0;
	for (const RID &E : parameters.exclude) {
		ret[idx++] = E;
	}
	return ret;
}

void PhysicsShapeQueryParameters3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &PhysicsShapeQueryParameters3D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &PhysicsShapeQueryParameters3D::get_shape);

	ClassDB::bind_method(D_METHOD("set_shape_rid", "shape"), &PhysicsShapeQueryParameters3D::set_shape_rid);
	ClassDB::bind_method(D_METHOD("get_shape_rid"), &PhysicsShapeQueryParameters3D::get_shape_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsShapeQueryParameters3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsShapeQueryParameters3D::get_transform);

	ClassDB::bind_method(D_METHOD("set_motion", "motion"), &PhysicsShapeQueryParameters3D::set_motion);
	ClassDB::bind_method(D_METHOD("get_motion"), &PhysicsShapeQueryParameters3D::get_motion);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &PhysicsShapeQueryParameters3D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &PhysicsShapeQueryParameters3D::get_margin);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &PhysicsShapeQueryParameters3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PhysicsShapeQueryParameters3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &PhysicsShapeQueryParameters3D::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &PhysicsShapeQueryParameters3D::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &PhysicsShapeQueryParameters3D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &PhysicsShapeQueryParameters3D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &PhysicsShapeQueryParameters3D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &PhysicsShapeQueryParameters3D::is_collide_with_areas_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_ARRAY_TYPE, "RID"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "motion"), "set_motion", "get_motion");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape3D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "shape_rid"), "set_shape_rid", "get_shape_rid");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
}
