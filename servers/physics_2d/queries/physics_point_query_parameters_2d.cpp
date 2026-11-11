/**************************************************************************/
/*  physics_point_query_parameters_2d.cpp                                 */
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

#include "physics_point_query_parameters_2d.h"

#include "core/object/class_db.h"
#include "core/variant/typed_array.h"

void PhysicsPointQueryParameters2D::set_exclude(const TypedArray<RID> &p_exclude) {
	parameters.exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++) {
		parameters.exclude.insert(p_exclude[i]);
	}
}

TypedArray<RID> PhysicsPointQueryParameters2D::get_exclude() const {
	TypedArray<RID> ret;
	ret.resize(parameters.exclude.size());
	int idx = 0;
	for (const RID &E : parameters.exclude) {
		ret[idx++] = E;
	}
	return ret;
}

void PhysicsPointQueryParameters2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_position", "position"), &PhysicsPointQueryParameters2D::set_position);
	ClassDB::bind_method(D_METHOD("get_position"), &PhysicsPointQueryParameters2D::get_position);

	ClassDB::bind_method(D_METHOD("set_canvas_instance_id", "canvas_instance_id"), &PhysicsPointQueryParameters2D::set_canvas_instance_id);
	ClassDB::bind_method(D_METHOD("get_canvas_instance_id"), &PhysicsPointQueryParameters2D::get_canvas_instance_id);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &PhysicsPointQueryParameters2D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PhysicsPointQueryParameters2D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &PhysicsPointQueryParameters2D::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &PhysicsPointQueryParameters2D::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &PhysicsPointQueryParameters2D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &PhysicsPointQueryParameters2D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &PhysicsPointQueryParameters2D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &PhysicsPointQueryParameters2D::is_collide_with_areas_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "position"), "set_position", "get_position");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "canvas_instance_id", PROPERTY_HINT_OBJECT_ID), "set_canvas_instance_id", "get_canvas_instance_id");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_ARRAY_TYPE, "RID"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
}
