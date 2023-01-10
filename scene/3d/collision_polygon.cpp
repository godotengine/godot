/**************************************************************************/
/*  collision_polygon.cpp                                                 */
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

#include "collision_polygon.h"

#include "collision_object.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"

void CollisionPolygon::_build_polygon() {
	if (!parent) {
		return;
	}

	parent->shape_owner_clear_shapes(owner_id);

	if (polygon.size() == 0) {
		return;
	}

	Vector<Vector<Vector2>> decomp = Geometry::decompose_polygon_in_convex(polygon);
	if (decomp.size() == 0) {
		return;
	}

	//here comes the sun, lalalala
	//decompose concave into multiple convex polygons and add them

	for (int i = 0; i < decomp.size(); i++) {
		Ref<ConvexPolygonShape> convex = memnew(ConvexPolygonShape);
		PoolVector<Vector3> cp;
		int cs = decomp[i].size();
		cp.resize(cs * 2);
		{
			PoolVector<Vector3>::Write w = cp.write();
			int idx = 0;
			for (int j = 0; j < cs; j++) {
				Vector2 d = decomp[i][j];
				w[idx++] = Vector3(d.x, d.y, depth * 0.5);
				w[idx++] = Vector3(d.x, d.y, -depth * 0.5);
			}
		}

		convex->set_points(cp);
		convex->set_margin(margin);
		parent->shape_owner_add_shape(owner_id, convex);
		parent->shape_owner_set_disabled(owner_id, disabled);
	}
}

void CollisionPolygon::_update_in_shape_owner(bool p_xform_only) {
	parent->shape_owner_set_transform(owner_id, get_transform());
	if (p_xform_only) {
		return;
	}
	parent->shape_owner_set_disabled(owner_id, disabled);
}

void CollisionPolygon::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PARENTED: {
			parent = Object::cast_to<CollisionObject>(get_parent());
			if (parent) {
				owner_id = parent->create_shape_owner(this);
				_build_polygon();
				_update_in_shape_owner();
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (parent) {
				_update_in_shape_owner();
			}

		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			if (parent) {
				_update_in_shape_owner(true);
			}

		} break;
		case NOTIFICATION_UNPARENTED: {
			if (parent) {
				parent->remove_shape_owner(owner_id);
			}
			owner_id = 0;
			parent = nullptr;
		} break;
	}
}

void CollisionPolygon::set_polygon(const Vector<Point2> &p_polygon) {
	polygon = p_polygon;
	if (parent) {
		_build_polygon();
	}
	update_configuration_warning();
	update_gizmo();
}

Vector<Point2> CollisionPolygon::get_polygon() const {
	return polygon;
}

AABB CollisionPolygon::get_item_rect() const {
	return aabb;
}

void CollisionPolygon::set_depth(float p_depth) {
	depth = p_depth;
	_build_polygon();
	update_gizmo();
}

float CollisionPolygon::get_depth() const {
	return depth;
}

void CollisionPolygon::set_disabled(bool p_disabled) {
	disabled = p_disabled;
	update_gizmo();

	if (parent) {
		parent->shape_owner_set_disabled(owner_id, p_disabled);
	}
}

bool CollisionPolygon::is_disabled() const {
	return disabled;
}

real_t CollisionPolygon::get_margin() const {
	return margin;
}

void CollisionPolygon::set_margin(real_t p_margin) {
	margin = p_margin;
	if (parent) {
		_build_polygon();
	}
}

String CollisionPolygon::get_configuration_warning() const {
	String warning = Spatial::get_configuration_warning();
	if (!Object::cast_to<CollisionObject>(get_parent())) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("CollisionPolygon only serves to provide a collision shape to a CollisionObject derived node. Please only use it as a child of Area, StaticBody, RigidBody, KinematicBody, etc. to give them a shape.");
	}

	if (polygon.empty()) {
		if (warning != String()) {
			warning += "\n\n";
		}
		warning += TTR("An empty CollisionPolygon has no effect on collision.");
	}

	return warning;
}

bool CollisionPolygon::_is_editable_3d_polygon() const {
	return true;
}
void CollisionPolygon::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CollisionPolygon::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CollisionPolygon::get_depth);

	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CollisionPolygon::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CollisionPolygon::get_polygon);

	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &CollisionPolygon::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &CollisionPolygon::is_disabled);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &CollisionPolygon::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &CollisionPolygon::get_margin);

	ClassDB::bind_method(D_METHOD("_is_editable_3d_polygon"), &CollisionPolygon::_is_editable_3d_polygon);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "depth"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::POOL_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "margin", PROPERTY_HINT_RANGE, "0.001,10,0.001"), "set_margin", "get_margin");
}

CollisionPolygon::CollisionPolygon() {
	aabb = AABB(Vector3(-1, -1, -1), Vector3(2, 2, 2));
	depth = 1.0;
	set_notify_local_transform(true);
	parent = nullptr;
	owner_id = 0;
	disabled = false;
}
