/*************************************************************************/
/*  collision_polygon_3d.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "collision_polygon_3d.h"

#include "collision_object_3d.h"
#include "core/math/geometry_2d.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/convex_polygon_shape_3d.h"

void CollisionPolygon3D::_build_polygon() {
	if (!parent) {
		return;
	}

	parent->shape_owner_clear_shapes(owner_id);

	if (polygon.size() == 0) {
		return;
	}

	Vector<Vector<Vector2>> decomp = Geometry2D::decompose_polygon_in_convex(polygon);
	if (decomp.size() == 0) {
		return;
	}

	//here comes the sun, lalalala
	//decompose concave into multiple convex polygons and add them

	for (int i = 0; i < decomp.size(); i++) {
		Ref<ConvexPolygonShape3D> convex = memnew(ConvexPolygonShape3D);
		Vector<Vector3> cp;
		int cs = decomp[i].size();
		cp.resize(cs * 2);
		{
			Vector3 *w = cp.ptrw();
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

void CollisionPolygon3D::_update_in_shape_owner(bool p_xform_only) {
	parent->shape_owner_set_transform(owner_id, get_transform());
	if (p_xform_only) {
		return;
	}
	parent->shape_owner_set_disabled(owner_id, disabled);
}

void CollisionPolygon3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PARENTED: {
			parent = Object::cast_to<CollisionObject3D>(get_parent());
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

void CollisionPolygon3D::set_polygon(const Vector<Point2> &p_polygon) {
	polygon = p_polygon;
	if (parent) {
		_build_polygon();
	}
	update_configuration_warnings();
	update_gizmo();
}

Vector<Point2> CollisionPolygon3D::get_polygon() const {
	return polygon;
}

AABB CollisionPolygon3D::get_item_rect() const {
	return aabb;
}

void CollisionPolygon3D::set_depth(real_t p_depth) {
	depth = p_depth;
	_build_polygon();
	update_gizmo();
}

real_t CollisionPolygon3D::get_depth() const {
	return depth;
}

void CollisionPolygon3D::set_disabled(bool p_disabled) {
	disabled = p_disabled;
	update_gizmo();

	if (parent) {
		parent->shape_owner_set_disabled(owner_id, p_disabled);
	}
}

bool CollisionPolygon3D::is_disabled() const {
	return disabled;
}

real_t CollisionPolygon3D::get_margin() const {
	return margin;
}

void CollisionPolygon3D::set_margin(real_t p_margin) {
	margin = p_margin;
	if (parent) {
		_build_polygon();
	}
}

TypedArray<String> CollisionPolygon3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node::get_configuration_warnings();

	if (!Object::cast_to<CollisionObject3D>(get_parent())) {
		warnings.push_back(TTR("CollisionPolygon3D only serves to provide a collision shape to a CollisionObject3D derived node. Please only use it as a child of Area3D, StaticBody3D, RigidBody3D, KinematicBody3D, etc. to give them a shape."));
	}

	if (polygon.is_empty()) {
		warnings.push_back(TTR("An empty CollisionPolygon3D has no effect on collision."));
	}

	return warnings;
}

bool CollisionPolygon3D::_is_editable_3d_polygon() const {
	return true;
}

void CollisionPolygon3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_depth", "depth"), &CollisionPolygon3D::set_depth);
	ClassDB::bind_method(D_METHOD("get_depth"), &CollisionPolygon3D::get_depth);

	ClassDB::bind_method(D_METHOD("set_polygon", "polygon"), &CollisionPolygon3D::set_polygon);
	ClassDB::bind_method(D_METHOD("get_polygon"), &CollisionPolygon3D::get_polygon);

	ClassDB::bind_method(D_METHOD("set_disabled", "disabled"), &CollisionPolygon3D::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &CollisionPolygon3D::is_disabled);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &CollisionPolygon3D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &CollisionPolygon3D::get_margin);

	ClassDB::bind_method(D_METHOD("_is_editable_3d_polygon"), &CollisionPolygon3D::_is_editable_3d_polygon);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "depth"), "set_depth", "get_depth");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
	ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "polygon"), "set_polygon", "get_polygon");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0.001,10,0.001"), "set_margin", "get_margin");
}

CollisionPolygon3D::CollisionPolygon3D() {
	set_notify_local_transform(true);
}
