/**************************************************************************/
/*  collision_shape_3d.cpp                                                */
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

#include "collision_shape_3d.h"

#include "mesh_instance_3d.h"
#include "physics_body_3d.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/convex_polygon_shape_3d.h"
#include "scene/resources/world_boundary_shape_3d.h"

void CollisionShape3D::make_convex_from_siblings() {
	Node *p = get_parent();
	if (!p) {
		return;
	}

	Vector<Vector3> vertices;

	for (int i = 0; i < p->get_child_count(); i++) {
		Node *n = p->get_child(i);
		MeshInstance3D *mi = Object::cast_to<MeshInstance3D>(n);
		if (mi) {
			Ref<Mesh> m = mi->get_mesh();
			if (m.is_valid()) {
				for (int j = 0; j < m->get_surface_count(); j++) {
					Array a = m->surface_get_arrays(j);
					if (!a.is_empty()) {
						Vector<Vector3> v = a[RenderingServer::ARRAY_VERTEX];
						for (int k = 0; k < v.size(); k++) {
							vertices.append(mi->get_transform().xform(v[k]));
						}
					}
				}
			}
		}
	}

	Ref<ConvexPolygonShape3D> shape_new = memnew(ConvexPolygonShape3D);
	shape_new->set_points(vertices);
	set_shape(shape_new);
}

void CollisionShape3D::_update_in_shape_owner(bool p_xform_only) {
	collision_object->shape_owner_set_transform(owner_id, get_transform());
	if (p_xform_only) {
		return;
	}
	collision_object->shape_owner_set_disabled(owner_id, disabled);
}

void CollisionShape3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PARENTED: {
			collision_object = Object::cast_to<CollisionObject3D>(get_parent());
			if (collision_object) {
				owner_id = collision_object->create_shape_owner(this);
				if (shape.is_valid()) {
					collision_object->shape_owner_add_shape(owner_id, shape);
				}
				_update_in_shape_owner();
			}
		} break;

		case NOTIFICATION_ENTER_TREE: {
			if (collision_object) {
				_update_in_shape_owner();
			}
		} break;

		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			if (collision_object) {
				_update_in_shape_owner(true);
			}
			update_configuration_warnings();
		} break;

		case NOTIFICATION_UNPARENTED: {
			if (collision_object) {
				collision_object->remove_shape_owner(owner_id);
			}
			owner_id = 0;
			collision_object = nullptr;
		} break;
	}
}

#ifndef DISABLE_DEPRECATED
void CollisionShape3D::resource_changed(Ref<Resource> res) {
}
#endif

PackedStringArray CollisionShape3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	CollisionObject3D *col_object = Object::cast_to<CollisionObject3D>(get_parent());
	if (col_object == nullptr) {
		warnings.push_back(RTR("CollisionShape3D only serves to provide a collision shape to a CollisionObject3D derived node.\nPlease only use it as a child of Area3D, StaticBody3D, RigidBody3D, CharacterBody3D, etc. to give them a shape."));
	}

	if (!shape.is_valid()) {
		warnings.push_back(RTR("A shape must be provided for CollisionShape3D to function. Please create a shape resource for it."));
	}

	if (shape.is_valid() && Object::cast_to<RigidBody3D>(col_object)) {
		if (Object::cast_to<ConcavePolygonShape3D>(*shape)) {
			warnings.push_back(RTR("ConcavePolygonShape3D doesn't support RigidBody3D in another mode than static."));
		} else if (Object::cast_to<WorldBoundaryShape3D>(*shape)) {
			warnings.push_back(RTR("WorldBoundaryShape3D doesn't support RigidBody3D in another mode than static."));
		}
	}

	Vector3 scale = get_transform().get_basis().get_scale();
	if (!(Math::is_zero_approx(scale.x - scale.y) && Math::is_zero_approx(scale.y - scale.z))) {
		warnings.push_back(RTR("A non-uniformly scaled CollisionShape3D node will probably not function as expected.\nPlease make its scale uniform (i.e. the same on all axes), and change the size of its shape resource instead."));
	}

	return warnings;
}

void CollisionShape3D::_bind_methods() {
#ifndef DISABLE_DEPRECATED
	ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &CollisionShape3D::resource_changed);
#endif
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &CollisionShape3D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &CollisionShape3D::get_shape);
	ClassDB::bind_method(D_METHOD("set_disabled", "enable"), &CollisionShape3D::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &CollisionShape3D::is_disabled);
	ClassDB::bind_method(D_METHOD("make_convex_from_siblings"), &CollisionShape3D::make_convex_from_siblings);
	ClassDB::set_method_flags("CollisionShape3D", "make_convex_from_siblings", METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape3D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
}

void CollisionShape3D::set_shape(const Ref<Shape3D> &p_shape) {
	if (p_shape == shape) {
		return;
	}
	if (shape.is_valid()) {
		shape->disconnect_changed(callable_mp((Node3D *)this, &Node3D::update_gizmos));
	}
	shape = p_shape;
	if (shape.is_valid()) {
		shape->connect_changed(callable_mp((Node3D *)this, &Node3D::update_gizmos));
	}
	update_gizmos();
	if (collision_object) {
		collision_object->shape_owner_clear_shapes(owner_id);
		if (shape.is_valid()) {
			collision_object->shape_owner_add_shape(owner_id, shape);
		}
	}

	if (is_inside_tree() && collision_object) {
		// If this is a heightfield shape our center may have changed
		_update_in_shape_owner(true);
	}
	update_configuration_warnings();
}

Ref<Shape3D> CollisionShape3D::get_shape() const {
	return shape;
}

void CollisionShape3D::set_disabled(bool p_disabled) {
	disabled = p_disabled;
	update_gizmos();
	if (collision_object) {
		collision_object->shape_owner_set_disabled(owner_id, p_disabled);
	}
}

bool CollisionShape3D::is_disabled() const {
	return disabled;
}

CollisionShape3D::CollisionShape3D() {
	//indicator = RenderingServer::get_singleton()->mesh_create();
	set_notify_local_transform(true);
}

CollisionShape3D::~CollisionShape3D() {
	//RenderingServer::get_singleton()->free(indicator);
}
