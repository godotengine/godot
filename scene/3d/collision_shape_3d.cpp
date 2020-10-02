/*************************************************************************/
/*  collision_shape_3d.cpp                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "collision_shape_3d.h"

#include "core/math/quick_hull.h"
#include "mesh_instance_3d.h"
#include "physics_body_3d.h"
#include "scene/resources/box_shape_3d.h"
#include "scene/resources/capsule_shape_3d.h"
#include "scene/resources/concave_polygon_shape_3d.h"
#include "scene/resources/convex_polygon_shape_3d.h"
#include "scene/resources/ray_shape_3d.h"
#include "scene/resources/sphere_shape_3d.h"
#include "scene/resources/world_margin_shape_3d.h"
#include "servers/rendering_server.h"

//TODO: Implement CylinderShape and HeightMapShape?

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
					if (!a.empty()) {
						Vector<Vector3> v = a[RenderingServer::ARRAY_VERTEX];
						for (int k = 0; k < v.size(); k++) {
							vertices.append(mi->get_transform().xform(v[k]));
						}
					}
				}
			}
		}
	}

	Ref<ConvexPolygonShape3D> shape = memnew(ConvexPolygonShape3D);
	shape->set_points(vertices);
	set_shape(shape);
}

void CollisionShape3D::_update_in_shape_owner(bool p_xform_only) {
	parent->shape_owner_set_transform(owner_id, get_transform());
	if (p_xform_only) {
		return;
	}
	parent->shape_owner_set_disabled(owner_id, disabled);
}

void CollisionShape3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_PARENTED: {
			parent = Object::cast_to<CollisionObject3D>(get_parent());
			if (parent) {
				owner_id = parent->create_shape_owner(this);
				if (shape.is_valid()) {
					parent->shape_owner_add_shape(owner_id, shape);
				}
				_update_in_shape_owner();
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (parent) {
				_update_in_shape_owner();
			}
			if (get_tree()->is_debugging_collisions_hint()) {
				_update_debug_shape();
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

void CollisionShape3D::resource_changed(RES res) {
	update_gizmo();
}

String CollisionShape3D::get_configuration_warning() const {
	String warning = Node3D::get_configuration_warning();

	if (!Object::cast_to<CollisionObject3D>(get_parent())) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("CollisionShape3D only serves to provide a collision shape to a CollisionObject3D derived node. Please only use it as a child of Area3D, StaticBody3D, RigidBody3D, KinematicBody3D, etc. to give them a shape.");
	}

	if (!shape.is_valid()) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("A shape must be provided for CollisionShape3D to function. Please create a shape resource for it.");
	}

	if (shape.is_valid() &&
			Object::cast_to<RigidBody3D>(get_parent()) &&
			Object::cast_to<ConcavePolygonShape3D>(*shape) &&
			Object::cast_to<RigidBody3D>(get_parent())->get_mode() != RigidBody3D::MODE_STATIC) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("ConcavePolygonShape3D doesn't support RigidBody3D in another mode than static.");
	}

	return warning;
}

void CollisionShape3D::_bind_methods() {
	//not sure if this should do anything
	ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &CollisionShape3D::resource_changed);
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &CollisionShape3D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &CollisionShape3D::get_shape);
	ClassDB::bind_method(D_METHOD("set_disabled", "enable"), &CollisionShape3D::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &CollisionShape3D::is_disabled);
	ClassDB::bind_method(D_METHOD("make_convex_from_siblings"), &CollisionShape3D::make_convex_from_siblings);
	ClassDB::set_method_flags("CollisionShape3D", "make_convex_from_siblings", METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ClassDB::bind_method(D_METHOD("_update_debug_shape"), &CollisionShape3D::_update_debug_shape);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape3D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
}

void CollisionShape3D::set_shape(const Ref<Shape3D> &p_shape) {
	if (!shape.is_null()) {
		shape->unregister_owner(this);
		shape->disconnect("changed", callable_mp(this, &CollisionShape3D::_shape_changed));
	}
	shape = p_shape;
	if (!shape.is_null()) {
		shape->register_owner(this);
		shape->connect("changed", callable_mp(this, &CollisionShape3D::_shape_changed));
	}
	update_gizmo();
	if (parent) {
		parent->shape_owner_clear_shapes(owner_id);
		if (shape.is_valid()) {
			parent->shape_owner_add_shape(owner_id, shape);
		}
	}

	if (is_inside_tree()) {
		_shape_changed();
	}
	update_configuration_warning();
}

Ref<Shape3D> CollisionShape3D::get_shape() const {
	return shape;
}

void CollisionShape3D::set_disabled(bool p_disabled) {
	disabled = p_disabled;
	update_gizmo();
	if (parent) {
		parent->shape_owner_set_disabled(owner_id, p_disabled);
	}
}

bool CollisionShape3D::is_disabled() const {
	return disabled;
}

CollisionShape3D::CollisionShape3D() {
	//indicator = RenderingServer::get_singleton()->mesh_create();
	disabled = false;
	debug_shape = nullptr;
	parent = nullptr;
	owner_id = 0;
	set_notify_local_transform(true);
}

CollisionShape3D::~CollisionShape3D() {
	if (!shape.is_null()) {
		shape->unregister_owner(this);
	}
	//RenderingServer::get_singleton()->free(indicator);
}

void CollisionShape3D::_update_debug_shape() {
	debug_shape_dirty = false;

	if (debug_shape) {
		debug_shape->queue_delete();
		debug_shape = nullptr;
	}

	Ref<Shape3D> s = get_shape();
	if (s.is_null()) {
		return;
	}

	Ref<Mesh> mesh = s->get_debug_mesh();
	MeshInstance3D *mi = memnew(MeshInstance3D);
	mi->set_mesh(mesh);
	add_child(mi);
	debug_shape = mi;
}

void CollisionShape3D::_shape_changed() {
	// If this is a heightfield shape our center may have changed
	if (parent) {
		_update_in_shape_owner(true);
	}

	if (is_inside_tree() && get_tree()->is_debugging_collisions_hint() && !debug_shape_dirty) {
		debug_shape_dirty = true;
		call_deferred("_update_debug_shape");
	}
}
