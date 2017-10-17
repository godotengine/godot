/*************************************************************************/
/*  collision_shape.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "collision_shape.h"
#include "scene/resources/box_shape.h"
#include "scene/resources/capsule_shape.h"
#include "scene/resources/concave_polygon_shape.h"
#include "scene/resources/convex_polygon_shape.h"
#include "scene/resources/plane_shape.h"
#include "scene/resources/ray_shape.h"
#include "scene/resources/sphere_shape.h"
#include "servers/visual_server.h"
//TODO: Implement CylinderShape and HeightMapShape?
#include "mesh_instance.h"
#include "physics_body.h"
#include "quick_hull.h"

void CollisionShape::make_convex_from_brothers() {

	Node *p = get_parent();
	if (!p)
		return;

	for (int i = 0; i < p->get_child_count(); i++) {

		Node *n = p->get_child(i);
		MeshInstance *mi = Object::cast_to<MeshInstance>(n);
		if (mi) {

			Ref<Mesh> m = mi->get_mesh();
			if (m.is_valid()) {

				Ref<Shape> s = m->create_convex_shape();
				set_shape(s);
			}
		}
	}
}

void CollisionShape::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_PARENTED: {
			parent = Object::cast_to<CollisionObject>(get_parent());
			if (parent) {
				owner_id = parent->create_shape_owner(this);
				if (shape.is_valid()) {
					parent->shape_owner_add_shape(owner_id, shape);
				}
				parent->shape_owner_set_transform(owner_id, get_transform());
				parent->shape_owner_set_disabled(owner_id, disabled);
			}
		} break;
		case NOTIFICATION_ENTER_TREE: {
			if (get_tree()->is_debugging_collisions_hint()) {
				_create_debug_shape();
			}

		} break;
		case NOTIFICATION_LOCAL_TRANSFORM_CHANGED: {
			if (parent) {
				parent->shape_owner_set_transform(owner_id, get_transform());
			}
		} break;
		case NOTIFICATION_UNPARENTED: {
			if (parent) {
				parent->remove_shape_owner(owner_id);
			}
			owner_id = 0;
			parent = NULL;
		} break;
	}
}

void CollisionShape::resource_changed(RES res) {

	update_gizmo();
}

String CollisionShape::get_configuration_warning() const {

	if (!Object::cast_to<CollisionObject>(get_parent())) {
		return TTR("CollisionShape only serves to provide a collision shape to a CollisionObject derived node. Please only use it as a child of Area, StaticBody, RigidBody, KinematicBody, etc. to give them a shape.");
	}

	if (!shape.is_valid()) {
		return TTR("A shape must be provided for CollisionShape to function. Please create a shape resource for it!");
	}

	return String();
}

void CollisionShape::_bind_methods() {

	//not sure if this should do anything
	ClassDB::bind_method(D_METHOD("resource_changed", "resource"), &CollisionShape::resource_changed);
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &CollisionShape::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &CollisionShape::get_shape);
	ClassDB::bind_method(D_METHOD("set_disabled", "enable"), &CollisionShape::set_disabled);
	ClassDB::bind_method(D_METHOD("is_disabled"), &CollisionShape::is_disabled);
	ClassDB::bind_method(D_METHOD("make_convex_from_brothers"), &CollisionShape::make_convex_from_brothers);
	ClassDB::set_method_flags("CollisionShape", "make_convex_from_brothers", METHOD_FLAGS_DEFAULT | METHOD_FLAG_EDITOR);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "disabled"), "set_disabled", "is_disabled");
}

void CollisionShape::set_shape(const Ref<Shape> &p_shape) {

	if (!shape.is_null())
		shape->unregister_owner(this);
	shape = p_shape;
	if (!shape.is_null())
		shape->register_owner(this);
	update_gizmo();
	if (parent) {
		parent->shape_owner_clear_shapes(owner_id);
		if (shape.is_valid()) {
			parent->shape_owner_add_shape(owner_id, shape);
		}
	}

	update_configuration_warning();
}

Ref<Shape> CollisionShape::get_shape() const {

	return shape;
}

void CollisionShape::set_disabled(bool p_disabled) {

	disabled = p_disabled;
	update_gizmo();
	if (parent) {
		parent->shape_owner_set_disabled(owner_id, p_disabled);
	}
}

bool CollisionShape::is_disabled() const {

	return disabled;
}

CollisionShape::CollisionShape() {

	//indicator = VisualServer::get_singleton()->mesh_create();
	disabled = false;
	debug_shape = NULL;
	parent = NULL;
	owner_id = 0;
	set_notify_local_transform(true);
}

CollisionShape::~CollisionShape() {
	if (!shape.is_null())
		shape->unregister_owner(this);
	//VisualServer::get_singleton()->free(indicator);
}

void CollisionShape::_create_debug_shape() {

	if (debug_shape) {
		debug_shape->queue_delete();
		debug_shape = NULL;
	}

	Ref<Shape> s = get_shape();

	if (s.is_null())
		return;

	Ref<Mesh> mesh = s->get_debug_mesh();

	MeshInstance *mi = memnew(MeshInstance);
	mi->set_mesh(mesh);

	add_child(mi);
	debug_shape = mi;
}
