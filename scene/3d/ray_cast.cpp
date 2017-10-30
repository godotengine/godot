/*************************************************************************/
/*  ray_cast.cpp                                                         */
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
#include "ray_cast.h"

#include "collision_object.h"
#include "engine.h"
#include "mesh_instance.h"
#include "servers/physics_server.h"

void RayCast::set_cast_to(const Vector3 &p_point) {

	cast_to = p_point;
	if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || get_tree()->is_debugging_collisions_hint()))
		update_gizmo();
	if (is_inside_tree() && get_tree()->is_debugging_collisions_hint())
		_update_debug_shape();
}

Vector3 RayCast::get_cast_to() const {

	return cast_to;
}

void RayCast::set_collision_mask(uint32_t p_mask) {

	collision_mask = p_mask;
}

uint32_t RayCast::get_collision_mask() const {

	return collision_mask;
}

void RayCast::set_type_mask(uint32_t p_mask) {

	type_mask = p_mask;
}

void RayCast::set_collision_mask_bit(int p_bit, bool p_value) {

	uint32_t mask = get_collision_mask();
	if (p_value)
		mask |= 1 << p_bit;
	else
		mask &= ~(1 << p_bit);
	set_collision_mask(mask);
}

bool RayCast::get_collision_mask_bit(int p_bit) const {

	return get_collision_mask() & (1 << p_bit);
}

uint32_t RayCast::get_type_mask() const {

	return type_mask;
}

bool RayCast::is_colliding() const {

	return collided;
}
Object *RayCast::get_collider() const {

	if (against == 0)
		return NULL;

	return ObjectDB::get_instance(against);
}

int RayCast::get_collider_shape() const {

	return against_shape;
}
Vector3 RayCast::get_collision_point() const {

	return collision_point;
}
Vector3 RayCast::get_collision_normal() const {

	return collision_normal;
}

void RayCast::set_enabled(bool p_enabled) {

	enabled = p_enabled;
	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint())
		set_physics_process(p_enabled);
	if (!p_enabled)
		collided = false;

	if (is_inside_tree() && get_tree()->is_debugging_collisions_hint()) {
		if (p_enabled)
			_update_debug_shape();
		else
			_clear_debug_shape();
	}
}

bool RayCast::is_enabled() const {

	return enabled;
}

void RayCast::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_ENTER_TREE: {

			if (enabled && !Engine::get_singleton()->is_editor_hint()) {
				set_physics_process(true);

				if (get_tree()->is_debugging_collisions_hint())
					_update_debug_shape();
			} else
				set_physics_process(false);

		} break;
		case NOTIFICATION_EXIT_TREE: {

			if (enabled) {
				set_physics_process(false);
			}

			if (debug_shape)
				_clear_debug_shape();

		} break;
		case NOTIFICATION_PHYSICS_PROCESS: {

			if (!enabled)
				break;

			bool prev_collision_state = collided;
			_update_raycast_state();
			if (prev_collision_state != collided && get_tree()->is_debugging_collisions_hint()) {
				if (debug_material.is_valid()) {
					Ref<SpatialMaterial> line_material = static_cast<Ref<SpatialMaterial> >(debug_material);
					line_material->set_albedo(collided ? Color(1.0, 0, 0) : Color(1.0, 0.8, 0.6));
				}
			}

		} break;
	}
}

void RayCast::_update_raycast_state() {
	Ref<World> w3d = get_world();
	ERR_FAIL_COND(w3d.is_null());

	PhysicsDirectSpaceState *dss = PhysicsServer::get_singleton()->space_get_direct_state(w3d->get_space());
	ERR_FAIL_COND(!dss);

	Transform gt = get_global_transform();

	Vector3 to = cast_to;
	if (to == Vector3())
		to = Vector3(0, 0.01, 0);

	PhysicsDirectSpaceState::RayResult rr;

	if (dss->intersect_ray(gt.get_origin(), gt.xform(to), rr, exclude, collision_mask, type_mask)) {

		collided = true;
		against = rr.collider_id;
		collision_point = rr.position;
		collision_normal = rr.normal;
		against_shape = rr.shape;
	} else {
		collided = false;
	}
}

void RayCast::force_raycast_update() {
	_update_raycast_state();
}

void RayCast::add_exception_rid(const RID &p_rid) {

	exclude.insert(p_rid);
}

void RayCast::add_exception(const Object *p_object) {

	ERR_FAIL_NULL(p_object);
	const CollisionObject *co = Object::cast_to<CollisionObject>(p_object);
	if (!co)
		return;
	add_exception_rid(co->get_rid());
}

void RayCast::remove_exception_rid(const RID &p_rid) {

	exclude.erase(p_rid);
}

void RayCast::remove_exception(const Object *p_object) {

	ERR_FAIL_NULL(p_object);
	const CollisionObject *co = Object::cast_to<CollisionObject>(p_object);
	if (!co)
		return;
	remove_exception_rid(co->get_rid());
}

void RayCast::clear_exceptions() {

	exclude.clear();
}

void RayCast::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &RayCast::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &RayCast::is_enabled);

	ClassDB::bind_method(D_METHOD("set_cast_to", "local_point"), &RayCast::set_cast_to);
	ClassDB::bind_method(D_METHOD("get_cast_to"), &RayCast::get_cast_to);

	ClassDB::bind_method(D_METHOD("is_colliding"), &RayCast::is_colliding);
	ClassDB::bind_method(D_METHOD("force_raycast_update"), &RayCast::force_raycast_update);

	ClassDB::bind_method(D_METHOD("get_collider"), &RayCast::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &RayCast::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collision_point"), &RayCast::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &RayCast::get_collision_normal);

	ClassDB::bind_method(D_METHOD("add_exception_rid", "rid"), &RayCast::add_exception_rid);
	ClassDB::bind_method(D_METHOD("add_exception", "node"), &RayCast::add_exception);

	ClassDB::bind_method(D_METHOD("remove_exception_rid", "rid"), &RayCast::remove_exception_rid);
	ClassDB::bind_method(D_METHOD("remove_exception", "node"), &RayCast::remove_exception);

	ClassDB::bind_method(D_METHOD("clear_exceptions"), &RayCast::clear_exceptions);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &RayCast::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &RayCast::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &RayCast::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &RayCast::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("set_type_mask", "mask"), &RayCast::set_type_mask);
	ClassDB::bind_method(D_METHOD("get_type_mask"), &RayCast::get_type_mask);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "cast_to"), "set_cast_to", "get_cast_to");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "type_mask", PROPERTY_HINT_FLAGS, "Static,Kinematic,Rigid,Character,Area"), "set_type_mask", "get_type_mask");
}

void RayCast::_create_debug_shape() {

	if (!debug_material.is_valid()) {
		debug_material = Ref<SpatialMaterial>(memnew(SpatialMaterial));

		Ref<SpatialMaterial> line_material = static_cast<Ref<SpatialMaterial> >(debug_material);
		line_material->set_flag(SpatialMaterial::FLAG_UNSHADED, true);
		line_material->set_line_width(3.0);
		line_material->set_albedo(Color(1.0, 0.8, 0.6));
	}

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);

	MeshInstance *mi = memnew(MeshInstance);
	mi->set_mesh(mesh);

	add_child(mi);
	debug_shape = mi;
}

void RayCast::_update_debug_shape() {

	if (!enabled)
		return;

	if (!debug_shape)
		_create_debug_shape();

	MeshInstance *mi = static_cast<MeshInstance *>(debug_shape);
	if (!mi->get_mesh().is_valid())
		return;

	Ref<ArrayMesh> mesh = mi->get_mesh();
	if (mesh->get_surface_count() > 0)
		mesh->surface_remove(0);

	Array a;
	a.resize(Mesh::ARRAY_MAX);

	Vector<Vector3> verts;
	verts.push_back(Vector3());
	verts.push_back(cast_to);
	a[Mesh::ARRAY_VERTEX] = verts;

	mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, a);
	mesh->surface_set_material(0, debug_material);
}

void RayCast::_clear_debug_shape() {

	if (!debug_shape)
		return;

	MeshInstance *mi = static_cast<MeshInstance *>(debug_shape);
	if (mi->is_inside_tree())
		mi->queue_delete();
	else
		memdelete(mi);

	debug_shape = NULL;
}

RayCast::RayCast() {

	enabled = false;
	against = 0;
	collided = false;
	against_shape = 0;
	collision_mask = 1;
	type_mask = PhysicsDirectSpaceState::TYPE_MASK_COLLISION;
	cast_to = Vector3(0, -1, 0);
	debug_shape = NULL;
}
