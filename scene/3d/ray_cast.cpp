/**************************************************************************/
/*  ray_cast.cpp                                                          */
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

#include "ray_cast.h"

#include "collision_object.h"
#include "core/engine.h"
#include "mesh_instance.h"
#include "servers/physics_server.h"

void RayCast::set_cast_to(const Vector3 &p_point) {
	cast_to = p_point;
	update_gizmo();

	if (Engine::get_singleton()->is_editor_hint()) {
		if (is_inside_tree()) {
			_update_debug_shape_vertices();
		}
	} else if (debug_shape) {
		_update_debug_shape();
	}
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

void RayCast::set_collision_mask_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision mask bit must be between 0 and 31 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool RayCast::get_collision_mask_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision mask bit must be between 0 and 31 inclusive.");
	return get_collision_mask() & (1 << p_bit);
}

bool RayCast::is_colliding() const {
	return collided;
}
Object *RayCast::get_collider() const {
	if (against == 0) {
		return nullptr;
	}

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
	update_gizmo();

	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
		set_physics_process_internal(p_enabled);
	}
	if (!p_enabled) {
		collided = false;
	}

	if (is_inside_tree() && get_tree()->is_debugging_collisions_hint()) {
		if (p_enabled) {
			_update_debug_shape();
		} else {
			_clear_debug_shape();
		}
	}
}

bool RayCast::is_enabled() const {
	return enabled;
}

void RayCast::set_exclude_parent_body(bool p_exclude_parent_body) {
	if (exclude_parent_body == p_exclude_parent_body) {
		return;
	}

	exclude_parent_body = p_exclude_parent_body;

	if (!is_inside_tree()) {
		return;
	}

	if (Object::cast_to<CollisionObject>(get_parent())) {
		if (exclude_parent_body) {
			exclude.insert(Object::cast_to<CollisionObject>(get_parent())->get_rid());
		} else {
			exclude.erase(Object::cast_to<CollisionObject>(get_parent())->get_rid());
		}
	}
}

bool RayCast::get_exclude_parent_body() const {
	return exclude_parent_body;
}

void RayCast::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (Engine::get_singleton()->is_editor_hint()) {
				_update_debug_shape_vertices();
			}
			if (enabled && !Engine::get_singleton()->is_editor_hint()) {
				set_physics_process_internal(true);
			} else {
				set_physics_process_internal(false);
			}

			if (get_tree()->is_debugging_collisions_hint()) {
				_update_debug_shape();
			}

			if (Object::cast_to<CollisionObject>(get_parent())) {
				if (exclude_parent_body) {
					exclude.insert(Object::cast_to<CollisionObject>(get_parent())->get_rid());
				} else {
					exclude.erase(Object::cast_to<CollisionObject>(get_parent())->get_rid());
				}
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (enabled) {
				set_physics_process_internal(false);
			}

			if (debug_shape) {
				_clear_debug_shape();
			}

		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!enabled) {
				break;
			}

			bool prev_collision_state = collided;
			_update_raycast_state();
			if (prev_collision_state != collided && get_tree()->is_debugging_collisions_hint()) {
				_update_debug_shape_material(true);
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
	if (to == Vector3()) {
		to = Vector3(0, 0.01, 0);
	}

	PhysicsDirectSpaceState::RayResult rr;

	if (dss->intersect_ray(gt.get_origin(), gt.xform(to), rr, exclude, collision_mask, collide_with_bodies, collide_with_areas)) {
		collided = true;
		against = rr.collider_id;
		collision_point = rr.position;
		collision_normal = rr.normal;
		against_shape = rr.shape;
	} else {
		collided = false;
		against = 0;
		against_shape = 0;
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
	ERR_FAIL_COND_MSG(!co, "The passed Node must be an instance of CollisionObject.");
	add_exception_rid(co->get_rid());
}

void RayCast::remove_exception_rid(const RID &p_rid) {
	exclude.erase(p_rid);
}

void RayCast::remove_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject *co = Object::cast_to<CollisionObject>(p_object);
	ERR_FAIL_COND_MSG(!co, "The passed Node must be an instance of CollisionObject.");
	remove_exception_rid(co->get_rid());
}

void RayCast::clear_exceptions() {
	exclude.clear();

	if (exclude_parent_body && is_inside_tree()) {
		CollisionObject *parent = Object::cast_to<CollisionObject>(get_parent());
		if (parent) {
			exclude.insert(parent->get_rid());
		}
	}
}

void RayCast::set_collide_with_areas(bool p_clip) {
	collide_with_areas = p_clip;
}

bool RayCast::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void RayCast::set_collide_with_bodies(bool p_clip) {
	collide_with_bodies = p_clip;
}

bool RayCast::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
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

	ClassDB::bind_method(D_METHOD("set_exclude_parent_body", "mask"), &RayCast::set_exclude_parent_body);
	ClassDB::bind_method(D_METHOD("get_exclude_parent_body"), &RayCast::get_exclude_parent_body);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &RayCast::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &RayCast::is_collide_with_areas_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &RayCast::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &RayCast::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_debug_shape_custom_color", "debug_shape_custom_color"), &RayCast::set_debug_shape_custom_color);
	ClassDB::bind_method(D_METHOD("get_debug_shape_custom_color"), &RayCast::get_debug_shape_custom_color);

	ClassDB::bind_method(D_METHOD("set_debug_shape_thickness", "debug_shape_thickness"), &RayCast::set_debug_shape_thickness);
	ClassDB::bind_method(D_METHOD("get_debug_shape_thickness"), &RayCast::get_debug_shape_thickness);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclude_parent"), "set_exclude_parent_body", "get_exclude_parent_body");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "cast_to"), "set_cast_to", "get_cast_to");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	ADD_GROUP("Collide With", "collide_with");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_areas", "is_collide_with_areas_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_bodies", "is_collide_with_bodies_enabled");

	ADD_GROUP("Debug Shape", "debug_shape");
	ADD_PROPERTY(PropertyInfo(Variant::COLOR, "debug_shape_custom_color"), "set_debug_shape_custom_color", "get_debug_shape_custom_color");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug_shape_thickness", PROPERTY_HINT_RANGE, "1,5"), "set_debug_shape_thickness", "get_debug_shape_thickness");
}

int RayCast::get_debug_shape_thickness() const {
	return debug_shape_thickness;
}

void RayCast::_update_debug_shape_vertices() {
	debug_shape_vertices.clear();
	debug_line_vertices.clear();

	if (cast_to == Vector3()) {
		return;
	}

	debug_line_vertices.push_back(Vector3());
	debug_line_vertices.push_back(cast_to);

	if (debug_shape_thickness > 1) {
		float scale_factor = 100.0;
		Vector3 dir = Vector3(cast_to).normalized();
		// Draw truncated pyramid
		Vector3 normal = (fabs(dir.x) + fabs(dir.y) > CMP_EPSILON) ? Vector3(-dir.y, dir.x, 0).normalized() : Vector3(0, -dir.z, dir.y).normalized();
		normal *= debug_shape_thickness / scale_factor;
		int vertices_strip_order[14] = { 4, 5, 0, 1, 2, 5, 6, 4, 7, 0, 3, 2, 7, 6 };
		for (int v = 0; v < 14; v++) {
			Vector3 vertex = vertices_strip_order[v] < 4 ? normal : normal / 3.0 + cast_to;
			debug_shape_vertices.push_back(vertex.rotated(dir, Math_PI * (0.5 * (vertices_strip_order[v] % 4) + 0.25)));
		}
	}
}

void RayCast::set_debug_shape_thickness(const int p_debug_shape_thickness) {
	debug_shape_thickness = p_debug_shape_thickness;
	update_gizmo();

	if (Engine::get_singleton()->is_editor_hint()) {
		if (is_inside_tree()) {
			_update_debug_shape_vertices();
		}
	} else if (debug_shape) {
		_update_debug_shape();
	}
}

const Vector<Vector3> &RayCast::get_debug_shape_vertices() const {
	return debug_shape_vertices;
}

const Vector<Vector3> &RayCast::get_debug_line_vertices() const {
	return debug_line_vertices;
}

void RayCast::set_debug_shape_custom_color(const Color &p_color) {
	debug_shape_custom_color = p_color;
	if (debug_material.is_valid()) {
		_update_debug_shape_material();
	}
}

Ref<Material3D> RayCast::get_debug_material() {
	_update_debug_shape_material();
	return debug_material;
}

const Color &RayCast::get_debug_shape_custom_color() const {
	return debug_shape_custom_color;
}

void RayCast::_create_debug_shape() {
	_update_debug_shape_material();

	Ref<ArrayMesh> mesh = memnew(ArrayMesh);

	MeshInstance *mi = memnew(MeshInstance);
#ifdef TOOLS_ENABLED
	// This enables the debug helper to show up in editor runs.
	// However it should not show up during export, because global mode
	// can slow the portal system, and this should only be used for debugging.
	mi->set_portal_mode(CullInstance::PORTAL_MODE_GLOBAL);
#endif
	mi->set_mesh(mesh);
	add_child(mi);

	debug_shape = mi;
}

void RayCast::_update_debug_shape_material(bool p_check_collision) {
	if (!debug_material.is_valid()) {
		Ref<SpatialMaterial> material = memnew(SpatialMaterial);
		debug_material = material;

		material->set_flag(Material3D::FLAG_UNSHADED, true);
		material->set_feature(Material3D::FEATURE_TRANSPARENT, true);
		// Use double-sided rendering so that the RayCast can be seen if the camera is inside.
		material->set_cull_mode(Material3D::CULL_DISABLED);
	}

	Color color = debug_shape_custom_color;
	if (color == Color(0.0, 0.0, 0.0)) {
		// Use the default debug shape color defined in the Project Settings.
		color = get_tree()->get_debug_collisions_color();
	}

	if (p_check_collision && collided) {
		if ((color.get_h() < 0.055 || color.get_h() > 0.945) && color.get_s() > 0.5 && color.get_v() > 0.5) {
			// If base color is already quite reddish, highlight collision with green color
			color = Color(0.0, 1.0, 0.0, color.a);
		} else {
			// Else, highlight collision with red color
			color = Color(1.0, 0, 0, color.a);
		}
	}

	Ref<Material3D> material = static_cast<Ref<Material3D>>(debug_material);
	material->set_albedo(color);
}

void RayCast::_update_debug_shape() {
	if (!enabled) {
		return;
	}

	if (!debug_shape) {
		_create_debug_shape();
	}

	MeshInstance *mi = static_cast<MeshInstance *>(debug_shape);
	Ref<ArrayMesh> mesh = mi->get_mesh();
	if (!mesh.is_valid()) {
		return;
	}

	_update_debug_shape_vertices();

	mesh->clear_surfaces();

	Array a;
	a.resize(Mesh::ARRAY_MAX);

	uint32_t flags = 0;
	int surface_count = 0;

	if (!debug_line_vertices.empty()) {
		a[Mesh::ARRAY_VERTEX] = debug_line_vertices;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_LINES, a, Array(), flags);
		mesh->surface_set_material(surface_count, debug_material);
		++surface_count;
	}

	if (!debug_shape_vertices.empty()) {
		a[Mesh::ARRAY_VERTEX] = debug_shape_vertices;
		mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLE_STRIP, a, Array(), flags);
		mesh->surface_set_material(surface_count, debug_material);
		++surface_count;
	}
}

void RayCast::_clear_debug_shape() {
	if (!debug_shape) {
		return;
	}

	MeshInstance *mi = static_cast<MeshInstance *>(debug_shape);
	if (mi->is_inside_tree()) {
		mi->queue_delete();
	} else {
		memdelete(mi);
	}

	debug_shape = nullptr;
}

RayCast::RayCast() {
	enabled = false;
	against = 0;
	collided = false;
	against_shape = 0;
	collision_mask = 1;
	cast_to = Vector3(0, -1, 0);
	debug_shape = nullptr;
	exclude_parent_body = true;
	collide_with_areas = false;
	collide_with_bodies = true;
}
