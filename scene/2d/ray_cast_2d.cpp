/*************************************************************************/
/*  ray_cast_2d.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "ray_cast_2d.h"

#include "collision_object_2d.h"

void RayCast2D::set_target_position(const Vector2 &p_point) {
	target_position = p_point;
	if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || get_tree()->is_debugging_collisions_hint())) {
		update();
	}
}

Vector2 RayCast2D::get_target_position() const {
	return target_position;
}

void RayCast2D::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t RayCast2D::get_collision_mask() const {
	return collision_mask;
}

void RayCast2D::set_collision_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_MSG(p_layer_number > 32, "Collision layer number must be between 1 and 32 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << (p_layer_number - 1);
	} else {
		mask &= ~(1 << (p_layer_number - 1));
	}
	set_collision_mask(mask);
}

bool RayCast2D::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

bool RayCast2D::is_colliding() const {
	return collided;
}

Object *RayCast2D::get_collider() const {
	if (against.is_null()) {
		return nullptr;
	}

	return ObjectDB::get_instance(against);
}

int RayCast2D::get_collider_shape() const {
	return against_shape;
}

Vector2 RayCast2D::get_collision_point() const {
	return collision_point;
}

Vector2 RayCast2D::get_collision_normal() const {
	return collision_normal;
}

void RayCast2D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	update();
	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
		set_physics_process_internal(p_enabled);
	}
	if (!p_enabled) {
		collided = false;
	}
}

bool RayCast2D::is_enabled() const {
	return enabled;
}

void RayCast2D::set_exclude_parent_body(bool p_exclude_parent_body) {
	if (exclude_parent_body == p_exclude_parent_body) {
		return;
	}

	exclude_parent_body = p_exclude_parent_body;

	if (!is_inside_tree()) {
		return;
	}

	if (Object::cast_to<CollisionObject2D>(get_parent())) {
		if (exclude_parent_body) {
			exclude.insert(Object::cast_to<CollisionObject2D>(get_parent())->get_rid());
		} else {
			exclude.erase(Object::cast_to<CollisionObject2D>(get_parent())->get_rid());
		}
	}
}

bool RayCast2D::get_exclude_parent_body() const {
	return exclude_parent_body;
}

void RayCast2D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (enabled && !Engine::get_singleton()->is_editor_hint()) {
				set_physics_process_internal(true);
			} else {
				set_physics_process_internal(false);
			}

			if (Object::cast_to<CollisionObject2D>(get_parent())) {
				if (exclude_parent_body) {
					exclude.insert(Object::cast_to<CollisionObject2D>(get_parent())->get_rid());
				} else {
					exclude.erase(Object::cast_to<CollisionObject2D>(get_parent())->get_rid());
				}
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (enabled) {
				set_physics_process_internal(false);
			}

		} break;

		case NOTIFICATION_DRAW: {
			ERR_FAIL_COND(!is_inside_tree());
			if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
				break;
			}
			_draw_debug_shape();

		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!enabled) {
				break;
			}

			_update_raycast_state();

		} break;
	}
}

void RayCast2D::_update_raycast_state() {
	Ref<World2D> w2d = get_world_2d();
	ERR_FAIL_COND(w2d.is_null());

	PhysicsDirectSpaceState2D *dss = PhysicsServer2D::get_singleton()->space_get_direct_state(w2d->get_space());
	ERR_FAIL_COND(!dss);

	Transform2D gt = get_global_transform();

	Vector2 to = target_position;
	if (to == Vector2()) {
		to = Vector2(0, 0.01);
	}

	PhysicsDirectSpaceState2D::RayResult rr;
	bool prev_collision_state = collided;

	PhysicsDirectSpaceState2D::RayParameters ray_params;
	ray_params.from = gt.get_origin();
	ray_params.to = gt.xform(to);
	ray_params.exclude = exclude;
	ray_params.collision_mask = collision_mask;
	ray_params.collide_with_bodies = collide_with_bodies;
	ray_params.collide_with_areas = collide_with_areas;
	ray_params.hit_from_inside = hit_from_inside;

	if (dss->intersect_ray(ray_params, rr)) {
		collided = true;
		against = rr.collider_id;
		collision_point = rr.position;
		collision_normal = rr.normal;
		against_shape = rr.shape;
	} else {
		collided = false;
		against = ObjectID();
		against_shape = 0;
	}

	if (prev_collision_state != collided) {
		update();
	}
}

void RayCast2D::_draw_debug_shape() {
	Color draw_col = collided ? Color(1.0, 0.01, 0) : get_tree()->get_debug_collisions_color();
	if (!enabled) {
		const float g = draw_col.get_v();
		draw_col.r = g;
		draw_col.g = g;
		draw_col.b = g;
	}

	// Draw an arrow indicating where the RayCast is pointing to
	const real_t max_arrow_size = 6;
	const real_t line_width = 1.4;
	bool no_line = target_position.length() < line_width;
	real_t arrow_size = CLAMP(target_position.length() * 2 / 3, line_width, max_arrow_size);

	if (no_line) {
		arrow_size = target_position.length();
	} else {
		draw_line(Vector2(), target_position - target_position.normalized() * arrow_size, draw_col, line_width);
	}

	Transform2D xf;
	xf.rotate(target_position.angle());
	xf.translate(Vector2(no_line ? 0 : target_position.length() - arrow_size, 0));

	Vector<Vector2> pts = {
		xf.xform(Vector2(arrow_size, 0)),
		xf.xform(Vector2(0, 0.5 * arrow_size)),
		xf.xform(Vector2(0, -0.5 * arrow_size))
	};

	Vector<Color> cols = { draw_col, draw_col, draw_col };

	draw_primitive(pts, cols, Vector<Vector2>());
}

void RayCast2D::force_raycast_update() {
	_update_raycast_state();
}

void RayCast2D::add_exception_rid(const RID &p_rid) {
	exclude.insert(p_rid);
}

void RayCast2D::add_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject2D *co = Object::cast_to<CollisionObject2D>(p_object);
	if (!co) {
		return;
	}
	add_exception_rid(co->get_rid());
}

void RayCast2D::remove_exception_rid(const RID &p_rid) {
	exclude.erase(p_rid);
}

void RayCast2D::remove_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject2D *co = Object::cast_to<CollisionObject2D>(p_object);
	if (!co) {
		return;
	}
	remove_exception_rid(co->get_rid());
}

void RayCast2D::clear_exceptions() {
	exclude.clear();
}

void RayCast2D::set_collide_with_areas(bool p_enabled) {
	collide_with_areas = p_enabled;
}

bool RayCast2D::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void RayCast2D::set_collide_with_bodies(bool p_enabled) {
	collide_with_bodies = p_enabled;
}

bool RayCast2D::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
}

void RayCast2D::set_hit_from_inside(bool p_enabled) {
	hit_from_inside = p_enabled;
}

bool RayCast2D::is_hit_from_inside_enabled() const {
	return hit_from_inside;
}

void RayCast2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &RayCast2D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &RayCast2D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_target_position", "local_point"), &RayCast2D::set_target_position);
	ClassDB::bind_method(D_METHOD("get_target_position"), &RayCast2D::get_target_position);

	ClassDB::bind_method(D_METHOD("is_colliding"), &RayCast2D::is_colliding);
	ClassDB::bind_method(D_METHOD("force_raycast_update"), &RayCast2D::force_raycast_update);

	ClassDB::bind_method(D_METHOD("get_collider"), &RayCast2D::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &RayCast2D::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collision_point"), &RayCast2D::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &RayCast2D::get_collision_normal);

	ClassDB::bind_method(D_METHOD("add_exception_rid", "rid"), &RayCast2D::add_exception_rid);
	ClassDB::bind_method(D_METHOD("add_exception", "node"), &RayCast2D::add_exception);

	ClassDB::bind_method(D_METHOD("remove_exception_rid", "rid"), &RayCast2D::remove_exception_rid);
	ClassDB::bind_method(D_METHOD("remove_exception", "node"), &RayCast2D::remove_exception);

	ClassDB::bind_method(D_METHOD("clear_exceptions"), &RayCast2D::clear_exceptions);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &RayCast2D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &RayCast2D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &RayCast2D::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &RayCast2D::get_collision_mask_value);

	ClassDB::bind_method(D_METHOD("set_exclude_parent_body", "mask"), &RayCast2D::set_exclude_parent_body);
	ClassDB::bind_method(D_METHOD("get_exclude_parent_body"), &RayCast2D::get_exclude_parent_body);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &RayCast2D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &RayCast2D::is_collide_with_areas_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &RayCast2D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &RayCast2D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_hit_from_inside", "enable"), &RayCast2D::set_hit_from_inside);
	ClassDB::bind_method(D_METHOD("is_hit_from_inside_enabled"), &RayCast2D::is_hit_from_inside_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclude_parent"), "set_exclude_parent_body", "get_exclude_parent_body");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "target_position"), "set_target_position", "get_target_position");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "hit_from_inside"), "set_hit_from_inside", "is_hit_from_inside_enabled");

	ADD_GROUP("Collide With", "collide_with");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_areas", "is_collide_with_areas_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
}

RayCast2D::RayCast2D() {
}
