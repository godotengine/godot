/*************************************************************************/
/*  shape_cast_3d.cpp                                                    */
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

#include "shape_cast_3d.h"

#include "core/config/engine.h"
#include "core/core_string_names.h"
#include "scene/2d/collision_object_2d.h"
#include "scene/2d/physics_body_2d.h"
#include "scene/3d/collision_object_3d.h"
#include "scene/3d/collision_shape_3d.h"
#include "scene/resources/circle_shape_2d.h"
#include "servers/physics_2d/godot_physics_server_2d.h"

void ShapeCast3D::set_target_position(const Vector3 &p_point) {
	target_position = p_point;
	if (is_inside_tree() && (Engine::get_singleton()->is_editor_hint() || get_tree()->is_debugging_collisions_hint())) {
		force_update_transform();
	}
}

Vector3 ShapeCast3D::get_target_position() const {
	return target_position;
}

void ShapeCast3D::set_margin(real_t p_margin) {
	margin = p_margin;
}

real_t ShapeCast3D::get_margin() const {
	return margin;
}

void ShapeCast3D::set_max_results(int p_max_results) {
	max_results = p_max_results;
}

int ShapeCast3D::get_max_results() const {
	return max_results;
}

void ShapeCast3D::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t ShapeCast3D::get_collision_mask() const {
	return collision_mask;
}

void ShapeCast3D::set_collision_mask_value(int p_layer_number, bool p_value) {
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

bool ShapeCast3D::get_collision_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1, false, "Collision layer number must be between 1 and 32 inclusive.");
	ERR_FAIL_COND_V_MSG(p_layer_number > 32, false, "Collision layer number must be between 1 and 32 inclusive.");
	return get_collision_mask() & (1 << (p_layer_number - 1));
}

int ShapeCast3D::get_collision_count() const {
	return result.size();
}

bool ShapeCast3D::is_colliding() const {
	return collided;
}

Object *ShapeCast3D::get_collider(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), nullptr, "No collider found.");

	if (result[p_idx].collider_id.is_null()) {
		return nullptr;
	}
	return ObjectDB::get_instance(result[p_idx].collider_id);
}

int ShapeCast3D::get_collider_shape(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), -1, "No collider shape found.");
	return result[p_idx].shape;
}

Vector3 ShapeCast3D::get_collision_point(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), Vector3(), "No collision point found.");
	return result[p_idx].point;
}

Vector3 ShapeCast3D::get_collision_normal(int p_idx) const {
	ERR_FAIL_INDEX_V_MSG(p_idx, result.size(), Vector3(), "No collision normal found.");
	return result[p_idx].normal;
}

real_t ShapeCast3D::get_closest_collision_safe_fraction() const {
	return collision_safe_fraction;
}

real_t ShapeCast3D::get_closest_collision_unsafe_fraction() const {
	return collision_unsafe_fraction;
}

void ShapeCast3D::set_enabled(bool p_enabled) {
	enabled = p_enabled;
	force_update_transform();
	if (is_inside_tree() && !Engine::get_singleton()->is_editor_hint()) {
		set_physics_process_internal(p_enabled);
	}
	if (!p_enabled) {
		collided = false;
	}
}

bool ShapeCast3D::is_enabled() const {
	return enabled;
}

void ShapeCast3D::set_shape(const Ref<Shape3D> &p_shape) {
	shape = p_shape;
	if (p_shape.is_valid()) {
		shape->connect(CoreStringNames::get_singleton()->changed, callable_mp(this, &ShapeCast3D::_redraw_shape));
		shape_rid = shape->get_rid();
	}
	update_configuration_warnings();
	force_update_transform();
}

Ref<Shape3D> ShapeCast3D::get_shape() const {
	return shape;
}

void ShapeCast3D::set_exclude_parent_body(bool p_exclude_parent_body) {
	if (exclude_parent_body == p_exclude_parent_body) {
		return;
	}
	exclude_parent_body = p_exclude_parent_body;

	if (!is_inside_tree()) {
		return;
	}
	if (Object::cast_to<CollisionObject3D>(get_parent())) {
		if (exclude_parent_body) {
			exclude.insert(Object::cast_to<CollisionObject3D>(get_parent())->get_rid());
		} else {
			exclude.erase(Object::cast_to<CollisionObject3D>(get_parent())->get_rid());
		}
	}
}

bool ShapeCast3D::get_exclude_parent_body() const {
	return exclude_parent_body;
}

void ShapeCast3D::_redraw_shape() {
	force_update_transform();
}

void ShapeCast3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			if (enabled && !Engine::get_singleton()->is_editor_hint()) {
				set_physics_process_internal(true);
			} else {
				set_physics_process_internal(false);
			}
			if (Object::cast_to<CollisionObject3D>(get_parent())) {
				if (exclude_parent_body) {
					exclude.insert(Object::cast_to<CollisionObject3D>(get_parent())->get_rid());
				} else {
					exclude.erase(Object::cast_to<CollisionObject3D>(get_parent())->get_rid());
				}
			}
		} break;
		case NOTIFICATION_EXIT_TREE: {
			if (enabled) {
				set_physics_process_internal(false);
			}
		} break;

		case NOTIFICATION_TRANSFORM_CHANGED: {
#ifdef TOOLS_ENABLED
			// ERR_FAIL_COND(!is_inside_tree());
			// if (!Engine::get_singleton()->is_editor_hint() && !get_tree()->is_debugging_collisions_hint()) {
			// 	break;
			// }
			// if (shape.is_null()) {
			// 	break;
			// }
			// Color draw_col = get_tree()->get_debug_collisions_color();
			// if (!enabled) {
			// 	float g = draw_col.get_v();
			// 	draw_col.r = g;
			// 	draw_col.g = g;
			// 	draw_col.b = g;
			// }
			// // Draw continuos chain of shapes along the cast.
			// const int steps = MAX(2, target_position.length() / shape->get_rect().get_size().length() * 4);
			// for (int i = 0; i <= steps; ++i) {
			// 	Vector3 t = (real_t(i) / steps) * target_position;
			// 	draw_set_transform(t, 0.0, Vector3(1, 1, 1));
			// 	shape->draw(get_canvas_item(), draw_col);
			// }
			// draw_set_transform(Vector3(), 0.0, Vector3(1, 1, 1));

			// // Draw an arrow indicating where the ShapeCast is pointing to.
			// if (target_position != Vector3()) {
			// 	Transform3D xf;
			// 	xf.rotate(target_position.angle());
			// 	xf.translate(Vector3(target_position.length(), 0));

			// 	draw_line(Vector3(), target_position, draw_col, 3);
			// 	Vector<Vector3> pts;
			// 	float tsize = 8;
			// 	pts.push_back(xf.xform(Vector3(0, tsize, 0)));
			// 	pts.push_back(xf.xform(Vector3(0, 0, Math_SQRT12 * tsize)));
			// 	pts.push_back(xf.xform(Vector3(0, 0, -Math_SQRT12 * tsize)));
			// 	Vector<Color> cols;
			// 	for (int i = 0; i < 3; i++) {
			// 		cols.push_back(draw_col);
			// 	}

			// 	draw_primitive(pts, cols, Vector<Vector3>());
			// }
#endif
		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (!enabled) {
				break;
			}
			_update_shapecast_state();
		} break;
	}
}

void ShapeCast3D::_update_shapecast_state() {
	result.clear();

	ERR_FAIL_COND_MSG(shape.is_null(), "Invalid shape.");

	Ref<World3D> w3d = get_world_3d();
	ERR_FAIL_COND(w3d.is_null());

	PhysicsDirectSpaceState3D *dss = PhysicsServer3D::get_singleton()->space_get_direct_state(w3d->get_space());
	ERR_FAIL_COND(!dss);

	Transform3D gt = get_global_transform();

	PhysicsDirectSpaceState3D::ShapeParameters params;
	params.shape_rid = shape_rid;
	params.transform = gt;
	params.motion = gt.basis.xform(target_position);
	params.margin = margin;
	params.exclude = exclude;
	params.collision_mask = collision_mask;
	params.collide_with_bodies = collide_with_bodies;
	params.collide_with_areas = collide_with_areas;

	collision_safe_fraction = 0.0;
	collision_unsafe_fraction = 0.0;

	if (target_position != Vector3()) {
		dss->cast_motion(params, collision_safe_fraction, collision_unsafe_fraction);
		if (collision_unsafe_fraction < 1.0) {
			// Move shape transform to the point of impact,
			// so we can collect contact info at that point.
			gt.set_origin(gt.get_origin() + params.motion * (collision_unsafe_fraction + CMP_EPSILON));
			params.transform = gt;
		}
	}
	// Regardless of whether the shape is stuck or it's moved along
	// the motion vector, we'll only consider static collisions from now on.
	params.motion = Vector3();

	bool intersected = true;
	while (intersected && result.size() < max_results) {
		PhysicsDirectSpaceState3D::ShapeRestInfo info;
		intersected = dss->rest_info(params, &info);
		if (intersected) {
			result.push_back(info);
			params.exclude.insert(info.rid);
		}
	}
	collided = !result.is_empty();
}

void ShapeCast3D::force_shapecast_update() {
	_update_shapecast_state();
}

void ShapeCast3D::add_exception_rid(const RID &p_rid) {
	exclude.insert(p_rid);
}

void ShapeCast3D::add_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject3D *co = Object::cast_to<CollisionObject3D>(p_object);
	if (!co) {
		return;
	}
	add_exception_rid(co->get_rid());
}

void ShapeCast3D::remove_exception_rid(const RID &p_rid) {
	exclude.erase(p_rid);
}

void ShapeCast3D::remove_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject3D *co = Object::cast_to<CollisionObject3D>(p_object);
	if (!co) {
		return;
	}
	remove_exception_rid(co->get_rid());
}

void ShapeCast3D::clear_exceptions() {
	exclude.clear();
}

void ShapeCast3D::set_collide_with_areas(bool p_clip) {
	collide_with_areas = p_clip;
}

bool ShapeCast3D::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void ShapeCast3D::set_collide_with_bodies(bool p_clip) {
	collide_with_bodies = p_clip;
}

bool ShapeCast3D::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
}

Array ShapeCast3D::_get_collision_result() const {
	Array ret;

	for (int i = 0; i < result.size(); ++i) {
		const PhysicsDirectSpaceState3D::ShapeRestInfo &sri = result[i];

		Dictionary col;
		col["point"] = sri.point;
		col["normal"] = sri.normal;
		col["rid"] = sri.rid;
		col["collider"] = ObjectDB::get_instance(sri.collider_id);
		col["collider_id"] = sri.collider_id;
		col["shape"] = sri.shape;
		col["linear_velocity"] = sri.linear_velocity;

		ret.push_back(col);
	}
	return ret;
}

TypedArray<String> ShapeCast3D::get_configuration_warnings() const {
	TypedArray<String> warnings = Node3D::get_configuration_warnings();

	if (shape.is_null()) {
		warnings.push_back(TTR("This node cannot interact with other objects unless a Shape3D is assigned."));
	}
	return warnings;
}

void ShapeCast3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_enabled", "enabled"), &ShapeCast3D::set_enabled);
	ClassDB::bind_method(D_METHOD("is_enabled"), &ShapeCast3D::is_enabled);

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &ShapeCast3D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &ShapeCast3D::get_shape);

	ClassDB::bind_method(D_METHOD("set_target_position", "local_point"), &ShapeCast3D::set_target_position);
	ClassDB::bind_method(D_METHOD("get_target_position"), &ShapeCast3D::get_target_position);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &ShapeCast3D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &ShapeCast3D::get_margin);

	ClassDB::bind_method(D_METHOD("set_max_results", "max_results"), &ShapeCast3D::set_max_results);
	ClassDB::bind_method(D_METHOD("get_max_results"), &ShapeCast3D::get_max_results);

	ClassDB::bind_method(D_METHOD("is_colliding"), &ShapeCast3D::is_colliding);
	ClassDB::bind_method(D_METHOD("get_collision_count"), &ShapeCast3D::get_collision_count);

	ClassDB::bind_method(D_METHOD("force_shapecast_update"), &ShapeCast3D::force_shapecast_update);

	ClassDB::bind_method(D_METHOD("get_collider", "index"), &ShapeCast3D::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape", "index"), &ShapeCast3D::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collision_point", "index"), &ShapeCast3D::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal", "index"), &ShapeCast3D::get_collision_normal);

	ClassDB::bind_method(D_METHOD("get_closest_collision_safe_fraction"), &ShapeCast3D::get_closest_collision_safe_fraction);
	ClassDB::bind_method(D_METHOD("get_closest_collision_unsafe_fraction"), &ShapeCast3D::get_closest_collision_unsafe_fraction);

	ClassDB::bind_method(D_METHOD("add_exception_rid", "rid"), &ShapeCast3D::add_exception_rid);
	ClassDB::bind_method(D_METHOD("add_exception", "node"), &ShapeCast3D::add_exception);

	ClassDB::bind_method(D_METHOD("remove_exception_rid", "rid"), &ShapeCast3D::remove_exception_rid);
	ClassDB::bind_method(D_METHOD("remove_exception", "node"), &ShapeCast3D::remove_exception);

	ClassDB::bind_method(D_METHOD("clear_exceptions"), &ShapeCast3D::clear_exceptions);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &ShapeCast3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &ShapeCast3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_value", "layer_number", "value"), &ShapeCast3D::set_collision_mask_value);
	ClassDB::bind_method(D_METHOD("get_collision_mask_value", "layer_number"), &ShapeCast3D::get_collision_mask_value);

	ClassDB::bind_method(D_METHOD("set_exclude_parent_body", "mask"), &ShapeCast3D::set_exclude_parent_body);
	ClassDB::bind_method(D_METHOD("get_exclude_parent_body"), &ShapeCast3D::get_exclude_parent_body);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &ShapeCast3D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &ShapeCast3D::is_collide_with_areas_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &ShapeCast3D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &ShapeCast3D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("_get_collision_result"), &ShapeCast3D::_get_collision_result);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enabled"), "set_enabled", "is_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape3D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "exclude_parent"), "set_exclude_parent_body", "get_exclude_parent_body");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "target_position"), "set_target_position", "get_target_position");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "max_results"), "set_max_results", "get_max_results");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "collision_result", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "", "_get_collision_result");
	ADD_GROUP("Collide With", "collide_with");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_areas", "is_collide_with_areas_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
}
