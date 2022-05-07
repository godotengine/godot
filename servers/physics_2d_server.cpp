/*************************************************************************/
/*  physics_2d_server.cpp                                                */
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

#include "physics_2d_server.h"

#include "core/method_bind_ext.gen.inc"
#include "core/print_string.h"
#include "core/project_settings.h"

Physics2DServer *Physics2DServer::singleton = nullptr;

void Physics2DDirectBodyState::integrate_forces() {
	real_t step = get_step();
	Vector2 lv = get_linear_velocity();
	lv += get_total_gravity() * step;

	real_t av = get_angular_velocity();

	float damp = 1.0 - step * get_total_linear_damp();

	if (damp < 0) { // reached zero in the given time
		damp = 0;
	}

	lv *= damp;

	damp = 1.0 - step * get_total_angular_damp();

	if (damp < 0) { // reached zero in the given time
		damp = 0;
	}

	av *= damp;

	set_linear_velocity(lv);
	set_angular_velocity(av);
}

Object *Physics2DDirectBodyState::get_contact_collider_object(int p_contact_idx) const {
	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance(objid);
	return obj;
}

Physics2DServer *Physics2DServer::get_singleton() {
	return singleton;
}

void Physics2DDirectBodyState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_total_gravity"), &Physics2DDirectBodyState::get_total_gravity);
	ClassDB::bind_method(D_METHOD("get_total_linear_damp"), &Physics2DDirectBodyState::get_total_linear_damp);
	ClassDB::bind_method(D_METHOD("get_total_angular_damp"), &Physics2DDirectBodyState::get_total_angular_damp);

	ClassDB::bind_method(D_METHOD("get_inverse_mass"), &Physics2DDirectBodyState::get_inverse_mass);
	ClassDB::bind_method(D_METHOD("get_inverse_inertia"), &Physics2DDirectBodyState::get_inverse_inertia);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &Physics2DDirectBodyState::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &Physics2DDirectBodyState::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &Physics2DDirectBodyState::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &Physics2DDirectBodyState::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &Physics2DDirectBodyState::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &Physics2DDirectBodyState::get_transform);

	ClassDB::bind_method(D_METHOD("get_velocity_at_local_position", "local_position"), &Physics2DDirectBodyState::get_velocity_at_local_position);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &Physics2DDirectBodyState::add_central_force);
	ClassDB::bind_method(D_METHOD("add_force", "offset", "force"), &Physics2DDirectBodyState::add_force);
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &Physics2DDirectBodyState::add_torque);
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &Physics2DDirectBodyState::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "impulse"), &Physics2DDirectBodyState::apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "offset", "impulse"), &Physics2DDirectBodyState::apply_impulse);

	ClassDB::bind_method(D_METHOD("set_sleep_state", "enabled"), &Physics2DDirectBodyState::set_sleep_state);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &Physics2DDirectBodyState::is_sleeping);

	ClassDB::bind_method(D_METHOD("get_contact_count"), &Physics2DDirectBodyState::get_contact_count);

	ClassDB::bind_method(D_METHOD("get_contact_local_position", "contact_idx"), &Physics2DDirectBodyState::get_contact_local_position);
	ClassDB::bind_method(D_METHOD("get_contact_local_normal", "contact_idx"), &Physics2DDirectBodyState::get_contact_local_normal);
	ClassDB::bind_method(D_METHOD("get_contact_local_shape", "contact_idx"), &Physics2DDirectBodyState::get_contact_local_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider);
	ClassDB::bind_method(D_METHOD("get_contact_collider_position", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider_id", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_id);
	ClassDB::bind_method(D_METHOD("get_contact_collider_object", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_object);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape_metadata", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_shape_metadata);
	ClassDB::bind_method(D_METHOD("get_contact_collider_velocity_at_position", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_step"), &Physics2DDirectBodyState::get_step);
	ClassDB::bind_method(D_METHOD("integrate_forces"), &Physics2DDirectBodyState::integrate_forces);
	ClassDB::bind_method(D_METHOD("get_space_state"), &Physics2DDirectBodyState::get_space_state);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "step"), "", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "inverse_mass"), "", "get_inverse_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "inverse_inertia"), "", "get_inverse_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "total_angular_damp"), "", "get_total_angular_damp");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "total_linear_damp"), "", "get_total_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "total_gravity"), "", "get_total_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleep_state", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "transform"), "set_transform", "get_transform");
}

Physics2DDirectBodyState::Physics2DDirectBodyState() {}

///////////////////////////////////////////////////////

void Physics2DShapeQueryParameters::set_shape(const RES &p_shape) {
	ERR_FAIL_COND(p_shape.is_null());
	shape = p_shape->get_rid();
}

void Physics2DShapeQueryParameters::set_shape_rid(const RID &p_shape) {
	shape = p_shape;
}

RID Physics2DShapeQueryParameters::get_shape_rid() const {
	return shape;
}

void Physics2DShapeQueryParameters::set_transform(const Transform2D &p_transform) {
	transform = p_transform;
}
Transform2D Physics2DShapeQueryParameters::get_transform() const {
	return transform;
}

void Physics2DShapeQueryParameters::set_motion(const Vector2 &p_motion) {
	motion = p_motion;
}
Vector2 Physics2DShapeQueryParameters::get_motion() const {
	return motion;
}

void Physics2DShapeQueryParameters::set_margin(float p_margin) {
	margin = p_margin;
}
float Physics2DShapeQueryParameters::get_margin() const {
	return margin;
}

void Physics2DShapeQueryParameters::set_collision_mask(uint32_t p_collision_mask) {
	collision_mask = p_collision_mask;
}

uint32_t Physics2DShapeQueryParameters::get_collision_mask() const {
	return collision_mask;
}

void Physics2DShapeQueryParameters::set_exclude(const Vector<RID> &p_exclude) {
	exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++) {
		exclude.insert(p_exclude[i]);
	}
}

Vector<RID> Physics2DShapeQueryParameters::get_exclude() const {
	Vector<RID> ret;
	ret.resize(exclude.size());
	int idx = 0;
	for (Set<RID>::Element *E = exclude.front(); E; E = E->next()) {
		ret.write[idx++] = E->get();
	}
	return ret;
}

void Physics2DShapeQueryParameters::set_collide_with_bodies(bool p_enable) {
	collide_with_bodies = p_enable;
}

bool Physics2DShapeQueryParameters::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
}

void Physics2DShapeQueryParameters::set_collide_with_areas(bool p_enable) {
	collide_with_areas = p_enable;
}

bool Physics2DShapeQueryParameters::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void Physics2DShapeQueryParameters::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &Physics2DShapeQueryParameters::set_shape);
	ClassDB::bind_method(D_METHOD("set_shape_rid", "shape"), &Physics2DShapeQueryParameters::set_shape_rid);
	ClassDB::bind_method(D_METHOD("get_shape_rid"), &Physics2DShapeQueryParameters::get_shape_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &Physics2DShapeQueryParameters::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &Physics2DShapeQueryParameters::get_transform);

	ClassDB::bind_method(D_METHOD("set_motion", "motion"), &Physics2DShapeQueryParameters::set_motion);
	ClassDB::bind_method(D_METHOD("get_motion"), &Physics2DShapeQueryParameters::get_motion);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &Physics2DShapeQueryParameters::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &Physics2DShapeQueryParameters::get_margin);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "collision_layer"), &Physics2DShapeQueryParameters::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &Physics2DShapeQueryParameters::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &Physics2DShapeQueryParameters::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &Physics2DShapeQueryParameters::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &Physics2DShapeQueryParameters::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &Physics2DShapeQueryParameters::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &Physics2DShapeQueryParameters::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &Physics2DShapeQueryParameters::is_collide_with_areas_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_NONE, itos(Variant::_RID) + ":"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "motion"), "set_motion", "get_motion");
	//ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D"), "set_shape", ""); // FIXME: Lacks a getter
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "shape_rid"), "set_shape_rid", "get_shape_rid");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
}

Physics2DShapeQueryParameters::Physics2DShapeQueryParameters() {
	margin = 0;
	collision_mask = 0x7FFFFFFF;
	collide_with_bodies = true;
	collide_with_areas = false;
}

Dictionary Physics2DDirectSpaceState::_intersect_ray(const Vector2 &p_from, const Vector2 &p_to, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas) {
	RayResult inters;
	Set<RID> exclude;
	for (int i = 0; i < p_exclude.size(); i++) {
		exclude.insert(p_exclude[i]);
	}

	bool res = intersect_ray(p_from, p_to, inters, exclude, p_layers, p_collide_with_bodies, p_collide_with_areas);

	if (!res) {
		return Dictionary();
	}

	Dictionary d;
	d["position"] = inters.position;
	d["normal"] = inters.normal;
	d["collider_id"] = inters.collider_id;
	d["collider"] = inters.collider;
	d["shape"] = inters.shape;
	d["rid"] = inters.rid;
	d["metadata"] = inters.metadata;

	return d;
}

Array Physics2DDirectSpaceState::_intersect_shape(const Ref<Physics2DShapeQueryParameters> &p_shape_query, int p_max_results) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Array());

	Vector<ShapeResult> sr;
	sr.resize(p_max_results);
	int rc = intersect_shape(p_shape_query->shape, p_shape_query->transform, p_shape_query->motion, p_shape_query->margin, sr.ptrw(), sr.size(), p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
	Array ret;
	ret.resize(rc);
	for (int i = 0; i < rc; i++) {
		Dictionary d;
		d["rid"] = sr[i].rid;
		d["collider_id"] = sr[i].collider_id;
		d["collider"] = sr[i].collider;
		d["shape"] = sr[i].shape;
		d["metadata"] = sr[i].metadata;
		ret[i] = d;
	}

	return ret;
}

Array Physics2DDirectSpaceState::_cast_motion(const Ref<Physics2DShapeQueryParameters> &p_shape_query) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Array());

	float closest_safe, closest_unsafe;
	bool res = cast_motion(p_shape_query->shape, p_shape_query->transform, p_shape_query->motion, p_shape_query->margin, closest_safe, closest_unsafe, p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
	if (!res) {
		return Array();
	}
	Array ret;
	ret.resize(2);
	ret[0] = closest_safe;
	ret[1] = closest_unsafe;
	return ret;
}

Array Physics2DDirectSpaceState::_intersect_point_impl(const Vector2 &p_point, int p_max_results, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_filter_by_canvas, ObjectID p_canvas_instance_id) {
	Set<RID> exclude;
	for (int i = 0; i < p_exclude.size(); i++) {
		exclude.insert(p_exclude[i]);
	}

	Vector<ShapeResult> ret;
	ret.resize(p_max_results);

	int rc;
	if (p_filter_by_canvas) {
		rc = intersect_point(p_point, ret.ptrw(), ret.size(), exclude, p_layers, p_collide_with_bodies, p_collide_with_areas);
	} else {
		rc = intersect_point_on_canvas(p_point, p_canvas_instance_id, ret.ptrw(), ret.size(), exclude, p_layers, p_collide_with_bodies, p_collide_with_areas);
	}

	if (rc == 0) {
		return Array();
	}

	Array r;
	r.resize(rc);
	for (int i = 0; i < rc; i++) {
		Dictionary d;
		d["rid"] = ret[i].rid;
		d["collider_id"] = ret[i].collider_id;
		d["collider"] = ret[i].collider;
		d["shape"] = ret[i].shape;
		d["metadata"] = ret[i].metadata;
		r[i] = d;
	}
	return r;
}

Array Physics2DDirectSpaceState::_intersect_point(const Vector2 &p_point, int p_max_results, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas) {
	return _intersect_point_impl(p_point, p_max_results, p_exclude, p_layers, p_collide_with_bodies, p_collide_with_areas);
}

Array Physics2DDirectSpaceState::_intersect_point_on_canvas(const Vector2 &p_point, ObjectID p_canvas_intance_id, int p_max_results, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas) {
	return _intersect_point_impl(p_point, p_max_results, p_exclude, p_layers, p_collide_with_bodies, p_collide_with_areas, true, p_canvas_intance_id);
}

Array Physics2DDirectSpaceState::_collide_shape(const Ref<Physics2DShapeQueryParameters> &p_shape_query, int p_max_results) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Array());

	Vector<Vector2> ret;
	ret.resize(p_max_results * 2);
	int rc = 0;
	bool res = collide_shape(p_shape_query->shape, p_shape_query->transform, p_shape_query->motion, p_shape_query->margin, ret.ptrw(), p_max_results, rc, p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
	if (!res) {
		return Array();
	}
	Array r;
	r.resize(rc * 2);
	for (int i = 0; i < rc * 2; i++) {
		r[i] = ret[i];
	}
	return r;
}
Dictionary Physics2DDirectSpaceState::_get_rest_info(const Ref<Physics2DShapeQueryParameters> &p_shape_query) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Dictionary());

	ShapeRestInfo sri;

	bool res = rest_info(p_shape_query->shape, p_shape_query->transform, p_shape_query->motion, p_shape_query->margin, &sri, p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
	Dictionary r;
	if (!res) {
		return r;
	}

	r["point"] = sri.point;
	r["normal"] = sri.normal;
	r["rid"] = sri.rid;
	r["collider_id"] = sri.collider_id;
	r["shape"] = sri.shape;
	r["linear_velocity"] = sri.linear_velocity;
	r["metadata"] = sri.metadata;

	return r;
}

Physics2DDirectSpaceState::Physics2DDirectSpaceState() {
}

void Physics2DDirectSpaceState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("intersect_point", "point", "max_results", "exclude", "collision_layer", "collide_with_bodies", "collide_with_areas"), &Physics2DDirectSpaceState::_intersect_point, DEFVAL(32), DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("intersect_point_on_canvas", "point", "canvas_instance_id", "max_results", "exclude", "collision_layer", "collide_with_bodies", "collide_with_areas"), &Physics2DDirectSpaceState::_intersect_point_on_canvas, DEFVAL(32), DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("intersect_ray", "from", "to", "exclude", "collision_layer", "collide_with_bodies", "collide_with_areas"), &Physics2DDirectSpaceState::_intersect_ray, DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("intersect_shape", "shape", "max_results"), &Physics2DDirectSpaceState::_intersect_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("cast_motion", "shape"), &Physics2DDirectSpaceState::_cast_motion);
	ClassDB::bind_method(D_METHOD("collide_shape", "shape", "max_results"), &Physics2DDirectSpaceState::_collide_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("get_rest_info", "shape"), &Physics2DDirectSpaceState::_get_rest_info);
}

///////////////////////////////

Vector2 Physics2DTestMotionResult::get_motion() const {
	return result.motion;
}

Vector2 Physics2DTestMotionResult::get_motion_remainder() const {
	return result.remainder;
}

Vector2 Physics2DTestMotionResult::get_collision_point() const {
	return result.collision_point;
}

Vector2 Physics2DTestMotionResult::get_collision_normal() const {
	return result.collision_normal;
}

Vector2 Physics2DTestMotionResult::get_collider_velocity() const {
	return result.collider_velocity;
}

ObjectID Physics2DTestMotionResult::get_collider_id() const {
	return result.collider_id;
}

RID Physics2DTestMotionResult::get_collider_rid() const {
	return result.collider;
}

Object *Physics2DTestMotionResult::get_collider() const {
	return ObjectDB::get_instance(result.collider_id);
}

int Physics2DTestMotionResult::get_collider_shape() const {
	return result.collider_shape;
}

real_t Physics2DTestMotionResult::get_collision_depth() const {
	return result.collision_depth;
}

real_t Physics2DTestMotionResult::get_collision_safe_fraction() const {
	return result.collision_safe_fraction;
}

real_t Physics2DTestMotionResult::get_collision_unsafe_fraction() const {
	return result.collision_unsafe_fraction;
}

void Physics2DTestMotionResult::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_motion"), &Physics2DTestMotionResult::get_motion);
	ClassDB::bind_method(D_METHOD("get_motion_remainder"), &Physics2DTestMotionResult::get_motion_remainder);
	ClassDB::bind_method(D_METHOD("get_collision_point"), &Physics2DTestMotionResult::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &Physics2DTestMotionResult::get_collision_normal);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &Physics2DTestMotionResult::get_collider_velocity);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &Physics2DTestMotionResult::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &Physics2DTestMotionResult::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider"), &Physics2DTestMotionResult::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &Physics2DTestMotionResult::get_collider_shape);
	ClassDB::bind_method(D_METHOD("get_collision_depth"), &Physics2DTestMotionResult::get_collision_depth);
	ClassDB::bind_method(D_METHOD("get_collision_safe_fraction"), &Physics2DTestMotionResult::get_collision_safe_fraction);
	ClassDB::bind_method(D_METHOD("get_collision_unsafe_fraction"), &Physics2DTestMotionResult::get_collision_unsafe_fraction);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "motion"), "", "get_motion");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "motion_remainder"), "", "get_motion_remainder");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collision_point"), "", "get_collision_point");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collision_normal"), "", "get_collision_normal");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collider_velocity"), "", "get_collider_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_id", PROPERTY_HINT_OBJECT_ID), "", "get_collider_id");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "collider_rid"), "", "get_collider_rid");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider"), "", "get_collider");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape"), "", "get_collider_shape");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "collision_depth"), "", "get_collision_depth");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "collision_safe_fraction"), "", "get_collision_safe_fraction");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "collision_unsafe_fraction"), "", "get_collision_unsafe_fraction");
}

///////////////////////////////////////

bool Physics2DServer::_body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, float p_margin, const Ref<Physics2DTestMotionResult> &p_result, bool p_exclude_raycast_shapes, const Vector<RID> &p_exclude) {
	MotionResult *r = nullptr;
	if (p_result.is_valid()) {
		r = p_result->get_result_ptr();
	}
	Set<RID> exclude;
	for (int i = 0; i < p_exclude.size(); i++) {
		exclude.insert(p_exclude[i]);
	}
	return body_test_motion(p_body, p_from, p_motion, p_infinite_inertia, p_margin, r, p_exclude_raycast_shapes, exclude);
}

void Physics2DServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("line_shape_create"), &Physics2DServer::line_shape_create);
	ClassDB::bind_method(D_METHOD("ray_shape_create"), &Physics2DServer::ray_shape_create);
	ClassDB::bind_method(D_METHOD("segment_shape_create"), &Physics2DServer::segment_shape_create);
	ClassDB::bind_method(D_METHOD("circle_shape_create"), &Physics2DServer::circle_shape_create);
	ClassDB::bind_method(D_METHOD("rectangle_shape_create"), &Physics2DServer::rectangle_shape_create);
	ClassDB::bind_method(D_METHOD("capsule_shape_create"), &Physics2DServer::capsule_shape_create);
	ClassDB::bind_method(D_METHOD("convex_polygon_shape_create"), &Physics2DServer::convex_polygon_shape_create);
	ClassDB::bind_method(D_METHOD("concave_polygon_shape_create"), &Physics2DServer::concave_polygon_shape_create);

	ClassDB::bind_method(D_METHOD("shape_set_data", "shape", "data"), &Physics2DServer::shape_set_data);

	ClassDB::bind_method(D_METHOD("shape_get_type", "shape"), &Physics2DServer::shape_get_type);
	ClassDB::bind_method(D_METHOD("shape_get_data", "shape"), &Physics2DServer::shape_get_data);

	ClassDB::bind_method(D_METHOD("space_create"), &Physics2DServer::space_create);
	ClassDB::bind_method(D_METHOD("space_set_active", "space", "active"), &Physics2DServer::space_set_active);
	ClassDB::bind_method(D_METHOD("space_is_active", "space"), &Physics2DServer::space_is_active);
	ClassDB::bind_method(D_METHOD("space_set_param", "space", "param", "value"), &Physics2DServer::space_set_param);
	ClassDB::bind_method(D_METHOD("space_get_param", "space", "param"), &Physics2DServer::space_get_param);
	ClassDB::bind_method(D_METHOD("space_get_direct_state", "space"), &Physics2DServer::space_get_direct_state);

	ClassDB::bind_method(D_METHOD("area_create"), &Physics2DServer::area_create);
	ClassDB::bind_method(D_METHOD("area_set_space", "area", "space"), &Physics2DServer::area_set_space);
	ClassDB::bind_method(D_METHOD("area_get_space", "area"), &Physics2DServer::area_get_space);

	ClassDB::bind_method(D_METHOD("area_set_space_override_mode", "area", "mode"), &Physics2DServer::area_set_space_override_mode);
	ClassDB::bind_method(D_METHOD("area_get_space_override_mode", "area"), &Physics2DServer::area_get_space_override_mode);

	ClassDB::bind_method(D_METHOD("area_add_shape", "area", "shape", "transform", "disabled"), &Physics2DServer::area_add_shape, DEFVAL(Transform2D()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("area_set_shape", "area", "shape_idx", "shape"), &Physics2DServer::area_set_shape);
	ClassDB::bind_method(D_METHOD("area_set_shape_transform", "area", "shape_idx", "transform"), &Physics2DServer::area_set_shape_transform);
	ClassDB::bind_method(D_METHOD("area_set_shape_disabled", "area", "shape_idx", "disabled"), &Physics2DServer::area_set_shape_disabled);

	ClassDB::bind_method(D_METHOD("area_get_shape_count", "area"), &Physics2DServer::area_get_shape_count);
	ClassDB::bind_method(D_METHOD("area_get_shape", "area", "shape_idx"), &Physics2DServer::area_get_shape);
	ClassDB::bind_method(D_METHOD("area_get_shape_transform", "area", "shape_idx"), &Physics2DServer::area_get_shape_transform);

	ClassDB::bind_method(D_METHOD("area_remove_shape", "area", "shape_idx"), &Physics2DServer::area_remove_shape);
	ClassDB::bind_method(D_METHOD("area_clear_shapes", "area"), &Physics2DServer::area_clear_shapes);

	ClassDB::bind_method(D_METHOD("area_set_collision_layer", "area", "layer"), &Physics2DServer::area_set_collision_layer);
	ClassDB::bind_method(D_METHOD("area_set_collision_mask", "area", "mask"), &Physics2DServer::area_set_collision_mask);

	ClassDB::bind_method(D_METHOD("area_set_param", "area", "param", "value"), &Physics2DServer::area_set_param);
	ClassDB::bind_method(D_METHOD("area_set_transform", "area", "transform"), &Physics2DServer::area_set_transform);

	ClassDB::bind_method(D_METHOD("area_get_param", "area", "param"), &Physics2DServer::area_get_param);
	ClassDB::bind_method(D_METHOD("area_get_transform", "area"), &Physics2DServer::area_get_transform);

	ClassDB::bind_method(D_METHOD("area_attach_object_instance_id", "area", "id"), &Physics2DServer::area_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_object_instance_id", "area"), &Physics2DServer::area_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("area_attach_canvas_instance_id", "area", "id"), &Physics2DServer::area_attach_canvas_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_canvas_instance_id", "area"), &Physics2DServer::area_get_canvas_instance_id);

	ClassDB::bind_method(D_METHOD("area_set_monitor_callback", "area", "receiver", "method"), &Physics2DServer::area_set_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_area_monitor_callback", "area", "receiver", "method"), &Physics2DServer::area_set_area_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_monitorable", "area", "monitorable"), &Physics2DServer::area_set_monitorable);

	ClassDB::bind_method(D_METHOD("body_create"), &Physics2DServer::body_create);

	ClassDB::bind_method(D_METHOD("body_set_space", "body", "space"), &Physics2DServer::body_set_space);
	ClassDB::bind_method(D_METHOD("body_get_space", "body"), &Physics2DServer::body_get_space);

	ClassDB::bind_method(D_METHOD("body_set_mode", "body", "mode"), &Physics2DServer::body_set_mode);
	ClassDB::bind_method(D_METHOD("body_get_mode", "body"), &Physics2DServer::body_get_mode);

	ClassDB::bind_method(D_METHOD("body_add_shape", "body", "shape", "transform", "disabled"), &Physics2DServer::body_add_shape, DEFVAL(Transform2D()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("body_set_shape", "body", "shape_idx", "shape"), &Physics2DServer::body_set_shape);
	ClassDB::bind_method(D_METHOD("body_set_shape_transform", "body", "shape_idx", "transform"), &Physics2DServer::body_set_shape_transform);
	ClassDB::bind_method(D_METHOD("body_set_shape_metadata", "body", "shape_idx", "metadata"), &Physics2DServer::body_set_shape_metadata);

	ClassDB::bind_method(D_METHOD("body_get_shape_count", "body"), &Physics2DServer::body_get_shape_count);
	ClassDB::bind_method(D_METHOD("body_get_shape", "body", "shape_idx"), &Physics2DServer::body_get_shape);
	ClassDB::bind_method(D_METHOD("body_get_shape_transform", "body", "shape_idx"), &Physics2DServer::body_get_shape_transform);
	ClassDB::bind_method(D_METHOD("body_get_shape_metadata", "body", "shape_idx"), &Physics2DServer::body_get_shape_metadata);

	ClassDB::bind_method(D_METHOD("body_remove_shape", "body", "shape_idx"), &Physics2DServer::body_remove_shape);
	ClassDB::bind_method(D_METHOD("body_clear_shapes", "body"), &Physics2DServer::body_clear_shapes);

	ClassDB::bind_method(D_METHOD("body_set_shape_disabled", "body", "shape_idx", "disabled"), &Physics2DServer::body_set_shape_disabled);
	ClassDB::bind_method(D_METHOD("body_set_shape_as_one_way_collision", "body", "shape_idx", "enable", "margin"), &Physics2DServer::body_set_shape_as_one_way_collision);

	ClassDB::bind_method(D_METHOD("body_attach_object_instance_id", "body", "id"), &Physics2DServer::body_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_object_instance_id", "body"), &Physics2DServer::body_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("body_attach_canvas_instance_id", "body", "id"), &Physics2DServer::body_attach_canvas_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_canvas_instance_id", "body"), &Physics2DServer::body_get_canvas_instance_id);

	ClassDB::bind_method(D_METHOD("body_set_continuous_collision_detection_mode", "body", "mode"), &Physics2DServer::body_set_continuous_collision_detection_mode);
	ClassDB::bind_method(D_METHOD("body_get_continuous_collision_detection_mode", "body"), &Physics2DServer::body_get_continuous_collision_detection_mode);

	ClassDB::bind_method(D_METHOD("body_set_collision_layer", "body", "layer"), &Physics2DServer::body_set_collision_layer);
	ClassDB::bind_method(D_METHOD("body_get_collision_layer", "body"), &Physics2DServer::body_get_collision_layer);

	ClassDB::bind_method(D_METHOD("body_set_collision_mask", "body", "mask"), &Physics2DServer::body_set_collision_mask);
	ClassDB::bind_method(D_METHOD("body_get_collision_mask", "body"), &Physics2DServer::body_get_collision_mask);

	ClassDB::bind_method(D_METHOD("body_set_param", "body", "param", "value"), &Physics2DServer::body_set_param);
	ClassDB::bind_method(D_METHOD("body_get_param", "body", "param"), &Physics2DServer::body_get_param);

	ClassDB::bind_method(D_METHOD("body_set_state", "body", "state", "value"), &Physics2DServer::body_set_state);
	ClassDB::bind_method(D_METHOD("body_get_state", "body", "state"), &Physics2DServer::body_get_state);

	ClassDB::bind_method(D_METHOD("body_apply_central_impulse", "body", "impulse"), &Physics2DServer::body_apply_central_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_torque_impulse", "body", "impulse"), &Physics2DServer::body_apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_impulse", "body", "position", "impulse"), &Physics2DServer::body_apply_impulse);
	ClassDB::bind_method(D_METHOD("body_add_central_force", "body", "force"), &Physics2DServer::body_add_central_force);
	ClassDB::bind_method(D_METHOD("body_add_force", "body", "offset", "force"), &Physics2DServer::body_add_force);
	ClassDB::bind_method(D_METHOD("body_add_torque", "body", "torque"), &Physics2DServer::body_add_torque);
	ClassDB::bind_method(D_METHOD("body_set_axis_velocity", "body", "axis_velocity"), &Physics2DServer::body_set_axis_velocity);

	ClassDB::bind_method(D_METHOD("body_add_collision_exception", "body", "excepted_body"), &Physics2DServer::body_add_collision_exception);
	ClassDB::bind_method(D_METHOD("body_remove_collision_exception", "body", "excepted_body"), &Physics2DServer::body_remove_collision_exception);

	ClassDB::bind_method(D_METHOD("body_set_max_contacts_reported", "body", "amount"), &Physics2DServer::body_set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("body_get_max_contacts_reported", "body"), &Physics2DServer::body_get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("body_set_omit_force_integration", "body", "enable"), &Physics2DServer::body_set_omit_force_integration);
	ClassDB::bind_method(D_METHOD("body_is_omitting_force_integration", "body"), &Physics2DServer::body_is_omitting_force_integration);

	ClassDB::bind_method(D_METHOD("body_set_force_integration_callback", "body", "receiver", "method", "userdata"), &Physics2DServer::body_set_force_integration_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_test_motion", "body", "from", "motion", "infinite_inertia", "margin", "result", "exclude_raycast_shapes", "exclude"), &Physics2DServer::_body_test_motion, DEFVAL(0.08), DEFVAL(Variant()), DEFVAL(true), DEFVAL(Array()));

	ClassDB::bind_method(D_METHOD("body_get_direct_state", "body"), &Physics2DServer::body_get_direct_state);

	/* JOINT API */

	ClassDB::bind_method(D_METHOD("joint_set_param", "joint", "param", "value"), &Physics2DServer::joint_set_param);
	ClassDB::bind_method(D_METHOD("joint_get_param", "joint", "param"), &Physics2DServer::joint_get_param);

	ClassDB::bind_method(D_METHOD("pin_joint_create", "anchor", "body_a", "body_b"), &Physics2DServer::pin_joint_create, DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("groove_joint_create", "groove1_a", "groove2_a", "anchor_b", "body_a", "body_b"), &Physics2DServer::groove_joint_create, DEFVAL(RID()), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("damped_spring_joint_create", "anchor_a", "anchor_b", "body_a", "body_b"), &Physics2DServer::damped_spring_joint_create, DEFVAL(RID()));

	ClassDB::bind_method(D_METHOD("damped_string_joint_set_param", "joint", "param", "value"), &Physics2DServer::damped_string_joint_set_param);
	ClassDB::bind_method(D_METHOD("damped_string_joint_get_param", "joint", "param"), &Physics2DServer::damped_string_joint_get_param);

	ClassDB::bind_method(D_METHOD("joint_get_type", "joint"), &Physics2DServer::joint_get_type);

	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &Physics2DServer::free);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &Physics2DServer::set_active);

	ClassDB::bind_method(D_METHOD("set_collision_iterations", "iterations"), &Physics2DServer::set_collision_iterations);

	ClassDB::bind_method(D_METHOD("get_process_info", "process_info"), &Physics2DServer::get_process_info);

	BIND_ENUM_CONSTANT(SPACE_PARAM_CONTACT_RECYCLE_RADIUS);
	BIND_ENUM_CONSTANT(SPACE_PARAM_CONTACT_MAX_SEPARATION);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_TIME_TO_SLEEP);
	BIND_ENUM_CONSTANT(SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS);

	BIND_ENUM_CONSTANT(SHAPE_LINE);
	BIND_ENUM_CONSTANT(SHAPE_RAY);
	BIND_ENUM_CONSTANT(SHAPE_SEGMENT);
	BIND_ENUM_CONSTANT(SHAPE_CIRCLE);
	BIND_ENUM_CONSTANT(SHAPE_RECTANGLE);
	BIND_ENUM_CONSTANT(SHAPE_CAPSULE);
	BIND_ENUM_CONSTANT(SHAPE_CONVEX_POLYGON);
	BIND_ENUM_CONSTANT(SHAPE_CONCAVE_POLYGON);
	BIND_ENUM_CONSTANT(SHAPE_CUSTOM);

	BIND_ENUM_CONSTANT(AREA_PARAM_GRAVITY);
	BIND_ENUM_CONSTANT(AREA_PARAM_GRAVITY_VECTOR);
	BIND_ENUM_CONSTANT(AREA_PARAM_GRAVITY_IS_POINT);
	BIND_ENUM_CONSTANT(AREA_PARAM_GRAVITY_DISTANCE_SCALE);
	BIND_ENUM_CONSTANT(AREA_PARAM_GRAVITY_POINT_ATTENUATION);
	BIND_ENUM_CONSTANT(AREA_PARAM_LINEAR_DAMP);
	BIND_ENUM_CONSTANT(AREA_PARAM_ANGULAR_DAMP);
	BIND_ENUM_CONSTANT(AREA_PARAM_PRIORITY);

	BIND_ENUM_CONSTANT(AREA_SPACE_OVERRIDE_DISABLED);
	BIND_ENUM_CONSTANT(AREA_SPACE_OVERRIDE_COMBINE);
	BIND_ENUM_CONSTANT(AREA_SPACE_OVERRIDE_COMBINE_REPLACE);
	BIND_ENUM_CONSTANT(AREA_SPACE_OVERRIDE_REPLACE);
	BIND_ENUM_CONSTANT(AREA_SPACE_OVERRIDE_REPLACE_COMBINE);

	BIND_ENUM_CONSTANT(BODY_MODE_STATIC);
	BIND_ENUM_CONSTANT(BODY_MODE_KINEMATIC);
	BIND_ENUM_CONSTANT(BODY_MODE_RIGID);
	BIND_ENUM_CONSTANT(BODY_MODE_CHARACTER);

	BIND_ENUM_CONSTANT(BODY_PARAM_BOUNCE);
	BIND_ENUM_CONSTANT(BODY_PARAM_FRICTION);
	BIND_ENUM_CONSTANT(BODY_PARAM_MASS);
	BIND_ENUM_CONSTANT(BODY_PARAM_INERTIA);
	BIND_ENUM_CONSTANT(BODY_PARAM_GRAVITY_SCALE);
	BIND_ENUM_CONSTANT(BODY_PARAM_LINEAR_DAMP);
	BIND_ENUM_CONSTANT(BODY_PARAM_ANGULAR_DAMP);
	BIND_ENUM_CONSTANT(BODY_PARAM_MAX);

	BIND_ENUM_CONSTANT(BODY_STATE_TRANSFORM);
	BIND_ENUM_CONSTANT(BODY_STATE_LINEAR_VELOCITY);
	BIND_ENUM_CONSTANT(BODY_STATE_ANGULAR_VELOCITY);
	BIND_ENUM_CONSTANT(BODY_STATE_SLEEPING);
	BIND_ENUM_CONSTANT(BODY_STATE_CAN_SLEEP);

	BIND_ENUM_CONSTANT(JOINT_PIN);
	BIND_ENUM_CONSTANT(JOINT_GROOVE);
	BIND_ENUM_CONSTANT(JOINT_DAMPED_SPRING);

	BIND_ENUM_CONSTANT(JOINT_PARAM_BIAS);
	BIND_ENUM_CONSTANT(JOINT_PARAM_MAX_BIAS);
	BIND_ENUM_CONSTANT(JOINT_PARAM_MAX_FORCE);

	BIND_ENUM_CONSTANT(DAMPED_STRING_REST_LENGTH);
	BIND_ENUM_CONSTANT(DAMPED_STRING_STIFFNESS);
	BIND_ENUM_CONSTANT(DAMPED_STRING_DAMPING);

	BIND_ENUM_CONSTANT(CCD_MODE_DISABLED);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_RAY);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_SHAPE);

	BIND_ENUM_CONSTANT(AREA_BODY_ADDED);
	BIND_ENUM_CONSTANT(AREA_BODY_REMOVED);

	BIND_ENUM_CONSTANT(INFO_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(INFO_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(INFO_ISLAND_COUNT);
}

Physics2DServer::Physics2DServer() {
	singleton = this;
}

Physics2DServer::~Physics2DServer() {
	singleton = nullptr;
}

Vector<Physics2DServerManager::ClassInfo> Physics2DServerManager::physics_2d_servers;
int Physics2DServerManager::default_server_id = -1;
int Physics2DServerManager::default_server_priority = -1;
const String Physics2DServerManager::setting_property_name("physics/2d/physics_engine");

void Physics2DServerManager::on_servers_changed() {
	String physics_servers("DEFAULT");
	for (int i = get_servers_count() - 1; 0 <= i; --i) {
		physics_servers += "," + get_server_name(i);
	}
	ProjectSettings::get_singleton()->set_custom_property_info(setting_property_name, PropertyInfo(Variant::STRING, setting_property_name, PROPERTY_HINT_ENUM, physics_servers));
}

void Physics2DServerManager::register_server(const String &p_name, CreatePhysics2DServerCallback p_creat_callback) {
	ERR_FAIL_COND(!p_creat_callback);
	ERR_FAIL_COND(find_server_id(p_name) != -1);
	physics_2d_servers.push_back(ClassInfo(p_name, p_creat_callback));
	on_servers_changed();
}

void Physics2DServerManager::set_default_server(const String &p_name, int p_priority) {
	const int id = find_server_id(p_name);
	ERR_FAIL_COND(id == -1); // Not found
	if (default_server_priority < p_priority) {
		default_server_id = id;
		default_server_priority = p_priority;
	}
}

int Physics2DServerManager::find_server_id(const String &p_name) {
	for (int i = physics_2d_servers.size() - 1; 0 <= i; --i) {
		if (p_name == physics_2d_servers[i].name) {
			return i;
		}
	}
	return -1;
}

int Physics2DServerManager::get_servers_count() {
	return physics_2d_servers.size();
}

String Physics2DServerManager::get_server_name(int p_id) {
	ERR_FAIL_INDEX_V(p_id, get_servers_count(), "");
	return physics_2d_servers[p_id].name;
}

Physics2DServer *Physics2DServerManager::new_default_server() {
	ERR_FAIL_COND_V(default_server_id == -1, nullptr);
	return physics_2d_servers[default_server_id].create_callback();
}

Physics2DServer *Physics2DServerManager::new_server(const String &p_name) {
	int id = find_server_id(p_name);
	if (id == -1) {
		return nullptr;
	} else {
		return physics_2d_servers[id].create_callback();
	}
}
