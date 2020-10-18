/*************************************************************************/
/*  physics_server_2d.cpp                                                */
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

#include "physics_server_2d.h"

#include "core/print_string.h"
#include "core/project_settings.h"

PhysicsServer2D *PhysicsServer2D::singleton = nullptr;

void PhysicsDirectBodyState2D::integrate_forces() {
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

Object *PhysicsDirectBodyState2D::get_contact_collider_object(int p_contact_idx) const {
	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance(objid);
	return obj;
}

PhysicsServer2D *PhysicsServer2D::get_singleton() {
	return singleton;
}

void PhysicsDirectBodyState2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_total_gravity"), &PhysicsDirectBodyState2D::get_total_gravity);
	ClassDB::bind_method(D_METHOD("get_total_linear_damp"), &PhysicsDirectBodyState2D::get_total_linear_damp);
	ClassDB::bind_method(D_METHOD("get_total_angular_damp"), &PhysicsDirectBodyState2D::get_total_angular_damp);

	ClassDB::bind_method(D_METHOD("get_inverse_mass"), &PhysicsDirectBodyState2D::get_inverse_mass);
	ClassDB::bind_method(D_METHOD("get_inverse_inertia"), &PhysicsDirectBodyState2D::get_inverse_inertia);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &PhysicsDirectBodyState2D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &PhysicsDirectBodyState2D::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &PhysicsDirectBodyState2D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &PhysicsDirectBodyState2D::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsDirectBodyState2D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsDirectBodyState2D::get_transform);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &PhysicsDirectBodyState2D::add_central_force);
	ClassDB::bind_method(D_METHOD("add_force", "force", "position"), &PhysicsDirectBodyState2D::add_force, Vector2());
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &PhysicsDirectBodyState2D::add_torque);
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &PhysicsDirectBodyState2D::apply_central_impulse);
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "impulse"), &PhysicsDirectBodyState2D::apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &PhysicsDirectBodyState2D::apply_impulse, Vector2());

	ClassDB::bind_method(D_METHOD("set_sleep_state", "enabled"), &PhysicsDirectBodyState2D::set_sleep_state);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &PhysicsDirectBodyState2D::is_sleeping);

	ClassDB::bind_method(D_METHOD("get_contact_count"), &PhysicsDirectBodyState2D::get_contact_count);

	ClassDB::bind_method(D_METHOD("get_contact_local_position", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_local_position);
	ClassDB::bind_method(D_METHOD("get_contact_local_normal", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_local_normal);
	ClassDB::bind_method(D_METHOD("get_contact_local_shape", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_local_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider);
	ClassDB::bind_method(D_METHOD("get_contact_collider_position", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider_id", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_id);
	ClassDB::bind_method(D_METHOD("get_contact_collider_object", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_object);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape_metadata", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_shape_metadata);
	ClassDB::bind_method(D_METHOD("get_contact_collider_velocity_at_position", "contact_idx"), &PhysicsDirectBodyState2D::get_contact_collider_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_step"), &PhysicsDirectBodyState2D::get_step);
	ClassDB::bind_method(D_METHOD("integrate_forces"), &PhysicsDirectBodyState2D::integrate_forces);
	ClassDB::bind_method(D_METHOD("get_space_state"), &PhysicsDirectBodyState2D::get_space_state);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step"), "", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inverse_mass"), "", "get_inverse_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inverse_inertia"), "", "get_inverse_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_angular_damp"), "", "get_total_angular_damp");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_linear_damp"), "", "get_total_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "total_gravity"), "", "get_total_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleep_state", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "transform"), "set_transform", "get_transform");
}

PhysicsDirectBodyState2D::PhysicsDirectBodyState2D() {}

///////////////////////////////////////////////////////

void PhysicsShapeQueryParameters2D::set_shape(const RES &p_shape_ref) {
	ERR_FAIL_COND(p_shape_ref.is_null());
	shape_ref = p_shape_ref;
	shape = p_shape_ref->get_rid();
}

RES PhysicsShapeQueryParameters2D::get_shape() const {
	return shape_ref;
}

void PhysicsShapeQueryParameters2D::set_shape_rid(const RID &p_shape) {
	if (shape != p_shape) {
		shape_ref = RES();
		shape = p_shape;
	}
}

RID PhysicsShapeQueryParameters2D::get_shape_rid() const {
	return shape;
}

void PhysicsShapeQueryParameters2D::set_transform(const Transform2D &p_transform) {
	transform = p_transform;
}

Transform2D PhysicsShapeQueryParameters2D::get_transform() const {
	return transform;
}

void PhysicsShapeQueryParameters2D::set_motion(const Vector2 &p_motion) {
	motion = p_motion;
}

Vector2 PhysicsShapeQueryParameters2D::get_motion() const {
	return motion;
}

void PhysicsShapeQueryParameters2D::set_margin(float p_margin) {
	margin = p_margin;
}

float PhysicsShapeQueryParameters2D::get_margin() const {
	return margin;
}

void PhysicsShapeQueryParameters2D::set_collision_mask(int p_collision_mask) {
	collision_mask = p_collision_mask;
}

int PhysicsShapeQueryParameters2D::get_collision_mask() const {
	return collision_mask;
}

void PhysicsShapeQueryParameters2D::set_exclude(const Vector<RID> &p_exclude) {
	exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++) {
		exclude.insert(p_exclude[i]);
	}
}

Vector<RID> PhysicsShapeQueryParameters2D::get_exclude() const {
	Vector<RID> ret;
	ret.resize(exclude.size());
	int idx = 0;
	for (Set<RID>::Element *E = exclude.front(); E; E = E->next()) {
		ret.write[idx] = E->get();
	}
	return ret;
}

void PhysicsShapeQueryParameters2D::set_collide_with_bodies(bool p_enable) {
	collide_with_bodies = p_enable;
}

bool PhysicsShapeQueryParameters2D::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
}

void PhysicsShapeQueryParameters2D::set_collide_with_areas(bool p_enable) {
	collide_with_areas = p_enable;
}

bool PhysicsShapeQueryParameters2D::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void PhysicsShapeQueryParameters2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &PhysicsShapeQueryParameters2D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &PhysicsShapeQueryParameters2D::get_shape);
	ClassDB::bind_method(D_METHOD("set_shape_rid", "shape"), &PhysicsShapeQueryParameters2D::set_shape_rid);
	ClassDB::bind_method(D_METHOD("get_shape_rid"), &PhysicsShapeQueryParameters2D::get_shape_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsShapeQueryParameters2D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsShapeQueryParameters2D::get_transform);

	ClassDB::bind_method(D_METHOD("set_motion", "motion"), &PhysicsShapeQueryParameters2D::set_motion);
	ClassDB::bind_method(D_METHOD("get_motion"), &PhysicsShapeQueryParameters2D::get_motion);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &PhysicsShapeQueryParameters2D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &PhysicsShapeQueryParameters2D::get_margin);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "collision_layer"), &PhysicsShapeQueryParameters2D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &PhysicsShapeQueryParameters2D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &PhysicsShapeQueryParameters2D::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &PhysicsShapeQueryParameters2D::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &PhysicsShapeQueryParameters2D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &PhysicsShapeQueryParameters2D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &PhysicsShapeQueryParameters2D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &PhysicsShapeQueryParameters2D::is_collide_with_areas_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_layer", PROPERTY_HINT_LAYERS_2D_PHYSICS), "set_collision_layer", "get_collision_layer");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_NONE, itos(Variant::_RID) + ":"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "motion"), "set_motion", "get_motion");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape2D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "shape_rid"), "set_shape_rid", "get_shape_rid");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM2D, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
}

PhysicsShapeQueryParameters2D::PhysicsShapeQueryParameters2D() {
	margin = 0;
	collision_mask = 0x7FFFFFFF;
	collide_with_bodies = true;
	collide_with_areas = false;
}

Dictionary PhysicsDirectSpaceState2D::_intersect_ray(const Vector2 &p_from, const Vector2 &p_to, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas) {
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

Array PhysicsDirectSpaceState2D::_intersect_shape(const Ref<PhysicsShapeQueryParameters2D> &p_shape_query, int p_max_results) {
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

Array PhysicsDirectSpaceState2D::_cast_motion(const Ref<PhysicsShapeQueryParameters2D> &p_shape_query) {
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

Array PhysicsDirectSpaceState2D::_intersect_point_impl(const Vector2 &p_point, int p_max_results, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_filter_by_canvas, ObjectID p_canvas_instance_id) {
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

Array PhysicsDirectSpaceState2D::_intersect_point(const Vector2 &p_point, int p_max_results, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas) {
	return _intersect_point_impl(p_point, p_max_results, p_exclude, p_layers, p_collide_with_bodies, p_collide_with_areas);
}

Array PhysicsDirectSpaceState2D::_intersect_point_on_canvas(const Vector2 &p_point, ObjectID p_canvas_intance_id, int p_max_results, const Vector<RID> &p_exclude, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas) {
	return _intersect_point_impl(p_point, p_max_results, p_exclude, p_layers, p_collide_with_bodies, p_collide_with_areas, true, p_canvas_intance_id);
}

Array PhysicsDirectSpaceState2D::_collide_shape(const Ref<PhysicsShapeQueryParameters2D> &p_shape_query, int p_max_results) {
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

Dictionary PhysicsDirectSpaceState2D::_get_rest_info(const Ref<PhysicsShapeQueryParameters2D> &p_shape_query) {
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

PhysicsDirectSpaceState2D::PhysicsDirectSpaceState2D() {
}

void PhysicsDirectSpaceState2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("intersect_point", "point", "max_results", "exclude", "collision_layer", "collide_with_bodies", "collide_with_areas"), &PhysicsDirectSpaceState2D::_intersect_point, DEFVAL(32), DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("intersect_point_on_canvas", "point", "canvas_instance_id", "max_results", "exclude", "collision_layer", "collide_with_bodies", "collide_with_areas"), &PhysicsDirectSpaceState2D::_intersect_point_on_canvas, DEFVAL(32), DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("intersect_ray", "from", "to", "exclude", "collision_layer", "collide_with_bodies", "collide_with_areas"), &PhysicsDirectSpaceState2D::_intersect_ray, DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("intersect_shape", "shape", "max_results"), &PhysicsDirectSpaceState2D::_intersect_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("cast_motion", "shape"), &PhysicsDirectSpaceState2D::_cast_motion);
	ClassDB::bind_method(D_METHOD("collide_shape", "shape", "max_results"), &PhysicsDirectSpaceState2D::_collide_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("get_rest_info", "shape"), &PhysicsDirectSpaceState2D::_get_rest_info);
}

int PhysicsShapeQueryResult2D::get_result_count() const {
	return result.size();
}

RID PhysicsShapeQueryResult2D::get_result_rid(int p_idx) const {
	return result[p_idx].rid;
}

ObjectID PhysicsShapeQueryResult2D::get_result_object_id(int p_idx) const {
	return result[p_idx].collider_id;
}

Object *PhysicsShapeQueryResult2D::get_result_object(int p_idx) const {
	return result[p_idx].collider;
}

int PhysicsShapeQueryResult2D::get_result_object_shape(int p_idx) const {
	return result[p_idx].shape;
}

PhysicsShapeQueryResult2D::PhysicsShapeQueryResult2D() {
}

void PhysicsShapeQueryResult2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_result_count"), &PhysicsShapeQueryResult2D::get_result_count);
	ClassDB::bind_method(D_METHOD("get_result_rid", "idx"), &PhysicsShapeQueryResult2D::get_result_rid);
	ClassDB::bind_method(D_METHOD("get_result_object_id", "idx"), &PhysicsShapeQueryResult2D::get_result_object_id);
	ClassDB::bind_method(D_METHOD("get_result_object", "idx"), &PhysicsShapeQueryResult2D::get_result_object);
	ClassDB::bind_method(D_METHOD("get_result_object_shape", "idx"), &PhysicsShapeQueryResult2D::get_result_object_shape);
}

///////////////////////////////

Vector2 PhysicsTestMotionResult2D::get_motion() const {
	return result.motion;
}

Vector2 PhysicsTestMotionResult2D::get_motion_remainder() const {
	return result.remainder;
}

Vector2 PhysicsTestMotionResult2D::get_collision_point() const {
	return result.collision_point;
}

Vector2 PhysicsTestMotionResult2D::get_collision_normal() const {
	return result.collision_normal;
}

Vector2 PhysicsTestMotionResult2D::get_collider_velocity() const {
	return result.collider_velocity;
}

ObjectID PhysicsTestMotionResult2D::get_collider_id() const {
	return result.collider_id;
}

RID PhysicsTestMotionResult2D::get_collider_rid() const {
	return result.collider;
}

Object *PhysicsTestMotionResult2D::get_collider() const {
	return ObjectDB::get_instance(result.collider_id);
}

int PhysicsTestMotionResult2D::get_collider_shape() const {
	return result.collider_shape;
}

void PhysicsTestMotionResult2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_motion"), &PhysicsTestMotionResult2D::get_motion);
	ClassDB::bind_method(D_METHOD("get_motion_remainder"), &PhysicsTestMotionResult2D::get_motion_remainder);
	ClassDB::bind_method(D_METHOD("get_collision_point"), &PhysicsTestMotionResult2D::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &PhysicsTestMotionResult2D::get_collision_normal);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &PhysicsTestMotionResult2D::get_collider_velocity);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &PhysicsTestMotionResult2D::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &PhysicsTestMotionResult2D::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider"), &PhysicsTestMotionResult2D::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &PhysicsTestMotionResult2D::get_collider_shape);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "motion"), "", "get_motion");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "motion_remainder"), "", "get_motion_remainder");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collision_point"), "", "get_collision_point");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collision_normal"), "", "get_collision_normal");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "collider_velocity"), "", "get_collider_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_id", PROPERTY_HINT_OBJECT_ID), "", "get_collider_id");
	ADD_PROPERTY(PropertyInfo(Variant::_RID, "collider_rid"), "", "get_collider_rid");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "collider"), "", "get_collider");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collider_shape"), "", "get_collider_shape");
}

PhysicsTestMotionResult2D::PhysicsTestMotionResult2D() {
	colliding = false;

	result.collider_shape = 0;
}

///////////////////////////////////////

bool PhysicsServer2D::_body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, float p_margin, const Ref<PhysicsTestMotionResult2D> &p_result) {
	MotionResult *r = nullptr;
	if (p_result.is_valid()) {
		r = p_result->get_result_ptr();
	}
	return body_test_motion(p_body, p_from, p_motion, p_infinite_inertia, p_margin, r);
}

void PhysicsServer2D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("line_shape_create"), &PhysicsServer2D::line_shape_create);
	ClassDB::bind_method(D_METHOD("ray_shape_create"), &PhysicsServer2D::ray_shape_create);
	ClassDB::bind_method(D_METHOD("segment_shape_create"), &PhysicsServer2D::segment_shape_create);
	ClassDB::bind_method(D_METHOD("circle_shape_create"), &PhysicsServer2D::circle_shape_create);
	ClassDB::bind_method(D_METHOD("rectangle_shape_create"), &PhysicsServer2D::rectangle_shape_create);
	ClassDB::bind_method(D_METHOD("capsule_shape_create"), &PhysicsServer2D::capsule_shape_create);
	ClassDB::bind_method(D_METHOD("convex_polygon_shape_create"), &PhysicsServer2D::convex_polygon_shape_create);
	ClassDB::bind_method(D_METHOD("concave_polygon_shape_create"), &PhysicsServer2D::concave_polygon_shape_create);

	ClassDB::bind_method(D_METHOD("shape_set_data", "shape", "data"), &PhysicsServer2D::shape_set_data);

	ClassDB::bind_method(D_METHOD("shape_get_type", "shape"), &PhysicsServer2D::shape_get_type);
	ClassDB::bind_method(D_METHOD("shape_get_data", "shape"), &PhysicsServer2D::shape_get_data);

	ClassDB::bind_method(D_METHOD("space_create"), &PhysicsServer2D::space_create);
	ClassDB::bind_method(D_METHOD("space_set_active", "space", "active"), &PhysicsServer2D::space_set_active);
	ClassDB::bind_method(D_METHOD("space_is_active", "space"), &PhysicsServer2D::space_is_active);
	ClassDB::bind_method(D_METHOD("space_set_param", "space", "param", "value"), &PhysicsServer2D::space_set_param);
	ClassDB::bind_method(D_METHOD("space_get_param", "space", "param"), &PhysicsServer2D::space_get_param);
	ClassDB::bind_method(D_METHOD("space_get_direct_state", "space"), &PhysicsServer2D::space_get_direct_state);

	ClassDB::bind_method(D_METHOD("area_create"), &PhysicsServer2D::area_create);
	ClassDB::bind_method(D_METHOD("area_set_space", "area", "space"), &PhysicsServer2D::area_set_space);
	ClassDB::bind_method(D_METHOD("area_get_space", "area"), &PhysicsServer2D::area_get_space);

	ClassDB::bind_method(D_METHOD("area_set_space_override_mode", "area", "mode"), &PhysicsServer2D::area_set_space_override_mode);
	ClassDB::bind_method(D_METHOD("area_get_space_override_mode", "area"), &PhysicsServer2D::area_get_space_override_mode);

	ClassDB::bind_method(D_METHOD("area_add_shape", "area", "shape", "transform", "disabled"), &PhysicsServer2D::area_add_shape, DEFVAL(Transform2D()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("area_set_shape", "area", "shape_idx", "shape"), &PhysicsServer2D::area_set_shape);
	ClassDB::bind_method(D_METHOD("area_set_shape_transform", "area", "shape_idx", "transform"), &PhysicsServer2D::area_set_shape_transform);
	ClassDB::bind_method(D_METHOD("area_set_shape_disabled", "area", "shape_idx", "disabled"), &PhysicsServer2D::area_set_shape_disabled);

	ClassDB::bind_method(D_METHOD("area_get_shape_count", "area"), &PhysicsServer2D::area_get_shape_count);
	ClassDB::bind_method(D_METHOD("area_get_shape", "area", "shape_idx"), &PhysicsServer2D::area_get_shape);
	ClassDB::bind_method(D_METHOD("area_get_shape_transform", "area", "shape_idx"), &PhysicsServer2D::area_get_shape_transform);

	ClassDB::bind_method(D_METHOD("area_remove_shape", "area", "shape_idx"), &PhysicsServer2D::area_remove_shape);
	ClassDB::bind_method(D_METHOD("area_clear_shapes", "area"), &PhysicsServer2D::area_clear_shapes);

	ClassDB::bind_method(D_METHOD("area_set_collision_layer", "area", "layer"), &PhysicsServer2D::area_set_collision_layer);
	ClassDB::bind_method(D_METHOD("area_set_collision_mask", "area", "mask"), &PhysicsServer2D::area_set_collision_mask);

	ClassDB::bind_method(D_METHOD("area_set_param", "area", "param", "value"), &PhysicsServer2D::area_set_param);
	ClassDB::bind_method(D_METHOD("area_set_transform", "area", "transform"), &PhysicsServer2D::area_set_transform);

	ClassDB::bind_method(D_METHOD("area_get_param", "area", "param"), &PhysicsServer2D::area_get_param);
	ClassDB::bind_method(D_METHOD("area_get_transform", "area"), &PhysicsServer2D::area_get_transform);

	ClassDB::bind_method(D_METHOD("area_attach_object_instance_id", "area", "id"), &PhysicsServer2D::area_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_object_instance_id", "area"), &PhysicsServer2D::area_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("area_attach_canvas_instance_id", "area", "id"), &PhysicsServer2D::area_attach_canvas_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_canvas_instance_id", "area"), &PhysicsServer2D::area_get_canvas_instance_id);

	ClassDB::bind_method(D_METHOD("area_set_monitor_callback", "area", "receiver", "method"), &PhysicsServer2D::area_set_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_area_monitor_callback", "area", "receiver", "method"), &PhysicsServer2D::area_set_area_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_monitorable", "area", "monitorable"), &PhysicsServer2D::area_set_monitorable);

	ClassDB::bind_method(D_METHOD("body_create"), &PhysicsServer2D::body_create);

	ClassDB::bind_method(D_METHOD("body_set_space", "body", "space"), &PhysicsServer2D::body_set_space);
	ClassDB::bind_method(D_METHOD("body_get_space", "body"), &PhysicsServer2D::body_get_space);

	ClassDB::bind_method(D_METHOD("body_set_mode", "body", "mode"), &PhysicsServer2D::body_set_mode);
	ClassDB::bind_method(D_METHOD("body_get_mode", "body"), &PhysicsServer2D::body_get_mode);

	ClassDB::bind_method(D_METHOD("body_add_shape", "body", "shape", "transform", "disabled"), &PhysicsServer2D::body_add_shape, DEFVAL(Transform2D()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("body_set_shape", "body", "shape_idx", "shape"), &PhysicsServer2D::body_set_shape);
	ClassDB::bind_method(D_METHOD("body_set_shape_transform", "body", "shape_idx", "transform"), &PhysicsServer2D::body_set_shape_transform);
	ClassDB::bind_method(D_METHOD("body_set_shape_metadata", "body", "shape_idx", "metadata"), &PhysicsServer2D::body_set_shape_metadata);

	ClassDB::bind_method(D_METHOD("body_get_shape_count", "body"), &PhysicsServer2D::body_get_shape_count);
	ClassDB::bind_method(D_METHOD("body_get_shape", "body", "shape_idx"), &PhysicsServer2D::body_get_shape);
	ClassDB::bind_method(D_METHOD("body_get_shape_transform", "body", "shape_idx"), &PhysicsServer2D::body_get_shape_transform);
	ClassDB::bind_method(D_METHOD("body_get_shape_metadata", "body", "shape_idx"), &PhysicsServer2D::body_get_shape_metadata);

	ClassDB::bind_method(D_METHOD("body_remove_shape", "body", "shape_idx"), &PhysicsServer2D::body_remove_shape);
	ClassDB::bind_method(D_METHOD("body_clear_shapes", "body"), &PhysicsServer2D::body_clear_shapes);

	ClassDB::bind_method(D_METHOD("body_set_shape_disabled", "body", "shape_idx", "disabled"), &PhysicsServer2D::body_set_shape_disabled);
	ClassDB::bind_method(D_METHOD("body_set_shape_as_one_way_collision", "body", "shape_idx", "enable", "margin"), &PhysicsServer2D::body_set_shape_as_one_way_collision);

	ClassDB::bind_method(D_METHOD("body_attach_object_instance_id", "body", "id"), &PhysicsServer2D::body_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_object_instance_id", "body"), &PhysicsServer2D::body_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("body_attach_canvas_instance_id", "body", "id"), &PhysicsServer2D::body_attach_canvas_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_canvas_instance_id", "body"), &PhysicsServer2D::body_get_canvas_instance_id);

	ClassDB::bind_method(D_METHOD("body_set_continuous_collision_detection_mode", "body", "mode"), &PhysicsServer2D::body_set_continuous_collision_detection_mode);
	ClassDB::bind_method(D_METHOD("body_get_continuous_collision_detection_mode", "body"), &PhysicsServer2D::body_get_continuous_collision_detection_mode);

	ClassDB::bind_method(D_METHOD("body_set_collision_layer", "body", "layer"), &PhysicsServer2D::body_set_collision_layer);
	ClassDB::bind_method(D_METHOD("body_get_collision_layer", "body"), &PhysicsServer2D::body_get_collision_layer);

	ClassDB::bind_method(D_METHOD("body_set_collision_mask", "body", "mask"), &PhysicsServer2D::body_set_collision_mask);
	ClassDB::bind_method(D_METHOD("body_get_collision_mask", "body"), &PhysicsServer2D::body_get_collision_mask);

	ClassDB::bind_method(D_METHOD("body_set_param", "body", "param", "value"), &PhysicsServer2D::body_set_param);
	ClassDB::bind_method(D_METHOD("body_get_param", "body", "param"), &PhysicsServer2D::body_get_param);

	ClassDB::bind_method(D_METHOD("body_set_state", "body", "state", "value"), &PhysicsServer2D::body_set_state);
	ClassDB::bind_method(D_METHOD("body_get_state", "body", "state"), &PhysicsServer2D::body_get_state);

	ClassDB::bind_method(D_METHOD("body_apply_central_impulse", "body", "impulse"), &PhysicsServer2D::body_apply_central_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_torque_impulse", "body", "impulse"), &PhysicsServer2D::body_apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_impulse", "body", "impulse", "position"), &PhysicsServer2D::body_apply_impulse, Vector2());
	ClassDB::bind_method(D_METHOD("body_add_central_force", "body", "force"), &PhysicsServer2D::body_add_central_force);
	ClassDB::bind_method(D_METHOD("body_add_force", "body", "force", "position"), &PhysicsServer2D::body_add_force, Vector2());
	ClassDB::bind_method(D_METHOD("body_add_torque", "body", "torque"), &PhysicsServer2D::body_add_torque);
	ClassDB::bind_method(D_METHOD("body_set_axis_velocity", "body", "axis_velocity"), &PhysicsServer2D::body_set_axis_velocity);

	ClassDB::bind_method(D_METHOD("body_add_collision_exception", "body", "excepted_body"), &PhysicsServer2D::body_add_collision_exception);
	ClassDB::bind_method(D_METHOD("body_remove_collision_exception", "body", "excepted_body"), &PhysicsServer2D::body_remove_collision_exception);

	ClassDB::bind_method(D_METHOD("body_set_max_contacts_reported", "body", "amount"), &PhysicsServer2D::body_set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("body_get_max_contacts_reported", "body"), &PhysicsServer2D::body_get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("body_set_omit_force_integration", "body", "enable"), &PhysicsServer2D::body_set_omit_force_integration);
	ClassDB::bind_method(D_METHOD("body_is_omitting_force_integration", "body"), &PhysicsServer2D::body_is_omitting_force_integration);

	ClassDB::bind_method(D_METHOD("body_set_force_integration_callback", "body", "receiver", "method", "userdata"), &PhysicsServer2D::body_set_force_integration_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_test_motion", "body", "from", "motion", "infinite_inertia", "margin", "result"), &PhysicsServer2D::_body_test_motion, DEFVAL(0.08), DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_get_direct_state", "body"), &PhysicsServer2D::body_get_direct_state);

	/* JOINT API */

	ClassDB::bind_method(D_METHOD("joint_set_param", "joint", "param", "value"), &PhysicsServer2D::joint_set_param);
	ClassDB::bind_method(D_METHOD("joint_get_param", "joint", "param"), &PhysicsServer2D::joint_get_param);

	ClassDB::bind_method(D_METHOD("pin_joint_create", "anchor", "body_a", "body_b"), &PhysicsServer2D::pin_joint_create, DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("groove_joint_create", "groove1_a", "groove2_a", "anchor_b", "body_a", "body_b"), &PhysicsServer2D::groove_joint_create, DEFVAL(RID()), DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("damped_spring_joint_create", "anchor_a", "anchor_b", "body_a", "body_b"), &PhysicsServer2D::damped_spring_joint_create, DEFVAL(RID()));

	ClassDB::bind_method(D_METHOD("damped_spring_joint_set_param", "joint", "param", "value"), &PhysicsServer2D::damped_spring_joint_set_param);
	ClassDB::bind_method(D_METHOD("damped_spring_joint_get_param", "joint", "param"), &PhysicsServer2D::damped_spring_joint_get_param);

	ClassDB::bind_method(D_METHOD("joint_get_type", "joint"), &PhysicsServer2D::joint_get_type);

	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &PhysicsServer2D::free);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &PhysicsServer2D::set_active);

	ClassDB::bind_method(D_METHOD("get_process_info", "process_info"), &PhysicsServer2D::get_process_info);

	BIND_ENUM_CONSTANT(SPACE_PARAM_CONTACT_RECYCLE_RADIUS);
	BIND_ENUM_CONSTANT(SPACE_PARAM_CONTACT_MAX_SEPARATION);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_TIME_TO_SLEEP);
	BIND_ENUM_CONSTANT(SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS);
	BIND_ENUM_CONSTANT(SPACE_PARAM_TEST_MOTION_MIN_CONTACT_DEPTH);

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

	BIND_ENUM_CONSTANT(DAMPED_SPRING_REST_LENGTH);
	BIND_ENUM_CONSTANT(DAMPED_SPRING_STIFFNESS);
	BIND_ENUM_CONSTANT(DAMPED_SPRING_DAMPING);

	BIND_ENUM_CONSTANT(CCD_MODE_DISABLED);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_RAY);
	BIND_ENUM_CONSTANT(CCD_MODE_CAST_SHAPE);

	BIND_ENUM_CONSTANT(AREA_BODY_ADDED);
	BIND_ENUM_CONSTANT(AREA_BODY_REMOVED);

	BIND_ENUM_CONSTANT(INFO_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(INFO_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(INFO_ISLAND_COUNT);
}

PhysicsServer2D::PhysicsServer2D() {
	singleton = this;
}

PhysicsServer2D::~PhysicsServer2D() {
	singleton = nullptr;
}

Vector<PhysicsServer2DManager::ClassInfo> PhysicsServer2DManager::physics_2d_servers;
int PhysicsServer2DManager::default_server_id = -1;
int PhysicsServer2DManager::default_server_priority = -1;
const String PhysicsServer2DManager::setting_property_name("physics/2d/physics_engine");

void PhysicsServer2DManager::on_servers_changed() {
	String physics_servers("DEFAULT");
	for (int i = get_servers_count() - 1; 0 <= i; --i) {
		physics_servers += "," + get_server_name(i);
	}
	ProjectSettings::get_singleton()->set_custom_property_info(setting_property_name, PropertyInfo(Variant::STRING, setting_property_name, PROPERTY_HINT_ENUM, physics_servers));
}

void PhysicsServer2DManager::register_server(const String &p_name, CreatePhysicsServer2DCallback p_creat_callback) {
	ERR_FAIL_COND(!p_creat_callback);
	ERR_FAIL_COND(find_server_id(p_name) != -1);
	physics_2d_servers.push_back(ClassInfo(p_name, p_creat_callback));
	on_servers_changed();
}

void PhysicsServer2DManager::set_default_server(const String &p_name, int p_priority) {
	const int id = find_server_id(p_name);
	ERR_FAIL_COND(id == -1); // Not found
	if (default_server_priority < p_priority) {
		default_server_id = id;
		default_server_priority = p_priority;
	}
}

int PhysicsServer2DManager::find_server_id(const String &p_name) {
	for (int i = physics_2d_servers.size() - 1; 0 <= i; --i) {
		if (p_name == physics_2d_servers[i].name) {
			return i;
		}
	}
	return -1;
}

int PhysicsServer2DManager::get_servers_count() {
	return physics_2d_servers.size();
}

String PhysicsServer2DManager::get_server_name(int p_id) {
	ERR_FAIL_INDEX_V(p_id, get_servers_count(), "");
	return physics_2d_servers[p_id].name;
}

PhysicsServer2D *PhysicsServer2DManager::new_default_server() {
	ERR_FAIL_COND_V(default_server_id == -1, nullptr);
	return physics_2d_servers[default_server_id].create_callback();
}

PhysicsServer2D *PhysicsServer2DManager::new_server(const String &p_name) {
	int id = find_server_id(p_name);
	if (id == -1) {
		return nullptr;
	} else {
		return physics_2d_servers[id].create_callback();
	}
}
