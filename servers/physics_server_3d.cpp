/*************************************************************************/
/*  physics_server_3d.cpp                                                */
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

#include "physics_server_3d.h"

#include "core/config/project_settings.h"
#include "core/string/print_string.h"

PhysicsServer3D *PhysicsServer3D::singleton = nullptr;

void PhysicsDirectBodyState3D::integrate_forces() {
	real_t step = get_step();
	Vector3 lv = get_linear_velocity();
	lv += get_total_gravity() * step;

	Vector3 av = get_angular_velocity();

	real_t linear_damp = 1.0 - step * get_total_linear_damp();

	if (linear_damp < 0) { // reached zero in the given time
		linear_damp = 0;
	}

	real_t angular_damp = 1.0 - step * get_total_angular_damp();

	if (angular_damp < 0) { // reached zero in the given time
		angular_damp = 0;
	}

	lv *= linear_damp;
	av *= angular_damp;

	set_linear_velocity(lv);
	set_angular_velocity(av);
}

Object *PhysicsDirectBodyState3D::get_contact_collider_object(int p_contact_idx) const {
	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance(objid);
	return obj;
}

PhysicsServer3D *PhysicsServer3D::get_singleton() {
	return singleton;
}

void PhysicsDirectBodyState3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_total_gravity"), &PhysicsDirectBodyState3D::get_total_gravity);
	ClassDB::bind_method(D_METHOD("get_total_linear_damp"), &PhysicsDirectBodyState3D::get_total_linear_damp);
	ClassDB::bind_method(D_METHOD("get_total_angular_damp"), &PhysicsDirectBodyState3D::get_total_angular_damp);

	ClassDB::bind_method(D_METHOD("get_center_of_mass"), &PhysicsDirectBodyState3D::get_center_of_mass);
	ClassDB::bind_method(D_METHOD("get_principal_inertia_axes"), &PhysicsDirectBodyState3D::get_principal_inertia_axes);

	ClassDB::bind_method(D_METHOD("get_inverse_mass"), &PhysicsDirectBodyState3D::get_inverse_mass);
	ClassDB::bind_method(D_METHOD("get_inverse_inertia"), &PhysicsDirectBodyState3D::get_inverse_inertia);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &PhysicsDirectBodyState3D::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &PhysicsDirectBodyState3D::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &PhysicsDirectBodyState3D::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &PhysicsDirectBodyState3D::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsDirectBodyState3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsDirectBodyState3D::get_transform);

	ClassDB::bind_method(D_METHOD("add_central_force", "force"), &PhysicsDirectBodyState3D::add_central_force, Vector3());
	ClassDB::bind_method(D_METHOD("add_force", "force", "position"), &PhysicsDirectBodyState3D::add_force, Vector3());
	ClassDB::bind_method(D_METHOD("add_torque", "torque"), &PhysicsDirectBodyState3D::add_torque);
	ClassDB::bind_method(D_METHOD("apply_central_impulse", "impulse"), &PhysicsDirectBodyState3D::apply_central_impulse, Vector3());
	ClassDB::bind_method(D_METHOD("apply_impulse", "impulse", "position"), &PhysicsDirectBodyState3D::apply_impulse, Vector3());
	ClassDB::bind_method(D_METHOD("apply_torque_impulse", "impulse"), &PhysicsDirectBodyState3D::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("set_sleep_state", "enabled"), &PhysicsDirectBodyState3D::set_sleep_state);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &PhysicsDirectBodyState3D::is_sleeping);

	ClassDB::bind_method(D_METHOD("get_contact_count"), &PhysicsDirectBodyState3D::get_contact_count);

	ClassDB::bind_method(D_METHOD("get_contact_local_position", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_local_position);
	ClassDB::bind_method(D_METHOD("get_contact_local_normal", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_local_normal);
	ClassDB::bind_method(D_METHOD("get_contact_impulse", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_impulse);
	ClassDB::bind_method(D_METHOD("get_contact_local_shape", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_local_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider);
	ClassDB::bind_method(D_METHOD("get_contact_collider_position", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider_id", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_id);
	ClassDB::bind_method(D_METHOD("get_contact_collider_object", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_object);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider_velocity_at_position", "contact_idx"), &PhysicsDirectBodyState3D::get_contact_collider_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_step"), &PhysicsDirectBodyState3D::get_step);
	ClassDB::bind_method(D_METHOD("integrate_forces"), &PhysicsDirectBodyState3D::integrate_forces);
	ClassDB::bind_method(D_METHOD("get_space_state"), &PhysicsDirectBodyState3D::get_space_state);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "step"), "", "get_step");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "inverse_mass"), "", "get_inverse_mass");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_angular_damp"), "", "get_total_angular_damp");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "total_linear_damp"), "", "get_total_linear_damp");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "inverse_inertia"), "", "get_inverse_inertia");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "total_gravity"), "", "get_total_gravity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "center_of_mass"), "", "get_center_of_mass");
	ADD_PROPERTY(PropertyInfo(Variant::BASIS, "principal_inertia_axes"), "", "get_principal_inertia_axes");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "angular_velocity"), "set_angular_velocity", "get_angular_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "linear_velocity"), "set_linear_velocity", "get_linear_velocity");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sleeping"), "set_sleep_state", "is_sleeping");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform"), "set_transform", "get_transform");
}

PhysicsDirectBodyState3D::PhysicsDirectBodyState3D() {}

///////////////////////////////////////////////////////

void PhysicsShapeQueryParameters3D::set_shape(const RES &p_shape_ref) {
	ERR_FAIL_COND(p_shape_ref.is_null());
	shape_ref = p_shape_ref;
	shape = p_shape_ref->get_rid();
}

RES PhysicsShapeQueryParameters3D::get_shape() const {
	return shape_ref;
}

void PhysicsShapeQueryParameters3D::set_shape_rid(const RID &p_shape) {
	if (shape != p_shape) {
		shape_ref = RES();
		shape = p_shape;
	}
}

RID PhysicsShapeQueryParameters3D::get_shape_rid() const {
	return shape;
}

void PhysicsShapeQueryParameters3D::set_transform(const Transform &p_transform) {
	transform = p_transform;
}

Transform PhysicsShapeQueryParameters3D::get_transform() const {
	return transform;
}

void PhysicsShapeQueryParameters3D::set_margin(real_t p_margin) {
	margin = p_margin;
}

real_t PhysicsShapeQueryParameters3D::get_margin() const {
	return margin;
}

void PhysicsShapeQueryParameters3D::set_collision_mask(uint32_t p_collision_mask) {
	collision_mask = p_collision_mask;
}

uint32_t PhysicsShapeQueryParameters3D::get_collision_mask() const {
	return collision_mask;
}

void PhysicsShapeQueryParameters3D::set_exclude(const Vector<RID> &p_exclude) {
	exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++) {
		exclude.insert(p_exclude[i]);
	}
}

Vector<RID> PhysicsShapeQueryParameters3D::get_exclude() const {
	Vector<RID> ret;
	ret.resize(exclude.size());
	int idx = 0;
	for (Set<RID>::Element *E = exclude.front(); E; E = E->next()) {
		ret.write[idx] = E->get();
	}
	return ret;
}

void PhysicsShapeQueryParameters3D::set_collide_with_bodies(bool p_enable) {
	collide_with_bodies = p_enable;
}

bool PhysicsShapeQueryParameters3D::is_collide_with_bodies_enabled() const {
	return collide_with_bodies;
}

void PhysicsShapeQueryParameters3D::set_collide_with_areas(bool p_enable) {
	collide_with_areas = p_enable;
}

bool PhysicsShapeQueryParameters3D::is_collide_with_areas_enabled() const {
	return collide_with_areas;
}

void PhysicsShapeQueryParameters3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &PhysicsShapeQueryParameters3D::set_shape);
	ClassDB::bind_method(D_METHOD("get_shape"), &PhysicsShapeQueryParameters3D::get_shape);
	ClassDB::bind_method(D_METHOD("set_shape_rid", "shape"), &PhysicsShapeQueryParameters3D::set_shape_rid);
	ClassDB::bind_method(D_METHOD("get_shape_rid"), &PhysicsShapeQueryParameters3D::get_shape_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsShapeQueryParameters3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsShapeQueryParameters3D::get_transform);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &PhysicsShapeQueryParameters3D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &PhysicsShapeQueryParameters3D::get_margin);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &PhysicsShapeQueryParameters3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &PhysicsShapeQueryParameters3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &PhysicsShapeQueryParameters3D::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &PhysicsShapeQueryParameters3D::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &PhysicsShapeQueryParameters3D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &PhysicsShapeQueryParameters3D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &PhysicsShapeQueryParameters3D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &PhysicsShapeQueryParameters3D::is_collide_with_areas_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_NONE, itos(Variant::RID) + ":"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "shape", PROPERTY_HINT_RESOURCE_TYPE, "Shape3D"), "set_shape", "get_shape");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "shape_rid"), "set_shape_rid", "get_shape_rid");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
}

PhysicsShapeQueryParameters3D::PhysicsShapeQueryParameters3D() {
	margin = 0;
	collision_mask = 0x7FFFFFFF;
	collide_with_bodies = true;
	collide_with_areas = false;
}

/////////////////////////////////////

Dictionary PhysicsDirectSpaceState3D::_intersect_ray(const Vector3 &p_from, const Vector3 &p_to, const Vector<RID> &p_exclude, uint32_t p_collision_mask, bool p_collide_with_bodies, bool p_collide_with_areas) {
	RayResult inters;
	Set<RID> exclude;
	for (int i = 0; i < p_exclude.size(); i++) {
		exclude.insert(p_exclude[i]);
	}

	bool res = intersect_ray(p_from, p_to, inters, exclude, p_collision_mask, p_collide_with_bodies, p_collide_with_areas);

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

	return d;
}

Array PhysicsDirectSpaceState3D::_intersect_shape(const Ref<PhysicsShapeQueryParameters3D> &p_shape_query, int p_max_results) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Array());

	Vector<ShapeResult> sr;
	sr.resize(p_max_results);
	int rc = intersect_shape(p_shape_query->shape, p_shape_query->transform, p_shape_query->margin, sr.ptrw(), sr.size(), p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
	Array ret;
	ret.resize(rc);
	for (int i = 0; i < rc; i++) {
		Dictionary d;
		d["rid"] = sr[i].rid;
		d["collider_id"] = sr[i].collider_id;
		d["collider"] = sr[i].collider;
		d["shape"] = sr[i].shape;
		ret[i] = d;
	}

	return ret;
}

Array PhysicsDirectSpaceState3D::_cast_motion(const Ref<PhysicsShapeQueryParameters3D> &p_shape_query, const Vector3 &p_motion) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Array());

	real_t closest_safe = 1.0f, closest_unsafe = 1.0f;
	bool res = cast_motion(p_shape_query->shape, p_shape_query->transform, p_motion, p_shape_query->margin, closest_safe, closest_unsafe, p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
	if (!res) {
		return Array();
	}
	Array ret;
	ret.resize(2);
	ret[0] = closest_safe;
	ret[1] = closest_unsafe;
	return ret;
}

Array PhysicsDirectSpaceState3D::_collide_shape(const Ref<PhysicsShapeQueryParameters3D> &p_shape_query, int p_max_results) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Array());

	Vector<Vector3> ret;
	ret.resize(p_max_results * 2);
	int rc = 0;
	bool res = collide_shape(p_shape_query->shape, p_shape_query->transform, p_shape_query->margin, ret.ptrw(), p_max_results, rc, p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
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

Dictionary PhysicsDirectSpaceState3D::_get_rest_info(const Ref<PhysicsShapeQueryParameters3D> &p_shape_query) {
	ERR_FAIL_COND_V(!p_shape_query.is_valid(), Dictionary());

	ShapeRestInfo sri;

	bool res = rest_info(p_shape_query->shape, p_shape_query->transform, p_shape_query->margin, &sri, p_shape_query->exclude, p_shape_query->collision_mask, p_shape_query->collide_with_bodies, p_shape_query->collide_with_areas);
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

	return r;
}

PhysicsDirectSpaceState3D::PhysicsDirectSpaceState3D() {
}

void PhysicsDirectSpaceState3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("intersect_ray", "from", "to", "exclude", "collision_mask", "collide_with_bodies", "collide_with_areas"), &PhysicsDirectSpaceState3D::_intersect_ray, DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("intersect_shape", "shape", "max_results"), &PhysicsDirectSpaceState3D::_intersect_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("cast_motion", "shape", "motion"), &PhysicsDirectSpaceState3D::_cast_motion);
	ClassDB::bind_method(D_METHOD("collide_shape", "shape", "max_results"), &PhysicsDirectSpaceState3D::_collide_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("get_rest_info", "shape"), &PhysicsDirectSpaceState3D::_get_rest_info);
}

int PhysicsShapeQueryResult3D::get_result_count() const {
	return result.size();
}

RID PhysicsShapeQueryResult3D::get_result_rid(int p_idx) const {
	return result[p_idx].rid;
}

ObjectID PhysicsShapeQueryResult3D::get_result_object_id(int p_idx) const {
	return result[p_idx].collider_id;
}

Object *PhysicsShapeQueryResult3D::get_result_object(int p_idx) const {
	return result[p_idx].collider;
}

int PhysicsShapeQueryResult3D::get_result_object_shape(int p_idx) const {
	return result[p_idx].shape;
}

PhysicsShapeQueryResult3D::PhysicsShapeQueryResult3D() {
}

void PhysicsShapeQueryResult3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_result_count"), &PhysicsShapeQueryResult3D::get_result_count);
	ClassDB::bind_method(D_METHOD("get_result_rid", "idx"), &PhysicsShapeQueryResult3D::get_result_rid);
	ClassDB::bind_method(D_METHOD("get_result_object_id", "idx"), &PhysicsShapeQueryResult3D::get_result_object_id);
	ClassDB::bind_method(D_METHOD("get_result_object", "idx"), &PhysicsShapeQueryResult3D::get_result_object);
	ClassDB::bind_method(D_METHOD("get_result_object_shape", "idx"), &PhysicsShapeQueryResult3D::get_result_object_shape);
}

///////////////////////////////////////

RID PhysicsServer3D::shape_create(ShapeType p_shape) {
	switch (p_shape) {
		case SHAPE_PLANE:
			return plane_shape_create();
		case SHAPE_RAY:
			return ray_shape_create();
		case SHAPE_SPHERE:
			return sphere_shape_create();
		case SHAPE_BOX:
			return box_shape_create();
		case SHAPE_CAPSULE:
			return capsule_shape_create();
		case SHAPE_CYLINDER:
			return cylinder_shape_create();
		case SHAPE_CONVEX_POLYGON:
			return convex_polygon_shape_create();
		case SHAPE_CONCAVE_POLYGON:
			return concave_polygon_shape_create();
		case SHAPE_HEIGHTMAP:
			return heightmap_shape_create();
		case SHAPE_CUSTOM:
			return custom_shape_create();
		default:
			return RID();
	}
}

void PhysicsServer3D::_bind_methods() {
#ifndef _3D_DISABLED

	ClassDB::bind_method(D_METHOD("plane_shape_create"), &PhysicsServer3D::plane_shape_create);
	ClassDB::bind_method(D_METHOD("ray_shape_create"), &PhysicsServer3D::ray_shape_create);
	ClassDB::bind_method(D_METHOD("sphere_shape_create"), &PhysicsServer3D::sphere_shape_create);
	ClassDB::bind_method(D_METHOD("box_shape_create"), &PhysicsServer3D::box_shape_create);
	ClassDB::bind_method(D_METHOD("capsule_shape_create"), &PhysicsServer3D::capsule_shape_create);
	ClassDB::bind_method(D_METHOD("cylinder_shape_create"), &PhysicsServer3D::cylinder_shape_create);
	ClassDB::bind_method(D_METHOD("convex_polygon_shape_create"), &PhysicsServer3D::convex_polygon_shape_create);
	ClassDB::bind_method(D_METHOD("concave_polygon_shape_create"), &PhysicsServer3D::concave_polygon_shape_create);
	ClassDB::bind_method(D_METHOD("heightmap_shape_create"), &PhysicsServer3D::heightmap_shape_create);
	ClassDB::bind_method(D_METHOD("custom_shape_create"), &PhysicsServer3D::custom_shape_create);

	ClassDB::bind_method(D_METHOD("shape_set_data", "shape", "data"), &PhysicsServer3D::shape_set_data);

	ClassDB::bind_method(D_METHOD("shape_get_type", "shape"), &PhysicsServer3D::shape_get_type);
	ClassDB::bind_method(D_METHOD("shape_get_data", "shape"), &PhysicsServer3D::shape_get_data);

	ClassDB::bind_method(D_METHOD("space_create"), &PhysicsServer3D::space_create);
	ClassDB::bind_method(D_METHOD("space_set_active", "space", "active"), &PhysicsServer3D::space_set_active);
	ClassDB::bind_method(D_METHOD("space_is_active", "space"), &PhysicsServer3D::space_is_active);
	ClassDB::bind_method(D_METHOD("space_set_param", "space", "param", "value"), &PhysicsServer3D::space_set_param);
	ClassDB::bind_method(D_METHOD("space_get_param", "space", "param"), &PhysicsServer3D::space_get_param);
	ClassDB::bind_method(D_METHOD("space_get_direct_state", "space"), &PhysicsServer3D::space_get_direct_state);

	ClassDB::bind_method(D_METHOD("area_create"), &PhysicsServer3D::area_create);
	ClassDB::bind_method(D_METHOD("area_set_space", "area", "space"), &PhysicsServer3D::area_set_space);
	ClassDB::bind_method(D_METHOD("area_get_space", "area"), &PhysicsServer3D::area_get_space);

	ClassDB::bind_method(D_METHOD("area_set_space_override_mode", "area", "mode"), &PhysicsServer3D::area_set_space_override_mode);
	ClassDB::bind_method(D_METHOD("area_get_space_override_mode", "area"), &PhysicsServer3D::area_get_space_override_mode);

	ClassDB::bind_method(D_METHOD("area_add_shape", "area", "shape", "transform", "disabled"), &PhysicsServer3D::area_add_shape, DEFVAL(Transform()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("area_set_shape", "area", "shape_idx", "shape"), &PhysicsServer3D::area_set_shape);
	ClassDB::bind_method(D_METHOD("area_set_shape_transform", "area", "shape_idx", "transform"), &PhysicsServer3D::area_set_shape_transform);
	ClassDB::bind_method(D_METHOD("area_set_shape_disabled", "area", "shape_idx", "disabled"), &PhysicsServer3D::area_set_shape_disabled);

	ClassDB::bind_method(D_METHOD("area_get_shape_count", "area"), &PhysicsServer3D::area_get_shape_count);
	ClassDB::bind_method(D_METHOD("area_get_shape", "area", "shape_idx"), &PhysicsServer3D::area_get_shape);
	ClassDB::bind_method(D_METHOD("area_get_shape_transform", "area", "shape_idx"), &PhysicsServer3D::area_get_shape_transform);

	ClassDB::bind_method(D_METHOD("area_remove_shape", "area", "shape_idx"), &PhysicsServer3D::area_remove_shape);
	ClassDB::bind_method(D_METHOD("area_clear_shapes", "area"), &PhysicsServer3D::area_clear_shapes);

	ClassDB::bind_method(D_METHOD("area_set_collision_layer", "area", "layer"), &PhysicsServer3D::area_set_collision_layer);
	ClassDB::bind_method(D_METHOD("area_set_collision_mask", "area", "mask"), &PhysicsServer3D::area_set_collision_mask);

	ClassDB::bind_method(D_METHOD("area_set_param", "area", "param", "value"), &PhysicsServer3D::area_set_param);
	ClassDB::bind_method(D_METHOD("area_set_transform", "area", "transform"), &PhysicsServer3D::area_set_transform);

	ClassDB::bind_method(D_METHOD("area_get_param", "area", "param"), &PhysicsServer3D::area_get_param);
	ClassDB::bind_method(D_METHOD("area_get_transform", "area"), &PhysicsServer3D::area_get_transform);

	ClassDB::bind_method(D_METHOD("area_attach_object_instance_id", "area", "id"), &PhysicsServer3D::area_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_object_instance_id", "area"), &PhysicsServer3D::area_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("area_set_monitor_callback", "area", "receiver", "method"), &PhysicsServer3D::area_set_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_area_monitor_callback", "area", "receiver", "method"), &PhysicsServer3D::area_set_area_monitor_callback);
	ClassDB::bind_method(D_METHOD("area_set_monitorable", "area", "monitorable"), &PhysicsServer3D::area_set_monitorable);

	ClassDB::bind_method(D_METHOD("area_set_ray_pickable", "area", "enable"), &PhysicsServer3D::area_set_ray_pickable);

	ClassDB::bind_method(D_METHOD("body_create"), &PhysicsServer3D::body_create);

	ClassDB::bind_method(D_METHOD("body_set_space", "body", "space"), &PhysicsServer3D::body_set_space);
	ClassDB::bind_method(D_METHOD("body_get_space", "body"), &PhysicsServer3D::body_get_space);

	ClassDB::bind_method(D_METHOD("body_set_mode", "body", "mode"), &PhysicsServer3D::body_set_mode);
	ClassDB::bind_method(D_METHOD("body_get_mode", "body"), &PhysicsServer3D::body_get_mode);

	ClassDB::bind_method(D_METHOD("body_set_collision_layer", "body", "layer"), &PhysicsServer3D::body_set_collision_layer);
	ClassDB::bind_method(D_METHOD("body_get_collision_layer", "body"), &PhysicsServer3D::body_get_collision_layer);

	ClassDB::bind_method(D_METHOD("body_set_collision_mask", "body", "mask"), &PhysicsServer3D::body_set_collision_mask);
	ClassDB::bind_method(D_METHOD("body_get_collision_mask", "body"), &PhysicsServer3D::body_get_collision_mask);

	ClassDB::bind_method(D_METHOD("body_add_shape", "body", "shape", "transform", "disabled"), &PhysicsServer3D::body_add_shape, DEFVAL(Transform()), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("body_set_shape", "body", "shape_idx", "shape"), &PhysicsServer3D::body_set_shape);
	ClassDB::bind_method(D_METHOD("body_set_shape_transform", "body", "shape_idx", "transform"), &PhysicsServer3D::body_set_shape_transform);
	ClassDB::bind_method(D_METHOD("body_set_shape_disabled", "body", "shape_idx", "disabled"), &PhysicsServer3D::body_set_shape_disabled);

	ClassDB::bind_method(D_METHOD("body_get_shape_count", "body"), &PhysicsServer3D::body_get_shape_count);
	ClassDB::bind_method(D_METHOD("body_get_shape", "body", "shape_idx"), &PhysicsServer3D::body_get_shape);
	ClassDB::bind_method(D_METHOD("body_get_shape_transform", "body", "shape_idx"), &PhysicsServer3D::body_get_shape_transform);

	ClassDB::bind_method(D_METHOD("body_remove_shape", "body", "shape_idx"), &PhysicsServer3D::body_remove_shape);
	ClassDB::bind_method(D_METHOD("body_clear_shapes", "body"), &PhysicsServer3D::body_clear_shapes);

	ClassDB::bind_method(D_METHOD("body_attach_object_instance_id", "body", "id"), &PhysicsServer3D::body_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_object_instance_id", "body"), &PhysicsServer3D::body_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("body_set_enable_continuous_collision_detection", "body", "enable"), &PhysicsServer3D::body_set_enable_continuous_collision_detection);
	ClassDB::bind_method(D_METHOD("body_is_continuous_collision_detection_enabled", "body"), &PhysicsServer3D::body_is_continuous_collision_detection_enabled);

	ClassDB::bind_method(D_METHOD("body_set_param", "body", "param", "value"), &PhysicsServer3D::body_set_param);
	ClassDB::bind_method(D_METHOD("body_get_param", "body", "param"), &PhysicsServer3D::body_get_param);

	ClassDB::bind_method(D_METHOD("body_set_kinematic_safe_margin", "body", "margin"), &PhysicsServer3D::body_set_kinematic_safe_margin);
	ClassDB::bind_method(D_METHOD("body_get_kinematic_safe_margin", "body"), &PhysicsServer3D::body_get_kinematic_safe_margin);

	ClassDB::bind_method(D_METHOD("body_set_state", "body", "state", "value"), &PhysicsServer3D::body_set_state);
	ClassDB::bind_method(D_METHOD("body_get_state", "body", "state"), &PhysicsServer3D::body_get_state);

	ClassDB::bind_method(D_METHOD("body_add_central_force", "body", "force"), &PhysicsServer3D::body_add_central_force);
	ClassDB::bind_method(D_METHOD("body_add_force", "body", "force", "position"), &PhysicsServer3D::body_add_force, Vector3());
	ClassDB::bind_method(D_METHOD("body_add_torque", "body", "torque"), &PhysicsServer3D::body_add_torque);

	ClassDB::bind_method(D_METHOD("body_apply_central_impulse", "body", "impulse"), &PhysicsServer3D::body_apply_central_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_impulse", "body", "impulse", "position"), &PhysicsServer3D::body_apply_impulse, Vector3());
	ClassDB::bind_method(D_METHOD("body_apply_torque_impulse", "body", "impulse"), &PhysicsServer3D::body_apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("body_set_axis_velocity", "body", "axis_velocity"), &PhysicsServer3D::body_set_axis_velocity);

	ClassDB::bind_method(D_METHOD("body_set_axis_lock", "body", "axis", "lock"), &PhysicsServer3D::body_set_axis_lock);
	ClassDB::bind_method(D_METHOD("body_is_axis_locked", "body", "axis"), &PhysicsServer3D::body_is_axis_locked);

	ClassDB::bind_method(D_METHOD("body_add_collision_exception", "body", "excepted_body"), &PhysicsServer3D::body_add_collision_exception);
	ClassDB::bind_method(D_METHOD("body_remove_collision_exception", "body", "excepted_body"), &PhysicsServer3D::body_remove_collision_exception);

	ClassDB::bind_method(D_METHOD("body_set_max_contacts_reported", "body", "amount"), &PhysicsServer3D::body_set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("body_get_max_contacts_reported", "body"), &PhysicsServer3D::body_get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("body_set_omit_force_integration", "body", "enable"), &PhysicsServer3D::body_set_omit_force_integration);
	ClassDB::bind_method(D_METHOD("body_is_omitting_force_integration", "body"), &PhysicsServer3D::body_is_omitting_force_integration);

	ClassDB::bind_method(D_METHOD("body_set_force_integration_callback", "body", "callable", "userdata"), &PhysicsServer3D::body_set_force_integration_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_set_ray_pickable", "body", "enable"), &PhysicsServer3D::body_set_ray_pickable);

	ClassDB::bind_method(D_METHOD("body_get_direct_state", "body"), &PhysicsServer3D::body_get_direct_state);

	/* SOFT BODY API */

	ClassDB::bind_method(D_METHOD("soft_body_get_bounds", "body"), &PhysicsServer3D::soft_body_get_bounds);

	/* JOINT API */

	ClassDB::bind_method(D_METHOD("joint_create"), &PhysicsServer3D::joint_create);
	ClassDB::bind_method(D_METHOD("joint_clear", "joint"), &PhysicsServer3D::joint_clear);

	BIND_ENUM_CONSTANT(JOINT_TYPE_PIN);
	BIND_ENUM_CONSTANT(JOINT_TYPE_HINGE);
	BIND_ENUM_CONSTANT(JOINT_TYPE_SLIDER);
	BIND_ENUM_CONSTANT(JOINT_TYPE_CONE_TWIST);
	BIND_ENUM_CONSTANT(JOINT_TYPE_6DOF);
	BIND_ENUM_CONSTANT(JOINT_TYPE_MAX);

	ClassDB::bind_method(D_METHOD("joint_make_pin", "joint", "body_A", "local_A", "body_B", "local_B"), &PhysicsServer3D::joint_make_pin);
	ClassDB::bind_method(D_METHOD("pin_joint_set_param", "joint", "param", "value"), &PhysicsServer3D::pin_joint_set_param);
	ClassDB::bind_method(D_METHOD("pin_joint_get_param", "joint", "param"), &PhysicsServer3D::pin_joint_get_param);

	ClassDB::bind_method(D_METHOD("pin_joint_set_local_a", "joint", "local_A"), &PhysicsServer3D::pin_joint_set_local_a);
	ClassDB::bind_method(D_METHOD("pin_joint_get_local_a", "joint"), &PhysicsServer3D::pin_joint_get_local_a);

	ClassDB::bind_method(D_METHOD("pin_joint_set_local_b", "joint", "local_B"), &PhysicsServer3D::pin_joint_set_local_b);
	ClassDB::bind_method(D_METHOD("pin_joint_get_local_b", "joint"), &PhysicsServer3D::pin_joint_get_local_b);

	BIND_ENUM_CONSTANT(PIN_JOINT_BIAS);
	BIND_ENUM_CONSTANT(PIN_JOINT_DAMPING);
	BIND_ENUM_CONSTANT(PIN_JOINT_IMPULSE_CLAMP);

	BIND_ENUM_CONSTANT(HINGE_JOINT_BIAS);
	BIND_ENUM_CONSTANT(HINGE_JOINT_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(HINGE_JOINT_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(HINGE_JOINT_LIMIT_BIAS);
	BIND_ENUM_CONSTANT(HINGE_JOINT_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(HINGE_JOINT_LIMIT_RELAXATION);
	BIND_ENUM_CONSTANT(HINGE_JOINT_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(HINGE_JOINT_MOTOR_MAX_IMPULSE);

	BIND_ENUM_CONSTANT(HINGE_JOINT_FLAG_USE_LIMIT);
	BIND_ENUM_CONSTANT(HINGE_JOINT_FLAG_ENABLE_MOTOR);

	ClassDB::bind_method(D_METHOD("joint_make_hinge", "joint", "body_A", "hinge_A", "body_B", "hinge_B"), &PhysicsServer3D::joint_make_hinge);

	ClassDB::bind_method(D_METHOD("hinge_joint_set_param", "joint", "param", "value"), &PhysicsServer3D::hinge_joint_set_param);
	ClassDB::bind_method(D_METHOD("hinge_joint_get_param", "joint", "param"), &PhysicsServer3D::hinge_joint_get_param);

	ClassDB::bind_method(D_METHOD("hinge_joint_set_flag", "joint", "flag", "enabled"), &PhysicsServer3D::hinge_joint_set_flag);
	ClassDB::bind_method(D_METHOD("hinge_joint_get_flag", "joint", "flag"), &PhysicsServer3D::hinge_joint_get_flag);

	ClassDB::bind_method(D_METHOD("joint_make_slider", "joint", "body_A", "local_ref_A", "body_B", "local_ref_B"), &PhysicsServer3D::joint_make_slider);

	ClassDB::bind_method(D_METHOD("slider_joint_set_param", "joint", "param", "value"), &PhysicsServer3D::slider_joint_set_param);
	ClassDB::bind_method(D_METHOD("slider_joint_get_param", "joint", "param"), &PhysicsServer3D::slider_joint_get_param);

	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_LIMIT_RESTITUTION);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_LIMIT_DAMPING);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_MOTION_SOFTNESS);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_MOTION_RESTITUTION);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_MOTION_DAMPING);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_ORTHOGONAL_SOFTNESS);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_ORTHOGONAL_RESTITUTION);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_LINEAR_ORTHOGONAL_DAMPING);

	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_LIMIT_UPPER);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_LIMIT_LOWER);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_LIMIT_RESTITUTION);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_LIMIT_DAMPING);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_MOTION_SOFTNESS);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_MOTION_RESTITUTION);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_MOTION_DAMPING);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_ORTHOGONAL_SOFTNESS);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_ORTHOGONAL_RESTITUTION);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_ANGULAR_ORTHOGONAL_DAMPING);
	BIND_ENUM_CONSTANT(SLIDER_JOINT_MAX);

	ClassDB::bind_method(D_METHOD("joint_make_cone_twist", "joint", "body_A", "local_ref_A", "body_B", "local_ref_B"), &PhysicsServer3D::joint_make_cone_twist);

	ClassDB::bind_method(D_METHOD("cone_twist_joint_set_param", "joint", "param", "value"), &PhysicsServer3D::cone_twist_joint_set_param);
	ClassDB::bind_method(D_METHOD("cone_twist_joint_get_param", "joint", "param"), &PhysicsServer3D::cone_twist_joint_get_param);

	BIND_ENUM_CONSTANT(CONE_TWIST_JOINT_SWING_SPAN);
	BIND_ENUM_CONSTANT(CONE_TWIST_JOINT_TWIST_SPAN);
	BIND_ENUM_CONSTANT(CONE_TWIST_JOINT_BIAS);
	BIND_ENUM_CONSTANT(CONE_TWIST_JOINT_SOFTNESS);
	BIND_ENUM_CONSTANT(CONE_TWIST_JOINT_RELAXATION);

	BIND_ENUM_CONSTANT(G6DOF_JOINT_LINEAR_LOWER_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_LINEAR_UPPER_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_LINEAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_LINEAR_RESTITUTION);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_LINEAR_DAMPING);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_LINEAR_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_LINEAR_MOTOR_FORCE_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_LOWER_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_UPPER_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_LIMIT_SOFTNESS);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_DAMPING);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_RESTITUTION);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_FORCE_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_ERP);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_MOTOR_TARGET_VELOCITY);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_ANGULAR_MOTOR_FORCE_LIMIT);

	BIND_ENUM_CONSTANT(G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_FLAG_ENABLE_MOTOR);
	BIND_ENUM_CONSTANT(G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR);

	ClassDB::bind_method(D_METHOD("joint_get_type", "joint"), &PhysicsServer3D::joint_get_type);

	ClassDB::bind_method(D_METHOD("joint_set_solver_priority", "joint", "priority"), &PhysicsServer3D::joint_set_solver_priority);
	ClassDB::bind_method(D_METHOD("joint_get_solver_priority", "joint"), &PhysicsServer3D::joint_get_solver_priority);

	ClassDB::bind_method(D_METHOD("joint_make_generic_6dof", "joint", "body_A", "local_ref_A", "body_B", "local_ref_B"), &PhysicsServer3D::joint_make_generic_6dof);

	ClassDB::bind_method(D_METHOD("generic_6dof_joint_set_param", "joint", "axis", "param", "value"), &PhysicsServer3D::generic_6dof_joint_set_param);
	ClassDB::bind_method(D_METHOD("generic_6dof_joint_get_param", "joint", "axis", "param"), &PhysicsServer3D::generic_6dof_joint_get_param);

	ClassDB::bind_method(D_METHOD("generic_6dof_joint_set_flag", "joint", "axis", "flag", "enable"), &PhysicsServer3D::generic_6dof_joint_set_flag);
	ClassDB::bind_method(D_METHOD("generic_6dof_joint_get_flag", "joint", "axis", "flag"), &PhysicsServer3D::generic_6dof_joint_get_flag);

	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &PhysicsServer3D::free);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &PhysicsServer3D::set_active);

	ClassDB::bind_method(D_METHOD("get_process_info", "process_info"), &PhysicsServer3D::get_process_info);

	BIND_ENUM_CONSTANT(SHAPE_PLANE);
	BIND_ENUM_CONSTANT(SHAPE_RAY);
	BIND_ENUM_CONSTANT(SHAPE_SPHERE);
	BIND_ENUM_CONSTANT(SHAPE_BOX);
	BIND_ENUM_CONSTANT(SHAPE_CAPSULE);
	BIND_ENUM_CONSTANT(SHAPE_CYLINDER);
	BIND_ENUM_CONSTANT(SHAPE_CONVEX_POLYGON);
	BIND_ENUM_CONSTANT(SHAPE_CONCAVE_POLYGON);
	BIND_ENUM_CONSTANT(SHAPE_HEIGHTMAP);
	BIND_ENUM_CONSTANT(SHAPE_SOFT_BODY);
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
	BIND_ENUM_CONSTANT(BODY_PARAM_GRAVITY_SCALE);
	BIND_ENUM_CONSTANT(BODY_PARAM_LINEAR_DAMP);
	BIND_ENUM_CONSTANT(BODY_PARAM_ANGULAR_DAMP);
	BIND_ENUM_CONSTANT(BODY_PARAM_MAX);

	BIND_ENUM_CONSTANT(BODY_STATE_TRANSFORM);
	BIND_ENUM_CONSTANT(BODY_STATE_LINEAR_VELOCITY);
	BIND_ENUM_CONSTANT(BODY_STATE_ANGULAR_VELOCITY);
	BIND_ENUM_CONSTANT(BODY_STATE_SLEEPING);
	BIND_ENUM_CONSTANT(BODY_STATE_CAN_SLEEP);

	BIND_ENUM_CONSTANT(AREA_BODY_ADDED);
	BIND_ENUM_CONSTANT(AREA_BODY_REMOVED);

	BIND_ENUM_CONSTANT(INFO_ACTIVE_OBJECTS);
	BIND_ENUM_CONSTANT(INFO_COLLISION_PAIRS);
	BIND_ENUM_CONSTANT(INFO_ISLAND_COUNT);

	BIND_ENUM_CONSTANT(SPACE_PARAM_CONTACT_RECYCLE_RADIUS);
	BIND_ENUM_CONSTANT(SPACE_PARAM_CONTACT_MAX_SEPARATION);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_TIME_TO_SLEEP);
	BIND_ENUM_CONSTANT(SPACE_PARAM_BODY_ANGULAR_VELOCITY_DAMP_RATIO);
	BIND_ENUM_CONSTANT(SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS);
	BIND_ENUM_CONSTANT(SPACE_PARAM_TEST_MOTION_MIN_CONTACT_DEPTH);

	BIND_ENUM_CONSTANT(BODY_AXIS_LINEAR_X);
	BIND_ENUM_CONSTANT(BODY_AXIS_LINEAR_Y);
	BIND_ENUM_CONSTANT(BODY_AXIS_LINEAR_Z);
	BIND_ENUM_CONSTANT(BODY_AXIS_ANGULAR_X);
	BIND_ENUM_CONSTANT(BODY_AXIS_ANGULAR_Y);
	BIND_ENUM_CONSTANT(BODY_AXIS_ANGULAR_Z);

#endif
}

PhysicsServer3D::PhysicsServer3D() {
	singleton = this;
}

PhysicsServer3D::~PhysicsServer3D() {
	singleton = nullptr;
}

Vector<PhysicsServer3DManager::ClassInfo> PhysicsServer3DManager::physics_servers;
int PhysicsServer3DManager::default_server_id = -1;
int PhysicsServer3DManager::default_server_priority = -1;
const String PhysicsServer3DManager::setting_property_name("physics/3d/physics_engine");

void PhysicsServer3DManager::on_servers_changed() {
	String physics_servers2("DEFAULT");
	for (int i = get_servers_count() - 1; 0 <= i; --i) {
		physics_servers2 += "," + get_server_name(i);
	}
	ProjectSettings::get_singleton()->set_custom_property_info(setting_property_name, PropertyInfo(Variant::STRING, setting_property_name, PROPERTY_HINT_ENUM, physics_servers2));
}

void PhysicsServer3DManager::register_server(const String &p_name, CreatePhysicsServer3DCallback p_creat_callback) {
	ERR_FAIL_COND(!p_creat_callback);
	ERR_FAIL_COND(find_server_id(p_name) != -1);
	physics_servers.push_back(ClassInfo(p_name, p_creat_callback));
	on_servers_changed();
}

void PhysicsServer3DManager::set_default_server(const String &p_name, int p_priority) {
	const int id = find_server_id(p_name);
	ERR_FAIL_COND(id == -1); // Not found
	if (default_server_priority < p_priority) {
		default_server_id = id;
		default_server_priority = p_priority;
	}
}

int PhysicsServer3DManager::find_server_id(const String &p_name) {
	for (int i = physics_servers.size() - 1; 0 <= i; --i) {
		if (p_name == physics_servers[i].name) {
			return i;
		}
	}
	return -1;
}

int PhysicsServer3DManager::get_servers_count() {
	return physics_servers.size();
}

String PhysicsServer3DManager::get_server_name(int p_id) {
	ERR_FAIL_INDEX_V(p_id, get_servers_count(), "");
	return physics_servers[p_id].name;
}

PhysicsServer3D *PhysicsServer3DManager::new_default_server() {
	ERR_FAIL_COND_V(default_server_id == -1, nullptr);
	return physics_servers[default_server_id].create_callback();
}

PhysicsServer3D *PhysicsServer3DManager::new_server(const String &p_name) {
	int id = find_server_id(p_name);
	if (id == -1) {
		return nullptr;
	} else {
		return physics_servers[id].create_callback();
	}
}
