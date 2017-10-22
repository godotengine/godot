/*************************************************************************/
/*  physics_server.cpp                                                   */
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
#include "physics_server.h"
#include "print_string.h"
PhysicsServer *PhysicsServer::singleton = NULL;

void PhysicsDirectBodyState::integrate_forces() {

	real_t step = get_step();
	Vector3 lv = get_linear_velocity();
	lv += get_total_gravity() * step;

	Vector3 av = get_angular_velocity();

	float linear_damp = 1.0 - step * get_total_linear_damp();

	if (linear_damp < 0) // reached zero in the given time
		linear_damp = 0;

	float angular_damp = 1.0 - step * get_total_angular_damp();

	if (angular_damp < 0) // reached zero in the given time
		angular_damp = 0;

	lv *= linear_damp;
	av *= angular_damp;

	set_linear_velocity(lv);
	set_angular_velocity(av);
}

Object *PhysicsDirectBodyState::get_contact_collider_object(int p_contact_idx) const {

	ObjectID objid = get_contact_collider_id(p_contact_idx);
	Object *obj = ObjectDB::get_instance(objid);
	return obj;
}

PhysicsServer *PhysicsServer::get_singleton() {

	return singleton;
}

void PhysicsDirectBodyState::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_total_gravity"), &PhysicsDirectBodyState::get_total_gravity);
	ClassDB::bind_method(D_METHOD("get_total_linear_damp"), &PhysicsDirectBodyState::get_total_linear_damp);
	ClassDB::bind_method(D_METHOD("get_total_angular_damp"), &PhysicsDirectBodyState::get_total_angular_damp);

	ClassDB::bind_method(D_METHOD("get_center_of_mass"), &PhysicsDirectBodyState::get_center_of_mass);
	ClassDB::bind_method(D_METHOD("get_principal_inertia_axes"), &PhysicsDirectBodyState::get_principal_inertia_axes);

	ClassDB::bind_method(D_METHOD("get_inverse_mass"), &PhysicsDirectBodyState::get_inverse_mass);
	ClassDB::bind_method(D_METHOD("get_inverse_inertia"), &PhysicsDirectBodyState::get_inverse_inertia);

	ClassDB::bind_method(D_METHOD("set_linear_velocity", "velocity"), &PhysicsDirectBodyState::set_linear_velocity);
	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &PhysicsDirectBodyState::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("set_angular_velocity", "velocity"), &PhysicsDirectBodyState::set_angular_velocity);
	ClassDB::bind_method(D_METHOD("get_angular_velocity"), &PhysicsDirectBodyState::get_angular_velocity);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsDirectBodyState::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsDirectBodyState::get_transform);

	ClassDB::bind_method(D_METHOD("add_force", "force", "position"), &PhysicsDirectBodyState::add_force);
	ClassDB::bind_method(D_METHOD("apply_impulse", "position", "j"), &PhysicsDirectBodyState::apply_impulse);
	ClassDB::bind_method(D_METHOD("apply_torqe_impulse", "j"), &PhysicsDirectBodyState::apply_torque_impulse);

	ClassDB::bind_method(D_METHOD("set_sleep_state", "enabled"), &PhysicsDirectBodyState::set_sleep_state);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &PhysicsDirectBodyState::is_sleeping);

	ClassDB::bind_method(D_METHOD("get_contact_count"), &PhysicsDirectBodyState::get_contact_count);

	ClassDB::bind_method(D_METHOD("get_contact_local_position", "contact_idx"), &PhysicsDirectBodyState::get_contact_local_position);
	ClassDB::bind_method(D_METHOD("get_contact_local_normal", "contact_idx"), &PhysicsDirectBodyState::get_contact_local_normal);
	ClassDB::bind_method(D_METHOD("get_contact_local_shape", "contact_idx"), &PhysicsDirectBodyState::get_contact_local_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider", "contact_idx"), &PhysicsDirectBodyState::get_contact_collider);
	ClassDB::bind_method(D_METHOD("get_contact_collider_position", "contact_idx"), &PhysicsDirectBodyState::get_contact_collider_position);
	ClassDB::bind_method(D_METHOD("get_contact_collider_id", "contact_idx"), &PhysicsDirectBodyState::get_contact_collider_id);
	ClassDB::bind_method(D_METHOD("get_contact_collider_object", "contact_idx"), &PhysicsDirectBodyState::get_contact_collider_object);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape", "contact_idx"), &PhysicsDirectBodyState::get_contact_collider_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider_velocity_at_position", "contact_idx"), &PhysicsDirectBodyState::get_contact_collider_velocity_at_position);
	ClassDB::bind_method(D_METHOD("get_step"), &PhysicsDirectBodyState::get_step);
	ClassDB::bind_method(D_METHOD("integrate_forces"), &PhysicsDirectBodyState::integrate_forces);
	ClassDB::bind_method(D_METHOD("get_space_state"), &PhysicsDirectBodyState::get_space_state);
}

PhysicsDirectBodyState::PhysicsDirectBodyState() {}

///////////////////////////////////////////////////////

void PhysicsShapeQueryParameters::set_shape(const RES &p_shape) {

	ERR_FAIL_COND(p_shape.is_null());
	shape = p_shape->get_rid();
}

void PhysicsShapeQueryParameters::set_shape_rid(const RID &p_shape) {

	shape = p_shape;
}

RID PhysicsShapeQueryParameters::get_shape_rid() const {

	return shape;
}

void PhysicsShapeQueryParameters::set_transform(const Transform &p_transform) {

	transform = p_transform;
}
Transform PhysicsShapeQueryParameters::get_transform() const {

	return transform;
}

void PhysicsShapeQueryParameters::set_margin(float p_margin) {

	margin = p_margin;
}

float PhysicsShapeQueryParameters::get_margin() const {

	return margin;
}

void PhysicsShapeQueryParameters::set_collision_layer(int p_collision_layer) {

	collision_layer = p_collision_layer;
}
int PhysicsShapeQueryParameters::get_collision_layer() const {

	return collision_layer;
}

void PhysicsShapeQueryParameters::set_object_type_mask(int p_object_type_mask) {

	object_type_mask = p_object_type_mask;
}
int PhysicsShapeQueryParameters::get_object_type_mask() const {

	return object_type_mask;
}
void PhysicsShapeQueryParameters::set_exclude(const Vector<RID> &p_exclude) {

	exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++)
		exclude.insert(p_exclude[i]);
}

Vector<RID> PhysicsShapeQueryParameters::get_exclude() const {

	Vector<RID> ret;
	ret.resize(exclude.size());
	int idx = 0;
	for (Set<RID>::Element *E = exclude.front(); E; E = E->next()) {
		ret[idx] = E->get();
	}
	return ret;
}

void PhysicsShapeQueryParameters::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_shape", "shape"), &PhysicsShapeQueryParameters::set_shape);
	ClassDB::bind_method(D_METHOD("set_shape_rid", "shape"), &PhysicsShapeQueryParameters::set_shape_rid);
	ClassDB::bind_method(D_METHOD("get_shape_rid"), &PhysicsShapeQueryParameters::get_shape_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &PhysicsShapeQueryParameters::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &PhysicsShapeQueryParameters::get_transform);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &PhysicsShapeQueryParameters::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &PhysicsShapeQueryParameters::get_margin);

	ClassDB::bind_method(D_METHOD("set_collision_layer", "collision_layer"), &PhysicsShapeQueryParameters::set_collision_layer);
	ClassDB::bind_method(D_METHOD("get_collision_layer"), &PhysicsShapeQueryParameters::get_collision_layer);

	ClassDB::bind_method(D_METHOD("set_object_type_mask", "object_type_mask"), &PhysicsShapeQueryParameters::set_object_type_mask);
	ClassDB::bind_method(D_METHOD("get_object_type_mask"), &PhysicsShapeQueryParameters::get_object_type_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &PhysicsShapeQueryParameters::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &PhysicsShapeQueryParameters::get_exclude);
}

PhysicsShapeQueryParameters::PhysicsShapeQueryParameters() {

	margin = 0;
	collision_layer = 0x7FFFFFFF;
	object_type_mask = PhysicsDirectSpaceState::TYPE_MASK_COLLISION;
}

/////////////////////////////////////

/*
Variant PhysicsDirectSpaceState::_intersect_shape(const RID& p_shape, const Transform& p_xform,int p_result_max,const Vector<RID>& p_exclude,uint32_t p_collision_mask) {



	ERR_FAIL_INDEX_V(p_result_max,4096,Variant());
	if (p_result_max<=0)
		return Variant();

	Set<RID> exclude;
	for(int i=0;i<p_exclude.size();i++)
		exclude.insert(p_exclude[i]);

	ShapeResult *res=(ShapeResult*)alloca(p_result_max*sizeof(ShapeResult));

	int rc = intersect_shape(p_shape,p_xform,0,res,p_result_max,exclude);

	if (rc==0)
		return Variant();

	Ref<PhysicsShapeQueryResult>  result = memnew( PhysicsShapeQueryResult );
	result->result.resize(rc);
	for(int i=0;i<rc;i++)
		result->result[i]=res[i];

	return result;

}
*/

Dictionary PhysicsDirectSpaceState::_intersect_ray(const Vector3 &p_from, const Vector3 &p_to, const Vector<RID> &p_exclude, uint32_t p_layers, uint32_t p_object_type_mask) {

	RayResult inters;
	Set<RID> exclude;
	for (int i = 0; i < p_exclude.size(); i++)
		exclude.insert(p_exclude[i]);

	bool res = intersect_ray(p_from, p_to, inters, exclude, p_layers, p_object_type_mask);

	if (!res)
		return Dictionary();

	Dictionary d;
	d["position"] = inters.position;
	d["normal"] = inters.normal;
	d["collider_id"] = inters.collider_id;
	d["collider"] = inters.collider;
	d["shape"] = inters.shape;
	d["rid"] = inters.rid;

	return d;
}

Array PhysicsDirectSpaceState::_intersect_shape(const Ref<PhysicsShapeQueryParameters> &p_shape_query, int p_max_results) {

	Vector<ShapeResult> sr;
	sr.resize(p_max_results);
	int rc = intersect_shape(p_shape_query->shape, p_shape_query->transform, p_shape_query->margin, sr.ptr(), sr.size(), p_shape_query->exclude, p_shape_query->collision_layer, p_shape_query->object_type_mask);
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

Array PhysicsDirectSpaceState::_cast_motion(const Ref<PhysicsShapeQueryParameters> &p_shape_query, const Vector3 &p_motion) {

	float closest_safe, closest_unsafe;
	bool res = cast_motion(p_shape_query->shape, p_shape_query->transform, p_motion, p_shape_query->margin, closest_safe, closest_unsafe, p_shape_query->exclude, p_shape_query->collision_layer, p_shape_query->object_type_mask);
	if (!res)
		return Array();
	Array ret;
	ret.resize(2);
	ret[0] = closest_safe;
	ret[1] = closest_unsafe;
	return ret;
}
Array PhysicsDirectSpaceState::_collide_shape(const Ref<PhysicsShapeQueryParameters> &p_shape_query, int p_max_results) {

	Vector<Vector3> ret;
	ret.resize(p_max_results * 2);
	int rc = 0;
	bool res = collide_shape(p_shape_query->shape, p_shape_query->transform, p_shape_query->margin, ret.ptr(), p_max_results, rc, p_shape_query->exclude, p_shape_query->collision_layer, p_shape_query->object_type_mask);
	if (!res)
		return Array();
	Array r;
	r.resize(rc * 2);
	for (int i = 0; i < rc * 2; i++)
		r[i] = ret[i];
	return r;
}
Dictionary PhysicsDirectSpaceState::_get_rest_info(const Ref<PhysicsShapeQueryParameters> &p_shape_query) {

	ShapeRestInfo sri;

	bool res = rest_info(p_shape_query->shape, p_shape_query->transform, p_shape_query->margin, &sri, p_shape_query->exclude, p_shape_query->collision_layer, p_shape_query->object_type_mask);
	Dictionary r;
	if (!res)
		return r;

	r["point"] = sri.point;
	r["normal"] = sri.normal;
	r["rid"] = sri.rid;
	r["collider_id"] = sri.collider_id;
	r["shape"] = sri.shape;
	r["linear_velocity"] = sri.linear_velocity;

	return r;
}

PhysicsDirectSpaceState::PhysicsDirectSpaceState() {
}

void PhysicsDirectSpaceState::_bind_methods() {

	//ClassDB::bind_method(D_METHOD("intersect_ray","from","to","exclude","umask"),&PhysicsDirectSpaceState::_intersect_ray,DEFVAL(Array()),DEFVAL(0));
	//ClassDB::bind_method(D_METHOD("intersect_shape","shape","xform","result_max","exclude","umask"),&PhysicsDirectSpaceState::_intersect_shape,DEFVAL(Array()),DEFVAL(0));

	ClassDB::bind_method(D_METHOD("intersect_ray", "from", "to", "exclude", "collision_layer", "type_mask"), &PhysicsDirectSpaceState::_intersect_ray, DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(TYPE_MASK_COLLISION));
	ClassDB::bind_method(D_METHOD("intersect_shape", "shape", "max_results"), &PhysicsDirectSpaceState::_intersect_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("cast_motion", "shape", "motion"), &PhysicsDirectSpaceState::_cast_motion);
	ClassDB::bind_method(D_METHOD("collide_shape", "shape", "max_results"), &PhysicsDirectSpaceState::_collide_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("get_rest_info", "shape"), &PhysicsDirectSpaceState::_get_rest_info);

	BIND_ENUM_CONSTANT(TYPE_MASK_STATIC_BODY);
	BIND_ENUM_CONSTANT(TYPE_MASK_KINEMATIC_BODY);
	BIND_ENUM_CONSTANT(TYPE_MASK_RIGID_BODY);
	BIND_ENUM_CONSTANT(TYPE_MASK_CHARACTER_BODY);
	BIND_ENUM_CONSTANT(TYPE_MASK_COLLISION);
	BIND_ENUM_CONSTANT(TYPE_MASK_AREA);
}

int PhysicsShapeQueryResult::get_result_count() const {

	return result.size();
}
RID PhysicsShapeQueryResult::get_result_rid(int p_idx) const {

	return result[p_idx].rid;
}
ObjectID PhysicsShapeQueryResult::get_result_object_id(int p_idx) const {

	return result[p_idx].collider_id;
}
Object *PhysicsShapeQueryResult::get_result_object(int p_idx) const {

	return result[p_idx].collider;
}
int PhysicsShapeQueryResult::get_result_object_shape(int p_idx) const {

	return result[p_idx].shape;
}

PhysicsShapeQueryResult::PhysicsShapeQueryResult() {
}

void PhysicsShapeQueryResult::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_result_count"), &PhysicsShapeQueryResult::get_result_count);
	ClassDB::bind_method(D_METHOD("get_result_rid", "idx"), &PhysicsShapeQueryResult::get_result_rid);
	ClassDB::bind_method(D_METHOD("get_result_object_id", "idx"), &PhysicsShapeQueryResult::get_result_object_id);
	ClassDB::bind_method(D_METHOD("get_result_object", "idx"), &PhysicsShapeQueryResult::get_result_object);
	ClassDB::bind_method(D_METHOD("get_result_object_shape", "idx"), &PhysicsShapeQueryResult::get_result_object_shape);
}

///////////////////////////////////////

void PhysicsServer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("shape_create", "type"), &PhysicsServer::shape_create);
	ClassDB::bind_method(D_METHOD("shape_set_data", "shape", "data"), &PhysicsServer::shape_set_data);

	ClassDB::bind_method(D_METHOD("shape_get_type", "shape"), &PhysicsServer::shape_get_type);
	ClassDB::bind_method(D_METHOD("shape_get_data", "shape"), &PhysicsServer::shape_get_data);

	ClassDB::bind_method(D_METHOD("space_create"), &PhysicsServer::space_create);
	ClassDB::bind_method(D_METHOD("space_set_active", "space", "active"), &PhysicsServer::space_set_active);
	ClassDB::bind_method(D_METHOD("space_is_active", "space"), &PhysicsServer::space_is_active);
	ClassDB::bind_method(D_METHOD("space_set_param", "space", "param", "value"), &PhysicsServer::space_set_param);
	ClassDB::bind_method(D_METHOD("space_get_param", "space", "param"), &PhysicsServer::space_get_param);
	ClassDB::bind_method(D_METHOD("space_get_direct_state", "space"), &PhysicsServer::space_get_direct_state);

	ClassDB::bind_method(D_METHOD("area_create"), &PhysicsServer::area_create);
	ClassDB::bind_method(D_METHOD("area_set_space", "area", "space"), &PhysicsServer::area_set_space);
	ClassDB::bind_method(D_METHOD("area_get_space", "area"), &PhysicsServer::area_get_space);

	ClassDB::bind_method(D_METHOD("area_set_space_override_mode", "area", "mode"), &PhysicsServer::area_set_space_override_mode);
	ClassDB::bind_method(D_METHOD("area_get_space_override_mode", "area"), &PhysicsServer::area_get_space_override_mode);

	ClassDB::bind_method(D_METHOD("area_add_shape", "area", "shape", "transform"), &PhysicsServer::area_add_shape, DEFVAL(Transform()));
	ClassDB::bind_method(D_METHOD("area_set_shape", "area", "shape_idx", "shape"), &PhysicsServer::area_set_shape);
	ClassDB::bind_method(D_METHOD("area_set_shape_transform", "area", "shape_idx", "transform"), &PhysicsServer::area_set_shape_transform);

	ClassDB::bind_method(D_METHOD("area_get_shape_count", "area"), &PhysicsServer::area_get_shape_count);
	ClassDB::bind_method(D_METHOD("area_get_shape", "area", "shape_idx"), &PhysicsServer::area_get_shape);
	ClassDB::bind_method(D_METHOD("area_get_shape_transform", "area", "shape_idx"), &PhysicsServer::area_get_shape_transform);

	ClassDB::bind_method(D_METHOD("area_remove_shape", "area", "shape_idx"), &PhysicsServer::area_remove_shape);
	ClassDB::bind_method(D_METHOD("area_clear_shapes", "area"), &PhysicsServer::area_clear_shapes);

	ClassDB::bind_method(D_METHOD("area_set_collision_layer", "area", "layer"), &PhysicsServer::area_set_collision_layer);
	ClassDB::bind_method(D_METHOD("area_set_collision_mask", "area", "mask"), &PhysicsServer::area_set_collision_mask);

	ClassDB::bind_method(D_METHOD("area_set_param", "area", "param", "value"), &PhysicsServer::area_set_param);
	ClassDB::bind_method(D_METHOD("area_set_transform", "area", "transform"), &PhysicsServer::area_set_transform);

	ClassDB::bind_method(D_METHOD("area_get_param", "area", "param"), &PhysicsServer::area_get_param);
	ClassDB::bind_method(D_METHOD("area_get_transform", "area"), &PhysicsServer::area_get_transform);

	ClassDB::bind_method(D_METHOD("area_attach_object_instance_id", "area", "id"), &PhysicsServer::area_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("area_get_object_instance_id", "area"), &PhysicsServer::area_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("area_set_monitor_callback", "area", "receiver", "method"), &PhysicsServer::area_set_monitor_callback);

	ClassDB::bind_method(D_METHOD("area_set_ray_pickable", "area", "enable"), &PhysicsServer::area_set_ray_pickable);
	ClassDB::bind_method(D_METHOD("area_is_ray_pickable", "area"), &PhysicsServer::area_is_ray_pickable);

	ClassDB::bind_method(D_METHOD("body_create", "mode", "init_sleeping"), &PhysicsServer::body_create, DEFVAL(BODY_MODE_RIGID), DEFVAL(false));

	ClassDB::bind_method(D_METHOD("body_set_space", "body", "space"), &PhysicsServer::body_set_space);
	ClassDB::bind_method(D_METHOD("body_get_space", "body"), &PhysicsServer::body_get_space);

	ClassDB::bind_method(D_METHOD("body_set_mode", "body", "mode"), &PhysicsServer::body_set_mode);
	ClassDB::bind_method(D_METHOD("body_get_mode", "body"), &PhysicsServer::body_get_mode);

	ClassDB::bind_method(D_METHOD("body_set_collision_layer", "body", "layer"), &PhysicsServer::body_set_collision_layer);
	ClassDB::bind_method(D_METHOD("body_get_collision_layer", "body"), &PhysicsServer::body_get_collision_layer);

	ClassDB::bind_method(D_METHOD("body_set_collision_mask", "body", "mask"), &PhysicsServer::body_set_collision_mask);
	ClassDB::bind_method(D_METHOD("body_get_collision_mask", "body"), &PhysicsServer::body_get_collision_mask);

	ClassDB::bind_method(D_METHOD("body_add_shape", "body", "shape", "transform"), &PhysicsServer::body_add_shape, DEFVAL(Transform()));
	ClassDB::bind_method(D_METHOD("body_set_shape", "body", "shape_idx", "shape"), &PhysicsServer::body_set_shape);
	ClassDB::bind_method(D_METHOD("body_set_shape_transform", "body", "shape_idx", "transform"), &PhysicsServer::body_set_shape_transform);

	ClassDB::bind_method(D_METHOD("body_get_shape_count", "body"), &PhysicsServer::body_get_shape_count);
	ClassDB::bind_method(D_METHOD("body_get_shape", "body", "shape_idx"), &PhysicsServer::body_get_shape);
	ClassDB::bind_method(D_METHOD("body_get_shape_transform", "body", "shape_idx"), &PhysicsServer::body_get_shape_transform);

	ClassDB::bind_method(D_METHOD("body_remove_shape", "body", "shape_idx"), &PhysicsServer::body_remove_shape);
	ClassDB::bind_method(D_METHOD("body_clear_shapes", "body"), &PhysicsServer::body_clear_shapes);

	ClassDB::bind_method(D_METHOD("body_attach_object_instance_id", "body", "id"), &PhysicsServer::body_attach_object_instance_id);
	ClassDB::bind_method(D_METHOD("body_get_object_instance_id", "body"), &PhysicsServer::body_get_object_instance_id);

	ClassDB::bind_method(D_METHOD("body_set_enable_continuous_collision_detection", "body", "enable"), &PhysicsServer::body_set_enable_continuous_collision_detection);
	ClassDB::bind_method(D_METHOD("body_is_continuous_collision_detection_enabled", "body"), &PhysicsServer::body_is_continuous_collision_detection_enabled);

	//ClassDB::bind_method(D_METHOD("body_set_user_flags","flags""),&PhysicsServer::body_set_shape,DEFVAL(Transform));
	//ClassDB::bind_method(D_METHOD("body_get_user_flags","body","shape_idx","shape"),&PhysicsServer::body_get_shape);

	ClassDB::bind_method(D_METHOD("body_set_param", "body", "param", "value"), &PhysicsServer::body_set_param);
	ClassDB::bind_method(D_METHOD("body_get_param", "body", "param"), &PhysicsServer::body_get_param);

	ClassDB::bind_method(D_METHOD("body_set_state", "body", "state", "value"), &PhysicsServer::body_set_state);
	ClassDB::bind_method(D_METHOD("body_get_state", "body", "state"), &PhysicsServer::body_get_state);

	ClassDB::bind_method(D_METHOD("body_apply_impulse", "body", "position", "impulse"), &PhysicsServer::body_apply_impulse);
	ClassDB::bind_method(D_METHOD("body_apply_torque_impulse", "body", "impulse"), &PhysicsServer::body_apply_torque_impulse);
	ClassDB::bind_method(D_METHOD("body_set_axis_velocity", "body", "axis_velocity"), &PhysicsServer::body_set_axis_velocity);

	ClassDB::bind_method(D_METHOD("body_set_axis_lock", "body", "axis"), &PhysicsServer::body_set_axis_lock);
	ClassDB::bind_method(D_METHOD("body_get_axis_lock", "body"), &PhysicsServer::body_get_axis_lock);

	ClassDB::bind_method(D_METHOD("body_add_collision_exception", "body", "excepted_body"), &PhysicsServer::body_add_collision_exception);
	ClassDB::bind_method(D_METHOD("body_remove_collision_exception", "body", "excepted_body"), &PhysicsServer::body_remove_collision_exception);
	//virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions)=0;

	ClassDB::bind_method(D_METHOD("body_set_max_contacts_reported", "body", "amount"), &PhysicsServer::body_set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("body_get_max_contacts_reported", "body"), &PhysicsServer::body_get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("body_set_omit_force_integration", "body", "enable"), &PhysicsServer::body_set_omit_force_integration);
	ClassDB::bind_method(D_METHOD("body_is_omitting_force_integration", "body"), &PhysicsServer::body_is_omitting_force_integration);

	ClassDB::bind_method(D_METHOD("body_set_force_integration_callback", "body", "receiver", "method", "userdata"), &PhysicsServer::body_set_force_integration_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_set_ray_pickable", "body", "enable"), &PhysicsServer::body_set_ray_pickable);
	ClassDB::bind_method(D_METHOD("body_is_ray_pickable", "body"), &PhysicsServer::body_is_ray_pickable);

	ClassDB::bind_method(D_METHOD("body_get_direct_state", "body"), &PhysicsServer::body_get_direct_state);

	/* JOINT API */

	BIND_ENUM_CONSTANT(JOINT_PIN);
	BIND_ENUM_CONSTANT(JOINT_HINGE);
	BIND_ENUM_CONSTANT(JOINT_SLIDER);
	BIND_ENUM_CONSTANT(JOINT_CONE_TWIST);
	BIND_ENUM_CONSTANT(JOINT_6DOF);

	ClassDB::bind_method(D_METHOD("joint_create_pin", "body_A", "local_A", "body_B", "local_B"), &PhysicsServer::joint_create_pin);
	ClassDB::bind_method(D_METHOD("pin_joint_set_param", "joint", "param", "value"), &PhysicsServer::pin_joint_set_param);
	ClassDB::bind_method(D_METHOD("pin_joint_get_param", "joint", "param"), &PhysicsServer::pin_joint_get_param);

	ClassDB::bind_method(D_METHOD("pin_joint_set_local_a", "joint", "local_A"), &PhysicsServer::pin_joint_set_local_a);
	ClassDB::bind_method(D_METHOD("pin_joint_get_local_a", "joint"), &PhysicsServer::pin_joint_get_local_a);

	ClassDB::bind_method(D_METHOD("pin_joint_set_local_b", "joint", "local_B"), &PhysicsServer::pin_joint_set_local_b);
	ClassDB::bind_method(D_METHOD("pin_joint_get_local_b", "joint"), &PhysicsServer::pin_joint_get_local_b);

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

	ClassDB::bind_method(D_METHOD("joint_create_hinge", "body_A", "hinge_A", "body_B", "hinge_B"), &PhysicsServer::joint_create_hinge);

	ClassDB::bind_method(D_METHOD("hinge_joint_set_param", "joint", "param", "value"), &PhysicsServer::hinge_joint_set_param);
	ClassDB::bind_method(D_METHOD("hinge_joint_get_param", "joint", "param"), &PhysicsServer::hinge_joint_get_param);

	ClassDB::bind_method(D_METHOD("hinge_joint_set_flag", "joint", "flag", "enabled"), &PhysicsServer::hinge_joint_set_flag);
	ClassDB::bind_method(D_METHOD("hinge_joint_get_flag", "joint", "flag"), &PhysicsServer::hinge_joint_get_flag);

	ClassDB::bind_method(D_METHOD("joint_create_slider", "body_A", "local_ref_A", "body_B", "local_ref_B"), &PhysicsServer::joint_create_slider);

	ClassDB::bind_method(D_METHOD("slider_joint_set_param", "joint", "param", "value"), &PhysicsServer::slider_joint_set_param);
	ClassDB::bind_method(D_METHOD("slider_joint_get_param", "joint", "param"), &PhysicsServer::slider_joint_get_param);

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

	ClassDB::bind_method(D_METHOD("joint_create_cone_twist", "body_A", "local_ref_A", "body_B", "local_ref_B"), &PhysicsServer::joint_create_cone_twist);

	ClassDB::bind_method(D_METHOD("cone_twist_joint_set_param", "joint", "param", "value"), &PhysicsServer::cone_twist_joint_set_param);
	ClassDB::bind_method(D_METHOD("cone_twist_joint_get_param", "joint", "param"), &PhysicsServer::cone_twist_joint_get_param);

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

	ClassDB::bind_method(D_METHOD("joint_get_type", "joint"), &PhysicsServer::joint_get_type);

	ClassDB::bind_method(D_METHOD("joint_set_solver_priority", "joint", "priority"), &PhysicsServer::joint_set_solver_priority);
	ClassDB::bind_method(D_METHOD("joint_get_solver_priority", "joint"), &PhysicsServer::joint_get_solver_priority);

	ClassDB::bind_method(D_METHOD("joint_create_generic_6dof", "body_A", "local_ref_A", "body_B", "local_ref_B"), &PhysicsServer::joint_create_generic_6dof);

	ClassDB::bind_method(D_METHOD("generic_6dof_joint_set_param", "joint", "axis", "param", "value"), &PhysicsServer::generic_6dof_joint_set_param);
	ClassDB::bind_method(D_METHOD("generic_6dof_joint_get_param", "joint", "axis", "param"), &PhysicsServer::generic_6dof_joint_get_param);

	ClassDB::bind_method(D_METHOD("generic_6dof_joint_set_flag", "joint", "axis", "flag", "enable"), &PhysicsServer::generic_6dof_joint_set_flag);
	ClassDB::bind_method(D_METHOD("generic_6dof_joint_get_flag", "joint", "axis", "flag"), &PhysicsServer::generic_6dof_joint_get_flag);

	/*
	ClassDB::bind_method(D_METHOD("joint_set_param","joint","param","value"),&PhysicsServer::joint_set_param);
	ClassDB::bind_method(D_METHOD("joint_get_param","joint","param"),&PhysicsServer::joint_get_param);

	ClassDB::bind_method(D_METHOD("pin_joint_create","anchor","body_a","body_b"),&PhysicsServer::pin_joint_create,DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("groove_joint_create","groove1_a","groove2_a","anchor_b","body_a","body_b"),&PhysicsServer::groove_joint_create,DEFVAL(RID()),DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("damped_spring_joint_create","anchor_a","anchor_b","body_a","body_b"),&PhysicsServer::damped_spring_joint_create,DEFVAL(RID()));

	ClassDB::bind_method(D_METHOD("damped_string_joint_set_param","joint","param","value"),&PhysicsServer::damped_string_joint_set_param,DEFVAL(RID()));
	ClassDB::bind_method(D_METHOD("damped_string_joint_get_param","joint","param"),&PhysicsServer::damped_string_joint_get_param);

	ClassDB::bind_method(D_METHOD("joint_get_type","joint"),&PhysicsServer::joint_get_type);
*/
	ClassDB::bind_method(D_METHOD("free_rid", "rid"), &PhysicsServer::free);

	ClassDB::bind_method(D_METHOD("set_active", "active"), &PhysicsServer::set_active);

	//ClassDB::bind_method(D_METHOD("init"),&PhysicsServer::init);
	//ClassDB::bind_method(D_METHOD("step"),&PhysicsServer::step);
	//ClassDB::bind_method(D_METHOD("sync"),&PhysicsServer::sync);
	//ClassDB::bind_method(D_METHOD("flush_queries"),&PhysicsServer::flush_queries);

	ClassDB::bind_method(D_METHOD("get_process_info", "process_info"), &PhysicsServer::get_process_info);

	BIND_ENUM_CONSTANT(SHAPE_PLANE);
	BIND_ENUM_CONSTANT(SHAPE_RAY);
	BIND_ENUM_CONSTANT(SHAPE_SPHERE);
	BIND_ENUM_CONSTANT(SHAPE_BOX);
	BIND_ENUM_CONSTANT(SHAPE_CAPSULE);
	BIND_ENUM_CONSTANT(SHAPE_CONVEX_POLYGON);
	BIND_ENUM_CONSTANT(SHAPE_CONCAVE_POLYGON);
	BIND_ENUM_CONSTANT(SHAPE_HEIGHTMAP);
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
	/*
	BIND_ENUM_CONSTANT( JOINT_PIN );
	BIND_ENUM_CONSTANT( JOINT_GROOVE );
	BIND_ENUM_CONSTANT( JOINT_DAMPED_SPRING );

	BIND_ENUM_CONSTANT( DAMPED_STRING_REST_LENGTH );
	BIND_ENUM_CONSTANT( DAMPED_STRING_STIFFNESS );
	BIND_ENUM_CONSTANT( DAMPED_STRING_DAMPING );
*/
	//BIND_ENUM_CONSTANT( TYPE_BODY );
	//BIND_ENUM_CONSTANT( TYPE_AREA );

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

	BIND_ENUM_CONSTANT(BODY_AXIS_LOCK_DISABLED);
	BIND_ENUM_CONSTANT(BODY_AXIS_LOCK_X);
	BIND_ENUM_CONSTANT(BODY_AXIS_LOCK_Y);
	BIND_ENUM_CONSTANT(BODY_AXIS_LOCK_Z);
}

PhysicsServer::PhysicsServer() {

	ERR_FAIL_COND(singleton != NULL);
	singleton = this;
}

PhysicsServer::~PhysicsServer() {

	singleton = NULL;
}
