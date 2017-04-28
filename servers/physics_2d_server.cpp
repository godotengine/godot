/*************************************************************************/
/*  physics_2d_server.cpp                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "physics_2d_server.h"
#include "print_string.h"
Physics2DServer *Physics2DServer::singleton = NULL;

void Physics2DDirectBodyState::integrate_forces() {

	real_t step = get_step();
	Vector2 lv = get_linear_velocity();
	lv += get_total_gravity() * step;

	real_t av = get_angular_velocity();

	float damp = 1.0 - step * get_total_linear_damp();

	if (damp < 0) // reached zero in the given time
		damp = 0;

	lv *= damp;

	damp = 1.0 - step * get_total_angular_damp();

	if (damp < 0) // reached zero in the given time
		damp = 0;

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

	ClassDB::bind_method(D_METHOD("set_sleep_state", "enabled"), &Physics2DDirectBodyState::set_sleep_state);
	ClassDB::bind_method(D_METHOD("is_sleeping"), &Physics2DDirectBodyState::is_sleeping);

	ClassDB::bind_method(D_METHOD("get_contact_count"), &Physics2DDirectBodyState::get_contact_count);

	ClassDB::bind_method(D_METHOD("get_contact_local_pos", "contact_idx"), &Physics2DDirectBodyState::get_contact_local_pos);
	ClassDB::bind_method(D_METHOD("get_contact_local_normal", "contact_idx"), &Physics2DDirectBodyState::get_contact_local_normal);
	ClassDB::bind_method(D_METHOD("get_contact_local_shape", "contact_idx"), &Physics2DDirectBodyState::get_contact_local_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider);
	ClassDB::bind_method(D_METHOD("get_contact_collider_pos", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_pos);
	ClassDB::bind_method(D_METHOD("get_contact_collider_id", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_id);
	ClassDB::bind_method(D_METHOD("get_contact_collider_object", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_object);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_shape);
	ClassDB::bind_method(D_METHOD("get_contact_collider_shape_metadata:Variant", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_shape_metadata);
	ClassDB::bind_method(D_METHOD("get_contact_collider_velocity_at_pos", "contact_idx"), &Physics2DDirectBodyState::get_contact_collider_velocity_at_pos);
	ClassDB::bind_method(D_METHOD("get_step"), &Physics2DDirectBodyState::get_step);
	ClassDB::bind_method(D_METHOD("integrate_forces"), &Physics2DDirectBodyState::integrate_forces);
	ClassDB::bind_method(D_METHOD("get_space_state:Physics2DDirectSpaceState"), &Physics2DDirectBodyState::get_space_state);
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

void Physics2DShapeQueryParameters::set_layer_mask(int p_layer_mask) {

	layer_mask = p_layer_mask;
}
int Physics2DShapeQueryParameters::get_layer_mask() const {

	return layer_mask;
}

void Physics2DShapeQueryParameters::set_object_type_mask(int p_object_type_mask) {

	object_type_mask = p_object_type_mask;
}
int Physics2DShapeQueryParameters::get_object_type_mask() const {

	return object_type_mask;
}
void Physics2DShapeQueryParameters::set_exclude(const Vector<RID> &p_exclude) {

	exclude.clear();
	for (int i = 0; i < p_exclude.size(); i++)
		exclude.insert(p_exclude[i]);
}

Vector<RID> Physics2DShapeQueryParameters::get_exclude() const {

	Vector<RID> ret;
	ret.resize(exclude.size());
	int idx = 0;
	for (Set<RID>::Element *E = exclude.front(); E; E = E->next()) {
		ret[idx] = E->get();
	}
	return ret;
}

void Physics2DShapeQueryParameters::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_shape", "shape:Shape2D"), &Physics2DShapeQueryParameters::set_shape);
	ClassDB::bind_method(D_METHOD("set_shape_rid", "shape"), &Physics2DShapeQueryParameters::set_shape_rid);
	ClassDB::bind_method(D_METHOD("get_shape_rid"), &Physics2DShapeQueryParameters::get_shape_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &Physics2DShapeQueryParameters::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &Physics2DShapeQueryParameters::get_transform);

	ClassDB::bind_method(D_METHOD("set_motion", "motion"), &Physics2DShapeQueryParameters::set_motion);
	ClassDB::bind_method(D_METHOD("get_motion"), &Physics2DShapeQueryParameters::get_motion);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &Physics2DShapeQueryParameters::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &Physics2DShapeQueryParameters::get_margin);

	ClassDB::bind_method(D_METHOD("set_layer_mask", "layer_mask"), &Physics2DShapeQueryParameters::set_layer_mask);
	ClassDB::bind_method(D_METHOD("get_layer_mask"), &Physics2DShapeQueryParameters::get_layer_mask);

	ClassDB::bind_method(D_METHOD("set_object_type_mask", "object_type_mask"), &Physics2DShapeQueryParameters::set_object_type_mask);
	ClassDB::bind_method(D_METHOD("get_object_type_mask"), &Physics2DShapeQueryParameters::get_object_type_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &Physics2DShapeQueryParameters::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &Physics2DShapeQueryParameters::get_exclude);
}

Physics2DShapeQueryParameters::Physics2DShapeQueryParameters() {

	margin = 0;
	layer_mask = 0x7FFFFFFF;
	object_type_mask = Physics2DDirectSpaceState::TYPE_MASK_COLLISION;
}

Dictionary Physics2DDirectSpaceState::_intersect_ray(const Vector2 &p_from, const Vector2 &p_to, const Vector<RID> &p_exclude, uint32_t p_layers, uint32_t p_object_type_mask) {

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
	d["metadata"] = inters.metadata;

	return d;
}

Array Physics2DDirectSpaceState::_intersect_shape(const Ref<Physics2DShapeQueryParameters> &psq, int p_max_results) {

	Vector<ShapeResult> sr;
	sr.resize(p_max_results);
	int rc = intersect_shape(psq->shape, psq->transform, psq->motion, psq->margin, sr.ptr(), sr.size(), psq->exclude, psq->layer_mask, psq->object_type_mask);
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

Array Physics2DDirectSpaceState::_cast_motion(const Ref<Physics2DShapeQueryParameters> &psq) {

	float closest_safe, closest_unsafe;
	bool res = cast_motion(psq->shape, psq->transform, psq->motion, psq->margin, closest_safe, closest_unsafe, psq->exclude, psq->layer_mask, psq->object_type_mask);
	if (!res)
		return Array();
	Array ret;
	ret.resize(2);
	ret[0] = closest_safe;
	ret[1] = closest_unsafe;
	return ret;
}

Array Physics2DDirectSpaceState::_intersect_point(const Vector2 &p_point, int p_max_results, const Vector<RID> &p_exclude, uint32_t p_layers, uint32_t p_object_type_mask) {

	Set<RID> exclude;
	for (int i = 0; i < p_exclude.size(); i++)
		exclude.insert(p_exclude[i]);

	Vector<ShapeResult> ret;
	ret.resize(p_max_results);

	int rc = intersect_point(p_point, ret.ptr(), ret.size(), exclude, p_layers, p_object_type_mask);
	if (rc == 0)
		return Array();

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

Array Physics2DDirectSpaceState::_collide_shape(const Ref<Physics2DShapeQueryParameters> &psq, int p_max_results) {

	Vector<Vector2> ret;
	ret.resize(p_max_results * 2);
	int rc = 0;
	bool res = collide_shape(psq->shape, psq->transform, psq->motion, psq->margin, ret.ptr(), p_max_results, rc, psq->exclude, psq->layer_mask, psq->object_type_mask);
	if (!res)
		return Array();
	Array r;
	r.resize(rc * 2);
	for (int i = 0; i < rc * 2; i++)
		r[i] = ret[i];
	return r;
}
Dictionary Physics2DDirectSpaceState::_get_rest_info(const Ref<Physics2DShapeQueryParameters> &psq) {

	ShapeRestInfo sri;

	bool res = rest_info(psq->shape, psq->transform, psq->motion, psq->margin, &sri, psq->exclude, psq->layer_mask, psq->object_type_mask);
	Dictionary r;
	if (!res)
		return r;

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

	ClassDB::bind_method(D_METHOD("intersect_point", "point", "max_results", "exclude", "layer_mask", "type_mask"), &Physics2DDirectSpaceState::_intersect_point, DEFVAL(32), DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(TYPE_MASK_COLLISION));
	ClassDB::bind_method(D_METHOD("intersect_ray:Dictionary", "from", "to", "exclude", "layer_mask", "type_mask"), &Physics2DDirectSpaceState::_intersect_ray, DEFVAL(Array()), DEFVAL(0x7FFFFFFF), DEFVAL(TYPE_MASK_COLLISION));
	ClassDB::bind_method(D_METHOD("intersect_shape", "shape:Physics2DShapeQueryParameters", "max_results"), &Physics2DDirectSpaceState::_intersect_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("cast_motion", "shape:Physics2DShapeQueryParameters"), &Physics2DDirectSpaceState::_cast_motion);
	ClassDB::bind_method(D_METHOD("collide_shape", "shape:Physics2DShapeQueryParameters", "max_results"), &Physics2DDirectSpaceState::_collide_shape, DEFVAL(32));
	ClassDB::bind_method(D_METHOD("get_rest_info", "shape:Physics2DShapeQueryParameters"), &Physics2DDirectSpaceState::_get_rest_info);
	//ClassDB::bind_method(D_METHOD("cast_motion","shape","xform","motion","exclude","umask"),&Physics2DDirectSpaceState::_intersect_shape,DEFVAL(Array()),DEFVAL(0));

	BIND_CONSTANT(TYPE_MASK_STATIC_BODY);
	BIND_CONSTANT(TYPE_MASK_KINEMATIC_BODY);
	BIND_CONSTANT(TYPE_MASK_RIGID_BODY);
	BIND_CONSTANT(TYPE_MASK_CHARACTER_BODY);
	BIND_CONSTANT(TYPE_MASK_AREA);
	BIND_CONSTANT(TYPE_MASK_COLLISION);
}

int Physics2DShapeQueryResult::get_result_count() const {

	return result.size();
}
RID Physics2DShapeQueryResult::get_result_rid(int p_idx) const {

	return result[p_idx].rid;
}
ObjectID Physics2DShapeQueryResult::get_result_object_id(int p_idx) const {

	return result[p_idx].collider_id;
}
Object *Physics2DShapeQueryResult::get_result_object(int p_idx) const {

	return result[p_idx].collider;
}
int Physics2DShapeQueryResult::get_result_object_shape(int p_idx) const {

	return result[p_idx].shape;
}

Physics2DShapeQueryResult::Physics2DShapeQueryResult() {
}

void Physics2DShapeQueryResult::_bind_methods() {

	ClassDB::bind_method(D_METHOD("get_result_count"), &Physics2DShapeQueryResult::get_result_count);
	ClassDB::bind_method(D_METHOD("get_result_rid", "idx"), &Physics2DShapeQueryResult::get_result_rid);
	ClassDB::bind_method(D_METHOD("get_result_object_id", "idx"), &Physics2DShapeQueryResult::get_result_object_id);
	ClassDB::bind_method(D_METHOD("get_result_object", "idx"), &Physics2DShapeQueryResult::get_result_object);
	ClassDB::bind_method(D_METHOD("get_result_object_shape", "idx"), &Physics2DShapeQueryResult::get_result_object_shape);
}

///////////////////////////////

/*bool Physics2DTestMotionResult::is_colliding() const {

	return colliding;
}*/
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

void Physics2DTestMotionResult::_bind_methods() {

	//ClassDB::bind_method(D_METHOD("is_colliding"),&Physics2DTestMotionResult::is_colliding);
	ClassDB::bind_method(D_METHOD("get_motion"), &Physics2DTestMotionResult::get_motion);
	ClassDB::bind_method(D_METHOD("get_motion_remainder"), &Physics2DTestMotionResult::get_motion_remainder);
	ClassDB::bind_method(D_METHOD("get_collision_point"), &Physics2DTestMotionResult::get_collision_point);
	ClassDB::bind_method(D_METHOD("get_collision_normal"), &Physics2DTestMotionResult::get_collision_normal);
	ClassDB::bind_method(D_METHOD("get_collider_velocity"), &Physics2DTestMotionResult::get_collider_velocity);
	ClassDB::bind_method(D_METHOD("get_collider_id"), &Physics2DTestMotionResult::get_collider_id);
	ClassDB::bind_method(D_METHOD("get_collider_rid"), &Physics2DTestMotionResult::get_collider_rid);
	ClassDB::bind_method(D_METHOD("get_collider"), &Physics2DTestMotionResult::get_collider);
	ClassDB::bind_method(D_METHOD("get_collider_shape"), &Physics2DTestMotionResult::get_collider_shape);
}

Physics2DTestMotionResult::Physics2DTestMotionResult() {

	colliding = false;
	result.collider_id = 0;
	result.collider_shape = 0;
}

///////////////////////////////////////

bool Physics2DServer::_body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, float p_margin, const Ref<Physics2DTestMotionResult> &p_result) {

	MotionResult *r = NULL;
	if (p_result.is_valid())
		r = p_result->get_result_ptr();
	return body_test_motion(p_body, p_from, p_motion, p_margin, r);
}

void Physics2DServer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("shape_create", "type"), &Physics2DServer::shape_create);
	ClassDB::bind_method(D_METHOD("shape_set_data", "shape", "data"), &Physics2DServer::shape_set_data);

	ClassDB::bind_method(D_METHOD("shape_get_type", "shape"), &Physics2DServer::shape_get_type);
	ClassDB::bind_method(D_METHOD("shape_get_data", "shape"), &Physics2DServer::shape_get_data);

	ClassDB::bind_method(D_METHOD("space_create"), &Physics2DServer::space_create);
	ClassDB::bind_method(D_METHOD("space_set_active", "space", "active"), &Physics2DServer::space_set_active);
	ClassDB::bind_method(D_METHOD("space_is_active", "space"), &Physics2DServer::space_is_active);
	ClassDB::bind_method(D_METHOD("space_set_param", "space", "param", "value"), &Physics2DServer::space_set_param);
	ClassDB::bind_method(D_METHOD("space_get_param", "space", "param"), &Physics2DServer::space_get_param);
	ClassDB::bind_method(D_METHOD("space_get_direct_state:Physics2DDirectSpaceState", "space"), &Physics2DServer::space_get_direct_state);

	ClassDB::bind_method(D_METHOD("area_create"), &Physics2DServer::area_create);
	ClassDB::bind_method(D_METHOD("area_set_space", "area", "space"), &Physics2DServer::area_set_space);
	ClassDB::bind_method(D_METHOD("area_get_space", "area"), &Physics2DServer::area_get_space);

	ClassDB::bind_method(D_METHOD("area_set_space_override_mode", "area", "mode"), &Physics2DServer::area_set_space_override_mode);
	ClassDB::bind_method(D_METHOD("area_get_space_override_mode", "area"), &Physics2DServer::area_get_space_override_mode);

	ClassDB::bind_method(D_METHOD("area_add_shape", "area", "shape", "transform"), &Physics2DServer::area_add_shape, DEFVAL(Transform2D()));
	ClassDB::bind_method(D_METHOD("area_set_shape", "area", "shape_idx", "shape"), &Physics2DServer::area_set_shape);
	ClassDB::bind_method(D_METHOD("area_set_shape_transform", "area", "shape_idx", "transform"), &Physics2DServer::area_set_shape_transform);

	ClassDB::bind_method(D_METHOD("area_get_shape_count", "area"), &Physics2DServer::area_get_shape_count);
	ClassDB::bind_method(D_METHOD("area_get_shape", "area", "shape_idx"), &Physics2DServer::area_get_shape);
	ClassDB::bind_method(D_METHOD("area_get_shape_transform", "area", "shape_idx"), &Physics2DServer::area_get_shape_transform);

	ClassDB::bind_method(D_METHOD("area_remove_shape", "area", "shape_idx"), &Physics2DServer::area_remove_shape);
	ClassDB::bind_method(D_METHOD("area_clear_shapes", "area"), &Physics2DServer::area_clear_shapes);

	ClassDB::bind_method(D_METHOD("area_set_layer_mask", "area", "mask"), &Physics2DServer::area_set_layer_mask);
	ClassDB::bind_method(D_METHOD("area_set_collision_mask", "area", "mask"), &Physics2DServer::area_set_collision_mask);

	ClassDB::bind_method(D_METHOD("area_set_param", "area", "param", "value"), &Physics2DServer::area_set_param);
	ClassDB::bind_method(D_METHOD("area_set_transform", "area", "transform"), &Physics2DServer::area_set_transform);

	ClassDB::bind_method(D_METHOD("area_get_param", "area", "param"), &Physics2DServer::area_get_param);
	ClassDB::bind_method(D_METHOD("area_get_transform", "area"), &Physics2DServer::area_get_transform);

	ClassDB::bind_method(D_METHOD("area_attach_object_instance_ID", "area", "id"), &Physics2DServer::area_attach_object_instance_ID);
	ClassDB::bind_method(D_METHOD("area_get_object_instance_ID", "area"), &Physics2DServer::area_get_object_instance_ID);

	ClassDB::bind_method(D_METHOD("area_set_monitor_callback", "area", "receiver", "method"), &Physics2DServer::area_set_monitor_callback);

	ClassDB::bind_method(D_METHOD("body_create", "mode", "init_sleeping"), &Physics2DServer::body_create, DEFVAL(BODY_MODE_RIGID), DEFVAL(false));

	ClassDB::bind_method(D_METHOD("body_set_space", "body", "space"), &Physics2DServer::body_set_space);
	ClassDB::bind_method(D_METHOD("body_get_space", "body"), &Physics2DServer::body_get_space);

	ClassDB::bind_method(D_METHOD("body_set_mode", "body", "mode"), &Physics2DServer::body_set_mode);
	ClassDB::bind_method(D_METHOD("body_get_mode", "body"), &Physics2DServer::body_get_mode);

	ClassDB::bind_method(D_METHOD("body_add_shape", "body", "shape", "transform"), &Physics2DServer::body_add_shape, DEFVAL(Transform2D()));
	ClassDB::bind_method(D_METHOD("body_set_shape", "body", "shape_idx", "shape"), &Physics2DServer::body_set_shape);
	ClassDB::bind_method(D_METHOD("body_set_shape_transform", "body", "shape_idx", "transform"), &Physics2DServer::body_set_shape_transform);
	ClassDB::bind_method(D_METHOD("body_set_shape_metadata", "body", "shape_idx", "metadata"), &Physics2DServer::body_set_shape_metadata);

	ClassDB::bind_method(D_METHOD("body_get_shape_count", "body"), &Physics2DServer::body_get_shape_count);
	ClassDB::bind_method(D_METHOD("body_get_shape", "body", "shape_idx"), &Physics2DServer::body_get_shape);
	ClassDB::bind_method(D_METHOD("body_get_shape_transform", "body", "shape_idx"), &Physics2DServer::body_get_shape_transform);
	ClassDB::bind_method(D_METHOD("body_get_shape_metadata", "body", "shape_idx"), &Physics2DServer::body_get_shape_metadata);

	ClassDB::bind_method(D_METHOD("body_remove_shape", "body", "shape_idx"), &Physics2DServer::body_remove_shape);
	ClassDB::bind_method(D_METHOD("body_clear_shapes", "body"), &Physics2DServer::body_clear_shapes);

	ClassDB::bind_method(D_METHOD("body_set_shape_as_trigger", "body", "shape_idx", "enable"), &Physics2DServer::body_set_shape_as_trigger);
	ClassDB::bind_method(D_METHOD("body_is_shape_set_as_trigger", "body", "shape_idx"), &Physics2DServer::body_is_shape_set_as_trigger);

	ClassDB::bind_method(D_METHOD("body_attach_object_instance_ID", "body", "id"), &Physics2DServer::body_attach_object_instance_ID);
	ClassDB::bind_method(D_METHOD("body_get_object_instance_ID", "body"), &Physics2DServer::body_get_object_instance_ID);

	ClassDB::bind_method(D_METHOD("body_set_continuous_collision_detection_mode", "body", "mode"), &Physics2DServer::body_set_continuous_collision_detection_mode);
	ClassDB::bind_method(D_METHOD("body_get_continuous_collision_detection_mode", "body"), &Physics2DServer::body_get_continuous_collision_detection_mode);

	ClassDB::bind_method(D_METHOD("body_set_layer_mask", "body", "mask"), &Physics2DServer::body_set_layer_mask);
	ClassDB::bind_method(D_METHOD("body_get_layer_mask", "body"), &Physics2DServer::body_get_layer_mask);

	ClassDB::bind_method(D_METHOD("body_set_collision_mask", "body", "mask"), &Physics2DServer::body_set_collision_mask);
	ClassDB::bind_method(D_METHOD("body_get_collision_mask", "body"), &Physics2DServer::body_get_collision_mask);

	ClassDB::bind_method(D_METHOD("body_set_param", "body", "param", "value"), &Physics2DServer::body_set_param);
	ClassDB::bind_method(D_METHOD("body_get_param", "body", "param"), &Physics2DServer::body_get_param);

	ClassDB::bind_method(D_METHOD("body_set_state", "body", "state", "value"), &Physics2DServer::body_set_state);
	ClassDB::bind_method(D_METHOD("body_get_state", "body", "state"), &Physics2DServer::body_get_state);

	ClassDB::bind_method(D_METHOD("body_apply_impulse", "body", "pos", "impulse"), &Physics2DServer::body_apply_impulse);
	ClassDB::bind_method(D_METHOD("body_add_force", "body", "offset", "force"), &Physics2DServer::body_add_force);
	ClassDB::bind_method(D_METHOD("body_set_axis_velocity", "body", "axis_velocity"), &Physics2DServer::body_set_axis_velocity);

	ClassDB::bind_method(D_METHOD("body_add_collision_exception", "body", "excepted_body"), &Physics2DServer::body_add_collision_exception);
	ClassDB::bind_method(D_METHOD("body_remove_collision_exception", "body", "excepted_body"), &Physics2DServer::body_remove_collision_exception);
	//virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions)=0;

	ClassDB::bind_method(D_METHOD("body_set_max_contacts_reported", "body", "amount"), &Physics2DServer::body_set_max_contacts_reported);
	ClassDB::bind_method(D_METHOD("body_get_max_contacts_reported", "body"), &Physics2DServer::body_get_max_contacts_reported);

	ClassDB::bind_method(D_METHOD("body_set_one_way_collision_direction", "body", "normal"), &Physics2DServer::body_set_one_way_collision_direction);
	ClassDB::bind_method(D_METHOD("body_get_one_way_collision_direction", "body"), &Physics2DServer::body_get_one_way_collision_direction);

	ClassDB::bind_method(D_METHOD("body_set_one_way_collision_max_depth", "body", "depth"), &Physics2DServer::body_set_one_way_collision_max_depth);
	ClassDB::bind_method(D_METHOD("body_get_one_way_collision_max_depth", "body"), &Physics2DServer::body_get_one_way_collision_max_depth);

	ClassDB::bind_method(D_METHOD("body_set_omit_force_integration", "body", "enable"), &Physics2DServer::body_set_omit_force_integration);
	ClassDB::bind_method(D_METHOD("body_is_omitting_force_integration", "body"), &Physics2DServer::body_is_omitting_force_integration);

	ClassDB::bind_method(D_METHOD("body_set_force_integration_callback", "body", "receiver", "method", "userdata"), &Physics2DServer::body_set_force_integration_callback, DEFVAL(Variant()));

	ClassDB::bind_method(D_METHOD("body_test_motion", "body", "from", "motion", "margin", "result:Physics2DTestMotionResult"), &Physics2DServer::_body_test_motion, DEFVAL(0.08), DEFVAL(Variant()));

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

	ClassDB::bind_method(D_METHOD("get_process_info", "process_info"), &Physics2DServer::get_process_info);

	//ClassDB::bind_method(D_METHOD("init"),&Physics2DServer::init);
	//ClassDB::bind_method(D_METHOD("step"),&Physics2DServer::step);
	//ClassDB::bind_method(D_METHOD("sync"),&Physics2DServer::sync);
	//ClassDB::bind_method(D_METHOD("flush_queries"),&Physics2DServer::flush_queries);

	BIND_CONSTANT(SPACE_PARAM_CONTACT_RECYCLE_RADIUS);
	BIND_CONSTANT(SPACE_PARAM_CONTACT_MAX_SEPARATION);
	BIND_CONSTANT(SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION);
	BIND_CONSTANT(SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_TRESHOLD);
	BIND_CONSTANT(SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_TRESHOLD);
	BIND_CONSTANT(SPACE_PARAM_BODY_TIME_TO_SLEEP);
	BIND_CONSTANT(SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS);

	BIND_CONSTANT(SHAPE_LINE);
	BIND_CONSTANT(SHAPE_SEGMENT);
	BIND_CONSTANT(SHAPE_CIRCLE);
	BIND_CONSTANT(SHAPE_RECTANGLE);
	BIND_CONSTANT(SHAPE_CAPSULE);
	BIND_CONSTANT(SHAPE_CONVEX_POLYGON);
	BIND_CONSTANT(SHAPE_CONCAVE_POLYGON);
	BIND_CONSTANT(SHAPE_CUSTOM);

	BIND_CONSTANT(AREA_PARAM_GRAVITY);
	BIND_CONSTANT(AREA_PARAM_GRAVITY_VECTOR);
	BIND_CONSTANT(AREA_PARAM_GRAVITY_IS_POINT);
	BIND_CONSTANT(AREA_PARAM_GRAVITY_DISTANCE_SCALE);
	BIND_CONSTANT(AREA_PARAM_GRAVITY_POINT_ATTENUATION);
	BIND_CONSTANT(AREA_PARAM_LINEAR_DAMP);
	BIND_CONSTANT(AREA_PARAM_ANGULAR_DAMP);
	BIND_CONSTANT(AREA_PARAM_PRIORITY);

	BIND_CONSTANT(AREA_SPACE_OVERRIDE_DISABLED);
	BIND_CONSTANT(AREA_SPACE_OVERRIDE_COMBINE);
	BIND_CONSTANT(AREA_SPACE_OVERRIDE_COMBINE_REPLACE);
	BIND_CONSTANT(AREA_SPACE_OVERRIDE_REPLACE);
	BIND_CONSTANT(AREA_SPACE_OVERRIDE_REPLACE_COMBINE);

	BIND_CONSTANT(BODY_MODE_STATIC);
	BIND_CONSTANT(BODY_MODE_KINEMATIC);
	BIND_CONSTANT(BODY_MODE_RIGID);
	BIND_CONSTANT(BODY_MODE_CHARACTER);

	BIND_CONSTANT(BODY_PARAM_BOUNCE);
	BIND_CONSTANT(BODY_PARAM_FRICTION);
	BIND_CONSTANT(BODY_PARAM_MASS);
	BIND_CONSTANT(BODY_PARAM_INERTIA);
	BIND_CONSTANT(BODY_PARAM_GRAVITY_SCALE);
	BIND_CONSTANT(BODY_PARAM_LINEAR_DAMP);
	BIND_CONSTANT(BODY_PARAM_ANGULAR_DAMP);
	BIND_CONSTANT(BODY_PARAM_MAX);

	BIND_CONSTANT(BODY_STATE_TRANSFORM);
	BIND_CONSTANT(BODY_STATE_LINEAR_VELOCITY);
	BIND_CONSTANT(BODY_STATE_ANGULAR_VELOCITY);
	BIND_CONSTANT(BODY_STATE_SLEEPING);
	BIND_CONSTANT(BODY_STATE_CAN_SLEEP);

	BIND_CONSTANT(JOINT_PIN);
	BIND_CONSTANT(JOINT_GROOVE);
	BIND_CONSTANT(JOINT_DAMPED_SPRING);

	BIND_CONSTANT(DAMPED_STRING_REST_LENGTH);
	BIND_CONSTANT(DAMPED_STRING_STIFFNESS);
	BIND_CONSTANT(DAMPED_STRING_DAMPING);

	BIND_CONSTANT(CCD_MODE_DISABLED);
	BIND_CONSTANT(CCD_MODE_CAST_RAY);
	BIND_CONSTANT(CCD_MODE_CAST_SHAPE);

	//BIND_CONSTANT( TYPE_BODY );
	//BIND_CONSTANT( TYPE_AREA );

	BIND_CONSTANT(AREA_BODY_ADDED);
	BIND_CONSTANT(AREA_BODY_REMOVED);

	BIND_CONSTANT(INFO_ACTIVE_OBJECTS);
	BIND_CONSTANT(INFO_COLLISION_PAIRS);
	BIND_CONSTANT(INFO_ISLAND_COUNT);
}

Physics2DServer::Physics2DServer() {

	//ERR_FAIL_COND( singleton!=NULL );
	singleton = this;
}

Physics2DServer::~Physics2DServer() {

	singleton = NULL;
}
