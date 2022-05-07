/*************************************************************************/
/*  physics_2d_server.h                                                  */
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

#ifndef PHYSICS_2D_SERVER_H
#define PHYSICS_2D_SERVER_H

#include "core/object.h"
#include "core/reference.h"
#include "core/resource.h"

class Physics2DDirectSpaceState;

class Physics2DDirectBodyState : public Object {
	GDCLASS(Physics2DDirectBodyState, Object);

protected:
	static void _bind_methods();

public:
	virtual Vector2 get_total_gravity() const = 0; // get gravity vector working on this body space/area
	virtual float get_total_linear_damp() const = 0; // get density of this body space/area
	virtual float get_total_angular_damp() const = 0; // get density of this body space/area

	virtual float get_inverse_mass() const = 0; // get the mass
	virtual real_t get_inverse_inertia() const = 0; // get density of this body space

	virtual void set_linear_velocity(const Vector2 &p_velocity) = 0;
	virtual Vector2 get_linear_velocity() const = 0;

	virtual void set_angular_velocity(real_t p_velocity) = 0;
	virtual real_t get_angular_velocity() const = 0;

	virtual void set_transform(const Transform2D &p_transform) = 0;
	virtual Transform2D get_transform() const = 0;

	virtual Vector2 get_velocity_at_local_position(const Vector2 &p_position) const = 0;

	virtual void add_central_force(const Vector2 &p_force) = 0;
	virtual void add_force(const Vector2 &p_offset, const Vector2 &p_force) = 0;
	virtual void add_torque(real_t p_torque) = 0;
	virtual void apply_central_impulse(const Vector2 &p_impulse) = 0;
	virtual void apply_torque_impulse(real_t p_torque) = 0;
	virtual void apply_impulse(const Vector2 &p_offset, const Vector2 &p_impulse) = 0;

	virtual void set_sleep_state(bool p_enable) = 0;
	virtual bool is_sleeping() const = 0;

	virtual int get_contact_count() const = 0;

	virtual Vector2 get_contact_local_position(int p_contact_idx) const = 0;
	virtual Vector2 get_contact_local_normal(int p_contact_idx) const = 0;
	virtual int get_contact_local_shape(int p_contact_idx) const = 0;

	virtual RID get_contact_collider(int p_contact_idx) const = 0;
	virtual Vector2 get_contact_collider_position(int p_contact_idx) const = 0;
	virtual ObjectID get_contact_collider_id(int p_contact_idx) const = 0;
	virtual Object *get_contact_collider_object(int p_contact_idx) const;
	virtual int get_contact_collider_shape(int p_contact_idx) const = 0;
	virtual Variant get_contact_collider_shape_metadata(int p_contact_idx) const = 0;
	virtual Vector2 get_contact_collider_velocity_at_position(int p_contact_idx) const = 0;

	virtual real_t get_step() const = 0;
	virtual void integrate_forces();

	virtual Physics2DDirectSpaceState *get_space_state() = 0;

	Physics2DDirectBodyState();
};

//used for script
class Physics2DShapeQueryParameters : public Reference {
	GDCLASS(Physics2DShapeQueryParameters, Reference);
	friend class Physics2DDirectSpaceState;
	RID shape;
	Transform2D transform;
	Vector2 motion;
	float margin;
	Set<RID> exclude;
	uint32_t collision_mask;

	bool collide_with_bodies;
	bool collide_with_areas;

protected:
	static void _bind_methods();

public:
	void set_shape(const RES &p_shape);
	void set_shape_rid(const RID &p_shape);
	RID get_shape_rid() const;

	void set_transform(const Transform2D &p_transform);
	Transform2D get_transform() const;

	void set_motion(const Vector2 &p_motion);
	Vector2 get_motion() const;

	void set_margin(float p_margin);
	float get_margin() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collide_with_bodies(bool p_enable);
	bool is_collide_with_bodies_enabled() const;

	void set_collide_with_areas(bool p_enable);
	bool is_collide_with_areas_enabled() const;

	void set_exclude(const Vector<RID> &p_exclude);
	Vector<RID> get_exclude() const;

	Physics2DShapeQueryParameters();
};

class Physics2DDirectSpaceState : public Object {
	GDCLASS(Physics2DDirectSpaceState, Object);

	Dictionary _intersect_ray(const Vector2 &p_from, const Vector2 &p_to, const Vector<RID> &p_exclude = Vector<RID>(), uint32_t p_layers = 0, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);
	Array _intersect_point(const Vector2 &p_point, int p_max_results = 32, const Vector<RID> &p_exclude = Vector<RID>(), uint32_t p_layers = 0, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);
	Array _intersect_point_on_canvas(const Vector2 &p_point, ObjectID p_canvas_intance_id, int p_max_results = 32, const Vector<RID> &p_exclude = Vector<RID>(), uint32_t p_layers = 0, bool p_collide_with_bodies = true, bool p_collide_with_areas = false);
	Array _intersect_point_impl(const Vector2 &p_point, int p_max_results, const Vector<RID> &p_exclud, uint32_t p_layers, bool p_collide_with_bodies, bool p_collide_with_areas, bool p_filter_by_canvas = false, ObjectID p_canvas_instance_id = 0);
	Array _intersect_shape(const Ref<Physics2DShapeQueryParameters> &p_shape_query, int p_max_results = 32);
	Array _cast_motion(const Ref<Physics2DShapeQueryParameters> &p_shape_query);
	Array _collide_shape(const Ref<Physics2DShapeQueryParameters> &p_shape_query, int p_max_results = 32);
	Dictionary _get_rest_info(const Ref<Physics2DShapeQueryParameters> &p_shape_query);

protected:
	static void _bind_methods();

public:
	struct RayResult {
		Vector2 position;
		Vector2 normal;
		RID rid;
		ObjectID collider_id = 0;
		Object *collider = nullptr;
		int shape = 0;
		Variant metadata;
	};

	virtual bool intersect_ray(const Vector2 &p_from, const Vector2 &p_to, RayResult &r_result, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false) = 0;

	struct ShapeResult {
		RID rid;
		ObjectID collider_id = 0;
		Object *collider = nullptr;
		int shape = 0;
		Variant metadata;
	};

	virtual int intersect_point(const Vector2 &p_point, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false, bool p_pick_point = false) = 0;
	virtual int intersect_point_on_canvas(const Vector2 &p_point, ObjectID p_canvas_instance_id, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false, bool p_pick_point = false) = 0;

	virtual int intersect_shape(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, float p_margin, ShapeResult *r_results, int p_result_max, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false) = 0;

	virtual bool cast_motion(const RID &p_shape, const Transform2D &p_xform, const Vector2 &p_motion, float p_margin, float &p_closest_safe, float &p_closest_unsafe, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false) = 0;

	virtual bool collide_shape(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, float p_margin, Vector2 *r_results, int p_result_max, int &r_result_count, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false) = 0;

	struct ShapeRestInfo {
		Vector2 point;
		Vector2 normal;
		RID rid;
		ObjectID collider_id = 0;
		int shape = 0;
		Vector2 linear_velocity; //velocity at contact point
		Variant metadata;
	};

	virtual bool rest_info(RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, float p_margin, ShapeRestInfo *r_info, const Set<RID> &p_exclude = Set<RID>(), uint32_t p_collision_layer = 0xFFFFFFFF, bool p_collide_with_bodies = true, bool p_collide_with_areas = false) = 0;

	Physics2DDirectSpaceState();
};

class Physics2DTestMotionResult;

class Physics2DServer : public Object {
	GDCLASS(Physics2DServer, Object);

	static Physics2DServer *singleton;

	virtual bool _body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, float p_margin = 0.08, const Ref<Physics2DTestMotionResult> &p_result = Ref<Physics2DTestMotionResult>(), bool p_exclude_raycast_shapes = true, const Vector<RID> &p_exclude = Vector<RID>());

protected:
	static void _bind_methods();

public:
	static Physics2DServer *get_singleton();

	enum ShapeType {
		SHAPE_LINE, ///< plane:"plane"
		SHAPE_RAY, ///< float:"length"
		SHAPE_SEGMENT, ///< float:"length"
		SHAPE_CIRCLE, ///< float:"radius"
		SHAPE_RECTANGLE, ///< vec3:"extents"
		SHAPE_CAPSULE,
		SHAPE_CONVEX_POLYGON, ///< array of planes:"planes"
		SHAPE_CONCAVE_POLYGON, ///< Vector2 array:"triangles" , or Dictionary with "indices" (int array) and "triangles" (Vector2 array)
		SHAPE_CUSTOM, ///< Server-Implementation based custom shape, calling shape_create() with this value will result in an error
	};

	virtual RID line_shape_create() = 0;
	virtual RID ray_shape_create() = 0;
	virtual RID segment_shape_create() = 0;
	virtual RID circle_shape_create() = 0;
	virtual RID rectangle_shape_create() = 0;
	virtual RID capsule_shape_create() = 0;
	virtual RID convex_polygon_shape_create() = 0;
	virtual RID concave_polygon_shape_create() = 0;

	virtual void shape_set_data(RID p_shape, const Variant &p_data) = 0;
	virtual void shape_set_custom_solver_bias(RID p_shape, real_t p_bias) = 0;

	virtual ShapeType shape_get_type(RID p_shape) const = 0;
	virtual Variant shape_get_data(RID p_shape) const = 0;
	virtual real_t shape_get_custom_solver_bias(RID p_shape) const = 0;

	//these work well, but should be used from the main thread only
	virtual bool shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count) = 0;

	/* SPACE API */

	virtual RID space_create() = 0;
	virtual void space_set_active(RID p_space, bool p_active) = 0;
	virtual bool space_is_active(RID p_space) const = 0;

	enum SpaceParameter {

		SPACE_PARAM_CONTACT_RECYCLE_RADIUS,
		SPACE_PARAM_CONTACT_MAX_SEPARATION,
		SPACE_PARAM_BODY_MAX_ALLOWED_PENETRATION,
		SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD,
		SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD,
		SPACE_PARAM_BODY_TIME_TO_SLEEP,
		SPACE_PARAM_CONSTRAINT_DEFAULT_BIAS,
	};

	virtual void space_set_param(RID p_space, SpaceParameter p_param, real_t p_value) = 0;
	virtual real_t space_get_param(RID p_space, SpaceParameter p_param) const = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual Physics2DDirectSpaceState *space_get_direct_state(RID p_space) = 0;

	virtual void space_set_debug_contacts(RID p_space, int p_max_contacts) = 0;
	virtual Vector<Vector2> space_get_contacts(RID p_space) const = 0;
	virtual int space_get_contact_count(RID p_space) const = 0;

	//missing space parameters

	/* AREA API */

	//missing attenuation? missing better override?

	enum AreaParameter {
		AREA_PARAM_GRAVITY,
		AREA_PARAM_GRAVITY_VECTOR,
		AREA_PARAM_GRAVITY_IS_POINT,
		AREA_PARAM_GRAVITY_DISTANCE_SCALE,
		AREA_PARAM_GRAVITY_POINT_ATTENUATION,
		AREA_PARAM_LINEAR_DAMP,
		AREA_PARAM_ANGULAR_DAMP,
		AREA_PARAM_PRIORITY
	};

	virtual RID area_create() = 0;

	virtual void area_set_space(RID p_area, RID p_space) = 0;
	virtual RID area_get_space(RID p_area) const = 0;

	enum AreaSpaceOverrideMode {
		AREA_SPACE_OVERRIDE_DISABLED,
		AREA_SPACE_OVERRIDE_COMBINE,
		AREA_SPACE_OVERRIDE_COMBINE_REPLACE, // Combines, then discards all subsequent calculations
		AREA_SPACE_OVERRIDE_REPLACE,
		AREA_SPACE_OVERRIDE_REPLACE_COMBINE // Discards all previous calculations, then keeps combining
	};

	virtual void area_set_space_override_mode(RID p_area, AreaSpaceOverrideMode p_mode) = 0;
	virtual AreaSpaceOverrideMode area_get_space_override_mode(RID p_area) const = 0;

	virtual void area_add_shape(RID p_area, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) = 0;
	virtual void area_set_shape(RID p_area, int p_shape_idx, RID p_shape) = 0;
	virtual void area_set_shape_transform(RID p_area, int p_shape_idx, const Transform2D &p_transform) = 0;

	virtual int area_get_shape_count(RID p_area) const = 0;
	virtual RID area_get_shape(RID p_area, int p_shape_idx) const = 0;
	virtual Transform2D area_get_shape_transform(RID p_area, int p_shape_idx) const = 0;

	virtual void area_remove_shape(RID p_area, int p_shape_idx) = 0;
	virtual void area_clear_shapes(RID p_area) = 0;

	virtual void area_set_shape_disabled(RID p_area, int p_shape, bool p_disabled) = 0;

	virtual void area_attach_object_instance_id(RID p_area, ObjectID p_id) = 0;
	virtual ObjectID area_get_object_instance_id(RID p_area) const = 0;

	virtual void area_attach_canvas_instance_id(RID p_area, ObjectID p_id) = 0;
	virtual ObjectID area_get_canvas_instance_id(RID p_area) const = 0;

	virtual void area_set_param(RID p_area, AreaParameter p_param, const Variant &p_value) = 0;
	virtual void area_set_transform(RID p_area, const Transform2D &p_transform) = 0;

	virtual Variant area_get_param(RID p_parea, AreaParameter p_param) const = 0;
	virtual Transform2D area_get_transform(RID p_area) const = 0;

	virtual void area_set_collision_mask(RID p_area, uint32_t p_mask) = 0;
	virtual void area_set_collision_layer(RID p_area, uint32_t p_layer) = 0;

	virtual void area_set_monitorable(RID p_area, bool p_monitorable) = 0;
	virtual void area_set_pickable(RID p_area, bool p_pickable) = 0;

	virtual void area_set_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) = 0;
	virtual void area_set_area_monitor_callback(RID p_area, Object *p_receiver, const StringName &p_method) = 0;

	/* BODY API */

	//missing ccd?

	enum BodyMode {
		BODY_MODE_STATIC,
		BODY_MODE_KINEMATIC,
		BODY_MODE_RIGID,
		BODY_MODE_CHARACTER
	};

	virtual RID body_create() = 0;

	virtual void body_set_space(RID p_body, RID p_space) = 0;
	virtual RID body_get_space(RID p_body) const = 0;

	virtual void body_set_mode(RID p_body, BodyMode p_mode) = 0;
	virtual BodyMode body_get_mode(RID p_body) const = 0;

	virtual void body_add_shape(RID p_body, RID p_shape, const Transform2D &p_transform = Transform2D(), bool p_disabled = false) = 0;
	virtual void body_set_shape(RID p_body, int p_shape_idx, RID p_shape) = 0;
	virtual void body_set_shape_transform(RID p_body, int p_shape_idx, const Transform2D &p_transform) = 0;
	virtual void body_set_shape_metadata(RID p_body, int p_shape_idx, const Variant &p_metadata) = 0;

	virtual int body_get_shape_count(RID p_body) const = 0;
	virtual RID body_get_shape(RID p_body, int p_shape_idx) const = 0;
	virtual Transform2D body_get_shape_transform(RID p_body, int p_shape_idx) const = 0;
	virtual Variant body_get_shape_metadata(RID p_body, int p_shape_idx) const = 0;

	virtual void body_set_shape_disabled(RID p_body, int p_shape, bool p_disabled) = 0;
	virtual void body_set_shape_as_one_way_collision(RID p_body, int p_shape, bool p_enabled, float p_margin = 0) = 0;

	virtual void body_remove_shape(RID p_body, int p_shape_idx) = 0;
	virtual void body_clear_shapes(RID p_body) = 0;

	virtual void body_attach_object_instance_id(RID p_body, uint32_t p_id) = 0;
	virtual uint32_t body_get_object_instance_id(RID p_body) const = 0;

	virtual void body_attach_canvas_instance_id(RID p_body, uint32_t p_id) = 0;
	virtual uint32_t body_get_canvas_instance_id(RID p_body) const = 0;

	enum CCDMode {
		CCD_MODE_DISABLED,
		CCD_MODE_CAST_RAY,
		CCD_MODE_CAST_SHAPE,
	};

	virtual void body_set_continuous_collision_detection_mode(RID p_body, CCDMode p_mode) = 0;
	virtual CCDMode body_get_continuous_collision_detection_mode(RID p_body) const = 0;

	virtual void body_set_collision_layer(RID p_body, uint32_t p_layer) = 0;
	virtual uint32_t body_get_collision_layer(RID p_body) const = 0;

	virtual void body_set_collision_mask(RID p_body, uint32_t p_mask) = 0;
	virtual uint32_t body_get_collision_mask(RID p_body) const = 0;

	// common body variables
	enum BodyParameter {
		BODY_PARAM_BOUNCE,
		BODY_PARAM_FRICTION,
		BODY_PARAM_MASS, ///< unused for static, always infinite
		BODY_PARAM_INERTIA, // read-only: computed from mass & shapes
		BODY_PARAM_GRAVITY_SCALE,
		BODY_PARAM_LINEAR_DAMP,
		BODY_PARAM_ANGULAR_DAMP,
		BODY_PARAM_MAX,
	};

	virtual void body_set_param(RID p_body, BodyParameter p_param, float p_value) = 0;
	virtual float body_get_param(RID p_body, BodyParameter p_param) const = 0;

	//state
	enum BodyState {
		BODY_STATE_TRANSFORM,
		BODY_STATE_LINEAR_VELOCITY,
		BODY_STATE_ANGULAR_VELOCITY,
		BODY_STATE_SLEEPING,
		BODY_STATE_CAN_SLEEP,
	};

	virtual void body_set_state(RID p_body, BodyState p_state, const Variant &p_variant) = 0;
	virtual Variant body_get_state(RID p_body, BodyState p_state) const = 0;

	//do something about it
	virtual void body_set_applied_force(RID p_body, const Vector2 &p_force) = 0;
	virtual Vector2 body_get_applied_force(RID p_body) const = 0;

	virtual void body_set_applied_torque(RID p_body, float p_torque) = 0;
	virtual float body_get_applied_torque(RID p_body) const = 0;

	virtual void body_add_central_force(RID p_body, const Vector2 &p_force) = 0;
	virtual void body_add_force(RID p_body, const Vector2 &p_offset, const Vector2 &p_force) = 0;
	virtual void body_add_torque(RID p_body, float p_torque) = 0;

	virtual void body_apply_central_impulse(RID p_body, const Vector2 &p_impulse) = 0;
	virtual void body_apply_torque_impulse(RID p_body, float p_torque) = 0;
	virtual void body_apply_impulse(RID p_body, const Vector2 &p_offset, const Vector2 &p_impulse) = 0;
	virtual void body_set_axis_velocity(RID p_body, const Vector2 &p_axis_velocity) = 0;

	//fix
	virtual void body_add_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void body_remove_collision_exception(RID p_body, RID p_body_b) = 0;
	virtual void body_get_collision_exceptions(RID p_body, List<RID> *p_exceptions) = 0;

	virtual void body_set_max_contacts_reported(RID p_body, int p_contacts) = 0;
	virtual int body_get_max_contacts_reported(RID p_body) const = 0;

	//missing remove
	virtual void body_set_contacts_reported_depth_threshold(RID p_body, float p_threshold) = 0;
	virtual float body_get_contacts_reported_depth_threshold(RID p_body) const = 0;

	virtual void body_set_omit_force_integration(RID p_body, bool p_omit) = 0;
	virtual bool body_is_omitting_force_integration(RID p_body) const = 0;

	virtual void body_set_force_integration_callback(RID p_body, Object *p_receiver, const StringName &p_method, const Variant &p_udata = Variant()) = 0;

	virtual bool body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count) = 0;

	virtual void body_set_pickable(RID p_body, bool p_pickable) = 0;

	// this function only works on physics process, errors and returns null otherwise
	virtual Physics2DDirectBodyState *body_get_direct_state(RID p_body) = 0;

	struct MotionResult {
		Vector2 motion;
		Vector2 remainder;

		Vector2 collision_point;
		Vector2 collision_normal;
		Vector2 collider_velocity;
		real_t collision_depth = 0.0;
		real_t collision_safe_fraction = 0.0;
		real_t collision_unsafe_fraction = 0.0;
		int collision_local_shape = 0;
		ObjectID collider_id = 0;
		RID collider;
		int collider_shape = 0;
		Variant collider_metadata;
	};

	virtual bool body_test_motion(RID p_body, const Transform2D &p_from, const Vector2 &p_motion, bool p_infinite_inertia, float p_margin = 0.08, MotionResult *r_result = nullptr, bool p_exclude_raycast_shapes = true, const Set<RID> &p_exclude = Set<RID>()) = 0;

	struct SeparationResult {
		float collision_depth;
		Vector2 collision_point;
		Vector2 collision_normal;
		Vector2 collider_velocity;
		int collision_local_shape;
		ObjectID collider_id;
		RID collider;
		int collider_shape;
		Variant collider_metadata;
	};

	virtual int body_test_ray_separation(RID p_body, const Transform2D &p_transform, bool p_infinite_inertia, Vector2 &r_recover_motion, SeparationResult *r_results, int p_result_max, float p_margin = 0.08) = 0;

	/* JOINT API */

	enum JointType {

		JOINT_PIN,
		JOINT_GROOVE,
		JOINT_DAMPED_SPRING
	};

	enum JointParam {
		JOINT_PARAM_BIAS,
		JOINT_PARAM_MAX_BIAS,
		JOINT_PARAM_MAX_FORCE,
	};

	virtual void joint_set_param(RID p_joint, JointParam p_param, real_t p_value) = 0;
	virtual real_t joint_get_param(RID p_joint, JointParam p_param) const = 0;

	virtual void joint_disable_collisions_between_bodies(RID p_joint, const bool p_disable) = 0;
	virtual bool joint_is_disabled_collisions_between_bodies(RID p_joint) const = 0;

	virtual RID pin_joint_create(const Vector2 &p_anchor, RID p_body_a, RID p_body_b = RID()) = 0;
	virtual RID groove_joint_create(const Vector2 &p_a_groove1, const Vector2 &p_a_groove2, const Vector2 &p_b_anchor, RID p_body_a, RID p_body_b) = 0;
	virtual RID damped_spring_joint_create(const Vector2 &p_anchor_a, const Vector2 &p_anchor_b, RID p_body_a, RID p_body_b = RID()) = 0;

	enum PinJointParam {
		PIN_JOINT_SOFTNESS
	};

	virtual void pin_joint_set_param(RID p_joint, PinJointParam p_param, real_t p_value) = 0;
	virtual real_t pin_joint_get_param(RID p_joint, PinJointParam p_param) const = 0;

	enum DampedStringParam {
		DAMPED_STRING_REST_LENGTH,
		DAMPED_STRING_STIFFNESS,
		DAMPED_STRING_DAMPING
	};
	virtual void damped_string_joint_set_param(RID p_joint, DampedStringParam p_param, real_t p_value) = 0;
	virtual real_t damped_string_joint_get_param(RID p_joint, DampedStringParam p_param) const = 0;

	virtual JointType joint_get_type(RID p_joint) const = 0;

	/* QUERY API */

	enum AreaBodyStatus {
		AREA_BODY_ADDED,
		AREA_BODY_REMOVED
	};

	/* MISC */

	virtual void free(RID p_rid) = 0;

	virtual void set_active(bool p_active) = 0;
	virtual void init() = 0;
	virtual void step(float p_step) = 0;
	virtual void sync() = 0;
	virtual void flush_queries() = 0;
	virtual void end_sync() = 0;
	virtual void finish() = 0;

	virtual bool is_flushing_queries() const = 0;

	virtual void set_collision_iterations(int p_iterations) = 0;

	enum ProcessInfo {

		INFO_ACTIVE_OBJECTS,
		INFO_COLLISION_PAIRS,
		INFO_ISLAND_COUNT
	};

	virtual int get_process_info(ProcessInfo p_info) = 0;

	Physics2DServer();
	~Physics2DServer();
};

class Physics2DTestMotionResult : public Reference {
	GDCLASS(Physics2DTestMotionResult, Reference);

	Physics2DServer::MotionResult result;
	bool colliding = false;
	friend class Physics2DServer;

protected:
	static void _bind_methods();

public:
	Physics2DServer::MotionResult *get_result_ptr() const { return const_cast<Physics2DServer::MotionResult *>(&result); }

	//bool is_colliding() const;
	Vector2 get_motion() const;
	Vector2 get_motion_remainder() const;

	Vector2 get_collision_point() const;
	Vector2 get_collision_normal() const;
	Vector2 get_collider_velocity() const;
	ObjectID get_collider_id() const;
	RID get_collider_rid() const;
	Object *get_collider() const;
	int get_collider_shape() const;
	real_t get_collision_depth() const;
	real_t get_collision_safe_fraction() const;
	real_t get_collision_unsafe_fraction() const;
};

typedef Physics2DServer *(*CreatePhysics2DServerCallback)();

class Physics2DServerManager {
	struct ClassInfo {
		String name;
		CreatePhysics2DServerCallback create_callback;

		ClassInfo() :
				name(""),
				create_callback(nullptr) {}

		ClassInfo(String p_name, CreatePhysics2DServerCallback p_create_callback) :
				name(p_name),
				create_callback(p_create_callback) {}

		ClassInfo(const ClassInfo &p_ci) :
				name(p_ci.name),
				create_callback(p_ci.create_callback) {}

		ClassInfo operator=(const ClassInfo &p_ci) {
			name = p_ci.name;
			create_callback = p_ci.create_callback;
			return *this;
		}
	};

	static Vector<ClassInfo> physics_2d_servers;
	static int default_server_id;
	static int default_server_priority;

public:
	static const String setting_property_name;

private:
	static void on_servers_changed();

public:
	static void register_server(const String &p_name, CreatePhysics2DServerCallback p_creat_callback);
	static void set_default_server(const String &p_name, int p_priority = 0);
	static int find_server_id(const String &p_name);
	static int get_servers_count();
	static String get_server_name(int p_id);
	static Physics2DServer *new_default_server();
	static Physics2DServer *new_server(const String &p_name);
};

VARIANT_ENUM_CAST(Physics2DServer::ShapeType);
VARIANT_ENUM_CAST(Physics2DServer::SpaceParameter);
VARIANT_ENUM_CAST(Physics2DServer::AreaParameter);
VARIANT_ENUM_CAST(Physics2DServer::AreaSpaceOverrideMode);
VARIANT_ENUM_CAST(Physics2DServer::BodyMode);
VARIANT_ENUM_CAST(Physics2DServer::BodyParameter);
VARIANT_ENUM_CAST(Physics2DServer::BodyState);
VARIANT_ENUM_CAST(Physics2DServer::CCDMode);
VARIANT_ENUM_CAST(Physics2DServer::JointParam);
VARIANT_ENUM_CAST(Physics2DServer::JointType);
VARIANT_ENUM_CAST(Physics2DServer::DampedStringParam);
//VARIANT_ENUM_CAST( Physics2DServer::ObjectType );
VARIANT_ENUM_CAST(Physics2DServer::AreaBodyStatus);
VARIANT_ENUM_CAST(Physics2DServer::ProcessInfo);

#endif
