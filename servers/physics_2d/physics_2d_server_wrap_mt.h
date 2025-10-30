/**************************************************************************/
/*  physics_2d_server_wrap_mt.h                                           */
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

#ifndef PHYSICS_2D_SERVER_WRAP_MT_H
#define PHYSICS_2D_SERVER_WRAP_MT_H

#include "core/command_queue_mt.h"
#include "core/os/thread.h"
#include "core/project_settings.h"
#include "core/safe_refcount.h"
#include "servers/physics_2d_server.h"

#ifdef DEBUG_SYNC
#define SYNC_DEBUG print_line("sync on: " + String(__FUNCTION__));
#else
#define SYNC_DEBUG
#endif

class Physics2DServerWrapMT : public Physics2DServer {
	mutable Physics2DServer *physics_2d_server;

	mutable CommandQueueMT command_queue;

	static void _thread_callback(void *_instance);
	void thread_loop();

	Thread::ID server_thread;
	Thread::ID main_thread;
	SafeFlag exit;
	Thread thread;
	SafeFlag step_thread_up;
	bool create_thread;

	Semaphore step_sem;
	void thread_step(real_t p_delta);

	void thread_exit();

	bool first_frame;

	Mutex alloc_mutex;
	int pool_max_size;

public:
#define ServerName Physics2DServer
#define ServerNameWrapMT Physics2DServerWrapMT
#define server_name physics_2d_server
#include "servers/server_wrap_mt_common.h"

	//FUNC1RID(shape,ShapeType); todo fix
	FUNCRID(line_shape)
	FUNCRID(ray_shape)
	FUNCRID(segment_shape)
	FUNCRID(circle_shape)
	FUNCRID(rectangle_shape)
	FUNCRID(capsule_shape)
	FUNCRID(convex_polygon_shape)
	FUNCRID(concave_polygon_shape)

	FUNC2(shape_set_data, RID, const Variant &);
	FUNC2(shape_set_custom_solver_bias, RID, real_t);

	FUNC1RC(ShapeType, shape_get_type, RID);
	FUNC1RC(Variant, shape_get_data, RID);
	FUNC1RC(real_t, shape_get_custom_solver_bias, RID);

	//these work well, but should be used from the main thread only
	bool shape_collide(RID p_shape_A, const Transform2D &p_xform_A, const Vector2 &p_motion_A, RID p_shape_B, const Transform2D &p_xform_B, const Vector2 &p_motion_B, Vector2 *r_results, int p_result_max, int &r_result_count) {
		ERR_FAIL_COND_V(main_thread != Thread::get_caller_id(), false);
		return physics_2d_server->shape_collide(p_shape_A, p_xform_A, p_motion_A, p_shape_B, p_xform_B, p_motion_B, r_results, p_result_max, r_result_count);
	}

	/* SPACE API */

	FUNCRID(space);
	FUNC2(space_set_active, RID, bool);
	FUNC1RC(bool, space_is_active, RID);

	FUNC3(space_set_param, RID, SpaceParameter, real_t);
	FUNC2RC(real_t, space_get_param, RID, SpaceParameter);

	// this function only works on physics process, errors and returns null otherwise
	Physics2DDirectSpaceState *space_get_direct_state(RID p_space) {
		ERR_FAIL_COND_V(main_thread != Thread::get_caller_id(), nullptr);
		return physics_2d_server->space_get_direct_state(p_space);
	}

	FUNC2(space_set_debug_contacts, RID, int);
	virtual Vector<Vector2> space_get_contacts(RID p_space) const {
		ERR_FAIL_COND_V(main_thread != Thread::get_caller_id(), Vector<Vector2>());
		return physics_2d_server->space_get_contacts(p_space);
	}

	virtual int space_get_contact_count(RID p_space) const {
		ERR_FAIL_COND_V(main_thread != Thread::get_caller_id(), 0);
		return physics_2d_server->space_get_contact_count(p_space);
	}

	/* AREA API */

	//FUNC0RID(area);
	FUNCRID(area);

	FUNC2(area_set_space, RID, RID);
	FUNC1RC(RID, area_get_space, RID);

	FUNC2(area_set_space_override_mode, RID, AreaSpaceOverrideMode);
	FUNC1RC(AreaSpaceOverrideMode, area_get_space_override_mode, RID);

	FUNC4(area_add_shape, RID, RID, const Transform2D &, bool);
	FUNC3(area_set_shape, RID, int, RID);
	FUNC3(area_set_shape_transform, RID, int, const Transform2D &);
	FUNC3(area_set_shape_disabled, RID, int, bool);

	FUNC1RC(int, area_get_shape_count, RID);
	FUNC2RC(RID, area_get_shape, RID, int);
	FUNC2RC(Transform2D, area_get_shape_transform, RID, int);
	FUNC2(area_remove_shape, RID, int);
	FUNC1(area_clear_shapes, RID);

	FUNC2(area_attach_object_instance_id, RID, ObjectID);
	FUNC1RC(ObjectID, area_get_object_instance_id, RID);

	FUNC2(area_attach_canvas_instance_id, RID, ObjectID);
	FUNC1RC(ObjectID, area_get_canvas_instance_id, RID);

	FUNC3(area_set_param, RID, AreaParameter, const Variant &);
	FUNC2(area_set_transform, RID, const Transform2D &);

	FUNC2RC(Variant, area_get_param, RID, AreaParameter);
	FUNC1RC(Transform2D, area_get_transform, RID);

	FUNC2(area_set_collision_mask, RID, uint32_t);
	FUNC2(area_set_collision_layer, RID, uint32_t);

	FUNC2(area_set_monitorable, RID, bool);
	FUNC2(area_set_pickable, RID, bool);

	FUNC3(area_set_monitor_callback, RID, Object *, const StringName &);
	FUNC3(area_set_area_monitor_callback, RID, Object *, const StringName &);

	/* BODY API */

	//FUNC2RID(body,BodyMode,bool);
	FUNCRID(body)

	FUNC2(body_set_space, RID, RID);
	FUNC1RC(RID, body_get_space, RID);

	FUNC2(body_set_mode, RID, BodyMode);
	FUNC1RC(BodyMode, body_get_mode, RID);

	FUNC4(body_add_shape, RID, RID, const Transform2D &, bool);
	FUNC3(body_set_shape, RID, int, RID);
	FUNC3(body_set_shape_transform, RID, int, const Transform2D &);
	FUNC3(body_set_shape_metadata, RID, int, const Variant &);

	FUNC1RC(int, body_get_shape_count, RID);
	FUNC2RC(Transform2D, body_get_shape_transform, RID, int);
	FUNC2RC(Variant, body_get_shape_metadata, RID, int);
	FUNC2RC(RID, body_get_shape, RID, int);

	FUNC3(body_set_shape_disabled, RID, int, bool);
	FUNC4(body_set_shape_as_one_way_collision, RID, int, bool, float);

	FUNC2(body_remove_shape, RID, int);
	FUNC1(body_clear_shapes, RID);

	FUNC2(body_attach_object_instance_id, RID, ObjectID);
	FUNC1RC(ObjectID, body_get_object_instance_id, RID);

	FUNC2(body_attach_canvas_instance_id, RID, ObjectID);
	FUNC1RC(ObjectID, body_get_canvas_instance_id, RID);

	FUNC2(body_set_continuous_collision_detection_mode, RID, CCDMode);
	FUNC1RC(CCDMode, body_get_continuous_collision_detection_mode, RID);

	FUNC2(body_set_collision_layer, RID, uint32_t);
	FUNC1RC(uint32_t, body_get_collision_layer, RID);

	FUNC2(body_set_collision_mask, RID, uint32_t);
	FUNC1RC(uint32_t, body_get_collision_mask, RID);

	FUNC3(body_set_param, RID, BodyParameter, real_t);
	FUNC2RC(real_t, body_get_param, RID, BodyParameter);

	FUNC3(body_set_state, RID, BodyState, const Variant &);
	FUNC2RC(Variant, body_get_state, RID, BodyState);

	FUNC2(body_set_applied_force, RID, const Vector2 &);
	FUNC1RC(Vector2, body_get_applied_force, RID);

	FUNC2(body_set_applied_torque, RID, real_t);
	FUNC1RC(real_t, body_get_applied_torque, RID);

	FUNC2(body_add_central_force, RID, const Vector2 &);
	FUNC3(body_add_force, RID, const Vector2 &, const Vector2 &);
	FUNC2(body_add_torque, RID, real_t);
	FUNC2(body_apply_central_impulse, RID, const Vector2 &);
	FUNC2(body_apply_torque_impulse, RID, real_t);
	FUNC3(body_apply_impulse, RID, const Vector2 &, const Vector2 &);
	FUNC2(body_set_axis_velocity, RID, const Vector2 &);

	FUNC2(body_add_collision_exception, RID, RID);
	FUNC2(body_remove_collision_exception, RID, RID);
	FUNC2S(body_get_collision_exceptions, RID, List<RID> *);

	FUNC2(body_set_max_contacts_reported, RID, int);
	FUNC1RC(int, body_get_max_contacts_reported, RID);

	FUNC2(body_set_contacts_reported_depth_threshold, RID, real_t);
	FUNC1RC(real_t, body_get_contacts_reported_depth_threshold, RID);

	FUNC2(body_set_omit_force_integration, RID, bool);
	FUNC1RC(bool, body_is_omitting_force_integration, RID);

	FUNC4(body_set_force_integration_callback, RID, Object *, const StringName &, const Variant &);

	bool body_collide_shape(RID p_body, int p_body_shape, RID p_shape, const Transform2D &p_shape_xform, const Vector2 &p_motion, Vector2 *r_results, int p_result_max, int &r_result_count) {
		return physics_2d_server->body_collide_shape(p_body, p_body_shape, p_shape, p_shape_xform, p_motion, r_results, p_result_max, r_result_count);
	}

	FUNC2(body_set_pickable, RID, bool);

	FUNC8R(bool, body_test_motion, RID, const Transform2D &, const Vector2 &, bool, real_t, MotionResult *, bool, const Set<RID> &);
	FUNC7R(int, body_test_ray_separation, RID, const Transform2D &, bool, Vector2 &, SeparationResult *, int, float);

	// this function only works on physics process, errors and returns null otherwise
	Physics2DDirectBodyState *body_get_direct_state(RID p_body) {
		ERR_FAIL_COND_V(main_thread != Thread::get_caller_id(), nullptr);
		return physics_2d_server->body_get_direct_state(p_body);
	}

	/* JOINT API */

	FUNC3(joint_set_param, RID, JointParam, real_t);
	FUNC2RC(real_t, joint_get_param, RID, JointParam);

	FUNC2(joint_disable_collisions_between_bodies, RID, const bool);
	FUNC1RC(bool, joint_is_disabled_collisions_between_bodies, RID);

	///FUNC3RID(pin_joint,const Vector2&,RID,RID);
	///FUNC5RID(groove_joint,const Vector2&,const Vector2&,const Vector2&,RID,RID);
	///FUNC4RID(damped_spring_joint,const Vector2&,const Vector2&,RID,RID);

	//TODO need to convert this to FUNCRID, but it's a hassle..

	FUNC3R(RID, pin_joint_create, const Vector2 &, RID, RID);
	FUNC5R(RID, groove_joint_create, const Vector2 &, const Vector2 &, const Vector2 &, RID, RID);
	FUNC4R(RID, damped_spring_joint_create, const Vector2 &, const Vector2 &, RID, RID);

	FUNC3(pin_joint_set_param, RID, PinJointParam, real_t);
	FUNC2RC(real_t, pin_joint_get_param, RID, PinJointParam);

	FUNC3(damped_string_joint_set_param, RID, DampedStringParam, real_t);
	FUNC2RC(real_t, damped_string_joint_get_param, RID, DampedStringParam);

	FUNC1RC(JointType, joint_get_type, RID);

	/* MISC */

	FUNC1(free, RID);
	FUNC1(set_active, bool);
	FUNC1(set_collision_iterations, int);

	virtual void init();
	virtual void step(real_t p_step);
	virtual void sync();
	virtual void end_sync();
	virtual void flush_queries();
	virtual void finish();

	virtual bool is_flushing_queries() const {
		return physics_2d_server->is_flushing_queries();
	}

	int get_process_info(ProcessInfo p_info) {
		return physics_2d_server->get_process_info(p_info);
	}

	Physics2DServerWrapMT(Physics2DServer *p_contained, bool p_create_thread);
	~Physics2DServerWrapMT();

	template <class T>
	static Physics2DServer *init_server() {
#ifdef NO_THREADS
		return memnew(T); // Always single unsafe when no threads are available.
#else
		int tm = GLOBAL_DEF("physics/2d/thread_model", 1);
		if (tm == 0) { // single unsafe
			return memnew(T);
		} else if (tm == 1) { // single safe
			return memnew(Physics2DServerWrapMT(memnew(T), false));
		} else { // multi threaded
			return memnew(Physics2DServerWrapMT(memnew(T), true));
		}
#endif
	}

#undef ServerNameWrapMT
#undef ServerName
#undef server_name
};

#ifdef DEBUG_SYNC
#undef DEBUG_SYNC
#endif
#undef SYNC_DEBUG

#endif // PHYSICS_2D_SERVER_WRAP_MT_H
