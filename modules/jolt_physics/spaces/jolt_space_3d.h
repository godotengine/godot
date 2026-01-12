/**************************************************************************/
/*  jolt_space_3d.h                                                       */
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

#pragma once

#include "servers/physics_3d/physics_server_3d.h"

#include "Jolt/Jolt.h"

#include "Jolt/Core/JobSystem.h"
#include "Jolt/Core/TempAllocator.h"
#include "Jolt/Physics/Body/BodyInterface.h"
#include "Jolt/Physics/Collision/BroadPhase/BroadPhaseQuery.h"
#include "Jolt/Physics/Collision/NarrowPhaseQuery.h"
#include "Jolt/Physics/Constraints/Constraint.h"
#include "Jolt/Physics/PhysicsSystem.h"

class JoltArea3D;
class JoltBody3D;
class JoltBodyActivationListener3D;
class JoltContactListener3D;
class JoltJoint3D;
class JoltLayers;
class JoltObject3D;
class JoltPhysicsDirectSpaceState3D;
class JoltShapedObject3D;
class JoltSoftBody3D;

class JoltSpace3D {
	Mutex pending_objects_mutex;
	Mutex body_call_queries_mutex;

	SelfList<JoltBody3D>::List body_call_queries_list;
	SelfList<JoltArea3D>::List area_call_queries_list;
	SelfList<JoltShapedObject3D>::List shapes_changed_list;
	SelfList<JoltShapedObject3D>::List needs_optimization_list;

	LocalVector<JPH::BodyID> pending_objects_sleeping;
	LocalVector<JPH::BodyID> pending_objects_awake;

	RID rid;

	JPH::JobSystem *job_system = nullptr;
	JPH::TempAllocator *temp_allocator = nullptr;
	JoltLayers *layers = nullptr;
	JoltContactListener3D *contact_listener = nullptr;
	JoltBodyActivationListener3D *body_activation_listener = nullptr;
	JPH::PhysicsSystem *physics_system = nullptr;
	JoltPhysicsDirectSpaceState3D *direct_state = nullptr;
	JoltArea3D *default_area = nullptr;

	float last_step = 0.0f;

	bool active = false;
	bool stepping = false;

	void _pre_step(float p_step);
	void _post_step(float p_step);

public:
	explicit JoltSpace3D(JPH::JobSystem *p_job_system);
	~JoltSpace3D();

	void step(float p_step);

	void call_queries();

	RID get_rid() const { return rid; }
	void set_rid(const RID &p_rid) { rid = p_rid; }

	bool is_active() const { return active; }
	void set_active(bool p_active) { active = p_active; }

	bool is_stepping() const { return stepping; }

	double get_param(PhysicsServer3D::SpaceParameter p_param) const;
	void set_param(PhysicsServer3D::SpaceParameter p_param, double p_value);

	JPH::PhysicsSystem &get_physics_system() const { return *physics_system; }

	JPH::TempAllocator &get_temp_allocator() const { return *temp_allocator; }

	JPH::BodyInterface &get_body_iface();
	const JPH::BodyInterface &get_body_iface() const;
	const JPH::BodyLockInterface &get_lock_iface() const;

	const JPH::BroadPhaseQuery &get_broad_phase_query() const;
	const JPH::NarrowPhaseQuery &get_narrow_phase_query() const;

	JPH::ObjectLayer map_to_object_layer(JPH::BroadPhaseLayer p_broad_phase_layer, uint32_t p_collision_layer, uint32_t p_collision_mask);
	void map_from_object_layer(JPH::ObjectLayer p_object_layer, JPH::BroadPhaseLayer &r_broad_phase_layer, uint32_t &r_collision_layer, uint32_t &r_collision_mask) const;

	JPH::Body *try_get_jolt_body(const JPH::BodyID &p_body_id) const;
	JoltObject3D *try_get_object(const JPH::BodyID &p_body_id) const;
	JoltShapedObject3D *try_get_shaped(const JPH::BodyID &p_body_id) const;
	JoltBody3D *try_get_body(const JPH::BodyID &p_body_id) const;
	JoltArea3D *try_get_area(const JPH::BodyID &p_body_id) const;
	JoltSoftBody3D *try_get_soft_body(const JPH::BodyID &p_body_id) const;

	JoltPhysicsDirectSpaceState3D *get_direct_state();

	JoltArea3D *get_default_area() const { return default_area; }
	void set_default_area(JoltArea3D *p_area);

	float get_last_step() const { return last_step; }

	JPH::Body *add_object(const JoltObject3D &p_object, const JPH::BodyCreationSettings &p_settings, bool p_sleeping = false);
	JPH::Body *add_object(const JoltObject3D &p_object, const JPH::SoftBodyCreationSettings &p_settings, bool p_sleeping = false);
	void remove_object(const JPH::BodyID &p_jolt_id);
	void flush_pending_objects();

	void set_is_object_sleeping(const JPH::BodyID &p_jolt_id, bool p_enable);

	void enqueue_call_queries(SelfList<JoltBody3D> *p_body);
	void enqueue_call_queries(SelfList<JoltArea3D> *p_area);
	void dequeue_call_queries(SelfList<JoltBody3D> *p_body);
	void dequeue_call_queries(SelfList<JoltArea3D> *p_area);

	void enqueue_shapes_changed(SelfList<JoltShapedObject3D> *p_object);
	void dequeue_shapes_changed(SelfList<JoltShapedObject3D> *p_object);

	void enqueue_needs_optimization(SelfList<JoltShapedObject3D> *p_object);
	void dequeue_needs_optimization(SelfList<JoltShapedObject3D> *p_object);

	void add_joint(JPH::Constraint *p_jolt_ref);
	void add_joint(JoltJoint3D *p_joint);
	void remove_joint(JPH::Constraint *p_jolt_ref);
	void remove_joint(JoltJoint3D *p_joint);

#ifdef DEBUG_ENABLED
	void dump_debug_snapshot(const String &p_dir);
	const PackedVector3Array &get_debug_contacts() const;
	int get_debug_contact_count() const;
	int get_max_debug_contacts() const;
	void set_max_debug_contacts(int p_count);
#endif
};
