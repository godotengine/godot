/**************************************************************************/
/*  jolt_space_3d.cpp                                                     */
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

#include "jolt_space_3d.h"

#include "../joints/jolt_joint_3d.h"
#include "../jolt_physics_server_3d.h"
#include "../jolt_project_settings.h"
#include "../misc/jolt_stream_wrappers.h"
#include "../objects/jolt_area_3d.h"
#include "../objects/jolt_body_3d.h"
#include "../shapes/jolt_shape_3d.h"
#include "jolt_body_activation_listener_3d.h"
#include "jolt_contact_listener_3d.h"
#include "jolt_layers.h"
#include "jolt_physics_direct_space_state_3d.h"

#include "core/io/file_access.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/os/time.h"
#include "core/string/print_string.h"
#include "core/variant/variant_utility.h"

#include <Jolt/Physics/Collision/CollideShapeVsShapePerLeaf.h>
#include <Jolt/Physics/Collision/CollisionCollectorImpl.h>
#include <Jolt/Physics/PhysicsScene.h>
#include "Jolt/Physics/Collision/CollideShapeVsShapePerLeaf.h"
#include "Jolt/Physics/Collision/CollisionCollectorImpl.h"
#include "Jolt/Physics/PhysicsScene.h"
#include "Jolt/Physics/StateRecorderImpl.h"

namespace {

constexpr double SPACE_DEFAULT_CONTACT_RECYCLE_RADIUS = 0.01;
constexpr double SPACE_DEFAULT_CONTACT_MAX_SEPARATION = 0.05;
constexpr double SPACE_DEFAULT_CONTACT_MAX_ALLOWED_PENETRATION = 0.01;
constexpr double SPACE_DEFAULT_CONTACT_DEFAULT_BIAS = 0.8;
constexpr double SPACE_DEFAULT_SLEEP_THRESHOLD_LINEAR = 0.1;
constexpr double SPACE_DEFAULT_SLEEP_THRESHOLD_ANGULAR = 8.0 * Math::PI / 180;
constexpr double SPACE_DEFAULT_SOLVER_ITERATIONS = 8;

} // namespace

void JoltSpace3D::_pre_step(float p_step) {
	flush_pending_objects();

	while (needs_optimization_list.first()) {
		JoltShapedObject3D *object = needs_optimization_list.first()->self();
		needs_optimization_list.remove(needs_optimization_list.first());
		object->commit_shapes(true);
	}

	contact_listener->pre_step();

	const JPH::BodyLockInterface &lock_iface = get_lock_iface();
	const JPH::BodyID *active_rigid_bodies = physics_system->GetActiveBodiesUnsafe(JPH::EBodyType::RigidBody);
	const JPH::uint32 active_rigid_body_count = physics_system->GetNumActiveBodies(JPH::EBodyType::RigidBody);

	for (JPH::uint32 i = 0; i < active_rigid_body_count; i++) {
		JPH::Body *jolt_body = lock_iface.TryGetBody(active_rigid_bodies[i]);
		JoltObject3D *object = reinterpret_cast<JoltObject3D *>(jolt_body->GetUserData());
		object->pre_step(p_step);
	}

	const JPH::BodyID *active_soft_bodies = physics_system->GetActiveBodiesUnsafe(JPH::EBodyType::SoftBody);
	const JPH::uint32 active_soft_body_count = physics_system->GetNumActiveBodies(JPH::EBodyType::SoftBody);

	for (JPH::uint32 i = 0; i < active_soft_body_count; i++) {
		JPH::Body *jolt_body = lock_iface.TryGetBody(active_soft_bodies[i]);
		JoltObject3D *object = reinterpret_cast<JoltObject3D *>(jolt_body->GetUserData());
		object->pre_step(p_step);
	}

	physics_system->SetBodyActivationListener(body_activation_listener);
}

void JoltSpace3D::_post_step(float p_step) {
	// We only want a listener during the step, as it will otherwise be called when pending bodies are flushed, which causes issues (e.g. GH-115322).
	physics_system->SetBodyActivationListener(nullptr);

	contact_listener->post_step();

	while (shapes_changed_list.first()) {
		JoltShapedObject3D *object = shapes_changed_list.first()->self();
		shapes_changed_list.remove(shapes_changed_list.first());
		object->clear_previous_shape();
	}
}

JoltSpace3D::JoltSpace3D(JPH::JobSystem *p_job_system, JPH::TempAllocator *p_temp_allocator) :
		job_system(p_job_system),
		temp_allocator(p_temp_allocator),
		layers(new JoltLayers()),
		contact_listener(new JoltContactListener3D(this)),
		body_activation_listener(new JoltBodyActivationListener3D()),
		physics_system(new JPH::PhysicsSystem()) {
	physics_system->Init((JPH::uint)JoltProjectSettings::max_bodies, 0, (JPH::uint)JoltProjectSettings::max_body_pairs, (JPH::uint)JoltProjectSettings::max_contact_constraints, *layers, *layers, *layers);

	JPH::PhysicsSettings settings;
	settings.mBaumgarte = JoltProjectSettings::baumgarte_stabilization_factor;
	settings.mSpeculativeContactDistance = JoltProjectSettings::speculative_contact_distance;
	settings.mPenetrationSlop = JoltProjectSettings::penetration_slop;
	settings.mLinearCastThreshold = JoltProjectSettings::ccd_movement_threshold;
	settings.mLinearCastMaxPenetration = JoltProjectSettings::ccd_max_penetration;
	settings.mBodyPairCacheMaxDeltaPositionSq = JoltProjectSettings::body_pair_cache_distance_sq;
	settings.mBodyPairCacheCosMaxDeltaRotationDiv2 = JoltProjectSettings::body_pair_cache_angle_cos_div2;
	settings.mNumVelocitySteps = (JPH::uint)JoltProjectSettings::simulation_velocity_steps;
	settings.mNumPositionSteps = (JPH::uint)JoltProjectSettings::simulation_position_steps;
	settings.mMinVelocityForRestitution = JoltProjectSettings::bounce_velocity_threshold;
	settings.mTimeBeforeSleep = JoltProjectSettings::sleep_time_threshold;
	settings.mPointVelocitySleepThreshold = JoltProjectSettings::sleep_velocity_threshold;
	settings.mUseBodyPairContactCache = JoltProjectSettings::body_pair_contact_cache_enabled;
	settings.mAllowSleeping = JoltProjectSettings::sleep_allowed;

	physics_system->SetPhysicsSettings(settings);
	physics_system->SetGravity(JPH::Vec3::sZero());
	physics_system->SetContactListener(contact_listener);
	physics_system->SetSoftBodyContactListener(contact_listener);

	physics_system->SetSimCollideBodyVsBody([](const JPH::Body &p_body1, const JPH::Body &p_body2, JPH::Mat44Arg p_transform_com1, JPH::Mat44Arg p_transform_com2, JPH::CollideShapeSettings &p_collide_shape_settings, JPH::CollideShapeCollector &p_collector, const JPH::ShapeFilter &p_shape_filter) {
		if (p_body1.IsSensor() || p_body2.IsSensor()) {
			JPH::CollideShapeSettings new_collide_shape_settings = p_collide_shape_settings;
			// Since we're breaking the sensor down into leaf shapes we'll end up stripping away our `JoltCustomDoubleSidedShape` decorator shape and thus any back-face collision, so we simply force-enable it like this rather than going through the trouble of reapplying the decorator.
			new_collide_shape_settings.mBackFaceMode = JPH::EBackFaceMode::CollideWithBackFaces;
			JPH::SubShapeIDCreator part1, part2;
			JPH::CollideShapeVsShapePerLeaf<JPH::AnyHitCollisionCollector<JPH::CollideShapeCollector>>(p_body1.GetShape(), p_body2.GetShape(), JPH::Vec3::sOne(), JPH::Vec3::sOne(), p_transform_com1, p_transform_com2, part1, part2, new_collide_shape_settings, p_collector, p_shape_filter);
		} else {
			JPH::PhysicsSystem::sDefaultSimCollideBodyVsBody(p_body1, p_body2, p_transform_com1, p_transform_com2, p_collide_shape_settings, p_collector, p_shape_filter);
		}
	});

	physics_system->SetCombineFriction([](const JPH::Body &p_body1, const JPH::SubShapeID &p_sub_shape_id1, const JPH::Body &p_body2, const JPH::SubShapeID &p_sub_shape_id2) {
		return Math::abs(MIN(p_body1.GetFriction(), p_body2.GetFriction()));
	});

	physics_system->SetCombineRestitution([](const JPH::Body &p_body1, const JPH::SubShapeID &p_sub_shape_id1, const JPH::Body &p_body2, const JPH::SubShapeID &p_sub_shape_id2) {
		return CLAMP(p_body1.GetRestitution() + p_body2.GetRestitution(), 0.0f, 1.0f);
	});
}

JoltSpace3D::~JoltSpace3D() {
	if (direct_state != nullptr) {
		memdelete(direct_state);
		direct_state = nullptr;
	}

	if (physics_system != nullptr) {
		delete physics_system;
		physics_system = nullptr;
	}

	if (body_activation_listener != nullptr) {
		delete body_activation_listener;
		body_activation_listener = nullptr;
	}

	if (contact_listener != nullptr) {
		delete contact_listener;
		contact_listener = nullptr;
	}

	if (layers != nullptr) {
		delete layers;
		layers = nullptr;
	}
}

void JoltSpace3D::step(float p_step) {
	stepping = true;
	last_step = p_step;

	_pre_step(p_step);

	const JPH::EPhysicsUpdateError update_error = physics_system->Update(p_step, 1, temp_allocator, job_system);

	if ((update_error & JPH::EPhysicsUpdateError::ManifoldCacheFull) != JPH::EPhysicsUpdateError::None) {
		WARN_PRINT_ONCE(vformat("Jolt Physics manifold cache exceeded capacity and contacts were ignored. "
								"Consider increasing maximum number of contact constraints in project settings. "
								"Maximum number of contact constraints is currently set to %d.",
				JoltProjectSettings::max_contact_constraints));
	}

	if ((update_error & JPH::EPhysicsUpdateError::BodyPairCacheFull) != JPH::EPhysicsUpdateError::None) {
		WARN_PRINT_ONCE(vformat("Jolt Physics body pair cache exceeded capacity and contacts were ignored. "
								"Consider increasing maximum number of body pairs in project settings. "
								"Maximum number of body pairs is currently set to %d.",
				JoltProjectSettings::max_body_pairs));
	}

	if ((update_error & JPH::EPhysicsUpdateError::ContactConstraintsFull) != JPH::EPhysicsUpdateError::None) {
		WARN_PRINT_ONCE(vformat("Jolt Physics contact constraint buffer exceeded capacity and contacts were ignored. "
								"Consider increasing maximum number of contact constraints in project settings. "
								"Maximum number of contact constraints is currently set to %d.",
				JoltProjectSettings::max_contact_constraints));
	}

	_post_step(p_step);

	stepping = false;
}

void JoltSpace3D::call_queries() {
	while (body_call_queries_list.first()) {
		JoltBody3D *body = body_call_queries_list.first()->self();
		body_call_queries_list.remove(body_call_queries_list.first());
		body->call_queries();
	}

	while (area_call_queries_list.first()) {
		JoltArea3D *body = area_call_queries_list.first()->self();
		area_call_queries_list.remove(area_call_queries_list.first());
		body->call_queries();
	}
}

double JoltSpace3D::get_param(PhysicsServer3D::SpaceParameter p_param) const {
	switch (p_param) {
		case PhysicsServer3D::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: {
			return SPACE_DEFAULT_CONTACT_RECYCLE_RADIUS;
		}
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_SEPARATION: {
			return SPACE_DEFAULT_CONTACT_MAX_SEPARATION;
		}
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION: {
			return SPACE_DEFAULT_CONTACT_MAX_ALLOWED_PENETRATION;
		}
		case PhysicsServer3D::SPACE_PARAM_CONTACT_DEFAULT_BIAS: {
			return SPACE_DEFAULT_CONTACT_DEFAULT_BIAS;
		}
		case PhysicsServer3D::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD: {
			return SPACE_DEFAULT_SLEEP_THRESHOLD_LINEAR;
		}
		case PhysicsServer3D::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD: {
			return SPACE_DEFAULT_SLEEP_THRESHOLD_ANGULAR;
		}
		case PhysicsServer3D::SPACE_PARAM_BODY_TIME_TO_SLEEP: {
			return JoltProjectSettings::sleep_time_threshold;
		}
		case PhysicsServer3D::SPACE_PARAM_SOLVER_ITERATIONS: {
			return SPACE_DEFAULT_SOLVER_ITERATIONS;
		}
		default: {
			ERR_FAIL_V_MSG(0.0, vformat("Unhandled space parameter: '%d'. This should not happen. Please report this.", p_param));
		}
	}
}

void JoltSpace3D::set_param(PhysicsServer3D::SpaceParameter p_param, double p_value) {
	switch (p_param) {
		case PhysicsServer3D::SPACE_PARAM_CONTACT_RECYCLE_RADIUS: {
			WARN_PRINT("Space-specific contact recycle radius is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_SEPARATION: {
			WARN_PRINT("Space-specific contact max separation is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		case PhysicsServer3D::SPACE_PARAM_CONTACT_MAX_ALLOWED_PENETRATION: {
			WARN_PRINT("Space-specific contact max allowed penetration is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		case PhysicsServer3D::SPACE_PARAM_CONTACT_DEFAULT_BIAS: {
			WARN_PRINT("Space-specific contact default bias is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		case PhysicsServer3D::SPACE_PARAM_BODY_LINEAR_VELOCITY_SLEEP_THRESHOLD: {
			WARN_PRINT("Space-specific linear velocity sleep threshold is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		case PhysicsServer3D::SPACE_PARAM_BODY_ANGULAR_VELOCITY_SLEEP_THRESHOLD: {
			WARN_PRINT("Space-specific angular velocity sleep threshold is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		case PhysicsServer3D::SPACE_PARAM_BODY_TIME_TO_SLEEP: {
			WARN_PRINT("Space-specific body sleep time is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		case PhysicsServer3D::SPACE_PARAM_SOLVER_ITERATIONS: {
			WARN_PRINT("Space-specific solver iterations is not supported when using Jolt Physics. Any such value will be ignored.");
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unhandled space parameter: '%d'. This should not happen. Please report this.", p_param));
		} break;
	}
}

JPH::BodyInterface &JoltSpace3D::get_body_iface() {
	return physics_system->GetBodyInterfaceNoLock();
}

const JPH::BodyInterface &JoltSpace3D::get_body_iface() const {
	return physics_system->GetBodyInterfaceNoLock();
}

const JPH::BodyLockInterface &JoltSpace3D::get_lock_iface() const {
	return physics_system->GetBodyLockInterfaceNoLock();
}

const JPH::BroadPhaseQuery &JoltSpace3D::get_broad_phase_query() const {
	return physics_system->GetBroadPhaseQuery();
}

const JPH::NarrowPhaseQuery &JoltSpace3D::get_narrow_phase_query() const {
	return physics_system->GetNarrowPhaseQueryNoLock();
}

JPH::ObjectLayer JoltSpace3D::map_to_object_layer(JPH::BroadPhaseLayer p_broad_phase_layer, uint32_t p_collision_layer, uint32_t p_collision_mask) {
	return layers->to_object_layer(p_broad_phase_layer, p_collision_layer, p_collision_mask);
}

void JoltSpace3D::map_from_object_layer(JPH::ObjectLayer p_object_layer, JPH::BroadPhaseLayer &r_broad_phase_layer, uint32_t &r_collision_layer, uint32_t &r_collision_mask) const {
	layers->from_object_layer(p_object_layer, r_broad_phase_layer, r_collision_layer, r_collision_mask);
}

JPH::Body *JoltSpace3D::try_get_jolt_body(const JPH::BodyID &p_body_id) const {
	return get_lock_iface().TryGetBody(p_body_id);
}

JoltObject3D *JoltSpace3D::try_get_object(const JPH::BodyID &p_body_id) const {
	const JPH::Body *jolt_body = try_get_jolt_body(p_body_id);
	if (unlikely(jolt_body == nullptr)) {
		return nullptr;
	}

	return reinterpret_cast<JoltObject3D *>(jolt_body->GetUserData());
}

JoltShapedObject3D *JoltSpace3D::try_get_shaped(const JPH::BodyID &p_body_id) const {
	JoltObject3D *object = try_get_object(p_body_id);
	if (unlikely(object == nullptr)) {
		return nullptr;
	}

	return object->as_shaped();
}

JoltBody3D *JoltSpace3D::try_get_body(const JPH::BodyID &p_body_id) const {
	JoltObject3D *object = try_get_object(p_body_id);
	if (unlikely(object == nullptr)) {
		return nullptr;
	}

	return object->as_body();
}

JoltArea3D *JoltSpace3D::try_get_area(const JPH::BodyID &p_body_id) const {
	JoltObject3D *object = try_get_object(p_body_id);
	if (unlikely(object == nullptr)) {
		return nullptr;
	}

	return object->as_area();
}

JoltSoftBody3D *JoltSpace3D::try_get_soft_body(const JPH::BodyID &p_body_id) const {
	JoltObject3D *object = try_get_object(p_body_id);
	if (unlikely(object == nullptr)) {
		return nullptr;
	}

	return object->as_soft_body();
}

JoltPhysicsDirectSpaceState3D *JoltSpace3D::get_direct_state() {
	if (direct_state == nullptr) {
		direct_state = memnew(JoltPhysicsDirectSpaceState3D(this));
	}

	return direct_state;
}

JPH::Body *JoltSpace3D::add_object(const JoltObject3D &p_object, const JPH::BodyCreationSettings &p_settings, bool p_sleeping) {
	JPH::BodyInterface &body_iface = get_body_iface();
	JPH::Body *jolt_body = &p_object == snapshot_object ? body_iface.CreateBodyWithID(snapshot_body_id, p_settings) : body_iface.CreateBody(p_settings);
	if (unlikely(jolt_body == nullptr)) {
		ERR_PRINT_ONCE(vformat("Failed to create underlying Jolt Physics body for '%s'. "
							   "Consider increasing maximum number of bodies in project settings. "
							   "Maximum number of bodies is currently set to %d.",
				p_object.to_string(), JoltProjectSettings::max_bodies));

		return nullptr;
	}

	if (p_sleeping) {
		pending_objects_sleeping.push_back(jolt_body->GetID());
	} else {
		pending_objects_awake.push_back(jolt_body->GetID());
	}

	return jolt_body;
}

JPH::Body *JoltSpace3D::add_object(const JoltObject3D &p_object, const JPH::SoftBodyCreationSettings &p_settings, bool p_sleeping) {
	JPH::BodyInterface &body_iface = get_body_iface();
	JPH::Body *jolt_body = &p_object == snapshot_object ? body_iface.CreateSoftBodyWithID(snapshot_body_id, p_settings) : body_iface.CreateSoftBody(p_settings);
	if (unlikely(jolt_body == nullptr)) {
		ERR_PRINT_ONCE(vformat("Failed to create underlying Jolt Physics body for '%s'. "
							   "Consider increasing maximum number of bodies in project settings. "
							   "Maximum number of bodies is currently set to %d.",
				p_object.to_string(), JoltProjectSettings::max_bodies));

		return nullptr;
	}

	if (p_sleeping) {
		pending_objects_sleeping.push_back(jolt_body->GetID());
	} else {
		pending_objects_awake.push_back(jolt_body->GetID());
	}

	return jolt_body;
}

void JoltSpace3D::remove_object(const JPH::BodyID &p_jolt_id) {
	JPH::BodyInterface &body_iface = get_body_iface();

	if (!pending_objects_sleeping.erase_unordered(p_jolt_id) && !pending_objects_awake.erase_unordered(p_jolt_id)) {
		body_iface.RemoveBody(p_jolt_id);
	}

	body_iface.DestroyBody(p_jolt_id);

	// If we're never going to step this space, like in the editor viewport, we need to manually clean up Jolt's broad phase instead, otherwise performance can degrade when doing things like switching scenes.
	// We'll never actually have zero bodies in any space though, since we always have the default area, so we check if there's one or fewer left instead.
	if (!JoltPhysicsServer3D::get_singleton()->is_active() && physics_system->GetNumBodies() <= 1) {
		physics_system->OptimizeBroadPhase();
	}
}

void JoltSpace3D::flush_pending_objects() {
	if (pending_objects_sleeping.is_empty() && pending_objects_awake.is_empty()) {
		return;
	}

	// We only care about locking within this method, because it's called when performing queries, which aren't covered by `PhysicsServer3DWrapMT`.
	MutexLock pending_objects_lock(pending_objects_mutex);

	JPH::BodyInterface &body_iface = get_body_iface();

	if (!pending_objects_sleeping.is_empty()) {
		JPH::BodyInterface::AddState add_state = body_iface.AddBodiesPrepare(pending_objects_sleeping.ptr(), pending_objects_sleeping.size());
		body_iface.AddBodiesFinalize(pending_objects_sleeping.ptr(), pending_objects_sleeping.size(), add_state, JPH::EActivation::DontActivate);
		pending_objects_sleeping.reset();
	}

	if (!pending_objects_awake.is_empty()) {
		JPH::BodyInterface::AddState add_state = body_iface.AddBodiesPrepare(pending_objects_awake.ptr(), pending_objects_awake.size());
		body_iface.AddBodiesFinalize(pending_objects_awake.ptr(), pending_objects_awake.size(), add_state, JPH::EActivation::Activate);
		pending_objects_awake.reset();
	}
}

void JoltSpace3D::set_is_object_sleeping(const JPH::BodyID &p_jolt_id, bool p_enable) {
	if (p_enable) {
		if (pending_objects_awake.erase_unordered(p_jolt_id)) {
			pending_objects_sleeping.push_back(p_jolt_id);
		} else if (pending_objects_sleeping.has(p_jolt_id)) {
			// Do nothing.
		} else {
			get_body_iface().DeactivateBody(p_jolt_id);
		}
	} else {
		if (pending_objects_sleeping.erase_unordered(p_jolt_id)) {
			pending_objects_awake.push_back(p_jolt_id);
		} else if (pending_objects_awake.has(p_jolt_id)) {
			// Do nothing.
		} else {
			get_body_iface().ActivateBody(p_jolt_id);
		}
	}
}

void JoltSpace3D::enqueue_call_queries(SelfList<JoltBody3D> *p_body) {
	// This method will be called from the body activation listener on multiple threads during the simulation step.
	MutexLock body_call_queries_lock(body_call_queries_mutex);

	if (!p_body->in_list()) {
		body_call_queries_list.add(p_body);
	}
}

void JoltSpace3D::enqueue_call_queries(SelfList<JoltArea3D> *p_area) {
	if (!p_area->in_list()) {
		area_call_queries_list.add(p_area);
	}
}

void JoltSpace3D::dequeue_call_queries(SelfList<JoltBody3D> *p_body) {
	if (p_body->in_list()) {
		body_call_queries_list.remove(p_body);
	}
}

void JoltSpace3D::dequeue_call_queries(SelfList<JoltArea3D> *p_area) {
	if (p_area->in_list()) {
		area_call_queries_list.remove(p_area);
	}
}

void JoltSpace3D::enqueue_shapes_changed(SelfList<JoltShapedObject3D> *p_object) {
	if (!p_object->in_list()) {
		shapes_changed_list.add(p_object);
	}
}

void JoltSpace3D::dequeue_shapes_changed(SelfList<JoltShapedObject3D> *p_object) {
	if (p_object->in_list()) {
		shapes_changed_list.remove(p_object);
	}
}

void JoltSpace3D::enqueue_needs_optimization(SelfList<JoltShapedObject3D> *p_object) {
	if (!p_object->in_list()) {
		needs_optimization_list.add(p_object);
	}
}

void JoltSpace3D::dequeue_needs_optimization(SelfList<JoltShapedObject3D> *p_object) {
	if (p_object->in_list()) {
		needs_optimization_list.remove(p_object);
	}
}

void JoltSpace3D::add_joint(JPH::Constraint *p_jolt_ref) {
	physics_system->AddConstraint(p_jolt_ref);
}

void JoltSpace3D::add_joint(JoltJoint3D *p_joint) {
	add_joint(p_joint->get_jolt_ref());
}

void JoltSpace3D::remove_joint(JPH::Constraint *p_jolt_ref) {
	physics_system->RemoveConstraint(p_jolt_ref);
}

void JoltSpace3D::remove_joint(JoltJoint3D *p_joint) {
	remove_joint(p_joint->get_jolt_ref());
}

#ifdef DEBUG_ENABLED

void JoltSpace3D::dump_debug_snapshot(const String &p_dir) {
	const Dictionary datetime = Time::get_singleton()->get_datetime_dict_from_system();
	const String datetime_str = vformat("%04d-%02d-%02d_%02d-%02d-%02d", datetime["year"], datetime["month"], datetime["day"], datetime["hour"], datetime["minute"], datetime["second"]);
	const String path = p_dir + vformat("/jolt_snapshot_%s_%d.bin", datetime_str, rid.get_id());

	Ref<FileAccess> file_access = FileAccess::open(path, FileAccess::ModeFlags::WRITE);
	ERR_FAIL_COND_MSG(file_access.is_null(), vformat("Failed to open '%s' for writing when saving snapshot of physics space with RID '%d'.", path, rid.get_id()));

	JPH::PhysicsScene physics_scene;
	physics_scene.FromPhysicsSystem(physics_system);

	for (JPH::BodyCreationSettings &settings : physics_scene.GetBodies()) {
		const JoltObject3D *object = reinterpret_cast<const JoltObject3D *>(settings.mUserData);

		if (const JoltBody3D *body = object->as_body()) {
			// Since we apply our own damping, while leaving Jolt's own values at zero, we need to transfer over the correct values.
			settings.mLinearDamping = body->get_total_linear_damp();
			settings.mAngularDamping = body->get_total_angular_damp();
		}

		settings.SetShape(JoltShape3D::without_custom_shapes(settings.GetShape()));
	}

	JoltStreamOutputWrapper output_stream(file_access);
	physics_scene.SaveBinaryState(output_stream, true, false);

	ERR_FAIL_COND_MSG(file_access->get_error() != OK, vformat("Writing snapshot of physics space with RID '%d' to '%s' failed with error '%s'.", rid.get_id(), path, VariantUtilityFunctions::error_string(file_access->get_error())));

	print_line(vformat("Snapshot of physics space with RID '%d' saved to '%s'.", rid.get_id(), path));
}

const PackedVector3Array &JoltSpace3D::get_debug_contacts() const {
	return contact_listener->get_debug_contacts();
}

int JoltSpace3D::get_debug_contact_count() const {
	return contact_listener->get_debug_contact_count();
}

int JoltSpace3D::get_max_debug_contacts() const {
	return contact_listener->get_max_debug_contacts();
}

void JoltSpace3D::set_max_debug_contacts(int p_count) {
	contact_listener->set_max_debug_contacts(p_count);
}

#endif

// The space's areas and bodies but its default area, in the order they were
// made (their RIDs').
LocalVector<JoltObject3D *> JoltSpace3D::snapshot_objects() {
	flush_pending_objects();
	JPH::BodyIDVector ids;
	physics_system->GetBodies(ids);
	const JPH::BodyLockInterface &lock_iface = get_lock_iface();
	LocalVector<JoltObject3D *> out;
	for (const JPH::BodyID &id : ids) {
		const JPH::Body *jolt_body = lock_iface.TryGetBody(id);
		JoltObject3D *object = jolt_body != nullptr ? reinterpret_cast<JoltObject3D *>(jolt_body->GetUserData()) : nullptr;
		if (object != nullptr && object != default_area) {
			out.push_back(object);
		}
	}
	struct MadeFirst {
		bool operator()(const JoltObject3D *p_a, const JoltObject3D *p_b) const { return p_a->get_rid().get_id() < p_b->get_rid().get_id(); }
	};
	out.sort_custom<MadeFirst>();
	return out;
}

// The joints between the listed bodies, each with its key: its bodies' places
// in the list (UINT32_MAX for none) and its type, in the order of the keys,
// then of when they were made.
struct SnapshotJoint {
	uint32_t a = UINT32_MAX;
	uint32_t b = UINT32_MAX;
	uint32_t type = 0;
	JoltJoint3D *joint = nullptr;
};

static LocalVector<SnapshotJoint> _snapshot_joints(const LocalVector<JoltObject3D *> &p_objects) {
	HashMap<const JoltObject3D *, uint32_t> index;
	for (uint32_t i = 0; i < p_objects.size(); i++) {
		index[p_objects[i]] = i;
	}
	HashSet<JoltJoint3D *> seen;
	LocalVector<SnapshotJoint> out;
	for (JoltObject3D *object : p_objects) {
		const JoltBody3D *body = object->as_body();
		if (body == nullptr) {
			continue;
		}
		for (JoltJoint3D *joint : body->get_joints()) {
			if (seen.has(joint)) {
				continue;
			}
			seen.insert(joint);
			SnapshotJoint j;
			j.joint = joint;
			j.type = uint32_t(joint->get_type());
			if (const uint32_t *a = joint->get_body_a() ? index.getptr(joint->get_body_a()) : nullptr) {
				j.a = *a;
			}
			if (const uint32_t *b = joint->get_body_b() ? index.getptr(joint->get_body_b()) : nullptr) {
				j.b = *b;
			}
			out.push_back(j);
		}
	}
	struct ByKey {
		bool operator()(const SnapshotJoint &p_x, const SnapshotJoint &p_y) const {
			if (p_x.a != p_y.a) {
				return p_x.a < p_y.a;
			}
			if (p_x.b != p_y.b) {
				return p_x.b < p_y.b;
			}
			if (p_x.type != p_y.type) {
				return p_x.type < p_y.type;
			}
			return p_x.joint->get_rid().get_id() < p_y.joint->get_rid().get_id();
		}
	};
	out.sort_custom<ByKey>();
	return out;
}

static void _snapshot_write(JPH::StateRecorder &p_recorder, const Vector3 &p_v) {
	p_recorder.Write(p_v.x);
	p_recorder.Write(p_v.y);
	p_recorder.Write(p_v.z);
}

static void _snapshot_read(JPH::StateRecorder &p_recorder, Vector3 &r_v) {
	p_recorder.Read(r_v.x);
	p_recorder.Read(r_v.y);
	p_recorder.Read(r_v.z);
}

// The state, the same for the same world however it was made: the objects in
// the order of their body ids; the default area's and the objects' body ids
// and each one's place among its owner's (in the order they were made); the
// objects whose queries (state sync, area events) are due at the next sync,
// in their order; the objects whose contact cache a change made invalid for
// the next step; what the last step left each body for the game to read
// (gravity, damping, contacts, a contact's collider by its place in the
// list); each joint's solver state (its warm start, found again by its
// bodies' places and its type); the active bodies in Jolt's order (each step
// goes by it) and the body ids it gives out next (Jolt's own state leaves
// both out); then Jolt's own state of the system (bodies
// and contacts; its constraints are the joints', made again in their own
// order).
Dictionary JoltSpace3D::snapshot_get_state() {
	LocalVector<JoltObject3D *> objects = snapshot_objects();
	HashMap<const JoltObject3D *, uint32_t> rank;
	HashMap<uint64_t, uint32_t> made;
	for (const JoltObject3D *object : objects) {
		rank[object] = made[uint64_t(object->get_instance_id())]++;
	}
	struct IdFirst {
		bool operator()(const JoltObject3D *p_a, const JoltObject3D *p_b) const { return p_a->get_jolt_id() < p_b->get_jolt_id(); }
	};
	objects.sort_custom<IdFirst>();
	JPH::StateRecorderImpl recorder;
	const uint32_t count = objects.size();
	recorder.Write(count);
	recorder.Write(default_area != nullptr && default_area->in_space() ? default_area->get_jolt_id() : JPH::BodyID());
	PackedInt64Array owners;
	owners.resize(count);
	HashMap<const JoltObject3D *, uint32_t> index;
	for (uint32_t i = 0; i < count; i++) {
		recorder.Write(objects[i]->get_jolt_id());
		recorder.Write(rank[objects[i]]);
		owners.set(i, int64_t(uint64_t(objects[i]->get_instance_id())));
		index[objects[i]] = i;
	}
	LocalVector<uint32_t> queued;
	for (SelfList<JoltBody3D> *e = body_call_queries_list.first(); e != nullptr; e = e->next()) {
		if (const uint32_t *i = index.getptr(e->self())) {
			queued.push_back(*i);
		}
	}
	for (SelfList<JoltArea3D> *e = area_call_queries_list.first(); e != nullptr; e = e->next()) {
		if (const uint32_t *i = index.getptr(e->self())) {
			queued.push_back(*i);
		}
	}
	recorder.Write(queued.size());
	for (uint32_t i : queued) {
		recorder.Write(i);
	}
	LocalVector<uint32_t> invalid;
	const JPH::BodyLockInterface &lock_iface = get_lock_iface();
	for (uint32_t i = 0; i < count; i++) {
		const JPH::Body *jolt_body = lock_iface.TryGetBody(objects[i]->get_jolt_id());
		if (jolt_body != nullptr && jolt_body->IsCollisionCacheInvalid()) {
			invalid.push_back(i);
		}
	}
	recorder.Write(invalid.size());
	for (uint32_t i : invalid) {
		recorder.Write(i);
	}
	HashMap<RID, uint32_t> by_rid;
	for (uint32_t i = 0; i < count; i++) {
		by_rid[objects[i]->get_rid()] = i;
	}
	for (uint32_t i = 0; i < count; i++) {
		JoltBody3D *body = objects[i]->as_body();
		if (body == nullptr) {
			continue;
		}
		_snapshot_write(recorder, body->get_gravity());
		recorder.Write(body->get_total_linear_damp());
		recorder.Write(body->get_total_angular_damp());
		const uint32_t contact_count = body->get_contact_count();
		recorder.Write(contact_count);
		for (uint32_t k = 0; k < contact_count; k++) {
			const JoltBody3D::Contact &c = body->get_contact(k);
			_snapshot_write(recorder, c.normal);
			_snapshot_write(recorder, c.position);
			_snapshot_write(recorder, c.collider_position);
			_snapshot_write(recorder, c.velocity);
			_snapshot_write(recorder, c.collider_velocity);
			_snapshot_write(recorder, c.impulse);
			recorder.Write(c.depth);
			recorder.Write(c.shape_index);
			recorder.Write(c.collider_shape_index);
			const uint32_t *collider = by_rid.getptr(c.collider_rid);
			recorder.Write(collider != nullptr ? *collider : UINT32_MAX);
		}
	}
	const LocalVector<SnapshotJoint> joints = _snapshot_joints(objects);
	HashMap<const JPH::Constraint *, uint32_t> solve_order; // the solver goes by the system's order
	{
		const JPH::Constraints constraints = physics_system->GetConstraints();
		for (uint32_t k = 0; k < constraints.size(); k++) {
			solve_order[constraints[k].GetPtr()] = k;
		}
	}
	recorder.Write(joints.size());
	for (const SnapshotJoint &j : joints) {
		recorder.Write(j.a);
		recorder.Write(j.b);
		recorder.Write(j.type);
		const uint32_t *order = j.joint->get_jolt_ref() != nullptr ? solve_order.getptr(j.joint->get_jolt_ref()) : nullptr;
		recorder.Write(order != nullptr ? *order : UINT32_MAX);
		std::string blob;
		if (j.joint->get_jolt_ref() != nullptr) {
			JPH::StateRecorderImpl joint_recorder;
			j.joint->get_jolt_ref()->SaveState(joint_recorder);
			blob = joint_recorder.GetData();
		}
		recorder.Write(uint32_t(blob.size()));
		recorder.WriteBytes(blob.data(), blob.size());
	}
	// The active bodies in Jolt's order (each step goes by it: the queries
	// due are queued in it), which its own state leaves out.
	{
		HashMap<uint32_t, uint32_t> by_id;
		for (uint32_t i = 0; i < count; i++) {
			by_id[objects[i]->get_jolt_id().GetIndexAndSequenceNumber()] = i;
		}
		for (const JPH::EBodyType type : { JPH::EBodyType::RigidBody, JPH::EBodyType::SoftBody }) {
			JPH::BodyIDVector active;
			physics_system->GetActiveBodies(type, active);
			LocalVector<uint32_t> listed;
			for (const JPH::BodyID &id : active) {
				if (const uint32_t *i = by_id.getptr(id.GetIndexAndSequenceNumber())) {
					listed.push_back(*i);
				}
			}
			recorder.Write(listed.size());
			for (uint32_t i : listed) {
				recorder.Write(i);
			}
		}
	}
	physics_system->SaveBodyIDState(recorder);
	physics_system->SaveState(recorder, JPH::EStateRecorderState(uint8_t(JPH::EStateRecorderState::Global) | uint8_t(JPH::EStateRecorderState::Bodies) | uint8_t(JPH::EStateRecorderState::Contacts)));
	const std::string data = recorder.GetData();
	PackedByteArray state;
	state.resize(data.size());
	memcpy(state.ptrw(), data.data(), data.size());
	Dictionary out;
	out["owners"] = owners;
	out["state"] = state;
	return out;
}

// Each saved object is found again by its owner (the k-th of an owner's, in
// the order they were made), gets the body id it had (made again with it
// where it has another), then Jolt's state is set. Fails before anything is
// changed unless the space holds the same objects by owner, and its default
// area is the one saved.
bool JoltSpace3D::snapshot_set_state(const PackedInt64Array &p_owners, const PackedByteArray &p_state) {
	JPH::StateRecorderImpl recorder;
	recorder.WriteBytes(p_state.ptr(), p_state.size());
	uint32_t count = 0;
	recorder.Read(count);
	ERR_FAIL_COND_V_MSG(count != uint32_t(p_owners.size()), false, vformat("Physics snapshot holds %d objects, %d owners given.", count, p_owners.size()));
	const LocalVector<JoltObject3D *> current = snapshot_objects();
	ERR_FAIL_COND_V_MSG(count != current.size(), false, vformat("Physics snapshot holds %d objects, the space %d.", count, current.size()));
	HashMap<uint64_t, LocalVector<JoltObject3D *>> by_owner;
	for (JoltObject3D *object : current) {
		by_owner[uint64_t(object->get_instance_id())].push_back(object);
	}
	JPH::BodyID area_id;
	recorder.Read(area_id);
	const JPH::BodyID current_area_id = default_area != nullptr && default_area->in_space() ? default_area->get_jolt_id() : JPH::BodyID();
	ERR_FAIL_COND_V_MSG(area_id != current_area_id, false, "Physics snapshot of another space (its default area differs).");
	LocalVector<JPH::BodyID> ids;
	ids.resize(count);
	LocalVector<JoltObject3D *> objects;
	objects.resize(count);
	HashSet<JoltObject3D *> found;
	for (uint32_t i = 0; i < count; i++) {
		uint32_t k = 0;
		recorder.Read(ids[i]);
		recorder.Read(k);
		LocalVector<JoltObject3D *> *own = by_owner.getptr(uint64_t(p_owners[i]));
		ERR_FAIL_COND_V_MSG(own == nullptr || k >= own->size(), false, "Physics snapshot: an owner has fewer areas and bodies than saved.");
		objects[i] = (*own)[k];
		ERR_FAIL_COND_V_MSG(found.has(objects[i]), false, "Physics snapshot: broken owner places.");
		found.insert(objects[i]);
	}
	uint32_t queued_count = 0;
	recorder.Read(queued_count);
	LocalVector<uint32_t> queued;
	queued.resize(queued_count);
	for (uint32_t &i : queued) {
		recorder.Read(i);
		ERR_FAIL_COND_V_MSG(i >= count, false, "Physics snapshot: broken queue.");
	}
	uint32_t invalid_count = 0;
	recorder.Read(invalid_count);
	LocalVector<uint32_t> invalid;
	invalid.resize(invalid_count);
	for (uint32_t &i : invalid) {
		recorder.Read(i);
		ERR_FAIL_COND_V_MSG(i >= count, false, "Physics snapshot: broken contact cache list.");
	}
	struct Step {
		Vector3 gravity;
		float linear_damp = 0.0f;
		float angular_damp = 0.0f;
		LocalVector<JoltBody3D::Contact> contacts;
	};
	HashMap<uint32_t, Step> steps;
	for (uint32_t i = 0; i < count; i++) {
		if (objects[i]->as_body() == nullptr) {
			continue;
		}
		Step &step = steps[i];
		_snapshot_read(recorder, step.gravity);
		recorder.Read(step.linear_damp);
		recorder.Read(step.angular_damp);
		uint32_t contact_count = 0;
		recorder.Read(contact_count);
		ERR_FAIL_COND_V_MSG(contact_count > MAX_CONTACTS_REPORTED_3D_MAX, false, "Physics snapshot: broken contacts.");
		step.contacts.resize(contact_count);
		for (JoltBody3D::Contact &c : step.contacts) {
			_snapshot_read(recorder, c.normal);
			_snapshot_read(recorder, c.position);
			_snapshot_read(recorder, c.collider_position);
			_snapshot_read(recorder, c.velocity);
			_snapshot_read(recorder, c.collider_velocity);
			_snapshot_read(recorder, c.impulse);
			recorder.Read(c.depth);
			recorder.Read(c.shape_index);
			recorder.Read(c.collider_shape_index);
			uint32_t collider = UINT32_MAX;
			recorder.Read(collider);
			if (collider < count) {
				c.collider_id = objects[collider]->get_instance_id();
				c.collider_rid = objects[collider]->get_rid();
			}
		}
	}
	uint32_t joint_count = 0;
	recorder.Read(joint_count);
	const LocalVector<SnapshotJoint> joints = _snapshot_joints(objects);
	ERR_FAIL_COND_V_MSG(joint_count != joints.size(), false, vformat("Physics snapshot holds %d joints, the space %d.", joint_count, joints.size()));
	LocalVector<std::string> joint_states;
	joint_states.resize(joint_count);
	LocalVector<uint32_t> joint_orders;
	joint_orders.resize(joint_count);
	for (uint32_t i = 0; i < joint_count; i++) {
		SnapshotJoint j;
		recorder.Read(j.a);
		recorder.Read(j.b);
		recorder.Read(j.type);
		ERR_FAIL_COND_V_MSG(j.a != joints[i].a || j.b != joints[i].b || j.type != joints[i].type, false, "Physics snapshot: the joints differ.");
		recorder.Read(joint_orders[i]);
		uint32_t size = 0;
		recorder.Read(size);
		ERR_FAIL_COND_V_MSG(size > p_state.size(), false, "Physics snapshot: broken joint state.");
		joint_states[i].resize(size);
		recorder.ReadBytes(joint_states[i].data(), size);
	}
	LocalVector<uint32_t> active[2];
	for (LocalVector<uint32_t> &listed : active) {
		uint32_t active_count = 0;
		recorder.Read(active_count);
		ERR_FAIL_COND_V_MSG(active_count > count, false, "Physics snapshot: broken active list.");
		listed.resize(active_count);
		for (uint32_t &i : listed) {
			recorder.Read(i);
			ERR_FAIL_COND_V_MSG(i >= count, false, "Physics snapshot: broken active list.");
		}
	}
	// Out first, all of them: an id one wants back may be another one's now.
	LocalVector<uint32_t> moved;
	for (uint32_t i = 0; i < count; i++) {
		if (objects[i]->get_jolt_id() != ids[i]) {
			objects[i]->snapshot_take_out();
			moved.push_back(i);
		}
	}
	for (uint32_t i : moved) {
		snapshot_object = objects[i];
		snapshot_body_id = ids[i];
		objects[i]->snapshot_put_back();
		snapshot_object = nullptr;
	}
	flush_pending_objects();
	for (uint32_t i = 0; i < count; i++) {
		ERR_FAIL_COND_V_MSG(objects[i]->get_jolt_id() != ids[i], false, "Physics snapshot: a body could not get its saved id back.");
	}
	// The ids given out next: those that would have been.
	ERR_FAIL_COND_V_MSG(!physics_system->RestoreBodyIDState(recorder), false, "Physics snapshot: broken body id state.");
	// Active in the saved order: all put to sleep, then woken in it. Jolt's
	// state then wakes none (they are as saved) and sets their velocities.
	{
		JPH::BodyInterface &body_iface = physics_system->GetBodyInterfaceNoLock();
		for (const JPH::EBodyType type : { JPH::EBodyType::RigidBody, JPH::EBodyType::SoftBody }) {
			JPH::BodyIDVector awake;
			physics_system->GetActiveBodies(type, awake);
			if (!awake.empty()) {
				body_iface.DeactivateBodies(awake.data(), int(awake.size()));
			}
		}
		for (const LocalVector<uint32_t> &listed : active) {
			JPH::BodyIDVector wake;
			for (uint32_t i : listed) {
				wake.push_back(ids[i]);
			}
			if (!wake.empty()) {
				body_iface.ActivateBodies(wake.data(), int(wake.size()));
			}
		}
	}
	if (!physics_system->RestoreState(recorder)) {
		return false;
	}
	// The contact caches valid as saved: the load set shapes (made the bodies'
	// collision again), each of which made its body's invalid.
	physics_system->ResetContactCacheInvalidation();
	for (uint32_t i : invalid) {
		get_body_iface().InvalidateContactCache(ids[i]);
	}
	// Joints after the bodies are back: putting a body back makes its joints
	// again, and the system's order of them (the solver's) changes. Taken out
	// and added again in the saved order, the joints' constraints (the only
	// ones) are in that order.
	struct Ordered {
		uint32_t order = 0;
		JPH::Constraint *constraint = nullptr;
		bool operator<(const Ordered &p_other) const { return order < p_other.order; }
	};
	LocalVector<Ordered> ordered;
	HashSet<const JPH::Constraint *> in_system;
	for (const JPH::Ref<JPH::Constraint> &c : physics_system->GetConstraints()) {
		in_system.insert(c.GetPtr());
	}
	for (uint32_t i = 0; i < joint_count; i++) {
		JPH::Constraint *constraint = joints[i].joint->get_jolt_ref();
		if (constraint == nullptr) {
			continue;
		}
		if (!joint_states[i].empty()) {
			JPH::StateRecorderImpl joint_recorder;
			joint_recorder.WriteBytes(joint_states[i].data(), joint_states[i].size());
			constraint->RestoreState(joint_recorder);
		}
		if (in_system.has(constraint)) {
			ordered.push_back({ joint_orders[i], constraint });
		}
	}
	ordered.sort();
	for (const Ordered &o : ordered) {
		physics_system->RemoveConstraint(o.constraint);
	}
	for (const Ordered &o : ordered) {
		physics_system->AddConstraint(o.constraint);
	}
	for (KeyValue<uint32_t, Step> &E : steps) {
		objects[E.key]->as_body()->snapshot_set_step(E.value.gravity, E.value.linear_damp, E.value.angular_damp, E.value.contacts);
	}
	// The queries due at the next sync: those, in that order.
	for (JoltObject3D *object : objects) {
		if (JoltBody3D *body = object->as_body()) {
			dequeue_call_queries(body->get_call_queries_element());
		} else if (JoltArea3D *area = object->as_area()) {
			dequeue_call_queries(area->get_call_queries_element());
		}
	}
	for (int64_t k = int64_t(queued.size()) - 1; k >= 0; k--) { // each goes first: the last one first
		const uint32_t i = queued[k];
		if (JoltBody3D *body = objects[i]->as_body()) {
			enqueue_call_queries(body->get_call_queries_element());
		} else if (JoltArea3D *area = objects[i]->as_area()) {
			enqueue_call_queries(area->get_call_queries_element());
		}
	}
	return true;
}
