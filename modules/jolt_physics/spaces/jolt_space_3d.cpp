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
#include "../shapes/jolt_custom_shape_type.h"
#include "../shapes/jolt_shape_3d.h"
#include "jolt_body_activation_listener_3d.h"
#include "jolt_contact_listener_3d.h"
#include "jolt_layers.h"
#include "jolt_physics_direct_space_state_3d.h"
#include "jolt_temp_allocator.h"

#include "core/io/file_access.h"
#include "core/os/time.h"
#include "core/string/print_string.h"
#include "core/variant/variant_utility.h"

#include "Jolt/Physics/Collision/CollideShapeVsShapePerLeaf.h"
#include "Jolt/Physics/Collision/CollisionCollectorImpl.h"
#include "Jolt/Physics/PhysicsScene.h"

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
		object->pre_step(p_step, *jolt_body);
	}
}

void JoltSpace3D::_post_step(float p_step) {
	contact_listener->post_step();

	while (shapes_changed_list.first()) {
		JoltShapedObject3D *object = shapes_changed_list.first()->self();
		shapes_changed_list.remove(shapes_changed_list.first());
		object->clear_previous_shape();
	}
}

JoltSpace3D::JoltSpace3D(JPH::JobSystem *p_job_system) :
		job_system(p_job_system),
		temp_allocator(new JoltTempAllocator()),
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
	physics_system->SetBodyActivationListener(body_activation_listener);

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

	if (temp_allocator != nullptr) {
		delete temp_allocator;
		temp_allocator = nullptr;
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

void JoltSpace3D::set_default_area(JoltArea3D *p_area) {
	if (default_area == p_area) {
		return;
	}

	if (default_area != nullptr) {
		default_area->set_default_area(false);
	}

	default_area = p_area;

	if (default_area != nullptr) {
		default_area->set_default_area(true);
	}
}

JPH::Body *JoltSpace3D::add_object(const JoltObject3D &p_object, const JPH::BodyCreationSettings &p_settings, bool p_sleeping) {
	JPH::BodyInterface &body_iface = get_body_iface();
	JPH::Body *jolt_body = body_iface.CreateBody(p_settings);
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
	JPH::Body *jolt_body = body_iface.CreateSoftBody(p_settings);
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
			// Since we do our own integration of gravity and damping, while leaving Jolt's own values at zero, we need to transfer over the correct values.
			settings.mGravityFactor = body->get_gravity_scale();
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
