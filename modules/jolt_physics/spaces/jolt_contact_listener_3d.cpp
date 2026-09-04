/**************************************************************************/
/*  jolt_contact_listener_3d.cpp                                          */
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

#include "jolt_contact_listener_3d.h"

#include "../jolt_project_settings.h"
#include "../misc/jolt_type_conversions.h"
#include "../objects/jolt_area_3d.h"
#include "../objects/jolt_body_3d.h"
#include "../objects/jolt_soft_body_3d.h"
#include "jolt_space_3d.h"

#include "core/templates/a_hash_map.h"

#include <Jolt/Physics/Collision/EstimateCollisionResponse.h>
#include <Jolt/Physics/SoftBody/SoftBodyManifold.h>

void JoltContactListener3D::OnContactAdded(const JPH::Body &p_body1, const JPH::Body &p_body2, const JPH::ContactManifold &p_manifold, JPH::ContactSettings &p_settings) {
	_try_override_collision_response(p_body1, p_body2, p_settings);
	_try_apply_surface_velocities(p_body1, p_body2, p_settings);
	_try_add_contacts(p_body1, p_body2, p_manifold, p_settings);
	_try_evaluate_area_overlap(p_body1, p_body2, p_manifold.mSubShapeID1, p_manifold.mSubShapeID2);

#ifdef DEBUG_ENABLED
	_try_add_debug_contacts(p_body1, p_body2, p_manifold);
#endif
}

void JoltContactListener3D::OnContactPersisted(const JPH::Body &p_body1, const JPH::Body &p_body2, const JPH::ContactManifold &p_manifold, JPH::ContactSettings &p_settings) {
	_try_override_collision_response(p_body1, p_body2, p_settings);
	_try_apply_surface_velocities(p_body1, p_body2, p_settings);
	_try_add_contacts(p_body1, p_body2, p_manifold, p_settings);
	_try_evaluate_area_overlap(p_body1, p_body2, p_manifold.mSubShapeID1, p_manifold.mSubShapeID2);

#ifdef DEBUG_ENABLED
	_try_add_debug_contacts(p_body1, p_body2, p_manifold);
#endif
}

void JoltContactListener3D::OnContactRemoved(const JPH::SubShapeIDPair &p_shape_pair) {
	_try_remove_area_overlap(p_shape_pair);
}

JPH::SoftBodyValidateResult JoltContactListener3D::OnSoftBodyContactValidate(const JPH::Body &p_soft_body, const JPH::Body &p_other_body, JPH::SoftBodyContactSettings &p_settings) {
	_try_override_collision_response(p_soft_body, p_other_body, p_settings);
	return JPH::SoftBodyValidateResult::AcceptContact;
}

void JoltContactListener3D::OnSoftBodyContactAdded(const JPH::Body &p_soft_body, const JPH::SoftBodyManifold &p_manifold) {
	for (JPH::uint i = 0; i < p_manifold.GetNumSensorContacts(); i++) {
		if (JPH::Body *other_jolt_body = space->try_get_jolt_body(p_manifold.GetSensorContactBodyID(i))) {
			_try_evaluate_area_overlap(p_soft_body, *other_jolt_body, JPH::SubShapeID(), JPH::SubShapeID());
		}
	}

#ifdef DEBUG_ENABLED
	_try_add_debug_contacts(p_soft_body, p_manifold);
#endif
}

bool JoltContactListener3D::_try_override_collision_response(const JPH::Body &p_jolt_body1, const JPH::Body &p_jolt_body2, JPH::ContactSettings &p_settings) {
	if (p_jolt_body1.IsSensor() || p_jolt_body2.IsSensor()) {
		return false;
	}

	if (!p_jolt_body1.IsDynamic() && !p_jolt_body2.IsDynamic()) {
		return false;
	}

	const JoltBody3D *body1 = reinterpret_cast<JoltBody3D *>(p_jolt_body1.GetUserData());
	const JoltBody3D *body2 = reinterpret_cast<JoltBody3D *>(p_jolt_body2.GetUserData());

	const bool can_collide1 = body1->can_collide_with(*body2);
	const bool can_collide2 = body2->can_collide_with(*body1);

	if (can_collide1 && !can_collide2) {
		p_settings.mInvMassScale2 = 0.0f;
		p_settings.mInvInertiaScale2 = 0.0f;
	} else if (can_collide2 && !can_collide1) {
		p_settings.mInvMassScale1 = 0.0f;
		p_settings.mInvInertiaScale1 = 0.0f;
	}

	return true;
}

bool JoltContactListener3D::_try_override_collision_response(const JPH::Body &p_jolt_soft_body, const JPH::Body &p_jolt_other_body, JPH::SoftBodyContactSettings &p_settings) {
	if (p_jolt_other_body.IsSensor()) {
		return false;
	}

	const JoltSoftBody3D *soft_body = reinterpret_cast<JoltSoftBody3D *>(p_jolt_soft_body.GetUserData());
	const JoltBody3D *other_body = reinterpret_cast<JoltBody3D *>(p_jolt_other_body.GetUserData());

	const bool can_collide1 = soft_body->can_collide_with(*other_body);
	const bool can_collide2 = other_body->can_collide_with(*soft_body);

	if (can_collide1 && !can_collide2) {
		p_settings.mInvMassScale2 = 0.0f;
		p_settings.mInvInertiaScale2 = 0.0f;
	} else if (can_collide2 && !can_collide1) {
		p_settings.mInvMassScale1 = 0.0f;
	}

	return true;
}

bool JoltContactListener3D::_try_apply_surface_velocities(const JPH::Body &p_jolt_body1, const JPH::Body &p_jolt_body2, JPH::ContactSettings &p_settings) {
	if (p_jolt_body1.IsSensor() || p_jolt_body2.IsSensor()) {
		return false;
	}

	const bool supports_surface_velocity1 = !p_jolt_body1.IsDynamic();
	const bool supports_surface_velocity2 = !p_jolt_body2.IsDynamic();

	if (supports_surface_velocity1 == supports_surface_velocity2) {
		return false;
	}

	const JoltBody3D *body1 = reinterpret_cast<JoltBody3D *>(p_jolt_body1.GetUserData());
	const JoltBody3D *body2 = reinterpret_cast<JoltBody3D *>(p_jolt_body2.GetUserData());

	const bool has_surface_velocity1 = supports_surface_velocity1 && (body1->get_linear_surface_velocity() != Vector3() || body1->get_angular_surface_velocity() != Vector3());
	const bool has_surface_velocity2 = supports_surface_velocity2 && (body2->get_linear_surface_velocity() != Vector3() || body2->get_angular_surface_velocity() != Vector3());

	if (has_surface_velocity1 == has_surface_velocity2) {
		return false;
	}

	const JPH::Vec3 linear_velocity1 = to_jolt(body1->get_linear_surface_velocity());
	const JPH::Vec3 angular_velocity1 = to_jolt(body1->get_angular_surface_velocity());

	const JPH::Vec3 linear_velocity2 = to_jolt(body2->get_linear_surface_velocity());
	const JPH::Vec3 angular_velocity2 = to_jolt(body2->get_angular_surface_velocity());

	const JPH::RVec3 com1 = p_jolt_body1.GetCenterOfMassPosition();
	const JPH::RVec3 com2 = p_jolt_body2.GetCenterOfMassPosition();
	const JPH::Vec3 rel_com2 = JPH::Vec3(com2 - com1);

	const JPH::Vec3 angular_linear_velocity2 = rel_com2.Cross(angular_velocity2);
	const JPH::Vec3 total_linear_velocity2 = linear_velocity2 + angular_linear_velocity2;

	p_settings.mRelativeLinearSurfaceVelocity = total_linear_velocity2 - linear_velocity1;
	p_settings.mRelativeAngularSurfaceVelocity = angular_velocity2 - angular_velocity1;

	return true;
}

bool JoltContactListener3D::_try_add_contacts(const JPH::Body &p_jolt_body1, const JPH::Body &p_jolt_body2, const JPH::ContactManifold &p_manifold, JPH::ContactSettings &p_settings) {
	if (p_jolt_body1.IsSensor() || p_jolt_body2.IsSensor()) {
		return false;
	}

	const JoltBody3D *body1 = reinterpret_cast<JoltBody3D *>(p_jolt_body1.GetUserData());
	const JoltBody3D *body2 = reinterpret_cast<JoltBody3D *>(p_jolt_body2.GetUserData());

	if (!body1->reports_contacts() && !body2->reports_contacts()) {
		return false;
	}

	LocalVector<Manifold> &manifolds = _get_thread_locals().manifolds;
	manifolds.resize(manifolds.size() + 1);
	Manifold &manifold = manifolds[manifolds.size() - 1];

	const JPH::uint contact_count = p_manifold.mRelativeContactPointsOn1.size();

	manifold.shape_pair = JPH::SubShapeIDPair(p_jolt_body1.GetID(), p_manifold.mSubShapeID1, p_jolt_body2.GetID(), p_manifold.mSubShapeID2);
	manifold.normal1 = to_godot(-p_manifold.mWorldSpaceNormal);
	manifold.depth = p_manifold.mPenetrationDepth;

	JPH::CollisionEstimationResult collision;
	JPH::EstimateCollisionResponse(p_jolt_body1, p_jolt_body2, p_manifold, collision, p_settings.mCombinedFriction, p_settings.mCombinedRestitution, JoltProjectSettings::bounce_velocity_threshold, 5);

	const JPH::Vec3 friction_impulse1 = contact_count > 0 ? (collision.mTangent1 * collision.mFrictionImpulse1) / contact_count : JPH::Vec3::sZero();
	const JPH::Vec3 friction_impulse2 = contact_count > 0 ? (collision.mTangent2 * collision.mFrictionImpulse2) / contact_count : JPH::Vec3::sZero();

	manifold.contacts.resize(contact_count);

	for (JPH::uint i = 0; i < contact_count; ++i) {
		const JPH::RVec3 relative_point1 = JPH::RVec3(p_manifold.mRelativeContactPointsOn1[i]);
		const JPH::RVec3 relative_point2 = JPH::RVec3(p_manifold.mRelativeContactPointsOn2[i]);

		const JPH::RVec3 world_point1 = p_manifold.mBaseOffset + relative_point1;
		const JPH::RVec3 world_point2 = p_manifold.mBaseOffset + relative_point2;

		const JPH::Vec3 velocity1 = p_jolt_body1.GetPointVelocity(world_point1);
		const JPH::Vec3 velocity2 = p_jolt_body2.GetPointVelocity(world_point2);

		const JPH::Vec3 contact_impulse = p_manifold.mWorldSpaceNormal * collision.mContactImpulse[i];
		const JPH::Vec3 combined_impulse = contact_impulse + friction_impulse1 + friction_impulse2;

		Contact &contact = manifold.contacts[i];
		contact.point1 = to_godot(world_point1);
		contact.point2 = to_godot(world_point2);
		contact.velocity1 = to_godot(velocity1);
		contact.velocity2 = to_godot(velocity2);
		contact.impulse1 = to_godot(-combined_impulse);
	}

	return true;
}

bool JoltContactListener3D::_try_evaluate_area_overlap(const JPH::Body &p_body1, const JPH::Body &p_body2, const JPH::SubShapeID &p_shape_id1, const JPH::SubShapeID &p_shape_id2) {
	if (!p_body1.IsSensor() && !p_body2.IsSensor()) {
		return false;
	}

	const JPH::SubShapeIDPair shape_pair1(p_body1.GetID(), p_shape_id1, p_body2.GetID(), p_shape_id2);
	const JPH::SubShapeIDPair shape_pair2(p_body2.GetID(), p_shape_id2, p_body1.GetID(), p_shape_id1);

	const JoltObject3D *object1 = reinterpret_cast<JoltObject3D *>(p_body1.GetUserData());
	const JoltObject3D *object2 = reinterpret_cast<JoltObject3D *>(p_body2.GetUserData());

	const JoltArea3D *area1 = object1->as_area();
	const JoltArea3D *area2 = object2->as_area();

	const JoltBody3D *body1 = object1->as_body();
	const JoltBody3D *body2 = object2->as_body();

	const JoltSoftBody3D *soft_body1 = object1->as_soft_body();
	const JoltSoftBody3D *soft_body2 = object2->as_soft_body();

	if (area1 != nullptr && area2 != nullptr) {
		_evaluate_area_overlap(*area1, *area2, shape_pair1);
		_evaluate_area_overlap(*area2, *area1, shape_pair2);
	} else if (area1 != nullptr && body2 != nullptr) {
		_evaluate_area_overlap(*area1, *body2, shape_pair1);
	} else if (area2 != nullptr && body1 != nullptr) {
		_evaluate_area_overlap(*area2, *body1, shape_pair2);
	} else if (area1 != nullptr && soft_body2 != nullptr) {
		_evaluate_area_overlap(*area1, *soft_body2, shape_pair1);
	} else if (area2 != nullptr && soft_body1 != nullptr) {
		_evaluate_area_overlap(*area2, *soft_body1, shape_pair2);
	}

	return true;
}

void JoltContactListener3D::_try_remove_area_overlap(const JPH::SubShapeIDPair &p_shape_pair) {
	const JPH::SubShapeIDPair swapped_shape_pair(p_shape_pair.GetBody2ID(), p_shape_pair.GetSubShapeID2(), p_shape_pair.GetBody1ID(), p_shape_pair.GetSubShapeID1());

	if (area_overlaps.has(p_shape_pair)) {
		_get_thread_locals().area_exits.push_back(p_shape_pair);
	}

	if (area_overlaps.has(swapped_shape_pair)) {
		_get_thread_locals().area_exits.push_back(swapped_shape_pair);
	}
}

#ifdef DEBUG_ENABLED

bool JoltContactListener3D::_try_add_debug_contacts(const JPH::Body &p_body1, const JPH::Body &p_body2, const JPH::ContactManifold &p_manifold) {
	if (p_body1.IsSensor() || p_body2.IsSensor()) {
		return false;
	}

	const int64_t max_count = debug_contacts.size();

	if (max_count == 0) {
		return false;
	}

	const int additional_pairs = (int)p_manifold.mRelativeContactPointsOn1.size();
	const int additional_contacts = additional_pairs * 2;

	int current_count = debug_contact_count.load(std::memory_order_relaxed);
	bool exchanged = false;

	do {
		const int new_count = current_count + additional_contacts;

		if (new_count > max_count) {
			return false;
		}

		exchanged = debug_contact_count.compare_exchange_weak(current_count, new_count, std::memory_order_release, std::memory_order_relaxed);
	} while (!exchanged);

	for (int i = 0; i < additional_pairs; ++i) {
		const int pair_index = current_count + i * 2;

		const JPH::RVec3 point_on_1 = p_manifold.GetWorldSpaceContactPointOn1((JPH::uint)i);
		const JPH::RVec3 point_on_2 = p_manifold.GetWorldSpaceContactPointOn2((JPH::uint)i);

		debug_contacts.write[pair_index + 0] = to_godot(point_on_1);
		debug_contacts.write[pair_index + 1] = to_godot(point_on_2);
	}

	return true;
}

bool JoltContactListener3D::_try_add_debug_contacts(const JPH::Body &p_soft_body, const JPH::SoftBodyManifold &p_manifold) {
	const int64_t max_count = debug_contacts.size();
	if (max_count == 0) {
		return false;
	}

	int additional_contacts = 0;

	for (const JPH::SoftBodyVertex &vertex : p_manifold.GetVertices()) {
		if (p_manifold.HasContact(vertex)) {
			additional_contacts += 1;
		}
	}

	int current_count = debug_contact_count.load(std::memory_order_relaxed);
	bool exchanged = false;

	do {
		const int new_count = current_count + additional_contacts;

		if (new_count > max_count) {
			return false;
		}

		exchanged = debug_contact_count.compare_exchange_weak(current_count, new_count, std::memory_order_release, std::memory_order_relaxed);
	} while (!exchanged);

	int contact_index = current_count;

	for (const JPH::SoftBodyVertex &vertex : p_manifold.GetVertices()) {
		if (!p_manifold.HasContact(vertex)) {
			continue;
		}

		const JPH::RMat44 body_com_transform = p_soft_body.GetCenterOfMassTransform();
		const JPH::Vec3 local_contact_point = p_manifold.GetLocalContactPoint(vertex);
		const JPH::RVec3 contact_point = body_com_transform * local_contact_point;

		debug_contacts.write[contact_index++] = to_godot(contact_point);
	}

	return true;
}

#endif

bool JoltContactListener3D::_has_shape_shifted(const JoltShapedObject3D &p_object, const JPH::SubShapeID &p_sub_shape_id) {
	return p_object.get_previous_jolt_shape() != nullptr && p_object.get_jolt_shape()->GetSubShapeUserData(p_sub_shape_id) != p_object.get_previous_jolt_shape()->GetSubShapeUserData(p_sub_shape_id);
}

void JoltContactListener3D::_evaluate_area_overlap(const JoltArea3D &p_area, const JoltArea3D &p_other_area, const JPH::SubShapeIDPair &p_shape_pair) {
	if (p_area.can_monitor(p_other_area)) {
		if (!area_overlaps.has(p_shape_pair)) {
			_get_thread_locals().area_enters.push_back(p_shape_pair);
		} else if (_has_shape_shifted(p_area, p_shape_pair.GetSubShapeID1()) || _has_shape_shifted(p_other_area, p_shape_pair.GetSubShapeID2())) {
			// A shape has taken on the `JPH::SubShapeID` value of another shape, likely because of the other shape having been replaced or moved
			// in some way, so we force the area to refresh its internal mappings by exiting and entering this shape pair.
			ThreadLocals &tl = _get_thread_locals();
			tl.area_exits.push_back(p_shape_pair);
			tl.area_enters.push_back(p_shape_pair);
		}
	} else {
		if (area_overlaps.has(p_shape_pair)) {
			_get_thread_locals().area_exits.push_back(p_shape_pair);
		}
	}
}

void JoltContactListener3D::_evaluate_area_overlap(const JoltArea3D &p_area, const JoltBody3D &p_body, const JPH::SubShapeIDPair &p_shape_pair) {
	if (p_area.can_monitor(p_body)) {
		if (!area_overlaps.has(p_shape_pair)) {
			_get_thread_locals().area_enters.push_back(p_shape_pair);
		} else if (_has_shape_shifted(p_area, p_shape_pair.GetSubShapeID1()) || _has_shape_shifted(p_body, p_shape_pair.GetSubShapeID2())) {
			// A shape has taken on the `JPH::SubShapeID` value of another shape, likely because of the other shape having been replaced or moved
			// in some way, so we force the area to refresh its internal mappings by exiting and entering this shape pair.
			ThreadLocals &tl = _get_thread_locals();
			tl.area_exits.push_back(p_shape_pair);
			tl.area_enters.push_back(p_shape_pair);
		}
	} else {
		if (area_overlaps.has(p_shape_pair)) {
			_get_thread_locals().area_exits.push_back(p_shape_pair);
		}
	}
}

void JoltContactListener3D::_evaluate_area_overlap(const JoltArea3D &p_area, const JoltSoftBody3D &p_body, const JPH::SubShapeIDPair &p_shape_pair) {
	if (p_area.can_monitor(p_body)) {
		_get_thread_locals().area_soft_body_overlaps.push_back(p_shape_pair);
	}
}

void JoltContactListener3D::_flush_contacts() {
	thread_local AHashMap<JPH::SubShapeIDPair, Manifold *, ShapePairHasher> deepest_manifolds;

	for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
		for (Manifold &manifold : tl->self()->manifolds) {
			Manifold *&deepest_manifold = deepest_manifolds[manifold.shape_pair];
			if (deepest_manifold == nullptr || manifold.depth > deepest_manifold->depth) {
				deepest_manifold = &manifold;
			}
		}
	}

	for (const KeyValue<JPH::SubShapeIDPair, Manifold *> &E : deepest_manifolds) {
		const JPH::SubShapeIDPair &shape_pair = E.key;
		const Manifold &manifold = *E.value;

		JoltBody3D *body1 = space->try_get_body(shape_pair.GetBody1ID());
		ERR_CONTINUE(body1 == nullptr);

		JoltBody3D *body2 = space->try_get_body(shape_pair.GetBody2ID());
		ERR_CONTINUE(body2 == nullptr);

		const int shape_index1 = body1->find_shape_index(shape_pair.GetSubShapeID1());
		const int shape_index2 = body2->find_shape_index(shape_pair.GetSubShapeID2());

		for (const Contact &contact : manifold.contacts) {
			body1->add_contact(body2, manifold.depth, shape_index1, shape_index2, manifold.normal1, contact.point1, contact.point2, contact.velocity1, contact.velocity2, contact.impulse1);
			body2->add_contact(body1, manifold.depth, shape_index2, shape_index1, -manifold.normal1, contact.point2, contact.point1, contact.velocity2, contact.velocity1, -contact.impulse1);
		}
	}

	deepest_manifolds.clear();

	for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
		tl->self()->manifolds.clear();
	}
}

void JoltContactListener3D::_flush_area_body_events() {
	uint32_t new_overlap_count = area_overlaps.size();
	for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
		new_overlap_count -= tl->self()->area_exits.size();
		new_overlap_count += tl->self()->area_enters.size();
	}

	// Exits must be dispatched before enters, as shape-shifting relies on it.

	for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
		for (const JPH::SubShapeIDPair &shape_pair : tl->self()->area_exits) {
			area_overlaps.erase(shape_pair);
			_dispatch_area_exit(shape_pair);
		}

		tl->self()->area_exits.clear();
	}

	area_overlaps.reserve(new_overlap_count);

	for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
		for (const JPH::SubShapeIDPair &shape_pair : tl->self()->area_enters) {
			area_overlaps.insert(shape_pair);
			_dispatch_area_enter(shape_pair);
		}

		tl->self()->area_enters.clear();
	}
}

void JoltContactListener3D::_flush_area_soft_body_events() {
	int current_count = area_soft_body_overlaps.size();
	int persisting_count = 0;
	int entered_count = 0;
	for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
		for (const JPH::SubShapeIDPair &shape_pair : tl->self()->area_soft_body_overlaps) {
			if (area_soft_body_overlaps.has(shape_pair)) {
				persisting_count += 1;
			} else {
				area_soft_body_overlaps.insert(shape_pair);
				_dispatch_area_enter(shape_pair);
				entered_count += 1;
			}
		}
	}

	int exited_count = current_count - persisting_count;
	if (exited_count > 0) {
		HashSet<JPH::SubShapeIDPair, ShapePairHasher> new_overlaps;
		new_overlaps.reserve(persisting_count + entered_count);

		for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
			for (const JPH::SubShapeIDPair &shape_pair : tl->self()->area_soft_body_overlaps) {
				new_overlaps.insert(shape_pair);
			}
		}

		for (const JPH::SubShapeIDPair &shape_pair : area_soft_body_overlaps) {
			if (!new_overlaps.has(shape_pair)) {
				_dispatch_area_exit(shape_pair);
			}
		}

		area_soft_body_overlaps = std::move(new_overlaps);
	} else if (unlikely(exited_count < 0)) {
		ERR_PRINT_ONCE("Duplicate area soft body overlaps found. This should not happen. Please report this.");
	}

	for (SelfList<ThreadLocals> *tl = ThreadLocals::instances.first(); tl != nullptr; tl = tl->next()) {
		tl->self()->area_soft_body_overlaps.clear();
	}
}

void JoltContactListener3D::_dispatch_area_enter(const JPH::SubShapeIDPair &p_shape_pair) {
	const JPH::BodyID &body_id1 = p_shape_pair.GetBody1ID();
	const JPH::BodyID &body_id2 = p_shape_pair.GetBody2ID();

	JoltObject3D *object1 = space->try_get_object(body_id1);
	JoltObject3D *object2 = space->try_get_object(body_id2);

	if (object1 == nullptr || object2 == nullptr) {
		return;
	}

	JoltArea3D *area1 = object1->as_area();
	JoltArea3D *area2 = object2->as_area();

	const JPH::SubShapeID &sub_shape_id1 = p_shape_pair.GetSubShapeID1();
	const JPH::SubShapeID &sub_shape_id2 = p_shape_pair.GetSubShapeID2();

	if (area1 != nullptr && area2 != nullptr) {
		area1->area_shape_entered(body_id2, sub_shape_id2, sub_shape_id1);
	} else if (area1 != nullptr && area2 == nullptr) {
		area1->body_shape_entered(body_id2, sub_shape_id2, sub_shape_id1);
	} else if (area1 == nullptr && area2 != nullptr) {
		area2->body_shape_entered(body_id1, sub_shape_id1, sub_shape_id2);
	}
}

void JoltContactListener3D::_dispatch_area_exit(const JPH::SubShapeIDPair &p_shape_pair) {
	const JPH::BodyID &body_id1 = p_shape_pair.GetBody1ID();
	const JPH::BodyID &body_id2 = p_shape_pair.GetBody2ID();

	JoltObject3D *object1 = space->try_get_object(body_id1);
	JoltObject3D *object2 = space->try_get_object(body_id2);

	JoltArea3D *area1 = object1 != nullptr ? object1->as_area() : nullptr;
	JoltArea3D *area2 = object2 != nullptr ? object2->as_area() : nullptr;

	const JPH::SubShapeID &sub_shape_id1 = p_shape_pair.GetSubShapeID1();
	const JPH::SubShapeID &sub_shape_id2 = p_shape_pair.GetSubShapeID2();

	if (area1 != nullptr && area2 != nullptr) {
		area1->area_shape_exited(body_id2, sub_shape_id2, sub_shape_id1);
	} else if (area1 != nullptr && object2 != nullptr) {
		area1->body_shape_exited(body_id2, sub_shape_id2, sub_shape_id1);
	} else if (object1 != nullptr && area2 != nullptr) {
		area2->body_shape_exited(body_id1, sub_shape_id1, sub_shape_id2);
	} else if (area1 != nullptr) {
		area1->shape_exited(body_id2, sub_shape_id2, sub_shape_id1);
	} else if (area2 != nullptr) {
		area2->shape_exited(body_id1, sub_shape_id1, sub_shape_id2);
	}
}

void JoltContactListener3D::pre_step() {
#ifdef DEBUG_ENABLED
	debug_contact_count = 0;
#endif
}

void JoltContactListener3D::post_step() {
	const MutexLock lock(ThreadLocals::instances_mutex);

	_flush_contacts();
	_flush_area_soft_body_events();
	_flush_area_body_events();
}
