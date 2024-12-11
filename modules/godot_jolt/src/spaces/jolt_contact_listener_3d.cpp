#include "jolt_contact_listener_3d.hpp"

#include "objects/jolt_area_impl_3d.hpp"
#include "objects/jolt_body_impl_3d.hpp"
#include "objects/jolt_soft_body_impl_3d.hpp"
#include "servers/jolt_project_settings.hpp"
#include "spaces/jolt_space_3d.hpp"

void JoltContactListener3D::listen_for(JoltShapedObjectImpl3D* p_object) {
	listening_for.insert(p_object->get_jolt_id());
}

void JoltContactListener3D::pre_step() {
	listening_for.clear();

#ifdef TOOLS_ENABLED
	debug_contact_count = 0;
#endif // TOOLS_ENABLED
}

void JoltContactListener3D::post_step() {
	_flush_contacts();
	_flush_area_shifts();
	_flush_area_exits();
	_flush_area_enters();
}

void JoltContactListener3D::OnContactAdded(
	const JPH::Body& p_body1,
	const JPH::Body& p_body2,
	const JPH::ContactManifold& p_manifold,
	JPH::ContactSettings& p_settings
) {
	_try_override_collision_response(p_body1, p_body2, p_settings);
	_try_apply_surface_velocities(p_body1, p_body2, p_settings);
	_try_add_contacts(p_body1, p_body2, p_manifold, p_settings);
	_try_evaluate_area_overlap(p_body1, p_body2, p_manifold);

#ifdef TOOLS_ENABLED
	_try_add_debug_contacts(p_body1, p_body2, p_manifold);
#endif // TOOLS_ENABLED
}

void JoltContactListener3D::OnContactPersisted(
	const JPH::Body& p_body1,
	const JPH::Body& p_body2,
	const JPH::ContactManifold& p_manifold,
	JPH::ContactSettings& p_settings
) {
	_try_override_collision_response(p_body1, p_body2, p_settings);
	_try_apply_surface_velocities(p_body1, p_body2, p_settings);
	_try_add_contacts(p_body1, p_body2, p_manifold, p_settings);
	_try_evaluate_area_overlap(p_body1, p_body2, p_manifold);

#ifdef TOOLS_ENABLED
	_try_add_debug_contacts(p_body1, p_body2, p_manifold);
#endif // TOOLS_ENABLED
}

void JoltContactListener3D::OnContactRemoved(const JPH::SubShapeIDPair& p_shape_pair) {
	if (!_try_remove_contacts(p_shape_pair)) {
		_try_remove_area_overlap(p_shape_pair);
	}
}

JPH::SoftBodyValidateResult JoltContactListener3D::OnSoftBodyContactValidate(
	const JPH::Body& p_soft_body,
	const JPH::Body& p_other_body,
	JPH::SoftBodyContactSettings& p_settings
) {
	_try_override_collision_response(p_soft_body, p_other_body, p_settings);

	return JPH::SoftBodyValidateResult::AcceptContact;
}

#ifdef TOOLS_ENABLED

void JoltContactListener3D::OnSoftBodyContactAdded(
	const JPH::Body& p_soft_body,
	const JPH::SoftBodyManifold& p_manifold
) {
	_try_add_debug_contacts(p_soft_body, p_manifold);
}

#endif // TOOLS_ENABLED

bool JoltContactListener3D::_is_listening_for(const JPH::Body& p_body) const {
	return listening_for.has(p_body.GetID());
}

bool JoltContactListener3D::_try_override_collision_response(
	const JPH::Body& p_jolt_body1,
	const JPH::Body& p_jolt_body2,
	JPH::ContactSettings& p_settings
) {
	if (p_jolt_body1.IsSensor() || p_jolt_body2.IsSensor()) {
		return false;
	}

	if (!p_jolt_body1.IsDynamic() && !p_jolt_body2.IsDynamic()) {
		return false;
	}

	const auto* body1 = reinterpret_cast<JoltBodyImpl3D*>(p_jolt_body1.GetUserData());
	const auto* body2 = reinterpret_cast<JoltBodyImpl3D*>(p_jolt_body2.GetUserData());

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

bool JoltContactListener3D::_try_override_collision_response(
	const JPH::Body& p_jolt_soft_body,
	const JPH::Body& p_jolt_other_body,
	JPH::SoftBodyContactSettings& p_settings
) {
	if (p_jolt_other_body.IsSensor()) {
		return false;
	}

	const auto* soft_body = reinterpret_cast<JoltSoftBodyImpl3D*>(p_jolt_soft_body.GetUserData());
	const auto* other_body = reinterpret_cast<JoltBodyImpl3D*>(p_jolt_other_body.GetUserData());

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

bool JoltContactListener3D::_try_apply_surface_velocities(
	const JPH::Body& p_jolt_body1,
	const JPH::Body& p_jolt_body2,
	JPH::ContactSettings& p_settings
) {
	if (p_jolt_body1.IsSensor() || p_jolt_body2.IsSensor()) {
		return false;
	}

	const bool supports_surface_velocity1 = !p_jolt_body1.IsDynamic();
	const bool supports_surface_velocity2 = !p_jolt_body2.IsDynamic();

	if (supports_surface_velocity1 == supports_surface_velocity2) {
		return false;
	}

	const auto* body1 = reinterpret_cast<JoltBodyImpl3D*>(p_jolt_body1.GetUserData());
	const auto* body2 = reinterpret_cast<JoltBodyImpl3D*>(p_jolt_body2.GetUserData());

	const bool has_surface_velocity1 = supports_surface_velocity1 &&
		(body1->get_linear_surface_velocity() != Vector3() ||
		 body1->get_angular_surface_velocity() != Vector3());

	const bool has_surface_velocity2 = supports_surface_velocity2 &&
		(body2->get_linear_surface_velocity() != Vector3() ||
		 body2->get_angular_surface_velocity() != Vector3());

	if (has_surface_velocity1 == has_surface_velocity2) {
		return false;
	}

	const JPH::Vec3 linear_velocity1 = to_jolt(body1->get_linear_surface_velocity());
	const JPH::Vec3 angular_velocity1 = to_jolt(body1->get_angular_surface_velocity());

	const JPH::Vec3 linear_velocity2 = to_jolt(body2->get_linear_surface_velocity());
	const JPH::Vec3 angular_velocity2 = to_jolt(body2->get_angular_surface_velocity());

	const JPH::RVec3 com1 = p_jolt_body1.GetCenterOfMassPosition();
	const JPH::RVec3 com2 = p_jolt_body2.GetCenterOfMassPosition();
	const auto rel_com2 = JPH::Vec3(com2 - com1);

	const JPH::Vec3 angular_linear_velocity2 = rel_com2.Cross(angular_velocity2);
	const JPH::Vec3 total_linear_velocity2 = linear_velocity2 + angular_linear_velocity2;

	p_settings.mRelativeLinearSurfaceVelocity = total_linear_velocity2 - linear_velocity1;
	p_settings.mRelativeAngularSurfaceVelocity = angular_velocity2 - angular_velocity1;

	return true;
}

bool JoltContactListener3D::_try_add_contacts(
	const JPH::Body& p_body1,
	const JPH::Body& p_body2,
	const JPH::ContactManifold& p_manifold,
	JPH::ContactSettings& p_settings
) {
	if (p_body1.IsSensor() || p_body2.IsSensor()) {
		return false;
	}

	if (!_is_listening_for(p_body1) && !_is_listening_for(p_body2)) {
		return false;
	}

	const JPH::SubShapeIDPair shape_pair(
		p_body1.GetID(),
		p_manifold.mSubShapeID1,
		p_body2.GetID(),
		p_manifold.mSubShapeID2
	);

	auto& manifold = [&]() -> Manifold& {
		const MutexLock write_lock(write_mutex);
		return manifolds_by_shape_pair[shape_pair];
	}();

	const JPH::uint contact_count = p_manifold.mRelativeContactPointsOn1.size();

	manifold.contacts1.reserve((int32_t)contact_count);
	manifold.contacts2.reserve((int32_t)contact_count);
	manifold.depth = p_manifold.mPenetrationDepth;

	JPH::CollisionEstimationResult collision;

	JPH::EstimateCollisionResponse(
		p_body1,
		p_body2,
		p_manifold,
		collision,
		p_settings.mCombinedFriction,
		p_settings.mCombinedRestitution,
		JoltProjectSettings::get_bounce_velocity_threshold(),
		5
	);

	for (JPH::uint i = 0; i < contact_count; ++i) {
		Contact& contact1 = manifold.contacts1.emplace_back();
		Contact& contact2 = manifold.contacts2.emplace_back();

		const auto relative_point1 = JPH::RVec3(p_manifold.mRelativeContactPointsOn1[i]);
		const auto relative_point2 = JPH::RVec3(p_manifold.mRelativeContactPointsOn2[i]);

		const JPH::RVec3 world_point1 = p_manifold.mBaseOffset + relative_point1;
		const JPH::RVec3 world_point2 = p_manifold.mBaseOffset + relative_point2;

		const JPH::Vec3 velocity1 = p_body1.GetPointVelocity(world_point1);
		const JPH::Vec3 velocity2 = p_body2.GetPointVelocity(world_point2);

		const JPH::CollisionEstimationResult::Impulse& impulse = collision.mImpulses[i];

		const JPH::Vec3 contact_impulse = p_manifold.mWorldSpaceNormal * impulse.mContactImpulse;
		const JPH::Vec3 friction_impulse1 = collision.mTangent1 * impulse.mFrictionImpulse1;
		const JPH::Vec3 friction_impulse2 = collision.mTangent2 * impulse.mFrictionImpulse2;
		const JPH::Vec3 combined_impulse = contact_impulse + friction_impulse1 + friction_impulse2;

		contact1.normal = -p_manifold.mWorldSpaceNormal;
		contact1.point_self = world_point1;
		contact1.point_other = world_point2;
		contact1.velocity_self = velocity1;
		contact1.velocity_other = velocity2;
		contact1.impulse = -combined_impulse;

		contact2.normal = p_manifold.mWorldSpaceNormal;
		contact2.point_self = world_point2;
		contact2.point_other = world_point1;
		contact2.velocity_self = velocity2;
		contact2.velocity_other = velocity1;
		contact2.impulse = combined_impulse;
	}

	return true;
}

bool JoltContactListener3D::_try_evaluate_area_overlap(
	const JPH::Body& p_body1,
	const JPH::Body& p_body2,
	const JPH::ContactManifold& p_manifold
) {
	if (!p_body1.IsSensor() && !p_body2.IsSensor()) {
		return false;
	}

	auto evaluate = [&](auto&& p_area, auto&& p_object, const JPH::SubShapeIDPair& p_shape_pair) {
		const MutexLock write_lock(write_mutex);

		if (p_area.can_monitor(p_object)) {
			if (!area_overlaps.has(p_shape_pair)) {
				area_overlaps.insert(p_shape_pair);
				area_enters.insert(p_shape_pair);
			}
		} else {
			if (area_overlaps.erase(p_shape_pair)) {
				area_exits.insert(p_shape_pair);
			}
		}
	};

	const JPH::SubShapeIDPair shape_pair1(
		p_body1.GetID(),
		p_manifold.mSubShapeID1,
		p_body2.GetID(),
		p_manifold.mSubShapeID2
	);

	const JPH::SubShapeIDPair shape_pair2(
		p_body2.GetID(),
		p_manifold.mSubShapeID2,
		p_body1.GetID(),
		p_manifold.mSubShapeID1
	);

	const auto* object1 = reinterpret_cast<JoltObjectImpl3D*>(p_body1.GetUserData());
	const auto* object2 = reinterpret_cast<JoltObjectImpl3D*>(p_body2.GetUserData());

	const JoltAreaImpl3D* area1 = object1->as_area();
	const JoltAreaImpl3D* area2 = object2->as_area();

	const JoltBodyImpl3D* body1 = object1->as_body();
	const JoltBodyImpl3D* body2 = object2->as_body();

	if (area1 != nullptr && area2 != nullptr) {
		evaluate(*area1, *area2, shape_pair1);
		evaluate(*area2, *area1, shape_pair2);
	} else if (area1 != nullptr && body2 != nullptr) {
		evaluate(*area1, *body2, shape_pair1);
	} else if (area2 != nullptr && body1 != nullptr) {
		evaluate(*area2, *body1, shape_pair2);
	}

	return true;
}

bool JoltContactListener3D::_try_remove_contacts(const JPH::SubShapeIDPair& p_shape_pair) {
	const MutexLock write_lock(write_mutex);

	return manifolds_by_shape_pair.erase(p_shape_pair);
}

bool JoltContactListener3D::_try_remove_area_overlap(const JPH::SubShapeIDPair& p_shape_pair) {
	const JPH::SubShapeIDPair swapped_shape_pair(
		p_shape_pair.GetBody2ID(),
		p_shape_pair.GetSubShapeID2(),
		p_shape_pair.GetBody1ID(),
		p_shape_pair.GetSubShapeID1()
	);

	const MutexLock write_lock(write_mutex);

	bool removed = false;

	if (area_overlaps.erase(p_shape_pair)) {
		area_exits.insert(p_shape_pair);
		removed = true;
	}

	if (area_overlaps.erase(swapped_shape_pair)) {
		area_exits.insert(swapped_shape_pair);
		removed = true;
	}

	return removed;
}

#ifdef TOOLS_ENABLED

bool JoltContactListener3D::_try_add_debug_contacts(
	const JPH::Body& p_body1,
	const JPH::Body& p_body2,
	const JPH::ContactManifold& p_manifold
) {
	if (p_body1.IsSensor() || p_body2.IsSensor()) {
		return false;
	}

	const int64_t max_count = debug_contacts.size();

	if (max_count == 0) {
		return false;
	}

	const auto additional_pairs = (int32_t)p_manifold.mRelativeContactPointsOn1.size();
	const int32_t additional_contacts = additional_pairs * 2;

	int32_t current_count = debug_contact_count;
	bool exchanged = false;

	do {
		const int32_t new_count = current_count + additional_contacts;

		if (new_count > max_count) {
			return false;
		}

		exchanged = debug_contact_count.compare_exchange_weak(current_count, new_count);
	} while (!exchanged);

	for (int32_t i = 0; i < additional_pairs; ++i) {
		const int32_t pair_index = current_count + i * 2;

		const JPH::RVec3 point_on_1 = p_manifold.GetWorldSpaceContactPointOn1((JPH::uint)i);
		const JPH::RVec3 point_on_2 = p_manifold.GetWorldSpaceContactPointOn2((JPH::uint)i);

		debug_contacts.set(pair_index + 0, to_godot(point_on_1));
		debug_contacts.set(pair_index + 1, to_godot(point_on_2));
	}

	return true;
}

bool JoltContactListener3D::_try_add_debug_contacts(
	const JPH::Body& p_soft_body,
	const JPH::SoftBodyManifold& p_manifold
) {
	const int64_t max_count = debug_contacts.size();

	if (max_count == 0) {
		return false;
	}

	int32_t additional_contacts = 0;

	for (const JPH::SoftBodyVertex& vertex : p_manifold.GetVertices()) {
		if (p_manifold.HasContact(vertex)) {
			additional_contacts += 1;
		}
	}

	int32_t current_count = debug_contact_count;
	bool exchanged = false;

	do {
		const int32_t new_count = current_count + additional_contacts;

		if (new_count > max_count) {
			return false;
		}

		exchanged = debug_contact_count.compare_exchange_weak(current_count, new_count);
	} while (!exchanged);

	int32_t contact_index = current_count;

	for (const JPH::SoftBodyVertex& vertex : p_manifold.GetVertices()) {
		if (!p_manifold.HasContact(vertex)) {
			continue;
		}

		const JPH::RMat44 body_com_transform = p_soft_body.GetCenterOfMassTransform();
		const JPH::Vec3 local_contact_point = p_manifold.GetLocalContactPoint(vertex);
		const JPH::RVec3 contact_point = body_com_transform * local_contact_point;

		debug_contacts.set(contact_index++, to_godot(contact_point));
	}

	return true;
}

#endif // TOOLS_ENABLED

void JoltContactListener3D::_flush_contacts() {
	for (auto&& [shape_pair, manifold] : manifolds_by_shape_pair) {
		const JPH::BodyID body_ids[] = {shape_pair.GetBody1ID(), shape_pair.GetBody2ID()};

		const JoltReadableBodies3D jolt_bodies = space->read_bodies(body_ids, count_of(body_ids));

		JoltBodyImpl3D* body1 = jolt_bodies[0].as_body();
		ERR_FAIL_NULL(body1);

		JoltBodyImpl3D* body2 = jolt_bodies[1].as_body();
		ERR_FAIL_NULL(body2);

		const int32_t shape_index1 = body1->find_shape_index(shape_pair.GetSubShapeID1());
		const int32_t shape_index2 = body2->find_shape_index(shape_pair.GetSubShapeID2());

		for (const Contact& contact : manifold.contacts1) {
			body1->add_contact(
				body2,
				manifold.depth,
				shape_index1,
				shape_index2,
				to_godot(contact.normal),
				to_godot(contact.point_self),
				to_godot(contact.point_other),
				to_godot(contact.velocity_self),
				to_godot(contact.velocity_other),
				to_godot(contact.impulse)
			);
		}

		for (const Contact& contact : manifold.contacts2) {
			body2->add_contact(
				body1,
				manifold.depth,
				shape_index2,
				shape_index1,
				to_godot(contact.normal),
				to_godot(contact.point_self),
				to_godot(contact.point_other),
				to_godot(contact.velocity_self),
				to_godot(contact.velocity_other),
				to_godot(contact.impulse)
			);
		}

		manifold.contacts1.clear();
		manifold.contacts2.clear();
	}
}

void JoltContactListener3D::_flush_area_enters() {
	for (const JPH::SubShapeIDPair& shape_pair : area_enters) {
		const JPH::BodyID& body_id1 = shape_pair.GetBody1ID();
		const JPH::BodyID& body_id2 = shape_pair.GetBody2ID();

		const JPH::SubShapeID& sub_shape_id1 = shape_pair.GetSubShapeID1();
		const JPH::SubShapeID& sub_shape_id2 = shape_pair.GetSubShapeID2();

		const JPH::BodyID body_ids[] = {body_id1, body_id2};

		const JoltReadableBodies3D jolt_bodies = space->read_bodies(body_ids, count_of(body_ids));

		const JoltReadableBody3D jolt_body1 = jolt_bodies[0];
		const JoltReadableBody3D jolt_body2 = jolt_bodies[1];

		if (jolt_body1.is_invalid() || jolt_body2.is_invalid()) {
			continue;
		}

		JoltAreaImpl3D* area1 = jolt_body1.as_area();
		JoltAreaImpl3D* area2 = jolt_body2.as_area();

		if (area1 != nullptr && area2 != nullptr) {
			area1->area_shape_entered(body_id2, sub_shape_id2, sub_shape_id1);
		} else if (area1 != nullptr && area2 == nullptr) {
			area1->body_shape_entered(body_id2, sub_shape_id2, sub_shape_id1);
		} else if (area1 == nullptr && area2 != nullptr) {
			area2->body_shape_entered(body_id1, sub_shape_id1, sub_shape_id2);
		}
	}

	area_enters.clear();
}

void JoltContactListener3D::_flush_area_shifts() {
	for (const JPH::SubShapeIDPair& shape_pair : area_overlaps) {
		auto is_shifted = [&](const JPH::BodyID& p_body_id, const JPH::SubShapeID& p_sub_shape_id) {
			const JoltReadableBody3D jolt_body = space->read_body(p_body_id);
			const JoltShapedObjectImpl3D* object = jolt_body.as_shaped();
			ERR_FAIL_NULL_V(object, false);

			if (object->get_previous_jolt_shape() == nullptr) {
				return false;
			}

			const JPH::Shape& current_shape = *object->get_jolt_shape();
			const JPH::Shape& previous_shape = *object->get_previous_jolt_shape();

			const auto current_id = (uint32_t)current_shape.GetSubShapeUserData(p_sub_shape_id);
			const auto previous_id = (uint32_t)previous_shape.GetSubShapeUserData(p_sub_shape_id);

			return current_id != previous_id;
		};

		if (is_shifted(shape_pair.GetBody1ID(), shape_pair.GetSubShapeID1()) ||
			is_shifted(shape_pair.GetBody2ID(), shape_pair.GetSubShapeID2()))
		{
			area_enters.insert(shape_pair);
			area_exits.insert(shape_pair);
		}
	}
}

void JoltContactListener3D::_flush_area_exits() {
	for (const JPH::SubShapeIDPair& shape_pair : area_exits) {
		const JPH::BodyID& body_id1 = shape_pair.GetBody1ID();
		const JPH::BodyID& body_id2 = shape_pair.GetBody2ID();

		const JPH::SubShapeID& sub_shape_id1 = shape_pair.GetSubShapeID1();
		const JPH::SubShapeID& sub_shape_id2 = shape_pair.GetSubShapeID2();

		const JPH::BodyID body_ids[] = {body_id1, body_id2};

		const JoltReadableBodies3D jolt_bodies = space->read_bodies(body_ids, count_of(body_ids));

		const JoltReadableBody3D jolt_body1 = jolt_bodies[0];
		const JoltReadableBody3D jolt_body2 = jolt_bodies[1];

		JoltAreaImpl3D* area1 = jolt_body1.as_area();
		JoltAreaImpl3D* area2 = jolt_body2.as_area();

		const JoltBodyImpl3D* body1 = jolt_body1.as_body();
		const JoltBodyImpl3D* body2 = jolt_body2.as_body();

		if (area1 != nullptr && area2 != nullptr) {
			area1->area_shape_exited(body_id2, sub_shape_id2, sub_shape_id1);
		} else if (area1 != nullptr && body2 != nullptr) {
			area1->body_shape_exited(body_id2, sub_shape_id2, sub_shape_id1);
		} else if (body1 != nullptr && area2 != nullptr) {
			area2->body_shape_exited(body_id1, sub_shape_id1, sub_shape_id2);
		} else if (area1 != nullptr) {
			area1->shape_exited(body_id2, sub_shape_id2, sub_shape_id1);
		} else if (area2 != nullptr) {
			area2->shape_exited(body_id1, sub_shape_id1, sub_shape_id2);
		}
	}

	area_exits.clear();
}
