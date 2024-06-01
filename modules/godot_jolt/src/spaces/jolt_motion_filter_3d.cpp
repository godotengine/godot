#include "jolt_motion_filter_3d.hpp"

#include "objects/jolt_body_impl_3d.hpp"
#include "objects/jolt_object_impl_3d.hpp"
#include "servers/jolt_physics_server_3d.hpp"
#include "shapes/jolt_custom_motion_shape.hpp"
#include "shapes/jolt_custom_shape_type.hpp"
#include "shapes/jolt_shape_impl_3d.hpp"
#include "spaces/jolt_broad_phase_layer.hpp"
#include "spaces/jolt_space_3d.hpp"

JoltMotionFilter3D::JoltMotionFilter3D(const JoltBodyImpl3D& p_body, bool p_collide_separation_ray)
	: physics_server(*static_cast<JoltPhysicsServer3D*>(PhysicsServer3D::get_singleton()))
	, body_self(p_body)
	, space(*body_self.get_space())
	, collide_separation_ray(p_collide_separation_ray) { }

bool JoltMotionFilter3D::ShouldCollide(JPH::BroadPhaseLayer p_broad_phase_layer) const {
	const auto broad_phase_layer = (JPH::BroadPhaseLayer::Type)p_broad_phase_layer;

	switch (broad_phase_layer) {
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::BODY_STATIC:
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::BODY_DYNAMIC: {
			return true;
		} break;
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::AREA_DETECTABLE:
		case (JPH::BroadPhaseLayer::Type)JoltBroadPhaseLayer::AREA_UNDETECTABLE: {
			return false;
		} break;
		default: {
			ERR_FAIL_D_MSG(vformat("Unhandled broad phase layer: '%d'", broad_phase_layer));
		}
	}
}

bool JoltMotionFilter3D::ShouldCollide(JPH::ObjectLayer p_object_layer) const {
	JPH::BroadPhaseLayer object_broad_phase_layer = {};
	uint32_t object_collision_layer = 0;
	uint32_t object_collision_mask = 0;

	space.map_from_object_layer(
		p_object_layer,
		object_broad_phase_layer,
		object_collision_layer,
		object_collision_mask
	);

	return (body_self.get_collision_mask() & object_collision_layer) != 0;
}

bool JoltMotionFilter3D::ShouldCollide(const JPH::BodyID& p_jolt_id) const {
	return p_jolt_id != body_self.get_jolt_id();
}

bool JoltMotionFilter3D::ShouldCollideLocked(const JPH::Body& p_jolt_body) const {
	if (p_jolt_body.IsSoftBody()) {
		return false;
	}

	const auto* object = reinterpret_cast<const JoltObjectImpl3D*>(p_jolt_body.GetUserData());

	if (physics_server.body_test_motion_is_excluding_object(object->get_instance_id()) ||
		physics_server.body_test_motion_is_excluding_body(object->get_rid()))
	{
		return false;
	}

	const JoltReadableBody3D jolt_body_self = space.read_body(body_self);

	return jolt_body_self->GetCollisionGroup().CanCollide(p_jolt_body.GetCollisionGroup());
}

bool JoltMotionFilter3D::ShouldCollide(
	[[maybe_unused]] const JPH::Shape* p_jolt_shape,
	[[maybe_unused]] const JPH::SubShapeID& p_jolt_shape_id
) const {
	return true;
}

bool JoltMotionFilter3D::ShouldCollide(
	const JPH::Shape* p_jolt_shape_self,
	[[maybe_unused]] const JPH::SubShapeID& p_jolt_shape_id_self,
	[[maybe_unused]] const JPH::Shape* p_jolt_shape_other,
	[[maybe_unused]] const JPH::SubShapeID& p_jolt_shape_id_other
) const {
	if (collide_separation_ray) {
		return true;
	}

	const auto* motion_shape = static_cast<const JoltCustomMotionShape*>(p_jolt_shape_self);
	const JPH::ConvexShape& actual_shape_self = motion_shape->get_inner_shape();

	return actual_shape_self.GetSubType() != JoltCustomShapeSubType::RAY;
}
