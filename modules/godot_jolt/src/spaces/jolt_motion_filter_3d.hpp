#pragma once
#include "../common.h"
class JoltBodyImpl3D;
class JoltPhysicsServer3D;
class JoltSpace3D;

class JoltMotionFilter3D final
	: public JPH::BroadPhaseLayerFilter
	, public JPH::ObjectLayerFilter
	, public JPH::BodyFilter
	, public JPH::ShapeFilter {
public:
	explicit JoltMotionFilter3D(const JoltBodyImpl3D& p_body, bool p_collide_separation_ray = true);

	bool ShouldCollide(JPH::BroadPhaseLayer p_broad_phase_layer) const override;

	bool ShouldCollide(JPH::ObjectLayer p_object_layer) const override;

	bool ShouldCollide(const JPH::BodyID& p_jolt_id) const override;

	bool ShouldCollideLocked(const JPH::Body& p_jolt_body) const override;

	bool ShouldCollide(const JPH::Shape* p_jolt_shape, const JPH::SubShapeID& p_jolt_shape_id)
		const override;

	bool ShouldCollide(
		const JPH::Shape* p_jolt_shape_self,
		const JPH::SubShapeID& p_jolt_shape_id_self,
		const JPH::Shape* p_jolt_shape_other,
		const JPH::SubShapeID& p_jolt_shape_id_other
	) const override;

private:
	const JoltPhysicsServer3D& physics_server;

	const JoltBodyImpl3D& body_self;

	const JoltSpace3D& space;

	bool collide_separation_ray = false;
};
