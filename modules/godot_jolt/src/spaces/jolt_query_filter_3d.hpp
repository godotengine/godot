#pragma once

#include "../common.h"


class JoltPhysicsDirectSpaceState3D;
class JoltSpace3D;

class JoltQueryFilter3D final
	: public JPH::BroadPhaseLayerFilter
	, public JPH::ObjectLayerFilter
	, public JPH::BodyFilter {
public:
	JoltQueryFilter3D(
		const JoltPhysicsDirectSpaceState3D& p_space_state,
		uint32_t p_collision_mask,
		bool p_collide_with_bodies,
		bool p_collide_with_areas,
		bool p_picking = false
	);

	bool ShouldCollide(JPH::BroadPhaseLayer p_broad_phase_layer) const override;

	bool ShouldCollide(JPH::ObjectLayer p_object_layer) const override;

	bool ShouldCollide(const JPH::BodyID& p_body_id) const override;

	bool ShouldCollideLocked(const JPH::Body& p_body) const override;

private:
	const JoltPhysicsDirectSpaceState3D& space_state;

	const JoltSpace3D& space;

	uint32_t collision_mask = 0;

	bool collide_with_bodies = false;

	bool collide_with_areas = false;

	bool picking = false;
};
