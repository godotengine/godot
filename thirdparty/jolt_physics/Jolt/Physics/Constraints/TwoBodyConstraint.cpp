// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/Constraints/TwoBodyConstraint.h>
#include <Jolt/Physics/IslandBuilder.h>
#include <Jolt/Physics/LargeIslandSplitter.h>
#include <Jolt/Physics/Body/BodyManager.h>

#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN

JPH_IMPLEMENT_SERIALIZABLE_ABSTRACT(TwoBodyConstraintSettings)
{
	JPH_ADD_BASE_CLASS(TwoBodyConstraintSettings, ConstraintSettings)
}

void TwoBodyConstraint::BuildIslands(uint32 inConstraintIndex, IslandBuilder &ioBuilder, BodyManager &inBodyManager)
{
#ifdef JPH_ENABLE_ASSERTS
	// Validates that a body that is sleeping has zero velocity.
	mBody1->ValidateMotion();
	mBody2->ValidateMotion();
#endif

	bool body1_dynamic = mBody1->IsDynamic();
	bool body2_dynamic = mBody2->IsDynamic();

	// Activate bodies
	BodyID body_ids[2];
	int num_bodies = 0;
	if (body1_dynamic && !mBody1->IsActive())
		body_ids[num_bodies++] = mBody1->GetID();
	if (body2_dynamic && !mBody2->IsActive())
		body_ids[num_bodies++] = mBody2->GetID();
	if (num_bodies > 0)
		inBodyManager.ActivateBodies(body_ids, num_bodies);

	// Link the two bodies only if both are dynamic. If one of them is static or kinematic they don't need to go into
	// the same simulation island as a constraint cannot affect the velocity of a kinematic body.
	if (body1_dynamic && body2_dynamic)
		ioBuilder.LinkBodies(mBody1->GetIndexInActiveBodiesInternal(), mBody2->GetIndexInActiveBodiesInternal());

	// Link the constraint to the first dynamic body
	if (body1_dynamic)
		ioBuilder.LinkConstraint(inConstraintIndex, mBody1->GetIndexInActiveBodiesInternal());
	else
	{
		JPH_ASSERT(body2_dynamic);
		ioBuilder.LinkConstraint(inConstraintIndex, mBody2->GetIndexInActiveBodiesInternal());
	}
}

uint TwoBodyConstraint::BuildIslandSplits(LargeIslandSplitter &ioSplitter) const
{
	return ioSplitter.AssignSplit(mBody1, mBody2);
}

#ifdef JPH_DEBUG_RENDERER

void TwoBodyConstraint::DrawConstraintReferenceFrame(DebugRenderer *inRenderer) const
{
	RMat44 transform1 = mBody1->GetCenterOfMassTransform() * GetConstraintToBody1Matrix();
	RMat44 transform2 = mBody2->GetCenterOfMassTransform() * GetConstraintToBody2Matrix();
	inRenderer->DrawCoordinateSystem(transform1, 1.1f * mDrawConstraintSize);
	inRenderer->DrawCoordinateSystem(transform2, mDrawConstraintSize);
}

#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_END
