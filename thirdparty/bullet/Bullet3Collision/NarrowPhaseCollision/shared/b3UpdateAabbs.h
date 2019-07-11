#ifndef B3_UPDATE_AABBS_H
#define B3_UPDATE_AABBS_H

#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

void b3ComputeWorldAabb(int bodyId, __global const b3RigidBodyData_t* bodies, __global const b3Collidable_t* collidables, __global const b3Aabb_t* localShapeAABB, __global b3Aabb_t* worldAabbs)
{
	__global const b3RigidBodyData_t* body = &bodies[bodyId];

	b3Float4 position = body->m_pos;
	b3Quat orientation = body->m_quat;

	int collidableIndex = body->m_collidableIdx;
	int shapeIndex = collidables[collidableIndex].m_shapeIndex;

	if (shapeIndex >= 0)
	{
		b3Aabb_t localAabb = localShapeAABB[collidableIndex];
		b3Aabb_t worldAabb;

		b3Float4 aabbAMinOut, aabbAMaxOut;
		float margin = 0.f;
		b3TransformAabb2(localAabb.m_minVec, localAabb.m_maxVec, margin, position, orientation, &aabbAMinOut, &aabbAMaxOut);

		worldAabb.m_minVec = aabbAMinOut;
		worldAabb.m_minIndices[3] = bodyId;
		worldAabb.m_maxVec = aabbAMaxOut;
		worldAabb.m_signedMaxIndices[3] = body[bodyId].m_invMass == 0.f ? 0 : 1;
		worldAabbs[bodyId] = worldAabb;
	}
}

#endif  //B3_UPDATE_AABBS_H
