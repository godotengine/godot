// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Geometry/Plane.h>

JPH_NAMESPACE_BEGIN

/// Run time information for a single particle of a soft body
/// Note that at run-time you should only modify the inverse mass and/or velocity of a vertex to control the soft body.
/// Modifying the position can lead to missed collisions.
/// The other members are used internally by the soft body solver.
class SoftBodyVertex
{
public:
	Vec3			mPreviousPosition;					///< Position at the previous time step
	Vec3 			mPosition;							///< Position, relative to the center of mass of the soft body
	Vec3 			mVelocity;							///< Velocity, relative to the center of mass of the soft body
	Plane			mCollisionPlane;					///< Nearest collision plane, relative to the center of mass of the soft body
	int				mCollidingShapeIndex;				///< Index in the colliding shapes list of the body we may collide with
	bool			mHasContact;						///< True if the vertex has collided with anything in the last update
	float			mLargestPenetration;				///< Used while finding the collision plane, stores the largest penetration found so far
	float			mInvMass;							///< Inverse mass (1 / mass)
};

JPH_NAMESPACE_END
