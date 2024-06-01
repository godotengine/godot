// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2024 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>

JPH_NAMESPACE_BEGIN

/// An interface to query which vertices of a soft body are colliding with other bodies
class SoftBodyManifold
{
public:
	/// Get the vertices of the soft body for iterating
	const Array<SoftBodyVertex> &	GetVertices() const							{ return mVertices; }

	/// Check if a vertex has collided with something in this update
	JPH_INLINE bool					HasContact(const SoftBodyVertex &inVertex) const
	{
		return inVertex.mHasContact;
	}

	/// Get the local space contact point (multiply by GetCenterOfMassTransform() of the soft body to get world space)
	JPH_INLINE Vec3					GetLocalContactPoint(const SoftBodyVertex &inVertex) const
	{
		return inVertex.mPosition - inVertex.mCollisionPlane.SignedDistance(inVertex.mPosition) * inVertex.mCollisionPlane.GetNormal();
	}

	/// Get the contact normal for the vertex (assumes there is a contact).
	JPH_INLINE Vec3					GetContactNormal(const SoftBodyVertex &inVertex) const
	{
		return -inVertex.mCollisionPlane.GetNormal();
	}

	/// Get the body with which the vertex has collided in this update
	JPH_INLINE BodyID				GetContactBodyID(const SoftBodyVertex &inVertex) const
	{
		return inVertex.mHasContact? mCollidingShapes[inVertex.mCollidingShapeIndex].mBodyID : BodyID();
	}

private:
	/// Allow SoftBodyMotionProperties to construct us
	friend class SoftBodyMotionProperties;

	/// Constructor
	explicit						SoftBodyManifold(const SoftBodyMotionProperties *inMotionProperties) :
										mVertices(inMotionProperties->mVertices),
										mCollidingShapes(inMotionProperties->mCollidingShapes)
	{
	}

	using CollidingShape = SoftBodyMotionProperties::CollidingShape;

	const Array<SoftBodyVertex> &	mVertices;
	const Array<CollidingShape>	&	mCollidingShapes;
};

JPH_NAMESPACE_END
