// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2024 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/CollideSoftBodyVertexIterator.h>
#include <Jolt/Geometry/ClosestPoint.h>

JPH_NAMESPACE_BEGIN

/// Collision detection helper that collides soft body vertices vs triangles
class JPH_EXPORT CollideSoftBodyVerticesVsTriangles
{
public:
						CollideSoftBodyVerticesVsTriangles(Mat44Arg inCenterOfMassTransform, Vec3Arg inScale) :
		mTransform(inCenterOfMassTransform),
		mInvTransform(mTransform.InversedRotationTranslation()),
		mScale(inScale),
		mNormalSign(ScaleHelpers::IsInsideOut(inScale)? -1.0f : 1.0f)
	{
	}

	JPH_INLINE void		StartVertex(const CollideSoftBodyVertexIterator &inVertex)
	{
		mLocalPosition = mInvTransform * inVertex.GetPosition();
		mClosestDistanceSq = FLT_MAX;
	}

	JPH_INLINE void		ProcessTriangle(Vec3Arg inV0, Vec3Arg inV1, Vec3Arg inV2)
	{
		// Apply the scale to the triangle
		Vec3 v0 = mScale * inV0;
		Vec3 v1 = mScale * inV1;
		Vec3 v2 = mScale * inV2;

		// Get the closest point from the vertex to the triangle
		uint32 set;
		Vec3 closest_point = ClosestPoint::GetClosestPointOnTriangle(v0 - mLocalPosition, v1 - mLocalPosition, v2 - mLocalPosition, set);
		float dist_sq = closest_point.LengthSq();
		if (dist_sq < mClosestDistanceSq)
		{
			mV0 = v0;
			mV1 = v1;
			mV2 = v2;
			mClosestPoint = closest_point;
			mClosestDistanceSq = dist_sq;
			mSet = set;
		}
	}

	JPH_INLINE void		FinishVertex(const CollideSoftBodyVertexIterator &ioVertex, int inCollidingShapeIndex) const
	{
		if (mClosestDistanceSq < FLT_MAX)
		{
			// Convert triangle to world space
			Vec3 v0 = mTransform * mV0;
			Vec3 v1 = mTransform * mV1;
			Vec3 v2 = mTransform * mV2;
			Vec3 triangle_normal = mNormalSign * (v1 - v0).Cross(v2 - v0).NormalizedOr(Vec3::sAxisY());

			if (mSet == 0b111)
			{
				// Closest is interior to the triangle, use plane as collision plane but don't allow more than sTriangleThickness penetration
				// because otherwise a triangle half a level a way will have a huge penetration if it is back facing
				float penetration = triangle_normal.Dot(v0 - ioVertex.GetPosition());
				if (penetration < sTriangleThickness && ioVertex.UpdatePenetration(penetration))
					ioVertex.SetCollision(Plane::sFromPointAndNormal(v0, triangle_normal), inCollidingShapeIndex);
			}
			else
			{
				// Closest point is on an edge or vertex, use closest point as collision plane
				Vec3 closest_point = mTransform * (mLocalPosition + mClosestPoint);
				Vec3 normal = ioVertex.GetPosition() - closest_point;
				if (normal.Dot(triangle_normal) > 0.0f) // Ignore back facing edges
				{
					float normal_length = normal.Length();
					float penetration = -normal_length;
					if (ioVertex.UpdatePenetration(penetration))
						ioVertex.SetCollision(Plane::sFromPointAndNormal(closest_point, normal_length > 0.0f? normal / normal_length : triangle_normal), inCollidingShapeIndex);
				}
			}
		}
	}

	/// Triangles are considered to have some thickness. This thickness extends backwards along the negative triangle normal.
	/// Make this value smaller than the smallest 'wall thickness' so that the back side of the triangle doesn't protrude through the other side.
	/// Make this value too small and tunneling is more likely to occur.
	static inline float	sTriangleThickness = 0.1f;

	Mat44				mTransform;
	Mat44				mInvTransform;
	Vec3				mScale;
	Vec3				mLocalPosition;
	Vec3				mV0, mV1, mV2;
	Vec3				mClosestPoint;
	float				mNormalSign;
	float				mClosestDistanceSq;
	uint32				mSet;
};

JPH_NAMESPACE_END
