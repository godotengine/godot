// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2025 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

#include <Jolt/Physics/Collision/CollideShape.h>
#include <Jolt/Physics/Collision/CollisionDispatch.h>
#include <Jolt/Core/STLLocalAllocator.h>

JPH_NAMESPACE_BEGIN

/// Collide 2 shapes and returns at most 1 hit per leaf shape pairs that overlapping. This can be used when not all contacts between the shapes are needed.
/// E.g. when testing a compound with 2 MeshShapes A and B against a compound with 2 SphereShapes C and D, then at most you'll get 4 collisions: AC, AD, BC, BD.
/// The default CollisionDispatch::sCollideShapeVsShape function would return all intersecting triangles in A against C, all in B against C etc.
/// @param inShape1 The first shape
/// @param inShape2 The second shape
/// @param inScale1 Local space scale of shape 1 (scales relative to its center of mass)
/// @param inScale2 Local space scale of shape 2 (scales relative to its center of mass)
/// @param inCenterOfMassTransform1 Transform to transform center of mass of shape 1 into world space
/// @param inCenterOfMassTransform2 Transform to transform center of mass of shape 2 into world space
/// @param inSubShapeIDCreator1 Class that tracks the current sub shape ID for shape 1
/// @param inSubShapeIDCreator2 Class that tracks the current sub shape ID for shape 2
/// @param inCollideShapeSettings Options for the CollideShape test
/// @param ioCollector The collector that receives the results.
/// @param inShapeFilter allows selectively disabling collisions between pairs of (sub) shapes.
/// @tparam LeafCollector The type of the collector that will be used to collect hits between leaf pairs. Must be either AnyHitCollisionCollector<CollideShapeCollector> to get any hit (cheapest) or ClosestHitCollisionCollector<CollideShapeCollector> to get the deepest hit (more expensive).
template <class LeafCollector>
void CollideShapeVsShapePerLeaf(const Shape *inShape1, const Shape *inShape2, Vec3Arg inScale1, Vec3Arg inScale2, Mat44Arg inCenterOfMassTransform1, Mat44Arg inCenterOfMassTransform2, const SubShapeIDCreator &inSubShapeIDCreator1, const SubShapeIDCreator &inSubShapeIDCreator2, const CollideShapeSettings &inCollideShapeSettings, CollideShapeCollector &ioCollector, const ShapeFilter &inShapeFilter = { })
{
	// Tracks information we need about a leaf shape
	struct LeafShape
	{
							LeafShape() = default;

							LeafShape(const AABox &inBounds, Mat44Arg inCenterOfMassTransform, Vec3Arg inScale, const Shape *inShape, const SubShapeIDCreator &inSubShapeIDCreator) :
			mBounds(inBounds),
			mCenterOfMassTransform(inCenterOfMassTransform),
			mScale(inScale),
			mShape(inShape),
			mSubShapeIDCreator(inSubShapeIDCreator)
		{
		}

		AABox				mBounds;
		Mat44				mCenterOfMassTransform;
		Vec3				mScale;
		const Shape *		mShape;
		SubShapeIDCreator	mSubShapeIDCreator;
	};

	constexpr uint cMaxLocalLeafShapes = 32;

	// A collector that stores the information we need from a leaf shape in an array that is usually on the stack but can fall back to the heap if needed
	class MyCollector : public TransformedShapeCollector
	{
	public:
							MyCollector()
		{
			mHits.reserve(cMaxLocalLeafShapes);
		}

		void				AddHit(const TransformedShape &inShape) override
		{
			mHits.emplace_back(inShape.GetWorldSpaceBounds(), inShape.GetCenterOfMassTransform().ToMat44(), inShape.GetShapeScale(), inShape.mShape, inShape.mSubShapeIDCreator);
		}

		Array<LeafShape, STLLocalAllocator<LeafShape, cMaxLocalLeafShapes>> mHits;
	};

	// Get bounds of both shapes
	AABox bounds1 = inShape1->GetWorldSpaceBounds(inCenterOfMassTransform1, inScale1);
	AABox bounds2 = inShape2->GetWorldSpaceBounds(inCenterOfMassTransform2, inScale2);

	// Get leaf shapes that overlap with the bounds of the other shape
	MyCollector leaf_shapes1, leaf_shapes2;
	inShape1->CollectTransformedShapes(bounds2, inCenterOfMassTransform1.GetTranslation(), inCenterOfMassTransform1.GetQuaternion(), inScale1, inSubShapeIDCreator1, leaf_shapes1, inShapeFilter);
	inShape2->CollectTransformedShapes(bounds1, inCenterOfMassTransform2.GetTranslation(), inCenterOfMassTransform2.GetQuaternion(), inScale2, inSubShapeIDCreator2, leaf_shapes2, inShapeFilter);

	// Now test each leaf shape against each other leaf
	for (const LeafShape &leaf1 : leaf_shapes1.mHits)
		for (const LeafShape &leaf2 : leaf_shapes2.mHits)
			if (leaf1.mBounds.Overlaps(leaf2.mBounds))
			{
				// Use the leaf collector to collect max 1 hit for this pair and pass it on to ioCollector
				LeafCollector collector;
				CollisionDispatch::sCollideShapeVsShape(leaf1.mShape, leaf2.mShape, leaf1.mScale, leaf2.mScale, leaf1.mCenterOfMassTransform, leaf2.mCenterOfMassTransform, leaf1.mSubShapeIDCreator, leaf2.mSubShapeIDCreator, inCollideShapeSettings, collector, inShapeFilter);
				if (collector.HadHit())
					ioCollector.AddHit(collector.mHit);
			}
}

JPH_NAMESPACE_END
