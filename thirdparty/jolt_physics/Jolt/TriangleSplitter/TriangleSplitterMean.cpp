// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/TriangleSplitter/TriangleSplitterMean.h>

JPH_NAMESPACE_BEGIN

TriangleSplitterMean::TriangleSplitterMean(const VertexList &inVertices, const IndexedTriangleList &inTriangles) :
	TriangleSplitter(inVertices, inTriangles)
{
}

bool TriangleSplitterMean::Split(const Range &inTriangles, Range &outLeft, Range &outRight)
{
	const uint *begin = mSortedTriangleIdx.data() + inTriangles.mBegin;
	const uint *end = mSortedTriangleIdx.data() + inTriangles.mEnd;

	// Calculate mean value for these triangles
	Vec3 mean = Vec3::sZero();
	for (const uint *t = begin; t < end; ++t)
		mean += Vec3::sLoadFloat3Unsafe(mCentroids[*t]);
	mean *= 1.0f / inTriangles.Count();

	// Calculate deviation
	Vec3 deviation = Vec3::sZero();
	for (const uint *t = begin; t < end; ++t)
	{
		Vec3 delta = Vec3::sLoadFloat3Unsafe(mCentroids[*t]) - mean;
		deviation += delta * delta;
	}
	deviation *= 1.0f / inTriangles.Count();

	// Calculate split plane
	uint dimension = deviation.GetHighestComponentIndex();
	float split = mean[dimension];

	return SplitInternal(inTriangles, dimension, split, outLeft, outRight);
}

JPH_NAMESPACE_END
