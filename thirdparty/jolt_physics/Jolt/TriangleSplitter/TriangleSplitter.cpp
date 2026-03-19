// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2021 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/TriangleSplitter/TriangleSplitter.h>

JPH_NAMESPACE_BEGIN

TriangleSplitter::TriangleSplitter(const VertexList &inVertices, const IndexedTriangleList &inTriangles) :
	mVertices(inVertices),
	mTriangles(inTriangles)
{
	mSortedTriangleIdx.resize(inTriangles.size());
	mCentroids.resize(inTriangles.size() + 1); // Add 1 so we can load with Vec3::sLoadFloat3Unsafe

	for (uint t = 0; t < inTriangles.size(); ++t)
	{
		// Initially triangles start unsorted
		mSortedTriangleIdx[t] = t;

		// Calculate centroid
		inTriangles[t].GetCentroid(inVertices).StoreFloat3(&mCentroids[t]);
	}

	// Make sure Vec3::sLoatFloat3Unsafe doesn't read uninitialized data
	mCentroids.back() = Float3(0, 0, 0);
}

bool TriangleSplitter::SplitInternal(const Range &inTriangles, uint inDimension, float inSplit, Range &outLeft, Range &outRight)
{
	// Divide triangles
	uint *start = mSortedTriangleIdx.data() + inTriangles.mBegin;
	uint *end = mSortedTriangleIdx.data() + inTriangles.mEnd;
	while (start < end)
	{
		// Search for first element that is on the right hand side of the split plane
		while (start < end && mCentroids[*start][inDimension] < inSplit)
			++start;

		// Search for the first element that is on the left hand side of the split plane
		while (start < end && mCentroids[*(end - 1)][inDimension] >= inSplit)
			--end;

		if (start < end)
		{
			// Swap the two elements
			--end;
			std::swap(*start, *end);
			++start;
		}
	}
	JPH_ASSERT(start == end);

	uint start_idx = uint(start - mSortedTriangleIdx.data());

#ifdef JPH_ENABLE_ASSERTS
	// Validate division algorithm
	JPH_ASSERT(inTriangles.mBegin <= start_idx);
	JPH_ASSERT(start_idx <= inTriangles.mEnd);
	for (uint i = inTriangles.mBegin; i < start_idx; ++i)
		JPH_ASSERT(mCentroids[mSortedTriangleIdx[i]][inDimension] < inSplit);
	for (uint i = start_idx; i < inTriangles.mEnd; ++i)
		JPH_ASSERT(mCentroids[mSortedTriangleIdx[i]][inDimension] >= inSplit);
#endif

	outLeft = Range(inTriangles.mBegin, start_idx);
	outRight = Range(start_idx, inTriangles.mEnd);
	return outLeft.Count() > 0 && outRight.Count() > 0;
}

JPH_NAMESPACE_END
