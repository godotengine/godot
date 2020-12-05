
#ifndef B3_SUPPORT_MAPPINGS_H
#define B3_SUPPORT_MAPPINGS_H

#include "Bullet3Common/b3Transform.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "b3VectorFloat4.h"

struct b3GjkPairDetector;

inline b3Vector3 localGetSupportVertexWithMargin(const float4& supportVec, const struct b3ConvexPolyhedronData* hull,
												 const b3AlignedObjectArray<b3Vector3>& verticesA, b3Scalar margin)
{
	b3Vector3 supVec = b3MakeVector3(b3Scalar(0.), b3Scalar(0.), b3Scalar(0.));
	b3Scalar maxDot = b3Scalar(-B3_LARGE_FLOAT);

	// Here we take advantage of dot(a, b*c) = dot(a*b, c).  Note: This is true mathematically, but not numerically.
	if (0 < hull->m_numVertices)
	{
		const b3Vector3 scaled = supportVec;
		int index = (int)scaled.maxDot(&verticesA[hull->m_vertexOffset], hull->m_numVertices, maxDot);
		return verticesA[hull->m_vertexOffset + index];
	}

	return supVec;
}

inline b3Vector3 localGetSupportVertexWithoutMargin(const float4& supportVec, const struct b3ConvexPolyhedronData* hull,
													const b3AlignedObjectArray<b3Vector3>& verticesA)
{
	return localGetSupportVertexWithMargin(supportVec, hull, verticesA, 0.f);
}

#endif  //B3_SUPPORT_MAPPINGS_H
