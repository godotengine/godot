
#ifndef B3_CONVEX_POLYHEDRON_DATA_H
#define B3_CONVEX_POLYHEDRON_DATA_H



#include "Bullet3Common/shared/b3Float4.h"
#include "Bullet3Common/shared/b3Quat.h"

typedef struct b3GpuFace b3GpuFace_t;
struct b3GpuFace
{
	b3Float4 m_plane;
	int m_indexOffset;
	int m_numIndices;
	int m_unusedPadding1;
	int m_unusedPadding2;
};

typedef struct b3ConvexPolyhedronData b3ConvexPolyhedronData_t;

struct b3ConvexPolyhedronData
{
	b3Float4		m_localCenter;
	b3Float4		m_extents;
	b3Float4		mC;
	b3Float4		mE;

	float			m_radius;
	int	m_faceOffset;
	int m_numFaces;
	int	m_numVertices;

	int m_vertexOffset;
	int	m_uniqueEdgesOffset;
	int	m_numUniqueEdges;
	int m_unused;
};

#endif //B3_CONVEX_POLYHEDRON_DATA_H
