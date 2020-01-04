/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_VORONOI_SIMPLEX_SOLVER_H
#define B3_VORONOI_SIMPLEX_SOLVER_H

#include "Bullet3Common/b3Vector3.h"

#define VORONOI_SIMPLEX_MAX_VERTS 5

///disable next define, or use defaultCollisionConfiguration->getSimplexSolver()->setEqualVertexThreshold(0.f) to disable/configure
//#define BT_USE_EQUAL_VERTEX_THRESHOLD
#define VORONOI_DEFAULT_EQUAL_VERTEX_THRESHOLD 0.0001f

struct b3UsageBitfield
{
	b3UsageBitfield()
	{
		reset();
	}

	void reset()
	{
		usedVertexA = false;
		usedVertexB = false;
		usedVertexC = false;
		usedVertexD = false;
	}
	unsigned short usedVertexA : 1;
	unsigned short usedVertexB : 1;
	unsigned short usedVertexC : 1;
	unsigned short usedVertexD : 1;
	unsigned short unused1 : 1;
	unsigned short unused2 : 1;
	unsigned short unused3 : 1;
	unsigned short unused4 : 1;
};

struct b3SubSimplexClosestResult
{
	b3Vector3 m_closestPointOnSimplex;
	//MASK for m_usedVertices
	//stores the simplex vertex-usage, using the MASK,
	// if m_usedVertices & MASK then the related vertex is used
	b3UsageBitfield m_usedVertices;
	b3Scalar m_barycentricCoords[4];
	bool m_degenerate;

	void reset()
	{
		m_degenerate = false;
		setBarycentricCoordinates();
		m_usedVertices.reset();
	}
	bool isValid()
	{
		bool valid = (m_barycentricCoords[0] >= b3Scalar(0.)) &&
					 (m_barycentricCoords[1] >= b3Scalar(0.)) &&
					 (m_barycentricCoords[2] >= b3Scalar(0.)) &&
					 (m_barycentricCoords[3] >= b3Scalar(0.));

		return valid;
	}
	void setBarycentricCoordinates(b3Scalar a = b3Scalar(0.), b3Scalar b = b3Scalar(0.), b3Scalar c = b3Scalar(0.), b3Scalar d = b3Scalar(0.))
	{
		m_barycentricCoords[0] = a;
		m_barycentricCoords[1] = b;
		m_barycentricCoords[2] = c;
		m_barycentricCoords[3] = d;
	}
};

/// b3VoronoiSimplexSolver is an implementation of the closest point distance algorithm from a 1-4 points simplex to the origin.
/// Can be used with GJK, as an alternative to Johnson distance algorithm.

B3_ATTRIBUTE_ALIGNED16(class)
b3VoronoiSimplexSolver
{
public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

	int m_numVertices;

	b3Vector3 m_simplexVectorW[VORONOI_SIMPLEX_MAX_VERTS];
	b3Vector3 m_simplexPointsP[VORONOI_SIMPLEX_MAX_VERTS];
	b3Vector3 m_simplexPointsQ[VORONOI_SIMPLEX_MAX_VERTS];

	b3Vector3 m_cachedP1;
	b3Vector3 m_cachedP2;
	b3Vector3 m_cachedV;
	b3Vector3 m_lastW;

	b3Scalar m_equalVertexThreshold;
	bool m_cachedValidClosest;

	b3SubSimplexClosestResult m_cachedBC;

	bool m_needsUpdate;

	void removeVertex(int index);
	void reduceVertices(const b3UsageBitfield& usedVerts);
	bool updateClosestVectorAndPoints();

	bool closestPtPointTetrahedron(const b3Vector3& p, const b3Vector3& a, const b3Vector3& b, const b3Vector3& c, const b3Vector3& d, b3SubSimplexClosestResult& finalResult);
	int pointOutsideOfPlane(const b3Vector3& p, const b3Vector3& a, const b3Vector3& b, const b3Vector3& c, const b3Vector3& d);
	bool closestPtPointTriangle(const b3Vector3& p, const b3Vector3& a, const b3Vector3& b, const b3Vector3& c, b3SubSimplexClosestResult& result);

public:
	b3VoronoiSimplexSolver()
		: m_equalVertexThreshold(VORONOI_DEFAULT_EQUAL_VERTEX_THRESHOLD)
	{
	}
	void reset();

	void addVertex(const b3Vector3& w, const b3Vector3& p, const b3Vector3& q);

	void setEqualVertexThreshold(b3Scalar threshold)
	{
		m_equalVertexThreshold = threshold;
	}

	b3Scalar getEqualVertexThreshold() const
	{
		return m_equalVertexThreshold;
	}

	bool closest(b3Vector3 & v);

	b3Scalar maxVertex();

	bool fullSimplex() const
	{
		return (m_numVertices == 4);
	}

	int getSimplex(b3Vector3 * pBuf, b3Vector3 * qBuf, b3Vector3 * yBuf) const;

	bool inSimplex(const b3Vector3& w);

	void backup_closest(b3Vector3 & v);

	bool emptySimplex() const;

	void compute_points(b3Vector3 & p1, b3Vector3 & p2);

	int numVertices() const
	{
		return m_numVertices;
	}
};

#endif  //B3_VORONOI_SIMPLEX_SOLVER_H
