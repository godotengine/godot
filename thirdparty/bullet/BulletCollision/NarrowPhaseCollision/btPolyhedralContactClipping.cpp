/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2011 Advanced Micro Devices, Inc.  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

///This file was written by Erwin Coumans
///Separating axis rest based on work from Pierre Terdiman, see
///And contact clipping based on work from Simon Hobbs

#include "btPolyhedralContactClipping.h"
#include "BulletCollision/CollisionShapes/btConvexPolyhedron.h"

#include <float.h>  //for FLT_MAX

int gExpectedNbTests = 0;
int gActualNbTests = 0;
bool gUseInternalObject = true;

// Clips a face to the back of a plane
void btPolyhedralContactClipping::clipFace(const btVertexArray& pVtxIn, btVertexArray& ppVtxOut, const btVector3& planeNormalWS, btScalar planeEqWS)
{
	int ve;
	btScalar ds, de;
	int numVerts = pVtxIn.size();
	if (numVerts < 2)
		return;

	btVector3 firstVertex = pVtxIn[pVtxIn.size() - 1];
	btVector3 endVertex = pVtxIn[0];

	ds = planeNormalWS.dot(firstVertex) + planeEqWS;

	for (ve = 0; ve < numVerts; ve++)
	{
		endVertex = pVtxIn[ve];

		de = planeNormalWS.dot(endVertex) + planeEqWS;

		if (ds < 0)
		{
			if (de < 0)
			{
				// Start < 0, end < 0, so output endVertex
				ppVtxOut.push_back(endVertex);
			}
			else
			{
				// Start < 0, end >= 0, so output intersection
				ppVtxOut.push_back(firstVertex.lerp(endVertex, btScalar(ds * 1.f / (ds - de))));
			}
		}
		else
		{
			if (de < 0)
			{
				// Start >= 0, end < 0 so output intersection and end
				ppVtxOut.push_back(firstVertex.lerp(endVertex, btScalar(ds * 1.f / (ds - de))));
				ppVtxOut.push_back(endVertex);
			}
		}
		firstVertex = endVertex;
		ds = de;
	}
}

static bool TestSepAxis(const btConvexPolyhedron& hullA, const btConvexPolyhedron& hullB, const btTransform& transA, const btTransform& transB, const btVector3& sep_axis, btScalar& depth, btVector3& witnessPointA, btVector3& witnessPointB)
{
	btScalar Min0, Max0;
	btScalar Min1, Max1;
	btVector3 witnesPtMinA, witnesPtMaxA;
	btVector3 witnesPtMinB, witnesPtMaxB;

	hullA.project(transA, sep_axis, Min0, Max0, witnesPtMinA, witnesPtMaxA);
	hullB.project(transB, sep_axis, Min1, Max1, witnesPtMinB, witnesPtMaxB);

	if (Max0 < Min1 || Max1 < Min0)
		return false;

	btScalar d0 = Max0 - Min1;
	btAssert(d0 >= 0.0f);
	btScalar d1 = Max1 - Min0;
	btAssert(d1 >= 0.0f);
	if (d0 < d1)
	{
		depth = d0;
		witnessPointA = witnesPtMaxA;
		witnessPointB = witnesPtMinB;
	}
	else
	{
		depth = d1;
		witnessPointA = witnesPtMinA;
		witnessPointB = witnesPtMaxB;
	}

	return true;
}

static int gActualSATPairTests = 0;

inline bool IsAlmostZero(const btVector3& v)
{
	if (btFabs(v.x()) > 1e-6 || btFabs(v.y()) > 1e-6 || btFabs(v.z()) > 1e-6) return false;
	return true;
}

#ifdef TEST_INTERNAL_OBJECTS

inline void BoxSupport(const btScalar extents[3], const btScalar sv[3], btScalar p[3])
{
	// This version is ~11.000 cycles (4%) faster overall in one of the tests.
	//	IR(p[0]) = IR(extents[0])|(IR(sv[0])&SIGN_BITMASK);
	//	IR(p[1]) = IR(extents[1])|(IR(sv[1])&SIGN_BITMASK);
	//	IR(p[2]) = IR(extents[2])|(IR(sv[2])&SIGN_BITMASK);
	p[0] = sv[0] < 0.0f ? -extents[0] : extents[0];
	p[1] = sv[1] < 0.0f ? -extents[1] : extents[1];
	p[2] = sv[2] < 0.0f ? -extents[2] : extents[2];
}

void InverseTransformPoint3x3(btVector3& out, const btVector3& in, const btTransform& tr)
{
	const btMatrix3x3& rot = tr.getBasis();
	const btVector3& r0 = rot[0];
	const btVector3& r1 = rot[1];
	const btVector3& r2 = rot[2];

	const btScalar x = r0.x() * in.x() + r1.x() * in.y() + r2.x() * in.z();
	const btScalar y = r0.y() * in.x() + r1.y() * in.y() + r2.y() * in.z();
	const btScalar z = r0.z() * in.x() + r1.z() * in.y() + r2.z() * in.z();

	out.setValue(x, y, z);
}

bool TestInternalObjects(const btTransform& trans0, const btTransform& trans1, const btVector3& delta_c, const btVector3& axis, const btConvexPolyhedron& convex0, const btConvexPolyhedron& convex1, btScalar dmin)
{
	const btScalar dp = delta_c.dot(axis);

	btVector3 localAxis0;
	InverseTransformPoint3x3(localAxis0, axis, trans0);
	btVector3 localAxis1;
	InverseTransformPoint3x3(localAxis1, axis, trans1);

	btScalar p0[3];
	BoxSupport(convex0.m_extents, localAxis0, p0);
	btScalar p1[3];
	BoxSupport(convex1.m_extents, localAxis1, p1);

	const btScalar Radius0 = p0[0] * localAxis0.x() + p0[1] * localAxis0.y() + p0[2] * localAxis0.z();
	const btScalar Radius1 = p1[0] * localAxis1.x() + p1[1] * localAxis1.y() + p1[2] * localAxis1.z();

	const btScalar MinRadius = Radius0 > convex0.m_radius ? Radius0 : convex0.m_radius;
	const btScalar MaxRadius = Radius1 > convex1.m_radius ? Radius1 : convex1.m_radius;

	const btScalar MinMaxRadius = MaxRadius + MinRadius;
	const btScalar d0 = MinMaxRadius + dp;
	const btScalar d1 = MinMaxRadius - dp;

	const btScalar depth = d0 < d1 ? d0 : d1;
	if (depth > dmin)
		return false;
	return true;
}
#endif  //TEST_INTERNAL_OBJECTS

SIMD_FORCE_INLINE void btSegmentsClosestPoints(
	btVector3& ptsVector,
	btVector3& offsetA,
	btVector3& offsetB,
	btScalar& tA, btScalar& tB,
	const btVector3& translation,
	const btVector3& dirA, btScalar hlenA,
	const btVector3& dirB, btScalar hlenB)
{
	// compute the parameters of the closest points on each line segment

	btScalar dirA_dot_dirB = btDot(dirA, dirB);
	btScalar dirA_dot_trans = btDot(dirA, translation);
	btScalar dirB_dot_trans = btDot(dirB, translation);

	btScalar denom = 1.0f - dirA_dot_dirB * dirA_dot_dirB;

	if (denom == 0.0f)
	{
		tA = 0.0f;
	}
	else
	{
		tA = (dirA_dot_trans - dirB_dot_trans * dirA_dot_dirB) / denom;
		if (tA < -hlenA)
			tA = -hlenA;
		else if (tA > hlenA)
			tA = hlenA;
	}

	tB = tA * dirA_dot_dirB - dirB_dot_trans;

	if (tB < -hlenB)
	{
		tB = -hlenB;
		tA = tB * dirA_dot_dirB + dirA_dot_trans;

		if (tA < -hlenA)
			tA = -hlenA;
		else if (tA > hlenA)
			tA = hlenA;
	}
	else if (tB > hlenB)
	{
		tB = hlenB;
		tA = tB * dirA_dot_dirB + dirA_dot_trans;

		if (tA < -hlenA)
			tA = -hlenA;
		else if (tA > hlenA)
			tA = hlenA;
	}

	// compute the closest points relative to segment centers.

	offsetA = dirA * tA;
	offsetB = dirB * tB;

	ptsVector = translation - offsetA + offsetB;
}

bool btPolyhedralContactClipping::findSeparatingAxis(const btConvexPolyhedron& hullA, const btConvexPolyhedron& hullB, const btTransform& transA, const btTransform& transB, btVector3& sep, btDiscreteCollisionDetectorInterface::Result& resultOut)
{
	gActualSATPairTests++;

	//#ifdef TEST_INTERNAL_OBJECTS
	const btVector3 c0 = transA * hullA.m_localCenter;
	const btVector3 c1 = transB * hullB.m_localCenter;
	const btVector3 DeltaC2 = c0 - c1;
	//#endif

	btScalar dmin = FLT_MAX;
	int curPlaneTests = 0;

	int numFacesA = hullA.m_faces.size();
	// Test normals from hullA
	for (int i = 0; i < numFacesA; i++)
	{
		const btVector3 Normal(hullA.m_faces[i].m_plane[0], hullA.m_faces[i].m_plane[1], hullA.m_faces[i].m_plane[2]);
		btVector3 faceANormalWS = transA.getBasis() * Normal;
		if (DeltaC2.dot(faceANormalWS) < 0)
			faceANormalWS *= -1.f;

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if (gUseInternalObject && !TestInternalObjects(transA, transB, DeltaC2, faceANormalWS, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		btScalar d;
		btVector3 wA, wB;
		if (!TestSepAxis(hullA, hullB, transA, transB, faceANormalWS, d, wA, wB))
			return false;

		if (d < dmin)
		{
			dmin = d;
			sep = faceANormalWS;
		}
	}

	int numFacesB = hullB.m_faces.size();
	// Test normals from hullB
	for (int i = 0; i < numFacesB; i++)
	{
		const btVector3 Normal(hullB.m_faces[i].m_plane[0], hullB.m_faces[i].m_plane[1], hullB.m_faces[i].m_plane[2]);
		btVector3 WorldNormal = transB.getBasis() * Normal;
		if (DeltaC2.dot(WorldNormal) < 0)
			WorldNormal *= -1.f;

		curPlaneTests++;
#ifdef TEST_INTERNAL_OBJECTS
		gExpectedNbTests++;
		if (gUseInternalObject && !TestInternalObjects(transA, transB, DeltaC2, WorldNormal, hullA, hullB, dmin))
			continue;
		gActualNbTests++;
#endif

		btScalar d;
		btVector3 wA, wB;
		if (!TestSepAxis(hullA, hullB, transA, transB, WorldNormal, d, wA, wB))
			return false;

		if (d < dmin)
		{
			dmin = d;
			sep = WorldNormal;
		}
	}

	btVector3 edgeAstart, edgeAend, edgeBstart, edgeBend;
	int edgeA = -1;
	int edgeB = -1;
	btVector3 worldEdgeA;
	btVector3 worldEdgeB;
	btVector3 witnessPointA(0, 0, 0), witnessPointB(0, 0, 0);

	int curEdgeEdge = 0;
	// Test edges
	for (int e0 = 0; e0 < hullA.m_uniqueEdges.size(); e0++)
	{
		const btVector3 edge0 = hullA.m_uniqueEdges[e0];
		const btVector3 WorldEdge0 = transA.getBasis() * edge0;
		for (int e1 = 0; e1 < hullB.m_uniqueEdges.size(); e1++)
		{
			const btVector3 edge1 = hullB.m_uniqueEdges[e1];
			const btVector3 WorldEdge1 = transB.getBasis() * edge1;

			btVector3 Cross = WorldEdge0.cross(WorldEdge1);
			curEdgeEdge++;
			if (!IsAlmostZero(Cross))
			{
				Cross = Cross.normalize();
				if (DeltaC2.dot(Cross) < 0)
					Cross *= -1.f;

#ifdef TEST_INTERNAL_OBJECTS
				gExpectedNbTests++;
				if (gUseInternalObject && !TestInternalObjects(transA, transB, DeltaC2, Cross, hullA, hullB, dmin))
					continue;
				gActualNbTests++;
#endif

				btScalar dist;
				btVector3 wA, wB;
				if (!TestSepAxis(hullA, hullB, transA, transB, Cross, dist, wA, wB))
					return false;

				if (dist < dmin)
				{
					dmin = dist;
					sep = Cross;
					edgeA = e0;
					edgeB = e1;
					worldEdgeA = WorldEdge0;
					worldEdgeB = WorldEdge1;
					witnessPointA = wA;
					witnessPointB = wB;
				}
			}
		}
	}

	if (edgeA >= 0 && edgeB >= 0)
	{
		//		printf("edge-edge\n");
		//add an edge-edge contact

		btVector3 ptsVector;
		btVector3 offsetA;
		btVector3 offsetB;
		btScalar tA;
		btScalar tB;

		btVector3 translation = witnessPointB - witnessPointA;

		btVector3 dirA = worldEdgeA;
		btVector3 dirB = worldEdgeB;

		btScalar hlenB = 1e30f;
		btScalar hlenA = 1e30f;

		btSegmentsClosestPoints(ptsVector, offsetA, offsetB, tA, tB,
								translation,
								dirA, hlenA,
								dirB, hlenB);

		btScalar nlSqrt = ptsVector.length2();
		if (nlSqrt > SIMD_EPSILON)
		{
			btScalar nl = btSqrt(nlSqrt);
			ptsVector *= 1.f / nl;
			if (ptsVector.dot(DeltaC2) < 0.f)
			{
				ptsVector *= -1.f;
			}
			btVector3 ptOnB = witnessPointB + offsetB;
			btScalar distance = nl;
			resultOut.addContactPoint(ptsVector, ptOnB, -distance);
		}
	}

	if ((DeltaC2.dot(sep)) < 0.0f)
		sep = -sep;

	return true;
}

void btPolyhedralContactClipping::clipFaceAgainstHull(const btVector3& separatingNormal, const btConvexPolyhedron& hullA, const btTransform& transA, btVertexArray& worldVertsB1, btVertexArray& worldVertsB2, const btScalar minDist, btScalar maxDist, btDiscreteCollisionDetectorInterface::Result& resultOut)
{
	worldVertsB2.resize(0);
	btVertexArray* pVtxIn = &worldVertsB1;
	btVertexArray* pVtxOut = &worldVertsB2;
	pVtxOut->reserve(pVtxIn->size());

	int closestFaceA = -1;
	{
		btScalar dmin = FLT_MAX;
		for (int face = 0; face < hullA.m_faces.size(); face++)
		{
			const btVector3 Normal(hullA.m_faces[face].m_plane[0], hullA.m_faces[face].m_plane[1], hullA.m_faces[face].m_plane[2]);
			const btVector3 faceANormalWS = transA.getBasis() * Normal;

			btScalar d = faceANormalWS.dot(separatingNormal);
			if (d < dmin)
			{
				dmin = d;
				closestFaceA = face;
			}
		}
	}
	if (closestFaceA < 0)
		return;

	const btFace& polyA = hullA.m_faces[closestFaceA];

	// clip polygon to back of planes of all faces of hull A that are adjacent to witness face
	int numVerticesA = polyA.m_indices.size();
	for (int e0 = 0; e0 < numVerticesA; e0++)
	{
		const btVector3& a = hullA.m_vertices[polyA.m_indices[e0]];
		const btVector3& b = hullA.m_vertices[polyA.m_indices[(e0 + 1) % numVerticesA]];
		const btVector3 edge0 = a - b;
		const btVector3 WorldEdge0 = transA.getBasis() * edge0;
		btVector3 worldPlaneAnormal1 = transA.getBasis() * btVector3(polyA.m_plane[0], polyA.m_plane[1], polyA.m_plane[2]);

		btVector3 planeNormalWS1 = -WorldEdge0.cross(worldPlaneAnormal1);  //.cross(WorldEdge0);
		btVector3 worldA1 = transA * a;
		btScalar planeEqWS1 = -worldA1.dot(planeNormalWS1);

//int otherFace=0;
#ifdef BLA1
		int otherFace = polyA.m_connectedFaces[e0];
		btVector3 localPlaneNormal(hullA.m_faces[otherFace].m_plane[0], hullA.m_faces[otherFace].m_plane[1], hullA.m_faces[otherFace].m_plane[2]);
		btScalar localPlaneEq = hullA.m_faces[otherFace].m_plane[3];

		btVector3 planeNormalWS = transA.getBasis() * localPlaneNormal;
		btScalar planeEqWS = localPlaneEq - planeNormalWS.dot(transA.getOrigin());
#else
		btVector3 planeNormalWS = planeNormalWS1;
		btScalar planeEqWS = planeEqWS1;

#endif
		//clip face

		clipFace(*pVtxIn, *pVtxOut, planeNormalWS, planeEqWS);
		btSwap(pVtxIn, pVtxOut);
		pVtxOut->resize(0);
	}

	//#define ONLY_REPORT_DEEPEST_POINT

	btVector3 point;

	// only keep points that are behind the witness face
	{
		btVector3 localPlaneNormal(polyA.m_plane[0], polyA.m_plane[1], polyA.m_plane[2]);
		btScalar localPlaneEq = polyA.m_plane[3];
		btVector3 planeNormalWS = transA.getBasis() * localPlaneNormal;
		btScalar planeEqWS = localPlaneEq - planeNormalWS.dot(transA.getOrigin());
		for (int i = 0; i < pVtxIn->size(); i++)
		{
			btVector3 vtx = pVtxIn->at(i);
			btScalar depth = planeNormalWS.dot(vtx) + planeEqWS;
			if (depth <= minDist)
			{
				//				printf("clamped: depth=%f to minDist=%f\n",depth,minDist);
				depth = minDist;
			}

			if (depth <= maxDist)
			{
				btVector3 point = pVtxIn->at(i);
#ifdef ONLY_REPORT_DEEPEST_POINT
				curMaxDist = depth;
#else
#if 0
				if (depth<-3)
				{
					printf("error in btPolyhedralContactClipping depth = %f\n", depth);
					printf("likely wrong separatingNormal passed in\n");
				}
#endif
				resultOut.addContactPoint(separatingNormal, point, depth);
#endif
			}
		}
	}
#ifdef ONLY_REPORT_DEEPEST_POINT
	if (curMaxDist < maxDist)
	{
		resultOut.addContactPoint(separatingNormal, point, curMaxDist);
	}
#endif  //ONLY_REPORT_DEEPEST_POINT
}

void btPolyhedralContactClipping::clipHullAgainstHull(const btVector3& separatingNormal1, const btConvexPolyhedron& hullA, const btConvexPolyhedron& hullB, const btTransform& transA, const btTransform& transB, const btScalar minDist, btScalar maxDist, btVertexArray& worldVertsB1, btVertexArray& worldVertsB2, btDiscreteCollisionDetectorInterface::Result& resultOut)
{
	btVector3 separatingNormal = separatingNormal1.normalized();
	//	const btVector3 c0 = transA * hullA.m_localCenter;
	//	const btVector3 c1 = transB * hullB.m_localCenter;
	//const btVector3 DeltaC2 = c0 - c1;

	int closestFaceB = -1;
	btScalar dmax = -FLT_MAX;
	{
		for (int face = 0; face < hullB.m_faces.size(); face++)
		{
			const btVector3 Normal(hullB.m_faces[face].m_plane[0], hullB.m_faces[face].m_plane[1], hullB.m_faces[face].m_plane[2]);
			const btVector3 WorldNormal = transB.getBasis() * Normal;
			btScalar d = WorldNormal.dot(separatingNormal);
			if (d > dmax)
			{
				dmax = d;
				closestFaceB = face;
			}
		}
	}
	worldVertsB1.resize(0);
	{
		const btFace& polyB = hullB.m_faces[closestFaceB];
		const int numVertices = polyB.m_indices.size();
		for (int e0 = 0; e0 < numVertices; e0++)
		{
			const btVector3& b = hullB.m_vertices[polyB.m_indices[e0]];
			worldVertsB1.push_back(transB * b);
		}
	}

	if (closestFaceB >= 0)
		clipFaceAgainstHull(separatingNormal, hullA, transA, worldVertsB1, worldVertsB2, minDist, maxDist, resultOut);
}
