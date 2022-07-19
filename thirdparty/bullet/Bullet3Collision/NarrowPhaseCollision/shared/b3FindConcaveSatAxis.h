#ifndef B3_FIND_CONCAVE_SEPARATING_AXIS_H
#define B3_FIND_CONCAVE_SEPARATING_AXIS_H

#define B3_TRIANGLE_NUM_CONVEX_FACES 5

#include "Bullet3Common/shared/b3Int4.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3BvhSubtreeInfoData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3QuantizedBvhNodeData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"

inline void b3Project(__global const b3ConvexPolyhedronData* hull, b3Float4ConstArg pos, b3QuatConstArg orn,
					  const b3Float4* dir, __global const b3Float4* vertices, float* min, float* max)
{
	min[0] = FLT_MAX;
	max[0] = -FLT_MAX;
	int numVerts = hull->m_numVertices;

	const b3Float4 localDir = b3QuatRotate(b3QuatInverse(orn), *dir);
	float offset = b3Dot(pos, *dir);
	for (int i = 0; i < numVerts; i++)
	{
		float dp = b3Dot(vertices[hull->m_vertexOffset + i], localDir);
		if (dp < min[0])
			min[0] = dp;
		if (dp > max[0])
			max[0] = dp;
	}
	if (min[0] > max[0])
	{
		float tmp = min[0];
		min[0] = max[0];
		max[0] = tmp;
	}
	min[0] += offset;
	max[0] += offset;
}

inline bool b3TestSepAxis(const b3ConvexPolyhedronData* hullA, __global const b3ConvexPolyhedronData* hullB,
						  b3Float4ConstArg posA, b3QuatConstArg ornA,
						  b3Float4ConstArg posB, b3QuatConstArg ornB,
						  b3Float4* sep_axis, const b3Float4* verticesA, __global const b3Float4* verticesB, float* depth)
{
	float Min0, Max0;
	float Min1, Max1;
	b3Project(hullA, posA, ornA, sep_axis, verticesA, &Min0, &Max0);
	b3Project(hullB, posB, ornB, sep_axis, verticesB, &Min1, &Max1);

	if (Max0 < Min1 || Max1 < Min0)
		return false;

	float d0 = Max0 - Min1;
	float d1 = Max1 - Min0;
	*depth = d0 < d1 ? d0 : d1;
	return true;
}

bool b3FindSeparatingAxis(const b3ConvexPolyhedronData* hullA, __global const b3ConvexPolyhedronData* hullB,
						  b3Float4ConstArg posA1,
						  b3QuatConstArg ornA,
						  b3Float4ConstArg posB1,
						  b3QuatConstArg ornB,
						  b3Float4ConstArg DeltaC2,

						  const b3Float4* verticesA,
						  const b3Float4* uniqueEdgesA,
						  const b3GpuFace* facesA,
						  const int* indicesA,

						  __global const b3Float4* verticesB,
						  __global const b3Float4* uniqueEdgesB,
						  __global const b3GpuFace* facesB,
						  __global const int* indicesB,
						  b3Float4* sep,
						  float* dmin)
{
	b3Float4 posA = posA1;
	posA.w = 0.f;
	b3Float4 posB = posB1;
	posB.w = 0.f;
	/*
	static int maxFaceVertex = 0;

	int curFaceVertexAB = hullA->m_numFaces*hullB->m_numVertices;
	curFaceVertexAB+= hullB->m_numFaces*hullA->m_numVertices;

	if (curFaceVertexAB>maxFaceVertex)
	{
		maxFaceVertex = curFaceVertexAB;
		printf("curFaceVertexAB = %d\n",curFaceVertexAB);
		printf("hullA->m_numFaces = %d\n",hullA->m_numFaces);
		printf("hullA->m_numVertices = %d\n",hullA->m_numVertices);
		printf("hullB->m_numVertices = %d\n",hullB->m_numVertices);
	}
*/

	int curPlaneTests = 0;
	{
		int numFacesA = hullA->m_numFaces;
		// Test normals from hullA
		for (int i = 0; i < numFacesA; i++)
		{
			const b3Float4 normal = facesA[hullA->m_faceOffset + i].m_plane;
			b3Float4 faceANormalWS = b3QuatRotate(ornA, normal);
			if (b3Dot(DeltaC2, faceANormalWS) < 0)
				faceANormalWS *= -1.f;
			curPlaneTests++;
			float d;
			if (!b3TestSepAxis(hullA, hullB, posA, ornA, posB, ornB, &faceANormalWS, verticesA, verticesB, &d))
				return false;
			if (d < *dmin)
			{
				*dmin = d;
				*sep = faceANormalWS;
			}
		}
	}
	if ((b3Dot(-DeltaC2, *sep)) > 0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}

b3Vector3 unitSphere162[] =
	{
		b3MakeVector3(0.000000, -1.000000, 0.000000),
		b3MakeVector3(0.203181, -0.967950, 0.147618),
		b3MakeVector3(-0.077607, -0.967950, 0.238853),
		b3MakeVector3(0.723607, -0.447220, 0.525725),
		b3MakeVector3(0.609547, -0.657519, 0.442856),
		b3MakeVector3(0.812729, -0.502301, 0.295238),
		b3MakeVector3(-0.251147, -0.967949, 0.000000),
		b3MakeVector3(-0.077607, -0.967950, -0.238853),
		b3MakeVector3(0.203181, -0.967950, -0.147618),
		b3MakeVector3(0.860698, -0.251151, 0.442858),
		b3MakeVector3(-0.276388, -0.447220, 0.850649),
		b3MakeVector3(-0.029639, -0.502302, 0.864184),
		b3MakeVector3(-0.155215, -0.251152, 0.955422),
		b3MakeVector3(-0.894426, -0.447216, 0.000000),
		b3MakeVector3(-0.831051, -0.502299, 0.238853),
		b3MakeVector3(-0.956626, -0.251149, 0.147618),
		b3MakeVector3(-0.276388, -0.447220, -0.850649),
		b3MakeVector3(-0.483971, -0.502302, -0.716565),
		b3MakeVector3(-0.436007, -0.251152, -0.864188),
		b3MakeVector3(0.723607, -0.447220, -0.525725),
		b3MakeVector3(0.531941, -0.502302, -0.681712),
		b3MakeVector3(0.687159, -0.251152, -0.681715),
		b3MakeVector3(0.687159, -0.251152, 0.681715),
		b3MakeVector3(-0.436007, -0.251152, 0.864188),
		b3MakeVector3(-0.956626, -0.251149, -0.147618),
		b3MakeVector3(-0.155215, -0.251152, -0.955422),
		b3MakeVector3(0.860698, -0.251151, -0.442858),
		b3MakeVector3(0.276388, 0.447220, 0.850649),
		b3MakeVector3(0.483971, 0.502302, 0.716565),
		b3MakeVector3(0.232822, 0.657519, 0.716563),
		b3MakeVector3(-0.723607, 0.447220, 0.525725),
		b3MakeVector3(-0.531941, 0.502302, 0.681712),
		b3MakeVector3(-0.609547, 0.657519, 0.442856),
		b3MakeVector3(-0.723607, 0.447220, -0.525725),
		b3MakeVector3(-0.812729, 0.502301, -0.295238),
		b3MakeVector3(-0.609547, 0.657519, -0.442856),
		b3MakeVector3(0.276388, 0.447220, -0.850649),
		b3MakeVector3(0.029639, 0.502302, -0.864184),
		b3MakeVector3(0.232822, 0.657519, -0.716563),
		b3MakeVector3(0.894426, 0.447216, 0.000000),
		b3MakeVector3(0.831051, 0.502299, -0.238853),
		b3MakeVector3(0.753442, 0.657515, 0.000000),
		b3MakeVector3(-0.232822, -0.657519, 0.716563),
		b3MakeVector3(-0.162456, -0.850654, 0.499995),
		b3MakeVector3(0.052790, -0.723612, 0.688185),
		b3MakeVector3(0.138199, -0.894429, 0.425321),
		b3MakeVector3(0.262869, -0.525738, 0.809012),
		b3MakeVector3(0.361805, -0.723611, 0.587779),
		b3MakeVector3(0.531941, -0.502302, 0.681712),
		b3MakeVector3(0.425323, -0.850654, 0.309011),
		b3MakeVector3(0.812729, -0.502301, -0.295238),
		b3MakeVector3(0.609547, -0.657519, -0.442856),
		b3MakeVector3(0.850648, -0.525736, 0.000000),
		b3MakeVector3(0.670817, -0.723611, -0.162457),
		b3MakeVector3(0.670817, -0.723610, 0.162458),
		b3MakeVector3(0.425323, -0.850654, -0.309011),
		b3MakeVector3(0.447211, -0.894428, 0.000001),
		b3MakeVector3(-0.753442, -0.657515, 0.000000),
		b3MakeVector3(-0.525730, -0.850652, 0.000000),
		b3MakeVector3(-0.638195, -0.723609, 0.262864),
		b3MakeVector3(-0.361801, -0.894428, 0.262864),
		b3MakeVector3(-0.688189, -0.525736, 0.499997),
		b3MakeVector3(-0.447211, -0.723610, 0.525729),
		b3MakeVector3(-0.483971, -0.502302, 0.716565),
		b3MakeVector3(-0.232822, -0.657519, -0.716563),
		b3MakeVector3(-0.162456, -0.850654, -0.499995),
		b3MakeVector3(-0.447211, -0.723611, -0.525727),
		b3MakeVector3(-0.361801, -0.894429, -0.262863),
		b3MakeVector3(-0.688189, -0.525736, -0.499997),
		b3MakeVector3(-0.638195, -0.723609, -0.262863),
		b3MakeVector3(-0.831051, -0.502299, -0.238853),
		b3MakeVector3(0.361804, -0.723612, -0.587779),
		b3MakeVector3(0.138197, -0.894429, -0.425321),
		b3MakeVector3(0.262869, -0.525738, -0.809012),
		b3MakeVector3(0.052789, -0.723611, -0.688186),
		b3MakeVector3(-0.029639, -0.502302, -0.864184),
		b3MakeVector3(0.956626, 0.251149, 0.147618),
		b3MakeVector3(0.956626, 0.251149, -0.147618),
		b3MakeVector3(0.951058, -0.000000, 0.309013),
		b3MakeVector3(1.000000, 0.000000, 0.000000),
		b3MakeVector3(0.947213, -0.276396, 0.162458),
		b3MakeVector3(0.951058, 0.000000, -0.309013),
		b3MakeVector3(0.947213, -0.276396, -0.162458),
		b3MakeVector3(0.155215, 0.251152, 0.955422),
		b3MakeVector3(0.436007, 0.251152, 0.864188),
		b3MakeVector3(-0.000000, -0.000000, 1.000000),
		b3MakeVector3(0.309017, 0.000000, 0.951056),
		b3MakeVector3(0.138199, -0.276398, 0.951055),
		b3MakeVector3(0.587786, 0.000000, 0.809017),
		b3MakeVector3(0.447216, -0.276398, 0.850648),
		b3MakeVector3(-0.860698, 0.251151, 0.442858),
		b3MakeVector3(-0.687159, 0.251152, 0.681715),
		b3MakeVector3(-0.951058, -0.000000, 0.309013),
		b3MakeVector3(-0.809018, 0.000000, 0.587783),
		b3MakeVector3(-0.861803, -0.276396, 0.425324),
		b3MakeVector3(-0.587786, 0.000000, 0.809017),
		b3MakeVector3(-0.670819, -0.276397, 0.688191),
		b3MakeVector3(-0.687159, 0.251152, -0.681715),
		b3MakeVector3(-0.860698, 0.251151, -0.442858),
		b3MakeVector3(-0.587786, -0.000000, -0.809017),
		b3MakeVector3(-0.809018, -0.000000, -0.587783),
		b3MakeVector3(-0.670819, -0.276397, -0.688191),
		b3MakeVector3(-0.951058, 0.000000, -0.309013),
		b3MakeVector3(-0.861803, -0.276396, -0.425324),
		b3MakeVector3(0.436007, 0.251152, -0.864188),
		b3MakeVector3(0.155215, 0.251152, -0.955422),
		b3MakeVector3(0.587786, -0.000000, -0.809017),
		b3MakeVector3(0.309017, -0.000000, -0.951056),
		b3MakeVector3(0.447216, -0.276398, -0.850648),
		b3MakeVector3(0.000000, 0.000000, -1.000000),
		b3MakeVector3(0.138199, -0.276398, -0.951055),
		b3MakeVector3(0.670820, 0.276396, 0.688190),
		b3MakeVector3(0.809019, -0.000002, 0.587783),
		b3MakeVector3(0.688189, 0.525736, 0.499997),
		b3MakeVector3(0.861804, 0.276394, 0.425323),
		b3MakeVector3(0.831051, 0.502299, 0.238853),
		b3MakeVector3(-0.447216, 0.276397, 0.850649),
		b3MakeVector3(-0.309017, -0.000001, 0.951056),
		b3MakeVector3(-0.262869, 0.525738, 0.809012),
		b3MakeVector3(-0.138199, 0.276397, 0.951055),
		b3MakeVector3(0.029639, 0.502302, 0.864184),
		b3MakeVector3(-0.947213, 0.276396, -0.162458),
		b3MakeVector3(-1.000000, 0.000001, 0.000000),
		b3MakeVector3(-0.850648, 0.525736, -0.000000),
		b3MakeVector3(-0.947213, 0.276397, 0.162458),
		b3MakeVector3(-0.812729, 0.502301, 0.295238),
		b3MakeVector3(-0.138199, 0.276397, -0.951055),
		b3MakeVector3(-0.309016, -0.000000, -0.951057),
		b3MakeVector3(-0.262869, 0.525738, -0.809012),
		b3MakeVector3(-0.447215, 0.276397, -0.850649),
		b3MakeVector3(-0.531941, 0.502302, -0.681712),
		b3MakeVector3(0.861804, 0.276396, -0.425322),
		b3MakeVector3(0.809019, 0.000000, -0.587782),
		b3MakeVector3(0.688189, 0.525736, -0.499997),
		b3MakeVector3(0.670821, 0.276397, -0.688189),
		b3MakeVector3(0.483971, 0.502302, -0.716565),
		b3MakeVector3(0.077607, 0.967950, 0.238853),
		b3MakeVector3(0.251147, 0.967949, 0.000000),
		b3MakeVector3(0.000000, 1.000000, 0.000000),
		b3MakeVector3(0.162456, 0.850654, 0.499995),
		b3MakeVector3(0.361800, 0.894429, 0.262863),
		b3MakeVector3(0.447209, 0.723612, 0.525728),
		b3MakeVector3(0.525730, 0.850652, 0.000000),
		b3MakeVector3(0.638194, 0.723610, 0.262864),
		b3MakeVector3(-0.203181, 0.967950, 0.147618),
		b3MakeVector3(-0.425323, 0.850654, 0.309011),
		b3MakeVector3(-0.138197, 0.894430, 0.425320),
		b3MakeVector3(-0.361804, 0.723612, 0.587778),
		b3MakeVector3(-0.052790, 0.723612, 0.688185),
		b3MakeVector3(-0.203181, 0.967950, -0.147618),
		b3MakeVector3(-0.425323, 0.850654, -0.309011),
		b3MakeVector3(-0.447210, 0.894429, 0.000000),
		b3MakeVector3(-0.670817, 0.723611, -0.162457),
		b3MakeVector3(-0.670817, 0.723611, 0.162457),
		b3MakeVector3(0.077607, 0.967950, -0.238853),
		b3MakeVector3(0.162456, 0.850654, -0.499995),
		b3MakeVector3(-0.138197, 0.894430, -0.425320),
		b3MakeVector3(-0.052790, 0.723612, -0.688185),
		b3MakeVector3(-0.361804, 0.723612, -0.587778),
		b3MakeVector3(0.361800, 0.894429, -0.262863),
		b3MakeVector3(0.638194, 0.723610, -0.262864),
		b3MakeVector3(0.447209, 0.723612, -0.525728)};

bool b3FindSeparatingAxisEdgeEdge(const b3ConvexPolyhedronData* hullA, __global const b3ConvexPolyhedronData* hullB,
								  b3Float4ConstArg posA1,
								  b3QuatConstArg ornA,
								  b3Float4ConstArg posB1,
								  b3QuatConstArg ornB,
								  b3Float4ConstArg DeltaC2,
								  const b3Float4* verticesA,
								  const b3Float4* uniqueEdgesA,
								  const b3GpuFace* facesA,
								  const int* indicesA,
								  __global const b3Float4* verticesB,
								  __global const b3Float4* uniqueEdgesB,
								  __global const b3GpuFace* facesB,
								  __global const int* indicesB,
								  b3Float4* sep,
								  float* dmin,
								  bool searchAllEdgeEdge)
{
	b3Float4 posA = posA1;
	posA.w = 0.f;
	b3Float4 posB = posB1;
	posB.w = 0.f;

	//	int curPlaneTests=0;

	int curEdgeEdge = 0;
	// Test edges
	static int maxEdgeTests = 0;
	int curEdgeTests = hullA->m_numUniqueEdges * hullB->m_numUniqueEdges;
	if (curEdgeTests > maxEdgeTests)
	{
		maxEdgeTests = curEdgeTests;
		printf("maxEdgeTests = %d\n", maxEdgeTests);
		printf("hullA->m_numUniqueEdges = %d\n", hullA->m_numUniqueEdges);
		printf("hullB->m_numUniqueEdges = %d\n", hullB->m_numUniqueEdges);
	}

	if (searchAllEdgeEdge)
	{
		for (int e0 = 0; e0 < hullA->m_numUniqueEdges; e0++)
		{
			const b3Float4 edge0 = uniqueEdgesA[hullA->m_uniqueEdgesOffset + e0];
			b3Float4 edge0World = b3QuatRotate(ornA, edge0);

			for (int e1 = 0; e1 < hullB->m_numUniqueEdges; e1++)
			{
				const b3Float4 edge1 = uniqueEdgesB[hullB->m_uniqueEdgesOffset + e1];
				b3Float4 edge1World = b3QuatRotate(ornB, edge1);

				b3Float4 crossje = b3Cross(edge0World, edge1World);

				curEdgeEdge++;
				if (!b3IsAlmostZero(crossje))
				{
					crossje = b3Normalized(crossje);
					if (b3Dot(DeltaC2, crossje) < 0)
						crossje *= -1.f;

					float dist;
					bool result = true;
					{
						float Min0, Max0;
						float Min1, Max1;
						b3Project(hullA, posA, ornA, &crossje, verticesA, &Min0, &Max0);
						b3Project(hullB, posB, ornB, &crossje, verticesB, &Min1, &Max1);

						if (Max0 < Min1 || Max1 < Min0)
							return false;

						float d0 = Max0 - Min1;
						float d1 = Max1 - Min0;
						dist = d0 < d1 ? d0 : d1;
						result = true;
					}

					if (dist < *dmin)
					{
						*dmin = dist;
						*sep = crossje;
					}
				}
			}
		}
	}
	else
	{
		int numDirections = sizeof(unitSphere162) / sizeof(b3Vector3);
		//printf("numDirections =%d\n",numDirections );

		for (int i = 0; i < numDirections; i++)
		{
			b3Float4 crossje = unitSphere162[i];
			{
				//if (b3Dot(DeltaC2,crossje)>0)
				{
					float dist;
					bool result = true;
					{
						float Min0, Max0;
						float Min1, Max1;
						b3Project(hullA, posA, ornA, &crossje, verticesA, &Min0, &Max0);
						b3Project(hullB, posB, ornB, &crossje, verticesB, &Min1, &Max1);

						if (Max0 < Min1 || Max1 < Min0)
							return false;

						float d0 = Max0 - Min1;
						float d1 = Max1 - Min0;
						dist = d0 < d1 ? d0 : d1;
						result = true;
					}

					if (dist < *dmin)
					{
						*dmin = dist;
						*sep = crossje;
					}
				}
			}
		}
	}

	if ((b3Dot(-DeltaC2, *sep)) > 0.0f)
	{
		*sep = -(*sep);
	}
	return true;
}

inline int b3FindClippingFaces(b3Float4ConstArg separatingNormal,
							   __global const b3ConvexPolyhedronData_t* hullA, __global const b3ConvexPolyhedronData_t* hullB,
							   b3Float4ConstArg posA, b3QuatConstArg ornA, b3Float4ConstArg posB, b3QuatConstArg ornB,
							   __global b3Float4* worldVertsA1,
							   __global b3Float4* worldNormalsA1,
							   __global b3Float4* worldVertsB1,
							   int capacityWorldVerts,
							   const float minDist, float maxDist,
							   __global const b3Float4* verticesA,
							   __global const b3GpuFace_t* facesA,
							   __global const int* indicesA,
							   __global const b3Float4* verticesB,
							   __global const b3GpuFace_t* facesB,
							   __global const int* indicesB,

							   __global b3Int4* clippingFaces, int pairIndex)
{
	int numContactsOut = 0;
	int numWorldVertsB1 = 0;

	int closestFaceB = -1;
	float dmax = -FLT_MAX;

	{
		for (int face = 0; face < hullB->m_numFaces; face++)
		{
			const b3Float4 Normal = b3MakeFloat4(facesB[hullB->m_faceOffset + face].m_plane.x,
												 facesB[hullB->m_faceOffset + face].m_plane.y, facesB[hullB->m_faceOffset + face].m_plane.z, 0.f);
			const b3Float4 WorldNormal = b3QuatRotate(ornB, Normal);
			float d = b3Dot(WorldNormal, separatingNormal);
			if (d > dmax)
			{
				dmax = d;
				closestFaceB = face;
			}
		}
	}

	{
		const b3GpuFace_t polyB = facesB[hullB->m_faceOffset + closestFaceB];
		const int numVertices = polyB.m_numIndices;
		for (int e0 = 0; e0 < numVertices; e0++)
		{
			const b3Float4 b = verticesB[hullB->m_vertexOffset + indicesB[polyB.m_indexOffset + e0]];
			worldVertsB1[pairIndex * capacityWorldVerts + numWorldVertsB1++] = b3TransformPoint(b, posB, ornB);
		}
	}

	int closestFaceA = -1;
	{
		float dmin = FLT_MAX;
		for (int face = 0; face < hullA->m_numFaces; face++)
		{
			const b3Float4 Normal = b3MakeFloat4(
				facesA[hullA->m_faceOffset + face].m_plane.x,
				facesA[hullA->m_faceOffset + face].m_plane.y,
				facesA[hullA->m_faceOffset + face].m_plane.z,
				0.f);
			const b3Float4 faceANormalWS = b3QuatRotate(ornA, Normal);

			float d = b3Dot(faceANormalWS, separatingNormal);
			if (d < dmin)
			{
				dmin = d;
				closestFaceA = face;
				worldNormalsA1[pairIndex] = faceANormalWS;
			}
		}
	}

	int numVerticesA = facesA[hullA->m_faceOffset + closestFaceA].m_numIndices;
	for (int e0 = 0; e0 < numVerticesA; e0++)
	{
		const b3Float4 a = verticesA[hullA->m_vertexOffset + indicesA[facesA[hullA->m_faceOffset + closestFaceA].m_indexOffset + e0]];
		worldVertsA1[pairIndex * capacityWorldVerts + e0] = b3TransformPoint(a, posA, ornA);
	}

	clippingFaces[pairIndex].x = closestFaceA;
	clippingFaces[pairIndex].y = closestFaceB;
	clippingFaces[pairIndex].z = numVerticesA;
	clippingFaces[pairIndex].w = numWorldVertsB1;

	return numContactsOut;
}

__kernel void b3FindConcaveSeparatingAxisKernel(__global b3Int4* concavePairs,
												__global const b3RigidBodyData* rigidBodies,
												__global const b3Collidable* collidables,
												__global const b3ConvexPolyhedronData* convexShapes,
												__global const b3Float4* vertices,
												__global const b3Float4* uniqueEdges,
												__global const b3GpuFace* faces,
												__global const int* indices,
												__global const b3GpuChildShape* gpuChildShapes,
												__global b3Aabb* aabbs,
												__global b3Float4* concaveSeparatingNormalsOut,
												__global b3Int4* clippingFacesOut,
												__global b3Vector3* worldVertsA1Out,
												__global b3Vector3* worldNormalsA1Out,
												__global b3Vector3* worldVertsB1Out,
												__global int* hasSeparatingNormals,
												int vertexFaceCapacity,
												int numConcavePairs,
												int pairIdx)
{
	int i = pairIdx;
	/*	int i = get_global_id(0);
	if (i>=numConcavePairs)
		return;
	int pairIdx = i;
	*/

	int bodyIndexA = concavePairs[i].x;
	int bodyIndexB = concavePairs[i].y;

	int collidableIndexA = rigidBodies[bodyIndexA].m_collidableIdx;
	int collidableIndexB = rigidBodies[bodyIndexB].m_collidableIdx;

	int shapeIndexA = collidables[collidableIndexA].m_shapeIndex;
	int shapeIndexB = collidables[collidableIndexB].m_shapeIndex;

	if (collidables[collidableIndexB].m_shapeType != SHAPE_CONVEX_HULL &&
		collidables[collidableIndexB].m_shapeType != SHAPE_COMPOUND_OF_CONVEX_HULLS)
	{
		concavePairs[pairIdx].w = -1;
		return;
	}

	hasSeparatingNormals[i] = 0;

	//	int numFacesA = convexShapes[shapeIndexA].m_numFaces;
	int numActualConcaveConvexTests = 0;

	int f = concavePairs[i].z;

	bool overlap = false;

	b3ConvexPolyhedronData convexPolyhedronA;

	//add 3 vertices of the triangle
	convexPolyhedronA.m_numVertices = 3;
	convexPolyhedronA.m_vertexOffset = 0;
	b3Float4 localCenter = b3MakeFloat4(0.f, 0.f, 0.f, 0.f);

	b3GpuFace face = faces[convexShapes[shapeIndexA].m_faceOffset + f];
	b3Aabb triAabb;
	triAabb.m_minVec = b3MakeFloat4(1e30f, 1e30f, 1e30f, 0.f);
	triAabb.m_maxVec = b3MakeFloat4(-1e30f, -1e30f, -1e30f, 0.f);

	b3Float4 verticesA[3];
	for (int i = 0; i < 3; i++)
	{
		int index = indices[face.m_indexOffset + i];
		b3Float4 vert = vertices[convexShapes[shapeIndexA].m_vertexOffset + index];
		verticesA[i] = vert;
		localCenter += vert;

		triAabb.m_minVec = b3MinFloat4(triAabb.m_minVec, vert);
		triAabb.m_maxVec = b3MaxFloat4(triAabb.m_maxVec, vert);
	}

	overlap = true;
	overlap = (triAabb.m_minVec.x > aabbs[bodyIndexB].m_maxVec.x || triAabb.m_maxVec.x < aabbs[bodyIndexB].m_minVec.x) ? false : overlap;
	overlap = (triAabb.m_minVec.z > aabbs[bodyIndexB].m_maxVec.z || triAabb.m_maxVec.z < aabbs[bodyIndexB].m_minVec.z) ? false : overlap;
	overlap = (triAabb.m_minVec.y > aabbs[bodyIndexB].m_maxVec.y || triAabb.m_maxVec.y < aabbs[bodyIndexB].m_minVec.y) ? false : overlap;

	if (overlap)
	{
		float dmin = FLT_MAX;
		int hasSeparatingAxis = 5;
		b3Float4 sepAxis = b3MakeFloat4(1, 2, 3, 4);

		//	int localCC=0;
		numActualConcaveConvexTests++;

		//a triangle has 3 unique edges
		convexPolyhedronA.m_numUniqueEdges = 3;
		convexPolyhedronA.m_uniqueEdgesOffset = 0;
		b3Float4 uniqueEdgesA[3];

		uniqueEdgesA[0] = (verticesA[1] - verticesA[0]);
		uniqueEdgesA[1] = (verticesA[2] - verticesA[1]);
		uniqueEdgesA[2] = (verticesA[0] - verticesA[2]);

		convexPolyhedronA.m_faceOffset = 0;

		b3Float4 normal = b3MakeFloat4(face.m_plane.x, face.m_plane.y, face.m_plane.z, 0.f);

		b3GpuFace facesA[B3_TRIANGLE_NUM_CONVEX_FACES];
		int indicesA[3 + 3 + 2 + 2 + 2];
		int curUsedIndices = 0;
		int fidx = 0;

		//front size of triangle
		{
			facesA[fidx].m_indexOffset = curUsedIndices;
			indicesA[0] = 0;
			indicesA[1] = 1;
			indicesA[2] = 2;
			curUsedIndices += 3;
			float c = face.m_plane.w;
			facesA[fidx].m_plane.x = normal.x;
			facesA[fidx].m_plane.y = normal.y;
			facesA[fidx].m_plane.z = normal.z;
			facesA[fidx].m_plane.w = c;
			facesA[fidx].m_numIndices = 3;
		}
		fidx++;
		//back size of triangle
		{
			facesA[fidx].m_indexOffset = curUsedIndices;
			indicesA[3] = 2;
			indicesA[4] = 1;
			indicesA[5] = 0;
			curUsedIndices += 3;
			float c = b3Dot(normal, verticesA[0]);
			//	float c1 = -face.m_plane.w;
			facesA[fidx].m_plane.x = -normal.x;
			facesA[fidx].m_plane.y = -normal.y;
			facesA[fidx].m_plane.z = -normal.z;
			facesA[fidx].m_plane.w = c;
			facesA[fidx].m_numIndices = 3;
		}
		fidx++;

		bool addEdgePlanes = true;
		if (addEdgePlanes)
		{
			int numVertices = 3;
			int prevVertex = numVertices - 1;
			for (int i = 0; i < numVertices; i++)
			{
				b3Float4 v0 = verticesA[i];
				b3Float4 v1 = verticesA[prevVertex];

				b3Float4 edgeNormal = b3Normalized(b3Cross(normal, v1 - v0));
				float c = -b3Dot(edgeNormal, v0);

				facesA[fidx].m_numIndices = 2;
				facesA[fidx].m_indexOffset = curUsedIndices;
				indicesA[curUsedIndices++] = i;
				indicesA[curUsedIndices++] = prevVertex;

				facesA[fidx].m_plane.x = edgeNormal.x;
				facesA[fidx].m_plane.y = edgeNormal.y;
				facesA[fidx].m_plane.z = edgeNormal.z;
				facesA[fidx].m_plane.w = c;
				fidx++;
				prevVertex = i;
			}
		}
		convexPolyhedronA.m_numFaces = B3_TRIANGLE_NUM_CONVEX_FACES;
		convexPolyhedronA.m_localCenter = localCenter * (1.f / 3.f);

		b3Float4 posA = rigidBodies[bodyIndexA].m_pos;
		posA.w = 0.f;
		b3Float4 posB = rigidBodies[bodyIndexB].m_pos;
		posB.w = 0.f;

		b3Quaternion ornA = rigidBodies[bodyIndexA].m_quat;
		b3Quaternion ornB = rigidBodies[bodyIndexB].m_quat;

		///////////////////
		///compound shape support

		if (collidables[collidableIndexB].m_shapeType == SHAPE_COMPOUND_OF_CONVEX_HULLS)
		{
			int compoundChild = concavePairs[pairIdx].w;
			int childShapeIndexB = compoundChild;  //collidables[collidableIndexB].m_shapeIndex+compoundChild;
			int childColIndexB = gpuChildShapes[childShapeIndexB].m_shapeIndex;
			b3Float4 childPosB = gpuChildShapes[childShapeIndexB].m_childPosition;
			b3Quaternion childOrnB = gpuChildShapes[childShapeIndexB].m_childOrientation;
			b3Float4 newPosB = b3TransformPoint(childPosB, posB, ornB);
			b3Quaternion newOrnB = b3QuatMul(ornB, childOrnB);
			posB = newPosB;
			ornB = newOrnB;
			shapeIndexB = collidables[childColIndexB].m_shapeIndex;
		}
		//////////////////

		b3Float4 c0local = convexPolyhedronA.m_localCenter;
		b3Float4 c0 = b3TransformPoint(c0local, posA, ornA);
		b3Float4 c1local = convexShapes[shapeIndexB].m_localCenter;
		b3Float4 c1 = b3TransformPoint(c1local, posB, ornB);
		const b3Float4 DeltaC2 = c0 - c1;

		bool sepA = b3FindSeparatingAxis(&convexPolyhedronA, &convexShapes[shapeIndexB],
										 posA, ornA,
										 posB, ornB,
										 DeltaC2,
										 verticesA, uniqueEdgesA, facesA, indicesA,
										 vertices, uniqueEdges, faces, indices,
										 &sepAxis, &dmin);
		hasSeparatingAxis = 4;
		if (!sepA)
		{
			hasSeparatingAxis = 0;
		}
		else
		{
			bool sepB = b3FindSeparatingAxis(&convexShapes[shapeIndexB], &convexPolyhedronA,
											 posB, ornB,
											 posA, ornA,
											 DeltaC2,
											 vertices, uniqueEdges, faces, indices,
											 verticesA, uniqueEdgesA, facesA, indicesA,
											 &sepAxis, &dmin);

			if (!sepB)
			{
				hasSeparatingAxis = 0;
			}
			else
			{
				bool sepEE = b3FindSeparatingAxisEdgeEdge(&convexPolyhedronA, &convexShapes[shapeIndexB],
														  posA, ornA,
														  posB, ornB,
														  DeltaC2,
														  verticesA, uniqueEdgesA, facesA, indicesA,
														  vertices, uniqueEdges, faces, indices,
														  &sepAxis, &dmin, true);

				if (!sepEE)
				{
					hasSeparatingAxis = 0;
				}
				else
				{
					hasSeparatingAxis = 1;
				}
			}
		}

		if (hasSeparatingAxis)
		{
			hasSeparatingNormals[i] = 1;
			sepAxis.w = dmin;
			concaveSeparatingNormalsOut[pairIdx] = sepAxis;

			//now compute clipping faces A and B, and world-space clipping vertices A and B...

			float minDist = -1e30f;
			float maxDist = 0.02f;

			b3FindClippingFaces(sepAxis,
								&convexPolyhedronA,
								&convexShapes[shapeIndexB],
								posA, ornA,
								posB, ornB,
								worldVertsA1Out,
								worldNormalsA1Out,
								worldVertsB1Out,
								vertexFaceCapacity,
								minDist, maxDist,
								verticesA,
								facesA,
								indicesA,

								vertices,
								faces,
								indices,
								clippingFacesOut, pairIdx);
		}
		else
		{
			//mark this pair as in-active
			concavePairs[pairIdx].w = -1;
		}
	}
	else
	{
		//mark this pair as in-active
		concavePairs[pairIdx].w = -1;
	}
}

#endif  //B3_FIND_CONCAVE_SEPARATING_AXIS_H
