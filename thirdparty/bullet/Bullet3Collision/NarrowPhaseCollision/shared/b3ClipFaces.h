#ifndef B3_CLIP_FACES_H
#define B3_CLIP_FACES_H

#include "Bullet3Common/shared/b3Int4.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3BvhSubtreeInfoData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3QuantizedBvhNodeData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3ConvexPolyhedronData.h"

inline b3Float4 b3Lerp3(b3Float4ConstArg a, b3Float4ConstArg b, float t)
{
	return b3MakeFloat4(a.x + (b.x - a.x) * t,
						a.y + (b.y - a.y) * t,
						a.z + (b.z - a.z) * t,
						0.f);
}

// Clips a face to the back of a plane, return the number of vertices out, stored in ppVtxOut
int clipFaceGlobal(__global const b3Float4* pVtxIn, int numVertsIn, b3Float4ConstArg planeNormalWS, float planeEqWS, __global b3Float4* ppVtxOut)
{
	int ve;
	float ds, de;
	int numVertsOut = 0;
	//double-check next test
	//	if (numVertsIn < 2)
	//		return 0;

	b3Float4 firstVertex = pVtxIn[numVertsIn - 1];
	b3Float4 endVertex = pVtxIn[0];

	ds = b3Dot(planeNormalWS, firstVertex) + planeEqWS;

	for (ve = 0; ve < numVertsIn; ve++)
	{
		endVertex = pVtxIn[ve];
		de = b3Dot(planeNormalWS, endVertex) + planeEqWS;
		if (ds < 0)
		{
			if (de < 0)
			{
				// Start < 0, end < 0, so output endVertex
				ppVtxOut[numVertsOut++] = endVertex;
			}
			else
			{
				// Start < 0, end >= 0, so output intersection
				ppVtxOut[numVertsOut++] = b3Lerp3(firstVertex, endVertex, (ds * 1.f / (ds - de)));
			}
		}
		else
		{
			if (de < 0)
			{
				// Start >= 0, end < 0 so output intersection and end
				ppVtxOut[numVertsOut++] = b3Lerp3(firstVertex, endVertex, (ds * 1.f / (ds - de)));
				ppVtxOut[numVertsOut++] = endVertex;
			}
		}
		firstVertex = endVertex;
		ds = de;
	}
	return numVertsOut;
}

__kernel void clipFacesAndFindContactsKernel(__global const b3Float4* separatingNormals,
											 __global const int* hasSeparatingAxis,
											 __global b3Int4* clippingFacesOut,
											 __global b3Float4* worldVertsA1,
											 __global b3Float4* worldNormalsA1,
											 __global b3Float4* worldVertsB1,
											 __global b3Float4* worldVertsB2,
											 int vertexFaceCapacity,
											 int pairIndex)
{
	//    int i = get_global_id(0);
	//int pairIndex = i;
	int i = pairIndex;

	float minDist = -1e30f;
	float maxDist = 0.02f;

	//	if (i<numPairs)
	{
		if (hasSeparatingAxis[i])
		{
			//			int bodyIndexA = pairs[i].x;
			//		int bodyIndexB = pairs[i].y;

			int numLocalContactsOut = 0;

			int capacityWorldVertsB2 = vertexFaceCapacity;

			__global b3Float4* pVtxIn = &worldVertsB1[pairIndex * capacityWorldVertsB2];
			__global b3Float4* pVtxOut = &worldVertsB2[pairIndex * capacityWorldVertsB2];

			{
				__global b3Int4* clippingFaces = clippingFacesOut;

				int closestFaceA = clippingFaces[pairIndex].x;
				// int closestFaceB = clippingFaces[pairIndex].y;
				int numVertsInA = clippingFaces[pairIndex].z;
				int numVertsInB = clippingFaces[pairIndex].w;

				int numVertsOut = 0;

				if (closestFaceA >= 0)
				{
					// clip polygon to back of planes of all faces of hull A that are adjacent to witness face

					for (int e0 = 0; e0 < numVertsInA; e0++)
					{
						const b3Float4 aw = worldVertsA1[pairIndex * capacityWorldVertsB2 + e0];
						const b3Float4 bw = worldVertsA1[pairIndex * capacityWorldVertsB2 + ((e0 + 1) % numVertsInA)];
						const b3Float4 WorldEdge0 = aw - bw;
						b3Float4 worldPlaneAnormal1 = worldNormalsA1[pairIndex];
						b3Float4 planeNormalWS1 = -b3Cross(WorldEdge0, worldPlaneAnormal1);
						b3Float4 worldA1 = aw;
						float planeEqWS1 = -b3Dot(worldA1, planeNormalWS1);
						b3Float4 planeNormalWS = planeNormalWS1;
						float planeEqWS = planeEqWS1;
						numVertsOut = clipFaceGlobal(pVtxIn, numVertsInB, planeNormalWS, planeEqWS, pVtxOut);
						__global b3Float4* tmp = pVtxOut;
						pVtxOut = pVtxIn;
						pVtxIn = tmp;
						numVertsInB = numVertsOut;
						numVertsOut = 0;
					}

					b3Float4 planeNormalWS = worldNormalsA1[pairIndex];
					float planeEqWS = -b3Dot(planeNormalWS, worldVertsA1[pairIndex * capacityWorldVertsB2]);

					for (int i = 0; i < numVertsInB; i++)
					{
						float depth = b3Dot(planeNormalWS, pVtxIn[i]) + planeEqWS;
						if (depth <= minDist)
						{
							depth = minDist;
						}
						/*
						static float maxDepth = 0.f;
						if (depth < maxDepth)
						{
							maxDepth = depth;
							if (maxDepth < -10)
							{
								printf("error at framecount %d?\n",myframecount);
							}
							printf("maxDepth = %f\n", maxDepth);

						}
*/
						if (depth <= maxDist)
						{
							b3Float4 pointInWorld = pVtxIn[i];
							pVtxOut[numLocalContactsOut++] = b3MakeFloat4(pointInWorld.x, pointInWorld.y, pointInWorld.z, depth);
						}
					}
				}
				clippingFaces[pairIndex].w = numLocalContactsOut;
			}

			for (int i = 0; i < numLocalContactsOut; i++)
				pVtxIn[i] = pVtxOut[i];

		}  //		if (hasSeparatingAxis[i])
	}      //	if (i<numPairs)
}

#endif  //B3_CLIP_FACES_H
