#include "btInternalEdgeUtility.h"

#include "BulletCollision/CollisionShapes/btBvhTriangleMeshShape.h"
#include "BulletCollision/CollisionShapes/btScaledBvhTriangleMeshShape.h"
#include "BulletCollision/CollisionShapes/btTriangleShape.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/NarrowPhaseCollision/btManifoldPoint.h"
#include "LinearMath/btIDebugDraw.h"
#include "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h"

//#define DEBUG_INTERNAL_EDGE

#ifdef DEBUG_INTERNAL_EDGE
#include <stdio.h>
#endif  //DEBUG_INTERNAL_EDGE

#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
static btIDebugDraw* gDebugDrawer = 0;

void btSetDebugDrawer(btIDebugDraw* debugDrawer)
{
	gDebugDrawer = debugDrawer;
}

static void btDebugDrawLine(const btVector3& from, const btVector3& to, const btVector3& color)
{
	if (gDebugDrawer)
		gDebugDrawer->drawLine(from, to, color);
}
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

static int btGetHash(int partId, int triangleIndex)
{
	int hash = (partId << (31 - MAX_NUM_PARTS_IN_BITS)) | triangleIndex;
	return hash;
}

static btScalar btGetAngle(const btVector3& edgeA, const btVector3& normalA, const btVector3& normalB)
{
	const btVector3 refAxis0 = edgeA;
	const btVector3 refAxis1 = normalA;
	const btVector3 swingAxis = normalB;
	btScalar angle = btAtan2(swingAxis.dot(refAxis0), swingAxis.dot(refAxis1));
	return angle;
}

struct btConnectivityProcessor : public btTriangleCallback
{
	int m_partIdA;
	int m_triangleIndexA;
	btVector3* m_triangleVerticesA;
	btTriangleInfoMap* m_triangleInfoMap;

	virtual void processTriangle(btVector3* triangle, int partId, int triangleIndex)
	{
		//skip self-collisions
		if ((m_partIdA == partId) && (m_triangleIndexA == triangleIndex))
			return;

		//skip duplicates (disabled for now)
		//if ((m_partIdA <= partId) && (m_triangleIndexA <= triangleIndex))
		//	return;

		//search for shared vertices and edges
		int numshared = 0;
		int sharedVertsA[3] = {-1, -1, -1};
		int sharedVertsB[3] = {-1, -1, -1};

		///skip degenerate triangles
		btScalar crossBSqr = ((triangle[1] - triangle[0]).cross(triangle[2] - triangle[0])).length2();
		if (crossBSqr < m_triangleInfoMap->m_equalVertexThreshold)
			return;

		btScalar crossASqr = ((m_triangleVerticesA[1] - m_triangleVerticesA[0]).cross(m_triangleVerticesA[2] - m_triangleVerticesA[0])).length2();
		///skip degenerate triangles
		if (crossASqr < m_triangleInfoMap->m_equalVertexThreshold)
			return;

#if 0
		printf("triangle A[0]	=	(%f,%f,%f)\ntriangle A[1]	=	(%f,%f,%f)\ntriangle A[2]	=	(%f,%f,%f)\n",
			m_triangleVerticesA[0].getX(),m_triangleVerticesA[0].getY(),m_triangleVerticesA[0].getZ(),
			m_triangleVerticesA[1].getX(),m_triangleVerticesA[1].getY(),m_triangleVerticesA[1].getZ(),
			m_triangleVerticesA[2].getX(),m_triangleVerticesA[2].getY(),m_triangleVerticesA[2].getZ());

		printf("partId=%d, triangleIndex=%d\n",partId,triangleIndex);
		printf("triangle B[0]	=	(%f,%f,%f)\ntriangle B[1]	=	(%f,%f,%f)\ntriangle B[2]	=	(%f,%f,%f)\n",
			triangle[0].getX(),triangle[0].getY(),triangle[0].getZ(),
			triangle[1].getX(),triangle[1].getY(),triangle[1].getZ(),
			triangle[2].getX(),triangle[2].getY(),triangle[2].getZ());
#endif

		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
			{
				if ((m_triangleVerticesA[i] - triangle[j]).length2() < m_triangleInfoMap->m_equalVertexThreshold)
				{
					sharedVertsA[numshared] = i;
					sharedVertsB[numshared] = j;
					numshared++;
					///degenerate case
					if (numshared >= 3)
						return;
				}
			}
			///degenerate case
			if (numshared >= 3)
				return;
		}
		switch (numshared)
		{
			case 0:
			{
				break;
			}
			case 1:
			{
				//shared vertex
				break;
			}
			case 2:
			{
				//shared edge
				//we need to make sure the edge is in the order V2V0 and not V0V2 so that the signs are correct
				if (sharedVertsA[0] == 0 && sharedVertsA[1] == 2)
				{
					sharedVertsA[0] = 2;
					sharedVertsA[1] = 0;
					int tmp = sharedVertsB[1];
					sharedVertsB[1] = sharedVertsB[0];
					sharedVertsB[0] = tmp;
				}

				int hash = btGetHash(m_partIdA, m_triangleIndexA);

				btTriangleInfo* info = m_triangleInfoMap->find(hash);
				if (!info)
				{
					btTriangleInfo tmp;
					m_triangleInfoMap->insert(hash, tmp);
					info = m_triangleInfoMap->find(hash);
				}

				int sumvertsA = sharedVertsA[0] + sharedVertsA[1];
				int otherIndexA = 3 - sumvertsA;

				btVector3 edge(m_triangleVerticesA[sharedVertsA[1]] - m_triangleVerticesA[sharedVertsA[0]]);

				btTriangleShape tA(m_triangleVerticesA[0], m_triangleVerticesA[1], m_triangleVerticesA[2]);
				int otherIndexB = 3 - (sharedVertsB[0] + sharedVertsB[1]);

				btTriangleShape tB(triangle[sharedVertsB[1]], triangle[sharedVertsB[0]], triangle[otherIndexB]);
				//btTriangleShape tB(triangle[0],triangle[1],triangle[2]);

				btVector3 normalA;
				btVector3 normalB;
				tA.calcNormal(normalA);
				tB.calcNormal(normalB);
				edge.normalize();
				btVector3 edgeCrossA = edge.cross(normalA).normalize();

				{
					btVector3 tmp = m_triangleVerticesA[otherIndexA] - m_triangleVerticesA[sharedVertsA[0]];
					if (edgeCrossA.dot(tmp) < 0)
					{
						edgeCrossA *= -1;
					}
				}

				btVector3 edgeCrossB = edge.cross(normalB).normalize();

				{
					btVector3 tmp = triangle[otherIndexB] - triangle[sharedVertsB[0]];
					if (edgeCrossB.dot(tmp) < 0)
					{
						edgeCrossB *= -1;
					}
				}

				btScalar angle2 = 0;
				btScalar ang4 = 0.f;

				btVector3 calculatedEdge = edgeCrossA.cross(edgeCrossB);
				btScalar len2 = calculatedEdge.length2();

				btScalar correctedAngle(0);
				//btVector3 calculatedNormalB = normalA;
				bool isConvex = false;

				if (len2 < m_triangleInfoMap->m_planarEpsilon)
				{
					angle2 = 0.f;
					ang4 = 0.f;
				}
				else
				{
					calculatedEdge.normalize();
					btVector3 calculatedNormalA = calculatedEdge.cross(edgeCrossA);
					calculatedNormalA.normalize();
					angle2 = btGetAngle(calculatedNormalA, edgeCrossA, edgeCrossB);
					ang4 = SIMD_PI - angle2;
					btScalar dotA = normalA.dot(edgeCrossB);
					///@todo: check if we need some epsilon, due to floating point imprecision
					isConvex = (dotA < 0.);

					correctedAngle = isConvex ? ang4 : -ang4;
				}

				//alternatively use
				//btVector3 calculatedNormalB2 = quatRotate(orn,normalA);

				switch (sumvertsA)
				{
					case 1:
					{
						btVector3 edge = m_triangleVerticesA[0] - m_triangleVerticesA[1];
						btQuaternion orn(edge, -correctedAngle);
						btVector3 computedNormalB = quatRotate(orn, normalA);
						btScalar bla = computedNormalB.dot(normalB);
						if (bla < 0)
						{
							computedNormalB *= -1;
							info->m_flags |= TRI_INFO_V0V1_SWAP_NORMALB;
						}
#ifdef DEBUG_INTERNAL_EDGE
						if ((computedNormalB - normalB).length() > 0.0001)
						{
							printf("warning: normals not identical\n");
						}
#endif  //DEBUG_INTERNAL_EDGE

						info->m_edgeV0V1Angle = -correctedAngle;

						if (isConvex)
							info->m_flags |= TRI_INFO_V0V1_CONVEX;
						break;
					}
					case 2:
					{
						btVector3 edge = m_triangleVerticesA[2] - m_triangleVerticesA[0];
						btQuaternion orn(edge, -correctedAngle);
						btVector3 computedNormalB = quatRotate(orn, normalA);
						if (computedNormalB.dot(normalB) < 0)
						{
							computedNormalB *= -1;
							info->m_flags |= TRI_INFO_V2V0_SWAP_NORMALB;
						}

#ifdef DEBUG_INTERNAL_EDGE
						if ((computedNormalB - normalB).length() > 0.0001)
						{
							printf("warning: normals not identical\n");
						}
#endif  //DEBUG_INTERNAL_EDGE
						info->m_edgeV2V0Angle = -correctedAngle;
						if (isConvex)
							info->m_flags |= TRI_INFO_V2V0_CONVEX;
						break;
					}
					case 3:
					{
						btVector3 edge = m_triangleVerticesA[1] - m_triangleVerticesA[2];
						btQuaternion orn(edge, -correctedAngle);
						btVector3 computedNormalB = quatRotate(orn, normalA);
						if (computedNormalB.dot(normalB) < 0)
						{
							info->m_flags |= TRI_INFO_V1V2_SWAP_NORMALB;
							computedNormalB *= -1;
						}
#ifdef DEBUG_INTERNAL_EDGE
						if ((computedNormalB - normalB).length() > 0.0001)
						{
							printf("warning: normals not identical\n");
						}
#endif  //DEBUG_INTERNAL_EDGE
						info->m_edgeV1V2Angle = -correctedAngle;

						if (isConvex)
							info->m_flags |= TRI_INFO_V1V2_CONVEX;
						break;
					}
				}

				break;
			}
			default:
			{
				//				printf("warning: duplicate triangle\n");
			}
		}
	}
};
/////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////

void btGenerateInternalEdgeInfo(btBvhTriangleMeshShape* trimeshShape, btTriangleInfoMap* triangleInfoMap)
{
	//the user pointer shouldn't already be used for other purposes, we intend to store connectivity info there!
	if (trimeshShape->getTriangleInfoMap())
		return;

	trimeshShape->setTriangleInfoMap(triangleInfoMap);

	btStridingMeshInterface* meshInterface = trimeshShape->getMeshInterface();
	const btVector3& meshScaling = meshInterface->getScaling();

	for (int partId = 0; partId < meshInterface->getNumSubParts(); partId++)
	{
		const unsigned char* vertexbase = 0;
		int numverts = 0;
		PHY_ScalarType type = PHY_INTEGER;
		int stride = 0;
		const unsigned char* indexbase = 0;
		int indexstride = 0;
		int numfaces = 0;
		PHY_ScalarType indicestype = PHY_INTEGER;
		//PHY_ScalarType indexType=0;

		btVector3 triangleVerts[3];
		meshInterface->getLockedReadOnlyVertexIndexBase(&vertexbase, numverts, type, stride, &indexbase, indexstride, numfaces, indicestype, partId);
		btVector3 aabbMin, aabbMax;

		for (int triangleIndex = 0; triangleIndex < numfaces; triangleIndex++)
		{
			unsigned int* gfxbase = (unsigned int*)(indexbase + triangleIndex * indexstride);

			for (int j = 2; j >= 0; j--)
			{
				int graphicsindex = indicestype == PHY_SHORT ? ((unsigned short*)gfxbase)[j] : gfxbase[j];
				if (type == PHY_FLOAT)
				{
					float* graphicsbase = (float*)(vertexbase + graphicsindex * stride);
					triangleVerts[j] = btVector3(
						graphicsbase[0] * meshScaling.getX(),
						graphicsbase[1] * meshScaling.getY(),
						graphicsbase[2] * meshScaling.getZ());
				}
				else
				{
					double* graphicsbase = (double*)(vertexbase + graphicsindex * stride);
					triangleVerts[j] = btVector3(btScalar(graphicsbase[0] * meshScaling.getX()), btScalar(graphicsbase[1] * meshScaling.getY()), btScalar(graphicsbase[2] * meshScaling.getZ()));
				}
			}
			aabbMin.setValue(btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT), btScalar(BT_LARGE_FLOAT));
			aabbMax.setValue(btScalar(-BT_LARGE_FLOAT), btScalar(-BT_LARGE_FLOAT), btScalar(-BT_LARGE_FLOAT));
			aabbMin.setMin(triangleVerts[0]);
			aabbMax.setMax(triangleVerts[0]);
			aabbMin.setMin(triangleVerts[1]);
			aabbMax.setMax(triangleVerts[1]);
			aabbMin.setMin(triangleVerts[2]);
			aabbMax.setMax(triangleVerts[2]);

			btConnectivityProcessor connectivityProcessor;
			connectivityProcessor.m_partIdA = partId;
			connectivityProcessor.m_triangleIndexA = triangleIndex;
			connectivityProcessor.m_triangleVerticesA = &triangleVerts[0];
			connectivityProcessor.m_triangleInfoMap = triangleInfoMap;

			trimeshShape->processAllTriangles(&connectivityProcessor, aabbMin, aabbMax);
		}
	}
}

// Given a point and a line segment (defined by two points), compute the closest point
// in the line.  Cap the point at the endpoints of the line segment.
void btNearestPointInLineSegment(const btVector3& point, const btVector3& line0, const btVector3& line1, btVector3& nearestPoint)
{
	btVector3 lineDelta = line1 - line0;

	// Handle degenerate lines
	if (lineDelta.fuzzyZero())
	{
		nearestPoint = line0;
	}
	else
	{
		btScalar delta = (point - line0).dot(lineDelta) / (lineDelta).dot(lineDelta);

		// Clamp the point to conform to the segment's endpoints
		if (delta < 0)
			delta = 0;
		else if (delta > 1)
			delta = 1;

		nearestPoint = line0 + lineDelta * delta;
	}
}

bool btClampNormal(const btVector3& edge, const btVector3& tri_normal_org, const btVector3& localContactNormalOnB, btScalar correctedEdgeAngle, btVector3& clampedLocalNormal)
{
	btVector3 tri_normal = tri_normal_org;
	//we only have a local triangle normal, not a local contact normal -> only normal in world space...
	//either compute the current angle all in local space, or all in world space

	btVector3 edgeCross = edge.cross(tri_normal).normalize();
	btScalar curAngle = btGetAngle(edgeCross, tri_normal, localContactNormalOnB);

	if (correctedEdgeAngle < 0)
	{
		if (curAngle < correctedEdgeAngle)
		{
			btScalar diffAngle = correctedEdgeAngle - curAngle;
			btQuaternion rotation(edge, diffAngle);
			clampedLocalNormal = btMatrix3x3(rotation) * localContactNormalOnB;
			return true;
		}
	}

	if (correctedEdgeAngle >= 0)
	{
		if (curAngle > correctedEdgeAngle)
		{
			btScalar diffAngle = correctedEdgeAngle - curAngle;
			btQuaternion rotation(edge, diffAngle);
			clampedLocalNormal = btMatrix3x3(rotation) * localContactNormalOnB;
			return true;
		}
	}
	return false;
}

/// Changes a btManifoldPoint collision normal to the normal from the mesh.
void btAdjustInternalEdgeContacts(btManifoldPoint& cp, const btCollisionObjectWrapper* colObj0Wrap, const btCollisionObjectWrapper* colObj1Wrap, int partId0, int index0, int normalAdjustFlags)
{
	//btAssert(colObj0->getCollisionShape()->getShapeType() == TRIANGLE_SHAPE_PROXYTYPE);
	if (colObj0Wrap->getCollisionShape()->getShapeType() != TRIANGLE_SHAPE_PROXYTYPE)
		return;

	btBvhTriangleMeshShape* trimesh = 0;

	if (colObj0Wrap->getCollisionObject()->getCollisionShape()->getShapeType() == SCALED_TRIANGLE_MESH_SHAPE_PROXYTYPE)
	{
		trimesh = ((btScaledBvhTriangleMeshShape*)colObj0Wrap->getCollisionObject()->getCollisionShape())->getChildShape();
	}
	else
	{
		if (colObj0Wrap->getCollisionObject()->getCollisionShape()->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE)
		{
			trimesh = (btBvhTriangleMeshShape*)colObj0Wrap->getCollisionObject()->getCollisionShape();
		}
	}
	if (trimesh == 0)
		return;

	btTriangleInfoMap* triangleInfoMapPtr = (btTriangleInfoMap*)trimesh->getTriangleInfoMap();
	if (!triangleInfoMapPtr)
		return;

	int hash = btGetHash(partId0, index0);

	btTriangleInfo* info = triangleInfoMapPtr->find(hash);
	if (!info)
		return;

	btScalar frontFacing = (normalAdjustFlags & BT_TRIANGLE_CONVEX_BACKFACE_MODE) == 0 ? 1.f : -1.f;

	const btTriangleShape* tri_shape = static_cast<const btTriangleShape*>(colObj0Wrap->getCollisionShape());
	btVector3 v0, v1, v2;
	tri_shape->getVertex(0, v0);
	tri_shape->getVertex(1, v1);
	tri_shape->getVertex(2, v2);

	//btVector3 center = (v0+v1+v2)*btScalar(1./3.);

	btVector3 red(1, 0, 0), green(0, 1, 0), blue(0, 0, 1), white(1, 1, 1), black(0, 0, 0);
	btVector3 tri_normal;
	tri_shape->calcNormal(tri_normal);

	//btScalar dot = tri_normal.dot(cp.m_normalWorldOnB);
	btVector3 nearest;
	btNearestPointInLineSegment(cp.m_localPointB, v0, v1, nearest);

	btVector3 contact = cp.m_localPointB;
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
	const btTransform& tr = colObj0->getWorldTransform();
	btDebugDrawLine(tr * nearest, tr * cp.m_localPointB, red);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

	bool isNearEdge = false;

	int numConcaveEdgeHits = 0;
	int numConvexEdgeHits = 0;

	btVector3 localContactNormalOnB = colObj0Wrap->getWorldTransform().getBasis().transpose() * cp.m_normalWorldOnB;
	localContactNormalOnB.normalize();  //is this necessary?

	// Get closest edge
	int bestedge = -1;
	btScalar disttobestedge = BT_LARGE_FLOAT;
	//
	// Edge 0 -> 1
	if (btFabs(info->m_edgeV0V1Angle) < triangleInfoMapPtr->m_maxEdgeAngleThreshold)
	{
		btVector3 nearest;
		btNearestPointInLineSegment(cp.m_localPointB, v0, v1, nearest);
		btScalar len = (contact - nearest).length();
		//
		if (len < disttobestedge)
		{
			bestedge = 0;
			disttobestedge = len;
		}
	}
	// Edge 1 -> 2
	if (btFabs(info->m_edgeV1V2Angle) < triangleInfoMapPtr->m_maxEdgeAngleThreshold)
	{
		btVector3 nearest;
		btNearestPointInLineSegment(cp.m_localPointB, v1, v2, nearest);
		btScalar len = (contact - nearest).length();
		//
		if (len < disttobestedge)
		{
			bestedge = 1;
			disttobestedge = len;
		}
	}
	// Edge 2 -> 0
	if (btFabs(info->m_edgeV2V0Angle) < triangleInfoMapPtr->m_maxEdgeAngleThreshold)
	{
		btVector3 nearest;
		btNearestPointInLineSegment(cp.m_localPointB, v2, v0, nearest);
		btScalar len = (contact - nearest).length();
		//
		if (len < disttobestedge)
		{
			bestedge = 2;
			disttobestedge = len;
		}
	}

#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
	btVector3 upfix = tri_normal * btVector3(0.1f, 0.1f, 0.1f);
	btDebugDrawLine(tr * v0 + upfix, tr * v1 + upfix, red);
#endif
	if (btFabs(info->m_edgeV0V1Angle) < triangleInfoMapPtr->m_maxEdgeAngleThreshold)
	{
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
		btDebugDrawLine(tr * contact, tr * (contact + cp.m_normalWorldOnB * 10), black);
#endif
		btScalar len = (contact - nearest).length();
		if (len < triangleInfoMapPtr->m_edgeDistanceThreshold)
			if (bestedge == 0)
			{
				btVector3 edge(v0 - v1);
				isNearEdge = true;

				if (info->m_edgeV0V1Angle == btScalar(0))
				{
					numConcaveEdgeHits++;
				}
				else
				{
					bool isEdgeConvex = (info->m_flags & TRI_INFO_V0V1_CONVEX);
					btScalar swapFactor = isEdgeConvex ? btScalar(1) : btScalar(-1);
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
					btDebugDrawLine(tr * nearest, tr * (nearest + swapFactor * tri_normal * 10), white);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

					btVector3 nA = swapFactor * tri_normal;

					btQuaternion orn(edge, info->m_edgeV0V1Angle);
					btVector3 computedNormalB = quatRotate(orn, tri_normal);
					if (info->m_flags & TRI_INFO_V0V1_SWAP_NORMALB)
						computedNormalB *= -1;
					btVector3 nB = swapFactor * computedNormalB;

					btScalar NdotA = localContactNormalOnB.dot(nA);
					btScalar NdotB = localContactNormalOnB.dot(nB);
					bool backFacingNormal = (NdotA < triangleInfoMapPtr->m_convexEpsilon) && (NdotB < triangleInfoMapPtr->m_convexEpsilon);

#ifdef DEBUG_INTERNAL_EDGE
					{
						btDebugDrawLine(cp.getPositionWorldOnB(), cp.getPositionWorldOnB() + tr.getBasis() * (nB * 20), red);
					}
#endif  //DEBUG_INTERNAL_EDGE

					if (backFacingNormal)
					{
						numConcaveEdgeHits++;
					}
					else
					{
						numConvexEdgeHits++;
						btVector3 clampedLocalNormal;
						bool isClamped = btClampNormal(edge, swapFactor * tri_normal, localContactNormalOnB, info->m_edgeV0V1Angle, clampedLocalNormal);
						if (isClamped)
						{
							if (((normalAdjustFlags & BT_TRIANGLE_CONVEX_DOUBLE_SIDED) != 0) || (clampedLocalNormal.dot(frontFacing * tri_normal) > 0))
							{
								btVector3 newNormal = colObj0Wrap->getWorldTransform().getBasis() * clampedLocalNormal;
								//					cp.m_distance1 = cp.m_distance1 * newNormal.dot(cp.m_normalWorldOnB);
								cp.m_normalWorldOnB = newNormal;
								// Reproject collision point along normal. (what about cp.m_distance1?)
								cp.m_positionWorldOnB = cp.m_positionWorldOnA - cp.m_normalWorldOnB * cp.m_distance1;
								cp.m_localPointB = colObj0Wrap->getWorldTransform().invXform(cp.m_positionWorldOnB);
							}
						}
					}
				}
			}
	}

	btNearestPointInLineSegment(contact, v1, v2, nearest);
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
	btDebugDrawLine(tr * nearest, tr * cp.m_localPointB, green);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
	btDebugDrawLine(tr * v1 + upfix, tr * v2 + upfix, green);
#endif

	if (btFabs(info->m_edgeV1V2Angle) < triangleInfoMapPtr->m_maxEdgeAngleThreshold)
	{
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
		btDebugDrawLine(tr * contact, tr * (contact + cp.m_normalWorldOnB * 10), black);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

		btScalar len = (contact - nearest).length();
		if (len < triangleInfoMapPtr->m_edgeDistanceThreshold)
			if (bestedge == 1)
			{
				isNearEdge = true;
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
				btDebugDrawLine(tr * nearest, tr * (nearest + tri_normal * 10), white);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

				btVector3 edge(v1 - v2);

				isNearEdge = true;

				if (info->m_edgeV1V2Angle == btScalar(0))
				{
					numConcaveEdgeHits++;
				}
				else
				{
					bool isEdgeConvex = (info->m_flags & TRI_INFO_V1V2_CONVEX) != 0;
					btScalar swapFactor = isEdgeConvex ? btScalar(1) : btScalar(-1);
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
					btDebugDrawLine(tr * nearest, tr * (nearest + swapFactor * tri_normal * 10), white);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

					btVector3 nA = swapFactor * tri_normal;

					btQuaternion orn(edge, info->m_edgeV1V2Angle);
					btVector3 computedNormalB = quatRotate(orn, tri_normal);
					if (info->m_flags & TRI_INFO_V1V2_SWAP_NORMALB)
						computedNormalB *= -1;
					btVector3 nB = swapFactor * computedNormalB;

#ifdef DEBUG_INTERNAL_EDGE
					{
						btDebugDrawLine(cp.getPositionWorldOnB(), cp.getPositionWorldOnB() + tr.getBasis() * (nB * 20), red);
					}
#endif  //DEBUG_INTERNAL_EDGE

					btScalar NdotA = localContactNormalOnB.dot(nA);
					btScalar NdotB = localContactNormalOnB.dot(nB);
					bool backFacingNormal = (NdotA < triangleInfoMapPtr->m_convexEpsilon) && (NdotB < triangleInfoMapPtr->m_convexEpsilon);

					if (backFacingNormal)
					{
						numConcaveEdgeHits++;
					}
					else
					{
						numConvexEdgeHits++;
						btVector3 localContactNormalOnB = colObj0Wrap->getWorldTransform().getBasis().transpose() * cp.m_normalWorldOnB;
						btVector3 clampedLocalNormal;
						bool isClamped = btClampNormal(edge, swapFactor * tri_normal, localContactNormalOnB, info->m_edgeV1V2Angle, clampedLocalNormal);
						if (isClamped)
						{
							if (((normalAdjustFlags & BT_TRIANGLE_CONVEX_DOUBLE_SIDED) != 0) || (clampedLocalNormal.dot(frontFacing * tri_normal) > 0))
							{
								btVector3 newNormal = colObj0Wrap->getWorldTransform().getBasis() * clampedLocalNormal;
								//					cp.m_distance1 = cp.m_distance1 * newNormal.dot(cp.m_normalWorldOnB);
								cp.m_normalWorldOnB = newNormal;
								// Reproject collision point along normal.
								cp.m_positionWorldOnB = cp.m_positionWorldOnA - cp.m_normalWorldOnB * cp.m_distance1;
								cp.m_localPointB = colObj0Wrap->getWorldTransform().invXform(cp.m_positionWorldOnB);
							}
						}
					}
				}
			}
	}

	btNearestPointInLineSegment(contact, v2, v0, nearest);
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
	btDebugDrawLine(tr * nearest, tr * cp.m_localPointB, blue);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
	btDebugDrawLine(tr * v2 + upfix, tr * v0 + upfix, blue);
#endif

	if (btFabs(info->m_edgeV2V0Angle) < triangleInfoMapPtr->m_maxEdgeAngleThreshold)
	{
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
		btDebugDrawLine(tr * contact, tr * (contact + cp.m_normalWorldOnB * 10), black);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

		btScalar len = (contact - nearest).length();
		if (len < triangleInfoMapPtr->m_edgeDistanceThreshold)
			if (bestedge == 2)
			{
				isNearEdge = true;
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
				btDebugDrawLine(tr * nearest, tr * (nearest + tri_normal * 10), white);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

				btVector3 edge(v2 - v0);

				if (info->m_edgeV2V0Angle == btScalar(0))
				{
					numConcaveEdgeHits++;
				}
				else
				{
					bool isEdgeConvex = (info->m_flags & TRI_INFO_V2V0_CONVEX) != 0;
					btScalar swapFactor = isEdgeConvex ? btScalar(1) : btScalar(-1);
#ifdef BT_INTERNAL_EDGE_DEBUG_DRAW
					btDebugDrawLine(tr * nearest, tr * (nearest + swapFactor * tri_normal * 10), white);
#endif  //BT_INTERNAL_EDGE_DEBUG_DRAW

					btVector3 nA = swapFactor * tri_normal;
					btQuaternion orn(edge, info->m_edgeV2V0Angle);
					btVector3 computedNormalB = quatRotate(orn, tri_normal);
					if (info->m_flags & TRI_INFO_V2V0_SWAP_NORMALB)
						computedNormalB *= -1;
					btVector3 nB = swapFactor * computedNormalB;

#ifdef DEBUG_INTERNAL_EDGE
					{
						btDebugDrawLine(cp.getPositionWorldOnB(), cp.getPositionWorldOnB() + tr.getBasis() * (nB * 20), red);
					}
#endif  //DEBUG_INTERNAL_EDGE

					btScalar NdotA = localContactNormalOnB.dot(nA);
					btScalar NdotB = localContactNormalOnB.dot(nB);
					bool backFacingNormal = (NdotA < triangleInfoMapPtr->m_convexEpsilon) && (NdotB < triangleInfoMapPtr->m_convexEpsilon);

					if (backFacingNormal)
					{
						numConcaveEdgeHits++;
					}
					else
					{
						numConvexEdgeHits++;
						//				printf("hitting convex edge\n");

						btVector3 localContactNormalOnB = colObj0Wrap->getWorldTransform().getBasis().transpose() * cp.m_normalWorldOnB;
						btVector3 clampedLocalNormal;
						bool isClamped = btClampNormal(edge, swapFactor * tri_normal, localContactNormalOnB, info->m_edgeV2V0Angle, clampedLocalNormal);
						if (isClamped)
						{
							if (((normalAdjustFlags & BT_TRIANGLE_CONVEX_DOUBLE_SIDED) != 0) || (clampedLocalNormal.dot(frontFacing * tri_normal) > 0))
							{
								btVector3 newNormal = colObj0Wrap->getWorldTransform().getBasis() * clampedLocalNormal;
								//					cp.m_distance1 = cp.m_distance1 * newNormal.dot(cp.m_normalWorldOnB);
								cp.m_normalWorldOnB = newNormal;
								// Reproject collision point along normal.
								cp.m_positionWorldOnB = cp.m_positionWorldOnA - cp.m_normalWorldOnB * cp.m_distance1;
								cp.m_localPointB = colObj0Wrap->getWorldTransform().invXform(cp.m_positionWorldOnB);
							}
						}
					}
				}
			}
	}

#ifdef DEBUG_INTERNAL_EDGE
	{
		btVector3 color(0, 1, 1);
		btDebugDrawLine(cp.getPositionWorldOnB(), cp.getPositionWorldOnB() + cp.m_normalWorldOnB * 10, color);
	}
#endif  //DEBUG_INTERNAL_EDGE

	if (isNearEdge)
	{
		if (numConcaveEdgeHits > 0)
		{
			if ((normalAdjustFlags & BT_TRIANGLE_CONCAVE_DOUBLE_SIDED) != 0)
			{
				//fix tri_normal so it pointing the same direction as the current local contact normal
				if (tri_normal.dot(localContactNormalOnB) < 0)
				{
					tri_normal *= -1;
				}
				cp.m_normalWorldOnB = colObj0Wrap->getWorldTransform().getBasis() * tri_normal;
			}
			else
			{
				btVector3 newNormal = tri_normal * frontFacing;
				//if the tri_normal is pointing opposite direction as the current local contact normal, skip it
				btScalar d = newNormal.dot(localContactNormalOnB);
				if (d < 0)
				{
					return;
				}
				//modify the normal to be the triangle normal (or backfacing normal)
				cp.m_normalWorldOnB = colObj0Wrap->getWorldTransform().getBasis() * newNormal;
			}

			// Reproject collision point along normal.
			cp.m_positionWorldOnB = cp.m_positionWorldOnA - cp.m_normalWorldOnB * cp.m_distance1;
			cp.m_localPointB = colObj0Wrap->getWorldTransform().invXform(cp.m_positionWorldOnB);
		}
	}
}
