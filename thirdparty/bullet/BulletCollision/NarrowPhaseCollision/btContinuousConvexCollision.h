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

#ifndef BT_CONTINUOUS_COLLISION_CONVEX_CAST_H
#define BT_CONTINUOUS_COLLISION_CONVEX_CAST_H

#include "btConvexCast.h"
#include "btSimplexSolverInterface.h"
class btConvexPenetrationDepthSolver;
class btConvexShape;
class btStaticPlaneShape;

/// btContinuousConvexCollision implements angular and linear time of impact for convex objects.
/// Based on Brian Mirtich's Conservative Advancement idea (PhD thesis).
/// Algorithm operates in worldspace, in order to keep in between motion globally consistent.
/// It uses GJK at the moment. Future improvement would use minkowski sum / supporting vertex, merging innerloops
class btContinuousConvexCollision : public btConvexCast
{
	btSimplexSolverInterface* m_simplexSolver;
	btConvexPenetrationDepthSolver* m_penetrationDepthSolver;
	const btConvexShape* m_convexA;
	//second object is either a convex or a plane (code sharing)
	const btConvexShape* m_convexB1;
	const btStaticPlaneShape* m_planeShape;

	void computeClosestPoints(const btTransform& transA, const btTransform& transB, struct btPointCollector& pointCollector);

public:
	btContinuousConvexCollision(const btConvexShape* shapeA, const btConvexShape* shapeB, btSimplexSolverInterface* simplexSolver, btConvexPenetrationDepthSolver* penetrationDepthSolver);

	btContinuousConvexCollision(const btConvexShape* shapeA, const btStaticPlaneShape* plane);

	virtual bool calcTimeOfImpact(
		const btTransform& fromA,
		const btTransform& toA,
		const btTransform& fromB,
		const btTransform& toB,
		CastResult& result);
};

#endif  //BT_CONTINUOUS_COLLISION_CONVEX_CAST_H
