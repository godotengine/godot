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

#ifndef BT_MINKOWSKI_PENETRATION_DEPTH_SOLVER_H
#define BT_MINKOWSKI_PENETRATION_DEPTH_SOLVER_H

#include "btConvexPenetrationDepthSolver.h"

///MinkowskiPenetrationDepthSolver implements bruteforce penetration depth estimation.
///Implementation is based on sampling the depth using support mapping, and using GJK step to get the witness points.
class btMinkowskiPenetrationDepthSolver : public btConvexPenetrationDepthSolver
{
protected:
	static btVector3* getPenetrationDirections();

public:
	virtual bool calcPenDepth(btSimplexSolverInterface& simplexSolver,
							  const btConvexShape* convexA, const btConvexShape* convexB,
							  const btTransform& transA, const btTransform& transB,
							  btVector3& v, btVector3& pa, btVector3& pb,
							  class btIDebugDraw* debugDraw);
};

#endif  //BT_MINKOWSKI_PENETRATION_DEPTH_SOLVER_H
