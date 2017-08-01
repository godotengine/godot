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



#include "btGjkConvexCast.h"
#include "BulletCollision/CollisionShapes/btSphereShape.h"
#include "btGjkPairDetector.h"
#include "btPointCollector.h"
#include "LinearMath/btTransformUtil.h"

#ifdef BT_USE_DOUBLE_PRECISION
#define MAX_ITERATIONS 64
#else
#define MAX_ITERATIONS 32
#endif

btGjkConvexCast::btGjkConvexCast(const btConvexShape* convexA,const btConvexShape* convexB,btSimplexSolverInterface* simplexSolver)
:m_simplexSolver(simplexSolver),
m_convexA(convexA),
m_convexB(convexB)
{
}

bool	btGjkConvexCast::calcTimeOfImpact(
					const btTransform& fromA,
					const btTransform& toA,
					const btTransform& fromB,
					const btTransform& toB,
					CastResult& result)
{


	m_simplexSolver->reset();

	/// compute linear velocity for this interval, to interpolate
	//assume no rotation/angular velocity, assert here?
	btVector3 linVelA,linVelB;
	linVelA = toA.getOrigin()-fromA.getOrigin();
	linVelB = toB.getOrigin()-fromB.getOrigin();

	btScalar radius = btScalar(0.001);
	btScalar lambda = btScalar(0.);
	btVector3 v(1,0,0);

	int maxIter = MAX_ITERATIONS;

	btVector3 n;
	n.setValue(btScalar(0.),btScalar(0.),btScalar(0.));
	bool hasResult = false;
	btVector3 c;
	btVector3 r = (linVelA-linVelB);

	btScalar lastLambda = lambda;
	//btScalar epsilon = btScalar(0.001);

	int numIter = 0;
	//first solution, using GJK


	btTransform identityTrans;
	identityTrans.setIdentity();


//	result.drawCoordSystem(sphereTr);

	btPointCollector	pointCollector;

		
	btGjkPairDetector gjk(m_convexA,m_convexB,m_simplexSolver,0);//m_penetrationDepthSolver);		
	btGjkPairDetector::ClosestPointInput input;

	//we don't use margins during CCD
	//	gjk.setIgnoreMargin(true);

	input.m_transformA = fromA;
	input.m_transformB = fromB;
	gjk.getClosestPoints(input,pointCollector,0);

	hasResult = pointCollector.m_hasResult;
	c = pointCollector.m_pointInWorld;

	if (hasResult)
	{
		btScalar dist;
		dist = pointCollector.m_distance;
		n = pointCollector.m_normalOnBInWorld;

	

		//not close enough
		while (dist > radius)
		{
			numIter++;
			if (numIter > maxIter)
			{
				return false; //todo: report a failure
			}
			btScalar dLambda = btScalar(0.);

			btScalar projectedLinearVelocity = r.dot(n);
			
			dLambda = dist / (projectedLinearVelocity);

			lambda = lambda - dLambda;

			if (lambda > btScalar(1.))
				return false;

			if (lambda < btScalar(0.))
				return false;

			//todo: next check with relative epsilon
			if (lambda <= lastLambda)
			{
				return false;
				//n.setValue(0,0,0);
				break;
			}
			lastLambda = lambda;

			//interpolate to next lambda
			result.DebugDraw( lambda );
			input.m_transformA.getOrigin().setInterpolate3(fromA.getOrigin(),toA.getOrigin(),lambda);
			input.m_transformB.getOrigin().setInterpolate3(fromB.getOrigin(),toB.getOrigin(),lambda);
			
			gjk.getClosestPoints(input,pointCollector,0);
			if (pointCollector.m_hasResult)
			{
				if (pointCollector.m_distance < btScalar(0.))
				{
					result.m_fraction = lastLambda;
					n = pointCollector.m_normalOnBInWorld;
					result.m_normal=n;
					result.m_hitPoint = pointCollector.m_pointInWorld;
					return true;
				}
				c = pointCollector.m_pointInWorld;		
				n = pointCollector.m_normalOnBInWorld;
				dist = pointCollector.m_distance;
			} else
			{
				//??
				return false;
			}

		}

		//is n normalized?
		//don't report time of impact for motion away from the contact normal (or causes minor penetration)
		if (n.dot(r)>=-result.m_allowedPenetration)
			return false;

		result.m_fraction = lambda;
		result.m_normal = n;
		result.m_hitPoint = c;
		return true;
	}

	return false;


}

