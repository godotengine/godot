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




#ifndef BT_GJK_PAIR_DETECTOR_H
#define BT_GJK_PAIR_DETECTOR_H

#include "btDiscreteCollisionDetectorInterface.h"
#include "BulletCollision/CollisionShapes/btCollisionMargin.h"

class btConvexShape;
#include "btSimplexSolverInterface.h"
class btConvexPenetrationDepthSolver;

/// btGjkPairDetector uses GJK to implement the btDiscreteCollisionDetectorInterface
class btGjkPairDetector : public btDiscreteCollisionDetectorInterface
{
	

	btVector3	m_cachedSeparatingAxis;
	btConvexPenetrationDepthSolver*	m_penetrationDepthSolver;
	btSimplexSolverInterface* m_simplexSolver;
	const btConvexShape* m_minkowskiA;
	const btConvexShape* m_minkowskiB;
	int	m_shapeTypeA;
	int m_shapeTypeB;
	btScalar	m_marginA;
	btScalar	m_marginB;

	bool		m_ignoreMargin;
	btScalar	m_cachedSeparatingDistance;
	

public:

	//some debugging to fix degeneracy problems
	int			m_lastUsedMethod;
	int			m_curIter;
	int			m_degenerateSimplex;
	int			m_catchDegeneracies;
	int			m_fixContactNormalDirection;

	btGjkPairDetector(const btConvexShape* objectA,const btConvexShape* objectB,btSimplexSolverInterface* simplexSolver,btConvexPenetrationDepthSolver*	penetrationDepthSolver);
	btGjkPairDetector(const btConvexShape* objectA,const btConvexShape* objectB,int shapeTypeA,int shapeTypeB,btScalar marginA, btScalar marginB, btSimplexSolverInterface* simplexSolver,btConvexPenetrationDepthSolver*	penetrationDepthSolver);
	virtual ~btGjkPairDetector() {};

	virtual void	getClosestPoints(const ClosestPointInput& input,Result& output,class btIDebugDraw* debugDraw,bool swapResults=false);

	void	getClosestPointsNonVirtual(const ClosestPointInput& input,Result& output,class btIDebugDraw* debugDraw);
	

	void setMinkowskiA(const btConvexShape* minkA)
	{
		m_minkowskiA = minkA;
	}

	void setMinkowskiB(const btConvexShape* minkB)
	{
		m_minkowskiB = minkB;
	}
	void setCachedSeperatingAxis(const btVector3& seperatingAxis)
	{
		m_cachedSeparatingAxis = seperatingAxis;
	}

	const btVector3& getCachedSeparatingAxis() const
	{
		return m_cachedSeparatingAxis;
	}
	btScalar	getCachedSeparatingDistance() const
	{
		return m_cachedSeparatingDistance;
	}

	void	setPenetrationDepthSolver(btConvexPenetrationDepthSolver*	penetrationDepthSolver)
	{
		m_penetrationDepthSolver = penetrationDepthSolver;
	}

	///don't use setIgnoreMargin, it's for Bullet's internal use
	void	setIgnoreMargin(bool ignoreMargin)
	{
		m_ignoreMargin = ignoreMargin;
	}


};

#endif //BT_GJK_PAIR_DETECTOR_H
