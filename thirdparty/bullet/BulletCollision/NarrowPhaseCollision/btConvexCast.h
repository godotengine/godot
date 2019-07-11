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

#ifndef BT_CONVEX_CAST_H
#define BT_CONVEX_CAST_H

#include "LinearMath/btTransform.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btScalar.h"
class btMinkowskiSumShape;
#include "LinearMath/btIDebugDraw.h"

#ifdef BT_USE_DOUBLE_PRECISION
#define MAX_CONVEX_CAST_ITERATIONS 64
#define MAX_CONVEX_CAST_EPSILON (SIMD_EPSILON * 10)
#else
#define MAX_CONVEX_CAST_ITERATIONS 32
#define MAX_CONVEX_CAST_EPSILON btScalar(0.0001)
#endif
///Typically the conservative advancement reaches solution in a few iterations, clip it to 32 for degenerate cases.
///See discussion about this here http://continuousphysics.com/Bullet/phpBB2/viewtopic.php?t=565
//will need to digg deeper to make the algorithm more robust
//since, a large epsilon can cause an early termination with false
//positive results (ray intersections that shouldn't be there)

/// btConvexCast is an interface for Casting
class btConvexCast
{
public:
	virtual ~btConvexCast();

	///RayResult stores the closest result
	/// alternatively, add a callback method to decide about closest/all results
	struct CastResult
	{
		//virtual bool	addRayResult(const btVector3& normal,btScalar	fraction) = 0;

		virtual void DebugDraw(btScalar fraction) { (void)fraction; }
		virtual void drawCoordSystem(const btTransform& trans) { (void)trans; }
		virtual void reportFailure(int errNo, int numIterations)
		{
			(void)errNo;
			(void)numIterations;
		}
		CastResult()
			: m_fraction(btScalar(BT_LARGE_FLOAT)),
			  m_debugDrawer(0),
			  m_allowedPenetration(btScalar(0)),
			  m_subSimplexCastMaxIterations(MAX_CONVEX_CAST_ITERATIONS),
			  m_subSimplexCastEpsilon(MAX_CONVEX_CAST_EPSILON)
		{
		}

		virtual ~CastResult(){};

		btTransform m_hitTransformA;
		btTransform m_hitTransformB;
		btVector3 m_normal;
		btVector3 m_hitPoint;
		btScalar m_fraction;  //input and output
		btIDebugDraw* m_debugDrawer;
		btScalar m_allowedPenetration;
		
		int m_subSimplexCastMaxIterations;
		btScalar m_subSimplexCastEpsilon;

	};

	/// cast a convex against another convex object
	virtual bool calcTimeOfImpact(
		const btTransform& fromA,
		const btTransform& toA,
		const btTransform& fromB,
		const btTransform& toB,
		CastResult& result) = 0;
};

#endif  //BT_CONVEX_CAST_H
