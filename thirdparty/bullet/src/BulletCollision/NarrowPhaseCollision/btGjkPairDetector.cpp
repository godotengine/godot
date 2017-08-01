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

#include "btGjkPairDetector.h"
#include "BulletCollision/CollisionShapes/btConvexShape.h"
#include "BulletCollision/NarrowPhaseCollision/btSimplexSolverInterface.h"
#include "BulletCollision/NarrowPhaseCollision/btConvexPenetrationDepthSolver.h"



#if defined(DEBUG) || defined (_DEBUG)
//#define TEST_NON_VIRTUAL 1
#include <stdio.h> //for debug printf
#ifdef __SPU__
#include <spu_printf.h>
#define printf spu_printf
#endif //__SPU__
#endif

//must be above the machine epsilon
#ifdef  BT_USE_DOUBLE_PRECISION
	#define REL_ERROR2 btScalar(1.0e-12)
	btScalar gGjkEpaPenetrationTolerance = 1e-7;
#else
	#define REL_ERROR2 btScalar(1.0e-6)
	btScalar gGjkEpaPenetrationTolerance = 0.001;
#endif

//temp globals, to improve GJK/EPA/penetration calculations
int gNumDeepPenetrationChecks = 0;
int gNumGjkChecks = 0;


btGjkPairDetector::btGjkPairDetector(const btConvexShape* objectA,const btConvexShape* objectB,btSimplexSolverInterface* simplexSolver,btConvexPenetrationDepthSolver*	penetrationDepthSolver)
:m_cachedSeparatingAxis(btScalar(0.),btScalar(1.),btScalar(0.)),
m_penetrationDepthSolver(penetrationDepthSolver),
m_simplexSolver(simplexSolver),
m_minkowskiA(objectA),
m_minkowskiB(objectB),
m_shapeTypeA(objectA->getShapeType()),
m_shapeTypeB(objectB->getShapeType()),
m_marginA(objectA->getMargin()),
m_marginB(objectB->getMargin()),
m_ignoreMargin(false),
m_lastUsedMethod(-1),
m_catchDegeneracies(1),
m_fixContactNormalDirection(1)
{
}
btGjkPairDetector::btGjkPairDetector(const btConvexShape* objectA,const btConvexShape* objectB,int shapeTypeA,int shapeTypeB,btScalar marginA, btScalar marginB, btSimplexSolverInterface* simplexSolver,btConvexPenetrationDepthSolver*	penetrationDepthSolver)
:m_cachedSeparatingAxis(btScalar(0.),btScalar(1.),btScalar(0.)),
m_penetrationDepthSolver(penetrationDepthSolver),
m_simplexSolver(simplexSolver),
m_minkowskiA(objectA),
m_minkowskiB(objectB),
m_shapeTypeA(shapeTypeA),
m_shapeTypeB(shapeTypeB),
m_marginA(marginA),
m_marginB(marginB),
m_ignoreMargin(false),
m_lastUsedMethod(-1),
m_catchDegeneracies(1),
m_fixContactNormalDirection(1)
{
}

void	btGjkPairDetector::getClosestPoints(const ClosestPointInput& input,Result& output,class btIDebugDraw* debugDraw,bool swapResults)
{
	(void)swapResults;

	getClosestPointsNonVirtual(input,output,debugDraw);
}

#ifdef __SPU__
void btGjkPairDetector::getClosestPointsNonVirtual(const ClosestPointInput& input,Result& output,class btIDebugDraw* debugDraw)
#else
void btGjkPairDetector::getClosestPointsNonVirtual(const ClosestPointInput& input, Result& output, class btIDebugDraw* debugDraw)
#endif
{
	m_cachedSeparatingDistance = 0.f;

	btScalar distance=btScalar(0.);
	btVector3	normalInB(btScalar(0.),btScalar(0.),btScalar(0.));

	btVector3 pointOnA,pointOnB;
	btTransform	localTransA = input.m_transformA;
	btTransform localTransB = input.m_transformB;
	btVector3 positionOffset=(localTransA.getOrigin() + localTransB.getOrigin()) * btScalar(0.5);
	localTransA.getOrigin() -= positionOffset;
	localTransB.getOrigin() -= positionOffset;

	bool check2d = m_minkowskiA->isConvex2d() && m_minkowskiB->isConvex2d();

	btScalar marginA = m_marginA;
	btScalar marginB = m_marginB;

	gNumGjkChecks++;

	//for CCD we don't use margins
	if (m_ignoreMargin)
	{
		marginA = btScalar(0.);
		marginB = btScalar(0.);
	}

	m_curIter = 0;
	int gGjkMaxIter = 1000;//this is to catch invalid input, perhaps check for #NaN?
	m_cachedSeparatingAxis.setValue(0,1,0);

	bool isValid = false;
	bool checkSimplex = false;
	bool checkPenetration = true;
	m_degenerateSimplex = 0;

	m_lastUsedMethod = -1;

	{
		btScalar squaredDistance = BT_LARGE_FLOAT;
		btScalar delta = btScalar(0.);
		
		btScalar margin = marginA + marginB;
		
		

		m_simplexSolver->reset();
		
		for ( ; ; )
		//while (true)
		{

			btVector3 seperatingAxisInA = (-m_cachedSeparatingAxis)* input.m_transformA.getBasis();
			btVector3 seperatingAxisInB = m_cachedSeparatingAxis* input.m_transformB.getBasis();


			btVector3 pInA = m_minkowskiA->localGetSupportVertexWithoutMarginNonVirtual(seperatingAxisInA);
			btVector3 qInB = m_minkowskiB->localGetSupportVertexWithoutMarginNonVirtual(seperatingAxisInB);

			btVector3  pWorld = localTransA(pInA);	
			btVector3  qWorld = localTransB(qInB);


			if (check2d)
			{
				pWorld[2] = 0.f;
				qWorld[2] = 0.f;
			}

			btVector3 w	= pWorld - qWorld;
			delta = m_cachedSeparatingAxis.dot(w);

			// potential exit, they don't overlap
			if ((delta > btScalar(0.0)) && (delta * delta > squaredDistance * input.m_maximumDistanceSquared)) 
			{
				m_degenerateSimplex = 10;
				checkSimplex=true;
				//checkPenetration = false;
				break;
			}

			//exit 0: the new point is already in the simplex, or we didn't come any closer
			if (m_simplexSolver->inSimplex(w))
			{
				m_degenerateSimplex = 1;
				checkSimplex = true;
				break;
			}
			// are we getting any closer ?
			btScalar f0 = squaredDistance - delta;
			btScalar f1 = squaredDistance * REL_ERROR2;

			if (f0 <= f1)
			{
				if (f0 <= btScalar(0.))
				{
					m_degenerateSimplex = 2;
				} else
				{
					m_degenerateSimplex = 11;
				}
				checkSimplex = true;
				break;
			}

			//add current vertex to simplex
			m_simplexSolver->addVertex(w, pWorld, qWorld);
			btVector3 newCachedSeparatingAxis;

			//calculate the closest point to the origin (update vector v)
			if (!m_simplexSolver->closest(newCachedSeparatingAxis))
			{
				m_degenerateSimplex = 3;
				checkSimplex = true;
				break;
			}

			if(newCachedSeparatingAxis.length2()<REL_ERROR2)
            {
				m_cachedSeparatingAxis = newCachedSeparatingAxis;
                m_degenerateSimplex = 6;
                checkSimplex = true;
                break;
            }

			btScalar previousSquaredDistance = squaredDistance;
			squaredDistance = newCachedSeparatingAxis.length2();
#if 0
///warning: this termination condition leads to some problems in 2d test case see Bullet/Demos/Box2dDemo
			if (squaredDistance>previousSquaredDistance)
			{
				m_degenerateSimplex = 7;
				squaredDistance = previousSquaredDistance;
                checkSimplex = false;
                break;
			}
#endif //
			

			//redundant m_simplexSolver->compute_points(pointOnA, pointOnB);

			//are we getting any closer ?
			if (previousSquaredDistance - squaredDistance <= SIMD_EPSILON * previousSquaredDistance) 
			{ 
//				m_simplexSolver->backup_closest(m_cachedSeparatingAxis);
				checkSimplex = true;
				m_degenerateSimplex = 12;
				
				break;
			}

			m_cachedSeparatingAxis = newCachedSeparatingAxis;

			  //degeneracy, this is typically due to invalid/uninitialized worldtransforms for a btCollisionObject   
              if (m_curIter++ > gGjkMaxIter)   
              {   
                      #if defined(DEBUG) || defined (_DEBUG)

                              printf("btGjkPairDetector maxIter exceeded:%i\n",m_curIter);   
                              printf("sepAxis=(%f,%f,%f), squaredDistance = %f, shapeTypeA=%i,shapeTypeB=%i\n",   
                              m_cachedSeparatingAxis.getX(),   
                              m_cachedSeparatingAxis.getY(),   
                              m_cachedSeparatingAxis.getZ(),   
                              squaredDistance,   
                              m_minkowskiA->getShapeType(),   
                              m_minkowskiB->getShapeType());   

                      #endif   
                      break;   

              } 


			bool check = (!m_simplexSolver->fullSimplex());
			//bool check = (!m_simplexSolver->fullSimplex() && squaredDistance > SIMD_EPSILON * m_simplexSolver->maxVertex());

			if (!check)
			{
				//do we need this backup_closest here ?
//				m_simplexSolver->backup_closest(m_cachedSeparatingAxis);
				m_degenerateSimplex = 13;
				break;
			}
		}

		if (checkSimplex)
		{
			m_simplexSolver->compute_points(pointOnA, pointOnB);
			normalInB = m_cachedSeparatingAxis;

			btScalar lenSqr =m_cachedSeparatingAxis.length2();
			
			//valid normal
			if (lenSqr < REL_ERROR2)
			{
				m_degenerateSimplex = 5;
			} 
			if (lenSqr > SIMD_EPSILON*SIMD_EPSILON)
			{
				btScalar rlen = btScalar(1.) / btSqrt(lenSqr );
				normalInB *= rlen; //normalize

				btScalar s = btSqrt(squaredDistance);
			
				btAssert(s > btScalar(0.0));
				pointOnA -= m_cachedSeparatingAxis * (marginA / s);
				pointOnB += m_cachedSeparatingAxis * (marginB / s);
				distance = ((btScalar(1.)/rlen) - margin);
				isValid = true;
				
				m_lastUsedMethod = 1;
			} else
			{
				m_lastUsedMethod = 2;
			}
		}

		bool catchDegeneratePenetrationCase = 
			(m_catchDegeneracies && m_penetrationDepthSolver && m_degenerateSimplex && ((distance+margin) < gGjkEpaPenetrationTolerance));

		//if (checkPenetration && !isValid)
		if (checkPenetration && (!isValid || catchDegeneratePenetrationCase ))
		{
			//penetration case

			//if there is no way to handle penetrations, bail out
			if (m_penetrationDepthSolver)
			{
				// Penetration depth case.
				btVector3 tmpPointOnA,tmpPointOnB;
				
				gNumDeepPenetrationChecks++;
				m_cachedSeparatingAxis.setZero();

				bool isValid2 = m_penetrationDepthSolver->calcPenDepth( 
					*m_simplexSolver, 
					m_minkowskiA,m_minkowskiB,
					localTransA,localTransB,
					m_cachedSeparatingAxis, tmpPointOnA, tmpPointOnB,
					debugDraw
					);


				if (isValid2)
				{
					btVector3 tmpNormalInB = tmpPointOnB-tmpPointOnA;
					btScalar lenSqr = tmpNormalInB.length2();
					if (lenSqr <= (SIMD_EPSILON*SIMD_EPSILON))
					{
						tmpNormalInB = m_cachedSeparatingAxis;
						lenSqr = m_cachedSeparatingAxis.length2();
					}

					if (lenSqr > (SIMD_EPSILON*SIMD_EPSILON))
					{
						tmpNormalInB /= btSqrt(lenSqr);
						btScalar distance2 = -(tmpPointOnA-tmpPointOnB).length();
						m_lastUsedMethod = 3;
						//only replace valid penetrations when the result is deeper (check)
						if (!isValid || (distance2 < distance))
						{
							distance = distance2;
							pointOnA = tmpPointOnA;
							pointOnB = tmpPointOnB;
							normalInB = tmpNormalInB;
							
							isValid = true;
							
						} else
						{
							m_lastUsedMethod = 8;
						}
					} else
					{
						m_lastUsedMethod = 9;
					}
				} else

				{
					///this is another degenerate case, where the initial GJK calculation reports a degenerate case
					///EPA reports no penetration, and the second GJK (using the supporting vector without margin)
					///reports a valid positive distance. Use the results of the second GJK instead of failing.
					///thanks to Jacob.Langford for the reproduction case
					///http://code.google.com/p/bullet/issues/detail?id=250

				
					if (m_cachedSeparatingAxis.length2() > btScalar(0.))
					{
						btScalar distance2 = (tmpPointOnA-tmpPointOnB).length()-margin;
						//only replace valid distances when the distance is less
						if (!isValid || (distance2 < distance))
						{
							distance = distance2;
							pointOnA = tmpPointOnA;
							pointOnB = tmpPointOnB;
							pointOnA -= m_cachedSeparatingAxis * marginA ;
							pointOnB += m_cachedSeparatingAxis * marginB ;
							normalInB = m_cachedSeparatingAxis;
							normalInB.normalize();

							isValid = true;
							m_lastUsedMethod = 6;
						} else
						{
							m_lastUsedMethod = 5;
						}
					}
				}
				
			}

		}
	}

	

	if (isValid && ((distance < 0) || (distance*distance < input.m_maximumDistanceSquared)))
	{

		m_cachedSeparatingAxis = normalInB;
		m_cachedSeparatingDistance = distance;

		{
		///todo: need to track down this EPA penetration solver degeneracy
		///the penetration solver reports penetration but the contact normal
		///connecting the contact points is pointing in the opposite direction
		///until then, detect the issue and revert the normal

			btScalar d1=0;
			{
				btVector3 seperatingAxisInA = (normalInB)* input.m_transformA.getBasis();
				btVector3 seperatingAxisInB = -normalInB* input.m_transformB.getBasis();
			

				btVector3 pInA = m_minkowskiA->localGetSupportVertexWithoutMarginNonVirtual(seperatingAxisInA);
				btVector3 qInB = m_minkowskiB->localGetSupportVertexWithoutMarginNonVirtual(seperatingAxisInB);

				btVector3  pWorld = localTransA(pInA);	
				btVector3  qWorld = localTransB(qInB);
				btVector3 w	= pWorld - qWorld;
				d1 = (-normalInB).dot(w);
			}
			btScalar d0 = 0.f;
			{
				btVector3 seperatingAxisInA = (-normalInB)* input.m_transformA.getBasis();
				btVector3 seperatingAxisInB = normalInB* input.m_transformB.getBasis();
			

				btVector3 pInA = m_minkowskiA->localGetSupportVertexWithoutMarginNonVirtual(seperatingAxisInA);
				btVector3 qInB = m_minkowskiB->localGetSupportVertexWithoutMarginNonVirtual(seperatingAxisInB);

				btVector3  pWorld = localTransA(pInA);	
				btVector3  qWorld = localTransB(qInB);
				btVector3 w	= pWorld - qWorld;
				d0 = normalInB.dot(w);
			}
			if (d1>d0)
			{
				m_lastUsedMethod = 10;
				normalInB*=-1;
			} 

		}
		output.addContactPoint(
			normalInB,
			pointOnB+positionOffset,
			distance);

	}


}





