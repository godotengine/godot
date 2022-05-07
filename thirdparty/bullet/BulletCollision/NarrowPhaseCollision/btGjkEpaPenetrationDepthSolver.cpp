/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

EPA Copyright (c) Ricardo Padrela 2006

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "BulletCollision/CollisionShapes/btConvexShape.h"
#include "btGjkEpaPenetrationDepthSolver.h"

#include "BulletCollision/NarrowPhaseCollision/btGjkEpa2.h"

bool btGjkEpaPenetrationDepthSolver::calcPenDepth(btSimplexSolverInterface& simplexSolver,
												  const btConvexShape* pConvexA, const btConvexShape* pConvexB,
												  const btTransform& transformA, const btTransform& transformB,
												  btVector3& v, btVector3& wWitnessOnA, btVector3& wWitnessOnB,
												  class btIDebugDraw* debugDraw)
{
	(void)debugDraw;
	(void)v;
	(void)simplexSolver;

	btVector3 guessVectors[] = {
		btVector3(transformB.getOrigin() - transformA.getOrigin()).safeNormalize(),
		btVector3(transformA.getOrigin() - transformB.getOrigin()).safeNormalize(),
		btVector3(0, 0, 1),
		btVector3(0, 1, 0),
		btVector3(1, 0, 0),
		btVector3(1, 1, 0),
		btVector3(1, 1, 1),
		btVector3(0, 1, 1),
		btVector3(1, 0, 1),
	};

	int numVectors = sizeof(guessVectors) / sizeof(btVector3);

	for (int i = 0; i < numVectors; i++)
	{
		simplexSolver.reset();
		btVector3 guessVector = guessVectors[i];

		btGjkEpaSolver2::sResults results;

		if (btGjkEpaSolver2::Penetration(pConvexA, transformA,
										 pConvexB, transformB,
										 guessVector, results))

		{
			wWitnessOnA = results.witnesses[0];
			wWitnessOnB = results.witnesses[1];
			v = results.normal;
			return true;
		}
		else
		{
			if (btGjkEpaSolver2::Distance(pConvexA, transformA, pConvexB, transformB, guessVector, results))
			{
				wWitnessOnA = results.witnesses[0];
				wWitnessOnB = results.witnesses[1];
				v = results.normal;
				return false;
			}
		}
	}

	//failed to find a distance/penetration
	wWitnessOnA.setValue(0, 0, 0);
	wWitnessOnB.setValue(0, 0, 0);
	v.setValue(0, 0, 0);
	return false;
}
