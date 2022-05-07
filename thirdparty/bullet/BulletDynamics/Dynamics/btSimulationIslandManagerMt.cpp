/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "LinearMath/btScalar.h"
#include "LinearMath/btThreads.h"
#include "btSimulationIslandManagerMt.h"
#include "BulletCollision/BroadphaseCollision/btDispatcher.h"
#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionDispatch/btCollisionWorld.h"
#include "BulletDynamics/ConstraintSolver/btTypedConstraint.h"
#include "BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolverMt.h"  // for s_minimumContactManifoldsForBatching

//#include <stdio.h>
#include "LinearMath/btQuickprof.h"

SIMD_FORCE_INLINE int calcBatchCost(int bodies, int manifolds, int constraints)
{
	// rough estimate of the cost of a batch, used for merging
	int batchCost = bodies + 8 * manifolds + 4 * constraints;
	return batchCost;
}

SIMD_FORCE_INLINE int calcBatchCost(const btSimulationIslandManagerMt::Island* island)
{
	return calcBatchCost(island->bodyArray.size(), island->manifoldArray.size(), island->constraintArray.size());
}

btSimulationIslandManagerMt::btSimulationIslandManagerMt()
{
	m_minimumSolverBatchSize = calcBatchCost(0, 128, 0);
	m_batchIslandMinBodyCount = 32;
	m_islandDispatch = parallelIslandDispatch;
	m_batchIsland = NULL;
}

btSimulationIslandManagerMt::~btSimulationIslandManagerMt()
{
	for (int i = 0; i < m_allocatedIslands.size(); ++i)
	{
		delete m_allocatedIslands[i];
	}
	m_allocatedIslands.resize(0);
	m_activeIslands.resize(0);
	m_freeIslands.resize(0);
}

inline int getIslandId(const btPersistentManifold* lhs)
{
	const btCollisionObject* rcolObj0 = static_cast<const btCollisionObject*>(lhs->getBody0());
	const btCollisionObject* rcolObj1 = static_cast<const btCollisionObject*>(lhs->getBody1());
	int islandId = rcolObj0->getIslandTag() >= 0 ? rcolObj0->getIslandTag() : rcolObj1->getIslandTag();
	return islandId;
}

SIMD_FORCE_INLINE int btGetConstraintIslandId1(const btTypedConstraint* lhs)
{
	const btCollisionObject& rcolObj0 = lhs->getRigidBodyA();
	const btCollisionObject& rcolObj1 = lhs->getRigidBodyB();
	int islandId = rcolObj0.getIslandTag() >= 0 ? rcolObj0.getIslandTag() : rcolObj1.getIslandTag();
	return islandId;
}

/// function object that routes calls to operator<
class IslandBatchSizeSortPredicate
{
public:
	bool operator()(const btSimulationIslandManagerMt::Island* lhs, const btSimulationIslandManagerMt::Island* rhs) const
	{
		int lCost = calcBatchCost(lhs);
		int rCost = calcBatchCost(rhs);
		return lCost > rCost;
	}
};

class IslandBodyCapacitySortPredicate
{
public:
	bool operator()(const btSimulationIslandManagerMt::Island* lhs, const btSimulationIslandManagerMt::Island* rhs) const
	{
		return lhs->bodyArray.capacity() > rhs->bodyArray.capacity();
	}
};

void btSimulationIslandManagerMt::Island::append(const Island& other)
{
	// append bodies
	for (int i = 0; i < other.bodyArray.size(); ++i)
	{
		bodyArray.push_back(other.bodyArray[i]);
	}
	// append manifolds
	for (int i = 0; i < other.manifoldArray.size(); ++i)
	{
		manifoldArray.push_back(other.manifoldArray[i]);
	}
	// append constraints
	for (int i = 0; i < other.constraintArray.size(); ++i)
	{
		constraintArray.push_back(other.constraintArray[i]);
	}
}

bool btIsBodyInIsland(const btSimulationIslandManagerMt::Island& island, const btCollisionObject* obj)
{
	for (int i = 0; i < island.bodyArray.size(); ++i)
	{
		if (island.bodyArray[i] == obj)
		{
			return true;
		}
	}
	return false;
}

void btSimulationIslandManagerMt::initIslandPools()
{
	// reset island pools
	int numElem = getUnionFind().getNumElements();
	m_lookupIslandFromId.resize(numElem);
	for (int i = 0; i < m_lookupIslandFromId.size(); ++i)
	{
		m_lookupIslandFromId[i] = NULL;
	}
	m_activeIslands.resize(0);
	m_freeIslands.resize(0);
	// check whether allocated islands are sorted by body capacity (largest to smallest)
	int lastCapacity = 0;
	bool isSorted = true;
	for (int i = 0; i < m_allocatedIslands.size(); ++i)
	{
		Island* island = m_allocatedIslands[i];
		int cap = island->bodyArray.capacity();
		if (cap > lastCapacity)
		{
			isSorted = false;
			break;
		}
		lastCapacity = cap;
	}
	if (!isSorted)
	{
		m_allocatedIslands.quickSort(IslandBodyCapacitySortPredicate());
	}

	m_batchIsland = NULL;
	// mark all islands free (but avoid deallocation)
	for (int i = 0; i < m_allocatedIslands.size(); ++i)
	{
		Island* island = m_allocatedIslands[i];
		island->bodyArray.resize(0);
		island->manifoldArray.resize(0);
		island->constraintArray.resize(0);
		island->id = -1;
		island->isSleeping = true;
		m_freeIslands.push_back(island);
	}
}

btSimulationIslandManagerMt::Island* btSimulationIslandManagerMt::getIsland(int id)
{
	btAssert(id >= 0);
	btAssert(id < m_lookupIslandFromId.size());
	Island* island = m_lookupIslandFromId[id];
	if (island == NULL)
	{
		// search for existing island
		for (int i = 0; i < m_activeIslands.size(); ++i)
		{
			if (m_activeIslands[i]->id == id)
			{
				island = m_activeIslands[i];
				break;
			}
		}
		m_lookupIslandFromId[id] = island;
	}
	return island;
}

btSimulationIslandManagerMt::Island* btSimulationIslandManagerMt::allocateIsland(int id, int numBodies)
{
	Island* island = NULL;
	int allocSize = numBodies;
	if (numBodies < m_batchIslandMinBodyCount)
	{
		if (m_batchIsland)
		{
			island = m_batchIsland;
			m_lookupIslandFromId[id] = island;
			// if we've made a large enough batch,
			if (island->bodyArray.size() + numBodies >= m_batchIslandMinBodyCount)
			{
				// next time start a new batch
				m_batchIsland = NULL;
			}
			return island;
		}
		else
		{
			// need to allocate a batch island
			allocSize = m_batchIslandMinBodyCount * 2;
		}
	}
	btAlignedObjectArray<Island*>& freeIslands = m_freeIslands;

	// search for free island
	if (freeIslands.size() > 0)
	{
		// try to reuse a previously allocated island
		int iFound = freeIslands.size();
		// linear search for smallest island that can hold our bodies
		for (int i = freeIslands.size() - 1; i >= 0; --i)
		{
			if (freeIslands[i]->bodyArray.capacity() >= allocSize)
			{
				iFound = i;
				island = freeIslands[i];
				island->id = id;
				break;
			}
		}
		// if found, shrink array while maintaining ordering
		if (island)
		{
			int iDest = iFound;
			int iSrc = iDest + 1;
			while (iSrc < freeIslands.size())
			{
				freeIslands[iDest++] = freeIslands[iSrc++];
			}
			freeIslands.pop_back();
		}
	}
	if (island == NULL)
	{
		// no free island found, allocate
		island = new Island();  // TODO: change this to use the pool allocator
		island->id = id;
		island->bodyArray.reserve(allocSize);
		m_allocatedIslands.push_back(island);
	}
	m_lookupIslandFromId[id] = island;
	if (numBodies < m_batchIslandMinBodyCount)
	{
		m_batchIsland = island;
	}
	m_activeIslands.push_back(island);
	return island;
}

void btSimulationIslandManagerMt::buildIslands(btDispatcher* dispatcher, btCollisionWorld* collisionWorld)
{
	BT_PROFILE("buildIslands");

	btCollisionObjectArray& collisionObjects = collisionWorld->getCollisionObjectArray();

	//we are going to sort the unionfind array, and store the element id in the size
	//afterwards, we clean unionfind, to make sure no-one uses it anymore

	getUnionFind().sortIslands();
	int numElem = getUnionFind().getNumElements();

	int endIslandIndex = 1;
	int startIslandIndex;

	//update the sleeping state for bodies, if all are sleeping
	for (startIslandIndex = 0; startIslandIndex < numElem; startIslandIndex = endIslandIndex)
	{
		int islandId = getUnionFind().getElement(startIslandIndex).m_id;
		for (endIslandIndex = startIslandIndex + 1; (endIslandIndex < numElem) && (getUnionFind().getElement(endIslandIndex).m_id == islandId); endIslandIndex++)
		{
		}

		//int numSleeping = 0;

		bool allSleeping = true;

		int idx;
		for (idx = startIslandIndex; idx < endIslandIndex; idx++)
		{
			int i = getUnionFind().getElement(idx).m_sz;

			btCollisionObject* colObj0 = collisionObjects[i];
			if ((colObj0->getIslandTag() != islandId) && (colObj0->getIslandTag() != -1))
			{
				//				printf("error in island management\n");
			}

			btAssert((colObj0->getIslandTag() == islandId) || (colObj0->getIslandTag() == -1));
			if (colObj0->getIslandTag() == islandId)
			{
				if (colObj0->getActivationState() == ACTIVE_TAG ||
					colObj0->getActivationState() == DISABLE_DEACTIVATION)
				{
					allSleeping = false;
					break;
				}
			}
		}

		if (allSleeping)
		{
			int idx;
			for (idx = startIslandIndex; idx < endIslandIndex; idx++)
			{
				int i = getUnionFind().getElement(idx).m_sz;
				btCollisionObject* colObj0 = collisionObjects[i];
				if ((colObj0->getIslandTag() != islandId) && (colObj0->getIslandTag() != -1))
				{
					//					printf("error in island management\n");
				}

				btAssert((colObj0->getIslandTag() == islandId) || (colObj0->getIslandTag() == -1));

				if (colObj0->getIslandTag() == islandId)
				{
					colObj0->setActivationState(ISLAND_SLEEPING);
				}
			}
		}
		else
		{
			int idx;
			for (idx = startIslandIndex; idx < endIslandIndex; idx++)
			{
				int i = getUnionFind().getElement(idx).m_sz;

				btCollisionObject* colObj0 = collisionObjects[i];
				if ((colObj0->getIslandTag() != islandId) && (colObj0->getIslandTag() != -1))
				{
					//					printf("error in island management\n");
				}

				btAssert((colObj0->getIslandTag() == islandId) || (colObj0->getIslandTag() == -1));

				if (colObj0->getIslandTag() == islandId)
				{
					if (colObj0->getActivationState() == ISLAND_SLEEPING)
					{
						colObj0->setActivationState(WANTS_DEACTIVATION);
						colObj0->setDeactivationTime(0.f);
					}
				}
			}
		}
	}
}

void btSimulationIslandManagerMt::addBodiesToIslands(btCollisionWorld* collisionWorld)
{
	btCollisionObjectArray& collisionObjects = collisionWorld->getCollisionObjectArray();
	int endIslandIndex = 1;
	int startIslandIndex;
	int numElem = getUnionFind().getNumElements();

	// create explicit islands and add bodies to each
	for (startIslandIndex = 0; startIslandIndex < numElem; startIslandIndex = endIslandIndex)
	{
		int islandId = getUnionFind().getElement(startIslandIndex).m_id;

		// find end index
		for (endIslandIndex = startIslandIndex; (endIslandIndex < numElem) && (getUnionFind().getElement(endIslandIndex).m_id == islandId); endIslandIndex++)
		{
		}
		// check if island is sleeping
		bool islandSleeping = true;
		for (int iElem = startIslandIndex; iElem < endIslandIndex; iElem++)
		{
			int i = getUnionFind().getElement(iElem).m_sz;
			btCollisionObject* colObj = collisionObjects[i];
			if (colObj->isActive())
			{
				islandSleeping = false;
			}
		}
		if (!islandSleeping)
		{
			// want to count the number of bodies before allocating the island to optimize memory usage of the Island structures
			int numBodies = endIslandIndex - startIslandIndex;
			Island* island = allocateIsland(islandId, numBodies);
			island->isSleeping = false;

			// add bodies to island
			for (int iElem = startIslandIndex; iElem < endIslandIndex; iElem++)
			{
				int i = getUnionFind().getElement(iElem).m_sz;
				btCollisionObject* colObj = collisionObjects[i];
				island->bodyArray.push_back(colObj);
			}
		}
	}
}

void btSimulationIslandManagerMt::addManifoldsToIslands(btDispatcher* dispatcher)
{
	// walk all the manifolds, activating bodies touched by kinematic objects, and add each manifold to its Island
	int maxNumManifolds = dispatcher->getNumManifolds();
	for (int i = 0; i < maxNumManifolds; i++)
	{
		btPersistentManifold* manifold = dispatcher->getManifoldByIndexInternal(i);

		const btCollisionObject* colObj0 = static_cast<const btCollisionObject*>(manifold->getBody0());
		const btCollisionObject* colObj1 = static_cast<const btCollisionObject*>(manifold->getBody1());

		///@todo: check sleeping conditions!
		if (((colObj0) && colObj0->getActivationState() != ISLAND_SLEEPING) ||
			((colObj1) && colObj1->getActivationState() != ISLAND_SLEEPING))
		{
			//kinematic objects don't merge islands, but wake up all connected objects
			if (colObj0->isKinematicObject() && colObj0->getActivationState() != ISLAND_SLEEPING)
			{
				if (colObj0->hasContactResponse())
					colObj1->activate();
			}
			if (colObj1->isKinematicObject() && colObj1->getActivationState() != ISLAND_SLEEPING)
			{
				if (colObj1->hasContactResponse())
					colObj0->activate();
			}
			//filtering for response
			if (dispatcher->needsResponse(colObj0, colObj1))
			{
				// scatter manifolds into various islands
				int islandId = getIslandId(manifold);
				// if island not sleeping,
				if (Island* island = getIsland(islandId))
				{
					island->manifoldArray.push_back(manifold);
				}
			}
		}
	}
}

void btSimulationIslandManagerMt::addConstraintsToIslands(btAlignedObjectArray<btTypedConstraint*>& constraints)
{
	// walk constraints
	for (int i = 0; i < constraints.size(); i++)
	{
		// scatter constraints into various islands
		btTypedConstraint* constraint = constraints[i];
		if (constraint->isEnabled())
		{
			int islandId = btGetConstraintIslandId1(constraint);
			// if island is not sleeping,
			if (Island* island = getIsland(islandId))
			{
				island->constraintArray.push_back(constraint);
			}
		}
	}
}

void btSimulationIslandManagerMt::mergeIslands()
{
	// sort islands in order of decreasing batch size
	m_activeIslands.quickSort(IslandBatchSizeSortPredicate());

	// merge small islands to satisfy minimum batch size
	// find first small batch island
	int destIslandIndex = m_activeIslands.size();
	for (int i = 0; i < m_activeIslands.size(); ++i)
	{
		Island* island = m_activeIslands[i];
		int batchSize = calcBatchCost(island);
		if (batchSize < m_minimumSolverBatchSize)
		{
			destIslandIndex = i;
			break;
		}
	}
	int lastIndex = m_activeIslands.size() - 1;
	while (destIslandIndex < lastIndex)
	{
		// merge islands from the back of the list
		Island* island = m_activeIslands[destIslandIndex];
		int numBodies = island->bodyArray.size();
		int numManifolds = island->manifoldArray.size();
		int numConstraints = island->constraintArray.size();
		int firstIndex = lastIndex;
		// figure out how many islands we want to merge and find out how many bodies, manifolds and constraints we will have
		while (true)
		{
			Island* src = m_activeIslands[firstIndex];
			numBodies += src->bodyArray.size();
			numManifolds += src->manifoldArray.size();
			numConstraints += src->constraintArray.size();
			int batchCost = calcBatchCost(numBodies, numManifolds, numConstraints);
			if (batchCost >= m_minimumSolverBatchSize)
			{
				break;
			}
			if (firstIndex - 1 == destIslandIndex)
			{
				break;
			}
			firstIndex--;
		}
		// reserve space for these pointers to minimize reallocation
		island->bodyArray.reserve(numBodies);
		island->manifoldArray.reserve(numManifolds);
		island->constraintArray.reserve(numConstraints);
		// merge islands
		for (int i = firstIndex; i <= lastIndex; ++i)
		{
			island->append(*m_activeIslands[i]);
		}
		// shrink array to exclude the islands that were merged from
		m_activeIslands.resize(firstIndex);
		lastIndex = firstIndex - 1;
		destIslandIndex++;
	}
}

void btSimulationIslandManagerMt::solveIsland(btConstraintSolver* solver, Island& island, const SolverParams& solverParams)
{
	btPersistentManifold** manifolds = island.manifoldArray.size() ? &island.manifoldArray[0] : NULL;
	btTypedConstraint** constraintsPtr = island.constraintArray.size() ? &island.constraintArray[0] : NULL;
	solver->solveGroup(&island.bodyArray[0],
					   island.bodyArray.size(),
					   manifolds,
					   island.manifoldArray.size(),
					   constraintsPtr,
					   island.constraintArray.size(),
					   *solverParams.m_solverInfo,
					   solverParams.m_debugDrawer,
					   solverParams.m_dispatcher);
}

void btSimulationIslandManagerMt::serialIslandDispatch(btAlignedObjectArray<Island*>* islandsPtr, const SolverParams& solverParams)
{
	BT_PROFILE("serialIslandDispatch");
	// serial dispatch
	btAlignedObjectArray<Island*>& islands = *islandsPtr;
	btConstraintSolver* solver = solverParams.m_solverMt ? solverParams.m_solverMt : solverParams.m_solverPool;
	for (int i = 0; i < islands.size(); ++i)
	{
		solveIsland(solver, *islands[i], solverParams);
	}
}

struct UpdateIslandDispatcher : public btIParallelForBody
{
	btAlignedObjectArray<btSimulationIslandManagerMt::Island*>& m_islandsPtr;
	const btSimulationIslandManagerMt::SolverParams& m_solverParams;

	UpdateIslandDispatcher(btAlignedObjectArray<btSimulationIslandManagerMt::Island*>& islandsPtr, const btSimulationIslandManagerMt::SolverParams& solverParams)
		: m_islandsPtr(islandsPtr), m_solverParams(solverParams)
	{
	}

	void forLoop(int iBegin, int iEnd) const BT_OVERRIDE
	{
		btConstraintSolver* solver = m_solverParams.m_solverPool;
		for (int i = iBegin; i < iEnd; ++i)
		{
			btSimulationIslandManagerMt::Island* island = m_islandsPtr[i];
			btSimulationIslandManagerMt::solveIsland(solver, *island, m_solverParams);
		}
	}
};

void btSimulationIslandManagerMt::parallelIslandDispatch(btAlignedObjectArray<Island*>* islandsPtr, const SolverParams& solverParams)
{
	BT_PROFILE("parallelIslandDispatch");
	//
	// if there are islands with many contacts, it may be faster to submit these
	// large islands *serially* to a single parallel constraint solver, and then later
	// submit the remaining smaller islands in parallel to multiple sequential solvers.
	//
	// Some task schedulers do not deal well with nested parallelFor loops. One implementation
	// of OpenMP was actually slower than doing everything single-threaded. Intel TBB
	// on the other hand, seems to do a pretty respectable job with it.
	//
	// When solving islands in parallel, the worst case performance happens when there
	// is one very large island and then perhaps a smattering of very small
	// islands -- one worker thread takes the large island and the remaining workers
	// tear through the smaller islands and then sit idle waiting for the first worker
	// to finish. Solving islands in parallel works best when there are numerous small
	// islands, roughly equal in size.
	//
	// By contrast, the other approach -- the parallel constraint solver -- is only
	// able to deliver a worthwhile speedup when the island is large. For smaller islands,
	// it is difficult to extract a useful amount of parallelism -- the overhead of grouping
	// the constraints into batches and sending the batches to worker threads can nullify
	// any gains from parallelism.
	//

	UpdateIslandDispatcher dispatcher(*islandsPtr, solverParams);
	// We take advantage of the fact the islands are sorted in order of decreasing size
	int iBegin = 0;
	if (solverParams.m_solverMt)
	{
		while (iBegin < islandsPtr->size())
		{
			btSimulationIslandManagerMt::Island* island = (*islandsPtr)[iBegin];
			if (island->manifoldArray.size() < btSequentialImpulseConstraintSolverMt::s_minimumContactManifoldsForBatching)
			{
				// OK to submit the rest of the array in parallel
				break;
			}
			// serial dispatch to parallel solver for large islands (if any)
			solveIsland(solverParams.m_solverMt, *island, solverParams);
			++iBegin;
		}
	}
	// parallel dispatch to sequential solvers for rest
	btParallelFor(iBegin, islandsPtr->size(), 1, dispatcher);
}

///@todo: this is random access, it can be walked 'cache friendly'!
void btSimulationIslandManagerMt::buildAndProcessIslands(btDispatcher* dispatcher,
														 btCollisionWorld* collisionWorld,
														 btAlignedObjectArray<btTypedConstraint*>& constraints,
														 const SolverParams& solverParams)
{
	BT_PROFILE("buildAndProcessIslands");
	btCollisionObjectArray& collisionObjects = collisionWorld->getCollisionObjectArray();

	buildIslands(dispatcher, collisionWorld);

	if (!getSplitIslands())
	{
		btPersistentManifold** manifolds = dispatcher->getInternalManifoldPointer();
		int maxNumManifolds = dispatcher->getNumManifolds();

		for (int i = 0; i < maxNumManifolds; i++)
		{
			btPersistentManifold* manifold = manifolds[i];

			const btCollisionObject* colObj0 = static_cast<const btCollisionObject*>(manifold->getBody0());
			const btCollisionObject* colObj1 = static_cast<const btCollisionObject*>(manifold->getBody1());

			///@todo: check sleeping conditions!
			if (((colObj0) && colObj0->getActivationState() != ISLAND_SLEEPING) ||
				((colObj1) && colObj1->getActivationState() != ISLAND_SLEEPING))
			{
				//kinematic objects don't merge islands, but wake up all connected objects
				if (colObj0->isKinematicObject() && colObj0->getActivationState() != ISLAND_SLEEPING)
				{
					if (colObj0->hasContactResponse())
						colObj1->activate();
				}
				if (colObj1->isKinematicObject() && colObj1->getActivationState() != ISLAND_SLEEPING)
				{
					if (colObj1->hasContactResponse())
						colObj0->activate();
				}
			}
		}
		btTypedConstraint** constraintsPtr = constraints.size() ? &constraints[0] : NULL;
		btConstraintSolver* solver = solverParams.m_solverMt ? solverParams.m_solverMt : solverParams.m_solverPool;
		solver->solveGroup(&collisionObjects[0],
						   collisionObjects.size(),
						   manifolds,
						   maxNumManifolds,
						   constraintsPtr,
						   constraints.size(),
						   *solverParams.m_solverInfo,
						   solverParams.m_debugDrawer,
						   solverParams.m_dispatcher);
	}
	else
	{
		initIslandPools();

		//traverse the simulation islands, and call the solver, unless all objects are sleeping/deactivated
		addBodiesToIslands(collisionWorld);
		addManifoldsToIslands(dispatcher);
		addConstraintsToIslands(constraints);

		// m_activeIslands array should now contain all non-sleeping Islands, and each Island should
		// have all the necessary bodies, manifolds and constraints.

		// if we want to merge islands with small batch counts,
		if (m_minimumSolverBatchSize > 1)
		{
			mergeIslands();
		}
		// dispatch islands to solver
		m_islandDispatch(&m_activeIslands, solverParams);
	}
}
