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

#ifndef BT_SIMULATION_ISLAND_MANAGER_MT_H
#define BT_SIMULATION_ISLAND_MANAGER_MT_H

#include "BulletCollision/CollisionDispatch/btSimulationIslandManager.h"

class btTypedConstraint;
class btConstraintSolver;
struct btContactSolverInfo;
class btIDebugDraw;

///
/// SimulationIslandManagerMt -- Multithread capable version of SimulationIslandManager
///                       Splits the world up into islands which can be solved in parallel.
///                       In order to solve islands in parallel, an IslandDispatch function
///                       must be provided which will dispatch calls to multiple threads.
///                       The amount of parallelism that can be achieved depends on the number
///                       of islands. If only a single island exists, then no parallelism is
///                       possible.
///
class btSimulationIslandManagerMt : public btSimulationIslandManager
{
public:
	struct Island
	{
		// a simulation island consisting of bodies, manifolds and constraints,
		// to be passed into a constraint solver.
		btAlignedObjectArray<btCollisionObject*> bodyArray;
		btAlignedObjectArray<btPersistentManifold*> manifoldArray;
		btAlignedObjectArray<btTypedConstraint*> constraintArray;
		int id;  // island id
		bool isSleeping;

		void append(const Island& other);  // add bodies, manifolds, constraints to my own
	};
	struct SolverParams
	{
		btConstraintSolver* m_solverPool;
		btConstraintSolver* m_solverMt;
		btContactSolverInfo* m_solverInfo;
		btIDebugDraw* m_debugDrawer;
		btDispatcher* m_dispatcher;
	};
	static void solveIsland(btConstraintSolver* solver, Island& island, const SolverParams& solverParams);

	typedef void (*IslandDispatchFunc)(btAlignedObjectArray<Island*>* islands, const SolverParams& solverParams);
	static void serialIslandDispatch(btAlignedObjectArray<Island*>* islandsPtr, const SolverParams& solverParams);
	static void parallelIslandDispatch(btAlignedObjectArray<Island*>* islandsPtr, const SolverParams& solverParams);

protected:
	btAlignedObjectArray<Island*> m_allocatedIslands;    // owner of all Islands
	btAlignedObjectArray<Island*> m_activeIslands;       // islands actively in use
	btAlignedObjectArray<Island*> m_freeIslands;         // islands ready to be reused
	btAlignedObjectArray<Island*> m_lookupIslandFromId;  // big lookup table to map islandId to Island pointer
	Island* m_batchIsland;
	int m_minimumSolverBatchSize;
	int m_batchIslandMinBodyCount;
	IslandDispatchFunc m_islandDispatch;

	Island* getIsland(int id);
	virtual Island* allocateIsland(int id, int numBodies);
	virtual void initIslandPools();
	virtual void addBodiesToIslands(btCollisionWorld* collisionWorld);
	virtual void addManifoldsToIslands(btDispatcher* dispatcher);
	virtual void addConstraintsToIslands(btAlignedObjectArray<btTypedConstraint*>& constraints);
	virtual void mergeIslands();

public:
	btSimulationIslandManagerMt();
	virtual ~btSimulationIslandManagerMt();

	virtual void buildAndProcessIslands(btDispatcher* dispatcher,
										btCollisionWorld* collisionWorld,
										btAlignedObjectArray<btTypedConstraint*>& constraints,
										const SolverParams& solverParams);

	virtual void buildIslands(btDispatcher* dispatcher, btCollisionWorld* colWorld);

	int getMinimumSolverBatchSize() const
	{
		return m_minimumSolverBatchSize;
	}
	void setMinimumSolverBatchSize(int sz)
	{
		m_minimumSolverBatchSize = sz;
	}
	IslandDispatchFunc getIslandDispatchFunction() const
	{
		return m_islandDispatch;
	}
	// allow users to set their own dispatch function for multithreaded dispatch
	void setIslandDispatchFunction(IslandDispatchFunc func)
	{
		m_islandDispatch = func;
	}
};

#endif  //BT_SIMULATION_ISLAND_MANAGER_H
