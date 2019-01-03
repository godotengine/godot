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

#ifndef BT_BATCHED_CONSTRAINTS_H
#define BT_BATCHED_CONSTRAINTS_H

#include "LinearMath/btThreads.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "BulletDynamics/ConstraintSolver/btSolverBody.h"
#include "BulletDynamics/ConstraintSolver/btSolverConstraint.h"

class btIDebugDraw;

struct btBatchedConstraints
{
	enum BatchingMethod
	{
		BATCHING_METHOD_SPATIAL_GRID_2D,
		BATCHING_METHOD_SPATIAL_GRID_3D,
		BATCHING_METHOD_COUNT
	};
	struct Range
	{
		int begin;
		int end;

		Range() : begin(0), end(0) {}
		Range(int _beg, int _end) : begin(_beg), end(_end) {}
	};

	btAlignedObjectArray<int> m_constraintIndices;
	btAlignedObjectArray<Range> m_batches;        // each batch is a range of indices in the m_constraintIndices array
	btAlignedObjectArray<Range> m_phases;         // each phase is range of indices in the m_batches array
	btAlignedObjectArray<char> m_phaseGrainSize;  // max grain size for each phase
	btAlignedObjectArray<int> m_phaseOrder;       // phases can be done in any order, so we can randomize the order here
	btIDebugDraw* m_debugDrawer;

	static bool s_debugDrawBatches;

	btBatchedConstraints() { m_debugDrawer = NULL; }
	void setup(btConstraintArray* constraints,
			   const btAlignedObjectArray<btSolverBody>& bodies,
			   BatchingMethod batchingMethod,
			   int minBatchSize,
			   int maxBatchSize,
			   btAlignedObjectArray<char>* scratchMemory);
	bool validate(btConstraintArray* constraints, const btAlignedObjectArray<btSolverBody>& bodies) const;
};

#endif  // BT_BATCHED_CONSTRAINTS_H
