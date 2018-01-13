/*
Copyright (c) 2013 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

#ifndef B3_GPU_PGS_CONSTRAINT_SOLVER_H
#define B3_GPU_PGS_CONSTRAINT_SOLVER_H

struct b3Contact4;
struct b3ContactPoint;


class b3Dispatcher;

#include "Bullet3Dynamics/ConstraintSolver/b3TypedConstraint.h"
#include "Bullet3Dynamics/ConstraintSolver/b3ContactSolverInfo.h"
#include "b3GpuSolverBody.h"
#include "b3GpuSolverConstraint.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"
struct b3RigidBodyData;
struct b3InertiaData;

#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"
#include "b3GpuGenericConstraint.h"

class b3GpuPgsConstraintSolver
{
protected:
	int m_staticIdx;
	struct b3GpuPgsJacobiSolverInternalData* m_gpuData;
	protected:
	b3AlignedObjectArray<b3GpuSolverBody>      m_tmpSolverBodyPool;
	b3GpuConstraintArray			m_tmpSolverContactConstraintPool;
	b3GpuConstraintArray			m_tmpSolverNonContactConstraintPool;
	b3GpuConstraintArray			m_tmpSolverContactFrictionConstraintPool;
	b3GpuConstraintArray			m_tmpSolverContactRollingFrictionConstraintPool;

	b3AlignedObjectArray<unsigned int> m_tmpConstraintSizesPool;
	

	bool						m_usePgs;
	void						averageVelocities();

	int							m_maxOverrideNumSolverIterations;

	int							m_numSplitImpulseRecoveries;

//	int	getOrInitSolverBody(int bodyIndex, b3RigidBodyData* bodies,b3InertiaData* inertias);
	void	initSolverBody(int bodyIndex, b3GpuSolverBody* solverBody, b3RigidBodyData* rb);

public:
	b3GpuPgsConstraintSolver (cl_context ctx, cl_device_id device, cl_command_queue queue,bool usePgs);
	virtual~b3GpuPgsConstraintSolver ();

	virtual b3Scalar solveGroupCacheFriendlyIterations(b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints1,int numConstraints,const b3ContactSolverInfo& infoGlobal);
	virtual b3Scalar solveGroupCacheFriendlySetup(b3OpenCLArray<b3RigidBodyData>* gpuBodies, b3OpenCLArray<b3InertiaData>* gpuInertias, int numBodies,b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);
	b3Scalar solveGroupCacheFriendlyFinish(b3OpenCLArray<b3RigidBodyData>* gpuBodies,b3OpenCLArray<b3InertiaData>* gpuInertias,int numBodies,b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);


	b3Scalar solveGroup(b3OpenCLArray<b3RigidBodyData>* gpuBodies,b3OpenCLArray<b3InertiaData>* gpuInertias, int numBodies,b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);
	void	solveJoints(int numBodies, b3OpenCLArray<b3RigidBodyData>* gpuBodies, b3OpenCLArray<b3InertiaData>* gpuInertias, 
				int numConstraints, b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints);

	int sortConstraintByBatch3( struct b3BatchConstraint* cs, int numConstraints, int simdWidth , int staticIdx, int numBodies);
	void	recomputeBatches();
};

#endif //B3_GPU_PGS_CONSTRAINT_SOLVER_H
