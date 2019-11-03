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

#ifndef B3_GPU_RIGIDBODY_PIPELINE_INTERNAL_DATA_H
#define B3_GPU_RIGIDBODY_PIPELINE_INTERNAL_DATA_H

#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"
#include "Bullet3Common/b3AlignedObjectArray.h"

#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"

#include "Bullet3OpenCL/BroadphaseCollision/b3SapAabb.h"
#include "Bullet3Dynamics/ConstraintSolver/b3TypedConstraint.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Config.h"

#include "Bullet3Collision/BroadPhaseCollision/b3OverlappingPair.h"
#include "Bullet3OpenCL/RigidBody/b3GpuGenericConstraint.h"

struct b3GpuRigidBodyPipelineInternalData
{
	cl_context m_context;
	cl_device_id m_device;
	cl_command_queue m_queue;

	cl_kernel m_integrateTransformsKernel;
	cl_kernel m_updateAabbsKernel;
	cl_kernel m_clearOverlappingPairsKernel;

	class b3PgsJacobiSolver* m_solver;

	class b3GpuPgsConstraintSolver* m_gpuSolver;

	class b3GpuPgsContactSolver* m_solver2;
	class b3GpuJacobiContactSolver* m_solver3;
	class b3GpuRaycast* m_raycaster;

	class b3GpuBroadphaseInterface* m_broadphaseSap;

	struct b3DynamicBvhBroadphase* m_broadphaseDbvt;
	b3OpenCLArray<b3SapAabb>* m_allAabbsGPU;
	b3AlignedObjectArray<b3SapAabb> m_allAabbsCPU;
	b3OpenCLArray<b3BroadphasePair>* m_overlappingPairsGPU;

	b3OpenCLArray<b3GpuGenericConstraint>* m_gpuConstraints;
	b3AlignedObjectArray<b3GpuGenericConstraint> m_cpuConstraints;

	b3AlignedObjectArray<b3TypedConstraint*> m_joints;
	int m_constraintUid;
	class b3GpuNarrowPhase* m_narrowphase;
	b3Vector3 m_gravity;

	b3Config m_config;
};

#endif  //B3_GPU_RIGIDBODY_PIPELINE_INTERNAL_DATA_H
