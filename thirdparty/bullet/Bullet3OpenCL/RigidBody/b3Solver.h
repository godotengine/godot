/*
Copyright (c) 2012 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Takahiro Harada

#ifndef __ADL_SOLVER_H
#define __ADL_SOLVER_H

#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"
#include "b3GpuConstraint4.h"

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Contact4.h"

#include "Bullet3OpenCL/ParallelPrimitives/b3PrefixScanCL.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3RadixSort32CL.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3BoundSearchCL.h"

#include "Bullet3OpenCL/Initialize/b3OpenCLUtils.h"

#define B3NEXTMULTIPLEOF(num, alignment) (((num) / (alignment) + (((num) % (alignment) == 0) ? 0 : 1)) * (alignment))

enum
{
	B3_SOLVER_N_SPLIT_X = 8,  //16,//4,
	B3_SOLVER_N_SPLIT_Y = 4,  //16,//4,
	B3_SOLVER_N_SPLIT_Z = 8,  //,
	B3_SOLVER_N_CELLS = B3_SOLVER_N_SPLIT_X * B3_SOLVER_N_SPLIT_Y * B3_SOLVER_N_SPLIT_Z,
	B3_SOLVER_N_BATCHES = 8,  //4,//8,//4,
	B3_MAX_NUM_BATCHES = 128,
};

class b3SolverBase
{
public:
	struct ConstraintCfg
	{
		ConstraintCfg(float dt = 0.f) : m_positionDrift(0.005f), m_positionConstraintCoeff(0.2f), m_dt(dt), m_staticIdx(-1) {}

		float m_positionDrift;
		float m_positionConstraintCoeff;
		float m_dt;
		bool m_enableParallelSolve;
		float m_batchCellSize;
		int m_staticIdx;
	};
};

class b3Solver : public b3SolverBase
{
public:
	cl_context m_context;
	cl_device_id m_device;
	cl_command_queue m_queue;

	b3OpenCLArray<unsigned int>* m_numConstraints;
	b3OpenCLArray<unsigned int>* m_offsets;
	b3OpenCLArray<int> m_batchSizes;

	int m_nIterations;
	cl_kernel m_batchingKernel;
	cl_kernel m_batchingKernelNew;
	cl_kernel m_solveContactKernel;
	cl_kernel m_solveFrictionKernel;
	cl_kernel m_contactToConstraintKernel;
	cl_kernel m_setSortDataKernel;
	cl_kernel m_reorderContactKernel;
	cl_kernel m_copyConstraintKernel;

	class b3RadixSort32CL* m_sort32;
	class b3BoundSearchCL* m_search;
	class b3PrefixScanCL* m_scan;

	b3OpenCLArray<b3SortData>* m_sortDataBuffer;
	b3OpenCLArray<b3Contact4>* m_contactBuffer2;

	enum
	{
		DYNAMIC_CONTACT_ALLOCATION_THRESHOLD = 2000000,
	};

	b3Solver(cl_context ctx, cl_device_id device, cl_command_queue queue, int pairCapacity);

	virtual ~b3Solver();

	void solveContactConstraint(const b3OpenCLArray<b3RigidBodyData>* bodyBuf, const b3OpenCLArray<b3InertiaData>* inertiaBuf,
								b3OpenCLArray<b3GpuConstraint4>* constraint, void* additionalData, int n, int maxNumBatches);

	void solveContactConstraintHost(b3OpenCLArray<b3RigidBodyData>* bodyBuf, b3OpenCLArray<b3InertiaData>* shapeBuf,
									b3OpenCLArray<b3GpuConstraint4>* constraint, void* additionalData, int n, int maxNumBatches, b3AlignedObjectArray<int>* batchSizes);

	void convertToConstraints(const b3OpenCLArray<b3RigidBodyData>* bodyBuf,
							  const b3OpenCLArray<b3InertiaData>* shapeBuf,
							  b3OpenCLArray<b3Contact4>* contactsIn, b3OpenCLArray<b3GpuConstraint4>* contactCOut, void* additionalData,
							  int nContacts, const ConstraintCfg& cfg);

	void batchContacts(b3OpenCLArray<b3Contact4>* contacts, int nContacts, b3OpenCLArray<unsigned int>* n, b3OpenCLArray<unsigned int>* offsets, int staticIdx);
};

#endif  //__ADL_SOLVER_H
