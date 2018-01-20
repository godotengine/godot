
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


bool useGpuInitSolverBodies = true;
bool useGpuInfo1 = true;
bool useGpuInfo2= true;
bool useGpuSolveJointConstraintRows=true;
bool useGpuWriteBackVelocities = true;
bool gpuBreakConstraints = true;

#include "b3GpuPgsConstraintSolver.h"

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include "Bullet3Dynamics/ConstraintSolver/b3TypedConstraint.h"
#include <new>
#include "Bullet3Common/b3AlignedObjectArray.h"
#include <string.h> //for memset
#include "Bullet3Collision/NarrowPhaseCollision/b3Contact4.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h"

#include "Bullet3OpenCL/ParallelPrimitives/b3PrefixScanCL.h"

#include "Bullet3OpenCL/RigidBody/kernels/jointSolver.h" //solveConstraintRowsCL
#include "Bullet3OpenCL/Initialize/b3OpenCLUtils.h"

#define B3_JOINT_SOLVER_PATH "src/Bullet3OpenCL/RigidBody/kernels/jointSolver.cl"


struct b3GpuPgsJacobiSolverInternalData
{

	cl_context m_context;
	cl_device_id m_device;
	cl_command_queue m_queue;

	b3PrefixScanCL*	m_prefixScan;

	cl_kernel m_solveJointConstraintRowsKernels;
	cl_kernel m_initSolverBodiesKernel;
	cl_kernel m_getInfo1Kernel;
	cl_kernel m_initBatchConstraintsKernel;
	cl_kernel m_getInfo2Kernel;
	cl_kernel m_writeBackVelocitiesKernel;
	cl_kernel m_breakViolatedConstraintsKernel;

	b3OpenCLArray<unsigned int>*	m_gpuConstraintRowOffsets;

	b3OpenCLArray<b3GpuSolverBody>*			m_gpuSolverBodies;
	b3OpenCLArray<b3BatchConstraint>*		m_gpuBatchConstraints;
	b3OpenCLArray<b3GpuSolverConstraint>*		m_gpuConstraintRows;
	b3OpenCLArray<unsigned int>*			m_gpuConstraintInfo1;

//	b3AlignedObjectArray<b3GpuSolverBody>		m_cpuSolverBodies;
	b3AlignedObjectArray<b3BatchConstraint>		m_cpuBatchConstraints;
	b3AlignedObjectArray<b3GpuSolverConstraint>	m_cpuConstraintRows;
	b3AlignedObjectArray<unsigned int>			m_cpuConstraintInfo1;
	b3AlignedObjectArray<unsigned int>			m_cpuConstraintRowOffsets;

	b3AlignedObjectArray<b3RigidBodyData>			m_cpuBodies;
	b3AlignedObjectArray<b3InertiaData>			m_cpuInertias;

	
	b3AlignedObjectArray<b3GpuGenericConstraint> m_cpuConstraints;

	b3AlignedObjectArray<int>		m_batchSizes;


};


/*
static b3Transform	getWorldTransform(b3RigidBodyData* rb)
{
	b3Transform newTrans;
	newTrans.setOrigin(rb->m_pos);
	newTrans.setRotation(rb->m_quat);
	return newTrans;
}

static const b3Matrix3x3&	getInvInertiaTensorWorld(b3InertiaData* inertia)
{
	return inertia->m_invInertiaWorld;
}

*/

static const b3Vector3&	getLinearVelocity(b3RigidBodyData* rb)
{
	return rb->m_linVel;
}

static const b3Vector3&	getAngularVelocity(b3RigidBodyData* rb)
{
	return rb->m_angVel;
}

b3Vector3 getVelocityInLocalPoint(b3RigidBodyData* rb, const b3Vector3& rel_pos)
{
	//we also calculate lin/ang velocity for kinematic objects
	return getLinearVelocity(rb) + getAngularVelocity(rb).cross(rel_pos);
	
}



b3GpuPgsConstraintSolver::b3GpuPgsConstraintSolver (cl_context ctx, cl_device_id device, cl_command_queue queue,bool usePgs)
{
	m_usePgs = usePgs;
	m_gpuData = new b3GpuPgsJacobiSolverInternalData();
	m_gpuData->m_context = ctx;
	m_gpuData->m_device = device;
	m_gpuData->m_queue = queue;

	m_gpuData->m_prefixScan = new b3PrefixScanCL(ctx,device,queue);

	m_gpuData->m_gpuConstraintRowOffsets = new b3OpenCLArray<unsigned int>(m_gpuData->m_context,m_gpuData->m_queue);

	m_gpuData->m_gpuSolverBodies = new b3OpenCLArray<b3GpuSolverBody>(m_gpuData->m_context,m_gpuData->m_queue);
	m_gpuData->m_gpuBatchConstraints = new b3OpenCLArray<b3BatchConstraint>(m_gpuData->m_context,m_gpuData->m_queue);
	m_gpuData->m_gpuConstraintRows = new b3OpenCLArray<b3GpuSolverConstraint>(m_gpuData->m_context,m_gpuData->m_queue);
	m_gpuData->m_gpuConstraintInfo1 = new b3OpenCLArray<unsigned int>(m_gpuData->m_context,m_gpuData->m_queue);
	cl_int errNum=0;

	{
		cl_program prog = b3OpenCLUtils::compileCLProgramFromString(m_gpuData->m_context,m_gpuData->m_device,solveConstraintRowsCL,&errNum,"",B3_JOINT_SOLVER_PATH);
		//cl_program prog = b3OpenCLUtils::compileCLProgramFromString(m_gpuData->m_context,m_gpuData->m_device,0,&errNum,"",B3_JOINT_SOLVER_PATH,true);
		b3Assert(errNum==CL_SUCCESS);
		m_gpuData->m_solveJointConstraintRowsKernels = b3OpenCLUtils::compileCLKernelFromString(m_gpuData->m_context, m_gpuData->m_device,solveConstraintRowsCL, "solveJointConstraintRows",&errNum,prog);
		b3Assert(errNum==CL_SUCCESS);
		m_gpuData->m_initSolverBodiesKernel = b3OpenCLUtils::compileCLKernelFromString(m_gpuData->m_context,m_gpuData->m_device,solveConstraintRowsCL,"initSolverBodies",&errNum,prog);
		b3Assert(errNum==CL_SUCCESS);
		m_gpuData->m_getInfo1Kernel = b3OpenCLUtils::compileCLKernelFromString(m_gpuData->m_context,m_gpuData->m_device,solveConstraintRowsCL,"getInfo1Kernel",&errNum,prog);
		b3Assert(errNum==CL_SUCCESS);
		m_gpuData->m_initBatchConstraintsKernel = b3OpenCLUtils::compileCLKernelFromString(m_gpuData->m_context,m_gpuData->m_device,solveConstraintRowsCL,"initBatchConstraintsKernel",&errNum,prog);
		b3Assert(errNum==CL_SUCCESS);
		m_gpuData->m_getInfo2Kernel= b3OpenCLUtils::compileCLKernelFromString(m_gpuData->m_context,m_gpuData->m_device,solveConstraintRowsCL,"getInfo2Kernel",&errNum,prog);
		b3Assert(errNum==CL_SUCCESS);
		m_gpuData->m_writeBackVelocitiesKernel = b3OpenCLUtils::compileCLKernelFromString(m_gpuData->m_context,m_gpuData->m_device,solveConstraintRowsCL,"writeBackVelocitiesKernel",&errNum,prog);
		b3Assert(errNum==CL_SUCCESS);
		m_gpuData->m_breakViolatedConstraintsKernel = b3OpenCLUtils::compileCLKernelFromString(m_gpuData->m_context,m_gpuData->m_device,solveConstraintRowsCL,"breakViolatedConstraintsKernel",&errNum,prog);
		b3Assert(errNum==CL_SUCCESS);

		
		

		clReleaseProgram(prog);
	}


}

b3GpuPgsConstraintSolver::~b3GpuPgsConstraintSolver ()
{
	clReleaseKernel(m_gpuData->m_solveJointConstraintRowsKernels);
	clReleaseKernel(m_gpuData->m_initSolverBodiesKernel);
	clReleaseKernel(m_gpuData->m_getInfo1Kernel);
	clReleaseKernel(m_gpuData->m_initBatchConstraintsKernel);
	clReleaseKernel(m_gpuData->m_getInfo2Kernel);
	clReleaseKernel(m_gpuData->m_writeBackVelocitiesKernel);
	clReleaseKernel(m_gpuData->m_breakViolatedConstraintsKernel);

	delete m_gpuData->m_prefixScan;
	delete m_gpuData->m_gpuConstraintRowOffsets;
	delete m_gpuData->m_gpuSolverBodies;
	delete m_gpuData->m_gpuBatchConstraints;
	delete m_gpuData->m_gpuConstraintRows;
	delete m_gpuData->m_gpuConstraintInfo1;

	delete m_gpuData;
}

struct b3BatchConstraint
{
	int m_bodyAPtrAndSignBit;
	int m_bodyBPtrAndSignBit;
	int m_originalConstraintIndex;
	int m_batchId;
};

static b3AlignedObjectArray<b3BatchConstraint> batchConstraints;


void	b3GpuPgsConstraintSolver::recomputeBatches()
{
	m_gpuData->m_batchSizes.clear();
}




b3Scalar b3GpuPgsConstraintSolver::solveGroupCacheFriendlySetup(b3OpenCLArray<b3RigidBodyData>* gpuBodies, b3OpenCLArray<b3InertiaData>* gpuInertias, int numBodies, b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints,int numConstraints,const b3ContactSolverInfo& infoGlobal)
{
	B3_PROFILE("GPU solveGroupCacheFriendlySetup");
	batchConstraints.resize(numConstraints);
	m_gpuData->m_gpuBatchConstraints->resize(numConstraints);
	m_staticIdx = -1;
	m_maxOverrideNumSolverIterations = 0;


	/*	m_gpuData->m_gpuBodies->resize(numBodies);
	m_gpuData->m_gpuBodies->copyFromHostPointer(bodies,numBodies);

	b3OpenCLArray<b3InertiaData> gpuInertias(m_gpuData->m_context,m_gpuData->m_queue);
	gpuInertias.resize(numBodies);
	gpuInertias.copyFromHostPointer(inertias,numBodies);
	*/

	m_gpuData->m_gpuSolverBodies->resize(numBodies);


	m_tmpSolverBodyPool.resize(numBodies);
	{
		
		if (useGpuInitSolverBodies)
		{
			B3_PROFILE("m_initSolverBodiesKernel");

			b3LauncherCL launcher(m_gpuData->m_queue,m_gpuData->m_initSolverBodiesKernel,"m_initSolverBodiesKernel");
			launcher.setBuffer(m_gpuData->m_gpuSolverBodies->getBufferCL());
			launcher.setBuffer(gpuBodies->getBufferCL());
			launcher.setConst(numBodies);
			launcher.launch1D(numBodies);
			clFinish(m_gpuData->m_queue);

			//			m_gpuData->m_gpuSolverBodies->copyToHost(m_tmpSolverBodyPool);
		} else
		{
			gpuBodies->copyToHost(m_gpuData->m_cpuBodies);
			for (int i=0;i<numBodies;i++)
			{

				b3RigidBodyData& body = m_gpuData->m_cpuBodies[i];
				b3GpuSolverBody& solverBody = m_tmpSolverBodyPool[i];
				initSolverBody(i,&solverBody,&body);
				solverBody.m_originalBodyIndex = i;
			}
			m_gpuData->m_gpuSolverBodies->copyFromHost(m_tmpSolverBodyPool);
		}
	}

//	int totalBodies = 0;
	int totalNumRows = 0;
	//b3RigidBody* rb0=0,*rb1=0;
	//if (1)
	{
		{


			//			int i;

			m_tmpConstraintSizesPool.resizeNoInitialize(numConstraints);

			//			b3OpenCLArray<b3GpuGenericConstraint> gpuConstraints(m_gpuData->m_context,m_gpuData->m_queue);


			if (useGpuInfo1)
			{
				B3_PROFILE("info1 and init batchConstraint");
				
				m_gpuData->m_gpuConstraintInfo1->resize(numConstraints);

				
				if (1)
				{
					B3_PROFILE("getInfo1Kernel");

					b3LauncherCL launcher(m_gpuData->m_queue,m_gpuData->m_getInfo1Kernel,"m_getInfo1Kernel");
					launcher.setBuffer(m_gpuData->m_gpuConstraintInfo1->getBufferCL());
					launcher.setBuffer(gpuConstraints->getBufferCL());
					launcher.setConst(numConstraints);
					launcher.launch1D(numConstraints);
					clFinish(m_gpuData->m_queue);
				}

				if (m_gpuData->m_batchSizes.size()==0)
				{
					B3_PROFILE("initBatchConstraintsKernel");

					m_gpuData->m_gpuConstraintRowOffsets->resize(numConstraints);
					unsigned int total=0;
					m_gpuData->m_prefixScan->execute(*m_gpuData->m_gpuConstraintInfo1,*m_gpuData->m_gpuConstraintRowOffsets,numConstraints,&total);
					unsigned int lastElem = m_gpuData->m_gpuConstraintInfo1->at(numConstraints-1);
					totalNumRows = total+lastElem;

					{
						B3_PROFILE("init batch constraints");
						b3LauncherCL launcher(m_gpuData->m_queue,m_gpuData->m_initBatchConstraintsKernel,"m_initBatchConstraintsKernel");
						launcher.setBuffer(m_gpuData->m_gpuConstraintInfo1->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuConstraintRowOffsets->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuBatchConstraints->getBufferCL());
						launcher.setBuffer(gpuConstraints->getBufferCL());
						launcher.setBuffer(gpuBodies->getBufferCL());
						launcher.setConst(numConstraints);
						launcher.launch1D(numConstraints);
						clFinish(m_gpuData->m_queue);
					}
					//assume the batching happens on CPU, so copy the data
					m_gpuData->m_gpuBatchConstraints->copyToHost(batchConstraints);
				}
			} 
			else
			{
				totalNumRows  = 0;
				gpuConstraints->copyToHost(m_gpuData->m_cpuConstraints);
				//calculate the total number of contraint rows
				for (int i=0;i<numConstraints;i++)
				{
					unsigned int& info1= m_tmpConstraintSizesPool[i];
					//					unsigned int info1;
					if (m_gpuData->m_cpuConstraints[i].isEnabled())
					{

						m_gpuData->m_cpuConstraints[i].getInfo1(&info1,&m_gpuData->m_cpuBodies[0]);
					} else
					{
						info1 = 0;
					}
					
					totalNumRows += info1;
				}

				m_gpuData->m_gpuBatchConstraints->copyFromHost(batchConstraints);
				m_gpuData->m_gpuConstraintInfo1->copyFromHost(m_tmpConstraintSizesPool);

			}
			m_tmpSolverNonContactConstraintPool.resizeNoInitialize(totalNumRows);
			m_gpuData->m_gpuConstraintRows->resize(totalNumRows);
			
			//			b3GpuConstraintArray		verify;

			if (useGpuInfo2)
			{
				{
						B3_PROFILE("getInfo2Kernel");
						b3LauncherCL launcher(m_gpuData->m_queue,m_gpuData->m_getInfo2Kernel,"m_getInfo2Kernel");
						launcher.setBuffer(m_gpuData->m_gpuConstraintRows->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuConstraintInfo1->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuConstraintRowOffsets->getBufferCL());
						launcher.setBuffer(gpuConstraints->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuBatchConstraints->getBufferCL());
						launcher.setBuffer(gpuBodies->getBufferCL());
						launcher.setBuffer(gpuInertias->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuSolverBodies->getBufferCL());
						launcher.setConst(infoGlobal.m_timeStep);
						launcher.setConst(infoGlobal.m_erp);
						launcher.setConst(infoGlobal.m_globalCfm);
						launcher.setConst(infoGlobal.m_damping);
						launcher.setConst(infoGlobal.m_numIterations);
						launcher.setConst(numConstraints);
						launcher.launch1D(numConstraints);
						clFinish(m_gpuData->m_queue);

						if (m_gpuData->m_batchSizes.size()==0)
							m_gpuData->m_gpuBatchConstraints->copyToHost(batchConstraints);
						//m_gpuData->m_gpuConstraintRows->copyToHost(verify);
						//m_gpuData->m_gpuConstraintRows->copyToHost(m_tmpSolverNonContactConstraintPool);

						

					}
			} 
			else
			{
			
				gpuInertias->copyToHost(m_gpuData->m_cpuInertias);

					///setup the b3SolverConstraints
			
				for (int i=0;i<numConstraints;i++)
				{
					const int& info1 = m_tmpConstraintSizesPool[i];
				
					if (info1)
					{
						int constraintIndex = batchConstraints[i].m_originalConstraintIndex;
						int constraintRowOffset = m_gpuData->m_cpuConstraintRowOffsets[constraintIndex];

						b3GpuSolverConstraint* currentConstraintRow = &m_tmpSolverNonContactConstraintPool[constraintRowOffset];
						b3GpuGenericConstraint& constraint = m_gpuData->m_cpuConstraints[i];

						b3RigidBodyData& rbA = m_gpuData->m_cpuBodies[ constraint.getRigidBodyA()];
						//b3RigidBody& rbA = constraint.getRigidBodyA();
		//				b3RigidBody& rbB = constraint.getRigidBodyB();
						b3RigidBodyData& rbB = m_gpuData->m_cpuBodies[ constraint.getRigidBodyB()];
					
					

						int solverBodyIdA = constraint.getRigidBodyA();//getOrInitSolverBody(constraint.getRigidBodyA(),bodies,inertias);
						int solverBodyIdB = constraint.getRigidBodyB();//getOrInitSolverBody(constraint.getRigidBodyB(),bodies,inertias);

						b3GpuSolverBody* bodyAPtr = &m_tmpSolverBodyPool[solverBodyIdA];
						b3GpuSolverBody* bodyBPtr = &m_tmpSolverBodyPool[solverBodyIdB];

						if (rbA.m_invMass)
						{
							batchConstraints[i].m_bodyAPtrAndSignBit = solverBodyIdA;
						} else
						{
							if (!solverBodyIdA)
								m_staticIdx = 0;
							batchConstraints[i].m_bodyAPtrAndSignBit = -solverBodyIdA;
						}

						if (rbB.m_invMass)
						{
							batchConstraints[i].m_bodyBPtrAndSignBit = solverBodyIdB;
						} else
						{
							if (!solverBodyIdB)
								m_staticIdx = 0;
							batchConstraints[i].m_bodyBPtrAndSignBit = -solverBodyIdB;
						}


						int overrideNumSolverIterations = 0;//constraint->getOverrideNumSolverIterations() > 0 ? constraint->getOverrideNumSolverIterations() : infoGlobal.m_numIterations;
						if (overrideNumSolverIterations>m_maxOverrideNumSolverIterations)
							m_maxOverrideNumSolverIterations = overrideNumSolverIterations;


						int j;
						for ( j=0;j<info1;j++)
						{
							memset(&currentConstraintRow[j],0,sizeof(b3GpuSolverConstraint));
							currentConstraintRow[j].m_angularComponentA.setValue(0,0,0);
							currentConstraintRow[j].m_angularComponentB.setValue(0,0,0);
							currentConstraintRow[j].m_appliedImpulse = 0.f;
							currentConstraintRow[j].m_appliedPushImpulse = 0.f;
							currentConstraintRow[j].m_cfm = 0.f;
							currentConstraintRow[j].m_contactNormal.setValue(0,0,0);
							currentConstraintRow[j].m_friction = 0.f;
							currentConstraintRow[j].m_frictionIndex = 0;
							currentConstraintRow[j].m_jacDiagABInv = 0.f;
							currentConstraintRow[j].m_lowerLimit = 0.f;
							currentConstraintRow[j].m_upperLimit = 0.f;

							currentConstraintRow[j].m_originalContactPoint = 0;
							currentConstraintRow[j].m_overrideNumSolverIterations = 0;
							currentConstraintRow[j].m_relpos1CrossNormal.setValue(0,0,0);
							currentConstraintRow[j].m_relpos2CrossNormal.setValue(0,0,0);
							currentConstraintRow[j].m_rhs = 0.f;
							currentConstraintRow[j].m_rhsPenetration = 0.f;
							currentConstraintRow[j].m_solverBodyIdA = 0;
							currentConstraintRow[j].m_solverBodyIdB = 0;
							
							currentConstraintRow[j].m_lowerLimit = -B3_INFINITY;
							currentConstraintRow[j].m_upperLimit = B3_INFINITY;
							currentConstraintRow[j].m_appliedImpulse = 0.f;
							currentConstraintRow[j].m_appliedPushImpulse = 0.f;
							currentConstraintRow[j].m_solverBodyIdA = solverBodyIdA;
							currentConstraintRow[j].m_solverBodyIdB = solverBodyIdB;
							currentConstraintRow[j].m_overrideNumSolverIterations = overrideNumSolverIterations;
						}

						bodyAPtr->internalGetDeltaLinearVelocity().setValue(0.f,0.f,0.f);
						bodyAPtr->internalGetDeltaAngularVelocity().setValue(0.f,0.f,0.f);
						bodyAPtr->internalGetPushVelocity().setValue(0.f,0.f,0.f);
						bodyAPtr->internalGetTurnVelocity().setValue(0.f,0.f,0.f);
						bodyBPtr->internalGetDeltaLinearVelocity().setValue(0.f,0.f,0.f);
						bodyBPtr->internalGetDeltaAngularVelocity().setValue(0.f,0.f,0.f);
						bodyBPtr->internalGetPushVelocity().setValue(0.f,0.f,0.f);
						bodyBPtr->internalGetTurnVelocity().setValue(0.f,0.f,0.f);


						b3GpuConstraintInfo2 info2;
						info2.fps = 1.f/infoGlobal.m_timeStep;
						info2.erp = infoGlobal.m_erp;
						info2.m_J1linearAxis = currentConstraintRow->m_contactNormal;
						info2.m_J1angularAxis = currentConstraintRow->m_relpos1CrossNormal;
						info2.m_J2linearAxis = 0;
						info2.m_J2angularAxis = currentConstraintRow->m_relpos2CrossNormal;
						info2.rowskip = sizeof(b3GpuSolverConstraint)/sizeof(b3Scalar);//check this
						///the size of b3GpuSolverConstraint needs be a multiple of b3Scalar
						b3Assert(info2.rowskip*sizeof(b3Scalar)== sizeof(b3GpuSolverConstraint));
						info2.m_constraintError = &currentConstraintRow->m_rhs;
						currentConstraintRow->m_cfm = infoGlobal.m_globalCfm;
						info2.m_damping = infoGlobal.m_damping;
						info2.cfm = &currentConstraintRow->m_cfm;
						info2.m_lowerLimit = &currentConstraintRow->m_lowerLimit;
						info2.m_upperLimit = &currentConstraintRow->m_upperLimit;
						info2.m_numIterations = infoGlobal.m_numIterations;
						m_gpuData->m_cpuConstraints[i].getInfo2(&info2,&m_gpuData->m_cpuBodies[0]);

						///finalize the constraint setup
						for ( j=0;j<info1;j++)
						{
							b3GpuSolverConstraint& solverConstraint = currentConstraintRow[j];

							if (solverConstraint.m_upperLimit>=m_gpuData->m_cpuConstraints[i].getBreakingImpulseThreshold())
							{
								solverConstraint.m_upperLimit = m_gpuData->m_cpuConstraints[i].getBreakingImpulseThreshold();
							}

							if (solverConstraint.m_lowerLimit<=-m_gpuData->m_cpuConstraints[i].getBreakingImpulseThreshold())
							{
								solverConstraint.m_lowerLimit = -m_gpuData->m_cpuConstraints[i].getBreakingImpulseThreshold();
							}

	//						solverConstraint.m_originalContactPoint = constraint;
							
							b3Matrix3x3& invInertiaWorldA= m_gpuData->m_cpuInertias[constraint.getRigidBodyA()].m_invInertiaWorld;
							{

								//b3Vector3 angularFactorA(1,1,1);
								const b3Vector3& ftorqueAxis1 = solverConstraint.m_relpos1CrossNormal;
								solverConstraint.m_angularComponentA = invInertiaWorldA*ftorqueAxis1;//*angularFactorA;
							}
						
							b3Matrix3x3& invInertiaWorldB= m_gpuData->m_cpuInertias[constraint.getRigidBodyB()].m_invInertiaWorld;
							{

								const b3Vector3& ftorqueAxis2 = solverConstraint.m_relpos2CrossNormal;
								solverConstraint.m_angularComponentB = invInertiaWorldB*ftorqueAxis2;//*constraint.getRigidBodyB().getAngularFactor();
							}

							{
								//it is ok to use solverConstraint.m_contactNormal instead of -solverConstraint.m_contactNormal
								//because it gets multiplied iMJlB
								b3Vector3 iMJlA = solverConstraint.m_contactNormal*rbA.m_invMass;
								b3Vector3 iMJaA = invInertiaWorldA*solverConstraint.m_relpos1CrossNormal;
								b3Vector3 iMJlB = solverConstraint.m_contactNormal*rbB.m_invMass;//sign of normal?
								b3Vector3 iMJaB = invInertiaWorldB*solverConstraint.m_relpos2CrossNormal;

								b3Scalar sum = iMJlA.dot(solverConstraint.m_contactNormal);
								sum += iMJaA.dot(solverConstraint.m_relpos1CrossNormal);
								sum += iMJlB.dot(solverConstraint.m_contactNormal);
								sum += iMJaB.dot(solverConstraint.m_relpos2CrossNormal);
								b3Scalar fsum = b3Fabs(sum);
								b3Assert(fsum > B3_EPSILON);
								solverConstraint.m_jacDiagABInv = fsum>B3_EPSILON?b3Scalar(1.)/sum : 0.f;
							}


							///fix rhs
							///todo: add force/torque accelerators
							{
								b3Scalar rel_vel;
								b3Scalar vel1Dotn = solverConstraint.m_contactNormal.dot(rbA.m_linVel) + solverConstraint.m_relpos1CrossNormal.dot(rbA.m_angVel);
								b3Scalar vel2Dotn = -solverConstraint.m_contactNormal.dot(rbB.m_linVel) + solverConstraint.m_relpos2CrossNormal.dot(rbB.m_angVel);

								rel_vel = vel1Dotn+vel2Dotn;

								b3Scalar restitution = 0.f;
								b3Scalar positionalError = solverConstraint.m_rhs;//already filled in by getConstraintInfo2
								b3Scalar	velocityError = restitution - rel_vel * info2.m_damping;
								b3Scalar	penetrationImpulse = positionalError*solverConstraint.m_jacDiagABInv;
								b3Scalar	velocityImpulse = velocityError *solverConstraint.m_jacDiagABInv;
								solverConstraint.m_rhs = penetrationImpulse+velocityImpulse;
								solverConstraint.m_appliedImpulse = 0.f;

							}
						}

					}
				}



				m_gpuData->m_gpuConstraintRows->copyFromHost(m_tmpSolverNonContactConstraintPool);
				m_gpuData->m_gpuConstraintInfo1->copyFromHost(m_tmpConstraintSizesPool);

				if (m_gpuData->m_batchSizes.size()==0)
					m_gpuData->m_gpuBatchConstraints->copyFromHost(batchConstraints);
				else
					m_gpuData->m_gpuBatchConstraints->copyToHost(batchConstraints);

				m_gpuData->m_gpuSolverBodies->copyFromHost(m_tmpSolverBodyPool);



			}//end useGpuInfo2


		}

#ifdef B3_SUPPORT_CONTACT_CONSTRAINTS
		{
			int i;

			for (i=0;i<numManifolds;i++)
			{
				b3Contact4& manifold = manifoldPtr[i];
				convertContact(bodies,inertias,&manifold,infoGlobal);
			}
		}
#endif //B3_SUPPORT_CONTACT_CONSTRAINTS
	}

//	b3ContactSolverInfo info = infoGlobal;


//	int numNonContactPool = m_tmpSolverNonContactConstraintPool.size();
//	int numConstraintPool = m_tmpSolverContactConstraintPool.size();
//	int numFrictionPool = m_tmpSolverContactFrictionConstraintPool.size();


	return 0.f;

}



///a straight copy from GPU/OpenCL kernel, for debugging
__inline void internalApplyImpulse( b3GpuSolverBody* body,  const b3Vector3& linearComponent, const b3Vector3& angularComponent,float impulseMagnitude)
{
	body->m_deltaLinearVelocity += linearComponent*impulseMagnitude*body->m_linearFactor;
	body->m_deltaAngularVelocity += angularComponent*(impulseMagnitude*body->m_angularFactor);
}


void resolveSingleConstraintRowGeneric2( b3GpuSolverBody* body1,  b3GpuSolverBody* body2,  b3GpuSolverConstraint* c)
{
	float deltaImpulse = c->m_rhs-b3Scalar(c->m_appliedImpulse)*c->m_cfm;
	float deltaVel1Dotn	=	b3Dot(c->m_contactNormal,body1->m_deltaLinearVelocity) 	+ b3Dot(c->m_relpos1CrossNormal,body1->m_deltaAngularVelocity);
	float deltaVel2Dotn	=	-b3Dot(c->m_contactNormal,body2->m_deltaLinearVelocity) + b3Dot(c->m_relpos2CrossNormal,body2->m_deltaAngularVelocity);

	deltaImpulse	-=	deltaVel1Dotn*c->m_jacDiagABInv;
	deltaImpulse	-=	deltaVel2Dotn*c->m_jacDiagABInv;

	float sum = b3Scalar(c->m_appliedImpulse) + deltaImpulse;
	if (sum < c->m_lowerLimit)
	{
		deltaImpulse = c->m_lowerLimit-b3Scalar(c->m_appliedImpulse);
		c->m_appliedImpulse = c->m_lowerLimit;
	}
	else if (sum > c->m_upperLimit) 
	{
		deltaImpulse = c->m_upperLimit-b3Scalar(c->m_appliedImpulse);
		c->m_appliedImpulse = c->m_upperLimit;
	}
	else
	{
		c->m_appliedImpulse = sum;
	}

	internalApplyImpulse(body1,c->m_contactNormal*body1->m_invMass,c->m_angularComponentA,deltaImpulse);
	internalApplyImpulse(body2,-c->m_contactNormal*body2->m_invMass,c->m_angularComponentB,deltaImpulse);

}



void	b3GpuPgsConstraintSolver::initSolverBody(int bodyIndex, b3GpuSolverBody* solverBody, b3RigidBodyData* rb)
{

	solverBody->m_deltaLinearVelocity.setValue(0.f,0.f,0.f);
	solverBody->m_deltaAngularVelocity.setValue(0.f,0.f,0.f);
	solverBody->internalGetPushVelocity().setValue(0.f,0.f,0.f);
	solverBody->internalGetTurnVelocity().setValue(0.f,0.f,0.f);

	b3Assert(rb);
//	solverBody->m_worldTransform = getWorldTransform(rb);
	solverBody->internalSetInvMass(b3MakeVector3(rb->m_invMass,rb->m_invMass,rb->m_invMass));
	solverBody->m_originalBodyIndex = bodyIndex;
	solverBody->m_angularFactor = b3MakeVector3(1,1,1);
	solverBody->m_linearFactor = b3MakeVector3(1,1,1);
	solverBody->m_linearVelocity = getLinearVelocity(rb);
	solverBody->m_angularVelocity = getAngularVelocity(rb);
}


void	b3GpuPgsConstraintSolver::averageVelocities()
{
}


b3Scalar b3GpuPgsConstraintSolver::solveGroupCacheFriendlyIterations(b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints1,int numConstraints,const b3ContactSolverInfo& infoGlobal)
{
	//only create the batches once.
	//@todo: incrementally update batches when constraints are added/activated and/or removed/deactivated
	B3_PROFILE("GpuSolveGroupCacheFriendlyIterations");

	bool createBatches = m_gpuData->m_batchSizes.size()==0;
	{
		
		if (createBatches)
		{
			
			m_gpuData->m_batchSizes.resize(0);

			{
				m_gpuData->m_gpuBatchConstraints->copyToHost(batchConstraints);

				B3_PROFILE("batch joints");
				b3Assert(batchConstraints.size()==numConstraints);
				int simdWidth =numConstraints+1;
				int numBodies = m_tmpSolverBodyPool.size();
				sortConstraintByBatch3( &batchConstraints[0], numConstraints, simdWidth , m_staticIdx,  numBodies);

				m_gpuData->m_gpuBatchConstraints->copyFromHost(batchConstraints);

			}
		} else
		{
			/*b3AlignedObjectArray<b3BatchConstraint> cpuCheckBatches;
			m_gpuData->m_gpuBatchConstraints->copyToHost(cpuCheckBatches);
			b3Assert(cpuCheckBatches.size()==batchConstraints.size());
			printf(".\n");
			*/
			//>copyFromHost(batchConstraints);
		}
		int maxIterations = infoGlobal.m_numIterations;
	
		bool useBatching = true;

		if (useBatching )
		{
			
			if (!useGpuSolveJointConstraintRows)
			{
				B3_PROFILE("copy to host");
				m_gpuData->m_gpuSolverBodies->copyToHost(m_tmpSolverBodyPool);
				m_gpuData->m_gpuBatchConstraints->copyToHost(batchConstraints);
				m_gpuData->m_gpuConstraintRows->copyToHost(m_tmpSolverNonContactConstraintPool);
				m_gpuData->m_gpuConstraintInfo1->copyToHost(m_gpuData->m_cpuConstraintInfo1);
				m_gpuData->m_gpuConstraintRowOffsets->copyToHost(m_gpuData->m_cpuConstraintRowOffsets);
				gpuConstraints1->copyToHost(m_gpuData->m_cpuConstraints);

			}

			for ( int iteration = 0 ; iteration< maxIterations ; iteration++)
			{
				
				int batchOffset = 0;
				int constraintOffset=0;
				int numBatches = m_gpuData->m_batchSizes.size();
				for (int bb=0;bb<numBatches;bb++)
				{
					int numConstraintsInBatch = m_gpuData->m_batchSizes[bb];

					
					if (useGpuSolveJointConstraintRows)
					{
						B3_PROFILE("solveJointConstraintRowsKernels");
					
						/*
						__kernel void solveJointConstraintRows(__global b3GpuSolverBody* solverBodies,
					  __global b3BatchConstraint* batchConstraints,
					  	__global b3SolverConstraint* rows,
						__global unsigned int* numConstraintRowsInfo1, 
						__global unsigned int* rowOffsets,
						__global b3GpuGenericConstraint* constraints,
						int batchOffset,
						int numConstraintsInBatch*/


						b3LauncherCL launcher(m_gpuData->m_queue,m_gpuData->m_solveJointConstraintRowsKernels,"m_solveJointConstraintRowsKernels");
						launcher.setBuffer(m_gpuData->m_gpuSolverBodies->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuBatchConstraints->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuConstraintRows->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuConstraintInfo1->getBufferCL());
						launcher.setBuffer(m_gpuData->m_gpuConstraintRowOffsets->getBufferCL());
						launcher.setBuffer(gpuConstraints1->getBufferCL());//to detect disabled constraints
						launcher.setConst(batchOffset);
						launcher.setConst(numConstraintsInBatch);

						launcher.launch1D(numConstraintsInBatch);
						

					} else//useGpu
					{
						

						
						for (int b=0;b<numConstraintsInBatch;b++)
						{
							const b3BatchConstraint& c = batchConstraints[batchOffset+b];
							/*printf("-----------\n");
							printf("bb=%d\n",bb);
							printf("c.batchId = %d\n", c.m_batchId);
							*/
							b3Assert(c.m_batchId==bb);
							b3GpuGenericConstraint* constraint = &m_gpuData->m_cpuConstraints[c.m_originalConstraintIndex];
							if (constraint->m_flags&B3_CONSTRAINT_FLAG_ENABLED)
							{
								int numConstraintRows = m_gpuData->m_cpuConstraintInfo1[c.m_originalConstraintIndex];
								int constraintOffset = m_gpuData->m_cpuConstraintRowOffsets[c.m_originalConstraintIndex];
								
								for (int jj=0;jj<numConstraintRows;jj++)
								{
	//							
									b3GpuSolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[constraintOffset+jj];
									//resolveSingleConstraintRowGenericSIMD(m_tmpSolverBodyPool[constraint.m_solverBodyIdA],m_tmpSolverBodyPool[constraint.m_solverBodyIdB],constraint);
									resolveSingleConstraintRowGeneric2(&m_tmpSolverBodyPool[constraint.m_solverBodyIdA],&m_tmpSolverBodyPool[constraint.m_solverBodyIdB],&constraint);
								}
							}
						}
					}//useGpu
					batchOffset+=numConstraintsInBatch;
					constraintOffset+=numConstraintsInBatch;
				}
			}//for (int iteration...

			if (!useGpuSolveJointConstraintRows)
			{
				{
					B3_PROFILE("copy from host");
					m_gpuData->m_gpuSolverBodies->copyFromHost(m_tmpSolverBodyPool);
					m_gpuData->m_gpuBatchConstraints->copyFromHost(batchConstraints);
					m_gpuData->m_gpuConstraintRows->copyFromHost(m_tmpSolverNonContactConstraintPool);
				}

				//B3_PROFILE("copy to host");
				//m_gpuData->m_gpuSolverBodies->copyToHost(m_tmpSolverBodyPool);
			}
			//int sz = sizeof(b3GpuSolverBody);
			//printf("cpu sizeof(b3GpuSolverBody)=%d\n",sz);

			
			


		} else
		{
			for ( int iteration = 0 ; iteration< maxIterations ; iteration++)
			{			
				int numJoints =			m_tmpSolverNonContactConstraintPool.size();
				for (int j=0;j<numJoints;j++)
				{
					b3GpuSolverConstraint& constraint = m_tmpSolverNonContactConstraintPool[j];
					resolveSingleConstraintRowGeneric2(&m_tmpSolverBodyPool[constraint.m_solverBodyIdA],&m_tmpSolverBodyPool[constraint.m_solverBodyIdB],&constraint);
				}

				if (!m_usePgs)
				{
					averageVelocities();
				}
			}
		}
		
	}
	clFinish(m_gpuData->m_queue);
	return 0.f;
}




static b3AlignedObjectArray<int> bodyUsed;
static b3AlignedObjectArray<int> curUsed;



inline int b3GpuPgsConstraintSolver::sortConstraintByBatch3( b3BatchConstraint* cs, int numConstraints, int simdWidth , int staticIdx, int numBodies)
{
	//int sz = sizeof(b3BatchConstraint);

	B3_PROFILE("sortConstraintByBatch3");
	
	static int maxSwaps = 0;
	int numSwaps = 0;

	curUsed.resize(2*simdWidth);

	static int maxNumConstraints = 0;
	if (maxNumConstraints<numConstraints)
	{
		maxNumConstraints = numConstraints;
		//printf("maxNumConstraints  = %d\n",maxNumConstraints );
	}

	int numUsedArray = numBodies/32+1;
	bodyUsed.resize(numUsedArray);

	for (int q=0;q<numUsedArray;q++)
		bodyUsed[q]=0;

	
	int curBodyUsed = 0;

	int numIter = 0;
    
		
#if defined(_DEBUG)
	for(int i=0; i<numConstraints; i++)
		cs[i].m_batchId = -1;
#endif
	
	int numValidConstraints = 0;
//	int unprocessedConstraintIndex = 0;

	int batchIdx = 0;
    

	{
		B3_PROFILE("cpu batch innerloop");
		
		while( numValidConstraints < numConstraints)
		{
			numIter++;
			int nCurrentBatch = 0;
			//	clear flag
			for(int i=0; i<curBodyUsed; i++) 
				bodyUsed[curUsed[i]/32] = 0;

            curBodyUsed = 0;

			for(int i=numValidConstraints; i<numConstraints; i++)
			{
				int idx = i;
				b3Assert( idx < numConstraints );
				//	check if it can go
				int bodyAS = cs[idx].m_bodyAPtrAndSignBit;
				int bodyBS = cs[idx].m_bodyBPtrAndSignBit;
				int bodyA = abs(bodyAS);
				int bodyB = abs(bodyBS);
				bool aIsStatic = (bodyAS<0) || bodyAS==staticIdx;
				bool bIsStatic = (bodyBS<0) || bodyBS==staticIdx;
				int aUnavailable = 0;
				int bUnavailable = 0;
				if (!aIsStatic)
				{
					aUnavailable = bodyUsed[ bodyA/32 ] & (1<<(bodyA&31));
				}
				if (!aUnavailable)
				if (!bIsStatic)
				{
					bUnavailable = bodyUsed[ bodyB/32 ] & (1<<(bodyB&31));
				}
                
				if( aUnavailable==0 && bUnavailable==0 ) // ok
				{
					if (!aIsStatic)
					{
						bodyUsed[ bodyA/32 ] |= (1<<(bodyA&31));
						curUsed[curBodyUsed++]=bodyA;
					}
					if (!bIsStatic)
					{
						bodyUsed[ bodyB/32 ] |= (1<<(bodyB&31));
						curUsed[curBodyUsed++]=bodyB;
					}

					cs[idx].m_batchId = batchIdx;

					if (i!=numValidConstraints)
					{
						b3Swap(cs[i],cs[numValidConstraints]);
						numSwaps++;
					}

					numValidConstraints++;
					{
						nCurrentBatch++;
						if( nCurrentBatch == simdWidth )
						{
							nCurrentBatch = 0;
							for(int i=0; i<curBodyUsed; i++) 
								bodyUsed[curUsed[i]/32] = 0;
							curBodyUsed = 0;
						}
					}
				}
			}
			m_gpuData->m_batchSizes.push_back(nCurrentBatch);
			batchIdx ++;
		}
	}
	
#if defined(_DEBUG)
    //		debugPrintf( "nBatches: %d\n", batchIdx );
	for(int i=0; i<numConstraints; i++)
    {
        b3Assert( cs[i].m_batchId != -1 );
    }
#endif

	if (maxSwaps<numSwaps)
	{
		maxSwaps = numSwaps;
		//printf("maxSwaps = %d\n", maxSwaps);
	}
	
	return batchIdx;
}


/// b3PgsJacobiSolver Sequentially applies impulses
b3Scalar b3GpuPgsConstraintSolver::solveGroup(b3OpenCLArray<b3RigidBodyData>* gpuBodies, b3OpenCLArray<b3InertiaData>* gpuInertias, 
				int numBodies, b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints,int numConstraints, const b3ContactSolverInfo& infoGlobal)
{

	B3_PROFILE("solveJoints");
	//you need to provide at least some bodies
	
	solveGroupCacheFriendlySetup( gpuBodies, gpuInertias,numBodies,gpuConstraints, numConstraints,infoGlobal);

	solveGroupCacheFriendlyIterations(gpuConstraints, numConstraints,infoGlobal);

	solveGroupCacheFriendlyFinish(gpuBodies, gpuInertias,numBodies, gpuConstraints, numConstraints, infoGlobal);
	
	return 0.f;
}

void	b3GpuPgsConstraintSolver::solveJoints(int numBodies, b3OpenCLArray<b3RigidBodyData>* gpuBodies, b3OpenCLArray<b3InertiaData>* gpuInertias, 
				int numConstraints, b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints)
{
	b3ContactSolverInfo infoGlobal;
	infoGlobal.m_splitImpulse = false;
	infoGlobal.m_timeStep = 1.f/60.f;
	infoGlobal.m_numIterations = 4;//4;
//	infoGlobal.m_solverMode|=B3_SOLVER_USE_2_FRICTION_DIRECTIONS|B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS|B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION;
	//infoGlobal.m_solverMode|=B3_SOLVER_USE_2_FRICTION_DIRECTIONS|B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS;
	infoGlobal.m_solverMode|=B3_SOLVER_USE_2_FRICTION_DIRECTIONS;

	//if (infoGlobal.m_solverMode & B3_SOLVER_INTERLEAVE_CONTACT_AND_FRICTION_CONSTRAINTS)
	//if ((infoGlobal.m_solverMode & B3_SOLVER_USE_2_FRICTION_DIRECTIONS) && (infoGlobal.m_solverMode & B3_SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION))
				

	solveGroup(gpuBodies,gpuInertias,numBodies,gpuConstraints,numConstraints,infoGlobal);

}

//b3AlignedObjectArray<b3RigidBodyData> testBodies;


b3Scalar b3GpuPgsConstraintSolver::solveGroupCacheFriendlyFinish(b3OpenCLArray<b3RigidBodyData>* gpuBodies,b3OpenCLArray<b3InertiaData>* gpuInertias,int numBodies,b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints,int numConstraints,const b3ContactSolverInfo& infoGlobal)
{
	B3_PROFILE("solveGroupCacheFriendlyFinish");
//	int numPoolConstraints = m_tmpSolverContactConstraintPool.size();
//	int i,j;


	{
		if (gpuBreakConstraints)
		{
			B3_PROFILE("breakViolatedConstraintsKernel");
			b3LauncherCL launcher(m_gpuData->m_queue,m_gpuData->m_breakViolatedConstraintsKernel,"m_breakViolatedConstraintsKernel");
			launcher.setBuffer(gpuConstraints->getBufferCL());
			launcher.setBuffer(m_gpuData->m_gpuConstraintInfo1->getBufferCL());
			launcher.setBuffer(m_gpuData->m_gpuConstraintRowOffsets->getBufferCL());
			launcher.setBuffer(m_gpuData->m_gpuConstraintRows->getBufferCL());
			launcher.setConst(numConstraints);
			launcher.launch1D(numConstraints);
		} else
		{
			gpuConstraints->copyToHost(m_gpuData->m_cpuConstraints);
			m_gpuData->m_gpuBatchConstraints->copyToHost(m_gpuData->m_cpuBatchConstraints);
			m_gpuData->m_gpuConstraintRows->copyToHost(m_gpuData->m_cpuConstraintRows);
			gpuConstraints->copyToHost(m_gpuData->m_cpuConstraints);
			m_gpuData->m_gpuConstraintInfo1->copyToHost(m_gpuData->m_cpuConstraintInfo1);
			m_gpuData->m_gpuConstraintRowOffsets->copyToHost(m_gpuData->m_cpuConstraintRowOffsets);

			for (int cid=0;cid<numConstraints;cid++)
			{
				int originalConstraintIndex = batchConstraints[cid].m_originalConstraintIndex;
				int constraintRowOffset = m_gpuData->m_cpuConstraintRowOffsets[originalConstraintIndex];
				int numRows = m_gpuData->m_cpuConstraintInfo1[originalConstraintIndex];
				if (numRows)
				{
		
				//	printf("cid=%d, breakingThreshold =%f\n",cid,breakingThreshold);
					for (int i=0;i<numRows;i++)
					{
						int rowIndex =constraintRowOffset+i;
						int orgConstraintIndex = m_gpuData->m_cpuConstraintRows[rowIndex].m_originalConstraintIndex;
						float breakingThreshold = m_gpuData->m_cpuConstraints[orgConstraintIndex].m_breakingImpulseThreshold;
					//	printf("rows[%d].m_appliedImpulse=%f\n",rowIndex,rows[rowIndex].m_appliedImpulse);
						if (b3Fabs(m_gpuData->m_cpuConstraintRows[rowIndex].m_appliedImpulse) >= breakingThreshold)
						{

							m_gpuData->m_cpuConstraints[orgConstraintIndex].m_flags =0;//&= ~B3_CONSTRAINT_FLAG_ENABLED;
						}
					}
				}
			}


			gpuConstraints->copyFromHost(m_gpuData->m_cpuConstraints);
		}
	}

	{
		if (useGpuWriteBackVelocities)
		{
			B3_PROFILE("GPU write back velocities and transforms");

			b3LauncherCL launcher(m_gpuData->m_queue,m_gpuData->m_writeBackVelocitiesKernel,"m_writeBackVelocitiesKernel");
			launcher.setBuffer(gpuBodies->getBufferCL());
			launcher.setBuffer(m_gpuData->m_gpuSolverBodies->getBufferCL());
			launcher.setConst(numBodies);
			launcher.launch1D(numBodies);
			clFinish(m_gpuData->m_queue);
//			m_gpuData->m_gpuSolverBodies->copyToHost(m_tmpSolverBodyPool);
//			m_gpuData->m_gpuBodies->copyToHostPointer(bodies,numBodies);
			//m_gpuData->m_gpuBodies->copyToHost(testBodies);

		} 
		else
		{
			B3_PROFILE("CPU write back velocities and transforms");

			m_gpuData->m_gpuSolverBodies->copyToHost(m_tmpSolverBodyPool);
			gpuBodies->copyToHost(m_gpuData->m_cpuBodies);
			for ( int i=0;i<m_tmpSolverBodyPool.size();i++)
			{
				int bodyIndex = m_tmpSolverBodyPool[i].m_originalBodyIndex;
				//printf("bodyIndex=%d\n",bodyIndex);
				b3Assert(i==bodyIndex);

				b3RigidBodyData* body = &m_gpuData->m_cpuBodies[bodyIndex];
				if (body->m_invMass)
				{
					if (infoGlobal.m_splitImpulse)
						m_tmpSolverBodyPool[i].writebackVelocityAndTransform(infoGlobal.m_timeStep, infoGlobal.m_splitImpulseTurnErp);
					else
						m_tmpSolverBodyPool[i].writebackVelocity();

					if (m_usePgs)
					{
						body->m_linVel = m_tmpSolverBodyPool[i].m_linearVelocity;
						body->m_angVel = m_tmpSolverBodyPool[i].m_angularVelocity;
					} else
					{
						b3Assert(0);
					}
	/*			
					if (infoGlobal.m_splitImpulse)
					{
						body->m_pos = m_tmpSolverBodyPool[i].m_worldTransform.getOrigin();
						b3Quaternion orn;
						orn = m_tmpSolverBodyPool[i].m_worldTransform.getRotation();
						body->m_quat = orn;
					}
					*/
				}
			}//for

			gpuBodies->copyFromHost(m_gpuData->m_cpuBodies);

		}
	}

	clFinish(m_gpuData->m_queue);

	m_tmpSolverContactConstraintPool.resizeNoInitialize(0);
	m_tmpSolverNonContactConstraintPool.resizeNoInitialize(0);
	m_tmpSolverContactFrictionConstraintPool.resizeNoInitialize(0);
	m_tmpSolverContactRollingFrictionConstraintPool.resizeNoInitialize(0);

	m_tmpSolverBodyPool.resizeNoInitialize(0);
	return 0.f;
}
