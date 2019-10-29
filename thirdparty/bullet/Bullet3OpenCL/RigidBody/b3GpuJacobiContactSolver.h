
#ifndef B3_GPU_JACOBI_CONTACT_SOLVER_H
#define B3_GPU_JACOBI_CONTACT_SOLVER_H
#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"
//#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3OpenCLArray.h"

//struct b3InertiaData;
//b3InertiaData

class b3TypedConstraint;

struct b3JacobiSolverInfo
{
	int m_fixedBodyIndex;

	float m_deltaTime;
	float m_positionDrift;
	float m_positionConstraintCoeff;
	int m_numIterations;

	b3JacobiSolverInfo()
		: m_fixedBodyIndex(0),
		  m_deltaTime(1. / 60.f),
		  m_positionDrift(0.005f),
		  m_positionConstraintCoeff(0.99f),
		  m_numIterations(7)
	{
	}
};
class b3GpuJacobiContactSolver
{
protected:
	struct b3GpuJacobiSolverInternalData* m_data;

	cl_context m_context;
	cl_device_id m_device;
	cl_command_queue m_queue;

public:
	b3GpuJacobiContactSolver(cl_context ctx, cl_device_id device, cl_command_queue queue, int pairCapacity);
	virtual ~b3GpuJacobiContactSolver();

	void solveContacts(int numBodies, cl_mem bodyBuf, cl_mem inertiaBuf, int numContacts, cl_mem contactBuf, const struct b3Config& config, int static0Index);
	void solveGroupHost(b3RigidBodyData* bodies, b3InertiaData* inertias, int numBodies, struct b3Contact4* manifoldPtr, int numManifolds, const b3JacobiSolverInfo& solverInfo);
	//void  solveGroupHost(btRigidBodyCL* bodies,b3InertiaData* inertias,int numBodies,btContact4* manifoldPtr, int numManifolds,btTypedConstraint** constraints,int numConstraints,const btJacobiSolverInfo& solverInfo);

	//b3Scalar solveGroup(b3OpenCLArray<b3RigidBodyData>* gpuBodies,b3OpenCLArray<b3InertiaData>* gpuInertias, int numBodies,b3OpenCLArray<b3GpuGenericConstraint>* gpuConstraints,int numConstraints,const b3ContactSolverInfo& infoGlobal);

	//void  solveGroup(btOpenCLArray<btRigidBodyCL>* bodies,btOpenCLArray<btInertiaCL>* inertias,btOpenCLArray<btContact4>* manifoldPtr,const btJacobiSolverInfo& solverInfo);
	//void  solveGroupMixed(btOpenCLArray<btRigidBodyCL>* bodies,btOpenCLArray<btInertiaCL>* inertias,btOpenCLArray<btContact4>* manifoldPtr,const btJacobiSolverInfo& solverInfo);
};
#endif  //B3_GPU_JACOBI_CONTACT_SOLVER_H
