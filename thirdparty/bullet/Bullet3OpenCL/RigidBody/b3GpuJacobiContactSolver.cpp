
#include "b3GpuJacobiContactSolver.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Contact4.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3FillCL.h" //b3Int2
class b3Vector3;
#include "Bullet3OpenCL/ParallelPrimitives/b3RadixSort32CL.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3PrefixScanCL.h"
#include "Bullet3OpenCL/ParallelPrimitives/b3LauncherCL.h"
#include "Bullet3OpenCL/Initialize/b3OpenCLUtils.h"
#include "Bullet3OpenCL/RigidBody/kernels/solverUtils.h"
#include "Bullet3Common/b3Logging.h"
#include "b3GpuConstraint4.h"
#include "Bullet3Common/shared/b3Int2.h"
#include "Bullet3Common/shared/b3Int4.h"
#define SOLVER_UTILS_KERNEL_PATH "src/Bullet3OpenCL/RigidBody/kernels/solverUtils.cl"


struct b3GpuJacobiSolverInternalData
{
		//btRadixSort32CL*	m_sort32;
		//btBoundSearchCL*	m_search;
		b3PrefixScanCL*	m_scan;

		b3OpenCLArray<unsigned int>* m_bodyCount;
		b3OpenCLArray<b3Int2>*		m_contactConstraintOffsets;
		b3OpenCLArray<unsigned int>* m_offsetSplitBodies;

		b3OpenCLArray<b3Vector3>*	m_deltaLinearVelocities;
		b3OpenCLArray<b3Vector3>*	m_deltaAngularVelocities;

		b3AlignedObjectArray<b3Vector3>	m_deltaLinearVelocitiesCPU;
		b3AlignedObjectArray<b3Vector3>	m_deltaAngularVelocitiesCPU;



		b3OpenCLArray<b3GpuConstraint4>* m_contactConstraints;

		b3FillCL*	m_filler;
		

		cl_kernel	m_countBodiesKernel;
		cl_kernel	m_contactToConstraintSplitKernel;
		cl_kernel	m_clearVelocitiesKernel;
		cl_kernel	m_averageVelocitiesKernel;
		cl_kernel	m_updateBodyVelocitiesKernel;
		cl_kernel	m_solveContactKernel;
		cl_kernel	m_solveFrictionKernel;



};


b3GpuJacobiContactSolver::b3GpuJacobiContactSolver(cl_context ctx, cl_device_id device, cl_command_queue queue, int pairCapacity)
	:m_context(ctx),
	m_device(device),
	m_queue(queue)
{
	m_data = new b3GpuJacobiSolverInternalData;
	m_data->m_scan = new b3PrefixScanCL(m_context,m_device,m_queue);
	m_data->m_bodyCount = new b3OpenCLArray<unsigned int>(m_context,m_queue);
	m_data->m_filler = new b3FillCL(m_context,m_device,m_queue);
	m_data->m_contactConstraintOffsets = new b3OpenCLArray<b3Int2>(m_context,m_queue);
	m_data->m_offsetSplitBodies = new b3OpenCLArray<unsigned int>(m_context,m_queue);
	m_data->m_contactConstraints = new b3OpenCLArray<b3GpuConstraint4>(m_context,m_queue);
	m_data->m_deltaLinearVelocities = new b3OpenCLArray<b3Vector3>(m_context,m_queue);
	m_data->m_deltaAngularVelocities = new b3OpenCLArray<b3Vector3>(m_context,m_queue);

	cl_int pErrNum;
	const char* additionalMacros="";
	const char* solverUtilsSource = solverUtilsCL;
	{
		cl_program solverUtilsProg= b3OpenCLUtils::compileCLProgramFromString( ctx, device, solverUtilsSource, &pErrNum,additionalMacros, SOLVER_UTILS_KERNEL_PATH);
		b3Assert(solverUtilsProg);
		m_data->m_countBodiesKernel =  b3OpenCLUtils::compileCLKernelFromString( ctx, device, solverUtilsSource, "CountBodiesKernel", &pErrNum, solverUtilsProg,additionalMacros );
		b3Assert(m_data->m_countBodiesKernel);

		m_data->m_contactToConstraintSplitKernel  = b3OpenCLUtils::compileCLKernelFromString( ctx, device, solverUtilsSource, "ContactToConstraintSplitKernel", &pErrNum, solverUtilsProg,additionalMacros );
		b3Assert(m_data->m_contactToConstraintSplitKernel);
		m_data->m_clearVelocitiesKernel  = b3OpenCLUtils::compileCLKernelFromString( ctx, device, solverUtilsSource, "ClearVelocitiesKernel", &pErrNum, solverUtilsProg,additionalMacros );
		b3Assert(m_data->m_clearVelocitiesKernel);

		m_data->m_averageVelocitiesKernel  = b3OpenCLUtils::compileCLKernelFromString( ctx, device, solverUtilsSource, "AverageVelocitiesKernel", &pErrNum, solverUtilsProg,additionalMacros );
		b3Assert(m_data->m_averageVelocitiesKernel);

		m_data->m_updateBodyVelocitiesKernel = b3OpenCLUtils::compileCLKernelFromString( ctx, device, solverUtilsSource, "UpdateBodyVelocitiesKernel", &pErrNum, solverUtilsProg,additionalMacros );
		b3Assert(m_data->m_updateBodyVelocitiesKernel);

		
		m_data->m_solveContactKernel = b3OpenCLUtils::compileCLKernelFromString( ctx, device, solverUtilsSource, "SolveContactJacobiKernel", &pErrNum, solverUtilsProg,additionalMacros );
		b3Assert(m_data->m_solveContactKernel );

		m_data->m_solveFrictionKernel = b3OpenCLUtils::compileCLKernelFromString( ctx, device, solverUtilsSource, "SolveFrictionJacobiKernel", &pErrNum, solverUtilsProg,additionalMacros );
		b3Assert(m_data->m_solveFrictionKernel);
	}

}


b3GpuJacobiContactSolver::~b3GpuJacobiContactSolver()
{
	clReleaseKernel(m_data->m_solveContactKernel);
	clReleaseKernel(m_data->m_solveFrictionKernel);
	clReleaseKernel(m_data->m_countBodiesKernel);
	clReleaseKernel(m_data->m_contactToConstraintSplitKernel);
	clReleaseKernel(m_data->m_averageVelocitiesKernel);
	clReleaseKernel(m_data->m_updateBodyVelocitiesKernel);
	clReleaseKernel(m_data->m_clearVelocitiesKernel );

	delete m_data->m_deltaLinearVelocities;
	delete m_data->m_deltaAngularVelocities;
	delete m_data->m_contactConstraints;
	delete m_data->m_offsetSplitBodies;
	delete m_data->m_contactConstraintOffsets;
	delete m_data->m_bodyCount;
	delete m_data->m_filler;
	delete m_data->m_scan;
	delete m_data;
}



b3Vector3 make_float4(float v)
{
	return b3MakeVector3 (v,v,v);
}

b3Vector4 make_float4(float x,float y, float z, float w)
{
	return b3MakeVector4 (x,y,z,w);
}


	static
	inline
	float calcRelVel(const b3Vector3& l0, const b3Vector3& l1, const b3Vector3& a0, const b3Vector3& a1, 
					 const b3Vector3& linVel0, const b3Vector3& angVel0, const b3Vector3& linVel1, const b3Vector3& angVel1)
	{
		return b3Dot(l0, linVel0) + b3Dot(a0, angVel0) + b3Dot(l1, linVel1) + b3Dot(a1, angVel1);
	}


	static
	inline
	void setLinearAndAngular(const b3Vector3& n, const b3Vector3& r0, const b3Vector3& r1,
							 b3Vector3& linear, b3Vector3& angular0, b3Vector3& angular1)
	{
		linear = n;
		angular0 = b3Cross(r0, n);
		angular1 = -b3Cross(r1, n);
	}


static __inline void solveContact(b3GpuConstraint4& cs, 
	const b3Vector3& posA, const b3Vector3& linVelARO, const b3Vector3& angVelARO, float invMassA, const b3Matrix3x3& invInertiaA,
	const b3Vector3& posB, const b3Vector3& linVelBRO, const b3Vector3& angVelBRO, float invMassB, const b3Matrix3x3& invInertiaB, 
	float maxRambdaDt[4], float minRambdaDt[4], b3Vector3& dLinVelA, b3Vector3& dAngVelA, b3Vector3& dLinVelB, b3Vector3& dAngVelB)
{


	for(int ic=0; ic<4; ic++)
	{
		//	dont necessary because this makes change to 0
		if( cs.m_jacCoeffInv[ic] == 0.f ) continue;

		{
			b3Vector3 angular0, angular1, linear;
			b3Vector3 r0 = cs.m_worldPos[ic] - (b3Vector3&)posA;
			b3Vector3 r1 = cs.m_worldPos[ic] - (b3Vector3&)posB;
			setLinearAndAngular( (const b3Vector3 &)cs.m_linear, (const b3Vector3 &)r0, (const b3Vector3 &)r1, linear, angular0, angular1 );

			float rambdaDt = calcRelVel((const b3Vector3 &)cs.m_linear,(const b3Vector3 &) -cs.m_linear, angular0, angular1,
				linVelARO+dLinVelA, angVelARO+dAngVelA, linVelBRO+dLinVelB, angVelBRO+dAngVelB ) + cs.m_b[ic];
			rambdaDt *= cs.m_jacCoeffInv[ic];

			{
				float prevSum = cs.m_appliedRambdaDt[ic];
				float updated = prevSum;
				updated += rambdaDt;
				updated = b3Max( updated, minRambdaDt[ic] );
				updated = b3Min( updated, maxRambdaDt[ic] );
				rambdaDt = updated - prevSum;
				cs.m_appliedRambdaDt[ic] = updated;
			}

			b3Vector3 linImp0 = invMassA*linear*rambdaDt;
			b3Vector3 linImp1 = invMassB*(-linear)*rambdaDt;
			b3Vector3 angImp0 = (invInertiaA* angular0)*rambdaDt;
			b3Vector3 angImp1 = (invInertiaB* angular1)*rambdaDt;
#ifdef _WIN32
            b3Assert(_finite(linImp0.getX()));
			b3Assert(_finite(linImp1.getX()));
#endif
			
			if (invMassA)
			{
				dLinVelA += linImp0;
				dAngVelA += angImp0;
			}
			if (invMassB)
			{
				dLinVelB += linImp1;
				dAngVelB += angImp1;
			}
		}
	}
}



void solveContact3(b3GpuConstraint4* cs,
			b3Vector3* posAPtr, b3Vector3* linVelA, b3Vector3* angVelA, float invMassA, const b3Matrix3x3& invInertiaA,
			b3Vector3* posBPtr, b3Vector3* linVelB, b3Vector3* angVelB, float invMassB, const b3Matrix3x3& invInertiaB,
			b3Vector3* dLinVelA, b3Vector3* dAngVelA, b3Vector3* dLinVelB, b3Vector3* dAngVelB)
{
	float minRambdaDt = 0;
	float maxRambdaDt = FLT_MAX;

	for(int ic=0; ic<4; ic++)
	{
		if( cs->m_jacCoeffInv[ic] == 0.f ) continue;

		b3Vector3 angular0, angular1, linear;
		b3Vector3 r0 = cs->m_worldPos[ic] - *posAPtr;
		b3Vector3 r1 = cs->m_worldPos[ic] - *posBPtr;
		setLinearAndAngular( cs->m_linear, r0, r1, linear, angular0, angular1 );

		float rambdaDt = calcRelVel( cs->m_linear, -cs->m_linear, angular0, angular1, 
			*linVelA+*dLinVelA, *angVelA+*dAngVelA, *linVelB+*dLinVelB, *angVelB+*dAngVelB ) + cs->m_b[ic];
		rambdaDt *= cs->m_jacCoeffInv[ic];

		{
			float prevSum = cs->m_appliedRambdaDt[ic];
			float updated = prevSum;
			updated += rambdaDt;
			updated = b3Max( updated, minRambdaDt );
			updated = b3Min( updated, maxRambdaDt );
			rambdaDt = updated - prevSum;
			cs->m_appliedRambdaDt[ic] = updated;
		}

		b3Vector3 linImp0 = invMassA*linear*rambdaDt;
		b3Vector3 linImp1 = invMassB*(-linear)*rambdaDt;
		b3Vector3 angImp0 = (invInertiaA* angular0)*rambdaDt;
		b3Vector3 angImp1 = (invInertiaB* angular1)*rambdaDt;

		if (invMassA)
		{
			*dLinVelA += linImp0;
			*dAngVelA += angImp0;
		}
		if (invMassB)
		{
			*dLinVelB += linImp1;
			*dAngVelB += angImp1;
		}
	}
}


static inline void solveFriction(b3GpuConstraint4& cs, 
	const b3Vector3& posA, const b3Vector3& linVelARO, const b3Vector3& angVelARO, float invMassA, const b3Matrix3x3& invInertiaA,
	const b3Vector3& posB, const b3Vector3& linVelBRO, const b3Vector3& angVelBRO, float invMassB, const b3Matrix3x3& invInertiaB, 
	float maxRambdaDt[4], float minRambdaDt[4], b3Vector3& dLinVelA, b3Vector3& dAngVelA, b3Vector3& dLinVelB, b3Vector3& dAngVelB)
{

	b3Vector3 linVelA = linVelARO+dLinVelA;
	b3Vector3 linVelB = linVelBRO+dLinVelB;
	b3Vector3 angVelA = angVelARO+dAngVelA;
	b3Vector3 angVelB = angVelBRO+dAngVelB;

	if( cs.m_fJacCoeffInv[0] == 0 && cs.m_fJacCoeffInv[0] == 0 ) return;
	const b3Vector3& center = (const b3Vector3&)cs.m_center;

	b3Vector3 n = -(const b3Vector3&)cs.m_linear;

	b3Vector3 tangent[2];
#if 1		
	b3PlaneSpace1 (n, tangent[0],tangent[1]);
#else
	b3Vector3 r = cs.m_worldPos[0]-center;
	tangent[0] = cross3( n, r );
	tangent[1] = cross3( tangent[0], n );
	tangent[0] = normalize3( tangent[0] );
	tangent[1] = normalize3( tangent[1] );
#endif

	b3Vector3 angular0, angular1, linear;
	b3Vector3 r0 = center - posA;
	b3Vector3 r1 = center - posB;
	for(int i=0; i<2; i++)
	{
		setLinearAndAngular( tangent[i], r0, r1, linear, angular0, angular1 );
		float rambdaDt = calcRelVel(linear, -linear, angular0, angular1,
			linVelA, angVelA, linVelB, angVelB );
		rambdaDt *= cs.m_fJacCoeffInv[i];

			{
				float prevSum = cs.m_fAppliedRambdaDt[i];
				float updated = prevSum;
				updated += rambdaDt;
				updated = b3Max( updated, minRambdaDt[i] );
				updated = b3Min( updated, maxRambdaDt[i] );
				rambdaDt = updated - prevSum;
				cs.m_fAppliedRambdaDt[i] = updated;
			}

		b3Vector3 linImp0 = invMassA*linear*rambdaDt;
		b3Vector3 linImp1 = invMassB*(-linear)*rambdaDt;
		b3Vector3 angImp0 = (invInertiaA* angular0)*rambdaDt;
		b3Vector3 angImp1 = (invInertiaB* angular1)*rambdaDt;
#ifdef _WIN32
		b3Assert(_finite(linImp0.getX()));
		b3Assert(_finite(linImp1.getX()));
#endif
		if (invMassA)
		{
			dLinVelA += linImp0;
			dAngVelA += angImp0;
		}
		if (invMassB)
		{
			dLinVelB += linImp1;
			dAngVelB += angImp1;
		}
	}

	{	//	angular damping for point constraint
		b3Vector3 ab = ( posB - posA ).normalized();
		b3Vector3 ac = ( center - posA ).normalized();
		if( b3Dot( ab, ac ) > 0.95f || (invMassA == 0.f || invMassB == 0.f))
		{
			float angNA = b3Dot( n, angVelA );
			float angNB = b3Dot( n, angVelB );

			if (invMassA)
				dAngVelA -= (angNA*0.1f)*n;
			if (invMassB)
				dAngVelB -= (angNB*0.1f)*n;
		}
	}

}




float calcJacCoeff(const b3Vector3& linear0, const b3Vector3& linear1, const b3Vector3& angular0, const b3Vector3& angular1,
					float invMass0, const b3Matrix3x3* invInertia0, float invMass1, const b3Matrix3x3* invInertia1, float countA, float countB)
{
	//	linear0,1 are normlized
	float jmj0 = invMass0;//dot3F4(linear0, linear0)*invMass0;
	
	float jmj1 = b3Dot(mtMul3(angular0,*invInertia0), angular0);
	float jmj2 = invMass1;//dot3F4(linear1, linear1)*invMass1;
	float jmj3 = b3Dot(mtMul3(angular1,*invInertia1), angular1);
	return -1.f/((jmj0+jmj1)*countA+(jmj2+jmj3)*countB);
//	return -1.f/((jmj0+jmj1)+(jmj2+jmj3));

}


void setConstraint4( const b3Vector3& posA, const b3Vector3& linVelA, const b3Vector3& angVelA, float invMassA, const b3Matrix3x3& invInertiaA,
	const b3Vector3& posB, const b3Vector3& linVelB, const b3Vector3& angVelB, float invMassB, const b3Matrix3x3& invInertiaB, 
	 b3Contact4* src, float dt, float positionDrift, float positionConstraintCoeff, float countA, float countB,
	b3GpuConstraint4* dstC )
{
	dstC->m_bodyA = abs(src->m_bodyAPtrAndSignBit);
	dstC->m_bodyB = abs(src->m_bodyBPtrAndSignBit);

	float dtInv = 1.f/dt;
	for(int ic=0; ic<4; ic++)
	{
		dstC->m_appliedRambdaDt[ic] = 0.f;
	}
	dstC->m_fJacCoeffInv[0] = dstC->m_fJacCoeffInv[1] = 0.f;


	dstC->m_linear = src->m_worldNormalOnB;
	dstC->m_linear[3] = 0.7f ;//src->getFrictionCoeff() );
	for(int ic=0; ic<4; ic++)
	{
		b3Vector3 r0 = src->m_worldPosB[ic] - posA;
		b3Vector3 r1 = src->m_worldPosB[ic] - posB;

		if( ic >= src->m_worldNormalOnB[3] )//npoints
		{
			dstC->m_jacCoeffInv[ic] = 0.f;
			continue;
		}

		float relVelN;
		{
			b3Vector3 linear, angular0, angular1;
			setLinearAndAngular(src->m_worldNormalOnB, r0, r1, linear, angular0, angular1);

			dstC->m_jacCoeffInv[ic] = calcJacCoeff(linear, -linear, angular0, angular1,
				invMassA, &invInertiaA, invMassB, &invInertiaB ,countA,countB);

			relVelN = calcRelVel(linear, -linear, angular0, angular1,
				linVelA, angVelA, linVelB, angVelB);

			float e = 0.f;//src->getRestituitionCoeff();
			if( relVelN*relVelN < 0.004f ) 
			{
				e = 0.f;
			}

			dstC->m_b[ic] = e*relVelN;
			//float penetration = src->m_worldPos[ic].w;
			dstC->m_b[ic] += (src->m_worldPosB[ic][3] + positionDrift)*positionConstraintCoeff*dtInv;
			dstC->m_appliedRambdaDt[ic] = 0.f;
		}
	}

	if( src->m_worldNormalOnB[3] > 0 )//npoints
	{	//	prepare friction
		b3Vector3 center = make_float4(0.f);
		for(int i=0; i<src->m_worldNormalOnB[3]; i++) 
			center += src->m_worldPosB[i];
		center /= (float)src->m_worldNormalOnB[3];

		b3Vector3 tangent[2];
		b3PlaneSpace1(src->m_worldNormalOnB,tangent[0],tangent[1]);
		
		b3Vector3 r[2];
		r[0] = center - posA;
		r[1] = center - posB;

		for(int i=0; i<2; i++)
		{
			b3Vector3 linear, angular0, angular1;
			setLinearAndAngular(tangent[i], r[0], r[1], linear, angular0, angular1);

			dstC->m_fJacCoeffInv[i] = calcJacCoeff(linear, -linear, angular0, angular1,
				invMassA, &invInertiaA, invMassB, &invInertiaB ,countA,countB);
			dstC->m_fAppliedRambdaDt[i] = 0.f;
		}
		dstC->m_center = center;
	}

	for(int i=0; i<4; i++)
	{
		if( i<src->m_worldNormalOnB[3] )
		{
			dstC->m_worldPos[i] = src->m_worldPosB[i];
		}
		else
		{
			dstC->m_worldPos[i] = make_float4(0.f);
		}
	}
}



void ContactToConstraintKernel(b3Contact4* gContact, b3RigidBodyData* gBodies, b3InertiaData* gShapes, b3GpuConstraint4* gConstraintOut, int nContacts,
float dt,
float positionDrift,
float positionConstraintCoeff, int gIdx, b3AlignedObjectArray<unsigned int>& bodyCount
)
{
	//int gIdx = 0;//GET_GLOBAL_IDX;
	
	if( gIdx < nContacts )
	{
		int aIdx = abs(gContact[gIdx].m_bodyAPtrAndSignBit);
		int bIdx = abs(gContact[gIdx].m_bodyBPtrAndSignBit);

		b3Vector3 posA = gBodies[aIdx].m_pos;
		b3Vector3 linVelA = gBodies[aIdx].m_linVel;
		b3Vector3 angVelA = gBodies[aIdx].m_angVel;
		float invMassA = gBodies[aIdx].m_invMass;
		b3Matrix3x3 invInertiaA = gShapes[aIdx].m_invInertiaWorld;//.m_invInertia;

		b3Vector3 posB = gBodies[bIdx].m_pos;
		b3Vector3 linVelB = gBodies[bIdx].m_linVel;
		b3Vector3 angVelB = gBodies[bIdx].m_angVel;
		float invMassB = gBodies[bIdx].m_invMass;
		b3Matrix3x3 invInertiaB = gShapes[bIdx].m_invInertiaWorld;//m_invInertia;

		b3GpuConstraint4 cs;
		float countA = invMassA ? (float)(bodyCount[aIdx]) : 1;
		float countB = invMassB ? (float)(bodyCount[bIdx]) : 1;
    	setConstraint4( posA, linVelA, angVelA, invMassA, invInertiaA, posB, linVelB, angVelB, invMassB, invInertiaB,
			&gContact[gIdx], dt, positionDrift, positionConstraintCoeff,countA,countB,
			&cs );
		

		
		cs.m_batchIdx = gContact[gIdx].m_batchIdx;

		gConstraintOut[gIdx] = cs;
	}
}


void b3GpuJacobiContactSolver::solveGroupHost(b3RigidBodyData* bodies,b3InertiaData* inertias,int numBodies,b3Contact4* manifoldPtr, int numManifolds,const b3JacobiSolverInfo& solverInfo)
{
	B3_PROFILE("b3GpuJacobiContactSolver::solveGroup");

	b3AlignedObjectArray<unsigned int> bodyCount;
	bodyCount.resize(numBodies);
	for (int i=0;i<numBodies;i++)
		bodyCount[i] = 0;

	b3AlignedObjectArray<b3Int2> contactConstraintOffsets;
	contactConstraintOffsets.resize(numManifolds);


	for (int i=0;i<numManifolds;i++)
	{
		int pa = manifoldPtr[i].m_bodyAPtrAndSignBit;
		int pb = manifoldPtr[i].m_bodyBPtrAndSignBit;

		bool isFixedA = (pa <0) || (pa == solverInfo.m_fixedBodyIndex);
		bool isFixedB = (pb <0) || (pb == solverInfo.m_fixedBodyIndex);

		int bodyIndexA = manifoldPtr[i].getBodyA();
		int bodyIndexB = manifoldPtr[i].getBodyB();

		if (!isFixedA)
		{
			contactConstraintOffsets[i].x = bodyCount[bodyIndexA];
			bodyCount[bodyIndexA]++;
		}
		if (!isFixedB)
		{
			contactConstraintOffsets[i].y = bodyCount[bodyIndexB];
			bodyCount[bodyIndexB]++;
		} 
	}

	b3AlignedObjectArray<unsigned int> offsetSplitBodies;
	offsetSplitBodies.resize(numBodies);
	unsigned int totalNumSplitBodies;
	m_data->m_scan->executeHost(bodyCount,offsetSplitBodies,numBodies,&totalNumSplitBodies);
	int numlastBody = bodyCount[numBodies-1];
	totalNumSplitBodies += numlastBody;
	printf("totalNumSplitBodies = %d\n",totalNumSplitBodies);

	



	b3AlignedObjectArray<b3GpuConstraint4> contactConstraints;
	contactConstraints.resize(numManifolds);

	for (int i=0;i<numManifolds;i++)
	{
		ContactToConstraintKernel(&manifoldPtr[0],bodies,inertias,&contactConstraints[0],numManifolds,
			solverInfo.m_deltaTime,
			solverInfo.m_positionDrift,
			solverInfo.m_positionConstraintCoeff,
			i, bodyCount);
	}
	int maxIter = solverInfo.m_numIterations;


	b3AlignedObjectArray<b3Vector3> deltaLinearVelocities;
	b3AlignedObjectArray<b3Vector3> deltaAngularVelocities;
	deltaLinearVelocities.resize(totalNumSplitBodies);
	deltaAngularVelocities.resize(totalNumSplitBodies);
	for (unsigned int i=0;i<totalNumSplitBodies;i++)
	{
		deltaLinearVelocities[i].setZero();
		deltaAngularVelocities[i].setZero();
	}



	for (int iter = 0;iter<maxIter;iter++)
	{
		int i=0;
		for( i=0; i<numManifolds; i++)
		{

			//float frictionCoeff = contactConstraints[i].getFrictionCoeff();
			int aIdx = (int)contactConstraints[i].m_bodyA;
			int bIdx = (int)contactConstraints[i].m_bodyB;
			b3RigidBodyData& bodyA = bodies[aIdx];
			b3RigidBodyData& bodyB = bodies[bIdx];

			b3Vector3 zero = b3MakeVector3(0,0,0);
			
			b3Vector3* dlvAPtr=&zero;
			b3Vector3* davAPtr=&zero;
			b3Vector3* dlvBPtr=&zero;
			b3Vector3* davBPtr=&zero;
			
			if (bodyA.m_invMass)
			{
				int bodyOffsetA = offsetSplitBodies[aIdx];
				int constraintOffsetA = contactConstraintOffsets[i].x;
				int splitIndexA = bodyOffsetA+constraintOffsetA;
				dlvAPtr = &deltaLinearVelocities[splitIndexA];
				davAPtr = &deltaAngularVelocities[splitIndexA];
			}

			if (bodyB.m_invMass)
			{
				int bodyOffsetB = offsetSplitBodies[bIdx];
				int constraintOffsetB = contactConstraintOffsets[i].y;
				int splitIndexB= bodyOffsetB+constraintOffsetB;
				dlvBPtr =&deltaLinearVelocities[splitIndexB];
				davBPtr = &deltaAngularVelocities[splitIndexB];
			}



			{
				float maxRambdaDt[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
				float minRambdaDt[4] = {0.f,0.f,0.f,0.f};

				solveContact( contactConstraints[i], (b3Vector3&)bodyA.m_pos, (b3Vector3&)bodyA.m_linVel, (b3Vector3&)bodyA.m_angVel, bodyA.m_invMass, inertias[aIdx].m_invInertiaWorld, 
					(b3Vector3&)bodyB.m_pos, (b3Vector3&)bodyB.m_linVel, (b3Vector3&)bodyB.m_angVel, bodyB.m_invMass, inertias[bIdx].m_invInertiaWorld,
					maxRambdaDt, minRambdaDt , *dlvAPtr,*davAPtr,*dlvBPtr,*davBPtr		);


			}
			
		}

		
		//easy
		for (int i=0;i<numBodies;i++)
		{
			if (bodies[i].m_invMass)
			{
				int bodyOffset = offsetSplitBodies[i];
				int count = bodyCount[i];
				float factor = 1.f/float(count);
				b3Vector3 averageLinVel;
				averageLinVel.setZero();
				b3Vector3 averageAngVel;
				averageAngVel.setZero();
				for (int j=0;j<count;j++)
				{
					averageLinVel += deltaLinearVelocities[bodyOffset+j]*factor;
					averageAngVel += deltaAngularVelocities[bodyOffset+j]*factor;
				}
				for (int j=0;j<count;j++)
				{
					deltaLinearVelocities[bodyOffset+j] = averageLinVel;
					deltaAngularVelocities[bodyOffset+j] = averageAngVel;
				}
			}
		}
	}
	for (int iter = 0;iter<maxIter;iter++)
	{
		//int i=0;
	
		//solve friction

		for(int i=0; i<numManifolds; i++)
		{
			float maxRambdaDt[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
			float minRambdaDt[4] = {0.f,0.f,0.f,0.f};

			float sum = 0;
			for(int j=0; j<4; j++)
			{
				sum +=contactConstraints[i].m_appliedRambdaDt[j];
			}
			float frictionCoeff = contactConstraints[i].getFrictionCoeff();
			int aIdx = (int)contactConstraints[i].m_bodyA;
			int bIdx = (int)contactConstraints[i].m_bodyB;
			b3RigidBodyData& bodyA = bodies[aIdx];
			b3RigidBodyData& bodyB = bodies[bIdx];

			b3Vector3 zero = b3MakeVector3(0,0,0);
			
			b3Vector3* dlvAPtr=&zero;
			b3Vector3* davAPtr=&zero;
			b3Vector3* dlvBPtr=&zero;
			b3Vector3* davBPtr=&zero;
			
			if (bodyA.m_invMass)
			{
				int bodyOffsetA = offsetSplitBodies[aIdx];
				int constraintOffsetA = contactConstraintOffsets[i].x;
				int splitIndexA = bodyOffsetA+constraintOffsetA;
				dlvAPtr = &deltaLinearVelocities[splitIndexA];
				davAPtr = &deltaAngularVelocities[splitIndexA];
			}

			if (bodyB.m_invMass)
			{
				int bodyOffsetB = offsetSplitBodies[bIdx];
				int constraintOffsetB = contactConstraintOffsets[i].y;
				int splitIndexB= bodyOffsetB+constraintOffsetB;
				dlvBPtr =&deltaLinearVelocities[splitIndexB];
				davBPtr = &deltaAngularVelocities[splitIndexB];
			}

			for(int j=0; j<4; j++)
			{
				maxRambdaDt[j] = frictionCoeff*sum;
				minRambdaDt[j] = -maxRambdaDt[j];
			}

			solveFriction( contactConstraints[i], (b3Vector3&)bodyA.m_pos, (b3Vector3&)bodyA.m_linVel, (b3Vector3&)bodyA.m_angVel, bodyA.m_invMass,inertias[aIdx].m_invInertiaWorld, 
				(b3Vector3&)bodyB.m_pos, (b3Vector3&)bodyB.m_linVel, (b3Vector3&)bodyB.m_angVel, bodyB.m_invMass, inertias[bIdx].m_invInertiaWorld,
				maxRambdaDt, minRambdaDt , *dlvAPtr,*davAPtr,*dlvBPtr,*davBPtr);

		}

		//easy
		for (int i=0;i<numBodies;i++)
		{
			if (bodies[i].m_invMass)
			{
				int bodyOffset = offsetSplitBodies[i];
				int count = bodyCount[i];
				float factor = 1.f/float(count);
				b3Vector3 averageLinVel;
				averageLinVel.setZero();
				b3Vector3 averageAngVel;
				averageAngVel.setZero();
				for (int j=0;j<count;j++)
				{
					averageLinVel += deltaLinearVelocities[bodyOffset+j]*factor;
					averageAngVel += deltaAngularVelocities[bodyOffset+j]*factor;
				}
				for (int j=0;j<count;j++)
				{
					deltaLinearVelocities[bodyOffset+j] = averageLinVel;
					deltaAngularVelocities[bodyOffset+j] = averageAngVel;
				}
			}
		}



	}


	//easy
	for (int i=0;i<numBodies;i++)
	{
		if (bodies[i].m_invMass)
		{
			int bodyOffset = offsetSplitBodies[i];
			int count = bodyCount[i];
			if (count)
			{
				bodies[i].m_linVel += deltaLinearVelocities[bodyOffset];
				bodies[i].m_angVel += deltaAngularVelocities[bodyOffset];
			}
		}
	}
}



void b3GpuJacobiContactSolver::solveContacts(int numBodies, cl_mem bodyBuf, cl_mem inertiaBuf, int numContacts, cl_mem contactBuf, const struct b3Config& config, int static0Index)
//
//
//void  b3GpuJacobiContactSolver::solveGroup(b3OpenCLArray<b3RigidBodyData>* bodies,b3OpenCLArray<b3InertiaData>* inertias,b3OpenCLArray<b3Contact4>* manifoldPtr,const btJacobiSolverInfo& solverInfo)
{
	b3JacobiSolverInfo solverInfo;
	solverInfo.m_fixedBodyIndex = static0Index;
	

	B3_PROFILE("b3GpuJacobiContactSolver::solveGroup");

	//int numBodies = bodies->size();
	int numManifolds = numContacts;//manifoldPtr->size();

	{
		B3_PROFILE("resize");
		m_data->m_bodyCount->resize(numBodies);
	}
	
	unsigned int val=0;
	b3Int2 val2;
	val2.x=0;
	val2.y=0;

	 {
		B3_PROFILE("m_filler");
		m_data->m_contactConstraintOffsets->resize(numManifolds);
		m_data->m_filler->execute(*m_data->m_bodyCount,val,numBodies);
		
	
		m_data->m_filler->execute(*m_data->m_contactConstraintOffsets,val2,numManifolds);
	}

	{
		B3_PROFILE("m_countBodiesKernel");
		b3LauncherCL launcher(this->m_queue,m_data->m_countBodiesKernel,"m_countBodiesKernel");
		launcher.setBuffer(contactBuf);//manifoldPtr->getBufferCL());
		launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
		launcher.setBuffer(m_data->m_contactConstraintOffsets->getBufferCL());
		launcher.setConst(numManifolds);
		launcher.setConst(solverInfo.m_fixedBodyIndex);
		launcher.launch1D(numManifolds);
	}
	unsigned int totalNumSplitBodies=0;
	{
		B3_PROFILE("m_scan->execute");
		
		m_data->m_offsetSplitBodies->resize(numBodies);
		m_data->m_scan->execute(*m_data->m_bodyCount,*m_data->m_offsetSplitBodies,numBodies,&totalNumSplitBodies);
		totalNumSplitBodies+=m_data->m_bodyCount->at(numBodies-1);
	}

	{
		B3_PROFILE("m_data->m_contactConstraints->resize");
		//int numContacts = manifoldPtr->size();
		m_data->m_contactConstraints->resize(numContacts);
	}
	
	{
		B3_PROFILE("contactToConstraintSplitKernel");
		b3LauncherCL launcher( m_queue, m_data->m_contactToConstraintSplitKernel,"m_contactToConstraintSplitKernel");
		launcher.setBuffer(contactBuf);
		launcher.setBuffer(bodyBuf);
		launcher.setBuffer(inertiaBuf);
		launcher.setBuffer(m_data->m_contactConstraints->getBufferCL());
		launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
        launcher.setConst(numContacts);
		launcher.setConst(solverInfo.m_deltaTime);
		launcher.setConst(solverInfo.m_positionDrift);
		launcher.setConst(solverInfo.m_positionConstraintCoeff);
		launcher.launch1D( numContacts, 64 );
		
	}

	
	{
		B3_PROFILE("m_data->m_deltaLinearVelocities->resize");
		m_data->m_deltaLinearVelocities->resize(totalNumSplitBodies);
		m_data->m_deltaAngularVelocities->resize(totalNumSplitBodies);
	}


	
	{
		B3_PROFILE("m_clearVelocitiesKernel");
		b3LauncherCL launch(m_queue,m_data->m_clearVelocitiesKernel,"m_clearVelocitiesKernel");
		launch.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
		launch.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
		launch.setConst(totalNumSplitBodies);
		launch.launch1D(totalNumSplitBodies);
		clFinish(m_queue);
	}
	
	
	int maxIter = solverInfo.m_numIterations;

	for (int iter = 0;iter<maxIter;iter++)
	{
		{
			B3_PROFILE("m_solveContactKernel");
			b3LauncherCL launcher( m_queue, m_data->m_solveContactKernel,"m_solveContactKernel" );
			launcher.setBuffer(m_data->m_contactConstraints->getBufferCL());
			launcher.setBuffer(bodyBuf);
			launcher.setBuffer(inertiaBuf);
			launcher.setBuffer(m_data->m_contactConstraintOffsets->getBufferCL());
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(solverInfo.m_deltaTime);
			launcher.setConst(solverInfo.m_positionDrift);
			launcher.setConst(solverInfo.m_positionConstraintCoeff);
			launcher.setConst(solverInfo.m_fixedBodyIndex);
			launcher.setConst(numManifolds);

			launcher.launch1D(numManifolds);
			clFinish(m_queue);
		}

		

		{
			B3_PROFILE("average velocities");
			b3LauncherCL launcher( m_queue, m_data->m_averageVelocitiesKernel,"m_averageVelocitiesKernel");
			launcher.setBuffer(bodyBuf);
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(numBodies);
			launcher.launch1D(numBodies);
			clFinish(m_queue);
		}

		
		{
			B3_PROFILE("m_solveFrictionKernel");
			b3LauncherCL launcher( m_queue, m_data->m_solveFrictionKernel,"m_solveFrictionKernel");
			launcher.setBuffer(m_data->m_contactConstraints->getBufferCL());
			launcher.setBuffer(bodyBuf);
			launcher.setBuffer(inertiaBuf);
			launcher.setBuffer(m_data->m_contactConstraintOffsets->getBufferCL());
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(solverInfo.m_deltaTime);
			launcher.setConst(solverInfo.m_positionDrift);
			launcher.setConst(solverInfo.m_positionConstraintCoeff);
			launcher.setConst(solverInfo.m_fixedBodyIndex);
			launcher.setConst(numManifolds);

			launcher.launch1D(numManifolds);
			clFinish(m_queue);
		}

		
		{
			B3_PROFILE("average velocities");
			b3LauncherCL launcher( m_queue, m_data->m_averageVelocitiesKernel,"m_averageVelocitiesKernel");
			launcher.setBuffer(bodyBuf);
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(numBodies);
			launcher.launch1D(numBodies);
			clFinish(m_queue);
		}

		

	}

	
	{
			B3_PROFILE("update body velocities");
			b3LauncherCL launcher( m_queue, m_data->m_updateBodyVelocitiesKernel,"m_updateBodyVelocitiesKernel");
			launcher.setBuffer(bodyBuf);
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(numBodies);
			launcher.launch1D(numBodies);
			clFinish(m_queue);
		}



}

#if 0

void  b3GpuJacobiContactSolver::solveGroupMixed(b3OpenCLArray<b3RigidBodyData>* bodiesGPU,b3OpenCLArray<b3InertiaData>* inertiasGPU,b3OpenCLArray<b3Contact4>* manifoldPtrGPU,const btJacobiSolverInfo& solverInfo)
{

	b3AlignedObjectArray<b3RigidBodyData> bodiesCPU;
	bodiesGPU->copyToHost(bodiesCPU);
	b3AlignedObjectArray<b3InertiaData> inertiasCPU;
	inertiasGPU->copyToHost(inertiasCPU);
	b3AlignedObjectArray<b3Contact4> manifoldPtrCPU;
	manifoldPtrGPU->copyToHost(manifoldPtrCPU);
	
	int numBodiesCPU = bodiesGPU->size();
	int numManifoldsCPU = manifoldPtrGPU->size();
	B3_PROFILE("b3GpuJacobiContactSolver::solveGroupMixed");

	b3AlignedObjectArray<unsigned int> bodyCount;
	bodyCount.resize(numBodiesCPU);
	for (int i=0;i<numBodiesCPU;i++)
		bodyCount[i] = 0;

	b3AlignedObjectArray<b3Int2> contactConstraintOffsets;
	contactConstraintOffsets.resize(numManifoldsCPU);


	for (int i=0;i<numManifoldsCPU;i++)
	{
		int pa = manifoldPtrCPU[i].m_bodyAPtrAndSignBit;
		int pb = manifoldPtrCPU[i].m_bodyBPtrAndSignBit;

		bool isFixedA = (pa <0) || (pa == solverInfo.m_fixedBodyIndex);
		bool isFixedB = (pb <0) || (pb == solverInfo.m_fixedBodyIndex);

		int bodyIndexA = manifoldPtrCPU[i].getBodyA();
		int bodyIndexB = manifoldPtrCPU[i].getBodyB();

		if (!isFixedA)
		{
			contactConstraintOffsets[i].x = bodyCount[bodyIndexA];
			bodyCount[bodyIndexA]++;
		}
		if (!isFixedB)
		{
			contactConstraintOffsets[i].y = bodyCount[bodyIndexB];
			bodyCount[bodyIndexB]++;
		} 
	}

	b3AlignedObjectArray<unsigned int> offsetSplitBodies;
	offsetSplitBodies.resize(numBodiesCPU);
	unsigned int totalNumSplitBodiesCPU;
	m_data->m_scan->executeHost(bodyCount,offsetSplitBodies,numBodiesCPU,&totalNumSplitBodiesCPU);
	int numlastBody = bodyCount[numBodiesCPU-1];
	totalNumSplitBodiesCPU += numlastBody;

		int numBodies = bodiesGPU->size();
	int numManifolds = manifoldPtrGPU->size();

	m_data->m_bodyCount->resize(numBodies);
	
	unsigned int val=0;
	b3Int2 val2;
	val2.x=0;
	val2.y=0;

	 {
		B3_PROFILE("m_filler");
		m_data->m_contactConstraintOffsets->resize(numManifolds);
		m_data->m_filler->execute(*m_data->m_bodyCount,val,numBodies);
		
	
		m_data->m_filler->execute(*m_data->m_contactConstraintOffsets,val2,numManifolds);
	}

	{
		B3_PROFILE("m_countBodiesKernel");
		b3LauncherCL launcher(this->m_queue,m_data->m_countBodiesKernel);
		launcher.setBuffer(manifoldPtrGPU->getBufferCL());
		launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
		launcher.setBuffer(m_data->m_contactConstraintOffsets->getBufferCL());
		launcher.setConst(numManifolds);
		launcher.setConst(solverInfo.m_fixedBodyIndex);
		launcher.launch1D(numManifolds);
	}

	unsigned int totalNumSplitBodies=0;
	m_data->m_offsetSplitBodies->resize(numBodies);
	m_data->m_scan->execute(*m_data->m_bodyCount,*m_data->m_offsetSplitBodies,numBodies,&totalNumSplitBodies);
	totalNumSplitBodies+=m_data->m_bodyCount->at(numBodies-1);

	if (totalNumSplitBodies != totalNumSplitBodiesCPU)
	{
		printf("error in totalNumSplitBodies!\n");
	}

	int numContacts = manifoldPtrGPU->size();
	m_data->m_contactConstraints->resize(numContacts);

	
	{
		B3_PROFILE("contactToConstraintSplitKernel");
		b3LauncherCL launcher( m_queue, m_data->m_contactToConstraintSplitKernel);
		launcher.setBuffer(manifoldPtrGPU->getBufferCL());
		launcher.setBuffer(bodiesGPU->getBufferCL());
		launcher.setBuffer(inertiasGPU->getBufferCL());
		launcher.setBuffer(m_data->m_contactConstraints->getBufferCL());
		launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
        launcher.setConst(numContacts);
		launcher.setConst(solverInfo.m_deltaTime);
		launcher.setConst(solverInfo.m_positionDrift);
		launcher.setConst(solverInfo.m_positionConstraintCoeff);
		launcher.launch1D( numContacts, 64 );
		clFinish(m_queue);
	}



	b3AlignedObjectArray<b3GpuConstraint4> contactConstraints;
	contactConstraints.resize(numManifoldsCPU);

	for (int i=0;i<numManifoldsCPU;i++)
	{
		ContactToConstraintKernel(&manifoldPtrCPU[0],&bodiesCPU[0],&inertiasCPU[0],&contactConstraints[0],numManifoldsCPU,
			solverInfo.m_deltaTime,
			solverInfo.m_positionDrift,
			solverInfo.m_positionConstraintCoeff,
			i, bodyCount);
	}
	int maxIter = solverInfo.m_numIterations;


	b3AlignedObjectArray<b3Vector3> deltaLinearVelocities;
	b3AlignedObjectArray<b3Vector3> deltaAngularVelocities;
	deltaLinearVelocities.resize(totalNumSplitBodiesCPU);
	deltaAngularVelocities.resize(totalNumSplitBodiesCPU);
	for (int i=0;i<totalNumSplitBodiesCPU;i++)
	{
		deltaLinearVelocities[i].setZero();
		deltaAngularVelocities[i].setZero();
	}

	m_data->m_deltaLinearVelocities->resize(totalNumSplitBodies);
	m_data->m_deltaAngularVelocities->resize(totalNumSplitBodies);


	
	{
		B3_PROFILE("m_clearVelocitiesKernel");
		b3LauncherCL launch(m_queue,m_data->m_clearVelocitiesKernel);
		launch.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
		launch.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
		launch.setConst(totalNumSplitBodies);
		launch.launch1D(totalNumSplitBodies);
	}
	

		///!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


	m_data->m_contactConstraints->copyToHost(contactConstraints);
	m_data->m_offsetSplitBodies->copyToHost(offsetSplitBodies);
	m_data->m_contactConstraintOffsets->copyToHost(contactConstraintOffsets);
	m_data->m_deltaLinearVelocities->copyToHost(deltaLinearVelocities);
	m_data->m_deltaAngularVelocities->copyToHost(deltaAngularVelocities);

	for (int iter = 0;iter<maxIter;iter++)
	{

				{
			B3_PROFILE("m_solveContactKernel");
			b3LauncherCL launcher( m_queue, m_data->m_solveContactKernel );
			launcher.setBuffer(m_data->m_contactConstraints->getBufferCL());
			launcher.setBuffer(bodiesGPU->getBufferCL());
			launcher.setBuffer(inertiasGPU->getBufferCL());
			launcher.setBuffer(m_data->m_contactConstraintOffsets->getBufferCL());
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(solverInfo.m_deltaTime);
			launcher.setConst(solverInfo.m_positionDrift);
			launcher.setConst(solverInfo.m_positionConstraintCoeff);
			launcher.setConst(solverInfo.m_fixedBodyIndex);
			launcher.setConst(numManifolds);

			launcher.launch1D(numManifolds);
			clFinish(m_queue);
		}


		int i=0;
		for( i=0; i<numManifoldsCPU; i++)
		{

			float frictionCoeff = contactConstraints[i].getFrictionCoeff();
			int aIdx = (int)contactConstraints[i].m_bodyA;
			int bIdx = (int)contactConstraints[i].m_bodyB;
			b3RigidBodyData& bodyA = bodiesCPU[aIdx];
			b3RigidBodyData& bodyB = bodiesCPU[bIdx];

			b3Vector3 zero(0,0,0);
			
			b3Vector3* dlvAPtr=&zero;
			b3Vector3* davAPtr=&zero;
			b3Vector3* dlvBPtr=&zero;
			b3Vector3* davBPtr=&zero;
			
			if (bodyA.m_invMass)
			{
				int bodyOffsetA = offsetSplitBodies[aIdx];
				int constraintOffsetA = contactConstraintOffsets[i].x;
				int splitIndexA = bodyOffsetA+constraintOffsetA;
				dlvAPtr = &deltaLinearVelocities[splitIndexA];
				davAPtr = &deltaAngularVelocities[splitIndexA];
			}

			if (bodyB.m_invMass)
			{
				int bodyOffsetB = offsetSplitBodies[bIdx];
				int constraintOffsetB = contactConstraintOffsets[i].y;
				int splitIndexB= bodyOffsetB+constraintOffsetB;
				dlvBPtr =&deltaLinearVelocities[splitIndexB];
				davBPtr = &deltaAngularVelocities[splitIndexB];
			}



			{
				float maxRambdaDt[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
				float minRambdaDt[4] = {0.f,0.f,0.f,0.f};

				solveContact( contactConstraints[i], (b3Vector3&)bodyA.m_pos, (b3Vector3&)bodyA.m_linVel, (b3Vector3&)bodyA.m_angVel, bodyA.m_invMass, inertiasCPU[aIdx].m_invInertiaWorld, 
					(b3Vector3&)bodyB.m_pos, (b3Vector3&)bodyB.m_linVel, (b3Vector3&)bodyB.m_angVel, bodyB.m_invMass, inertiasCPU[bIdx].m_invInertiaWorld,
					maxRambdaDt, minRambdaDt , *dlvAPtr,*davAPtr,*dlvBPtr,*davBPtr		);


			}
		}

		
		{
			B3_PROFILE("average velocities");
			b3LauncherCL launcher( m_queue, m_data->m_averageVelocitiesKernel);
			launcher.setBuffer(bodiesGPU->getBufferCL());
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(numBodies);
			launcher.launch1D(numBodies);
			clFinish(m_queue);
		}

		//easy
		for (int i=0;i<numBodiesCPU;i++)
		{
			if (bodiesCPU[i].m_invMass)
			{
				int bodyOffset = offsetSplitBodies[i];
				int count = bodyCount[i];
				float factor = 1.f/float(count);
				b3Vector3 averageLinVel;
				averageLinVel.setZero();
				b3Vector3 averageAngVel;
				averageAngVel.setZero();
				for (int j=0;j<count;j++)
				{
					averageLinVel += deltaLinearVelocities[bodyOffset+j]*factor;
					averageAngVel += deltaAngularVelocities[bodyOffset+j]*factor;
				}
				for (int j=0;j<count;j++)
				{
					deltaLinearVelocities[bodyOffset+j] = averageLinVel;
					deltaAngularVelocities[bodyOffset+j] = averageAngVel;
				}
			}
		}
//	m_data->m_deltaAngularVelocities->copyFromHost(deltaAngularVelocities);
	//m_data->m_deltaLinearVelocities->copyFromHost(deltaLinearVelocities);
	m_data->m_deltaAngularVelocities->copyToHost(deltaAngularVelocities);
	m_data->m_deltaLinearVelocities->copyToHost(deltaLinearVelocities);

#if 0

		{
			B3_PROFILE("m_solveFrictionKernel");
			b3LauncherCL launcher( m_queue, m_data->m_solveFrictionKernel);
			launcher.setBuffer(m_data->m_contactConstraints->getBufferCL());
			launcher.setBuffer(bodiesGPU->getBufferCL());
			launcher.setBuffer(inertiasGPU->getBufferCL());
			launcher.setBuffer(m_data->m_contactConstraintOffsets->getBufferCL());
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(solverInfo.m_deltaTime);
			launcher.setConst(solverInfo.m_positionDrift);
			launcher.setConst(solverInfo.m_positionConstraintCoeff);
			launcher.setConst(solverInfo.m_fixedBodyIndex);
			launcher.setConst(numManifolds);

			launcher.launch1D(numManifolds);
			clFinish(m_queue);
		}

		//solve friction

		for(int i=0; i<numManifoldsCPU; i++)
		{
			float maxRambdaDt[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
			float minRambdaDt[4] = {0.f,0.f,0.f,0.f};

			float sum = 0;
			for(int j=0; j<4; j++)
			{
				sum +=contactConstraints[i].m_appliedRambdaDt[j];
			}
			float frictionCoeff = contactConstraints[i].getFrictionCoeff();
			int aIdx = (int)contactConstraints[i].m_bodyA;
			int bIdx = (int)contactConstraints[i].m_bodyB;
			b3RigidBodyData& bodyA = bodiesCPU[aIdx];
			b3RigidBodyData& bodyB = bodiesCPU[bIdx];

			b3Vector3 zero(0,0,0);
			
			b3Vector3* dlvAPtr=&zero;
			b3Vector3* davAPtr=&zero;
			b3Vector3* dlvBPtr=&zero;
			b3Vector3* davBPtr=&zero;
			
			if (bodyA.m_invMass)
			{
				int bodyOffsetA = offsetSplitBodies[aIdx];
				int constraintOffsetA = contactConstraintOffsets[i].x;
				int splitIndexA = bodyOffsetA+constraintOffsetA;
				dlvAPtr = &deltaLinearVelocities[splitIndexA];
				davAPtr = &deltaAngularVelocities[splitIndexA];
			}

			if (bodyB.m_invMass)
			{
				int bodyOffsetB = offsetSplitBodies[bIdx];
				int constraintOffsetB = contactConstraintOffsets[i].y;
				int splitIndexB= bodyOffsetB+constraintOffsetB;
				dlvBPtr =&deltaLinearVelocities[splitIndexB];
				davBPtr = &deltaAngularVelocities[splitIndexB];
			}

			for(int j=0; j<4; j++)
			{
				maxRambdaDt[j] = frictionCoeff*sum;
				minRambdaDt[j] = -maxRambdaDt[j];
			}

			solveFriction( contactConstraints[i], (b3Vector3&)bodyA.m_pos, (b3Vector3&)bodyA.m_linVel, (b3Vector3&)bodyA.m_angVel, bodyA.m_invMass,inertiasCPU[aIdx].m_invInertiaWorld, 
				(b3Vector3&)bodyB.m_pos, (b3Vector3&)bodyB.m_linVel, (b3Vector3&)bodyB.m_angVel, bodyB.m_invMass, inertiasCPU[bIdx].m_invInertiaWorld,
				maxRambdaDt, minRambdaDt , *dlvAPtr,*davAPtr,*dlvBPtr,*davBPtr);

		}

		{
			B3_PROFILE("average velocities");
			b3LauncherCL launcher( m_queue, m_data->m_averageVelocitiesKernel);
			launcher.setBuffer(bodiesGPU->getBufferCL());
			launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
			launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
			launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
			launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
			launcher.setConst(numBodies);
			launcher.launch1D(numBodies);
			clFinish(m_queue);
		}

		//easy
		for (int i=0;i<numBodiesCPU;i++)
		{
			if (bodiesCPU[i].m_invMass)
			{
				int bodyOffset = offsetSplitBodies[i];
				int count = bodyCount[i];
				float factor = 1.f/float(count);
				b3Vector3 averageLinVel;
				averageLinVel.setZero();
				b3Vector3 averageAngVel;
				averageAngVel.setZero();
				for (int j=0;j<count;j++)
				{
					averageLinVel += deltaLinearVelocities[bodyOffset+j]*factor;
					averageAngVel += deltaAngularVelocities[bodyOffset+j]*factor;
				}
				for (int j=0;j<count;j++)
				{
					deltaLinearVelocities[bodyOffset+j] = averageLinVel;
					deltaAngularVelocities[bodyOffset+j] = averageAngVel;
				}
			}
		}

#endif

	}

	{
		B3_PROFILE("update body velocities");
		b3LauncherCL launcher( m_queue, m_data->m_updateBodyVelocitiesKernel);
		launcher.setBuffer(bodiesGPU->getBufferCL());
		launcher.setBuffer(m_data->m_offsetSplitBodies->getBufferCL());
		launcher.setBuffer(m_data->m_bodyCount->getBufferCL());
		launcher.setBuffer(m_data->m_deltaLinearVelocities->getBufferCL());
		launcher.setBuffer(m_data->m_deltaAngularVelocities->getBufferCL());
		launcher.setConst(numBodies);
		launcher.launch1D(numBodies);
		clFinish(m_queue);
	}


	//easy
	for (int i=0;i<numBodiesCPU;i++)
	{
		if (bodiesCPU[i].m_invMass)
		{
			int bodyOffset = offsetSplitBodies[i];
			int count = bodyCount[i];
			if (count)
			{
				bodiesCPU[i].m_linVel += deltaLinearVelocities[bodyOffset];
				bodiesCPU[i].m_angVel += deltaAngularVelocities[bodyOffset];
			}
		}
	}


//	bodiesGPU->copyFromHost(bodiesCPU);


}
#endif
