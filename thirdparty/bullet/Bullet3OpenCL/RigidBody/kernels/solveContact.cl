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


//#pragma OPENCL EXTENSION cl_amd_printf : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable


#ifdef cl_ext_atomic_counters_32
#pragma OPENCL EXTENSION cl_ext_atomic_counters_32 : enable
#else
#define counter32_t volatile global int*
#endif

typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;

#define GET_GROUP_IDX get_group_id(0)
#define GET_LOCAL_IDX get_local_id(0)
#define GET_GLOBAL_IDX get_global_id(0)
#define GET_GROUP_SIZE get_local_size(0)
#define GET_NUM_GROUPS get_num_groups(0)
#define GROUP_LDS_BARRIER barrier(CLK_LOCAL_MEM_FENCE)
#define GROUP_MEM_FENCE mem_fence(CLK_LOCAL_MEM_FENCE)
#define AtomInc(x) atom_inc(&(x))
#define AtomInc1(x, out) out = atom_inc(&(x))
#define AppendInc(x, out) out = atomic_inc(x)
#define AtomAdd(x, value) atom_add(&(x), value)
#define AtomCmpxhg(x, cmp, value) atom_cmpxchg( &(x), cmp, value )
#define AtomXhg(x, value) atom_xchg ( &(x), value )


#define SELECT_UINT4( b, a, condition ) select( b,a,condition )

#define mymake_float4 (float4)
//#define make_float2 (float2)
//#define make_uint4 (uint4)
//#define make_int4 (int4)
//#define make_uint2 (uint2)
//#define make_int2 (int2)


#define max2 max
#define min2 min


///////////////////////////////////////
//	Vector
///////////////////////////////////////




__inline
float4 fastNormalize4(float4 v)
{
	return fast_normalize(v);
}



__inline
float4 cross3(float4 a, float4 b)
{
	return cross(a,b);
}

__inline
float dot3F4(float4 a, float4 b)
{
	float4 a1 = mymake_float4(a.xyz,0.f);
	float4 b1 = mymake_float4(b.xyz,0.f);
	return dot(a1, b1);
}




__inline
float4 normalize3(const float4 a)
{
	float4 n = mymake_float4(a.x, a.y, a.z, 0.f);
	return fastNormalize4( n );
//	float length = sqrtf(dot3F4(a, a));
//	return 1.f/length * a;
}




///////////////////////////////////////
//	Matrix3x3
///////////////////////////////////////

typedef struct
{
	float4 m_row[3];
}Matrix3x3;






__inline
float4 mtMul1(Matrix3x3 a, float4 b);

__inline
float4 mtMul3(float4 a, Matrix3x3 b);




__inline
float4 mtMul1(Matrix3x3 a, float4 b)
{
	float4 ans;
	ans.x = dot3F4( a.m_row[0], b );
	ans.y = dot3F4( a.m_row[1], b );
	ans.z = dot3F4( a.m_row[2], b );
	ans.w = 0.f;
	return ans;
}

__inline
float4 mtMul3(float4 a, Matrix3x3 b)
{
	float4 colx = mymake_float4(b.m_row[0].x, b.m_row[1].x, b.m_row[2].x, 0);
	float4 coly = mymake_float4(b.m_row[0].y, b.m_row[1].y, b.m_row[2].y, 0);
	float4 colz = mymake_float4(b.m_row[0].z, b.m_row[1].z, b.m_row[2].z, 0);

	float4 ans;
	ans.x = dot3F4( a, colx );
	ans.y = dot3F4( a, coly );
	ans.z = dot3F4( a, colz );
	return ans;
}

///////////////////////////////////////
//	Quaternion
///////////////////////////////////////

typedef float4 Quaternion;







#define WG_SIZE 64

typedef struct
{
	float4 m_pos;
	Quaternion m_quat;
	float4 m_linVel;
	float4 m_angVel;

	u32 m_shapeIdx;
	float m_invMass;
	float m_restituitionCoeff;
	float m_frictionCoeff;
} Body;

typedef struct
{
	Matrix3x3 m_invInertia;
	Matrix3x3 m_initInvInertia;
} Shape;

typedef struct
{
	float4 m_linear;
	float4 m_worldPos[4];
	float4 m_center;	
	float m_jacCoeffInv[4];
	float m_b[4];
	float m_appliedRambdaDt[4];

	float m_fJacCoeffInv[2];	
	float m_fAppliedRambdaDt[2];	

	u32 m_bodyA;
	u32 m_bodyB;

	int m_batchIdx;
	u32 m_paddings[1];
} Constraint4;



typedef struct
{
	int m_nConstraints;
	int m_start;
	int m_batchIdx;
	int m_nSplit;
//	int m_paddings[1];
} ConstBuffer;

typedef struct
{
	int m_solveFriction;
	int m_maxBatch;	//	long batch really kills the performance
	int m_batchIdx;
	int m_nSplit;
//	int m_paddings[1];
} ConstBufferBatchSolve;

void setLinearAndAngular( float4 n, float4 r0, float4 r1, float4* linear, float4* angular0, float4* angular1);

void setLinearAndAngular( float4 n, float4 r0, float4 r1, float4* linear, float4* angular0, float4* angular1)
{
	*linear = mymake_float4(-n.xyz,0.f);
	*angular0 = -cross3(r0, n);
	*angular1 = cross3(r1, n);
}

float calcRelVel( float4 l0, float4 l1, float4 a0, float4 a1, float4 linVel0, float4 angVel0, float4 linVel1, float4 angVel1 );

float calcRelVel( float4 l0, float4 l1, float4 a0, float4 a1, float4 linVel0, float4 angVel0, float4 linVel1, float4 angVel1 )
{
	return dot3F4(l0, linVel0) + dot3F4(a0, angVel0) + dot3F4(l1, linVel1) + dot3F4(a1, angVel1);
}


float calcJacCoeff(const float4 linear0, const float4 linear1, const float4 angular0, const float4 angular1,
				   float invMass0, const Matrix3x3* invInertia0, float invMass1, const Matrix3x3* invInertia1);

float calcJacCoeff(const float4 linear0, const float4 linear1, const float4 angular0, const float4 angular1,
					float invMass0, const Matrix3x3* invInertia0, float invMass1, const Matrix3x3* invInertia1)
{
	//	linear0,1 are normlized
	float jmj0 = invMass0;//dot3F4(linear0, linear0)*invMass0;
	float jmj1 = dot3F4(mtMul3(angular0,*invInertia0), angular0);
	float jmj2 = invMass1;//dot3F4(linear1, linear1)*invMass1;
	float jmj3 = dot3F4(mtMul3(angular1,*invInertia1), angular1);
	return -1.f/(jmj0+jmj1+jmj2+jmj3);
}


void solveContact(__global Constraint4* cs,
				  float4 posA, float4* linVelA, float4* angVelA, float invMassA, Matrix3x3 invInertiaA,
				  float4 posB, float4* linVelB, float4* angVelB, float invMassB, Matrix3x3 invInertiaB);

void solveContact(__global Constraint4* cs,
			float4 posA, float4* linVelA, float4* angVelA, float invMassA, Matrix3x3 invInertiaA,
			float4 posB, float4* linVelB, float4* angVelB, float invMassB, Matrix3x3 invInertiaB)
{
	float minRambdaDt = 0;
	float maxRambdaDt = FLT_MAX;

	for(int ic=0; ic<4; ic++)
	{
		if( cs->m_jacCoeffInv[ic] == 0.f ) continue;

		float4 angular0, angular1, linear;
		float4 r0 = cs->m_worldPos[ic] - posA;
		float4 r1 = cs->m_worldPos[ic] - posB;
		setLinearAndAngular( -cs->m_linear, r0, r1, &linear, &angular0, &angular1 );

		float rambdaDt = calcRelVel( cs->m_linear, -cs->m_linear, angular0, angular1, 
			*linVelA, *angVelA, *linVelB, *angVelB ) + cs->m_b[ic];
		rambdaDt *= cs->m_jacCoeffInv[ic];

		{
			float prevSum = cs->m_appliedRambdaDt[ic];
			float updated = prevSum;
			updated += rambdaDt;
			updated = max2( updated, minRambdaDt );
			updated = min2( updated, maxRambdaDt );
			rambdaDt = updated - prevSum;
			cs->m_appliedRambdaDt[ic] = updated;
		}

		float4 linImp0 = invMassA*linear*rambdaDt;
		float4 linImp1 = invMassB*(-linear)*rambdaDt;
		float4 angImp0 = mtMul1(invInertiaA, angular0)*rambdaDt;
		float4 angImp1 = mtMul1(invInertiaB, angular1)*rambdaDt;

		*linVelA += linImp0;
		*angVelA += angImp0;
		*linVelB += linImp1;
		*angVelB += angImp1;
	}
}

void btPlaneSpace1 (const float4* n, float4* p, float4* q);
 void btPlaneSpace1 (const float4* n, float4* p, float4* q)
{
  if (fabs(n[0].z) > 0.70710678f) {
    // choose p in y-z plane
    float a = n[0].y*n[0].y + n[0].z*n[0].z;
    float k = 1.f/sqrt(a);
    p[0].x = 0;
	p[0].y = -n[0].z*k;
	p[0].z = n[0].y*k;
    // set q = n x p
    q[0].x = a*k;
	q[0].y = -n[0].x*p[0].z;
	q[0].z = n[0].x*p[0].y;
  }
  else {
    // choose p in x-y plane
    float a = n[0].x*n[0].x + n[0].y*n[0].y;
    float k = 1.f/sqrt(a);
    p[0].x = -n[0].y*k;
	p[0].y = n[0].x*k;
	p[0].z = 0;
    // set q = n x p
    q[0].x = -n[0].z*p[0].y;
	q[0].y = n[0].z*p[0].x;
	q[0].z = a*k;
  }
}

void solveContactConstraint(__global Body* gBodies, __global Shape* gShapes, __global Constraint4* ldsCs);
void solveContactConstraint(__global Body* gBodies, __global Shape* gShapes, __global Constraint4* ldsCs)
{
	//float frictionCoeff = ldsCs[0].m_linear.w;
	int aIdx = ldsCs[0].m_bodyA;
	int bIdx = ldsCs[0].m_bodyB;

	float4 posA = gBodies[aIdx].m_pos;
	float4 linVelA = gBodies[aIdx].m_linVel;
	float4 angVelA = gBodies[aIdx].m_angVel;
	float invMassA = gBodies[aIdx].m_invMass;
	Matrix3x3 invInertiaA = gShapes[aIdx].m_invInertia;

	float4 posB = gBodies[bIdx].m_pos;
	float4 linVelB = gBodies[bIdx].m_linVel;
	float4 angVelB = gBodies[bIdx].m_angVel;
	float invMassB = gBodies[bIdx].m_invMass;
	Matrix3x3 invInertiaB = gShapes[bIdx].m_invInertia;

	solveContact( ldsCs, posA, &linVelA, &angVelA, invMassA, invInertiaA,
			posB, &linVelB, &angVelB, invMassB, invInertiaB );

  if (gBodies[aIdx].m_invMass)
  {
		gBodies[aIdx].m_linVel = linVelA;
		gBodies[aIdx].m_angVel = angVelA;
	} else
	{
		gBodies[aIdx].m_linVel = mymake_float4(0,0,0,0);
		gBodies[aIdx].m_angVel = mymake_float4(0,0,0,0);
	
	}
	if (gBodies[bIdx].m_invMass)
  {
		gBodies[bIdx].m_linVel = linVelB;
		gBodies[bIdx].m_angVel = angVelB;
	} else
	{
		gBodies[bIdx].m_linVel = mymake_float4(0,0,0,0);
		gBodies[bIdx].m_angVel = mymake_float4(0,0,0,0);
	
	}

}



typedef struct 
{
	int m_valInt0;
	int m_valInt1;
	int m_valInt2;
	int m_valInt3;

	float m_val0;
	float m_val1;
	float m_val2;
	float m_val3;
} SolverDebugInfo;




__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void BatchSolveKernelContact(__global Body* gBodies,
                      __global Shape* gShapes,
                      __global Constraint4* gConstraints,
                      __global int* gN,
                      __global int* gOffsets,
                      __global	int* batchSizes,
                       int maxBatch1,
                       int cellBatch,
                       int4 nSplit
                      )
{
	//__local int ldsBatchIdx[WG_SIZE+1];
	__local int ldsCurBatch;
	__local int ldsNextBatch;
	__local int ldsStart;

	int lIdx = GET_LOCAL_IDX;
	int wgIdx = GET_GROUP_IDX;

//	int gIdx = GET_GLOBAL_IDX;
//	debugInfo[gIdx].m_valInt0 = gIdx;
	//debugInfo[gIdx].m_valInt1 = GET_GROUP_SIZE;

	
	

	int zIdx = (wgIdx/((nSplit.x*nSplit.y)/4))*2+((cellBatch&4)>>2);
	int remain= (wgIdx%((nSplit.x*nSplit.y)/4));
	int yIdx = (remain/(nSplit.x/2))*2 + ((cellBatch&2)>>1);
	int xIdx = (remain%(nSplit.x/2))*2 + (cellBatch&1);
	int cellIdx = xIdx+yIdx*nSplit.x+zIdx*(nSplit.x*nSplit.y);

	//int xIdx = (wgIdx/(nSplit/2))*2 + (bIdx&1);
	//int yIdx = (wgIdx%(nSplit/2))*2 + (bIdx>>1);
	//int cellIdx = xIdx+yIdx*nSplit;
	
	if( gN[cellIdx] == 0 ) 
		return;

	int maxBatch = batchSizes[cellIdx];
	
	
	const int start = gOffsets[cellIdx];
	const int end = start + gN[cellIdx];

	
	
	
	if( lIdx == 0 )
	{
		ldsCurBatch = 0;
		ldsNextBatch = 0;
		ldsStart = start;
	}


	GROUP_LDS_BARRIER;

	int idx=ldsStart+lIdx;
	while (ldsCurBatch < maxBatch)
	{
		for(; idx<end; )
		{
			if (gConstraints[idx].m_batchIdx == ldsCurBatch)
			{
					solveContactConstraint( gBodies, gShapes, &gConstraints[idx] );

				 idx+=64;
			} else
			{
				break;
			}
		}
		GROUP_LDS_BARRIER;
	
		if( lIdx == 0 )
		{
			ldsCurBatch++;
		}
		GROUP_LDS_BARRIER;
	}
	
    
}



__kernel void solveSingleContactKernel(__global Body* gBodies,
                      __global Shape* gShapes,
                      __global Constraint4* gConstraints,
                       int cellIdx,
                       int batchOffset,
                       int numConstraintsInBatch
                      )
{

	int index = get_global_id(0);
	if (index < numConstraintsInBatch)
	{
		int idx=batchOffset+index;
		solveContactConstraint( gBodies, gShapes, &gConstraints[idx] );
	}    
}
