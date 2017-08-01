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

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"

#pragma OPENCL EXTENSION cl_amd_printf : enable
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

#define make_float4 (float4)
#define make_float2 (float2)
#define make_uint4 (uint4)
#define make_int4 (int4)
#define make_uint2 (uint2)
#define make_int2 (int2)


#define max2 max
#define min2 min


///////////////////////////////////////
//	Vector
///////////////////////////////////////
__inline
float fastDiv(float numerator, float denominator)
{
	return native_divide(numerator, denominator);	
//	return numerator/denominator;	
}

__inline
float4 fastDiv4(float4 numerator, float4 denominator)
{
	return native_divide(numerator, denominator);	
}

__inline
float fastSqrtf(float f2)
{
	return native_sqrt(f2);
//	return sqrt(f2);
}

__inline
float fastRSqrt(float f2)
{
	return native_rsqrt(f2);
}

__inline
float fastLength4(float4 v)
{
	return fast_length(v);
}

__inline
float4 fastNormalize4(float4 v)
{
	return fast_normalize(v);
}


__inline
float sqrtf(float a)
{
//	return sqrt(a);
	return native_sqrt(a);
}

__inline
float4 cross3(float4 a1, float4 b1)
{

	float4 	a=make_float4(a1.xyz,0.f);
	float4 	b=make_float4(b1.xyz,0.f);
	//float4 	a=a1;
	//float4 	b=b1;
	return cross(a,b);
}

__inline
float dot3F4(float4 a, float4 b)
{
	float4 a1 = make_float4(a.xyz,0.f);
	float4 b1 = make_float4(b.xyz,0.f);
	return dot(a1, b1);
}

__inline
float length3(const float4 a)
{
	return sqrtf(dot3F4(a,a));
}

__inline
float dot4(const float4 a, const float4 b)
{
	return dot( a, b );
}

//	for height
__inline
float dot3w1(const float4 point, const float4 eqn)
{
	return dot3F4(point,eqn) + eqn.w;
}

__inline
float4 normalize3(const float4 a)
{
	float4 n = make_float4(a.x, a.y, a.z, 0.f);
	return fastNormalize4( n );
//	float length = sqrtf(dot3F4(a, a));
//	return 1.f/length * a;
}

__inline
float4 normalize4(const float4 a)
{
	float length = sqrtf(dot4(a, a));
	return 1.f/length * a;
}

__inline
float4 createEquation(const float4 a, const float4 b, const float4 c)
{
	float4 eqn;
	float4 ab = b-a;
	float4 ac = c-a;
	eqn = normalize3( cross3(ab, ac) );
	eqn.w = -dot3F4(eqn,a);
	return eqn;
}

///////////////////////////////////////
//	Matrix3x3
///////////////////////////////////////

typedef struct
{
	float4 m_row[3];
}Matrix3x3;

__inline
Matrix3x3 mtZero();

__inline
Matrix3x3 mtIdentity();

__inline
Matrix3x3 mtTranspose(Matrix3x3 m);

__inline
Matrix3x3 mtMul(Matrix3x3 a, Matrix3x3 b);

__inline
float4 mtMul1(Matrix3x3 a, float4 b);

__inline
float4 mtMul3(float4 a, Matrix3x3 b);

__inline
Matrix3x3 mtZero()
{
	Matrix3x3 m;
	m.m_row[0] = (float4)(0.f);
	m.m_row[1] = (float4)(0.f);
	m.m_row[2] = (float4)(0.f);
	return m;
}

__inline
Matrix3x3 mtIdentity()
{
	Matrix3x3 m;
	m.m_row[0] = (float4)(1,0,0,0);
	m.m_row[1] = (float4)(0,1,0,0);
	m.m_row[2] = (float4)(0,0,1,0);
	return m;
}

__inline
Matrix3x3 mtTranspose(Matrix3x3 m)
{
	Matrix3x3 out;
	out.m_row[0] = (float4)(m.m_row[0].x, m.m_row[1].x, m.m_row[2].x, 0.f);
	out.m_row[1] = (float4)(m.m_row[0].y, m.m_row[1].y, m.m_row[2].y, 0.f);
	out.m_row[2] = (float4)(m.m_row[0].z, m.m_row[1].z, m.m_row[2].z, 0.f);
	return out;
}

__inline
Matrix3x3 mtMul(Matrix3x3 a, Matrix3x3 b)
{
	Matrix3x3 transB;
	transB = mtTranspose( b );
	Matrix3x3 ans;
	//	why this doesn't run when 0ing in the for{}
	a.m_row[0].w = 0.f;
	a.m_row[1].w = 0.f;
	a.m_row[2].w = 0.f;
	for(int i=0; i<3; i++)
	{
//	a.m_row[i].w = 0.f;
		ans.m_row[i].x = dot3F4(a.m_row[i],transB.m_row[0]);
		ans.m_row[i].y = dot3F4(a.m_row[i],transB.m_row[1]);
		ans.m_row[i].z = dot3F4(a.m_row[i],transB.m_row[2]);
		ans.m_row[i].w = 0.f;
	}
	return ans;
}

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
	float4 colx = make_float4(b.m_row[0].x, b.m_row[1].x, b.m_row[2].x, 0);
	float4 coly = make_float4(b.m_row[0].y, b.m_row[1].y, b.m_row[2].y, 0);
	float4 colz = make_float4(b.m_row[0].z, b.m_row[1].z, b.m_row[2].z, 0);

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

__inline
Quaternion qtMul(Quaternion a, Quaternion b);

__inline
Quaternion qtNormalize(Quaternion in);

__inline
float4 qtRotate(Quaternion q, float4 vec);

__inline
Quaternion qtInvert(Quaternion q);





__inline
Quaternion qtMul(Quaternion a, Quaternion b)
{
	Quaternion ans;
	ans = cross3( a, b );
	ans += a.w*b+b.w*a;
//	ans.w = a.w*b.w - (a.x*b.x+a.y*b.y+a.z*b.z);
	ans.w = a.w*b.w - dot3F4(a, b);
	return ans;
}

__inline
Quaternion qtNormalize(Quaternion in)
{
	return fastNormalize4(in);
//	in /= length( in );
//	return in;
}
__inline
float4 qtRotate(Quaternion q, float4 vec)
{
	Quaternion qInv = qtInvert( q );
	float4 vcpy = vec;
	vcpy.w = 0.f;
	float4 out = qtMul(qtMul(q,vcpy),qInv);
	return out;
}

__inline
Quaternion qtInvert(Quaternion q)
{
	return (Quaternion)(-q.xyz, q.w);
}

__inline
float4 qtInvRotate(const Quaternion q, float4 vec)
{
	return qtRotate( qtInvert( q ), vec );
}




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
	u32 m_paddings;
} Constraint4;






__kernel void CountBodiesKernel(__global struct b3Contact4Data* manifoldPtr, __global unsigned int* bodyCount, __global int2* contactConstraintOffsets, int numContactManifolds, int fixedBodyIndex)
{
	int i = GET_GLOBAL_IDX;
	
	if( i < numContactManifolds)
	{
		int pa = manifoldPtr[i].m_bodyAPtrAndSignBit;
		bool isFixedA = (pa <0) || (pa == fixedBodyIndex);
		int bodyIndexA = abs(pa);
		if (!isFixedA)
		{
			 AtomInc1(bodyCount[bodyIndexA],contactConstraintOffsets[i].x);
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		int pb = manifoldPtr[i].m_bodyBPtrAndSignBit;
		bool isFixedB = (pb <0) || (pb == fixedBodyIndex);
		int bodyIndexB = abs(pb);
		if (!isFixedB)
		{
			AtomInc1(bodyCount[bodyIndexB],contactConstraintOffsets[i].y);
		} 
	}
}

__kernel void ClearVelocitiesKernel(__global float4* linearVelocities,__global float4* angularVelocities, int numSplitBodies)
{
	int i = GET_GLOBAL_IDX;
	
	if( i < numSplitBodies)
	{
		linearVelocities[i] = make_float4(0);
		angularVelocities[i] = make_float4(0);
	}
}


__kernel void AverageVelocitiesKernel(__global Body* gBodies,__global int* offsetSplitBodies,__global const unsigned int* bodyCount,
__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities, int numBodies)
{
	int i = GET_GLOBAL_IDX;
	if (i<numBodies)
	{
		if (gBodies[i].m_invMass)
		{
			int bodyOffset = offsetSplitBodies[i];
			int count = bodyCount[i];
			float factor = 1.f/((float)count);
			float4 averageLinVel = make_float4(0.f);
			float4 averageAngVel = make_float4(0.f);
			
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
			
		}//bodies[i].m_invMass
	}//i<numBodies
}



void setLinearAndAngular( float4 n, float4 r0, float4 r1, float4* linear, float4* angular0, float4* angular1)
{
	*linear = make_float4(n.xyz,0.f);
	*angular0 = cross3(r0, n);
	*angular1 = -cross3(r1, n);
}


float calcRelVel( float4 l0, float4 l1, float4 a0, float4 a1, float4 linVel0, float4 angVel0, float4 linVel1, float4 angVel1 )
{
	return dot3F4(l0, linVel0) + dot3F4(a0, angVel0) + dot3F4(l1, linVel1) + dot3F4(a1, angVel1);
}


float calcJacCoeff(const float4 linear0, const float4 linear1, const float4 angular0, const float4 angular1,
					float invMass0, const Matrix3x3* invInertia0, float invMass1, const Matrix3x3* invInertia1, float countA, float countB)
{
	//	linear0,1 are normlized
	float jmj0 = invMass0;//dot3F4(linear0, linear0)*invMass0;
	float jmj1 = dot3F4(mtMul3(angular0,*invInertia0), angular0);
	float jmj2 = invMass1;//dot3F4(linear1, linear1)*invMass1;
	float jmj3 = dot3F4(mtMul3(angular1,*invInertia1), angular1);
	return -1.f/((jmj0+jmj1)*countA+(jmj2+jmj3)*countB);
}


void btPlaneSpace1 (float4 n, float4* p, float4* q);
 void btPlaneSpace1 (float4 n, float4* p, float4* q)
{
  if (fabs(n.z) > 0.70710678f) {
    // choose p in y-z plane
    float a = n.y*n.y + n.z*n.z;
    float k = 1.f/sqrt(a);
    p[0].x = 0;
	p[0].y = -n.z*k;
	p[0].z = n.y*k;
    // set q = n x p
    q[0].x = a*k;
	q[0].y = -n.x*p[0].z;
	q[0].z = n.x*p[0].y;
  }
  else {
    // choose p in x-y plane
    float a = n.x*n.x + n.y*n.y;
    float k = 1.f/sqrt(a);
    p[0].x = -n.y*k;
	p[0].y = n.x*k;
	p[0].z = 0;
    // set q = n x p
    q[0].x = -n.z*p[0].y;
	q[0].y = n.z*p[0].x;
	q[0].z = a*k;
  }
}





void solveContact(__global Constraint4* cs,
			float4 posA, float4* linVelA, float4* angVelA, float invMassA, Matrix3x3 invInertiaA,
			float4 posB, float4* linVelB, float4* angVelB, float invMassB, Matrix3x3 invInertiaB,
			float4* dLinVelA, float4* dAngVelA, float4* dLinVelB, float4* dAngVelB)
{
	float minRambdaDt = 0;
	float maxRambdaDt = FLT_MAX;

	for(int ic=0; ic<4; ic++)
	{
		if( cs->m_jacCoeffInv[ic] == 0.f ) continue;

		float4 angular0, angular1, linear;
		float4 r0 = cs->m_worldPos[ic] - posA;
		float4 r1 = cs->m_worldPos[ic] - posB;
		setLinearAndAngular( cs->m_linear, r0, r1, &linear, &angular0, &angular1 );
	


		float rambdaDt = calcRelVel( cs->m_linear, -cs->m_linear, angular0, angular1, 
			*linVelA+*dLinVelA, *angVelA+*dAngVelA, *linVelB+*dLinVelB, *angVelB+*dAngVelB ) + cs->m_b[ic];
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


//	solveContactConstraint( gBodies, gShapes, &gConstraints[i] ,contactConstraintOffsets,offsetSplitBodies, deltaLinearVelocities, deltaAngularVelocities);


void solveContactConstraint(__global Body* gBodies, __global Shape* gShapes, __global Constraint4* ldsCs, 
__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,
__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities)
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

			
	float4 dLinVelA = make_float4(0,0,0,0);
	float4 dAngVelA = make_float4(0,0,0,0);
	float4 dLinVelB = make_float4(0,0,0,0);
	float4 dAngVelB = make_float4(0,0,0,0);
			
	int bodyOffsetA = offsetSplitBodies[aIdx];
	int constraintOffsetA = contactConstraintOffsets[0].x;
	int splitIndexA = bodyOffsetA+constraintOffsetA;
	
	if (invMassA)
	{
		dLinVelA = deltaLinearVelocities[splitIndexA];
		dAngVelA = deltaAngularVelocities[splitIndexA];
	}

	int bodyOffsetB = offsetSplitBodies[bIdx];
	int constraintOffsetB = contactConstraintOffsets[0].y;
	int splitIndexB= bodyOffsetB+constraintOffsetB;

	if (invMassB)
	{
		dLinVelB = deltaLinearVelocities[splitIndexB];
		dAngVelB = deltaAngularVelocities[splitIndexB];
	}

	solveContact( ldsCs, posA, &linVelA, &angVelA, invMassA, invInertiaA,
			posB, &linVelB, &angVelB, invMassB, invInertiaB ,&dLinVelA, &dAngVelA, &dLinVelB, &dAngVelB);

	if (invMassA)
	{
		deltaLinearVelocities[splitIndexA] = dLinVelA;
		deltaAngularVelocities[splitIndexA] = dAngVelA;
	} 
	if (invMassB)
	{
		deltaLinearVelocities[splitIndexB] = dLinVelB;
		deltaAngularVelocities[splitIndexB] = dAngVelB;
	}

}


__kernel void SolveContactJacobiKernel(__global Constraint4* gConstraints, __global Body* gBodies, __global Shape* gShapes ,
__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities,
float deltaTime, float positionDrift, float positionConstraintCoeff, int fixedBodyIndex, int numManifolds
)
{
	int i = GET_GLOBAL_IDX;
	if (i<numManifolds)
	{
		solveContactConstraint( gBodies, gShapes, &gConstraints[i] ,&contactConstraintOffsets[i],offsetSplitBodies, deltaLinearVelocities, deltaAngularVelocities);
	}
}




void solveFrictionConstraint(__global Body* gBodies, __global Shape* gShapes, __global Constraint4* ldsCs,
							__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,
							__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities)
{
	float frictionCoeff = 0.7f;//ldsCs[0].m_linear.w;
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
	

	float4 dLinVelA = make_float4(0,0,0,0);
	float4 dAngVelA = make_float4(0,0,0,0);
	float4 dLinVelB = make_float4(0,0,0,0);
	float4 dAngVelB = make_float4(0,0,0,0);
			
	int bodyOffsetA = offsetSplitBodies[aIdx];
	int constraintOffsetA = contactConstraintOffsets[0].x;
	int splitIndexA = bodyOffsetA+constraintOffsetA;
	
	if (invMassA)
	{
		dLinVelA = deltaLinearVelocities[splitIndexA];
		dAngVelA = deltaAngularVelocities[splitIndexA];
	}

	int bodyOffsetB = offsetSplitBodies[bIdx];
	int constraintOffsetB = contactConstraintOffsets[0].y;
	int splitIndexB= bodyOffsetB+constraintOffsetB;

	if (invMassB)
	{
		dLinVelB = deltaLinearVelocities[splitIndexB];
		dAngVelB = deltaAngularVelocities[splitIndexB];
	}




	{
		float maxRambdaDt[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
		float minRambdaDt[4] = {0.f,0.f,0.f,0.f};

		float sum = 0;
		for(int j=0; j<4; j++)
		{
			sum +=ldsCs[0].m_appliedRambdaDt[j];
		}
		frictionCoeff = 0.7f;
		for(int j=0; j<4; j++)
		{
			maxRambdaDt[j] = frictionCoeff*sum;
			minRambdaDt[j] = -maxRambdaDt[j];
		}

		
//		solveFriction( ldsCs, posA, &linVelA, &angVelA, invMassA, invInertiaA,
//			posB, &linVelB, &angVelB, invMassB, invInertiaB, maxRambdaDt, minRambdaDt );
		
		
		{
			
			__global Constraint4* cs = ldsCs;
			
			if( cs->m_fJacCoeffInv[0] == 0 && cs->m_fJacCoeffInv[0] == 0 ) return;
			const float4 center = cs->m_center;
			
			float4 n = -cs->m_linear;
			
			float4 tangent[2];
			btPlaneSpace1(n,&tangent[0],&tangent[1]);
			float4 angular0, angular1, linear;
			float4 r0 = center - posA;
			float4 r1 = center - posB;
			for(int i=0; i<2; i++)
			{
				setLinearAndAngular( tangent[i], r0, r1, &linear, &angular0, &angular1 );
				float rambdaDt = calcRelVel(linear, -linear, angular0, angular1,
											linVelA+dLinVelA, angVelA+dAngVelA, linVelB+dLinVelB, angVelB+dAngVelB );
				rambdaDt *= cs->m_fJacCoeffInv[i];
				
				{
					float prevSum = cs->m_fAppliedRambdaDt[i];
					float updated = prevSum;
					updated += rambdaDt;
					updated = max2( updated, minRambdaDt[i] );
					updated = min2( updated, maxRambdaDt[i] );
					rambdaDt = updated - prevSum;
					cs->m_fAppliedRambdaDt[i] = updated;
				}
				
				float4 linImp0 = invMassA*linear*rambdaDt;
				float4 linImp1 = invMassB*(-linear)*rambdaDt;
				float4 angImp0 = mtMul1(invInertiaA, angular0)*rambdaDt;
				float4 angImp1 = mtMul1(invInertiaB, angular1)*rambdaDt;
				
				dLinVelA += linImp0;
				dAngVelA += angImp0;
				dLinVelB += linImp1;
				dAngVelB += angImp1;
			}
			{	//	angular damping for point constraint
				float4 ab = normalize3( posB - posA );
				float4 ac = normalize3( center - posA );
				if( dot3F4( ab, ac ) > 0.95f  || (invMassA == 0.f || invMassB == 0.f))
				{
					float angNA = dot3F4( n, angVelA );
					float angNB = dot3F4( n, angVelB );
					
					dAngVelA -= (angNA*0.1f)*n;
					dAngVelB -= (angNB*0.1f)*n;
				}
			}
		}

		
		
	}

	if (invMassA)
	{
		deltaLinearVelocities[splitIndexA] = dLinVelA;
		deltaAngularVelocities[splitIndexA] = dAngVelA;
	} 
	if (invMassB)
	{
		deltaLinearVelocities[splitIndexB] = dLinVelB;
		deltaAngularVelocities[splitIndexB] = dAngVelB;
	}
 

}


__kernel void SolveFrictionJacobiKernel(__global Constraint4* gConstraints, __global Body* gBodies, __global Shape* gShapes ,
										__global int2* contactConstraintOffsets,__global unsigned int* offsetSplitBodies,
										__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities,
										float deltaTime, float positionDrift, float positionConstraintCoeff, int fixedBodyIndex, int numManifolds
)
{
	int i = GET_GLOBAL_IDX;
	if (i<numManifolds)
	{
		solveFrictionConstraint( gBodies, gShapes, &gConstraints[i] ,&contactConstraintOffsets[i],offsetSplitBodies, deltaLinearVelocities, deltaAngularVelocities);
	}
}


__kernel void UpdateBodyVelocitiesKernel(__global Body* gBodies,__global int* offsetSplitBodies,__global const unsigned int* bodyCount,
									__global float4* deltaLinearVelocities, __global float4* deltaAngularVelocities, int numBodies)
{
	int i = GET_GLOBAL_IDX;
	if (i<numBodies)
	{
		if (gBodies[i].m_invMass)
		{
			int bodyOffset = offsetSplitBodies[i];
			int count = bodyCount[i];
			if (count)
			{
				gBodies[i].m_linVel += deltaLinearVelocities[bodyOffset];
				gBodies[i].m_angVel += deltaAngularVelocities[bodyOffset];
			}
		}
	}
}



void setConstraint4( const float4 posA, const float4 linVelA, const float4 angVelA, float invMassA, const Matrix3x3 invInertiaA,
	const float4 posB, const float4 linVelB, const float4 angVelB, float invMassB, const Matrix3x3 invInertiaB, 
	__global struct b3Contact4Data* src, float dt, float positionDrift, float positionConstraintCoeff,float countA, float countB,
	Constraint4* dstC )
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
	dstC->m_linear.w = 0.7f ;//src->getFrictionCoeff() );
	for(int ic=0; ic<4; ic++)
	{
		float4 r0 = src->m_worldPosB[ic] - posA;
		float4 r1 = src->m_worldPosB[ic] - posB;

		if( ic >= src->m_worldNormalOnB.w )//npoints
		{
			dstC->m_jacCoeffInv[ic] = 0.f;
			continue;
		}

		float relVelN;
		{
			float4 linear, angular0, angular1;
			setLinearAndAngular(src->m_worldNormalOnB, r0, r1, &linear, &angular0, &angular1);

			dstC->m_jacCoeffInv[ic] = calcJacCoeff(linear, -linear, angular0, angular1,
				invMassA, &invInertiaA, invMassB, &invInertiaB , countA, countB);

			relVelN = calcRelVel(linear, -linear, angular0, angular1,
				linVelA, angVelA, linVelB, angVelB);

			float e = 0.f;//src->getRestituitionCoeff();
			if( relVelN*relVelN < 0.004f ) e = 0.f;

			dstC->m_b[ic] = e*relVelN;
			//float penetration = src->m_worldPosB[ic].w;
			dstC->m_b[ic] += (src->m_worldPosB[ic].w + positionDrift)*positionConstraintCoeff*dtInv;
			dstC->m_appliedRambdaDt[ic] = 0.f;
		}
	}

	if( src->m_worldNormalOnB.w > 0 )//npoints
	{	//	prepare friction
		float4 center = make_float4(0.f);
		for(int i=0; i<src->m_worldNormalOnB.w; i++) 
			center += src->m_worldPosB[i];
		center /= (float)src->m_worldNormalOnB.w;

		float4 tangent[2];
		btPlaneSpace1(-src->m_worldNormalOnB,&tangent[0],&tangent[1]);
		
		float4 r[2];
		r[0] = center - posA;
		r[1] = center - posB;

		for(int i=0; i<2; i++)
		{
			float4 linear, angular0, angular1;
			setLinearAndAngular(tangent[i], r[0], r[1], &linear, &angular0, &angular1);

			dstC->m_fJacCoeffInv[i] = calcJacCoeff(linear, -linear, angular0, angular1,
				invMassA, &invInertiaA, invMassB, &invInertiaB ,countA, countB);
			dstC->m_fAppliedRambdaDt[i] = 0.f;
		}
		dstC->m_center = center;
	}

	for(int i=0; i<4; i++)
	{
		if( i<src->m_worldNormalOnB.w )
		{
			dstC->m_worldPos[i] = src->m_worldPosB[i];
		}
		else
		{
			dstC->m_worldPos[i] = make_float4(0.f);
		}
	}
}


__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void ContactToConstraintSplitKernel(__global const struct b3Contact4Data* gContact, __global const Body* gBodies, __global const Shape* gShapes, __global Constraint4* gConstraintOut, 
__global const unsigned int* bodyCount,
int nContacts,
float dt,
float positionDrift,
float positionConstraintCoeff
)
{
	int gIdx = GET_GLOBAL_IDX;
	
	if( gIdx < nContacts )
	{
		int aIdx = abs(gContact[gIdx].m_bodyAPtrAndSignBit);
		int bIdx = abs(gContact[gIdx].m_bodyBPtrAndSignBit);

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

		Constraint4 cs;

		float countA = invMassA != 0.f ? (float)bodyCount[aIdx] : 1;
		float countB = invMassB != 0.f ? (float)bodyCount[bIdx] : 1;

    	setConstraint4( posA, linVelA, angVelA, invMassA, invInertiaA, posB, linVelB, angVelB, invMassB, invInertiaB,
			&gContact[gIdx], dt, positionDrift, positionConstraintCoeff,countA,countB,
			&cs  );
		
		cs.m_batchIdx = gContact[gIdx].m_batchIdx;

		gConstraintOut[gIdx] = cs;
	}
}