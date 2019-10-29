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

#define B3_CONSTRAINT_FLAG_ENABLED 1

#define B3_GPU_POINT2POINT_CONSTRAINT_TYPE 3
#define B3_GPU_FIXED_CONSTRAINT_TYPE 4

#define MOTIONCLAMP 100000 //unused, for debugging/safety in case constraint solver fails
#define B3_INFINITY 1e30f

#define mymake_float4 (float4)


__inline float dot3F4(float4 a, float4 b)
{
	float4 a1 = mymake_float4(a.xyz,0.f);
	float4 b1 = mymake_float4(b.xyz,0.f);
	return dot(a1, b1);
}


typedef float4 Quaternion;


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



typedef struct
{
	Matrix3x3 m_invInertiaWorld;
	Matrix3x3 m_initInvInertia;
} BodyInertia;


typedef struct
{
	Matrix3x3 m_basis;//orientation
	float4	m_origin;//transform
}b3Transform;

typedef struct
{
//	b3Transform		m_worldTransformUnused;
	float4		m_deltaLinearVelocity;
	float4		m_deltaAngularVelocity;
	float4		m_angularFactor;
	float4		m_linearFactor;
	float4		m_invMass;
	float4		m_pushVelocity;
	float4		m_turnVelocity;
	float4		m_linearVelocity;
	float4		m_angularVelocity;

	union 
	{
		void*	m_originalBody;
		int		m_originalBodyIndex;
	};
	int padding[3];

} b3GpuSolverBody;

typedef struct
{
	float4 m_pos;
	Quaternion m_quat;
	float4 m_linVel;
	float4 m_angVel;

	unsigned int m_shapeIdx;
	float m_invMass;
	float m_restituitionCoeff;
	float m_frictionCoeff;
} b3RigidBodyCL;

typedef struct
{

	float4		m_relpos1CrossNormal;
	float4		m_contactNormal;

	float4		m_relpos2CrossNormal;
	//float4		m_contactNormal2;//usually m_contactNormal2 == -m_contactNormal

	float4		m_angularComponentA;
	float4		m_angularComponentB;
	
	float	m_appliedPushImpulse;
	float	m_appliedImpulse;
	int	m_padding1;
	int	m_padding2;
	float	m_friction;
	float	m_jacDiagABInv;
	float		m_rhs;
	float		m_cfm;
	
    float		m_lowerLimit;
	float		m_upperLimit;
	float		m_rhsPenetration;
	int			m_originalConstraint;


	int	m_overrideNumSolverIterations;
    int			m_frictionIndex;
	int m_solverBodyIdA;
	int m_solverBodyIdB;

} b3SolverConstraint;

typedef struct 
{
	int m_bodyAPtrAndSignBit;
	int m_bodyBPtrAndSignBit;
	int m_originalConstraintIndex;
	int m_batchId;
} b3BatchConstraint;






typedef struct 
{
	int				m_constraintType;
	int				m_rbA;
	int				m_rbB;
	float			m_breakingImpulseThreshold;

	float4 m_pivotInA;
	float4 m_pivotInB;
	Quaternion m_relTargetAB;

	int	m_flags;
	int m_padding[3];
} b3GpuGenericConstraint;


/*b3Transform	getWorldTransform(b3RigidBodyCL* rb)
{
	b3Transform newTrans;
	newTrans.setOrigin(rb->m_pos);
	newTrans.setRotation(rb->m_quat);
	return newTrans;
}*/




__inline
float4 cross3(float4 a, float4 b)
{
	return cross(a,b);
}

__inline
float4 fastNormalize4(float4 v)
{
	v = mymake_float4(v.xyz,0.f);
	return fast_normalize(v);
}


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


__inline void internalApplyImpulse(__global b3GpuSolverBody* body,  float4 linearComponent, float4 angularComponent,float impulseMagnitude)
{
	body->m_deltaLinearVelocity += linearComponent*impulseMagnitude*body->m_linearFactor;
	body->m_deltaAngularVelocity += angularComponent*(impulseMagnitude*body->m_angularFactor);
}


void resolveSingleConstraintRowGeneric(__global b3GpuSolverBody* body1, __global b3GpuSolverBody* body2, __global b3SolverConstraint* c)
{
	float deltaImpulse = c->m_rhs-c->m_appliedImpulse*c->m_cfm;
	float deltaVel1Dotn	=	dot3F4(c->m_contactNormal,body1->m_deltaLinearVelocity) 	+ dot3F4(c->m_relpos1CrossNormal,body1->m_deltaAngularVelocity);
	float deltaVel2Dotn	=	-dot3F4(c->m_contactNormal,body2->m_deltaLinearVelocity) + dot3F4(c->m_relpos2CrossNormal,body2->m_deltaAngularVelocity);

	deltaImpulse	-=	deltaVel1Dotn*c->m_jacDiagABInv;
	deltaImpulse	-=	deltaVel2Dotn*c->m_jacDiagABInv;

	float sum = c->m_appliedImpulse + deltaImpulse;
	if (sum < c->m_lowerLimit)
	{
		deltaImpulse = c->m_lowerLimit-c->m_appliedImpulse;
		c->m_appliedImpulse = c->m_lowerLimit;
	}
	else if (sum > c->m_upperLimit) 
	{
		deltaImpulse = c->m_upperLimit-c->m_appliedImpulse;
		c->m_appliedImpulse = c->m_upperLimit;
	}
	else
	{
		c->m_appliedImpulse = sum;
	}

	internalApplyImpulse(body1,c->m_contactNormal*body1->m_invMass,c->m_angularComponentA,deltaImpulse);
	internalApplyImpulse(body2,-c->m_contactNormal*body2->m_invMass,c->m_angularComponentB,deltaImpulse);

}

__kernel void solveJointConstraintRows(__global b3GpuSolverBody* solverBodies,
					  __global b3BatchConstraint* batchConstraints,
					  	__global b3SolverConstraint* rows,
						__global unsigned int* numConstraintRowsInfo1, 
						__global unsigned int* rowOffsets,
						__global b3GpuGenericConstraint* constraints,
						int batchOffset,
						int numConstraintsInBatch
                      )
{
	int b = get_global_id(0);
	if (b>=numConstraintsInBatch)
		return;

	__global b3BatchConstraint* c = &batchConstraints[b+batchOffset];
	int originalConstraintIndex = c->m_originalConstraintIndex;
	if (constraints[originalConstraintIndex].m_flags&B3_CONSTRAINT_FLAG_ENABLED)
	{
		int numConstraintRows = numConstraintRowsInfo1[originalConstraintIndex];
		int rowOffset = rowOffsets[originalConstraintIndex];
		for (int jj=0;jj<numConstraintRows;jj++)
		{
			__global b3SolverConstraint* constraint = &rows[rowOffset+jj];
			resolveSingleConstraintRowGeneric(&solverBodies[constraint->m_solverBodyIdA],&solverBodies[constraint->m_solverBodyIdB],constraint);
		}
	}
};

__kernel void initSolverBodies(__global b3GpuSolverBody* solverBodies,__global b3RigidBodyCL* bodiesCL, int numBodies)
{
	int i = get_global_id(0);
	if (i>=numBodies)
		return;

	__global b3GpuSolverBody* solverBody = &solverBodies[i];
	__global b3RigidBodyCL* bodyCL = &bodiesCL[i];

	solverBody->m_deltaLinearVelocity = (float4)(0.f,0.f,0.f,0.f);
	solverBody->m_deltaAngularVelocity  = (float4)(0.f,0.f,0.f,0.f);
	solverBody->m_pushVelocity = (float4)(0.f,0.f,0.f,0.f);
	solverBody->m_pushVelocity = (float4)(0.f,0.f,0.f,0.f);
	solverBody->m_invMass = (float4)(bodyCL->m_invMass,bodyCL->m_invMass,bodyCL->m_invMass,0.f);
	solverBody->m_originalBodyIndex = i;
	solverBody->m_angularFactor = (float4)(1,1,1,0);
	solverBody->m_linearFactor = (float4) (1,1,1,0);
	solverBody->m_linearVelocity = bodyCL->m_linVel;
	solverBody->m_angularVelocity = bodyCL->m_angVel;
}

__kernel void breakViolatedConstraintsKernel(__global b3GpuGenericConstraint* constraints, __global unsigned int* numConstraintRows, __global unsigned int* rowOffsets, __global b3SolverConstraint* rows, int numConstraints)
{
	int cid = get_global_id(0);
	if (cid>=numConstraints)
		return;
	int numRows = numConstraintRows[cid];
	if (numRows)
	{
		for (int i=0;i<numRows;i++)
		{
			int rowIndex = rowOffsets[cid]+i;
			float breakingThreshold = constraints[cid].m_breakingImpulseThreshold;
			if (fabs(rows[rowIndex].m_appliedImpulse) >= breakingThreshold)
			{
				constraints[cid].m_flags =0;//&= ~B3_CONSTRAINT_FLAG_ENABLED;
			}
		}
	}
}



__kernel void getInfo1Kernel(__global unsigned int* infos, __global b3GpuGenericConstraint* constraints, int numConstraints)
{
	int i = get_global_id(0);
	if (i>=numConstraints)
		return;

	__global b3GpuGenericConstraint* constraint = &constraints[i];

	switch (constraint->m_constraintType)
	{
		case B3_GPU_POINT2POINT_CONSTRAINT_TYPE:
		{
			infos[i] = 3;
			break;
		}
		case B3_GPU_FIXED_CONSTRAINT_TYPE:
		{
			infos[i] = 6;
			break;
		}
		default:
		{
		}
	}
}

__kernel void initBatchConstraintsKernel(__global unsigned int* numConstraintRows, __global unsigned int* rowOffsets, 
										__global b3BatchConstraint* batchConstraints, 
										__global b3GpuGenericConstraint* constraints,
										__global b3RigidBodyCL* bodies,
										int numConstraints)
{
	int i = get_global_id(0);
	if (i>=numConstraints)
		return;

	int rbA = constraints[i].m_rbA;
	int rbB = constraints[i].m_rbB;

	batchConstraints[i].m_bodyAPtrAndSignBit = bodies[rbA].m_invMass != 0.f ? rbA : -rbA;
	batchConstraints[i].m_bodyBPtrAndSignBit = bodies[rbB].m_invMass != 0.f ? rbB : -rbB;
	batchConstraints[i].m_batchId = -1;
	batchConstraints[i].m_originalConstraintIndex = i;

}




typedef struct
{
	// integrator parameters: frames per second (1/stepsize), default error
	// reduction parameter (0..1).
	float fps,erp;

	// for the first and second body, pointers to two (linear and angular)
	// n*3 jacobian sub matrices, stored by rows. these matrices will have
	// been initialized to 0 on entry. if the second body is zero then the
	// J2xx pointers may be 0.
	union 
	{
		__global float4* m_J1linearAxisFloat4;
		__global float* m_J1linearAxis;
	};
	union
	{
		__global float4* m_J1angularAxisFloat4;
		__global float* m_J1angularAxis;

	};
	union
	{
	__global float4* m_J2linearAxisFloat4;
	__global float* m_J2linearAxis;
	};
	union
	{
		__global float4* m_J2angularAxisFloat4;
		__global float* m_J2angularAxis;
	};
	// elements to jump from one row to the next in J's
	int rowskip;

	// right hand sides of the equation J*v = c + cfm * lambda. cfm is the
	// "constraint force mixing" vector. c is set to zero on entry, cfm is
	// set to a constant value (typically very small or zero) value on entry.
	__global float* m_constraintError;
	__global float* cfm;

	// lo and hi limits for variables (set to -/+ infinity on entry).
	__global float* m_lowerLimit;
	__global float* m_upperLimit;

	// findex vector for variables. see the LCP solver interface for a
	// description of what this does. this is set to -1 on entry.
	// note that the returned indexes are relative to the first index of
	// the constraint.
	__global int *findex;
	// number of solver iterations
	int m_numIterations;

	//damping of the velocity
	float	m_damping;
} b3GpuConstraintInfo2;


void	getSkewSymmetricMatrix(float4 vecIn, __global float4* v0,__global float4* v1,__global float4* v2)
{
	*v0 = (float4)(0.		,-vecIn.z		,vecIn.y,0.f);
	*v1 = (float4)(vecIn.z	,0.			,-vecIn.x,0.f);
	*v2 = (float4)(-vecIn.y	,vecIn.x	,0.f,0.f);
}


void getInfo2Point2Point(__global b3GpuGenericConstraint* constraint,b3GpuConstraintInfo2* info,__global b3RigidBodyCL* bodies)
{
	float4 posA = bodies[constraint->m_rbA].m_pos;
	Quaternion rotA = bodies[constraint->m_rbA].m_quat;

	float4 posB = bodies[constraint->m_rbB].m_pos;
	Quaternion rotB = bodies[constraint->m_rbB].m_quat;



		// anchor points in global coordinates with respect to body PORs.
   
    // set jacobian
    info->m_J1linearAxis[0] = 1;
	info->m_J1linearAxis[info->rowskip+1] = 1;
	info->m_J1linearAxis[2*info->rowskip+2] = 1;

	float4 a1 = qtRotate(rotA,constraint->m_pivotInA);

	{
		__global float4* angular0 = (__global float4*)(info->m_J1angularAxis);
		__global float4* angular1 = (__global float4*)(info->m_J1angularAxis+info->rowskip);
		__global float4* angular2 = (__global float4*)(info->m_J1angularAxis+2*info->rowskip);
		float4 a1neg = -a1;
		getSkewSymmetricMatrix(a1neg,angular0,angular1,angular2);
	}
	if (info->m_J2linearAxis)
	{
		info->m_J2linearAxis[0] = -1;
		info->m_J2linearAxis[info->rowskip+1] = -1;
		info->m_J2linearAxis[2*info->rowskip+2] = -1;
	}
	
	float4 a2 = qtRotate(rotB,constraint->m_pivotInB);
   
	{
	//	float4 a2n = -a2;
		__global float4* angular0 = (__global float4*)(info->m_J2angularAxis);
		__global float4* angular1 = (__global float4*)(info->m_J2angularAxis+info->rowskip);
		__global float4* angular2 = (__global float4*)(info->m_J2angularAxis+2*info->rowskip);
		getSkewSymmetricMatrix(a2,angular0,angular1,angular2);
	}
    
    // set right hand side
//	float currERP = (m_flags & B3_P2P_FLAGS_ERP) ? m_erp : info->erp;
	float currERP = info->erp;

	float k = info->fps * currERP;
    int j;
	float4 result = a2 + posB - a1 - posA;
	float* resultPtr = &result;

	for (j=0; j<3; j++)
    {
        info->m_constraintError[j*info->rowskip] = k * (resultPtr[j]);
    }
}

Quaternion nearest( Quaternion first, Quaternion qd)
{
	Quaternion diff,sum;
	diff = first- qd;
	sum = first + qd;
	
	if( dot(diff,diff) < dot(sum,sum) )
		return qd;
	return (-qd);
}

float b3Acos(float x) 
{ 
	if (x<-1)	
		x=-1; 
	if (x>1)	
		x=1;
	return acos(x); 
}

float getAngle(Quaternion orn)
{
	if (orn.w>=1.f)
		orn.w=1.f;
	float s = 2.f * b3Acos(orn.w);
	return s;
}

void calculateDiffAxisAngleQuaternion( Quaternion orn0,Quaternion orn1a,float4* axis,float* angle)
{
	Quaternion orn1 = nearest(orn0,orn1a);
	
	Quaternion dorn = qtMul(orn1,qtInvert(orn0));
	*angle = getAngle(dorn);
	*axis = (float4)(dorn.x,dorn.y,dorn.z,0.f);
	
	//check for axis length
	float len = dot3F4(*axis,*axis);
	if (len < FLT_EPSILON*FLT_EPSILON)
		*axis = (float4)(1,0,0,0);
	else
		*axis /= sqrt(len);
}



void getInfo2FixedOrientation(__global b3GpuGenericConstraint* constraint,b3GpuConstraintInfo2* info,__global b3RigidBodyCL* bodies, int start_row)
{
	Quaternion worldOrnA = bodies[constraint->m_rbA].m_quat;
	Quaternion worldOrnB = bodies[constraint->m_rbB].m_quat;

	int s = info->rowskip;
	int start_index = start_row * s;

	// 3 rows to make body rotations equal
	info->m_J1angularAxis[start_index] = 1;
	info->m_J1angularAxis[start_index + s + 1] = 1;
	info->m_J1angularAxis[start_index + s*2+2] = 1;
	if ( info->m_J2angularAxis)
	{
		info->m_J2angularAxis[start_index] = -1;
		info->m_J2angularAxis[start_index + s+1] = -1;
		info->m_J2angularAxis[start_index + s*2+2] = -1;
	}
	
	float currERP = info->erp;
	float k = info->fps * currERP;
	float4 diff;
	float angle;
	float4 qrelCur = qtMul(worldOrnA,qtInvert(worldOrnB));
	
	calculateDiffAxisAngleQuaternion(constraint->m_relTargetAB,qrelCur,&diff,&angle);
	diff*=-angle;
		
	float* resultPtr = &diff;
	
	for (int j=0; j<3; j++)
    {
        info->m_constraintError[(3+j)*info->rowskip] = k * resultPtr[j];
    }
	

}


__kernel void writeBackVelocitiesKernel(__global b3RigidBodyCL* bodies,__global b3GpuSolverBody* solverBodies,int numBodies)
{
	int i = get_global_id(0);
	if (i>=numBodies)
		return;

	if (bodies[i].m_invMass)
	{
//		if (length(solverBodies[i].m_deltaLinearVelocity)<MOTIONCLAMP)
		{
			bodies[i].m_linVel += solverBodies[i].m_deltaLinearVelocity;
		}
//		if (length(solverBodies[i].m_deltaAngularVelocity)<MOTIONCLAMP)
		{
			bodies[i].m_angVel += solverBodies[i].m_deltaAngularVelocity;
		} 
	}
}


__kernel void getInfo2Kernel(__global b3SolverConstraint* solverConstraintRows, 
							__global unsigned int* infos, 
							__global unsigned int* constraintRowOffsets, 
							__global b3GpuGenericConstraint* constraints, 
							__global b3BatchConstraint* batchConstraints, 
							__global b3RigidBodyCL* bodies,
							__global BodyInertia* inertias,
							__global b3GpuSolverBody* solverBodies,
							float timeStep,
							float globalErp,
							float globalCfm,
							float globalDamping,
							int globalNumIterations,
							int numConstraints)
{

	int i = get_global_id(0);
	if (i>=numConstraints)
		return;
		
	//for now, always initialize the batch info
	int info1 = infos[i];
			
	__global b3SolverConstraint* currentConstraintRow = &solverConstraintRows[constraintRowOffsets[i]];
	__global b3GpuGenericConstraint* constraint = &constraints[i];

	__global b3RigidBodyCL* rbA = &bodies[ constraint->m_rbA];
	__global b3RigidBodyCL* rbB = &bodies[ constraint->m_rbB];

	int solverBodyIdA = constraint->m_rbA;
	int solverBodyIdB = constraint->m_rbB;

	__global b3GpuSolverBody* bodyAPtr = &solverBodies[solverBodyIdA];
	__global b3GpuSolverBody* bodyBPtr = &solverBodies[solverBodyIdB];


	if (rbA->m_invMass)
	{
		batchConstraints[i].m_bodyAPtrAndSignBit = solverBodyIdA;
	} else
	{
//			if (!solverBodyIdA)
//				m_staticIdx = 0;
		batchConstraints[i].m_bodyAPtrAndSignBit = -solverBodyIdA;
	}

	if (rbB->m_invMass)
	{
		batchConstraints[i].m_bodyBPtrAndSignBit = solverBodyIdB;
	} else
	{
//			if (!solverBodyIdB)
//				m_staticIdx = 0;
		batchConstraints[i].m_bodyBPtrAndSignBit = -solverBodyIdB;
	}

	if (info1)
	{
		int overrideNumSolverIterations = 0;//constraint->getOverrideNumSolverIterations() > 0 ? constraint->getOverrideNumSolverIterations() : infoGlobal.m_numIterations;
//		if (overrideNumSolverIterations>m_maxOverrideNumSolverIterations)
	//		m_maxOverrideNumSolverIterations = overrideNumSolverIterations;


		int j;
		for ( j=0;j<info1;j++)
		{
//			memset(&currentConstraintRow[j],0,sizeof(b3SolverConstraint));
			currentConstraintRow[j].m_angularComponentA = (float4)(0,0,0,0);
			currentConstraintRow[j].m_angularComponentB = (float4)(0,0,0,0);
			currentConstraintRow[j].m_appliedImpulse = 0.f;
			currentConstraintRow[j].m_appliedPushImpulse = 0.f;
			currentConstraintRow[j].m_cfm = 0.f;
			currentConstraintRow[j].m_contactNormal = (float4)(0,0,0,0);
			currentConstraintRow[j].m_friction = 0.f;
			currentConstraintRow[j].m_frictionIndex = 0;
			currentConstraintRow[j].m_jacDiagABInv = 0.f;
			currentConstraintRow[j].m_lowerLimit = 0.f;
			currentConstraintRow[j].m_upperLimit = 0.f;

			currentConstraintRow[j].m_originalConstraint = i;
			currentConstraintRow[j].m_overrideNumSolverIterations = 0;
			currentConstraintRow[j].m_relpos1CrossNormal = (float4)(0,0,0,0);
			currentConstraintRow[j].m_relpos2CrossNormal = (float4)(0,0,0,0);
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

		bodyAPtr->m_deltaLinearVelocity = (float4)(0,0,0,0);
		bodyAPtr->m_deltaAngularVelocity = (float4)(0,0,0,0);
		bodyAPtr->m_pushVelocity = (float4)(0,0,0,0);
		bodyAPtr->m_turnVelocity = (float4)(0,0,0,0);
		bodyBPtr->m_deltaLinearVelocity = (float4)(0,0,0,0);
		bodyBPtr->m_deltaAngularVelocity = (float4)(0,0,0,0);
		bodyBPtr->m_pushVelocity = (float4)(0,0,0,0);
		bodyBPtr->m_turnVelocity  = (float4)(0,0,0,0);

		int rowskip = sizeof(b3SolverConstraint)/sizeof(float);//check this

		


		b3GpuConstraintInfo2 info2;
		info2.fps = 1.f/timeStep;
		info2.erp = globalErp;
		info2.m_J1linearAxisFloat4 = &currentConstraintRow->m_contactNormal;
		info2.m_J1angularAxisFloat4 = &currentConstraintRow->m_relpos1CrossNormal;
		info2.m_J2linearAxisFloat4 = 0;
		info2.m_J2angularAxisFloat4 = &currentConstraintRow->m_relpos2CrossNormal;
		info2.rowskip = sizeof(b3SolverConstraint)/sizeof(float);//check this

		///the size of b3SolverConstraint needs be a multiple of float
//		b3Assert(info2.rowskip*sizeof(float)== sizeof(b3SolverConstraint));
		info2.m_constraintError = &currentConstraintRow->m_rhs;
		currentConstraintRow->m_cfm = globalCfm;
		info2.m_damping = globalDamping;
		info2.cfm = &currentConstraintRow->m_cfm;
		info2.m_lowerLimit = &currentConstraintRow->m_lowerLimit;
		info2.m_upperLimit = &currentConstraintRow->m_upperLimit;
		info2.m_numIterations = globalNumIterations;

		switch (constraint->m_constraintType)
		{
			case B3_GPU_POINT2POINT_CONSTRAINT_TYPE:
			{
				getInfo2Point2Point(constraint,&info2,bodies);
				break;
			}
			case B3_GPU_FIXED_CONSTRAINT_TYPE:
			{
				getInfo2Point2Point(constraint,&info2,bodies);

				getInfo2FixedOrientation(constraint,&info2,bodies,3);

				break;
			}

			default:
			{
			}
		}

		///finalize the constraint setup
		for ( j=0;j<info1;j++)
		{
			__global b3SolverConstraint* solverConstraint = &currentConstraintRow[j];

			if (solverConstraint->m_upperLimit>=constraint->m_breakingImpulseThreshold)
			{
				solverConstraint->m_upperLimit = constraint->m_breakingImpulseThreshold;
			}

			if (solverConstraint->m_lowerLimit<=-constraint->m_breakingImpulseThreshold)
			{
				solverConstraint->m_lowerLimit = -constraint->m_breakingImpulseThreshold;
			}

//						solverConstraint->m_originalContactPoint = constraint;
							
			Matrix3x3 invInertiaWorldA= inertias[constraint->m_rbA].m_invInertiaWorld;
			{

				//float4 angularFactorA(1,1,1);
				float4 ftorqueAxis1 = solverConstraint->m_relpos1CrossNormal;
				solverConstraint->m_angularComponentA = mtMul1(invInertiaWorldA,ftorqueAxis1);//*angularFactorA;
			}
						
			Matrix3x3 invInertiaWorldB= inertias[constraint->m_rbB].m_invInertiaWorld;
			{

				float4 ftorqueAxis2 = solverConstraint->m_relpos2CrossNormal;
				solverConstraint->m_angularComponentB = mtMul1(invInertiaWorldB,ftorqueAxis2);//*constraint->m_rbB.getAngularFactor();
			}

			{
				//it is ok to use solverConstraint->m_contactNormal instead of -solverConstraint->m_contactNormal
				//because it gets multiplied iMJlB
				float4 iMJlA = solverConstraint->m_contactNormal*rbA->m_invMass;
				float4 iMJaA = mtMul3(solverConstraint->m_relpos1CrossNormal,invInertiaWorldA);
				float4 iMJlB = solverConstraint->m_contactNormal*rbB->m_invMass;//sign of normal?
				float4 iMJaB = mtMul3(solverConstraint->m_relpos2CrossNormal,invInertiaWorldB);

				float sum = dot3F4(iMJlA,solverConstraint->m_contactNormal);
				sum += dot3F4(iMJaA,solverConstraint->m_relpos1CrossNormal);
				sum += dot3F4(iMJlB,solverConstraint->m_contactNormal);
				sum += dot3F4(iMJaB,solverConstraint->m_relpos2CrossNormal);
				float fsum = fabs(sum);
				if (fsum>FLT_EPSILON)
				{
					solverConstraint->m_jacDiagABInv = 1.f/sum;
				} else
				{
					solverConstraint->m_jacDiagABInv = 0.f;
				}
			}


			///fix rhs
			///todo: add force/torque accelerators
			{
				float rel_vel;
				float vel1Dotn = dot3F4(solverConstraint->m_contactNormal,rbA->m_linVel) + dot3F4(solverConstraint->m_relpos1CrossNormal,rbA->m_angVel);
				float vel2Dotn = -dot3F4(solverConstraint->m_contactNormal,rbB->m_linVel) + dot3F4(solverConstraint->m_relpos2CrossNormal,rbB->m_angVel);

				rel_vel = vel1Dotn+vel2Dotn;

				float restitution = 0.f;
				float positionalError = solverConstraint->m_rhs;//already filled in by getConstraintInfo2
				float	velocityError = restitution - rel_vel * info2.m_damping;
				float	penetrationImpulse = positionalError*solverConstraint->m_jacDiagABInv;
				float	velocityImpulse = velocityError *solverConstraint->m_jacDiagABInv;
				solverConstraint->m_rhs = penetrationImpulse+velocityImpulse;
				solverConstraint->m_appliedImpulse = 0.f;

			}
		}
	}
}
