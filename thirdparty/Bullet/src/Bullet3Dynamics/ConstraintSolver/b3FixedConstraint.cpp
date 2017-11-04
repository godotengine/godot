
#include "b3FixedConstraint.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Common/b3TransformUtil.h"
#include <new>


b3FixedConstraint::b3FixedConstraint(int rbA,int rbB, const b3Transform& frameInA,const b3Transform& frameInB)
:b3TypedConstraint(B3_FIXED_CONSTRAINT_TYPE,rbA,rbB)
{
	m_pivotInA = frameInA.getOrigin();
	m_pivotInB = frameInB.getOrigin();
	m_relTargetAB = frameInA.getRotation()*frameInB.getRotation().inverse();

}

b3FixedConstraint::~b3FixedConstraint ()
{
}

	
void b3FixedConstraint::getInfo1 (b3ConstraintInfo1* info,const b3RigidBodyData* bodies)
{
	info->m_numConstraintRows = 6;
	info->nub = 6;
}

void b3FixedConstraint::getInfo2 (b3ConstraintInfo2* info, const b3RigidBodyData* bodies)
{
	//fix the 3 linear degrees of freedom

	const b3Vector3& worldPosA = bodies[m_rbA].m_pos;
	const b3Quaternion& worldOrnA = bodies[m_rbA].m_quat;
	const b3Vector3& worldPosB= bodies[m_rbB].m_pos;
	const b3Quaternion& worldOrnB = bodies[m_rbB].m_quat;

	info->m_J1linearAxis[0] = 1;
	info->m_J1linearAxis[info->rowskip+1] = 1;
	info->m_J1linearAxis[2*info->rowskip+2] = 1;

	b3Vector3 a1 = b3QuatRotate(worldOrnA,m_pivotInA);
	{
		b3Vector3* angular0 = (b3Vector3*)(info->m_J1angularAxis);
		b3Vector3* angular1 = (b3Vector3*)(info->m_J1angularAxis+info->rowskip);
		b3Vector3* angular2 = (b3Vector3*)(info->m_J1angularAxis+2*info->rowskip);
		b3Vector3 a1neg = -a1;
		a1neg.getSkewSymmetricMatrix(angular0,angular1,angular2);
	}
    
	if (info->m_J2linearAxis)
	{
		info->m_J2linearAxis[0] = -1;
		info->m_J2linearAxis[info->rowskip+1] = -1;
		info->m_J2linearAxis[2*info->rowskip+2] = -1;
	}
	
	b3Vector3 a2 = b3QuatRotate(worldOrnB,m_pivotInB);
   
	{
	//	b3Vector3 a2n = -a2;
		b3Vector3* angular0 = (b3Vector3*)(info->m_J2angularAxis);
		b3Vector3* angular1 = (b3Vector3*)(info->m_J2angularAxis+info->rowskip);
		b3Vector3* angular2 = (b3Vector3*)(info->m_J2angularAxis+2*info->rowskip);
		a2.getSkewSymmetricMatrix(angular0,angular1,angular2);
	}

    // set right hand side for the linear dofs
	b3Scalar k = info->fps * info->erp;
	b3Vector3 linearError = k*(a2+worldPosB-a1-worldPosA);
    int j;
	for (j=0; j<3; j++)
    {
        info->m_constraintError[j*info->rowskip] = linearError[j];
		//printf("info->m_constraintError[%d]=%f\n",j,info->m_constraintError[j]);
    }

		//fix the 3 angular degrees of freedom

	int start_row = 3;
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


    // set right hand side for the angular dofs

	b3Vector3 diff;
	b3Scalar angle;
	b3Quaternion qrelCur = worldOrnA *worldOrnB.inverse();

	b3TransformUtil::calculateDiffAxisAngleQuaternion(m_relTargetAB,qrelCur,diff,angle);
	diff*=-angle;
	for (j=0; j<3; j++)
    {
        info->m_constraintError[(3+j)*info->rowskip] = k * diff[j];
    }

}