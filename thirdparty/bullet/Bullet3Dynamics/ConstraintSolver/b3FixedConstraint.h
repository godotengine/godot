
#ifndef B3_FIXED_CONSTRAINT_H
#define B3_FIXED_CONSTRAINT_H

#include "b3TypedConstraint.h"

B3_ATTRIBUTE_ALIGNED16(class)
b3FixedConstraint : public b3TypedConstraint
{
	b3Vector3 m_pivotInA;
	b3Vector3 m_pivotInB;
	b3Quaternion m_relTargetAB;

public:
	b3FixedConstraint(int rbA, int rbB, const b3Transform& frameInA, const b3Transform& frameInB);

	virtual ~b3FixedConstraint();

	virtual void getInfo1(b3ConstraintInfo1 * info, const b3RigidBodyData* bodies);

	virtual void getInfo2(b3ConstraintInfo2 * info, const b3RigidBodyData* bodies);

	virtual void setParam(int num, b3Scalar value, int axis = -1)
	{
		b3Assert(0);
	}
	virtual b3Scalar getParam(int num, int axis = -1) const
	{
		b3Assert(0);
		return 0.f;
	}
};

#endif  //B3_FIXED_CONSTRAINT_H
