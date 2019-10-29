#ifndef B3_CONTACT_CONSTRAINT5_H
#define B3_CONTACT_CONSTRAINT5_H

#include "Bullet3Common/shared/b3Float4.h"

typedef struct b3ContactConstraint4 b3ContactConstraint4_t;

struct b3ContactConstraint4
{
	b3Float4 m_linear;  //normal?
	b3Float4 m_worldPos[4];
	b3Float4 m_center;  //	friction
	float m_jacCoeffInv[4];
	float m_b[4];
	float m_appliedRambdaDt[4];
	float m_fJacCoeffInv[2];      //	friction
	float m_fAppliedRambdaDt[2];  //	friction

	unsigned int m_bodyA;
	unsigned int m_bodyB;
	int m_batchIdx;
	unsigned int m_paddings;
};

//inline	void setFrictionCoeff(float value) { m_linear[3] = value; }
inline float b3GetFrictionCoeff(b3ContactConstraint4_t* constraint)
{
	return constraint->m_linear.w;
}

#endif  //B3_CONTACT_CONSTRAINT5_H
