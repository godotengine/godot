#ifndef B3_RIGIDBODY_DATA_H
#define B3_RIGIDBODY_DATA_H

#include "Bullet3Common/shared/b3Float4.h"
#include "Bullet3Common/shared/b3Quat.h"
#include "Bullet3Common/shared/b3Mat3x3.h"

typedef struct b3RigidBodyData b3RigidBodyData_t;

struct b3RigidBodyData
{
	b3Float4 m_pos;
	b3Quat m_quat;
	b3Float4 m_linVel;
	b3Float4 m_angVel;

	int m_collidableIdx;
	float m_invMass;
	float m_restituitionCoeff;
	float m_frictionCoeff;
};

typedef struct b3InertiaData b3InertiaData_t;

struct b3InertiaData
{
	b3Mat3x3 m_invInertiaWorld;
	b3Mat3x3 m_initInvInertia;
};

#endif  //B3_RIGIDBODY_DATA_H
