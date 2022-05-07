/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btTypedConstraint.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btSerializer.h"

#define DEFAULT_DEBUGDRAW_SIZE btScalar(0.05f)

btTypedConstraint::btTypedConstraint(btTypedConstraintType type, btRigidBody& rbA)
	: btTypedObject(type),
	  m_userConstraintType(-1),
	  m_userConstraintPtr((void*)-1),
	  m_breakingImpulseThreshold(SIMD_INFINITY),
	  m_isEnabled(true),
	  m_needsFeedback(false),
	  m_overrideNumSolverIterations(-1),
	  m_rbA(rbA),
	  m_rbB(getFixedBody()),
	  m_appliedImpulse(btScalar(0.)),
	  m_dbgDrawSize(DEFAULT_DEBUGDRAW_SIZE),
	  m_jointFeedback(0)
{
}

btTypedConstraint::btTypedConstraint(btTypedConstraintType type, btRigidBody& rbA, btRigidBody& rbB)
	: btTypedObject(type),
	  m_userConstraintType(-1),
	  m_userConstraintPtr((void*)-1),
	  m_breakingImpulseThreshold(SIMD_INFINITY),
	  m_isEnabled(true),
	  m_needsFeedback(false),
	  m_overrideNumSolverIterations(-1),
	  m_rbA(rbA),
	  m_rbB(rbB),
	  m_appliedImpulse(btScalar(0.)),
	  m_dbgDrawSize(DEFAULT_DEBUGDRAW_SIZE),
	  m_jointFeedback(0)
{
}

btScalar btTypedConstraint::getMotorFactor(btScalar pos, btScalar lowLim, btScalar uppLim, btScalar vel, btScalar timeFact)
{
	if (lowLim > uppLim)
	{
		return btScalar(1.0f);
	}
	else if (lowLim == uppLim)
	{
		return btScalar(0.0f);
	}
	btScalar lim_fact = btScalar(1.0f);
	btScalar delta_max = vel / timeFact;
	if (delta_max < btScalar(0.0f))
	{
		if ((pos >= lowLim) && (pos < (lowLim - delta_max)))
		{
			lim_fact = (lowLim - pos) / delta_max;
		}
		else if (pos < lowLim)
		{
			lim_fact = btScalar(0.0f);
		}
		else
		{
			lim_fact = btScalar(1.0f);
		}
	}
	else if (delta_max > btScalar(0.0f))
	{
		if ((pos <= uppLim) && (pos > (uppLim - delta_max)))
		{
			lim_fact = (uppLim - pos) / delta_max;
		}
		else if (pos > uppLim)
		{
			lim_fact = btScalar(0.0f);
		}
		else
		{
			lim_fact = btScalar(1.0f);
		}
	}
	else
	{
		lim_fact = btScalar(0.0f);
	}
	return lim_fact;
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
const char* btTypedConstraint::serialize(void* dataBuffer, btSerializer* serializer) const
{
	btTypedConstraintData2* tcd = (btTypedConstraintData2*)dataBuffer;

	tcd->m_rbA = (btRigidBodyData*)serializer->getUniquePointer(&m_rbA);
	tcd->m_rbB = (btRigidBodyData*)serializer->getUniquePointer(&m_rbB);
	char* name = (char*)serializer->findNameForPointer(this);
	tcd->m_name = (char*)serializer->getUniquePointer(name);
	if (tcd->m_name)
	{
		serializer->serializeName(name);
	}

	tcd->m_objectType = m_objectType;
	tcd->m_needsFeedback = m_needsFeedback;
	tcd->m_overrideNumSolverIterations = m_overrideNumSolverIterations;
	tcd->m_breakingImpulseThreshold = m_breakingImpulseThreshold;
	tcd->m_isEnabled = m_isEnabled ? 1 : 0;

	tcd->m_userConstraintId = m_userConstraintId;
	tcd->m_userConstraintType = m_userConstraintType;

	tcd->m_appliedImpulse = m_appliedImpulse;
	tcd->m_dbgDrawSize = m_dbgDrawSize;

	tcd->m_disableCollisionsBetweenLinkedBodies = false;

	int i;
	for (i = 0; i < m_rbA.getNumConstraintRefs(); i++)
		if (m_rbA.getConstraintRef(i) == this)
			tcd->m_disableCollisionsBetweenLinkedBodies = true;
	for (i = 0; i < m_rbB.getNumConstraintRefs(); i++)
		if (m_rbB.getConstraintRef(i) == this)
			tcd->m_disableCollisionsBetweenLinkedBodies = true;

	return btTypedConstraintDataName;
}

btRigidBody& btTypedConstraint::getFixedBody()
{
	static btRigidBody s_fixed(0, 0, 0);
	s_fixed.setMassProps(btScalar(0.), btVector3(btScalar(0.), btScalar(0.), btScalar(0.)));
	return s_fixed;
}

void btAngularLimit::set(btScalar low, btScalar high, btScalar _softness, btScalar _biasFactor, btScalar _relaxationFactor)
{
	m_halfRange = (high - low) / 2.0f;
	m_center = btNormalizeAngle(low + m_halfRange);
	m_softness = _softness;
	m_biasFactor = _biasFactor;
	m_relaxationFactor = _relaxationFactor;
}

void btAngularLimit::test(const btScalar angle)
{
	m_correction = 0.0f;
	m_sign = 0.0f;
	m_solveLimit = false;

	if (m_halfRange >= 0.0f)
	{
		btScalar deviation = btNormalizeAngle(angle - m_center);
		if (deviation < -m_halfRange)
		{
			m_solveLimit = true;
			m_correction = -(deviation + m_halfRange);
			m_sign = +1.0f;
		}
		else if (deviation > m_halfRange)
		{
			m_solveLimit = true;
			m_correction = m_halfRange - deviation;
			m_sign = -1.0f;
		}
	}
}

btScalar btAngularLimit::getError() const
{
	return m_correction * m_sign;
}

void btAngularLimit::fit(btScalar& angle) const
{
	if (m_halfRange > 0.0f)
	{
		btScalar relativeAngle = btNormalizeAngle(angle - m_center);
		if (!btEqual(relativeAngle, m_halfRange))
		{
			if (relativeAngle > 0.0f)
			{
				angle = getHigh();
			}
			else
			{
				angle = getLow();
			}
		}
	}
}

btScalar btAngularLimit::getLow() const
{
	return btNormalizeAngle(m_center - m_halfRange);
}

btScalar btAngularLimit::getHigh() const
{
	return btNormalizeAngle(m_center + m_halfRange);
}
