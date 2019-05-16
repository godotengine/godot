/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_MANIFOLD_RESULT_H
#define BT_MANIFOLD_RESULT_H

class btCollisionObject;
struct btCollisionObjectWrapper;

#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
class btManifoldPoint;

#include "BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h"

#include "LinearMath/btTransform.h"
#include "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"

typedef bool (*ContactAddedCallback)(btManifoldPoint& cp, const btCollisionObjectWrapper* colObj0Wrap, int partId0, int index0, const btCollisionObjectWrapper* colObj1Wrap, int partId1, int index1);
extern ContactAddedCallback gContactAddedCallback;

//#define DEBUG_PART_INDEX 1

/// These callbacks are used to customize the algorith that combine restitution, friction, damping, Stiffness
typedef btScalar (*CalculateCombinedCallback)(const btCollisionObject* body0, const btCollisionObject* body1);

extern CalculateCombinedCallback gCalculateCombinedRestitutionCallback;
extern CalculateCombinedCallback gCalculateCombinedFrictionCallback;
extern CalculateCombinedCallback gCalculateCombinedRollingFrictionCallback;
extern CalculateCombinedCallback gCalculateCombinedSpinningFrictionCallback;
extern CalculateCombinedCallback gCalculateCombinedContactDampingCallback;
extern CalculateCombinedCallback gCalculateCombinedContactStiffnessCallback;

///btManifoldResult is a helper class to manage  contact results.
class btManifoldResult : public btDiscreteCollisionDetectorInterface::Result
{
protected:
	btPersistentManifold* m_manifoldPtr;

	const btCollisionObjectWrapper* m_body0Wrap;
	const btCollisionObjectWrapper* m_body1Wrap;
	int m_partId0;
	int m_partId1;
	int m_index0;
	int m_index1;

public:
	btManifoldResult()
		:
#ifdef DEBUG_PART_INDEX

		  m_partId0(-1),
		  m_partId1(-1),
		  m_index0(-1),
		  m_index1(-1)
#endif  //DEBUG_PART_INDEX
			  m_closestPointDistanceThreshold(0)
	{
	}

	btManifoldResult(const btCollisionObjectWrapper* body0Wrap, const btCollisionObjectWrapper* body1Wrap);

	virtual ~btManifoldResult(){};

	void setPersistentManifold(btPersistentManifold* manifoldPtr)
	{
		m_manifoldPtr = manifoldPtr;
	}

	const btPersistentManifold* getPersistentManifold() const
	{
		return m_manifoldPtr;
	}
	btPersistentManifold* getPersistentManifold()
	{
		return m_manifoldPtr;
	}

	virtual void setShapeIdentifiersA(int partId0, int index0)
	{
		m_partId0 = partId0;
		m_index0 = index0;
	}

	virtual void setShapeIdentifiersB(int partId1, int index1)
	{
		m_partId1 = partId1;
		m_index1 = index1;
	}

	virtual void addContactPoint(const btVector3& normalOnBInWorld, const btVector3& pointInWorld, btScalar depth);

	SIMD_FORCE_INLINE void refreshContactPoints()
	{
		btAssert(m_manifoldPtr);
		if (!m_manifoldPtr->getNumContacts())
			return;

		bool isSwapped = m_manifoldPtr->getBody0() != m_body0Wrap->getCollisionObject();

		if (isSwapped)
		{
			m_manifoldPtr->refreshContactPoints(m_body1Wrap->getCollisionObject()->getWorldTransform(), m_body0Wrap->getCollisionObject()->getWorldTransform());
		}
		else
		{
			m_manifoldPtr->refreshContactPoints(m_body0Wrap->getCollisionObject()->getWorldTransform(), m_body1Wrap->getCollisionObject()->getWorldTransform());
		}
	}

	const btCollisionObjectWrapper* getBody0Wrap() const
	{
		return m_body0Wrap;
	}
	const btCollisionObjectWrapper* getBody1Wrap() const
	{
		return m_body1Wrap;
	}

	void setBody0Wrap(const btCollisionObjectWrapper* obj0Wrap)
	{
		m_body0Wrap = obj0Wrap;
	}

	void setBody1Wrap(const btCollisionObjectWrapper* obj1Wrap)
	{
		m_body1Wrap = obj1Wrap;
	}

	const btCollisionObject* getBody0Internal() const
	{
		return m_body0Wrap->getCollisionObject();
	}

	const btCollisionObject* getBody1Internal() const
	{
		return m_body1Wrap->getCollisionObject();
	}

	btScalar m_closestPointDistanceThreshold;

	/// in the future we can let the user override the methods to combine restitution and friction
	static btScalar calculateCombinedRestitution(const btCollisionObject* body0, const btCollisionObject* body1);
	static btScalar calculateCombinedFriction(const btCollisionObject* body0, const btCollisionObject* body1);
	static btScalar calculateCombinedRollingFriction(const btCollisionObject* body0, const btCollisionObject* body1);
	static btScalar calculateCombinedSpinningFriction(const btCollisionObject* body0, const btCollisionObject* body1);
	static btScalar calculateCombinedContactDamping(const btCollisionObject* body0, const btCollisionObject* body1);
	static btScalar calculateCombinedContactStiffness(const btCollisionObject* body0, const btCollisionObject* body1);
};

#endif  //BT_MANIFOLD_RESULT_H
