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


#include "btManifoldResult.h"
#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"
#include "BulletCollision/CollisionDispatch/btCollisionObjectWrapper.h"

///This is to allow MaterialCombiner/Custom Friction/Restitution values
ContactAddedCallback		gContactAddedCallback=0;



btScalar	btManifoldResult::calculateCombinedRollingFriction(const btCollisionObject* body0,const btCollisionObject* body1)
{
	btScalar friction = body0->getRollingFriction() * body1->getFriction() + body1->getRollingFriction() * body0->getFriction();

	const btScalar MAX_FRICTION  = btScalar(10.);
	if (friction < -MAX_FRICTION)
		friction = -MAX_FRICTION;
	if (friction > MAX_FRICTION)
		friction = MAX_FRICTION;
	return friction;

}

btScalar	btManifoldResult::calculateCombinedSpinningFriction(const btCollisionObject* body0,const btCollisionObject* body1)
{
    btScalar friction = body0->getSpinningFriction() * body1->getFriction() + body1->getSpinningFriction() * body0->getFriction();
    
    const btScalar MAX_FRICTION  = btScalar(10.);
    if (friction < -MAX_FRICTION)
        friction = -MAX_FRICTION;
    if (friction > MAX_FRICTION)
        friction = MAX_FRICTION;
    return friction;
}

///User can override this material combiner by implementing gContactAddedCallback and setting body0->m_collisionFlags |= btCollisionObject::customMaterialCallback;
btScalar	btManifoldResult::calculateCombinedFriction(const btCollisionObject* body0,const btCollisionObject* body1)
{
	btScalar friction = body0->getFriction() * body1->getFriction();

	const btScalar MAX_FRICTION  = btScalar(10.);
	if (friction < -MAX_FRICTION)
		friction = -MAX_FRICTION;
	if (friction > MAX_FRICTION)
		friction = MAX_FRICTION;
	return friction;

}

btScalar	btManifoldResult::calculateCombinedRestitution(const btCollisionObject* body0,const btCollisionObject* body1)
{
	return body0->getRestitution() * body1->getRestitution();
}

btScalar	btManifoldResult::calculateCombinedContactDamping(const btCollisionObject* body0,const btCollisionObject* body1)
{
    return body0->getContactDamping() + body1->getContactDamping();
}

btScalar	btManifoldResult::calculateCombinedContactStiffness(const btCollisionObject* body0,const btCollisionObject* body1)
{
    
    btScalar s0 = body0->getContactStiffness();
    btScalar s1 = body1->getContactStiffness();
    
    btScalar tmp0 = btScalar(1)/s0;
    btScalar tmp1 = btScalar(1)/s1;
    btScalar combinedStiffness = btScalar(1) / (tmp0+tmp1);
    return combinedStiffness;
}


btManifoldResult::btManifoldResult(const btCollisionObjectWrapper* body0Wrap,const btCollisionObjectWrapper* body1Wrap)
		:m_manifoldPtr(0),
		m_body0Wrap(body0Wrap),
		m_body1Wrap(body1Wrap)
#ifdef DEBUG_PART_INDEX
		,m_partId0(-1),
	m_partId1(-1),
	m_index0(-1),
	m_index1(-1)
#endif //DEBUG_PART_INDEX
	, m_closestPointDistanceThreshold(0)
{
}


void btManifoldResult::addContactPoint(const btVector3& normalOnBInWorld,const btVector3& pointInWorld,btScalar depth)
{
	btAssert(m_manifoldPtr);
	//order in manifold needs to match

	if (depth > m_manifoldPtr->getContactBreakingThreshold())
//	if (depth > m_manifoldPtr->getContactProcessingThreshold())
		return;

	bool isSwapped = m_manifoldPtr->getBody0() != m_body0Wrap->getCollisionObject();
	bool isNewCollision = m_manifoldPtr->getNumContacts() == 0;

	btVector3 pointA = pointInWorld + normalOnBInWorld * depth;

	btVector3 localA;
	btVector3 localB;
	
	if (isSwapped)
	{
		localA = m_body1Wrap->getCollisionObject()->getWorldTransform().invXform(pointA );
		localB = m_body0Wrap->getCollisionObject()->getWorldTransform().invXform(pointInWorld);
	} else
	{
		localA = m_body0Wrap->getCollisionObject()->getWorldTransform().invXform(pointA );
		localB = m_body1Wrap->getCollisionObject()->getWorldTransform().invXform(pointInWorld);
	}

	btManifoldPoint newPt(localA,localB,normalOnBInWorld,depth);
	newPt.m_positionWorldOnA = pointA;
	newPt.m_positionWorldOnB = pointInWorld;
	
	int insertIndex = m_manifoldPtr->getCacheEntry(newPt);

	newPt.m_combinedFriction = calculateCombinedFriction(m_body0Wrap->getCollisionObject(),m_body1Wrap->getCollisionObject());
	newPt.m_combinedRestitution = calculateCombinedRestitution(m_body0Wrap->getCollisionObject(),m_body1Wrap->getCollisionObject());
	newPt.m_combinedRollingFriction = calculateCombinedRollingFriction(m_body0Wrap->getCollisionObject(),m_body1Wrap->getCollisionObject());
    newPt.m_combinedSpinningFriction = calculateCombinedSpinningFriction(m_body0Wrap->getCollisionObject(),m_body1Wrap->getCollisionObject());
	
	if (    (m_body0Wrap->getCollisionObject()->getCollisionFlags()& btCollisionObject::CF_HAS_CONTACT_STIFFNESS_DAMPING) ||
            (m_body1Wrap->getCollisionObject()->getCollisionFlags()& btCollisionObject::CF_HAS_CONTACT_STIFFNESS_DAMPING))
    {
        newPt.m_combinedContactDamping1 = calculateCombinedContactDamping(m_body0Wrap->getCollisionObject(),m_body1Wrap->getCollisionObject());
        newPt.m_combinedContactStiffness1 = calculateCombinedContactStiffness(m_body0Wrap->getCollisionObject(),m_body1Wrap->getCollisionObject());
        newPt.m_contactPointFlags |= BT_CONTACT_FLAG_CONTACT_STIFFNESS_DAMPING;
    }

	if (    (m_body0Wrap->getCollisionObject()->getCollisionFlags()& btCollisionObject::CF_HAS_FRICTION_ANCHOR) ||
            (m_body1Wrap->getCollisionObject()->getCollisionFlags()& btCollisionObject::CF_HAS_FRICTION_ANCHOR))
    {
        newPt.m_contactPointFlags |= BT_CONTACT_FLAG_FRICTION_ANCHOR;
    }

	btPlaneSpace1(newPt.m_normalWorldOnB,newPt.m_lateralFrictionDir1,newPt.m_lateralFrictionDir2);
	

	
   //BP mod, store contact triangles.
	if (isSwapped)
	{
		newPt.m_partId0 = m_partId1;
		newPt.m_partId1 = m_partId0;
		newPt.m_index0  = m_index1;
		newPt.m_index1  = m_index0;
	} else
	{
		newPt.m_partId0 = m_partId0;
		newPt.m_partId1 = m_partId1;
		newPt.m_index0  = m_index0;
		newPt.m_index1  = m_index1;
	}
	//printf("depth=%f\n",depth);
	///@todo, check this for any side effects
	if (insertIndex >= 0)
	{
		//const btManifoldPoint& oldPoint = m_manifoldPtr->getContactPoint(insertIndex);
		m_manifoldPtr->replaceContactPoint(newPt,insertIndex);
	} else
	{
		insertIndex = m_manifoldPtr->addManifoldPoint(newPt);
	}
	
	//User can override friction and/or restitution
	if (gContactAddedCallback &&
		//and if either of the two bodies requires custom material
		 ((m_body0Wrap->getCollisionObject()->getCollisionFlags() & btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK) ||
		   (m_body1Wrap->getCollisionObject()->getCollisionFlags() & btCollisionObject::CF_CUSTOM_MATERIAL_CALLBACK)))
	{
		//experimental feature info, for per-triangle material etc.
		const btCollisionObjectWrapper* obj0Wrap = isSwapped? m_body1Wrap : m_body0Wrap;
		const btCollisionObjectWrapper* obj1Wrap = isSwapped? m_body0Wrap : m_body1Wrap;
		(*gContactAddedCallback)(m_manifoldPtr->getContactPoint(insertIndex),obj0Wrap,newPt.m_partId0,newPt.m_index0,obj1Wrap,newPt.m_partId1,newPt.m_index1);
	}

	if (gContactStartedCallback && isNewCollision)
	{
		gContactStartedCallback(m_manifoldPtr);
	}
}

