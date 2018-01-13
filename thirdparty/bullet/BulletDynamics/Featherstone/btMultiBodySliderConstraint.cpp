/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

///This file was written by Erwin Coumans

#include "btMultiBodySliderConstraint.h"
#include "btMultiBodyLinkCollider.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "BulletDynamics/ConstraintSolver/btGeneric6DofSpring2Constraint.h"
#include "LinearMath/btIDebugDraw.h"

#define BTMBSLIDERCONSTRAINT_DIM 5
#define EPSILON 0.000001

btMultiBodySliderConstraint::btMultiBodySliderConstraint(btMultiBody* body, int link, btRigidBody* bodyB, const btVector3& pivotInA, const btVector3& pivotInB, const btMatrix3x3& frameInA, const btMatrix3x3& frameInB, const btVector3& jointAxis)
	:btMultiBodyConstraint(body,0,link,-1,BTMBSLIDERCONSTRAINT_DIM,false),
	m_rigidBodyA(0),
	m_rigidBodyB(bodyB),
	m_pivotInA(pivotInA),
	m_pivotInB(pivotInB),
    m_frameInA(frameInA),
    m_frameInB(frameInB),
    m_jointAxis(jointAxis)
{
    m_data.resize(BTMBSLIDERCONSTRAINT_DIM);//at least store the applied impulses
}

btMultiBodySliderConstraint::btMultiBodySliderConstraint(btMultiBody* bodyA, int linkA, btMultiBody* bodyB, int linkB, const btVector3& pivotInA, const btVector3& pivotInB, const btMatrix3x3& frameInA, const btMatrix3x3& frameInB, const btVector3& jointAxis)
	:btMultiBodyConstraint(bodyA,bodyB,linkA,linkB,BTMBSLIDERCONSTRAINT_DIM,false),
	m_rigidBodyA(0),
	m_rigidBodyB(0),
	m_pivotInA(pivotInA),
	m_pivotInB(pivotInB),
    m_frameInA(frameInA),
    m_frameInB(frameInB),
    m_jointAxis(jointAxis)
{
    m_data.resize(BTMBSLIDERCONSTRAINT_DIM);//at least store the applied impulses
}

void btMultiBodySliderConstraint::finalizeMultiDof()
{
	//not implemented yet
	btAssert(0);
}

btMultiBodySliderConstraint::~btMultiBodySliderConstraint()
{
}


int btMultiBodySliderConstraint::getIslandIdA() const
{
	if (m_rigidBodyA)
		return m_rigidBodyA->getIslandTag();

	if (m_bodyA)
	{
		btMultiBodyLinkCollider* col = m_bodyA->getBaseCollider();
		if (col)
			return col->getIslandTag();
		for (int i=0;i<m_bodyA->getNumLinks();i++)
		{
			if (m_bodyA->getLink(i).m_collider)
				return m_bodyA->getLink(i).m_collider->getIslandTag();
		}
	}
	return -1;
}

int btMultiBodySliderConstraint::getIslandIdB() const
{
	if (m_rigidBodyB)
		return m_rigidBodyB->getIslandTag();
	if (m_bodyB)
	{
		btMultiBodyLinkCollider* col = m_bodyB->getBaseCollider();
		if (col)
			return col->getIslandTag();

		for (int i=0;i<m_bodyB->getNumLinks();i++)
		{
			col = m_bodyB->getLink(i).m_collider;
			if (col)
				return col->getIslandTag();
		}
	}
	return -1;
}

void btMultiBodySliderConstraint::createConstraintRows(btMultiBodyConstraintArray& constraintRows, btMultiBodyJacobianData& data, const btContactSolverInfo& infoGlobal)
{
    // Convert local points back to world
    btVector3 pivotAworld = m_pivotInA;
    btMatrix3x3 frameAworld = m_frameInA;
    btVector3 jointAxis = m_jointAxis;
    if (m_rigidBodyA)
    {
        pivotAworld = m_rigidBodyA->getCenterOfMassTransform()*m_pivotInA;
        frameAworld = m_frameInA.transpose()*btMatrix3x3(m_rigidBodyA->getOrientation());
        jointAxis = quatRotate(m_rigidBodyA->getOrientation(),m_jointAxis);
        
    } else if (m_bodyA) {
        pivotAworld = m_bodyA->localPosToWorld(m_linkA, m_pivotInA);
        frameAworld = m_bodyA->localFrameToWorld(m_linkA, m_frameInA);
        jointAxis = m_bodyA->localDirToWorld(m_linkA, m_jointAxis);
    }
    btVector3 pivotBworld = m_pivotInB;
    btMatrix3x3 frameBworld = m_frameInB;
    if (m_rigidBodyB)
    {
        pivotBworld = m_rigidBodyB->getCenterOfMassTransform()*m_pivotInB;
        frameBworld = m_frameInB.transpose()*btMatrix3x3(m_rigidBodyB->getOrientation());
        
    } else if (m_bodyB) {
        pivotBworld = m_bodyB->localPosToWorld(m_linkB, m_pivotInB);
        frameBworld = m_bodyB->localFrameToWorld(m_linkB, m_frameInB);
    }
    
    btVector3 constraintAxis[2];
    for (int i = 0; i < 3; ++i)
    {
        constraintAxis[0] = frameAworld.getColumn(i).cross(jointAxis);
        if (constraintAxis[0].safeNorm() > EPSILON)
        {
            constraintAxis[0] = constraintAxis[0].normalized();
            constraintAxis[1] = jointAxis.cross(constraintAxis[0]);
            constraintAxis[1] = constraintAxis[1].normalized();
            break;
        }
    }
    
    btMatrix3x3 relRot = frameAworld.inverse()*frameBworld;
    btVector3 angleDiff;
    btGeneric6DofSpring2Constraint::matrixToEulerXYZ(relRot,angleDiff);
    
    int numDim = BTMBSLIDERCONSTRAINT_DIM;
    for (int i=0;i<numDim;i++)
	{
        btMultiBodySolverConstraint& constraintRow = constraintRows.expandNonInitializing();
        constraintRow.m_orgConstraint = this;
        constraintRow.m_orgDofIndex = i;
        constraintRow.m_relpos1CrossNormal.setValue(0,0,0);
        constraintRow.m_contactNormal1.setValue(0,0,0);
        constraintRow.m_relpos2CrossNormal.setValue(0,0,0);
        constraintRow.m_contactNormal2.setValue(0,0,0);
        constraintRow.m_angularComponentA.setValue(0,0,0);
        constraintRow.m_angularComponentB.setValue(0,0,0);
        
        constraintRow.m_solverBodyIdA = data.m_fixedBodyId;
        constraintRow.m_solverBodyIdB = data.m_fixedBodyId;
        
        if (m_rigidBodyA)
        {
            constraintRow.m_solverBodyIdA = m_rigidBodyA->getCompanionId();
        }
        if (m_rigidBodyB)
        {
            constraintRow.m_solverBodyIdB = m_rigidBodyB->getCompanionId();
        }
        
        btVector3 constraintNormalLin(0,0,0);
        btVector3 constraintNormalAng(0,0,0);
        btScalar posError = 0.0;
        if (i < 2) {
            constraintNormalLin = constraintAxis[i];
            posError = (pivotAworld-pivotBworld).dot(constraintNormalLin);
            fillMultiBodyConstraint(constraintRow, data, 0, 0, constraintNormalAng,
                                    constraintNormalLin, pivotAworld, pivotBworld,
                                    posError,
                                    infoGlobal,
                                    -m_maxAppliedImpulse, m_maxAppliedImpulse
                                    );
        }
        else { //i>=2
            constraintNormalAng = frameAworld.getColumn(i%3);
            posError = angleDiff[i%3];
            fillMultiBodyConstraint(constraintRow, data, 0, 0, constraintNormalAng,
                                    constraintNormalLin, pivotAworld, pivotBworld,
                                    posError,
                                    infoGlobal,
                                    -m_maxAppliedImpulse, m_maxAppliedImpulse, true
                                    );
        }
	}
}

void btMultiBodySliderConstraint::debugDraw(class btIDebugDraw* drawer)
{
	btTransform tr;
	tr.setIdentity();

	if (m_rigidBodyA)
	{
		btVector3 pivot = m_rigidBodyA->getCenterOfMassTransform() * m_pivotInA;
		tr.setOrigin(pivot);
		drawer->drawTransform(tr, 0.1);
	}
	if (m_bodyA)
	{
		btVector3 pivotAworld = m_bodyA->localPosToWorld(m_linkA, m_pivotInA);
		tr.setOrigin(pivotAworld);
		drawer->drawTransform(tr, 0.1);
	}
	if (m_rigidBodyB)
	{
		// that ideally should draw the same frame
		btVector3 pivot = m_rigidBodyB->getCenterOfMassTransform() * m_pivotInB;
		tr.setOrigin(pivot);
		drawer->drawTransform(tr, 0.1);
	}
	if (m_bodyB)
	{
		btVector3 pivotBworld = m_bodyB->localPosToWorld(m_linkB, m_pivotInB);
		tr.setOrigin(pivotBworld);
		drawer->drawTransform(tr, 0.1);
	}
}
