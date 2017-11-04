/*
Bullet Continuous Collision Detection and Physics Library, http://bulletphysics.org
Copyright (C) 2006, 2007 Sony Computer Entertainment Inc. 

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/



#include "btHinge2Constraint.h"
#include "BulletDynamics/Dynamics/btRigidBody.h"
#include "LinearMath/btTransformUtil.h"



// constructor
// anchor, axis1 and axis2 are in world coordinate system
// axis1 must be orthogonal to axis2
btHinge2Constraint::btHinge2Constraint(btRigidBody& rbA, btRigidBody& rbB, btVector3& anchor, btVector3& axis1, btVector3& axis2)
: btGeneric6DofSpring2Constraint(rbA, rbB, btTransform::getIdentity(), btTransform::getIdentity(),RO_XYZ),
 m_anchor(anchor),
 m_axis1(axis1),
 m_axis2(axis2)
{
	// build frame basis
	// 6DOF constraint uses Euler angles and to define limits
	// it is assumed that rotational order is :
	// Z - first, allowed limits are (-PI,PI);
	// new position of Y - second (allowed limits are (-PI/2 + epsilon, PI/2 - epsilon), where epsilon is a small positive number 
	// used to prevent constraint from instability on poles;
	// new position of X, allowed limits are (-PI,PI);
	// So to simulate ODE Universal joint we should use parent axis as Z, child axis as Y and limit all other DOFs
	// Build the frame in world coordinate system first
	btVector3 zAxis = axis1.normalize();
	btVector3 xAxis = axis2.normalize();
	btVector3 yAxis = zAxis.cross(xAxis); // we want right coordinate system
	btTransform frameInW;
	frameInW.setIdentity();
	frameInW.getBasis().setValue(	xAxis[0], yAxis[0], zAxis[0],	
									xAxis[1], yAxis[1], zAxis[1],
									xAxis[2], yAxis[2], zAxis[2]);
	frameInW.setOrigin(anchor);
	// now get constraint frame in local coordinate systems
	m_frameInA = rbA.getCenterOfMassTransform().inverse() * frameInW;
	m_frameInB = rbB.getCenterOfMassTransform().inverse() * frameInW;
	// sei limits
	setLinearLowerLimit(btVector3(0.f, 0.f, -1.f));
	setLinearUpperLimit(btVector3(0.f, 0.f,  1.f));
	// like front wheels of a car
	setAngularLowerLimit(btVector3(1.f,  0.f, -SIMD_HALF_PI * 0.5f)); 
	setAngularUpperLimit(btVector3(-1.f, 0.f,  SIMD_HALF_PI * 0.5f));
	// enable suspension
	enableSpring(2, true);
	setStiffness(2, SIMD_PI * SIMD_PI * 4.f);
	setDamping(2, 0.01f);
	setEquilibriumPoint();
}

