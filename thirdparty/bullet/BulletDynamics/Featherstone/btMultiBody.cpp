/*
 * PURPOSE:
 *   Class representing an articulated rigid body. Stores the body's
 *   current state, allows forces and torques to be set, handles
 *   timestepping and implements Featherstone's algorithm.
 *   
 * COPYRIGHT:
 *   Copyright (C) Stephen Thompson, <stephen@solarflare.org.uk>, 2011-2013
 *   Portions written By Erwin Coumans: connection to LCP solver, various multibody constraints, replacing Eigen math library by Bullet LinearMath and a dedicated 6x6 matrix inverse (solveImatrix)
 *   Portions written By Jakub Stepien: support for multi-DOF constraints, introduction of spatial algebra and several other improvements

 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it freely,
 subject to the following restrictions:
 
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 
 */

#include "btMultiBody.h"
#include "btMultiBodyLink.h"
#include "btMultiBodyLinkCollider.h"
#include "btMultiBodyJointFeedback.h"
#include "LinearMath/btTransformUtil.h"
#include "LinearMath/btSerializer.h"
//#include "Bullet3Common/b3Logging.h"
// #define INCLUDE_GYRO_TERM


namespace
{
const btScalar INITIAL_SLEEP_EPSILON = btScalar(0.05);  // this is a squared velocity (m^2 s^-2)
const btScalar INITIAL_SLEEP_TIMEOUT = btScalar(2);     // in seconds
}  // namespace

void btMultiBody::spatialTransform(const btMatrix3x3 &rotation_matrix,  // rotates vectors in 'from' frame to vectors in 'to' frame
	const btVector3 &displacement,       // vector from origin of 'from' frame to origin of 'to' frame, in 'to' coordinates
	const btVector3 &top_in,             // top part of input vector
	const btVector3 &bottom_in,          // bottom part of input vector
	btVector3 &top_out,                  // top part of output vector
	btVector3 &bottom_out)               // bottom part of output vector
{
	top_out = rotation_matrix * top_in;
	bottom_out = -displacement.cross(top_out) + rotation_matrix * bottom_in;
}

namespace
{


#if 0
    void InverseSpatialTransform(const btMatrix3x3 &rotation_matrix,
                                 const btVector3 &displacement,
                                 const btVector3 &top_in,
                                 const btVector3 &bottom_in,
                                 btVector3 &top_out,
                                 btVector3 &bottom_out)
    {
        top_out = rotation_matrix.transpose() * top_in;
        bottom_out = rotation_matrix.transpose() * (bottom_in + displacement.cross(top_in));		
    }

    btScalar SpatialDotProduct(const btVector3 &a_top,
                            const btVector3 &a_bottom,
                            const btVector3 &b_top,
                            const btVector3 &b_bottom)
    {
        return a_bottom.dot(b_top) + a_top.dot(b_bottom);
    }

	void SpatialCrossProduct(const btVector3 &a_top,
                            const btVector3 &a_bottom,
                            const btVector3 &b_top,
                            const btVector3 &b_bottom,
							btVector3 &top_out,
							btVector3 &bottom_out)
	{
		top_out = a_top.cross(b_top);
		bottom_out = a_bottom.cross(b_top) + a_top.cross(b_bottom);
	}
#endif

}  // namespace

//
// Implementation of class btMultiBody
//

btMultiBody::btMultiBody(int n_links,
						 btScalar mass,
						 const btVector3 &inertia,
						 bool fixedBase,
						 bool canSleep,
						 bool /*deprecatedUseMultiDof*/)
	: m_baseCollider(0),
	  m_baseName(0),
	  m_basePos(0, 0, 0),
	  m_baseQuat(0, 0, 0, 1),
      m_basePos_interpolate(0, 0, 0),
      m_baseQuat_interpolate(0, 0, 0, 1),
	  m_baseMass(mass),
	  m_baseInertia(inertia),

	  m_fixedBase(fixedBase),
	  m_awake(true),
	  m_canSleep(canSleep),
	  m_canWakeup(true),
	  m_sleepTimer(0),
      m_sleepEpsilon(INITIAL_SLEEP_EPSILON),
	  m_sleepTimeout(INITIAL_SLEEP_TIMEOUT),

	  m_userObjectPointer(0),
	  m_userIndex2(-1),
	  m_userIndex(-1),
	  m_companionId(-1),
	  m_linearDamping(0.04f),
	  m_angularDamping(0.04f),
	  m_useGyroTerm(true),
	  m_maxAppliedImpulse(1000.f),
	  m_maxCoordinateVelocity(100.f),
	  m_hasSelfCollision(true),
	  __posUpdated(false),
	  m_dofCount(0),
	  m_posVarCnt(0),
	  m_useRK4(false),
	  m_useGlobalVelocities(false),
	  m_internalNeedsJointFeedback(false),
		m_kinematic_calculate_velocity(false)
{
	m_cachedInertiaTopLeft.setValue(0, 0, 0, 0, 0, 0, 0, 0, 0);
	m_cachedInertiaTopRight.setValue(0, 0, 0, 0, 0, 0, 0, 0, 0);
	m_cachedInertiaLowerLeft.setValue(0, 0, 0, 0, 0, 0, 0, 0, 0);
	m_cachedInertiaLowerRight.setValue(0, 0, 0, 0, 0, 0, 0, 0, 0);
	m_cachedInertiaValid = false;

	m_links.resize(n_links);
	m_matrixBuf.resize(n_links + 1);

	m_baseForce.setValue(0, 0, 0);
	m_baseTorque.setValue(0, 0, 0);

	clearConstraintForces();
	clearForcesAndTorques();
}

btMultiBody::~btMultiBody()
{
}

void btMultiBody::setupFixed(int i,
							 btScalar mass,
							 const btVector3 &inertia,
							 int parent,
							 const btQuaternion &rotParentToThis,
							 const btVector3 &parentComToThisPivotOffset,
							 const btVector3 &thisPivotToThisComOffset, bool /*deprecatedDisableParentCollision*/)
{
	m_links[i].m_mass = mass;
	m_links[i].m_inertiaLocal = inertia;
	m_links[i].m_parent = parent;
	m_links[i].setAxisTop(0, 0., 0., 0.);
	m_links[i].setAxisBottom(0, btVector3(0, 0, 0));
	m_links[i].m_zeroRotParentToThis = rotParentToThis;
	m_links[i].m_dVector = thisPivotToThisComOffset;
	m_links[i].m_eVector = parentComToThisPivotOffset;

	m_links[i].m_jointType = btMultibodyLink::eFixed;
	m_links[i].m_dofCount = 0;
	m_links[i].m_posVarCount = 0;

	m_links[i].m_flags |= BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION;

	m_links[i].updateCacheMultiDof();

	updateLinksDofOffsets();
}

void btMultiBody::setupPrismatic(int i,
								 btScalar mass,
								 const btVector3 &inertia,
								 int parent,
								 const btQuaternion &rotParentToThis,
								 const btVector3 &jointAxis,
								 const btVector3 &parentComToThisPivotOffset,
								 const btVector3 &thisPivotToThisComOffset,
								 bool disableParentCollision)
{
	m_dofCount += 1;
	m_posVarCnt += 1;

	m_links[i].m_mass = mass;
	m_links[i].m_inertiaLocal = inertia;
	m_links[i].m_parent = parent;
	m_links[i].m_zeroRotParentToThis = rotParentToThis;
	m_links[i].setAxisTop(0, 0., 0., 0.);
	m_links[i].setAxisBottom(0, jointAxis);
	m_links[i].m_eVector = parentComToThisPivotOffset;
	m_links[i].m_dVector = thisPivotToThisComOffset;
	m_links[i].m_cachedRotParentToThis = rotParentToThis;

	m_links[i].m_jointType = btMultibodyLink::ePrismatic;
	m_links[i].m_dofCount = 1;
	m_links[i].m_posVarCount = 1;
	m_links[i].m_jointPos[0] = 0.f;
	m_links[i].m_jointTorque[0] = 0.f;

	if (disableParentCollision)
		m_links[i].m_flags |= BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION;
	//

	m_links[i].updateCacheMultiDof();

	updateLinksDofOffsets();
}

void btMultiBody::setupRevolute(int i,
								btScalar mass,
								const btVector3 &inertia,
								int parent,
								const btQuaternion &rotParentToThis,
								const btVector3 &jointAxis,
								const btVector3 &parentComToThisPivotOffset,
								const btVector3 &thisPivotToThisComOffset,
								bool disableParentCollision)
{
	m_dofCount += 1;
	m_posVarCnt += 1;

	m_links[i].m_mass = mass;
	m_links[i].m_inertiaLocal = inertia;
	m_links[i].m_parent = parent;
	m_links[i].m_zeroRotParentToThis = rotParentToThis;
	m_links[i].setAxisTop(0, jointAxis);
	m_links[i].setAxisBottom(0, jointAxis.cross(thisPivotToThisComOffset));
	m_links[i].m_dVector = thisPivotToThisComOffset;
	m_links[i].m_eVector = parentComToThisPivotOffset;

	m_links[i].m_jointType = btMultibodyLink::eRevolute;
	m_links[i].m_dofCount = 1;
	m_links[i].m_posVarCount = 1;
	m_links[i].m_jointPos[0] = 0.f;
	m_links[i].m_jointTorque[0] = 0.f;

	if (disableParentCollision)
		m_links[i].m_flags |= BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION;
	//
	m_links[i].updateCacheMultiDof();
	//
	updateLinksDofOffsets();
}

void btMultiBody::setupSpherical(int i,
								 btScalar mass,
								 const btVector3 &inertia,
								 int parent,
								 const btQuaternion &rotParentToThis,
								 const btVector3 &parentComToThisPivotOffset,
								 const btVector3 &thisPivotToThisComOffset,
								 bool disableParentCollision)
{
	m_dofCount += 3;
	m_posVarCnt += 4;

	m_links[i].m_mass = mass;
	m_links[i].m_inertiaLocal = inertia;
	m_links[i].m_parent = parent;
	m_links[i].m_zeroRotParentToThis = rotParentToThis;
	m_links[i].m_dVector = thisPivotToThisComOffset;
	m_links[i].m_eVector = parentComToThisPivotOffset;

	m_links[i].m_jointType = btMultibodyLink::eSpherical;
	m_links[i].m_dofCount = 3;
	m_links[i].m_posVarCount = 4;
	m_links[i].setAxisTop(0, 1.f, 0.f, 0.f);
	m_links[i].setAxisTop(1, 0.f, 1.f, 0.f);
	m_links[i].setAxisTop(2, 0.f, 0.f, 1.f);
	m_links[i].setAxisBottom(0, m_links[i].getAxisTop(0).cross(thisPivotToThisComOffset));
	m_links[i].setAxisBottom(1, m_links[i].getAxisTop(1).cross(thisPivotToThisComOffset));
	m_links[i].setAxisBottom(2, m_links[i].getAxisTop(2).cross(thisPivotToThisComOffset));
	m_links[i].m_jointPos[0] = m_links[i].m_jointPos[1] = m_links[i].m_jointPos[2] = 0.f;
	m_links[i].m_jointPos[3] = 1.f;
	m_links[i].m_jointTorque[0] = m_links[i].m_jointTorque[1] = m_links[i].m_jointTorque[2] = 0.f;

	if (disableParentCollision)
		m_links[i].m_flags |= BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION;
	//
	m_links[i].updateCacheMultiDof();
	//
	updateLinksDofOffsets();
}

void btMultiBody::setupPlanar(int i,
							  btScalar mass,
							  const btVector3 &inertia,
							  int parent,
							  const btQuaternion &rotParentToThis,
							  const btVector3 &rotationAxis,
							  const btVector3 &parentComToThisComOffset,
							  bool disableParentCollision)
{
	m_dofCount += 3;
	m_posVarCnt += 3;

	m_links[i].m_mass = mass;
	m_links[i].m_inertiaLocal = inertia;
	m_links[i].m_parent = parent;
	m_links[i].m_zeroRotParentToThis = rotParentToThis;
	m_links[i].m_dVector.setZero();
	m_links[i].m_eVector = parentComToThisComOffset;

	//
	btVector3 vecNonParallelToRotAxis(1, 0, 0);
	if (rotationAxis.normalized().dot(vecNonParallelToRotAxis) > 0.999)
		vecNonParallelToRotAxis.setValue(0, 1, 0);
	//

	m_links[i].m_jointType = btMultibodyLink::ePlanar;
	m_links[i].m_dofCount = 3;
	m_links[i].m_posVarCount = 3;
	btVector3 n = rotationAxis.normalized();
	m_links[i].setAxisTop(0, n[0], n[1], n[2]);
	m_links[i].setAxisTop(1, 0, 0, 0);
	m_links[i].setAxisTop(2, 0, 0, 0);
	m_links[i].setAxisBottom(0, 0, 0, 0);
	btVector3 cr = m_links[i].getAxisTop(0).cross(vecNonParallelToRotAxis);
	m_links[i].setAxisBottom(1, cr[0], cr[1], cr[2]);
	cr = m_links[i].getAxisBottom(1).cross(m_links[i].getAxisTop(0));
	m_links[i].setAxisBottom(2, cr[0], cr[1], cr[2]);
	m_links[i].m_jointPos[0] = m_links[i].m_jointPos[1] = m_links[i].m_jointPos[2] = 0.f;
	m_links[i].m_jointTorque[0] = m_links[i].m_jointTorque[1] = m_links[i].m_jointTorque[2] = 0.f;

	if (disableParentCollision)
		m_links[i].m_flags |= BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION;
	//
	m_links[i].updateCacheMultiDof();
	//
	updateLinksDofOffsets();

	m_links[i].setAxisBottom(1, m_links[i].getAxisBottom(1).normalized());
	m_links[i].setAxisBottom(2, m_links[i].getAxisBottom(2).normalized());
}

void btMultiBody::finalizeMultiDof()
{
	m_deltaV.resize(0);
	m_deltaV.resize(6 + m_dofCount);
    m_splitV.resize(0);
    m_splitV.resize(6 + m_dofCount);
	m_realBuf.resize(6 + m_dofCount + m_dofCount * m_dofCount + 6 + m_dofCount);  //m_dofCount for joint-space vels + m_dofCount^2 for "D" matrices + delta-pos vector (6 base "vels" + joint "vels")
	m_vectorBuf.resize(2 * m_dofCount);                                           //two 3-vectors (i.e. one six-vector) for each system dof	("h" matrices)
	m_matrixBuf.resize(m_links.size() + 1);
	for (int i = 0; i < m_vectorBuf.size(); i++)
	{
		m_vectorBuf[i].setValue(0, 0, 0);
	}
	updateLinksDofOffsets();
}

int btMultiBody::getParent(int link_num) const
{
	return m_links[link_num].m_parent;
}

btScalar btMultiBody::getLinkMass(int i) const
{
	return m_links[i].m_mass;
}

const btVector3 &btMultiBody::getLinkInertia(int i) const
{
	return m_links[i].m_inertiaLocal;
}

btScalar btMultiBody::getJointPos(int i) const
{
	return m_links[i].m_jointPos[0];
}

btScalar btMultiBody::getJointVel(int i) const
{
	return m_realBuf[6 + m_links[i].m_dofOffset];
}

btScalar *btMultiBody::getJointPosMultiDof(int i)
{
	return &m_links[i].m_jointPos[0];
}

btScalar *btMultiBody::getJointVelMultiDof(int i)
{
	return &m_realBuf[6 + m_links[i].m_dofOffset];
}

const btScalar *btMultiBody::getJointPosMultiDof(int i) const
{
	return &m_links[i].m_jointPos[0];
}

const btScalar *btMultiBody::getJointVelMultiDof(int i) const
{
	return &m_realBuf[6 + m_links[i].m_dofOffset];
}

void btMultiBody::setJointPos(int i, btScalar q)
{
	m_links[i].m_jointPos[0] = q;
	m_links[i].updateCacheMultiDof();
}


void btMultiBody::setJointPosMultiDof(int i, const double *q)
{
	for (int pos = 0; pos < m_links[i].m_posVarCount; ++pos)
		m_links[i].m_jointPos[pos] = (btScalar)q[pos];

	m_links[i].updateCacheMultiDof();
}

void btMultiBody::setJointPosMultiDof(int i, const float *q)
{
	for (int pos = 0; pos < m_links[i].m_posVarCount; ++pos)
		m_links[i].m_jointPos[pos] = (btScalar)q[pos];

	m_links[i].updateCacheMultiDof();
}



void btMultiBody::setJointVel(int i, btScalar qdot)
{
	m_realBuf[6 + m_links[i].m_dofOffset] = qdot;
}

void btMultiBody::setJointVelMultiDof(int i, const double *qdot)
{
	for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		m_realBuf[6 + m_links[i].m_dofOffset + dof] = (btScalar)qdot[dof];
}

void btMultiBody::setJointVelMultiDof(int i, const float* qdot)
{
	for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		m_realBuf[6 + m_links[i].m_dofOffset + dof] = (btScalar)qdot[dof];
}

const btVector3 &btMultiBody::getRVector(int i) const
{
	return m_links[i].m_cachedRVector;
}

const btQuaternion &btMultiBody::getParentToLocalRot(int i) const
{
	return m_links[i].m_cachedRotParentToThis;
}

const btVector3 &btMultiBody::getInterpolateRVector(int i) const
{
    return m_links[i].m_cachedRVector_interpolate;
}

const btQuaternion &btMultiBody::getInterpolateParentToLocalRot(int i) const
{
    return m_links[i].m_cachedRotParentToThis_interpolate;
}

btVector3 btMultiBody::localPosToWorld(int i, const btVector3 &local_pos) const
{
	btAssert(i >= -1);
	btAssert(i < m_links.size());
	if ((i < -1) || (i >= m_links.size()))
	{
		return btVector3(SIMD_INFINITY, SIMD_INFINITY, SIMD_INFINITY);
	}

	btVector3 result = local_pos;
	while (i != -1)
	{
		// 'result' is in frame i. transform it to frame parent(i)
		result += getRVector(i);
		result = quatRotate(getParentToLocalRot(i).inverse(), result);
		i = getParent(i);
	}

	// 'result' is now in the base frame. transform it to world frame
	result = quatRotate(getWorldToBaseRot().inverse(), result);
	result += getBasePos();

	return result;
}

btVector3 btMultiBody::worldPosToLocal(int i, const btVector3 &world_pos) const
{
	btAssert(i >= -1);
	btAssert(i < m_links.size());
	if ((i < -1) || (i >= m_links.size()))
	{
		return btVector3(SIMD_INFINITY, SIMD_INFINITY, SIMD_INFINITY);
	}

	if (i == -1)
	{
		// world to base
		return quatRotate(getWorldToBaseRot(), (world_pos - getBasePos()));
	}
	else
	{
		// find position in parent frame, then transform to current frame
		return quatRotate(getParentToLocalRot(i), worldPosToLocal(getParent(i), world_pos)) - getRVector(i);
	}
}

btVector3 btMultiBody::localDirToWorld(int i, const btVector3 &local_dir) const
{
	btAssert(i >= -1);
	btAssert(i < m_links.size());
	if ((i < -1) || (i >= m_links.size()))
	{
		return btVector3(SIMD_INFINITY, SIMD_INFINITY, SIMD_INFINITY);
	}

	btVector3 result = local_dir;
	while (i != -1)
	{
		result = quatRotate(getParentToLocalRot(i).inverse(), result);
		i = getParent(i);
	}
	result = quatRotate(getWorldToBaseRot().inverse(), result);
	return result;
}

btVector3 btMultiBody::worldDirToLocal(int i, const btVector3 &world_dir) const
{
	btAssert(i >= -1);
	btAssert(i < m_links.size());
	if ((i < -1) || (i >= m_links.size()))
	{
		return btVector3(SIMD_INFINITY, SIMD_INFINITY, SIMD_INFINITY);
	}

	if (i == -1)
	{
		return quatRotate(getWorldToBaseRot(), world_dir);
	}
	else
	{
		return quatRotate(getParentToLocalRot(i), worldDirToLocal(getParent(i), world_dir));
	}
}

btMatrix3x3 btMultiBody::localFrameToWorld(int i, const btMatrix3x3 &local_frame) const
{
	btMatrix3x3 result = local_frame;
	btVector3 frameInWorld0 = localDirToWorld(i, local_frame.getColumn(0));
	btVector3 frameInWorld1 = localDirToWorld(i, local_frame.getColumn(1));
	btVector3 frameInWorld2 = localDirToWorld(i, local_frame.getColumn(2));
	result.setValue(frameInWorld0[0], frameInWorld1[0], frameInWorld2[0], frameInWorld0[1], frameInWorld1[1], frameInWorld2[1], frameInWorld0[2], frameInWorld1[2], frameInWorld2[2]);
	return result;
}

void btMultiBody::compTreeLinkVelocities(btVector3 *omega, btVector3 *vel) const
{
	int num_links = getNumLinks();
	// Calculates the velocities of each link (and the base) in its local frame
	const btQuaternion& base_rot = getWorldToBaseRot();
	omega[0] = quatRotate(base_rot, getBaseOmega());
	vel[0] = quatRotate(base_rot, getBaseVel());

	for (int i = 0; i < num_links; ++i)
	{
		const btMultibodyLink& link = getLink(i);
		const int parent = link.m_parent;

		// transform parent vel into this frame, store in omega[i+1], vel[i+1]
		spatialTransform(btMatrix3x3(link.m_cachedRotParentToThis), link.m_cachedRVector,
			omega[parent + 1], vel[parent + 1],
			omega[i + 1], vel[i + 1]);

		// now add qidot * shat_i
		const btScalar* jointVel = getJointVelMultiDof(i);
		for (int dof = 0; dof < link.m_dofCount; ++dof)
		{
			omega[i + 1] += jointVel[dof] * link.getAxisTop(dof);
			vel[i + 1] += jointVel[dof] * link.getAxisBottom(dof);
		}
	}
}


void btMultiBody::clearConstraintForces()
{
	m_baseConstraintForce.setValue(0, 0, 0);
	m_baseConstraintTorque.setValue(0, 0, 0);

	for (int i = 0; i < getNumLinks(); ++i)
	{
		m_links[i].m_appliedConstraintForce.setValue(0, 0, 0);
		m_links[i].m_appliedConstraintTorque.setValue(0, 0, 0);
	}
}
void btMultiBody::clearForcesAndTorques()
{
	m_baseForce.setValue(0, 0, 0);
	m_baseTorque.setValue(0, 0, 0);

	for (int i = 0; i < getNumLinks(); ++i)
	{
		m_links[i].m_appliedForce.setValue(0, 0, 0);
		m_links[i].m_appliedTorque.setValue(0, 0, 0);
		m_links[i].m_jointTorque[0] = m_links[i].m_jointTorque[1] = m_links[i].m_jointTorque[2] = m_links[i].m_jointTorque[3] = m_links[i].m_jointTorque[4] = m_links[i].m_jointTorque[5] = 0.f;
	}
}

void btMultiBody::clearVelocities()
{
	for (int i = 0; i < 6 + getNumDofs(); ++i)
	{
		m_realBuf[i] = 0.f;
	}
}
void btMultiBody::addLinkForce(int i, const btVector3 &f)
{
	m_links[i].m_appliedForce += f;
}

void btMultiBody::addLinkTorque(int i, const btVector3 &t)
{
	m_links[i].m_appliedTorque += t;
}

void btMultiBody::addLinkConstraintForce(int i, const btVector3 &f)
{
	m_links[i].m_appliedConstraintForce += f;
}

void btMultiBody::addLinkConstraintTorque(int i, const btVector3 &t)
{
	m_links[i].m_appliedConstraintTorque += t;
}

void btMultiBody::addJointTorque(int i, btScalar Q)
{
	m_links[i].m_jointTorque[0] += Q;
}

void btMultiBody::addJointTorqueMultiDof(int i, int dof, btScalar Q)
{
	m_links[i].m_jointTorque[dof] += Q;
}

void btMultiBody::addJointTorqueMultiDof(int i, const btScalar *Q)
{
	for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		m_links[i].m_jointTorque[dof] = Q[dof];
}

const btVector3 &btMultiBody::getLinkForce(int i) const
{
	return m_links[i].m_appliedForce;
}

const btVector3 &btMultiBody::getLinkTorque(int i) const
{
	return m_links[i].m_appliedTorque;
}

btScalar btMultiBody::getJointTorque(int i) const
{
	return m_links[i].m_jointTorque[0];
}

btScalar *btMultiBody::getJointTorqueMultiDof(int i)
{
	return &m_links[i].m_jointTorque[0];
}

bool btMultiBody::hasFixedBase() const
{
	return m_fixedBase || (getBaseCollider() && getBaseCollider()->isStaticObject());
}

bool btMultiBody::isBaseStaticOrKinematic() const
{
	return m_fixedBase || (getBaseCollider() && getBaseCollider()->isStaticOrKinematicObject());
}

bool btMultiBody::isBaseKinematic() const
{
	return getBaseCollider() && getBaseCollider()->isKinematicObject();
}

void btMultiBody::setBaseDynamicType(int dynamicType)
{
	if(getBaseCollider()) {
		int oldFlags = getBaseCollider()->getCollisionFlags();
		oldFlags &= ~(btCollisionObject::CF_STATIC_OBJECT | btCollisionObject::CF_KINEMATIC_OBJECT);
		getBaseCollider()->setCollisionFlags(oldFlags | dynamicType);
	}
}

inline btMatrix3x3 outerProduct(const btVector3 &v0, const btVector3 &v1)  //renamed it from vecMulVecTranspose (http://en.wikipedia.org/wiki/Outer_product); maybe it should be moved to btVector3 like dot and cross?
{
	btVector3 row0 = btVector3(
		v0.x() * v1.x(),
		v0.x() * v1.y(),
		v0.x() * v1.z());
	btVector3 row1 = btVector3(
		v0.y() * v1.x(),
		v0.y() * v1.y(),
		v0.y() * v1.z());
	btVector3 row2 = btVector3(
		v0.z() * v1.x(),
		v0.z() * v1.y(),
		v0.z() * v1.z());

	btMatrix3x3 m(row0[0], row0[1], row0[2],
				  row1[0], row1[1], row1[2],
				  row2[0], row2[1], row2[2]);
	return m;
}

#define vecMulVecTranspose(v0, v1Transposed) outerProduct(v0, v1Transposed)
//

void btMultiBody::computeAccelerationsArticulatedBodyAlgorithmMultiDof(btScalar dt,
    btAlignedObjectArray<btScalar> &scratch_r,
    btAlignedObjectArray<btVector3> &scratch_v,
    btAlignedObjectArray<btMatrix3x3> &scratch_m,
	bool isConstraintPass,
	bool jointFeedbackInWorldSpace,
	bool jointFeedbackInJointFrame)
{
	// Implement Featherstone's algorithm to calculate joint accelerations (q_double_dot)
	// and the base linear & angular accelerations.

	// We apply damping forces in this routine as well as any external forces specified by the
	// caller (via addBaseForce etc).

	// output should point to an array of 6 + num_links reals.
	// Format is: 3 angular accelerations (in world frame), 3 linear accelerations (in world frame),
	// num_links joint acceleration values.

	// We added support for multi degree of freedom (multi dof) joints.
	// In addition we also can compute the joint reaction forces. This is performed in a second pass,
	// so that we can include the effect of the constraint solver forces (computed in the PGS LCP solver)

	m_internalNeedsJointFeedback = false;

	int num_links = getNumLinks();

	const btScalar DAMPING_K1_LINEAR = m_linearDamping;
	const btScalar DAMPING_K2_LINEAR = m_linearDamping;

	const btScalar DAMPING_K1_ANGULAR = m_angularDamping;
	const btScalar DAMPING_K2_ANGULAR = m_angularDamping;

	const btVector3 base_vel = getBaseVel();
	const btVector3 base_omega = getBaseOmega();

	// Temporary matrices/vectors -- use scratch space from caller
	// so that we don't have to keep reallocating every frame

	scratch_r.resize(2 * m_dofCount + 7);  //multidof? ("Y"s use it and it is used to store qdd) => 2 x m_dofCount
	scratch_v.resize(8 * num_links + 6);
	scratch_m.resize(4 * num_links + 4);

	//btScalar * r_ptr = &scratch_r[0];
	btScalar *output = &scratch_r[m_dofCount];  // "output" holds the q_double_dot results
	btVector3 *v_ptr = &scratch_v[0];

	// vhat_i  (top = angular, bottom = linear part)
	btSpatialMotionVector *spatVel = (btSpatialMotionVector *)v_ptr;
	v_ptr += num_links * 2 + 2;
	//
	// zhat_i^A
	btSpatialForceVector *zeroAccSpatFrc = (btSpatialForceVector *)v_ptr;
	v_ptr += num_links * 2 + 2;
	//
	// chat_i  (note NOT defined for the base)
	btSpatialMotionVector *spatCoriolisAcc = (btSpatialMotionVector *)v_ptr;
	v_ptr += num_links * 2;
	//
	// Ihat_i^A.
	btSymmetricSpatialDyad *spatInertia = (btSymmetricSpatialDyad *)&scratch_m[num_links + 1];

	// Cached 3x3 rotation matrices from parent frame to this frame.
	btMatrix3x3 *rot_from_parent = &m_matrixBuf[0];
	btMatrix3x3 *rot_from_world = &scratch_m[0];

	// hhat_i, ahat_i
	// hhat is NOT stored for the base (but ahat is)
	btSpatialForceVector *h = (btSpatialForceVector *)(m_dofCount > 0 ? &m_vectorBuf[0] : 0);
	btSpatialMotionVector *spatAcc = (btSpatialMotionVector *)v_ptr;
	v_ptr += num_links * 2 + 2;
	//
	// Y_i, invD_i
	btScalar *invD = m_dofCount > 0 ? &m_realBuf[6 + m_dofCount] : 0;
	btScalar *Y = &scratch_r[0];
	//
	//aux variables
	btSpatialMotionVector spatJointVel;         //spatial velocity due to the joint motion (i.e. without predecessors' influence)
	btScalar D[36];                             //"D" matrix; it's dofxdof for each body so asingle 6x6 D matrix will do
	btScalar invD_times_Y[6];                   //D^{-1} * Y [dofxdof x dofx1 = dofx1] <=> D^{-1} * u; better moved to buffers since it is recalced in calcAccelerationDeltasMultiDof; num_dof of btScalar would cover all bodies
	btSpatialMotionVector result;               //holds results of the SolveImatrix op; it is a spatial motion vector (accel)
	btScalar Y_minus_hT_a[6];                   //Y - h^{T} * a; it's dofx1 for each body so a single 6x1 temp is enough
	btSpatialForceVector spatForceVecTemps[6];  //6 temporary spatial force vectors
	btSpatialTransformationMatrix fromParent;   //spatial transform from parent to child
	btSymmetricSpatialDyad dyadTemp;            //inertia matrix temp
	btSpatialTransformationMatrix fromWorld;
	fromWorld.m_trnVec.setZero();
	/////////////////

	// ptr to the joint accel part of the output
	btScalar *joint_accel = output + 6;

	// Start of the algorithm proper.

	// First 'upward' loop.
	// Combines CompTreeLinkVelocities and InitTreeLinks from Mirtich.

	rot_from_parent[0] = btMatrix3x3(m_baseQuat);  //m_baseQuat assumed to be alias!?

	//create the vector of spatial velocity of the base by transforming global-coor linear and angular velocities into base-local coordinates
	spatVel[0].setVector(rot_from_parent[0] * base_omega, rot_from_parent[0] * base_vel);

	if (isBaseStaticOrKinematic())
	{
		zeroAccSpatFrc[0].setZero();
	}
	else
	{
		const btVector3 &baseForce = isConstraintPass ? m_baseConstraintForce : m_baseForce;
		const btVector3 &baseTorque = isConstraintPass ? m_baseConstraintTorque : m_baseTorque;
		//external forces
		zeroAccSpatFrc[0].setVector(-(rot_from_parent[0] * baseTorque), -(rot_from_parent[0] * baseForce));

		//adding damping terms (only)
		const btScalar linDampMult = 1., angDampMult = 1.;
		zeroAccSpatFrc[0].addVector(angDampMult * m_baseInertia * spatVel[0].getAngular() * (DAMPING_K1_ANGULAR + DAMPING_K2_ANGULAR * spatVel[0].getAngular().safeNorm()),
									linDampMult * m_baseMass * spatVel[0].getLinear() * (DAMPING_K1_LINEAR + DAMPING_K2_LINEAR * spatVel[0].getLinear().safeNorm()));

		//
		//p += vhat x Ihat vhat - done in a simpler way
		if (m_useGyroTerm)
			zeroAccSpatFrc[0].addAngular(spatVel[0].getAngular().cross(m_baseInertia * spatVel[0].getAngular()));
		//
		zeroAccSpatFrc[0].addLinear(m_baseMass * spatVel[0].getAngular().cross(spatVel[0].getLinear()));
	}

	//init the spatial AB inertia (it has the simple form thanks to choosing local body frames origins at their COMs)
	spatInertia[0].setMatrix(btMatrix3x3(0, 0, 0, 0, 0, 0, 0, 0, 0),
							 //
							 btMatrix3x3(m_baseMass, 0, 0,
										 0, m_baseMass, 0,
										 0, 0, m_baseMass),
							 //
							 btMatrix3x3(m_baseInertia[0], 0, 0,
										 0, m_baseInertia[1], 0,
										 0, 0, m_baseInertia[2]));

	rot_from_world[0] = rot_from_parent[0];

	//
	for (int i = 0; i < num_links; ++i)
	{
		const int parent = m_links[i].m_parent;
		rot_from_parent[i + 1] = btMatrix3x3(m_links[i].m_cachedRotParentToThis);
		rot_from_world[i + 1] = rot_from_parent[i + 1] * rot_from_world[parent + 1];

		fromParent.m_rotMat = rot_from_parent[i + 1];
		fromParent.m_trnVec = m_links[i].m_cachedRVector;
		fromWorld.m_rotMat = rot_from_world[i + 1];
		fromParent.transform(spatVel[parent + 1], spatVel[i + 1]);

		// now set vhat_i to its true value by doing
		// vhat_i += qidot * shat_i
		if (!m_useGlobalVelocities)
		{
			spatJointVel.setZero();

			for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
				spatJointVel += m_links[i].m_axes[dof] * getJointVelMultiDof(i)[dof];

			// remember vhat_i is really vhat_p(i) (but in current frame) at this point	=> we need to add velocity across the inboard joint
			spatVel[i + 1] += spatJointVel;

			//
			// vhat_i is vhat_p(i) transformed to local coors + the velocity across the i-th inboard joint
			//spatVel[i+1] = fromParent * spatVel[parent+1] + spatJointVel;
		}
		else
		{
			fromWorld.transformRotationOnly(m_links[i].m_absFrameTotVelocity, spatVel[i + 1]);
			fromWorld.transformRotationOnly(m_links[i].m_absFrameLocVelocity, spatJointVel);
		}

		// we can now calculate chat_i
		spatVel[i + 1].cross(spatJointVel, spatCoriolisAcc[i]);

		// calculate zhat_i^A
		//
		if (isLinkAndAllAncestorsKinematic(i))
		{
			zeroAccSpatFrc[i].setZero();
		}
		else{
			//external forces
			btVector3 linkAppliedForce = isConstraintPass ? m_links[i].m_appliedConstraintForce : m_links[i].m_appliedForce;
			btVector3 linkAppliedTorque = isConstraintPass ? m_links[i].m_appliedConstraintTorque : m_links[i].m_appliedTorque;

			zeroAccSpatFrc[i + 1].setVector(-(rot_from_world[i + 1] * linkAppliedTorque), -(rot_from_world[i + 1] * linkAppliedForce));

#if 0	
			{

				b3Printf("stepVelocitiesMultiDof zeroAccSpatFrc[%d] linear:%f,%f,%f, angular:%f,%f,%f",
				i+1,
				zeroAccSpatFrc[i+1].m_topVec[0],
				zeroAccSpatFrc[i+1].m_topVec[1],
				zeroAccSpatFrc[i+1].m_topVec[2],

				zeroAccSpatFrc[i+1].m_bottomVec[0],
				zeroAccSpatFrc[i+1].m_bottomVec[1],
				zeroAccSpatFrc[i+1].m_bottomVec[2]);
			}
#endif
			//
			//adding damping terms (only)
			btScalar linDampMult = 1., angDampMult = 1.;
			zeroAccSpatFrc[i + 1].addVector(angDampMult * m_links[i].m_inertiaLocal * spatVel[i + 1].getAngular() * (DAMPING_K1_ANGULAR + DAMPING_K2_ANGULAR * spatVel[i + 1].getAngular().safeNorm()),
											linDampMult * m_links[i].m_mass * spatVel[i + 1].getLinear() * (DAMPING_K1_LINEAR + DAMPING_K2_LINEAR * spatVel[i + 1].getLinear().safeNorm()));
			//p += vhat x Ihat vhat - done in a simpler way
			if (m_useGyroTerm)
				zeroAccSpatFrc[i + 1].addAngular(spatVel[i + 1].getAngular().cross(m_links[i].m_inertiaLocal * spatVel[i + 1].getAngular()));
			//
			zeroAccSpatFrc[i + 1].addLinear(m_links[i].m_mass * spatVel[i + 1].getAngular().cross(spatVel[i + 1].getLinear()));
			//
			//btVector3 temp = m_links[i].m_mass * spatVel[i+1].getAngular().cross(spatVel[i+1].getLinear());
			////clamp parent's omega
			//btScalar parOmegaMod = temp.length();
			//btScalar parOmegaModMax = 1000;
			//if(parOmegaMod > parOmegaModMax)
			//	temp *= parOmegaModMax / parOmegaMod;
			//zeroAccSpatFrc[i+1].addLinear(temp);
			//printf("|zeroAccSpatFrc[%d]| = %.4f\n", i+1, temp.length());
			//temp = spatCoriolisAcc[i].getLinear();
			//printf("|spatCoriolisAcc[%d]| = %.4f\n", i+1, temp.length());
		}

		// calculate Ihat_i^A
		//init the spatial AB inertia (it has the simple form thanks to choosing local body frames origins at their COMs)
		spatInertia[i + 1].setMatrix(btMatrix3x3(0, 0, 0, 0, 0, 0, 0, 0, 0),
									 //
									 btMatrix3x3(m_links[i].m_mass, 0, 0,
												 0, m_links[i].m_mass, 0,
												 0, 0, m_links[i].m_mass),
									 //
									 btMatrix3x3(m_links[i].m_inertiaLocal[0], 0, 0,
												 0, m_links[i].m_inertiaLocal[1], 0,
												 0, 0, m_links[i].m_inertiaLocal[2]));

		//printf("w[%d] = [%.4f %.4f %.4f]\n", i, vel_top_angular[i+1].x(), vel_top_angular[i+1].y(), vel_top_angular[i+1].z());
		//printf("v[%d] = [%.4f %.4f %.4f]\n", i, vel_bottom_linear[i+1].x(), vel_bottom_linear[i+1].y(), vel_bottom_linear[i+1].z());
		//printf("c[%d] = [%.4f %.4f %.4f]\n", i, coriolis_bottom_linear[i].x(), coriolis_bottom_linear[i].y(), coriolis_bottom_linear[i].z());
	}

	// 'Downward' loop.
	// (part of TreeForwardDynamics in Mirtich.)
	for (int i = num_links - 1; i >= 0; --i)
	{
		if(isLinkAndAllAncestorsKinematic(i))
			continue;
		const int parent = m_links[i].m_parent;
		fromParent.m_rotMat = rot_from_parent[i + 1];
		fromParent.m_trnVec = m_links[i].m_cachedRVector;

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			btSpatialForceVector &hDof = h[m_links[i].m_dofOffset + dof];
			//
			hDof = spatInertia[i + 1] * m_links[i].m_axes[dof];
			//
			Y[m_links[i].m_dofOffset + dof] = m_links[i].m_jointTorque[dof] - m_links[i].m_axes[dof].dot(zeroAccSpatFrc[i + 1]) - spatCoriolisAcc[i].dot(hDof);
		}
		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			btScalar *D_row = &D[dof * m_links[i].m_dofCount];
			for (int dof2 = 0; dof2 < m_links[i].m_dofCount; ++dof2)
			{
				const btSpatialForceVector &hDof2 = h[m_links[i].m_dofOffset + dof2];
				D_row[dof2] = m_links[i].m_axes[dof].dot(hDof2);
			}
		}

		btScalar *invDi = &invD[m_links[i].m_dofOffset * m_links[i].m_dofOffset];
		switch (m_links[i].m_jointType)
		{
			case btMultibodyLink::ePrismatic:
			case btMultibodyLink::eRevolute:
			{
				if (D[0] >= SIMD_EPSILON)
				{
					invDi[0] = 1.0f / D[0];
				}
				else
				{
					invDi[0] = 0;
				}
				break;
			}
			case btMultibodyLink::eSpherical:
			case btMultibodyLink::ePlanar:
			{
				const btMatrix3x3 D3x3(D[0], D[1], D[2], D[3], D[4], D[5], D[6], D[7], D[8]);
				const btMatrix3x3 invD3x3(D3x3.inverse());

				//unroll the loop?
				for (int row = 0; row < 3; ++row)
				{
					for (int col = 0; col < 3; ++col)
					{
						invDi[row * 3 + col] = invD3x3[row][col];
					}
				}

				break;
			}
			default:
			{
			}
		}

		//determine h*D^{-1}
		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			spatForceVecTemps[dof].setZero();

			for (int dof2 = 0; dof2 < m_links[i].m_dofCount; ++dof2)
			{
				const btSpatialForceVector &hDof2 = h[m_links[i].m_dofOffset + dof2];
				//
				spatForceVecTemps[dof] += hDof2 * invDi[dof2 * m_links[i].m_dofCount + dof];
			}
		}

		dyadTemp = spatInertia[i + 1];

		//determine (h*D^{-1}) * h^{T}
		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			const btSpatialForceVector &hDof = h[m_links[i].m_dofOffset + dof];
			//
			dyadTemp -= symmetricSpatialOuterProduct(hDof, spatForceVecTemps[dof]);
		}

		fromParent.transformInverse(dyadTemp, spatInertia[parent + 1], btSpatialTransformationMatrix::Add);

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			invD_times_Y[dof] = 0.f;

			for (int dof2 = 0; dof2 < m_links[i].m_dofCount; ++dof2)
			{
				invD_times_Y[dof] += invDi[dof * m_links[i].m_dofCount + dof2] * Y[m_links[i].m_dofOffset + dof2];
			}
		}

		spatForceVecTemps[0] = zeroAccSpatFrc[i + 1] + spatInertia[i + 1] * spatCoriolisAcc[i];

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			const btSpatialForceVector &hDof = h[m_links[i].m_dofOffset + dof];
			//
			spatForceVecTemps[0] += hDof * invD_times_Y[dof];
		}

		fromParent.transformInverse(spatForceVecTemps[0], spatForceVecTemps[1]);

		zeroAccSpatFrc[parent + 1] += spatForceVecTemps[1];
	}

	// Second 'upward' loop
	// (part of TreeForwardDynamics in Mirtich)

	if (isBaseStaticOrKinematic())
	{
		spatAcc[0].setZero();
	}
	else
	{
		if (num_links > 0)
		{
			m_cachedInertiaValid = true;
			m_cachedInertiaTopLeft = spatInertia[0].m_topLeftMat;
			m_cachedInertiaTopRight = spatInertia[0].m_topRightMat;
			m_cachedInertiaLowerLeft = spatInertia[0].m_bottomLeftMat;
			m_cachedInertiaLowerRight = spatInertia[0].m_topLeftMat.transpose();
		}

		solveImatrix(zeroAccSpatFrc[0], result);
		spatAcc[0] = -result;
	}

	// now do the loop over the m_links
	for (int i = 0; i < num_links; ++i)
	{
		//	qdd = D^{-1} * (Y - h^{T}*apar) = (S^{T}*I*S)^{-1} * (tau - S^{T}*I*cor - S^{T}*zeroAccFrc - S^{T}*I*apar)
		//	a = apar + cor + Sqdd
		//or
		//	qdd = D^{-1} * (Y - h^{T}*(apar+cor))
		//	a = apar + Sqdd

		const int parent = m_links[i].m_parent;
		fromParent.m_rotMat = rot_from_parent[i + 1];
		fromParent.m_trnVec = m_links[i].m_cachedRVector;

		fromParent.transform(spatAcc[parent + 1], spatAcc[i + 1]);

		if(!isLinkAndAllAncestorsKinematic(i))
		{
			for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
			{
				const btSpatialForceVector &hDof = h[m_links[i].m_dofOffset + dof];
				//
				Y_minus_hT_a[dof] = Y[m_links[i].m_dofOffset + dof] - spatAcc[i + 1].dot(hDof);
			}
			btScalar *invDi = &invD[m_links[i].m_dofOffset * m_links[i].m_dofOffset];
			//D^{-1} * (Y - h^{T}*apar)
			mulMatrix(invDi, Y_minus_hT_a, m_links[i].m_dofCount, m_links[i].m_dofCount, m_links[i].m_dofCount, 1, &joint_accel[m_links[i].m_dofOffset]);

			spatAcc[i + 1] += spatCoriolisAcc[i];

			for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
				spatAcc[i + 1] += m_links[i].m_axes[dof] * joint_accel[m_links[i].m_dofOffset + dof];
		}

		if (m_links[i].m_jointFeedback)
		{
			m_internalNeedsJointFeedback = true;

			btVector3 angularBotVec = (spatInertia[i + 1] * spatAcc[i + 1] + zeroAccSpatFrc[i + 1]).m_bottomVec;
			btVector3 linearTopVec = (spatInertia[i + 1] * spatAcc[i + 1] + zeroAccSpatFrc[i + 1]).m_topVec;

			if (jointFeedbackInJointFrame)
			{
				//shift the reaction forces to the joint frame
				//linear (force) component is the same
				//shift the angular (torque, moment) component using the relative position,  m_links[i].m_dVector
				angularBotVec = angularBotVec - linearTopVec.cross(m_links[i].m_dVector);
			}

			if (jointFeedbackInWorldSpace)
			{
				if (isConstraintPass)
				{
					m_links[i].m_jointFeedback->m_reactionForces.m_bottomVec += m_links[i].m_cachedWorldTransform.getBasis() * angularBotVec;
					m_links[i].m_jointFeedback->m_reactionForces.m_topVec += m_links[i].m_cachedWorldTransform.getBasis() * linearTopVec;
				}
				else
				{
					m_links[i].m_jointFeedback->m_reactionForces.m_bottomVec = m_links[i].m_cachedWorldTransform.getBasis() * angularBotVec;
					m_links[i].m_jointFeedback->m_reactionForces.m_topVec = m_links[i].m_cachedWorldTransform.getBasis() * linearTopVec;
				}
			}
			else
			{
				if (isConstraintPass)
				{
					m_links[i].m_jointFeedback->m_reactionForces.m_bottomVec += angularBotVec;
					m_links[i].m_jointFeedback->m_reactionForces.m_topVec += linearTopVec;
				}
				else
				{
					m_links[i].m_jointFeedback->m_reactionForces.m_bottomVec = angularBotVec;
					m_links[i].m_jointFeedback->m_reactionForces.m_topVec = linearTopVec;
				}
			}
		}
	}

	// transform base accelerations back to the world frame.
	const btVector3 omegadot_out = rot_from_parent[0].transpose() * spatAcc[0].getAngular();
	output[0] = omegadot_out[0];
	output[1] = omegadot_out[1];
	output[2] = omegadot_out[2];

	const btVector3 vdot_out = rot_from_parent[0].transpose() * (spatAcc[0].getLinear() + spatVel[0].getAngular().cross(spatVel[0].getLinear()));
	output[3] = vdot_out[0];
	output[4] = vdot_out[1];
	output[5] = vdot_out[2];

	/////////////////
	//printf("q = [");
	//printf("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f ", m_baseQuat.x(), m_baseQuat.y(), m_baseQuat.z(), m_baseQuat.w(), m_basePos.x(), m_basePos.y(), m_basePos.z());
	//for(int link = 0; link < getNumLinks(); ++link)
	//	for(int dof = 0; dof < m_links[link].m_dofCount; ++dof)
	//		printf("%.6f ", m_links[link].m_jointPos[dof]);
	//printf("]\n");
	////
	//printf("qd = [");
	//for(int dof = 0; dof < getNumDofs() + 6; ++dof)
	//	printf("%.6f ", m_realBuf[dof]);
	//printf("]\n");
	//printf("qdd = [");
	//for(int dof = 0; dof < getNumDofs() + 6; ++dof)
	//	printf("%.6f ", output[dof]);
	//printf("]\n");
	/////////////////

	// Final step: add the accelerations (times dt) to the velocities.

	if (!isConstraintPass)
	{
		if (dt > 0.)
			applyDeltaVeeMultiDof(output, dt);
	}
	/////
	//btScalar angularThres = 1;
	//btScalar maxAngVel = 0.;
	//bool scaleDown = 1.;
	//for(int link = 0; link < m_links.size(); ++link)
	//{
	//	if(spatVel[link+1].getAngular().length() > maxAngVel)
	//	{
	//		maxAngVel = spatVel[link+1].getAngular().length();
	//		scaleDown = angularThres / spatVel[link+1].getAngular().length();
	//		break;
	//	}
	//}

	//if(scaleDown != 1.)
	//{
	//	for(int link = 0; link < m_links.size(); ++link)
	//	{
	//		if(m_links[link].m_jointType == btMultibodyLink::eRevolute || m_links[link].m_jointType == btMultibodyLink::eSpherical)
	//		{
	//			for(int dof = 0; dof < m_links[link].m_dofCount; ++dof)
	//				getJointVelMultiDof(link)[dof] *= scaleDown;
	//		}
	//	}
	//}
	/////

	/////////////////////
	if (m_useGlobalVelocities)
	{
		for (int i = 0; i < num_links; ++i)
		{
			const int parent = m_links[i].m_parent;
			//rot_from_parent[i+1] = btMatrix3x3(m_links[i].m_cachedRotParentToThis);    /// <- done
			//rot_from_world[i+1] = rot_from_parent[i+1] * rot_from_world[parent+1];		/// <- done

			fromParent.m_rotMat = rot_from_parent[i + 1];
			fromParent.m_trnVec = m_links[i].m_cachedRVector;
			fromWorld.m_rotMat = rot_from_world[i + 1];

			// vhat_i = i_xhat_p(i) * vhat_p(i)
			fromParent.transform(spatVel[parent + 1], spatVel[i + 1]);
			//nice alternative below (using operator *) but it generates temps
			/////////////////////////////////////////////////////////////

			// now set vhat_i to its true value by doing
			// vhat_i += qidot * shat_i
			spatJointVel.setZero();

			for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
				spatJointVel += m_links[i].m_axes[dof] * getJointVelMultiDof(i)[dof];

			// remember vhat_i is really vhat_p(i) (but in current frame) at this point	=> we need to add velocity across the inboard joint
			spatVel[i + 1] += spatJointVel;

			fromWorld.transformInverseRotationOnly(spatVel[i + 1], m_links[i].m_absFrameTotVelocity);
			fromWorld.transformInverseRotationOnly(spatJointVel, m_links[i].m_absFrameLocVelocity);
		}
	}
}

void btMultiBody::solveImatrix(const btVector3 &rhs_top, const btVector3 &rhs_bot, btScalar result[6]) const
{
	int num_links = getNumLinks();
	///solve I * x = rhs, so the result = invI * rhs
	if (num_links == 0)
	{
		// in the case of 0 m_links (i.e. a plain rigid body, not a multibody) rhs * invI is easier

		if ((m_baseInertia[0] >= SIMD_EPSILON) && (m_baseInertia[1] >= SIMD_EPSILON) && (m_baseInertia[2] >= SIMD_EPSILON))
		{
			result[0] = rhs_bot[0] / m_baseInertia[0];
			result[1] = rhs_bot[1] / m_baseInertia[1];
			result[2] = rhs_bot[2] / m_baseInertia[2];
		}
		else
		{
			result[0] = 0;
			result[1] = 0;
			result[2] = 0;
		}
		if (m_baseMass >= SIMD_EPSILON)
		{
			result[3] = rhs_top[0] / m_baseMass;
			result[4] = rhs_top[1] / m_baseMass;
			result[5] = rhs_top[2] / m_baseMass;
		}
		else
		{
			result[3] = 0;
			result[4] = 0;
			result[5] = 0;
		}
	}
	else
	{
		if (!m_cachedInertiaValid)
		{
			for (int i = 0; i < 6; i++)
			{
				result[i] = 0.f;
			}
			return;
		}
		/// Special routine for calculating the inverse of a spatial inertia matrix
		///the 6x6 matrix is stored as 4 blocks of 3x3 matrices
		btMatrix3x3 Binv = m_cachedInertiaTopRight.inverse() * -1.f;
		btMatrix3x3 tmp = m_cachedInertiaLowerRight * Binv;
		btMatrix3x3 invIupper_right = (tmp * m_cachedInertiaTopLeft + m_cachedInertiaLowerLeft).inverse();
		tmp = invIupper_right * m_cachedInertiaLowerRight;
		btMatrix3x3 invI_upper_left = (tmp * Binv);
		btMatrix3x3 invI_lower_right = (invI_upper_left).transpose();
		tmp = m_cachedInertiaTopLeft * invI_upper_left;
		tmp[0][0] -= 1.0;
		tmp[1][1] -= 1.0;
		tmp[2][2] -= 1.0;
		btMatrix3x3 invI_lower_left = (Binv * tmp);

		//multiply result = invI * rhs
		{
			btVector3 vtop = invI_upper_left * rhs_top;
			btVector3 tmp;
			tmp = invIupper_right * rhs_bot;
			vtop += tmp;
			btVector3 vbot = invI_lower_left * rhs_top;
			tmp = invI_lower_right * rhs_bot;
			vbot += tmp;
			result[0] = vtop[0];
			result[1] = vtop[1];
			result[2] = vtop[2];
			result[3] = vbot[0];
			result[4] = vbot[1];
			result[5] = vbot[2];
		}
	}
}
void btMultiBody::solveImatrix(const btSpatialForceVector &rhs, btSpatialMotionVector &result) const
{
	int num_links = getNumLinks();
	///solve I * x = rhs, so the result = invI * rhs
	if (num_links == 0)
	{
		// in the case of 0 m_links (i.e. a plain rigid body, not a multibody) rhs * invI is easier
		if ((m_baseInertia[0] >= SIMD_EPSILON) && (m_baseInertia[1] >= SIMD_EPSILON) && (m_baseInertia[2] >= SIMD_EPSILON))
		{
			result.setAngular(rhs.getAngular() / m_baseInertia);
		}
		else
		{
			result.setAngular(btVector3(0, 0, 0));
		}
		if (m_baseMass >= SIMD_EPSILON)
		{
			result.setLinear(rhs.getLinear() / m_baseMass);
		}
		else
		{
			result.setLinear(btVector3(0, 0, 0));
		}
	}
	else
	{
		/// Special routine for calculating the inverse of a spatial inertia matrix
		///the 6x6 matrix is stored as 4 blocks of 3x3 matrices
		if (!m_cachedInertiaValid)
		{
			result.setLinear(btVector3(0, 0, 0));
			result.setAngular(btVector3(0, 0, 0));
			result.setVector(btVector3(0, 0, 0), btVector3(0, 0, 0));
			return;
		}
		btMatrix3x3 Binv = m_cachedInertiaTopRight.inverse() * -1.f;
		btMatrix3x3 tmp = m_cachedInertiaLowerRight * Binv;
		btMatrix3x3 invIupper_right = (tmp * m_cachedInertiaTopLeft + m_cachedInertiaLowerLeft).inverse();
		tmp = invIupper_right * m_cachedInertiaLowerRight;
		btMatrix3x3 invI_upper_left = (tmp * Binv);
		btMatrix3x3 invI_lower_right = (invI_upper_left).transpose();
		tmp = m_cachedInertiaTopLeft * invI_upper_left;
		tmp[0][0] -= 1.0;
		tmp[1][1] -= 1.0;
		tmp[2][2] -= 1.0;
		btMatrix3x3 invI_lower_left = (Binv * tmp);

		//multiply result = invI * rhs
		{
			btVector3 vtop = invI_upper_left * rhs.getLinear();
			btVector3 tmp;
			tmp = invIupper_right * rhs.getAngular();
			vtop += tmp;
			btVector3 vbot = invI_lower_left * rhs.getLinear();
			tmp = invI_lower_right * rhs.getAngular();
			vbot += tmp;
			result.setVector(vtop, vbot);
		}
	}
}

void btMultiBody::mulMatrix(const btScalar *pA, const btScalar *pB, int rowsA, int colsA, int rowsB, int colsB, btScalar *pC) const
{
	for (int row = 0; row < rowsA; row++)
	{
		for (int col = 0; col < colsB; col++)
		{
			pC[row * colsB + col] = 0.f;
			for (int inner = 0; inner < rowsB; inner++)
			{
				pC[row * colsB + col] += pA[row * colsA + inner] * pB[col + inner * colsB];
			}
		}
	}
}

void btMultiBody::calcAccelerationDeltasMultiDof(const btScalar *force, btScalar *output,
												 btAlignedObjectArray<btScalar> &scratch_r, btAlignedObjectArray<btVector3> &scratch_v) const
{
	// Temporary matrices/vectors -- use scratch space from caller
	// so that we don't have to keep reallocating every frame

	int num_links = getNumLinks();
	scratch_r.resize(m_dofCount);
	scratch_v.resize(4 * num_links + 4);

	btScalar *r_ptr = m_dofCount ? &scratch_r[0] : 0;
	btVector3 *v_ptr = &scratch_v[0];

	// zhat_i^A (scratch space)
	btSpatialForceVector *zeroAccSpatFrc = (btSpatialForceVector *)v_ptr;
	v_ptr += num_links * 2 + 2;

	// rot_from_parent (cached from calcAccelerations)
	const btMatrix3x3 *rot_from_parent = &m_matrixBuf[0];

	// hhat (cached), accel (scratch)
	// hhat is NOT stored for the base (but ahat is)
	const btSpatialForceVector *h = (btSpatialForceVector *)(m_dofCount > 0 ? &m_vectorBuf[0] : 0);
	btSpatialMotionVector *spatAcc = (btSpatialMotionVector *)v_ptr;
	v_ptr += num_links * 2 + 2;

	// Y_i (scratch), invD_i (cached)
	const btScalar *invD = m_dofCount > 0 ? &m_realBuf[6 + m_dofCount] : 0;
	btScalar *Y = r_ptr;
	////////////////
	//aux variables
	btScalar invD_times_Y[6];                   //D^{-1} * Y [dofxdof x dofx1 = dofx1] <=> D^{-1} * u; better moved to buffers since it is recalced in calcAccelerationDeltasMultiDof; num_dof of btScalar would cover all bodies
	btSpatialMotionVector result;               //holds results of the SolveImatrix op; it is a spatial motion vector (accel)
	btScalar Y_minus_hT_a[6];                   //Y - h^{T} * a; it's dofx1 for each body so a single 6x1 temp is enough
	btSpatialForceVector spatForceVecTemps[6];  //6 temporary spatial force vectors
	btSpatialTransformationMatrix fromParent;
	/////////////////

	// First 'upward' loop.
	// Combines CompTreeLinkVelocities and InitTreeLinks from Mirtich.

	// Fill in zero_acc
	// -- set to force/torque on the base, zero otherwise
	if (isBaseStaticOrKinematic())
	{
		zeroAccSpatFrc[0].setZero();
	}
	else
	{
		//test forces
		fromParent.m_rotMat = rot_from_parent[0];
		fromParent.transformRotationOnly(btSpatialForceVector(-force[0], -force[1], -force[2], -force[3], -force[4], -force[5]), zeroAccSpatFrc[0]);
	}
	for (int i = 0; i < num_links; ++i)
	{
		zeroAccSpatFrc[i + 1].setZero();
	}

	// 'Downward' loop.
	// (part of TreeForwardDynamics in Mirtich.)
	for (int i = num_links - 1; i >= 0; --i)
	{
		if(isLinkAndAllAncestorsKinematic(i))
			continue;
		const int parent = m_links[i].m_parent;
		fromParent.m_rotMat = rot_from_parent[i + 1];
		fromParent.m_trnVec = m_links[i].m_cachedRVector;

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			Y[m_links[i].m_dofOffset + dof] = force[6 + m_links[i].m_dofOffset + dof] - m_links[i].m_axes[dof].dot(zeroAccSpatFrc[i + 1]);
		}

		btVector3 in_top, in_bottom, out_top, out_bottom;
		const btScalar *invDi = &invD[m_links[i].m_dofOffset * m_links[i].m_dofOffset];

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			invD_times_Y[dof] = 0.f;

			for (int dof2 = 0; dof2 < m_links[i].m_dofCount; ++dof2)
			{
				invD_times_Y[dof] += invDi[dof * m_links[i].m_dofCount + dof2] * Y[m_links[i].m_dofOffset + dof2];
			}
		}

		// Zp += pXi * (Zi + hi*Yi/Di)
		spatForceVecTemps[0] = zeroAccSpatFrc[i + 1];

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			const btSpatialForceVector &hDof = h[m_links[i].m_dofOffset + dof];
			//
			spatForceVecTemps[0] += hDof * invD_times_Y[dof];
		}

		fromParent.transformInverse(spatForceVecTemps[0], spatForceVecTemps[1]);

		zeroAccSpatFrc[parent + 1] += spatForceVecTemps[1];
	}

	// ptr to the joint accel part of the output
	btScalar *joint_accel = output + 6;

	// Second 'upward' loop
	// (part of TreeForwardDynamics in Mirtich)

	if (isBaseStaticOrKinematic())
	{
		spatAcc[0].setZero();
	}
	else
	{
		solveImatrix(zeroAccSpatFrc[0], result);
		spatAcc[0] = -result;
	}

	// now do the loop over the m_links
	for (int i = 0; i < num_links; ++i)
	{
		if(isLinkAndAllAncestorsKinematic(i))
			continue;
		const int parent = m_links[i].m_parent;
		fromParent.m_rotMat = rot_from_parent[i + 1];
		fromParent.m_trnVec = m_links[i].m_cachedRVector;

		fromParent.transform(spatAcc[parent + 1], spatAcc[i + 1]);

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
		{
			const btSpatialForceVector &hDof = h[m_links[i].m_dofOffset + dof];
			//
			Y_minus_hT_a[dof] = Y[m_links[i].m_dofOffset + dof] - spatAcc[i + 1].dot(hDof);
		}

		const btScalar *invDi = &invD[m_links[i].m_dofOffset * m_links[i].m_dofOffset];
		mulMatrix(const_cast<btScalar *>(invDi), Y_minus_hT_a, m_links[i].m_dofCount, m_links[i].m_dofCount, m_links[i].m_dofCount, 1, &joint_accel[m_links[i].m_dofOffset]);

		for (int dof = 0; dof < m_links[i].m_dofCount; ++dof)
			spatAcc[i + 1] += m_links[i].m_axes[dof] * joint_accel[m_links[i].m_dofOffset + dof];
	}

	// transform base accelerations back to the world frame.
	btVector3 omegadot_out;
	omegadot_out = rot_from_parent[0].transpose() * spatAcc[0].getAngular();
	output[0] = omegadot_out[0];
	output[1] = omegadot_out[1];
	output[2] = omegadot_out[2];

	btVector3 vdot_out;
	vdot_out = rot_from_parent[0].transpose() * spatAcc[0].getLinear();
	output[3] = vdot_out[0];
	output[4] = vdot_out[1];
	output[5] = vdot_out[2];

	/////////////////
	//printf("delta = [");
	//for(int dof = 0; dof < getNumDofs() + 6; ++dof)
	//	printf("%.2f ", output[dof]);
	//printf("]\n");
	/////////////////
}
void btMultiBody::predictPositionsMultiDof(btScalar dt)
{
    int num_links = getNumLinks();
		if(!isBaseKinematic())
		{
      // step position by adding dt * velocity
      //btVector3 v = getBaseVel();
      //m_basePos += dt * v;
      //
      btScalar *pBasePos;
      btScalar *pBaseVel = &m_realBuf[3];  //note: the !pqd case assumes m_realBuf holds with base velocity at 3,4,5 (should be wrapped for safety)
    
    	// reset to current position
    	for (int i = 0; i < 3; ++i)
    	{
    	    m_basePos_interpolate[i] = m_basePos[i];
    	}
    	pBasePos = m_basePos_interpolate;
    	
    	pBasePos[0] += dt * pBaseVel[0];
    	pBasePos[1] += dt * pBaseVel[1];
    	pBasePos[2] += dt * pBaseVel[2];
		}
    
    ///////////////////////////////
    //local functor for quaternion integration (to avoid error prone redundancy)
    struct
    {
        //"exponential map" based on btTransformUtil::integrateTransform(..)
        void operator()(const btVector3 &omega, btQuaternion &quat, bool baseBody, btScalar dt)
        {
            //baseBody    =>    quat is alias and omega is global coor
            //!baseBody    =>    quat is alibi and omega is local coor
            
            btVector3 axis;
            btVector3 angvel;
            
            if (!baseBody)
                angvel = quatRotate(quat, omega);  //if quat is not m_baseQuat, it is alibi => ok
            else
                angvel = omega;
            
            btScalar fAngle = angvel.length();
            //limit the angular motion
            if (fAngle * dt > ANGULAR_MOTION_THRESHOLD)
            {
                fAngle = btScalar(0.5) * SIMD_HALF_PI / dt;
            }
            
            if (fAngle < btScalar(0.001))
            {
                // use Taylor's expansions of sync function
                axis = angvel * (btScalar(0.5) * dt - (dt * dt * dt) * (btScalar(0.020833333333)) * fAngle * fAngle);
            }
            else
            {
                // sync(fAngle) = sin(c*fAngle)/t
                axis = angvel * (btSin(btScalar(0.5) * fAngle * dt) / fAngle);
            }
            
            if (!baseBody)
                quat = btQuaternion(axis.x(), axis.y(), axis.z(), btCos(fAngle * dt * btScalar(0.5))) * quat;
            else
                quat = quat * btQuaternion(-axis.x(), -axis.y(), -axis.z(), btCos(fAngle * dt * btScalar(0.5)));
            //equivalent to: quat = (btQuaternion(axis.x(),axis.y(),axis.z(),btCos( fAngle*dt*btScalar(0.5) )) * quat.inverse()).inverse();
            
            quat.normalize();
        }
    } pQuatUpdateFun;
    ///////////////////////////////
    
    //pQuatUpdateFun(getBaseOmega(), m_baseQuat, true, dt);
    //
		if(!isBaseKinematic())
		{
        btScalar *pBaseQuat;

        // reset to current orientation
        for (int i = 0; i < 4; ++i)
        {
            m_baseQuat_interpolate[i] = m_baseQuat[i];
        }
        pBaseQuat = m_baseQuat_interpolate;

        btScalar *pBaseOmega = &m_realBuf[0];  //note: the !pqd case assumes m_realBuf starts with base omega (should be wrapped for safety)
        //
        btQuaternion baseQuat;
        baseQuat.setValue(pBaseQuat[0], pBaseQuat[1], pBaseQuat[2], pBaseQuat[3]);
        btVector3 baseOmega;
        baseOmega.setValue(pBaseOmega[0], pBaseOmega[1], pBaseOmega[2]);
        pQuatUpdateFun(baseOmega, baseQuat, true, dt);
        pBaseQuat[0] = baseQuat.x();
        pBaseQuat[1] = baseQuat.y();
        pBaseQuat[2] = baseQuat.z();
        pBaseQuat[3] = baseQuat.w();
		}

    // Finally we can update m_jointPos for each of the m_links
    for (int i = 0; i < num_links; ++i)
    {
        btScalar *pJointPos;
        pJointPos = &m_links[i].m_jointPos_interpolate[0];
        
        if (m_links[i].m_collider && m_links[i].m_collider->isStaticOrKinematic()) 
		{
            switch (m_links[i].m_jointType) 
						{
                case btMultibodyLink::ePrismatic:
                case btMultibodyLink::eRevolute:
                {
                    pJointPos[0] = m_links[i].m_jointPos[0];
                    break;
                }
                case btMultibodyLink::eSpherical:
                {
                    for (int j = 0; j < 4; ++j)
                    {
                        pJointPos[j] = m_links[i].m_jointPos[j];
                    }
                    break;
                }
                case btMultibodyLink::ePlanar:
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        pJointPos[j] = m_links[i].m_jointPos[j];
                    }
                    break;
                }
                default:
                   break;
            }
        }
        else
        {
            btScalar *pJointVel = getJointVelMultiDof(i); 

            switch (m_links[i].m_jointType)
            {
                case btMultibodyLink::ePrismatic:
                case btMultibodyLink::eRevolute:
                {
                    //reset to current pos
                    pJointPos[0] = m_links[i].m_jointPos[0];
                    btScalar jointVel = pJointVel[0];
                    pJointPos[0] += dt * jointVel;
                    break;
                }
                case btMultibodyLink::eSpherical:
                {
                    //reset to current pos

                    for (int j = 0; j < 4; ++j)
                    {
                        pJointPos[j] = m_links[i].m_jointPos[j];
                    }
                    
                    btVector3 jointVel;
                    jointVel.setValue(pJointVel[0], pJointVel[1], pJointVel[2]);
                    btQuaternion jointOri;
                    jointOri.setValue(pJointPos[0], pJointPos[1], pJointPos[2], pJointPos[3]);
                    pQuatUpdateFun(jointVel, jointOri, false, dt);
                    pJointPos[0] = jointOri.x();
                    pJointPos[1] = jointOri.y();
                    pJointPos[2] = jointOri.z();
                    pJointPos[3] = jointOri.w();
                    break;
                }
                case btMultibodyLink::ePlanar:
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        pJointPos[j] = m_links[i].m_jointPos[j];
                    }
                    pJointPos[0] += dt * getJointVelMultiDof(i)[0];
                    
                    btVector3 q0_coors_qd1qd2 = getJointVelMultiDof(i)[1] * m_links[i].getAxisBottom(1) + getJointVelMultiDof(i)[2] * m_links[i].getAxisBottom(2);
                    btVector3 no_q0_coors_qd1qd2 = quatRotate(btQuaternion(m_links[i].getAxisTop(0), pJointPos[0]), q0_coors_qd1qd2);
                    pJointPos[1] += m_links[i].getAxisBottom(1).dot(no_q0_coors_qd1qd2) * dt;
                    pJointPos[2] += m_links[i].getAxisBottom(2).dot(no_q0_coors_qd1qd2) * dt;
                    break;
                }
                default:
                {
                }
            }
        }
        
        m_links[i].updateInterpolationCacheMultiDof();
    }
}

void btMultiBody::stepPositionsMultiDof(btScalar dt, btScalar *pq, btScalar *pqd)
{
	int num_links = getNumLinks();
	if(!isBaseKinematic())
	{
		// step position by adding dt * velocity
		//btVector3 v = getBaseVel();
		//m_basePos += dt * v;
		//
  	  btScalar *pBasePos = (pq ? &pq[4] : m_basePos);
  	  btScalar *pBaseVel = (pqd ? &pqd[3] : &m_realBuf[3]);  //note: the !pqd case assumes m_realBuf holds with base velocity at 3,4,5 (should be wrapped for safety)
  	  
		pBasePos[0] += dt * pBaseVel[0];
		pBasePos[1] += dt * pBaseVel[1];
		pBasePos[2] += dt * pBaseVel[2];
	}

	///////////////////////////////
	//local functor for quaternion integration (to avoid error prone redundancy)
	struct
	{
		//"exponential map" based on btTransformUtil::integrateTransform(..)
		void operator()(const btVector3 &omega, btQuaternion &quat, bool baseBody, btScalar dt)
		{
			//baseBody	=>	quat is alias and omega is global coor
			//!baseBody	=>	quat is alibi and omega is local coor

			btVector3 axis;
			btVector3 angvel;

			if (!baseBody)
				angvel = quatRotate(quat, omega);  //if quat is not m_baseQuat, it is alibi => ok
			else
				angvel = omega;

			btScalar fAngle = angvel.length();
			//limit the angular motion
			if (fAngle * dt > ANGULAR_MOTION_THRESHOLD)
			{
				fAngle = btScalar(0.5) * SIMD_HALF_PI / dt;
			}

			if (fAngle < btScalar(0.001))
			{
				// use Taylor's expansions of sync function
				axis = angvel * (btScalar(0.5) * dt - (dt * dt * dt) * (btScalar(0.020833333333)) * fAngle * fAngle);
			}
			else
			{
				// sync(fAngle) = sin(c*fAngle)/t
				axis = angvel * (btSin(btScalar(0.5) * fAngle * dt) / fAngle);
			}

			if (!baseBody)
				quat = btQuaternion(axis.x(), axis.y(), axis.z(), btCos(fAngle * dt * btScalar(0.5))) * quat;
			else
				quat = quat * btQuaternion(-axis.x(), -axis.y(), -axis.z(), btCos(fAngle * dt * btScalar(0.5)));
			//equivalent to: quat = (btQuaternion(axis.x(),axis.y(),axis.z(),btCos( fAngle*dt*btScalar(0.5) )) * quat.inverse()).inverse();

			quat.normalize();
		}
	} pQuatUpdateFun;
	///////////////////////////////

	//pQuatUpdateFun(getBaseOmega(), m_baseQuat, true, dt);
	//
	if(!isBaseKinematic())
	{
		btScalar *pBaseQuat = pq ? pq : m_baseQuat;
		btScalar *pBaseOmega = pqd ? pqd : &m_realBuf[0];  //note: the !pqd case assumes m_realBuf starts with base omega (should be wrapped for safety)
		//
		btQuaternion baseQuat;
		baseQuat.setValue(pBaseQuat[0], pBaseQuat[1], pBaseQuat[2], pBaseQuat[3]);
		btVector3 baseOmega;
		baseOmega.setValue(pBaseOmega[0], pBaseOmega[1], pBaseOmega[2]);
		pQuatUpdateFun(baseOmega, baseQuat, true, dt);
		pBaseQuat[0] = baseQuat.x();
		pBaseQuat[1] = baseQuat.y();
		pBaseQuat[2] = baseQuat.z();
		pBaseQuat[3] = baseQuat.w();

		//printf("pBaseOmega = %.4f %.4f %.4f\n", pBaseOmega->x(), pBaseOmega->y(), pBaseOmega->z());
		//printf("pBaseVel = %.4f %.4f %.4f\n", pBaseVel->x(), pBaseVel->y(), pBaseVel->z());
		//printf("baseQuat = %.4f %.4f %.4f %.4f\n", pBaseQuat->x(), pBaseQuat->y(), pBaseQuat->z(), pBaseQuat->w());
	}

	if (pq)
		pq += 7;
	if (pqd)
		pqd += 6;

	// Finally we can update m_jointPos for each of the m_links
	for (int i = 0; i < num_links; ++i)
	{
		if (!(m_links[i].m_collider && m_links[i].m_collider->isStaticOrKinematic()))
		{
			btScalar *pJointPos;
			pJointPos= (pq ? pq : &m_links[i].m_jointPos[0]);
		
			btScalar *pJointVel = (pqd ? pqd : getJointVelMultiDof(i));

			switch (m_links[i].m_jointType)
			{
				case btMultibodyLink::ePrismatic:
				case btMultibodyLink::eRevolute:
				{
    	            //reset to current pos
					btScalar jointVel = pJointVel[0];
					pJointPos[0] += dt * jointVel;
					break;
				}
				case btMultibodyLink::eSpherical:
				{
    	            //reset to current pos
					btVector3 jointVel;
					jointVel.setValue(pJointVel[0], pJointVel[1], pJointVel[2]);
					btQuaternion jointOri;
					jointOri.setValue(pJointPos[0], pJointPos[1], pJointPos[2], pJointPos[3]);
					pQuatUpdateFun(jointVel, jointOri, false, dt);
					pJointPos[0] = jointOri.x();
					pJointPos[1] = jointOri.y();
					pJointPos[2] = jointOri.z();
					pJointPos[3] = jointOri.w();
					break;
				}
				case btMultibodyLink::ePlanar:
				{
					pJointPos[0] += dt * getJointVelMultiDof(i)[0];

					btVector3 q0_coors_qd1qd2 = getJointVelMultiDof(i)[1] * m_links[i].getAxisBottom(1) + getJointVelMultiDof(i)[2] * m_links[i].getAxisBottom(2);
					btVector3 no_q0_coors_qd1qd2 = quatRotate(btQuaternion(m_links[i].getAxisTop(0), pJointPos[0]), q0_coors_qd1qd2);
					pJointPos[1] += m_links[i].getAxisBottom(1).dot(no_q0_coors_qd1qd2) * dt;
					pJointPos[2] += m_links[i].getAxisBottom(2).dot(no_q0_coors_qd1qd2) * dt;

					break;
				}
				default:
				{
				}
			}
		}

		m_links[i].updateCacheMultiDof(pq);

		if (pq)
			pq += m_links[i].m_posVarCount;
		if (pqd)
			pqd += m_links[i].m_dofCount;
	}
}

void btMultiBody::fillConstraintJacobianMultiDof(int link,
												 const btVector3 &contact_point,
												 const btVector3 &normal_ang,
												 const btVector3 &normal_lin,
												 btScalar *jac,
												 btAlignedObjectArray<btScalar> &scratch_r1,
												 btAlignedObjectArray<btVector3> &scratch_v,
												 btAlignedObjectArray<btMatrix3x3> &scratch_m) const
{
	// temporary space
	int num_links = getNumLinks();
	int m_dofCount = getNumDofs();
	scratch_v.resize(3 * num_links + 3);  //(num_links + base) offsets + (num_links + base) normals_lin + (num_links + base) normals_ang
	scratch_m.resize(num_links + 1);

	btVector3 *v_ptr = &scratch_v[0];
	btVector3 *p_minus_com_local = v_ptr;
	v_ptr += num_links + 1;
	btVector3 *n_local_lin = v_ptr;
	v_ptr += num_links + 1;
	btVector3 *n_local_ang = v_ptr;
	v_ptr += num_links + 1;
	btAssert(v_ptr - &scratch_v[0] == scratch_v.size());

	//scratch_r.resize(m_dofCount);
	//btScalar *results = m_dofCount > 0 ? &scratch_r[0] : 0;

    scratch_r1.resize(m_dofCount+num_links);
    btScalar * results = m_dofCount > 0 ? &scratch_r1[0] : 0;
    btScalar* links = num_links? &scratch_r1[m_dofCount] : 0;
    int numLinksChildToRoot=0;
    int l = link;
    while (l != -1)
    {
        links[numLinksChildToRoot++]=l;
        l = m_links[l].m_parent;
    }
    
	btMatrix3x3 *rot_from_world = &scratch_m[0];

	const btVector3 p_minus_com_world = contact_point - m_basePos;
	const btVector3 &normal_lin_world = normal_lin;  //convenience
	const btVector3 &normal_ang_world = normal_ang;

	rot_from_world[0] = btMatrix3x3(m_baseQuat);

	// omega coeffients first.
	btVector3 omega_coeffs_world;
	omega_coeffs_world = p_minus_com_world.cross(normal_lin_world);
	jac[0] = omega_coeffs_world[0] + normal_ang_world[0];
	jac[1] = omega_coeffs_world[1] + normal_ang_world[1];
	jac[2] = omega_coeffs_world[2] + normal_ang_world[2];
	// then v coefficients
	jac[3] = normal_lin_world[0];
	jac[4] = normal_lin_world[1];
	jac[5] = normal_lin_world[2];

	//create link-local versions of p_minus_com and normal
	p_minus_com_local[0] = rot_from_world[0] * p_minus_com_world;
	n_local_lin[0] = rot_from_world[0] * normal_lin_world;
	n_local_ang[0] = rot_from_world[0] * normal_ang_world;

	// Set remaining jac values to zero for now.
	for (int i = 6; i < 6 + m_dofCount; ++i)
	{
		jac[i] = 0;
	}

	// Qdot coefficients, if necessary.
	if (num_links > 0 && link > -1)
	{
        // TODO: (Also, we are making 3 separate calls to this function, for the normal & the 2 friction directions,
		// which is resulting in repeated work being done...)

		// calculate required normals & positions in the local frames.
        for (int a = 0; a < numLinksChildToRoot; a++)
        {
            int i = links[numLinksChildToRoot-1-a];
        	// transform to local frame
			const int parent = m_links[i].m_parent;
			const btMatrix3x3 mtx(m_links[i].m_cachedRotParentToThis);
			rot_from_world[i + 1] = mtx * rot_from_world[parent + 1];

			n_local_lin[i + 1] = mtx * n_local_lin[parent + 1];
			n_local_ang[i + 1] = mtx * n_local_ang[parent + 1];
			p_minus_com_local[i + 1] = mtx * p_minus_com_local[parent + 1] - m_links[i].m_cachedRVector;

			// calculate the jacobian entry
			switch (m_links[i].m_jointType)
			{
				case btMultibodyLink::eRevolute:
				{
					results[m_links[i].m_dofOffset] = n_local_lin[i + 1].dot(m_links[i].getAxisTop(0).cross(p_minus_com_local[i + 1]) + m_links[i].getAxisBottom(0));
					results[m_links[i].m_dofOffset] += n_local_ang[i + 1].dot(m_links[i].getAxisTop(0));
					break;
				}
				case btMultibodyLink::ePrismatic:
				{
					results[m_links[i].m_dofOffset] = n_local_lin[i + 1].dot(m_links[i].getAxisBottom(0));
					break;
				}
				case btMultibodyLink::eSpherical:
				{
					results[m_links[i].m_dofOffset + 0] = n_local_lin[i + 1].dot(m_links[i].getAxisTop(0).cross(p_minus_com_local[i + 1]) + m_links[i].getAxisBottom(0));
					results[m_links[i].m_dofOffset + 1] = n_local_lin[i + 1].dot(m_links[i].getAxisTop(1).cross(p_minus_com_local[i + 1]) + m_links[i].getAxisBottom(1));
					results[m_links[i].m_dofOffset + 2] = n_local_lin[i + 1].dot(m_links[i].getAxisTop(2).cross(p_minus_com_local[i + 1]) + m_links[i].getAxisBottom(2));

					results[m_links[i].m_dofOffset + 0] += n_local_ang[i + 1].dot(m_links[i].getAxisTop(0));
					results[m_links[i].m_dofOffset + 1] += n_local_ang[i + 1].dot(m_links[i].getAxisTop(1));
					results[m_links[i].m_dofOffset + 2] += n_local_ang[i + 1].dot(m_links[i].getAxisTop(2));

					break;
				}
				case btMultibodyLink::ePlanar:
				{
					results[m_links[i].m_dofOffset + 0] = n_local_lin[i + 1].dot(m_links[i].getAxisTop(0).cross(p_minus_com_local[i + 1]));  // + m_links[i].getAxisBottom(0));
					results[m_links[i].m_dofOffset + 1] = n_local_lin[i + 1].dot(m_links[i].getAxisBottom(1));
					results[m_links[i].m_dofOffset + 2] = n_local_lin[i + 1].dot(m_links[i].getAxisBottom(2));

					break;
				}
				default:
				{
				}
			}
		}

		// Now copy through to output.
		//printf("jac[%d] = ", link);
		while (link != -1)
		{
			for (int dof = 0; dof < m_links[link].m_dofCount; ++dof)
			{
				jac[6 + m_links[link].m_dofOffset + dof] = results[m_links[link].m_dofOffset + dof];
				//printf("%.2f\t", jac[6 + m_links[link].m_dofOffset + dof]);
			}

			link = m_links[link].m_parent;
		}
		//printf("]\n");
	}
}

void btMultiBody::wakeUp()
{
	m_sleepTimer = 0;
	m_awake = true;
}

void btMultiBody::goToSleep()
{
	m_awake = false;
}

void btMultiBody::checkMotionAndSleepIfRequired(btScalar timestep)
{
	extern bool gDisableDeactivation;
	if (!m_canSleep || gDisableDeactivation)
	{
		m_awake = true;
		m_sleepTimer = 0;
		return;
	}

	

	// motion is computed as omega^2 + v^2 + (sum of squares of joint velocities)
	btScalar motion = 0;
	{
		for (int i = 0; i < 6 + m_dofCount; ++i)
			motion += m_realBuf[i] * m_realBuf[i];
	}

	if (motion < m_sleepEpsilon)
	{
		m_sleepTimer += timestep;
		if (m_sleepTimer > m_sleepTimeout)
		{
			goToSleep();
		}
	}
	else
	{
		m_sleepTimer = 0;
		if (m_canWakeup)
		{
			if (!m_awake)
				wakeUp();
		}
	}
}

void btMultiBody::forwardKinematics(btAlignedObjectArray<btQuaternion> &world_to_local, btAlignedObjectArray<btVector3> &local_origin)
{
	int num_links = getNumLinks();

	// Cached 3x3 rotation matrices from parent frame to this frame.
	btMatrix3x3 *rot_from_parent = (btMatrix3x3 *)&m_matrixBuf[0];

	rot_from_parent[0] = btMatrix3x3(m_baseQuat);  //m_baseQuat assumed to be alias!?

	for (int i = 0; i < num_links; ++i)
	{
		rot_from_parent[i + 1] = btMatrix3x3(m_links[i].m_cachedRotParentToThis);
	}

	int nLinks = getNumLinks();
	///base + num m_links
	world_to_local.resize(nLinks + 1);
	local_origin.resize(nLinks + 1);

	world_to_local[0] = getWorldToBaseRot();
	local_origin[0] = getBasePos();

	for (int k = 0; k < getNumLinks(); k++)
	{
		const int parent = getParent(k);
		world_to_local[k + 1] = getParentToLocalRot(k) * world_to_local[parent + 1];
		local_origin[k + 1] = local_origin[parent + 1] + (quatRotate(world_to_local[k + 1].inverse(), getRVector(k)));
	}

	for (int link = 0; link < getNumLinks(); link++)
	{
		int index = link + 1;

		btVector3 posr = local_origin[index];
		btScalar quat[4] = {-world_to_local[index].x(), -world_to_local[index].y(), -world_to_local[index].z(), world_to_local[index].w()};
		btTransform tr;
		tr.setIdentity();
		tr.setOrigin(posr);
		tr.setRotation(btQuaternion(quat[0], quat[1], quat[2], quat[3]));
		getLink(link).m_cachedWorldTransform = tr;
	}
}

void btMultiBody::updateCollisionObjectWorldTransforms(btAlignedObjectArray<btQuaternion> &world_to_local, btAlignedObjectArray<btVector3> &local_origin)
{
	world_to_local.resize(getNumLinks() + 1);
	local_origin.resize(getNumLinks() + 1);

	world_to_local[0] = getWorldToBaseRot();
	local_origin[0] = getBasePos();

	if (getBaseCollider())
	{
		btVector3 posr = local_origin[0];
		//	float pos[4]={posr.x(),posr.y(),posr.z(),1};
		btScalar quat[4] = {-world_to_local[0].x(), -world_to_local[0].y(), -world_to_local[0].z(), world_to_local[0].w()};
		btTransform tr;
		tr.setIdentity();
		tr.setOrigin(posr);
		tr.setRotation(btQuaternion(quat[0], quat[1], quat[2], quat[3]));

		getBaseCollider()->setWorldTransform(tr);
        getBaseCollider()->setInterpolationWorldTransform(tr);
	}

	for (int k = 0; k < getNumLinks(); k++)
	{
		const int parent = getParent(k);
		world_to_local[k + 1] = getParentToLocalRot(k) * world_to_local[parent + 1];
		local_origin[k + 1] = local_origin[parent + 1] + (quatRotate(world_to_local[k + 1].inverse(), getRVector(k)));
	}

	for (int m = 0; m < getNumLinks(); m++)
	{
		btMultiBodyLinkCollider *col = getLink(m).m_collider;
		if (col)
		{
			int link = col->m_link;
			btAssert(link == m);

			int index = link + 1;

			btVector3 posr = local_origin[index];
			//			float pos[4]={posr.x(),posr.y(),posr.z(),1};
			btScalar quat[4] = {-world_to_local[index].x(), -world_to_local[index].y(), -world_to_local[index].z(), world_to_local[index].w()};
			btTransform tr;
			tr.setIdentity();
			tr.setOrigin(posr);
			tr.setRotation(btQuaternion(quat[0], quat[1], quat[2], quat[3]));

			col->setWorldTransform(tr);
            col->setInterpolationWorldTransform(tr);
		}
	}
}

void btMultiBody::updateCollisionObjectInterpolationWorldTransforms(btAlignedObjectArray<btQuaternion> &world_to_local, btAlignedObjectArray<btVector3> &local_origin)
{
    world_to_local.resize(getNumLinks() + 1);
    local_origin.resize(getNumLinks() + 1);
    
		if(isBaseKinematic()){
        world_to_local[0] = getWorldToBaseRot();
        local_origin[0] = getBasePos();
		}
		else
		{
        world_to_local[0] = getInterpolateWorldToBaseRot();
        local_origin[0] = getInterpolateBasePos();
		}
    
    if (getBaseCollider())
    {
        btVector3 posr = local_origin[0];
        //    float pos[4]={posr.x(),posr.y(),posr.z(),1};
        btScalar quat[4] = {-world_to_local[0].x(), -world_to_local[0].y(), -world_to_local[0].z(), world_to_local[0].w()};
        btTransform tr;
        tr.setIdentity();
        tr.setOrigin(posr);
        tr.setRotation(btQuaternion(quat[0], quat[1], quat[2], quat[3]));
        
        getBaseCollider()->setInterpolationWorldTransform(tr);
    }
    
    for (int k = 0; k < getNumLinks(); k++)
    {
        const int parent = getParent(k);
        world_to_local[k + 1] = getInterpolateParentToLocalRot(k) * world_to_local[parent + 1];
        local_origin[k + 1] = local_origin[parent + 1] + (quatRotate(world_to_local[k + 1].inverse(), getInterpolateRVector(k)));
    }
    
    for (int m = 0; m < getNumLinks(); m++)
    {
        btMultiBodyLinkCollider *col = getLink(m).m_collider;
        if (col)
        {
            int link = col->m_link;
            btAssert(link == m);
            
            int index = link + 1;
            
            btVector3 posr = local_origin[index];
            //            float pos[4]={posr.x(),posr.y(),posr.z(),1};
            btScalar quat[4] = {-world_to_local[index].x(), -world_to_local[index].y(), -world_to_local[index].z(), world_to_local[index].w()};
            btTransform tr;
            tr.setIdentity();
            tr.setOrigin(posr);
            tr.setRotation(btQuaternion(quat[0], quat[1], quat[2], quat[3]));
            
            col->setInterpolationWorldTransform(tr);
        }
    }
}

int btMultiBody::calculateSerializeBufferSize() const
{
	int sz = sizeof(btMultiBodyData);
	return sz;
}

///fills the dataBuffer and returns the struct name (and 0 on failure)
const char *btMultiBody::serialize(void *dataBuffer, class btSerializer *serializer) const
{
	btMultiBodyData *mbd = (btMultiBodyData *)dataBuffer;
	getBasePos().serialize(mbd->m_baseWorldPosition);
	getWorldToBaseRot().inverse().serialize(mbd->m_baseWorldOrientation);
	getBaseVel().serialize(mbd->m_baseLinearVelocity);
	getBaseOmega().serialize(mbd->m_baseAngularVelocity);

	mbd->m_baseMass = this->getBaseMass();
	getBaseInertia().serialize(mbd->m_baseInertia);
	{
		char *name = (char *)serializer->findNameForPointer(m_baseName);
		mbd->m_baseName = (char *)serializer->getUniquePointer(name);
		if (mbd->m_baseName)
		{
			serializer->serializeName(name);
		}
	}
	mbd->m_numLinks = this->getNumLinks();
	if (mbd->m_numLinks)
	{
		int sz = sizeof(btMultiBodyLinkData);
		int numElem = mbd->m_numLinks;
		btChunk *chunk = serializer->allocate(sz, numElem);
		btMultiBodyLinkData *memPtr = (btMultiBodyLinkData *)chunk->m_oldPtr;
		for (int i = 0; i < numElem; i++, memPtr++)
		{
			memPtr->m_jointType = getLink(i).m_jointType;
			memPtr->m_dofCount = getLink(i).m_dofCount;
			memPtr->m_posVarCount = getLink(i).m_posVarCount;

			getLink(i).m_inertiaLocal.serialize(memPtr->m_linkInertia);

			getLink(i).m_absFrameTotVelocity.m_topVec.serialize(memPtr->m_absFrameTotVelocityTop);
			getLink(i).m_absFrameTotVelocity.m_bottomVec.serialize(memPtr->m_absFrameTotVelocityBottom);
			getLink(i).m_absFrameLocVelocity.m_topVec.serialize(memPtr->m_absFrameLocVelocityTop);
			getLink(i).m_absFrameLocVelocity.m_bottomVec.serialize(memPtr->m_absFrameLocVelocityBottom);

			memPtr->m_linkMass = getLink(i).m_mass;
			memPtr->m_parentIndex = getLink(i).m_parent;
			memPtr->m_jointDamping = getLink(i).m_jointDamping;
			memPtr->m_jointFriction = getLink(i).m_jointFriction;
			memPtr->m_jointLowerLimit = getLink(i).m_jointLowerLimit;
			memPtr->m_jointUpperLimit = getLink(i).m_jointUpperLimit;
			memPtr->m_jointMaxForce = getLink(i).m_jointMaxForce;
			memPtr->m_jointMaxVelocity = getLink(i).m_jointMaxVelocity;

			getLink(i).m_eVector.serialize(memPtr->m_parentComToThisPivotOffset);
			getLink(i).m_dVector.serialize(memPtr->m_thisPivotToThisComOffset);
			getLink(i).m_zeroRotParentToThis.serialize(memPtr->m_zeroRotParentToThis);
			btAssert(memPtr->m_dofCount <= 3);
			for (int dof = 0; dof < getLink(i).m_dofCount; dof++)
			{
				getLink(i).getAxisBottom(dof).serialize(memPtr->m_jointAxisBottom[dof]);
				getLink(i).getAxisTop(dof).serialize(memPtr->m_jointAxisTop[dof]);

				memPtr->m_jointTorque[dof] = getLink(i).m_jointTorque[dof];
				memPtr->m_jointVel[dof] = getJointVelMultiDof(i)[dof];
			}
			int numPosVar = getLink(i).m_posVarCount;
			for (int posvar = 0; posvar < numPosVar; posvar++)
			{
				memPtr->m_jointPos[posvar] = getLink(i).m_jointPos[posvar];
			}

			{
				char *name = (char *)serializer->findNameForPointer(m_links[i].m_linkName);
				memPtr->m_linkName = (char *)serializer->getUniquePointer(name);
				if (memPtr->m_linkName)
				{
					serializer->serializeName(name);
				}
			}
			{
				char *name = (char *)serializer->findNameForPointer(m_links[i].m_jointName);
				memPtr->m_jointName = (char *)serializer->getUniquePointer(name);
				if (memPtr->m_jointName)
				{
					serializer->serializeName(name);
				}
			}
			memPtr->m_linkCollider = (btCollisionObjectData *)serializer->getUniquePointer(getLink(i).m_collider);
		}
		serializer->finalizeChunk(chunk, btMultiBodyLinkDataName, BT_ARRAY_CODE, (void *)&m_links[0]);
	}
	mbd->m_links = mbd->m_numLinks ? (btMultiBodyLinkData *)serializer->getUniquePointer((void *)&m_links[0]) : 0;

	// Fill padding with zeros to appease msan.
#ifdef BT_USE_DOUBLE_PRECISION
	memset(mbd->m_padding, 0, sizeof(mbd->m_padding));
#endif

	return btMultiBodyDataName;
}

void btMultiBody::saveKinematicState(btScalar timeStep)
{
	//todo: clamp to some (user definable) safe minimum timestep, to limit maximum angular/linear velocities
	if (m_kinematic_calculate_velocity && timeStep != btScalar(0.))
	{
		btVector3 linearVelocity, angularVelocity;
		btTransformUtil::calculateVelocity(getInterpolateBaseWorldTransform(), getBaseWorldTransform(), timeStep, linearVelocity, angularVelocity);
		setBaseVel(linearVelocity);
		setBaseOmega(angularVelocity);
		setInterpolateBaseWorldTransform(getBaseWorldTransform());
	}
}

void btMultiBody::setLinkDynamicType(const int i, int type)
{
	if (i == -1)
	{
		setBaseDynamicType(type);
	}
	else if (i >= 0 && i < getNumLinks())
	{
		if (m_links[i].m_collider)
		{
			m_links[i].m_collider->setDynamicType(type);
		}
	}
}

bool btMultiBody::isLinkStaticOrKinematic(const int i) const
{
	if (i == -1)
	{
		return isBaseStaticOrKinematic();
	}
	else
	{
		if (m_links[i].m_collider)
			return m_links[i].m_collider->isStaticOrKinematic();
	}
	return false;
}

bool btMultiBody::isLinkKinematic(const int i) const
{
	if (i == -1)
	{
		return isBaseKinematic();
	}
	else
	{
		if (m_links[i].m_collider)
			return m_links[i].m_collider->isKinematic();
	}
	return false;
}

bool btMultiBody::isLinkAndAllAncestorsStaticOrKinematic(const int i) const
{
	int link = i;
	while (link != -1) {
		if (!isLinkStaticOrKinematic(link))
			return false;
		link = m_links[link].m_parent;
	}
	return isBaseStaticOrKinematic();
}

bool btMultiBody::isLinkAndAllAncestorsKinematic(const int i) const
{
	int link = i;
	while (link != -1) {
		if (!isLinkKinematic(link))
			return false;
		link = m_links[link].m_parent;
	}
	return isBaseKinematic();
}
