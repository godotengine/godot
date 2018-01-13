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

#ifndef BT_MULTIBODY_LINK_H
#define BT_MULTIBODY_LINK_H

#include "LinearMath/btQuaternion.h"
#include "LinearMath/btVector3.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"

enum	btMultiBodyLinkFlags
{
	BT_MULTIBODYLINKFLAGS_DISABLE_PARENT_COLLISION = 1,
	BT_MULTIBODYLINKFLAGS_DISABLE_ALL_PARENT_COLLISION = 2,
};

//both defines are now permanently enabled
#define BT_MULTIBODYLINK_INCLUDE_PLANAR_JOINTS
#define TEST_SPATIAL_ALGEBRA_LAYER

//
// Various spatial helper functions
//

//namespace {


#include "LinearMath/btSpatialAlgebra.h"

//}

//
// Link struct
//

struct btMultibodyLink 
{

	BT_DECLARE_ALIGNED_ALLOCATOR();

    btScalar m_mass;         // mass of link
    btVector3 m_inertiaLocal;   // inertia of link (local frame; diagonal)

    int m_parent;         // index of the parent link (assumed to be < index of this link), or -1 if parent is the base link.

    btQuaternion m_zeroRotParentToThis;    // rotates vectors in parent-frame to vectors in local-frame (when q=0). constant.

    btVector3 m_dVector;   // vector from the inboard joint pos to this link's COM. (local frame.) constant.
	//this is set to zero for planar joint (see also m_eVector comment)
	
    // m_eVector is constant, but depends on the joint type:
    // revolute, fixed, prismatic, spherical: vector from parent's COM to the pivot point, in PARENT's frame.
	// planar: vector from COM of parent to COM of this link, WHEN Q = 0. (local frame.)
	// todo: fix the planar so it is consistent with the other joints
	
    btVector3 m_eVector;

	btSpatialMotionVector m_absFrameTotVelocity, m_absFrameLocVelocity;

	enum eFeatherstoneJointType
	{
		eRevolute = 0,
		ePrismatic = 1,
		eSpherical = 2,
		ePlanar = 3,
		eFixed = 4,
		eInvalid
	};

	

	// "axis" = spatial joint axis (Mirtich Defn 9 p104). (expressed in local frame.) constant.
    // for prismatic: m_axesTop[0] = zero;
    //                m_axesBottom[0] = unit vector along the joint axis.
    // for revolute: m_axesTop[0] = unit vector along the rotation axis (u);
    //               m_axesBottom[0] = u cross m_dVector (i.e. COM linear motion due to the rotation at the joint)
	//
	// for spherical: m_axesTop[0][1][2] (u1,u2,u3) form a 3x3 identity matrix (3 rotation axes)
	//				  m_axesBottom[0][1][2] cross u1,u2,u3 (i.e. COM linear motion due to the rotation at the joint)
	//
	// for planar: m_axesTop[0] = unit vector along the rotation axis (u); defines the plane of motion
	//			   m_axesTop[1][2] = zero
	//			   m_axesBottom[0] = zero
	//			   m_axesBottom[1][2] = unit vectors along the translational axes on that plane		
	btSpatialMotionVector m_axes[6];
	void setAxisTop(int dof, const btVector3 &axis) { m_axes[dof].m_topVec = axis; }
	void setAxisBottom(int dof, const btVector3 &axis) 
	{ 
		m_axes[dof].m_bottomVec = axis; 
	}
	void setAxisTop(int dof, const btScalar &x, const btScalar &y, const btScalar &z) 
	{
		m_axes[dof].m_topVec.setValue(x, y, z); 
	}
	void setAxisBottom(int dof, const btScalar &x, const btScalar &y, const btScalar &z) 
	{ 
		m_axes[dof].m_bottomVec.setValue(x, y, z); 
	}
	const btVector3 & getAxisTop(int dof) const { return m_axes[dof].m_topVec; }
	const btVector3 & getAxisBottom(int dof) const { return m_axes[dof].m_bottomVec; }

	int m_dofOffset, m_cfgOffset;

    btQuaternion m_cachedRotParentToThis;   // rotates vectors in parent frame to vectors in local frame
    btVector3 m_cachedRVector;                // vector from COM of parent to COM of this link, in local frame.

    btVector3 m_appliedForce;    // In WORLD frame
    btVector3 m_appliedTorque;   // In WORLD frame

btVector3 m_appliedConstraintForce;    // In WORLD frame
    btVector3 m_appliedConstraintTorque;   // In WORLD frame

	btScalar m_jointPos[7];
    
    //m_jointTorque is the joint torque applied by the user using 'addJointTorque'.
    //It gets set to zero after each internal stepSimulation call
	btScalar m_jointTorque[6];
    
	class btMultiBodyLinkCollider* m_collider;
	int m_flags;
	
	
	int m_dofCount, m_posVarCount;				//redundant but handy
	
	eFeatherstoneJointType m_jointType;
	
	struct btMultiBodyJointFeedback*	m_jointFeedback;

	btTransform	m_cachedWorldTransform;//this cache is updated when calling btMultiBody::forwardKinematics

	const char* m_linkName;//m_linkName memory needs to be managed by the developer/user!
	const char* m_jointName;//m_jointName memory needs to be managed by the developer/user!
    const void* m_userPtr;//m_userPtr ptr needs to be managed by the developer/user!
    
	btScalar m_jointDamping; //todo: implement this internally. It is unused for now, it is set by a URDF loader. User can apply manual damping.
	btScalar m_jointFriction; //todo: implement this internally. It is unused for now, it is set by a URDF loader. User can apply manual friction using a velocity motor.
	btScalar m_jointLowerLimit; //todo: implement this internally. It is unused for now, it is set by a URDF loader. 
	btScalar m_jointUpperLimit; //todo: implement this internally. It is unused for now, it is set by a URDF loader.
	btScalar m_jointMaxForce; //todo: implement this internally. It is unused for now, it is set by a URDF loader. 
	btScalar m_jointMaxVelocity;//todo: implement this internally. It is unused for now, it is set by a URDF loader. 
	
	// ctor: set some sensible defaults
	btMultibodyLink()
		: 	m_mass(1),
			m_parent(-1),
			m_zeroRotParentToThis(0, 0, 0, 1),
			m_cachedRotParentToThis(0, 0, 0, 1),
			m_collider(0),
			m_flags(0),
			m_dofCount(0),
			m_posVarCount(0),
			m_jointType(btMultibodyLink::eInvalid),
			m_jointFeedback(0),
			m_linkName(0),
			m_jointName(0),
            m_userPtr(0),
			m_jointDamping(0),
			m_jointFriction(0),
			m_jointLowerLimit(0),
			m_jointUpperLimit(0),
			m_jointMaxForce(0),
			m_jointMaxVelocity(0)
	{
		
		m_inertiaLocal.setValue(1, 1, 1);
		setAxisTop(0, 0., 0., 0.);
		setAxisBottom(0, 1., 0., 0.);
		m_dVector.setValue(0, 0, 0);
		m_eVector.setValue(0, 0, 0);
		m_cachedRVector.setValue(0, 0, 0);
		m_appliedForce.setValue( 0, 0, 0);
		m_appliedTorque.setValue(0, 0, 0);
		//		
		m_jointPos[0] = m_jointPos[1] = m_jointPos[2] = m_jointPos[4] = m_jointPos[5] = m_jointPos[6] = 0.f;
		m_jointPos[3] = 1.f;			//"quat.w"
		m_jointTorque[0] = m_jointTorque[1] = m_jointTorque[2] = m_jointTorque[3] = m_jointTorque[4] = m_jointTorque[5] = 0.f;
		m_cachedWorldTransform.setIdentity();
	}

    // routine to update m_cachedRotParentToThis and m_cachedRVector
	void updateCacheMultiDof(btScalar *pq = 0)
	{
		btScalar *pJointPos = (pq ? pq : &m_jointPos[0]);

		switch(m_jointType)
		{
			case eRevolute:
			{
				m_cachedRotParentToThis = btQuaternion(getAxisTop(0),-pJointPos[0]) * m_zeroRotParentToThis;
				m_cachedRVector = m_dVector + quatRotate(m_cachedRotParentToThis,m_eVector);

				break;
			}
			case ePrismatic:
			{
				// m_cachedRotParentToThis never changes, so no need to update
				m_cachedRVector = m_dVector + quatRotate(m_cachedRotParentToThis,m_eVector) + pJointPos[0] * getAxisBottom(0);

				break;
			}
			case eSpherical:
			{
				m_cachedRotParentToThis = btQuaternion(pJointPos[0], pJointPos[1], pJointPos[2], -pJointPos[3]) * m_zeroRotParentToThis;
				m_cachedRVector = m_dVector + quatRotate(m_cachedRotParentToThis,m_eVector);

				break;
			}
			case ePlanar:
			{
				m_cachedRotParentToThis = btQuaternion(getAxisTop(0),-pJointPos[0]) * m_zeroRotParentToThis;				
				m_cachedRVector = quatRotate(btQuaternion(getAxisTop(0),-pJointPos[0]), pJointPos[1] * getAxisBottom(1) + pJointPos[2] * getAxisBottom(2)) + quatRotate(m_cachedRotParentToThis,m_eVector);				

				break;
			}
			case eFixed:
			{
				m_cachedRotParentToThis = m_zeroRotParentToThis;
				m_cachedRVector = m_dVector + quatRotate(m_cachedRotParentToThis,m_eVector);

				break;
			}
			default:
			{
				//invalid type
				btAssert(0);
			}
		}
	}
};


#endif //BT_MULTIBODY_LINK_H
