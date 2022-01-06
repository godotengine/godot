/*
 Written by Xuchen Han <xuchenhan2015@u.northwestern.edu>
 
 Bullet Continuous Collision Detection and Physics Library
 Copyright (c) 2019 Google Inc. http://bulletphysics.org
 This software is provided 'as-is', without any express or implied warranty.
 In no event will the authors be held liable for any damages arising from the use of this software.
 Permission is granted to anyone to use this software for any purpose,
 including commercial applications, and to alter it and redistribute it freely,
 subject to the following restrictions:
 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef BT_DEFORMABLE_CONTACT_CONSTRAINT_H
#define BT_DEFORMABLE_CONTACT_CONSTRAINT_H
#include "btSoftBody.h"

// btDeformableContactConstraint is an abstract class specifying the method that each type of contact constraint needs to implement
class btDeformableContactConstraint
{
public:
	// True if the friction is static
	// False if the friction is dynamic
	bool m_static;
	const btContactSolverInfo* m_infoGlobal;

	// normal of the contact
	btVector3 m_normal;

	btDeformableContactConstraint(const btVector3& normal, const btContactSolverInfo& infoGlobal) : m_static(false), m_normal(normal), m_infoGlobal(&infoGlobal)
	{
	}

	btDeformableContactConstraint(bool isStatic, const btVector3& normal, const btContactSolverInfo& infoGlobal) : m_static(isStatic), m_normal(normal), m_infoGlobal(&infoGlobal)
	{
	}

	btDeformableContactConstraint() : m_static(false) {}

	btDeformableContactConstraint(const btDeformableContactConstraint& other)
		: m_static(other.m_static), m_normal(other.m_normal), m_infoGlobal(other.m_infoGlobal)
	{
	}

	virtual ~btDeformableContactConstraint() {}

	// solve the constraint with inelastic impulse and return the error, which is the square of normal component of velocity diffrerence
	// the constraint is solved by calculating the impulse between object A and B in the contact and apply the impulse to both objects involved in the contact
	virtual btScalar solveConstraint(const btContactSolverInfo& infoGlobal) = 0;

	// get the velocity of the object A in the contact
	virtual btVector3 getVa() const = 0;

	// get the velocity of the object B in the contact
	virtual btVector3 getVb() const = 0;

	// get the velocity change of the soft body node in the constraint
	virtual btVector3 getDv(const btSoftBody::Node*) const = 0;

	// apply impulse to the soft body node and/or face involved
	virtual void applyImpulse(const btVector3& impulse) = 0;

	// scale the penetration depth by erp
	virtual void setPenetrationScale(btScalar scale) = 0;
};

//
// Constraint that a certain node in the deformable objects cannot move
class btDeformableStaticConstraint : public btDeformableContactConstraint
{
public:
	btSoftBody::Node* m_node;

	btDeformableStaticConstraint(btSoftBody::Node* node, const btContactSolverInfo& infoGlobal) : m_node(node), btDeformableContactConstraint(false, btVector3(0, 0, 0), infoGlobal)
	{
	}
	btDeformableStaticConstraint() {}
	btDeformableStaticConstraint(const btDeformableStaticConstraint& other)
		: m_node(other.m_node), btDeformableContactConstraint(other)
	{
	}

	virtual ~btDeformableStaticConstraint() {}

	virtual btScalar solveConstraint(const btContactSolverInfo& infoGlobal)
	{
		return 0;
	}

	virtual btVector3 getVa() const
	{
		return btVector3(0, 0, 0);
	}

	virtual btVector3 getVb() const
	{
		return btVector3(0, 0, 0);
	}

	virtual btVector3 getDv(const btSoftBody::Node* n) const
	{
		return btVector3(0, 0, 0);
	}

	virtual void applyImpulse(const btVector3& impulse) {}
	virtual void setPenetrationScale(btScalar scale) {}
};

//
// Anchor Constraint between rigid and deformable node
class btDeformableNodeAnchorConstraint : public btDeformableContactConstraint
{
public:
	const btSoftBody::DeformableNodeRigidAnchor* m_anchor;

	btDeformableNodeAnchorConstraint(const btSoftBody::DeformableNodeRigidAnchor& c, const btContactSolverInfo& infoGlobal);
	btDeformableNodeAnchorConstraint(const btDeformableNodeAnchorConstraint& other);
	btDeformableNodeAnchorConstraint() {}
	virtual ~btDeformableNodeAnchorConstraint()
	{
	}
	virtual btScalar solveConstraint(const btContactSolverInfo& infoGlobal);

	// object A is the rigid/multi body, and object B is the deformable node/face
	virtual btVector3 getVa() const;
	// get the velocity of the deformable node in contact
	virtual btVector3 getVb() const;
	virtual btVector3 getDv(const btSoftBody::Node* n) const
	{
		return btVector3(0, 0, 0);
	}
	virtual void applyImpulse(const btVector3& impulse);

	virtual void setPenetrationScale(btScalar scale) {}
};

//
// Constraint between rigid/multi body and deformable objects
class btDeformableRigidContactConstraint : public btDeformableContactConstraint
{
public:
	btVector3 m_total_normal_dv;
	btVector3 m_total_tangent_dv;
	btScalar m_penetration;
	btScalar m_total_split_impulse;
	bool m_binding;
	const btSoftBody::DeformableRigidContact* m_contact;

	btDeformableRigidContactConstraint(const btSoftBody::DeformableRigidContact& c, const btContactSolverInfo& infoGlobal);
	btDeformableRigidContactConstraint(const btDeformableRigidContactConstraint& other);
	btDeformableRigidContactConstraint() : m_binding(false) {}
	virtual ~btDeformableRigidContactConstraint()
	{
	}

	// object A is the rigid/multi body, and object B is the deformable node/face
	virtual btVector3 getVa() const;

	// get the split impulse velocity of the deformable face at the contact point
	virtual btVector3 getSplitVb() const = 0;

	// get the split impulse velocity of the rigid/multibdoy at the contaft
	virtual btVector3 getSplitVa() const;

	virtual btScalar solveConstraint(const btContactSolverInfo& infoGlobal);

	virtual void setPenetrationScale(btScalar scale)
	{
		m_penetration *= scale;
	}

	btScalar solveSplitImpulse(const btContactSolverInfo& infoGlobal);

	virtual void applySplitImpulse(const btVector3& impulse) = 0;
};

//
// Constraint between rigid/multi body and deformable objects nodes
class btDeformableNodeRigidContactConstraint : public btDeformableRigidContactConstraint
{
public:
	// the deformable node in contact
	btSoftBody::Node* m_node;

	btDeformableNodeRigidContactConstraint(const btSoftBody::DeformableNodeRigidContact& contact, const btContactSolverInfo& infoGlobal);
	btDeformableNodeRigidContactConstraint(const btDeformableNodeRigidContactConstraint& other);
	btDeformableNodeRigidContactConstraint() {}
	virtual ~btDeformableNodeRigidContactConstraint()
	{
	}

	// get the velocity of the deformable node in contact
	virtual btVector3 getVb() const;

	// get the split impulse velocity of the deformable face at the contact point
	virtual btVector3 getSplitVb() const;

	// get the velocity change of the input soft body node in the constraint
	virtual btVector3 getDv(const btSoftBody::Node*) const;

	// cast the contact to the desired type
	const btSoftBody::DeformableNodeRigidContact* getContact() const
	{
		return static_cast<const btSoftBody::DeformableNodeRigidContact*>(m_contact);
	}

	virtual void applyImpulse(const btVector3& impulse);

	virtual void applySplitImpulse(const btVector3& impulse);
};

//
// Constraint between rigid/multi body and deformable objects faces
class btDeformableFaceRigidContactConstraint : public btDeformableRigidContactConstraint
{
public:
	btSoftBody::Face* m_face;
	bool m_useStrainLimiting;
	btDeformableFaceRigidContactConstraint(const btSoftBody::DeformableFaceRigidContact& contact, const btContactSolverInfo& infoGlobal, bool useStrainLimiting);
	btDeformableFaceRigidContactConstraint(const btDeformableFaceRigidContactConstraint& other);
	btDeformableFaceRigidContactConstraint() : m_useStrainLimiting(false) {}
	virtual ~btDeformableFaceRigidContactConstraint()
	{
	}

	// get the velocity of the deformable face at the contact point
	virtual btVector3 getVb() const;

	// get the split impulse velocity of the deformable face at the contact point
	virtual btVector3 getSplitVb() const;

	// get the velocity change of the input soft body node in the constraint
	virtual btVector3 getDv(const btSoftBody::Node*) const;

	// cast the contact to the desired type
	const btSoftBody::DeformableFaceRigidContact* getContact() const
	{
		return static_cast<const btSoftBody::DeformableFaceRigidContact*>(m_contact);
	}

	virtual void applyImpulse(const btVector3& impulse);

	virtual void applySplitImpulse(const btVector3& impulse);
};

//
// Constraint between  deformable objects faces and deformable objects nodes
class btDeformableFaceNodeContactConstraint : public btDeformableContactConstraint
{
public:
	btSoftBody::Node* m_node;
	btSoftBody::Face* m_face;
	const btSoftBody::DeformableFaceNodeContact* m_contact;
	btVector3 m_total_normal_dv;
	btVector3 m_total_tangent_dv;

	btDeformableFaceNodeContactConstraint(const btSoftBody::DeformableFaceNodeContact& contact, const btContactSolverInfo& infoGlobal);
	btDeformableFaceNodeContactConstraint() {}
	virtual ~btDeformableFaceNodeContactConstraint() {}

	virtual btScalar solveConstraint(const btContactSolverInfo& infoGlobal);

	// get the velocity of the object A in the contact
	virtual btVector3 getVa() const;

	// get the velocity of the object B in the contact
	virtual btVector3 getVb() const;

	// get the velocity change of the input soft body node in the constraint
	virtual btVector3 getDv(const btSoftBody::Node*) const;

	// cast the contact to the desired type
	const btSoftBody::DeformableFaceNodeContact* getContact() const
	{
		return static_cast<const btSoftBody::DeformableFaceNodeContact*>(m_contact);
	}

	virtual void applyImpulse(const btVector3& impulse);

	virtual void setPenetrationScale(btScalar scale) {}
};
#endif /* BT_DEFORMABLE_CONTACT_CONSTRAINT_H */
