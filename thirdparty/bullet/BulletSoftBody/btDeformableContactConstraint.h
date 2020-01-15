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
    
    // normal of the contact
    btVector3 m_normal;
    
    btDeformableContactConstraint(const btVector3& normal): m_static(false), m_normal(normal)
    {
    }
    
    btDeformableContactConstraint(bool isStatic, const btVector3& normal): m_static(isStatic), m_normal(normal)
    {
    }
    
    btDeformableContactConstraint(const btDeformableContactConstraint& other)
    : m_static(other.m_static)
    , m_normal(other.m_normal)
    {
        
    }
    btDeformableContactConstraint(){}
    
    virtual ~btDeformableContactConstraint(){}
    
    // solve the constraint with inelastic impulse and return the error, which is the square of normal component of velocity diffrerence
    // the constraint is solved by calculating the impulse between object A and B in the contact and apply the impulse to both objects involved in the contact
    virtual btScalar solveConstraint() = 0;
    
    // solve the position error by applying an inelastic impulse that changes only the position (not velocity)
    virtual btScalar solveSplitImpulse(const btContactSolverInfo& infoGlobal) = 0;
    
    // get the velocity of the object A in the contact
    virtual btVector3 getVa() const = 0;
    
    // get the velocity of the object B in the contact
    virtual btVector3 getVb() const = 0;
    
    // get the velocity change of the soft body node in the constraint
    virtual btVector3 getDv(const btSoftBody::Node*) const = 0;
    
    // apply impulse to the soft body node and/or face involved
    virtual void applyImpulse(const btVector3& impulse) = 0;
    
    // apply position based impulse to the soft body node and/or face involved
    virtual void applySplitImpulse(const btVector3& impulse) = 0;
    
    // scale the penetration depth by erp
    virtual void setPenetrationScale(btScalar scale) = 0;
};

//
// Constraint that a certain node in the deformable objects cannot move
class btDeformableStaticConstraint : public btDeformableContactConstraint
{
public:
    const btSoftBody::Node* m_node;
    
    btDeformableStaticConstraint(){}
    
    btDeformableStaticConstraint(const btSoftBody::Node* node): m_node(node), btDeformableContactConstraint(false, btVector3(0,0,0))
    {
    }
    
    btDeformableStaticConstraint(const btDeformableStaticConstraint& other)
    : m_node(other.m_node)
    , btDeformableContactConstraint(other)
    {
        
    }
    
    virtual ~btDeformableStaticConstraint(){}
    
    virtual btScalar solveConstraint()
    {
        return 0;
    }
    
    virtual btScalar solveSplitImpulse(const btContactSolverInfo& infoGlobal)
    {
        return 0;
    }

    virtual btVector3 getVa() const
    {
        return btVector3(0,0,0);
    }
    
    virtual btVector3 getVb() const
    {
        return btVector3(0,0,0);
    }
    
    virtual btVector3 getDv(const btSoftBody::Node* n) const
    {
        return btVector3(0,0,0);
    }
    
    virtual void applyImpulse(const btVector3& impulse){}
    virtual void applySplitImpulse(const btVector3& impulse){}
    virtual void setPenetrationScale(btScalar scale){}
};

//
// Anchor Constraint between rigid and deformable node
class btDeformableNodeAnchorConstraint : public btDeformableContactConstraint
{
public:
    const btSoftBody::DeformableNodeRigidAnchor* m_anchor;
    
    btDeformableNodeAnchorConstraint(){}
    btDeformableNodeAnchorConstraint(const btSoftBody::DeformableNodeRigidAnchor& c);
    btDeformableNodeAnchorConstraint(const btDeformableNodeAnchorConstraint& other);
    virtual ~btDeformableNodeAnchorConstraint()
    {
    }
    virtual btScalar solveConstraint();
    virtual btScalar solveSplitImpulse(const btContactSolverInfo& infoGlobal)
    {
        // todo xuchenhan@
        return 0;
    }
    // object A is the rigid/multi body, and object B is the deformable node/face
    virtual btVector3 getVa() const;
    // get the velocity of the deformable node in contact
    virtual btVector3 getVb() const;
    virtual btVector3 getDv(const btSoftBody::Node* n) const
    {
        return btVector3(0,0,0);
    }
    virtual void applyImpulse(const btVector3& impulse);
    virtual void applySplitImpulse(const btVector3& impulse)
    {
        // todo xuchenhan@
    };
    virtual void setPenetrationScale(btScalar scale){}
};


//
// Constraint between rigid/multi body and deformable objects
class btDeformableRigidContactConstraint : public btDeformableContactConstraint
{
public:
    btVector3 m_total_normal_dv;
    btVector3 m_total_tangent_dv;
    btScalar m_penetration;
    const btSoftBody::DeformableRigidContact* m_contact;
    
    btDeformableRigidContactConstraint(){}
    btDeformableRigidContactConstraint(const btSoftBody::DeformableRigidContact& c);
    btDeformableRigidContactConstraint(const btDeformableRigidContactConstraint& other);
    virtual ~btDeformableRigidContactConstraint()
    {
    }
    
    // object A is the rigid/multi body, and object B is the deformable node/face
    virtual btVector3 getVa() const;
    
    virtual btScalar solveConstraint();
    
    virtual btScalar solveSplitImpulse(const btContactSolverInfo& infoGlobal);
    
    virtual void setPenetrationScale(btScalar scale)
    {
        m_penetration *= scale;
    }
};

//
// Constraint between rigid/multi body and deformable objects nodes
class btDeformableNodeRigidContactConstraint : public btDeformableRigidContactConstraint
{
public:
    // the deformable node in contact
    const btSoftBody::Node* m_node;
    
    btDeformableNodeRigidContactConstraint(){}
    btDeformableNodeRigidContactConstraint(const btSoftBody::DeformableNodeRigidContact& contact);
    btDeformableNodeRigidContactConstraint(const btDeformableNodeRigidContactConstraint& other);
    
    virtual ~btDeformableNodeRigidContactConstraint()
    {
    }
    
    // get the velocity of the deformable node in contact
    virtual btVector3 getVb() const;
    
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
    const btSoftBody::Face* m_face;
    btDeformableFaceRigidContactConstraint(){}
    btDeformableFaceRigidContactConstraint(const btSoftBody::DeformableFaceRigidContact& contact);
    btDeformableFaceRigidContactConstraint(const btDeformableFaceRigidContactConstraint& other);
    
    virtual ~btDeformableFaceRigidContactConstraint()
    {
    }
    
    // get the velocity of the deformable face at the contact point
    virtual btVector3 getVb() const;
    
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
    
    btDeformableFaceNodeContactConstraint(){}
    
    btDeformableFaceNodeContactConstraint(const btSoftBody::DeformableFaceNodeContact& contact);
    
    virtual ~btDeformableFaceNodeContactConstraint(){}
    
    virtual btScalar solveConstraint();
    
    virtual btScalar solveSplitImpulse(const btContactSolverInfo& infoGlobal)
    {
        // todo: xuchenhan@
        return 0;
    }
    
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
    virtual void applySplitImpulse(const btVector3& impulse)
    {
        // todo xuchenhan@
    }
    virtual void setPenetrationScale(btScalar scale){}
};
#endif /* BT_DEFORMABLE_CONTACT_CONSTRAINT_H */
