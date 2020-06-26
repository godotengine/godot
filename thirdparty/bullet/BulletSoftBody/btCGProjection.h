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

#ifndef BT_CG_PROJECTION_H
#define BT_CG_PROJECTION_H

#include "btSoftBody.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraint.h"

struct DeformableContactConstraint
{
    const btSoftBody::Node* m_node;
    btAlignedObjectArray<const btSoftBody::RContact*> m_contact;
    btAlignedObjectArray<btVector3> m_total_normal_dv;
    btAlignedObjectArray<btVector3> m_total_tangent_dv;
    btAlignedObjectArray<bool> m_static;
    btAlignedObjectArray<bool> m_can_be_dynamic;
    
    DeformableContactConstraint(const btSoftBody::RContact& rcontact): m_node(rcontact.m_node)
    {
        append(rcontact);
    }
    
    DeformableContactConstraint(): m_node(NULL)
    {
        m_contact.push_back(NULL);
    }
    
    void append(const btSoftBody::RContact& rcontact)
    {
        m_contact.push_back(&rcontact);
        m_total_normal_dv.push_back(btVector3(0,0,0));
        m_total_tangent_dv.push_back(btVector3(0,0,0));
        m_static.push_back(false);
        m_can_be_dynamic.push_back(true);
    }

    void replace(const btSoftBody::RContact& rcontact)
    {
        m_contact.clear();
        m_total_normal_dv.clear();
        m_total_tangent_dv.clear();
        m_static.clear();
        m_can_be_dynamic.clear();
        append(rcontact);
    }
    
    ~DeformableContactConstraint()
    {
    }
};

class btCGProjection
{
public:
    typedef btAlignedObjectArray<btVector3> TVStack;
    typedef btAlignedObjectArray<btAlignedObjectArray<btVector3> > TVArrayStack;
    typedef btAlignedObjectArray<btAlignedObjectArray<btScalar> > TArrayStack;
    btAlignedObjectArray<btSoftBody *>& m_softBodies;
    const btScalar& m_dt;
    // map from node indices to node pointers
    const btAlignedObjectArray<btSoftBody::Node*>* m_nodes;
    
    btCGProjection(btAlignedObjectArray<btSoftBody *>& softBodies, const btScalar& dt)
    : m_softBodies(softBodies)
    , m_dt(dt)
    {
    }
    
    virtual ~btCGProjection()
    {
    }
    
    // apply the constraints
    virtual void project(TVStack& x) = 0;
    
    virtual void setConstraints() = 0;
    
    // update the constraints
    virtual btScalar update() = 0;
    
    virtual void reinitialize(bool nodeUpdated)
    {
    }
    
    virtual void setIndices(const btAlignedObjectArray<btSoftBody::Node*>* nodes)
    {
        m_nodes = nodes;
    }
};


#endif /* btCGProjection_h */
