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

#ifndef BT_MASS_SPRING_H
#define BT_MASS_SPRING_H

#include "btDeformableLagrangianForce.h"

class btDeformableMassSpringForce : public btDeformableLagrangianForce
{
    // If true, the damping force will be in the direction of the spring
    // If false, the damping force will be in the direction of the velocity
    bool m_momentum_conserving;
    btScalar m_elasticStiffness, m_dampingStiffness, m_bendingStiffness;
public:
    typedef btAlignedObjectArray<btVector3> TVStack;
    btDeformableMassSpringForce() : m_momentum_conserving(false), m_elasticStiffness(1), m_dampingStiffness(0.05)
    {
    }
    btDeformableMassSpringForce(btScalar k, btScalar d, bool conserve_angular = true, double bending_k = -1) : m_momentum_conserving(conserve_angular), m_elasticStiffness(k), m_dampingStiffness(d), m_bendingStiffness(bending_k)
    {
        if (m_bendingStiffness < btScalar(0))
        {
            m_bendingStiffness = m_elasticStiffness;
        }
    }
    
    virtual void addScaledForces(btScalar scale, TVStack& force)
    {
        addScaledDampingForce(scale, force);
        addScaledElasticForce(scale, force);
    }
    
    virtual void addScaledExplicitForce(btScalar scale, TVStack& force)
    {
        addScaledElasticForce(scale, force);
    }
    
    virtual void addScaledDampingForce(btScalar scale, TVStack& force)
    {
        int numNodes = getNumNodes();
        btAssert(numNodes <= force.size());
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            const btSoftBody* psb = m_softBodies[i];
            if (!psb->isActive())
            {
                continue;
            }
            for (int j = 0; j < psb->m_links.size(); ++j)
            {
                const btSoftBody::Link& link = psb->m_links[j];
                btSoftBody::Node* node1 = link.m_n[0];
                btSoftBody::Node* node2 = link.m_n[1];
                size_t id1 = node1->index;
                size_t id2 = node2->index;
                
                // damping force
                btVector3 v_diff = (node2->m_v - node1->m_v);
                btVector3 scaled_force = scale * m_dampingStiffness * v_diff;
                if (m_momentum_conserving)
                {
                    if ((node2->m_x - node1->m_x).norm() > SIMD_EPSILON)
                    {
                        btVector3 dir = (node2->m_x - node1->m_x).normalized();
                        scaled_force = scale * m_dampingStiffness * v_diff.dot(dir) * dir;
                    }
                }
                force[id1] += scaled_force;
                force[id2] -= scaled_force;
            }
        }
    }
    
    virtual void addScaledElasticForce(btScalar scale, TVStack& force)
    {
        int numNodes = getNumNodes();
        btAssert(numNodes <= force.size());
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            const btSoftBody* psb = m_softBodies[i];
            if (!psb->isActive())
            {
                continue;
            }
            for (int j = 0; j < psb->m_links.size(); ++j)
            {
                const btSoftBody::Link& link = psb->m_links[j];
                btSoftBody::Node* node1 = link.m_n[0];
                btSoftBody::Node* node2 = link.m_n[1];
                btScalar r = link.m_rl;
                size_t id1 = node1->index;
                size_t id2 = node2->index;
                
                // elastic force
                btVector3 dir = (node2->m_q - node1->m_q);
                btVector3 dir_normalized = (dir.norm() > SIMD_EPSILON) ? dir.normalized() : btVector3(0,0,0);
                btScalar scaled_stiffness = scale * (link.m_bbending ? m_bendingStiffness : m_elasticStiffness);
                btVector3 scaled_force = scaled_stiffness * (dir - dir_normalized * r);
                force[id1] += scaled_force;
                force[id2] -= scaled_force;
            }
        }
    }
    
    virtual void addScaledDampingForceDifferential(btScalar scale, const TVStack& dv, TVStack& df)
    {
        // implicit damping force differential
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            btSoftBody* psb = m_softBodies[i];
            if (!psb->isActive())
            {
                continue;
            }
            btScalar scaled_k_damp = m_dampingStiffness * scale;
            for (int j = 0; j < psb->m_links.size(); ++j)
            {
                const btSoftBody::Link& link = psb->m_links[j];
                btSoftBody::Node* node1 = link.m_n[0];
                btSoftBody::Node* node2 = link.m_n[1];
                size_t id1 = node1->index;
                size_t id2 = node2->index;

                btVector3 local_scaled_df = scaled_k_damp * (dv[id2] - dv[id1]);
                if (m_momentum_conserving)
                {
                    if ((node2->m_x - node1->m_x).norm() > SIMD_EPSILON)
                    {
                        btVector3 dir = (node2->m_x - node1->m_x).normalized();
                        local_scaled_df= scaled_k_damp * (dv[id2] - dv[id1]).dot(dir) * dir;
                    }
                }
                df[id1] += local_scaled_df;
                df[id2] -= local_scaled_df;
            }
        }
    }
    
    virtual void buildDampingForceDifferentialDiagonal(btScalar scale, TVStack& diagA)
    {
        // implicit damping force differential
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            btSoftBody* psb = m_softBodies[i];
            if (!psb->isActive())
            {
                continue;
            }
            btScalar scaled_k_damp = m_dampingStiffness * scale;
            for (int j = 0; j < psb->m_links.size(); ++j)
            {
                const btSoftBody::Link& link = psb->m_links[j];
                btSoftBody::Node* node1 = link.m_n[0];
                btSoftBody::Node* node2 = link.m_n[1];
                size_t id1 = node1->index;
                size_t id2 = node2->index;
                if (m_momentum_conserving)
                {
                    if ((node2->m_x - node1->m_x).norm() > SIMD_EPSILON)
                    {
                        btVector3 dir = (node2->m_x - node1->m_x).normalized();
                        for (int d = 0; d < 3; ++d)
                        {
                            if (node1->m_im > 0)
                                diagA[id1][d] -= scaled_k_damp * dir[d] * dir[d];
                            if (node2->m_im > 0)
                                diagA[id2][d] -= scaled_k_damp * dir[d] * dir[d];
                        }
                    }
                }
                else
                {
                    for (int d = 0; d < 3; ++d)
                    {
                        if (node1->m_im > 0)
                            diagA[id1][d] -= scaled_k_damp;
                        if (node2->m_im > 0)
                            diagA[id2][d] -= scaled_k_damp;
                    }
                }
            }
        }
    }
    
    virtual double totalElasticEnergy(btScalar dt)
    {
        double energy = 0;
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            const btSoftBody* psb = m_softBodies[i];
            if (!psb->isActive())
            {
                continue;
            }
            for (int j = 0; j < psb->m_links.size(); ++j)
            {
                const btSoftBody::Link& link = psb->m_links[j];
                btSoftBody::Node* node1 = link.m_n[0];
                btSoftBody::Node* node2 = link.m_n[1];
                btScalar r = link.m_rl;

                // elastic force
                btVector3 dir = (node2->m_q - node1->m_q);
                energy += 0.5 * m_elasticStiffness * (dir.norm() - r) * (dir.norm() -r);
            }
        }
        return energy;
    }
    
    virtual double totalDampingEnergy(btScalar dt)
    {
        double energy = 0;
        int sz = 0;
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            btSoftBody* psb = m_softBodies[i];
            if (!psb->isActive())
            {
                continue;
            }
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                sz = btMax(sz, psb->m_nodes[j].index);
            }
        }
        TVStack dampingForce;
        dampingForce.resize(sz+1);
        for (int i = 0; i < dampingForce.size(); ++i)
            dampingForce[i].setZero();
        addScaledDampingForce(0.5, dampingForce);
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                const btSoftBody::Node& node = psb->m_nodes[j];
                energy -= dampingForce[node.index].dot(node.m_v) / dt;
            }
        }
        return energy;
    }
    
    virtual void addScaledElasticForceDifferential(btScalar scale, const TVStack& dx, TVStack& df)
    {
        // implicit damping force differential
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            const btSoftBody* psb = m_softBodies[i];
            if (!psb->isActive())
            {
                continue;
            }
            for (int j = 0; j < psb->m_links.size(); ++j)
            {
                const btSoftBody::Link& link = psb->m_links[j];
                btSoftBody::Node* node1 = link.m_n[0];
                btSoftBody::Node* node2 = link.m_n[1];
                size_t id1 = node1->index;
                size_t id2 = node2->index;
                btScalar r = link.m_rl;

                btVector3 dir = (node1->m_q - node2->m_q);
                btScalar dir_norm = dir.norm();
                btVector3 dir_normalized = (dir_norm > SIMD_EPSILON) ? dir.normalized() : btVector3(0,0,0);
                btVector3 dx_diff = dx[id1] - dx[id2];
                btVector3 scaled_df = btVector3(0,0,0);
                btScalar scaled_k = scale * (link.m_bbending ? m_bendingStiffness : m_elasticStiffness);
                if (dir_norm > SIMD_EPSILON)
                {
                    scaled_df -= scaled_k * dir_normalized.dot(dx_diff) * dir_normalized;
                    scaled_df += scaled_k * dir_normalized.dot(dx_diff) * ((dir_norm-r)/dir_norm) * dir_normalized;
                    scaled_df -= scaled_k * ((dir_norm-r)/dir_norm) * dx_diff;
                }
                
                df[id1] += scaled_df;
                df[id2] -= scaled_df;
            }
        }
    }
    
    virtual btDeformableLagrangianForceType getForceType()
    {
        return BT_MASSSPRING_FORCE;
    }
    
};

#endif /* btMassSpring_h */
