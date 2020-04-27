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

#ifndef BT_COROTATED_H
#define BT_COROTATED_H

#include "btDeformableLagrangianForce.h"
#include "LinearMath/btPolarDecomposition.h"

static inline int PolarDecomposition(const btMatrix3x3& m, btMatrix3x3& q, btMatrix3x3& s)
{
    static const btPolarDecomposition polar;
    return polar.decompose(m, q, s);
}

class btDeformableCorotatedForce : public btDeformableLagrangianForce
{
public:
    typedef btAlignedObjectArray<btVector3> TVStack;
    btScalar m_mu, m_lambda;
    btDeformableCorotatedForce(): m_mu(1), m_lambda(1)
    {
        
    }
    
    btDeformableCorotatedForce(btScalar mu, btScalar lambda): m_mu(mu), m_lambda(lambda)
    {
    }
    
    virtual void addScaledForces(btScalar scale, TVStack& force)
    {
        addScaledElasticForce(scale, force);
    }
    
    virtual void addScaledExplicitForce(btScalar scale, TVStack& force)
    {
        addScaledElasticForce(scale, force);
    }
    
    virtual void addScaledDampingForce(btScalar scale, TVStack& force)
    {
    }
    
    virtual void addScaledElasticForce(btScalar scale, TVStack& force)
    {
        int numNodes = getNumNodes();
        btAssert(numNodes <= force.size());
        btVector3 grad_N_hat_1st_col = btVector3(-1,-1,-1);
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_tetras.size(); ++j)
            {
                btSoftBody::Tetra& tetra = psb->m_tetras[j];
                btMatrix3x3 P;
                firstPiola(tetra.m_F,P);
                btVector3 force_on_node0 = P * (tetra.m_Dm_inverse.transpose()*grad_N_hat_1st_col);
                btMatrix3x3 force_on_node123 = P * tetra.m_Dm_inverse.transpose();
                
                btSoftBody::Node* node0 = tetra.m_n[0];
                btSoftBody::Node* node1 = tetra.m_n[1];
                btSoftBody::Node* node2 = tetra.m_n[2];
                btSoftBody::Node* node3 = tetra.m_n[3];
                size_t id0 = node0->index;
                size_t id1 = node1->index;
                size_t id2 = node2->index;
                size_t id3 = node3->index;
                
                // elastic force
                // explicit elastic force
                btScalar scale1 = scale * tetra.m_element_measure;
                force[id0] -= scale1 * force_on_node0;
                force[id1] -= scale1 * force_on_node123.getColumn(0);
                force[id2] -= scale1 * force_on_node123.getColumn(1);
                force[id3] -= scale1 * force_on_node123.getColumn(2);
            }
        }
    }
    
    void firstPiola(const btMatrix3x3& F, btMatrix3x3& P)
    {
        // btMatrix3x3 JFinvT = F.adjoint();
        btScalar J = F.determinant();
        P =  F.adjoint().transpose() * (m_lambda * (J-1));
        if (m_mu > SIMD_EPSILON)
        {
            btMatrix3x3 R,S;
            if (J < 1024 * SIMD_EPSILON)
                R.setIdentity();
            else
                PolarDecomposition(F, R, S); // this QR is not robust, consider using implicit shift svd
            /*https://fuchuyuan.github.io/research/svd/paper.pdf*/
            P += (F-R) * 2 * m_mu;
        }
    }
    
    virtual void addScaledElasticForceDifferential(btScalar scale, const TVStack& dx, TVStack& df)
    {
    }
    
    virtual void addScaledDampingForceDifferential(btScalar scale, const TVStack& dv, TVStack& df)
    {
    }
    
    virtual void buildDampingForceDifferentialDiagonal(btScalar scale, TVStack& diagA){}
    
    virtual btDeformableLagrangianForceType getForceType()
    {
        return BT_COROTATED_FORCE;
    }
    
};


#endif /* btCorotated_h */
