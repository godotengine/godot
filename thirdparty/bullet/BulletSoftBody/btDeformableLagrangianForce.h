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

#ifndef BT_DEFORMABLE_LAGRANGIAN_FORCE_H
#define BT_DEFORMABLE_LAGRANGIAN_FORCE_H

#include "btSoftBody.h"
#include <LinearMath/btHashMap.h>
#include <iostream>

enum btDeformableLagrangianForceType
{
    BT_GRAVITY_FORCE = 1,
    BT_MASSSPRING_FORCE = 2,
    BT_COROTATED_FORCE = 3,
    BT_NEOHOOKEAN_FORCE = 4,
    BT_LINEAR_ELASTICITY_FORCE = 5
};

static inline double randomDouble(double low, double high)
{
    return low + static_cast<double>(rand()) / RAND_MAX * (high - low);
}

class btDeformableLagrangianForce
{
public:
    typedef btAlignedObjectArray<btVector3> TVStack;
    btAlignedObjectArray<btSoftBody *> m_softBodies;
    const btAlignedObjectArray<btSoftBody::Node*>* m_nodes;
    
    btDeformableLagrangianForce()
    {
    }
    
    virtual ~btDeformableLagrangianForce(){}
    
    // add all forces
    virtual void addScaledForces(btScalar scale, TVStack& force) = 0;
    
    // add damping df
    virtual void addScaledDampingForceDifferential(btScalar scale, const TVStack& dv, TVStack& df) = 0;
    
    // add elastic df
    virtual void addScaledElasticForceDifferential(btScalar scale, const TVStack& dx, TVStack& df) = 0;
    
    // add all forces that are explicit in explicit solve
    virtual void addScaledExplicitForce(btScalar scale, TVStack& force) = 0;
    
    // add all damping forces 
    virtual void addScaledDampingForce(btScalar scale, TVStack& force) = 0;
    
    virtual btDeformableLagrangianForceType getForceType() = 0;
    
    virtual void reinitialize(bool nodeUpdated)
    {
    }
    
    // get number of nodes that have the force
    virtual int getNumNodes()
    {
        int numNodes = 0;
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            numNodes += m_softBodies[i]->m_nodes.size();
        }
        return numNodes;
    }
    
    // add a soft body to be affected by the particular lagrangian force
    virtual void addSoftBody(btSoftBody* psb)
    {
        m_softBodies.push_back(psb);
    }
    
    virtual void setIndices(const btAlignedObjectArray<btSoftBody::Node*>* nodes)
    {
        m_nodes = nodes;
    }
    
     // Calculate the incremental deformable generated from the input dx
    virtual btMatrix3x3 Ds(int id0, int id1, int id2, int id3, const TVStack& dx)
    {
        btVector3 c1 = dx[id1] - dx[id0];
        btVector3 c2 = dx[id2] - dx[id0];
        btVector3 c3 = dx[id3] - dx[id0];
        return btMatrix3x3(c1,c2,c3).transpose();
    }
    
    // Calculate the incremental deformable generated from the current velocity
    virtual btMatrix3x3 DsFromVelocity(const btSoftBody::Node* n0, const btSoftBody::Node* n1, const btSoftBody::Node* n2, const btSoftBody::Node* n3)
    {
        btVector3 c1 = n1->m_v - n0->m_v;
        btVector3 c2 = n2->m_v - n0->m_v;
        btVector3 c3 = n3->m_v - n0->m_v;
        return btMatrix3x3(c1,c2,c3).transpose();
    }
    
    // test for addScaledElasticForce function
    virtual void testDerivative()
    {
        for (int i = 0; i<m_softBodies.size();++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                psb->m_nodes[j].m_q += btVector3(randomDouble(-.1, .1), randomDouble(-.1, .1), randomDouble(-.1, .1));
            }
            psb->updateDeformation();
        }
        
        TVStack dx;
        dx.resize(getNumNodes());
        TVStack dphi_dx;
        dphi_dx.resize(dx.size());
        for (int i =0; i < dphi_dx.size();++i)
        {
            dphi_dx[i].setZero();
        }
        addScaledForces(-1, dphi_dx);
        
        // write down the current position
        TVStack x;
        x.resize(dx.size());
        int counter = 0;
        for (int i = 0; i<m_softBodies.size();++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                x[counter] = psb->m_nodes[j].m_q;
                counter++;
            }
        }
        counter = 0;
        
        // populate dx with random vectors
        for (int i = 0; i < dx.size(); ++i)
        {
            dx[i].setX(randomDouble(-1, 1));
            dx[i].setY(randomDouble(-1, 1));
            dx[i].setZ(randomDouble(-1, 1));
        }
        
        btAlignedObjectArray<double> errors;
        for (int it = 0; it < 10; ++it)
        {
            for (int i = 0; i < dx.size(); ++i)
            {
                dx[i] *= 0.5;
            }
            
            // get dphi/dx * dx
            double dphi = 0;
            for (int i = 0; i < dx.size(); ++i)
            {
                dphi += dphi_dx[i].dot(dx[i]);
            }
            

            for (int i = 0; i<m_softBodies.size();++i)
            {
                btSoftBody* psb = m_softBodies[i];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                {
                    psb->m_nodes[j].m_q = x[counter] + dx[counter];
                    counter++;
                }
                psb->updateDeformation();
            }
            counter = 0;
            double f1 = totalElasticEnergy(0);
            
            for (int i = 0; i<m_softBodies.size();++i)
            {
                btSoftBody* psb = m_softBodies[i];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                {
                    psb->m_nodes[j].m_q = x[counter] - dx[counter];
                    counter++;
                }
                psb->updateDeformation();
            }
            counter = 0;
            
            double f2 = totalElasticEnergy(0);
            
            //restore m_q
            for (int i = 0; i<m_softBodies.size();++i)
            {
                btSoftBody* psb = m_softBodies[i];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                {
                    psb->m_nodes[j].m_q = x[counter];
                    counter++;
                }
                psb->updateDeformation();
            }
            counter = 0;
            double error = f1-f2-2*dphi;
            errors.push_back(error);
            std::cout << "Iteration = " << it <<", f1 = " << f1 << ", f2 = " << f2 << ", error = " << error << std::endl;
        }
        for (int i = 1; i < errors.size(); ++i)
        {
            std::cout << "Iteration = " << i << ", ratio = " << errors[i-1]/errors[i] << std::endl;
        }
    }
    
    // test for addScaledElasticForce function
    virtual void testHessian()
    {
        for (int i = 0; i<m_softBodies.size();++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                psb->m_nodes[j].m_q += btVector3(randomDouble(-.1, .1), randomDouble(-.1, .1), randomDouble(-.1, .1));
            }
            psb->updateDeformation();
        }
        
        
        TVStack dx;
        dx.resize(getNumNodes());
        TVStack df;
        df.resize(dx.size());
        TVStack f1;
        f1.resize(dx.size());
        TVStack f2;
        f2.resize(dx.size());
        
        
        // write down the current position
        TVStack x;
        x.resize(dx.size());
        int counter = 0;
        for (int i = 0; i<m_softBodies.size();++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                x[counter] = psb->m_nodes[j].m_q;
                counter++;
            }
        }
        counter = 0;
        
        // populate dx with random vectors
        for (int i = 0; i < dx.size(); ++i)
        {
            dx[i].setX(randomDouble(-1, 1));
            dx[i].setY(randomDouble(-1, 1));
            dx[i].setZ(randomDouble(-1, 1));
        }
        
        btAlignedObjectArray<double> errors;
        for (int it = 0; it < 10; ++it)
        {
            for (int i = 0; i < dx.size(); ++i)
            {
                dx[i] *= 0.5;
            }
            
            // get df
            for (int i =0; i < df.size();++i)
            {
                df[i].setZero();
                f1[i].setZero();
                f2[i].setZero();
            }

            //set df
            addScaledElasticForceDifferential(-1, dx, df);
            
            for (int i = 0; i<m_softBodies.size();++i)
            {
                btSoftBody* psb = m_softBodies[i];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                {
                    psb->m_nodes[j].m_q = x[counter] + dx[counter];
                    counter++;
                }
                psb->updateDeformation();
            }
            counter = 0;
            
            //set f1
            addScaledForces(-1, f1);
            
            for (int i = 0; i<m_softBodies.size();++i)
            {
                btSoftBody* psb = m_softBodies[i];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                {
                    psb->m_nodes[j].m_q = x[counter] - dx[counter];
                    counter++;
                }
                psb->updateDeformation();
            }
            counter = 0;
            
            //set f2
            addScaledForces(-1, f2);
            
            //restore m_q
            for (int i = 0; i<m_softBodies.size();++i)
            {
                btSoftBody* psb = m_softBodies[i];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                {
                    psb->m_nodes[j].m_q = x[counter];
                    counter++;
                }
                psb->updateDeformation();
            }
            counter = 0;
            double error = 0;
            for (int i = 0; i < df.size();++i)
            {
                btVector3 error_vector = f1[i]-f2[i]-2*df[i];
                error += error_vector.length2();
            }
            error = btSqrt(error);
            errors.push_back(error);
            std::cout << "Iteration = " << it << ", error = " << error << std::endl;
        }
        for (int i = 1; i < errors.size(); ++i)
        {
            std::cout << "Iteration = " << i << ", ratio = " << errors[i-1]/errors[i] << std::endl;
        }
    }
    
    //
    virtual double totalElasticEnergy(btScalar dt)
    {
        return 0;
    }
    
    //
    virtual double totalDampingEnergy(btScalar dt)
    {
        return 0;
    }
    
    // total Energy takes dt as input because certain energies depend on dt
    virtual double totalEnergy(btScalar dt)
    {
        return totalElasticEnergy(dt) + totalDampingEnergy(dt);
    }
};
#endif /* BT_DEFORMABLE_LAGRANGIAN_FORCE */
