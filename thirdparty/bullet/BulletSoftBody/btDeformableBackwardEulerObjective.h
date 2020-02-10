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

#ifndef BT_BACKWARD_EULER_OBJECTIVE_H
#define BT_BACKWARD_EULER_OBJECTIVE_H
#include "btConjugateGradient.h"
#include "btDeformableLagrangianForce.h"
#include "btDeformableMassSpringForce.h"
#include "btDeformableGravityForce.h"
#include "btDeformableCorotatedForce.h"
#include "btDeformableLinearElasticityForce.h"
#include "btDeformableNeoHookeanForce.h"
#include "btDeformableContactProjection.h"
#include "btPreconditioner.h"
#include "btDeformableMultiBodyDynamicsWorld.h"
#include "LinearMath/btQuickprof.h"

class btDeformableBackwardEulerObjective
{
public:
    typedef btAlignedObjectArray<btVector3> TVStack;
    btScalar m_dt;
    btAlignedObjectArray<btDeformableLagrangianForce*> m_lf;
    btAlignedObjectArray<btSoftBody *>& m_softBodies;
    Preconditioner* m_preconditioner;
    btDeformableContactProjection m_projection;
    const TVStack& m_backupVelocity;
    btAlignedObjectArray<btSoftBody::Node* > m_nodes;
    bool m_implicit;

    btDeformableBackwardEulerObjective(btAlignedObjectArray<btSoftBody *>& softBodies, const TVStack& backup_v);
    
    virtual ~btDeformableBackwardEulerObjective();
    
    void initialize(){}
    
    // compute the rhs for CG solve, i.e, add the dt scaled implicit force to residual
    void computeResidual(btScalar dt, TVStack& residual);
    
    // add explicit force to the velocity
    void applyExplicitForce(TVStack& force);
    
    // apply force to velocity and optionally reset the force to zero
    void applyForce(TVStack& force, bool setZero);
    
    // compute the norm of the residual
    btScalar computeNorm(const TVStack& residual) const;
    
    // compute one step of the solve (there is only one solve if the system is linear)
    void computeStep(TVStack& dv, const TVStack& residual, const btScalar& dt);
    
    // perform A*x = b
    void multiply(const TVStack& x, TVStack& b) const;
    
    // set initial guess for CG solve
    void initialGuess(TVStack& dv, const TVStack& residual);
    
    // reset data structure and reset dt
    void reinitialize(bool nodeUpdated, btScalar dt);
    
    void setDt(btScalar dt);
    
    // add friction force to residual
    void applyDynamicFriction(TVStack& r);
    
    // add dv to velocity
    void updateVelocity(const TVStack& dv);
    
    //set constraints as projections
    void setConstraints();
    
    // update the projections and project the residual
    void project(TVStack& r)
    {
        BT_PROFILE("project");
        m_projection.project(r);
    }
    
    // perform precondition M^(-1) x = b
    void precondition(const TVStack& x, TVStack& b)
    {
        m_preconditioner->operator()(x,b);
    }

    // reindex all the vertices 
    virtual void updateId()
    {
        size_t node_id = 0;
        size_t face_id = 0;
        m_nodes.clear();
        for (int i = 0; i < m_softBodies.size(); ++i)
        {
            btSoftBody* psb = m_softBodies[i];
            for (int j = 0; j < psb->m_nodes.size(); ++j)
            {
                psb->m_nodes[j].index = node_id;
                m_nodes.push_back(&psb->m_nodes[j]);
                ++node_id;
            }
            for (int j = 0; j < psb->m_faces.size(); ++j)
            {
                psb->m_faces[j].m_index = face_id;
                ++face_id;
            }
        }
    }
    
    const btAlignedObjectArray<btSoftBody::Node*>* getIndices() const
    {
        return &m_nodes;
    }
    
    void setImplicit(bool implicit)
    {
        m_implicit = implicit;
    }

    // Calculate the total potential energy in the system
    btScalar totalEnergy(btScalar dt);
};

#endif /* btBackwardEulerObjective_h */
