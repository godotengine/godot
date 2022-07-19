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

#include "btDeformableBackwardEulerObjective.h"
#include "btPreconditioner.h"
#include "LinearMath/btQuickprof.h"

btDeformableBackwardEulerObjective::btDeformableBackwardEulerObjective(btAlignedObjectArray<btSoftBody *>& softBodies, const TVStack& backup_v)
: m_softBodies(softBodies)
, m_projection(softBodies)
, m_backupVelocity(backup_v)
, m_implicit(false)
{
    m_preconditioner = new MassPreconditioner(m_softBodies);
}

btDeformableBackwardEulerObjective::~btDeformableBackwardEulerObjective()
{
    delete m_preconditioner;
}

void btDeformableBackwardEulerObjective::reinitialize(bool nodeUpdated, btScalar dt)
{
    BT_PROFILE("reinitialize");
    if (dt > 0)
    {
        setDt(dt);
    }
    if(nodeUpdated)
    {
        updateId();
    }
    for (int i = 0; i < m_lf.size(); ++i)
    {
        m_lf[i]->reinitialize(nodeUpdated);
    }
    m_projection.reinitialize(nodeUpdated);
    m_preconditioner->reinitialize(nodeUpdated);
}

void btDeformableBackwardEulerObjective::setDt(btScalar dt)
{
    m_dt = dt;
}

void btDeformableBackwardEulerObjective::multiply(const TVStack& x, TVStack& b) const
{
    BT_PROFILE("multiply");
    // add in the mass term
    size_t counter = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        for (int j = 0; j < psb->m_nodes.size(); ++j)
        {
            const btSoftBody::Node& node = psb->m_nodes[j];
            b[counter] = (node.m_im == 0) ? btVector3(0,0,0) : x[counter] / node.m_im;
            ++counter;
        }
    }

    for (int i = 0; i < m_lf.size(); ++i)
    {
        // add damping matrix
        m_lf[i]->addScaledDampingForceDifferential(-m_dt, x, b);
        if (m_implicit)
        {
             m_lf[i]->addScaledElasticForceDifferential(-m_dt*m_dt, x, b);
        }
    }
}

void btDeformableBackwardEulerObjective::updateVelocity(const TVStack& dv)
{
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        for (int j = 0; j < psb->m_nodes.size(); ++j)
        {
            btSoftBody::Node& node = psb->m_nodes[j];
            node.m_v = m_backupVelocity[node.index] + dv[node.index];
        }
    }
}

void btDeformableBackwardEulerObjective::applyForce(TVStack& force, bool setZero)
{
    size_t counter = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        if (!psb->isActive())
        {
            counter += psb->m_nodes.size();
            continue;
        }
        for (int j = 0; j < psb->m_nodes.size(); ++j)
        {
            btScalar one_over_mass = (psb->m_nodes[j].m_im == 0) ? 0 : psb->m_nodes[j].m_im;
            psb->m_nodes[j].m_v += one_over_mass * force[counter++];
        }
    }
    if (setZero)
    {
        for (int i = 0; i < force.size(); ++i)
            force[i].setZero();
    }
}

void btDeformableBackwardEulerObjective::computeResidual(btScalar dt, TVStack &residual)
{
    BT_PROFILE("computeResidual");
    // add implicit force
    for (int i = 0; i < m_lf.size(); ++i)
    {
        if (m_implicit)
        {
            m_lf[i]->addScaledForces(dt, residual);
        }
        else
        {
            m_lf[i]->addScaledDampingForce(dt, residual);
        }
    }
    m_projection.project(residual);
}

btScalar btDeformableBackwardEulerObjective::computeNorm(const TVStack& residual) const
{
    btScalar mag = 0;
    for (int i = 0; i < residual.size(); ++i)
    {
        mag += residual[i].length2();
    }
    return std::sqrt(mag);
}

btScalar btDeformableBackwardEulerObjective::totalEnergy(btScalar dt)
{
    btScalar e = 0;
    for (int i = 0; i < m_lf.size(); ++i)
    {
        e += m_lf[i]->totalEnergy(dt);
    }
    return e;
}

void btDeformableBackwardEulerObjective::applyExplicitForce(TVStack& force)
{
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        m_softBodies[i]->advanceDeformation();
    }
    
    for (int i = 0; i < m_lf.size(); ++i)
    {
        m_lf[i]->addScaledExplicitForce(m_dt, force);
    }
    applyForce(force, true);
}

void btDeformableBackwardEulerObjective::initialGuess(TVStack& dv, const TVStack& residual)
{
    size_t counter = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        for (int j = 0; j < psb->m_nodes.size(); ++j)
        {
            dv[counter] = psb->m_nodes[j].m_im * residual[counter];
            ++counter;
        }
    }
}

//set constraints as projections
void btDeformableBackwardEulerObjective::setConstraints()
{
    m_projection.setConstraints();
}

void btDeformableBackwardEulerObjective::applyDynamicFriction(TVStack& r)
{
     m_projection.applyDynamicFriction(r);
}
