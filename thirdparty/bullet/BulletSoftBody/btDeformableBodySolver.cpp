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

#include <stdio.h>
#include <limits>
#include "btDeformableBodySolver.h"
#include "btSoftBodyInternals.h"
#include "LinearMath/btQuickprof.h"
static const int kMaxConjugateGradientIterations  = 50;
btDeformableBodySolver::btDeformableBodySolver()
: m_numNodes(0)
, m_cg(kMaxConjugateGradientIterations)
, m_maxNewtonIterations(5)
, m_newtonTolerance(1e-4)
, m_lineSearch(false)
{
    m_objective = new btDeformableBackwardEulerObjective(m_softBodies, m_backupVelocity);
}

btDeformableBodySolver::~btDeformableBodySolver()
{
    delete m_objective;
}

void btDeformableBodySolver::solveDeformableConstraints(btScalar solverdt)
{
    BT_PROFILE("solveDeformableConstraints");
    if (!m_implicit)
    {
        m_objective->computeResidual(solverdt, m_residual);
        m_objective->applyDynamicFriction(m_residual);
        computeStep(m_dv, m_residual);
        updateVelocity();
    }
    else
    {
        for (int i = 0; i < m_maxNewtonIterations; ++i)
        {
            updateState();
            // add the inertia term in the residual
            int counter = 0;
            for (int k = 0; k < m_softBodies.size(); ++k)
            {
                btSoftBody* psb = m_softBodies[k];
                for (int j = 0; j < psb->m_nodes.size(); ++j)
                {
                    if (psb->m_nodes[j].m_im > 0)
                    {
                        m_residual[counter] = (-1./psb->m_nodes[j].m_im) *  m_dv[counter];
                    }
                    ++counter;
                }
            }
            
            m_objective->computeResidual(solverdt, m_residual);
            if (m_objective->computeNorm(m_residual) < m_newtonTolerance && i > 0)
            {
                break;
            }
            // todo xuchenhan@: this really only needs to be calculated once
            m_objective->applyDynamicFriction(m_residual);
            if (m_lineSearch)
            {
                btScalar inner_product = computeDescentStep(m_ddv,m_residual);
                btScalar alpha = 0.01, beta = 0.5; // Boyd & Vandenberghe suggested alpha between 0.01 and 0.3, beta between 0.1 to 0.8
                btScalar scale = 2;
                btScalar f0 = m_objective->totalEnergy(solverdt)+kineticEnergy(), f1, f2;
                backupDv();
                do {
                    scale *= beta;
                    if (scale < 1e-8) {
                        return;
                    }
                    updateEnergy(scale);
                    f1 = m_objective->totalEnergy(solverdt)+kineticEnergy();
                    f2 = f0 - alpha * scale * inner_product;
                } while (!(f1 < f2+SIMD_EPSILON)); // if anything here is nan then the search continues
                revertDv();
                updateDv(scale);
            }
            else
            {
                computeStep(m_ddv, m_residual);
                updateDv();
            }
            for (int j = 0; j < m_numNodes; ++j)
            {
                m_ddv[j].setZero();
                m_residual[j].setZero();
            }
        }
        updateVelocity();
    }
}

btScalar btDeformableBodySolver::kineticEnergy()
{
    btScalar ke = 0;
    for (int i = 0; i < m_softBodies.size();++i)
    {
        btSoftBody* psb = m_softBodies[i];
        for (int j = 0; j < psb->m_nodes.size();++j)
        {
            btSoftBody::Node& node = psb->m_nodes[j];
            if (node.m_im > 0)
            {
                ke += m_dv[node.index].length2() * 0.5 / node.m_im;
            }
        }
    }
    return ke;
}

void btDeformableBodySolver::backupDv()
{
    m_backup_dv.resize(m_dv.size());
    for (int i = 0; i<m_backup_dv.size(); ++i)
    {
        m_backup_dv[i] = m_dv[i];
    }
}

void btDeformableBodySolver::revertDv()
{
    for (int i = 0; i<m_backup_dv.size(); ++i)
    {
        m_dv[i] = m_backup_dv[i];
    }
}

void btDeformableBodySolver::updateEnergy(btScalar scale)
{
    for (int i = 0; i<m_dv.size(); ++i)
    {
        m_dv[i] = m_backup_dv[i] + scale * m_ddv[i];
    }
    updateState();
}


btScalar btDeformableBodySolver::computeDescentStep(TVStack& ddv, const TVStack& residual, bool verbose)
{
    m_cg.solve(*m_objective, ddv, residual, false);
    btScalar inner_product = m_cg.dot(residual, m_ddv);
    btScalar res_norm = m_objective->computeNorm(residual);
    btScalar tol = 1e-5 * res_norm * m_objective->computeNorm(m_ddv);
    if (inner_product < -tol)
    {
        if (verbose)
        {
            std::cout << "Looking backwards!" << std::endl;
        }
        for (int i = 0; i < m_ddv.size();++i)
        {
            m_ddv[i] = -m_ddv[i];
        }
        inner_product = -inner_product;
    }
    else if (std::abs(inner_product) < tol)
    {
        if (verbose)
        {
            std::cout << "Gradient Descent!" << std::endl;
        }
        btScalar scale = m_objective->computeNorm(m_ddv) / res_norm;
        for (int i = 0; i < m_ddv.size();++i)
        {
            m_ddv[i] = scale * residual[i];
        }
        inner_product = scale * res_norm * res_norm;
    }
    return inner_product;
}

void btDeformableBodySolver::updateState()
{
    updateVelocity();
    updateTempPosition();
}

void btDeformableBodySolver::updateDv(btScalar scale)
{
    for (int i = 0; i < m_numNodes; ++i)
    {
        m_dv[i] += scale * m_ddv[i];
    }
}

void btDeformableBodySolver::computeStep(TVStack& ddv, const TVStack& residual)
{
    m_cg.solve(*m_objective, ddv, residual);
}

void btDeformableBodySolver::reinitialize(const btAlignedObjectArray<btSoftBody *>& softBodies, btScalar dt)
{
    m_softBodies.copyFromArray(softBodies);
    bool nodeUpdated = updateNodes();
    
    if (nodeUpdated)
    {
        m_dv.resize(m_numNodes, btVector3(0,0,0));
        m_ddv.resize(m_numNodes, btVector3(0,0,0));
        m_residual.resize(m_numNodes, btVector3(0,0,0));
        m_backupVelocity.resize(m_numNodes, btVector3(0,0,0));
    }
    
    // need to setZero here as resize only set value for newly allocated items
    for (int i = 0; i < m_numNodes; ++i)
    {
        m_dv[i].setZero();
        m_ddv[i].setZero();
        m_residual[i].setZero();
    }
    
    m_dt = dt;
    m_objective->reinitialize(nodeUpdated, dt);
}

void btDeformableBodySolver::setConstraints()
{
    BT_PROFILE("setConstraint");
    m_objective->setConstraints();
}

btScalar btDeformableBodySolver::solveContactConstraints(btCollisionObject** deformableBodies,int numDeformableBodies)
{
    BT_PROFILE("solveContactConstraints");
    btScalar maxSquaredResidual = m_objective->m_projection.update(deformableBodies,numDeformableBodies);
    return maxSquaredResidual;
}

btScalar btDeformableBodySolver::solveSplitImpulse(const btContactSolverInfo& infoGlobal)
{
    BT_PROFILE("solveSplitImpulse");
    return m_objective->m_projection.solveSplitImpulse(infoGlobal);
}

void btDeformableBodySolver::splitImpulseSetup(const btContactSolverInfo& infoGlobal)
{
     m_objective->m_projection.splitImpulseSetup(infoGlobal);
}

void btDeformableBodySolver::updateVelocity()
{
    int counter = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        psb->m_maxSpeedSquared = 0;
        if (!psb->isActive())
        {
            counter += psb->m_nodes.size();
            continue;
        }
        for (int j = 0; j < psb->m_nodes.size(); ++j)
        {
            // set NaN to zero;
            if (m_dv[counter] != m_dv[counter])
            {
                m_dv[counter].setZero();
            }
            psb->m_nodes[j].m_v = m_backupVelocity[counter]+m_dv[counter];
            psb->m_maxSpeedSquared = btMax(psb->m_maxSpeedSquared, psb->m_nodes[j].m_v.length2());
            ++counter;
        }
    }
}

void btDeformableBodySolver::updateTempPosition()
{
    int counter = 0;
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
            psb->m_nodes[j].m_q = psb->m_nodes[j].m_x + m_dt * psb->m_nodes[j].m_v;
            ++counter;
        }
        psb->updateDeformation();
    }
}

void btDeformableBodySolver::backupVelocity()
{
    int counter = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        for (int j = 0; j < psb->m_nodes.size(); ++j)
        {
            m_backupVelocity[counter++] = psb->m_nodes[j].m_v;
        }
    }
}

void btDeformableBodySolver::setupDeformableSolve(bool implicit)
{
    int counter = 0;
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
            if (implicit)
            {
                if ((psb->m_nodes[j].m_v - m_backupVelocity[counter]).norm() < SIMD_EPSILON)
                    m_dv[counter] = psb->m_nodes[j].m_v - m_backupVelocity[counter];
                else
                    m_dv[counter] = psb->m_nodes[j].m_v - psb->m_nodes[j].m_vn;
                m_backupVelocity[counter] = psb->m_nodes[j].m_vn;
            }
            else
                m_dv[counter] =  psb->m_nodes[j].m_v - m_backupVelocity[counter];
            psb->m_nodes[j].m_v = m_backupVelocity[counter] + psb->m_nodes[j].m_vsplit;
            ++counter;
        }
    }
}

void btDeformableBodySolver::revertVelocity()
{
    int counter = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        for (int j = 0; j < psb->m_nodes.size(); ++j)
        {
            psb->m_nodes[j].m_v = m_backupVelocity[counter++];
        }
    }
}

bool btDeformableBodySolver::updateNodes()
{
    int numNodes = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
        numNodes += m_softBodies[i]->m_nodes.size();
    if (numNodes != m_numNodes)
    {
        m_numNodes = numNodes;
        return true;
    }
    return false;
}


void btDeformableBodySolver::predictMotion(btScalar solverdt)
{
    // apply explicit forces to velocity
    m_objective->applyExplicitForce(m_residual);
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody *psb = m_softBodies[i];
        
        if (psb->isActive())
        {
            // predict motion for collision detection
            predictDeformableMotion(psb, solverdt);
        }
    }
}

void btDeformableBodySolver::predictDeformableMotion(btSoftBody* psb, btScalar dt)
{
    int i, ni;
    
    /* Update                */
    if (psb->m_bUpdateRtCst)
    {
        psb->m_bUpdateRtCst = false;
        psb->updateConstants();
        psb->m_fdbvt.clear();
        if (psb->m_cfg.collisions & btSoftBody::fCollision::SDF_RD)
        {
            psb->initializeFaceTree();
        }
    }
    
    /* Prepare                */
    psb->m_sst.sdt = dt * psb->m_cfg.timescale;
    psb->m_sst.isdt = 1 / psb->m_sst.sdt;
    psb->m_sst.velmrg = psb->m_sst.sdt * 3;
    psb->m_sst.radmrg = psb->getCollisionShape()->getMargin();
    psb->m_sst.updmrg = psb->m_sst.radmrg * (btScalar)0.25;
    /* Bounds                */
    psb->updateBounds();
    
    /* Integrate            */
    // do not allow particles to move more than the bounding box size
    btScalar max_v = (psb->m_bounds[1]-psb->m_bounds[0]).norm() / dt;
    for (i = 0, ni = psb->m_nodes.size(); i < ni; ++i)
    {
        btSoftBody::Node& n = psb->m_nodes[i];
        // apply drag
        n.m_v *= (1 - psb->m_cfg.drag);
        // scale velocity back
        if (n.m_v.norm() > max_v)
        {
            n.m_v.safeNormalize();
            n.m_v *= max_v;
        }
        n.m_q = n.m_x + n.m_v * dt;
    }

    /* Nodes                */
    ATTRIBUTE_ALIGNED16(btDbvtVolume)
    vol;
    for (i = 0, ni = psb->m_nodes.size(); i < ni; ++i)
    {
        btSoftBody::Node& n = psb->m_nodes[i];
        btVector3 points[2] = {n.m_x, n.m_q};
        vol = btDbvtVolume::FromPoints(points, 2);
        vol.Expand(btVector3(psb->m_sst.radmrg, psb->m_sst.radmrg, psb->m_sst.radmrg));
        psb->m_ndbvt.update(n.m_leaf, vol);
    }

    if (!psb->m_fdbvt.empty())
    {
        for (int i = 0; i < psb->m_faces.size(); ++i)
        {
            btSoftBody::Face& f = psb->m_faces[i];
            btVector3 points[6] = {f.m_n[0]->m_x, f.m_n[0]->m_q,
                                   f.m_n[1]->m_x, f.m_n[1]->m_q,
                                   f.m_n[2]->m_x, f.m_n[2]->m_q};
            vol = btDbvtVolume::FromPoints(points, 6);
            vol.Expand(btVector3(psb->m_sst.radmrg, psb->m_sst.radmrg, psb->m_sst.radmrg));
            psb->m_fdbvt.update(f.m_leaf, vol);
        }
    }
    /* Clear contacts        */
    psb->m_nodeRigidContacts.resize(0);
    psb->m_faceRigidContacts.resize(0);
    psb->m_faceNodeContacts.resize(0);
    /* Optimize dbvt's        */
    psb->m_ndbvt.optimizeIncremental(1);
    psb->m_fdbvt.optimizeIncremental(1);
}


void btDeformableBodySolver::updateSoftBodies()
{
    BT_PROFILE("updateSoftBodies");
    for (int i = 0; i < m_softBodies.size(); i++)
    {
        btSoftBody *psb = (btSoftBody *)m_softBodies[i];
        if (psb->isActive())
        {
            psb->updateNormals();
        }
    }
}

void btDeformableBodySolver::setImplicit(bool implicit)
{
    m_implicit = implicit;
    m_objective->setImplicit(implicit);
}

void btDeformableBodySolver::setLineSearch(bool lineSearch)
{
    m_lineSearch = lineSearch;
}
