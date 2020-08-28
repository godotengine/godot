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

#include "btDeformableContactProjection.h"
#include "btDeformableMultiBodyDynamicsWorld.h"
#include <algorithm>
#include <cmath>
btScalar btDeformableContactProjection::update(btCollisionObject** deformableBodies,int numDeformableBodies, const btContactSolverInfo& infoGlobal)
{
	btScalar residualSquare = 0;
	for (int i = 0; i < numDeformableBodies; ++i)
	{
		for (int j = 0; j < m_softBodies.size(); ++j)
		{
			btCollisionObject* psb = m_softBodies[j];
			if (psb != deformableBodies[i])
			{
				continue;
			}
			for (int k = 0; k < m_nodeRigidConstraints[j].size(); ++k)
			{
				btDeformableNodeRigidContactConstraint& constraint = m_nodeRigidConstraints[j][k];
				btScalar localResidualSquare = constraint.solveConstraint(infoGlobal);
				residualSquare = btMax(residualSquare, localResidualSquare);
			}
			for (int k = 0; k < m_nodeAnchorConstraints[j].size(); ++k)
			{
				btDeformableNodeAnchorConstraint& constraint = m_nodeAnchorConstraints[j][k];
				btScalar localResidualSquare = constraint.solveConstraint(infoGlobal);
				residualSquare = btMax(residualSquare, localResidualSquare);
			}
			for (int k = 0; k < m_faceRigidConstraints[j].size(); ++k)
			{
				btDeformableFaceRigidContactConstraint& constraint = m_faceRigidConstraints[j][k];
				btScalar localResidualSquare = constraint.solveConstraint(infoGlobal);
				residualSquare = btMax(residualSquare, localResidualSquare);
			}
			for (int k = 0; k < m_deformableConstraints[j].size(); ++k)
			{
				btDeformableFaceNodeContactConstraint& constraint = m_deformableConstraints[j][k];
				btScalar localResidualSquare = constraint.solveConstraint(infoGlobal);
				residualSquare = btMax(residualSquare, localResidualSquare);
			}
		}
	}
	return residualSquare;
}

void btDeformableContactProjection::splitImpulseSetup(const btContactSolverInfo& infoGlobal)
{
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		// node constraints
		for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
		{
			btDeformableNodeRigidContactConstraint& constraint = m_nodeRigidConstraints[i][j];
			constraint.setPenetrationScale(infoGlobal.m_deformable_erp);
		}
		// face constraints
		for (int j = 0; j < m_faceRigidConstraints[i].size(); ++j)
		{
			btDeformableFaceRigidContactConstraint& constraint = m_faceRigidConstraints[i][j];
			constraint.setPenetrationScale(infoGlobal.m_deformable_erp);
		}
	}
}

void btDeformableContactProjection::setConstraints(const btContactSolverInfo& infoGlobal)
{  
	BT_PROFILE("setConstraints");
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		btSoftBody* psb = m_softBodies[i];
		if (!psb->isActive())
		{
			continue;
		}

		// set Dirichlet constraint
		for (int j = 0; j < psb->m_nodes.size(); ++j)
		{
			if (psb->m_nodes[j].m_im == 0)
			{
				btDeformableStaticConstraint static_constraint(&psb->m_nodes[j], infoGlobal);
				m_staticConstraints[i].push_back(static_constraint);
			}
		}
		
		// set up deformable anchors
		for (int j = 0; j < psb->m_deformableAnchors.size(); ++j)
		{
			btSoftBody::DeformableNodeRigidAnchor& anchor = psb->m_deformableAnchors[j];
			// skip fixed points
			if (anchor.m_node->m_im == 0)
			{
				continue;
			}
			anchor.m_c1 = anchor.m_cti.m_colObj->getWorldTransform().getBasis() * anchor.m_local;
			btDeformableNodeAnchorConstraint constraint(anchor, infoGlobal);
			m_nodeAnchorConstraints[i].push_back(constraint);
		}
		
		// set Deformable Node vs. Rigid constraint
		for (int j = 0; j < psb->m_nodeRigidContacts.size(); ++j)
		{
			const btSoftBody::DeformableNodeRigidContact& contact = psb->m_nodeRigidContacts[j];
			// skip fixed points
			if (contact.m_node->m_im == 0)
			{
				continue;
			}
			btDeformableNodeRigidContactConstraint constraint(contact, infoGlobal);
			btVector3 va = constraint.getVa();
			btVector3 vb = constraint.getVb();
			const btVector3 vr = vb - va;
			const btSoftBody::sCti& cti = contact.m_cti;
			const btScalar dn = btDot(vr, cti.m_normal);
			if (dn < SIMD_EPSILON)
			{
				m_nodeRigidConstraints[i].push_back(constraint);
			}
		}
		
		// set Deformable Face vs. Rigid constraint
		for (int j = 0; j < psb->m_faceRigidContacts.size(); ++j)
		{
			const btSoftBody::DeformableFaceRigidContact& contact = psb->m_faceRigidContacts[j];
			// skip fixed faces
			if (contact.m_c2 == 0)
			{
				continue;
			}
			btDeformableFaceRigidContactConstraint constraint(contact, infoGlobal, m_useStrainLimiting);
			btVector3 va = constraint.getVa();
			btVector3 vb = constraint.getVb();
			const btVector3 vr = vb - va;
			const btSoftBody::sCti& cti = contact.m_cti;
			const btScalar dn = btDot(vr, cti.m_normal);
			if (dn < SIMD_EPSILON)
			{
				m_faceRigidConstraints[i].push_back(constraint);
			}
		}
	}
}

void btDeformableContactProjection::project(TVStack& x)
{
#ifndef USE_MGS
    const int dim = 3;
    for (int index = 0; index < m_projectionsDict.size(); ++index)
    {
        btAlignedObjectArray<btVector3>& projectionDirs = *m_projectionsDict.getAtIndex(index);
        size_t i = m_projectionsDict.getKeyAtIndex(index).getUid1();
        if (projectionDirs.size() >= dim)
        {
            // static node
            x[i].setZero();
            continue;
        }
        else if (projectionDirs.size() == 2)
        {
            btVector3 dir0 = projectionDirs[0];
            btVector3 dir1 = projectionDirs[1];
            btVector3 free_dir = btCross(dir0, dir1);
            if (free_dir.safeNorm() < SIMD_EPSILON)
            {
                x[i] -= x[i].dot(dir0) * dir0;
                x[i] -= x[i].dot(dir1) * dir1;
            }
            else
            {
                free_dir.normalize();
                x[i] = x[i].dot(free_dir) * free_dir;
            }
        }
        else
        {
            btAssert(projectionDirs.size() == 1);
            btVector3 dir0 = projectionDirs[0];
            x[i] -= x[i].dot(dir0) * dir0;
        }
    }
#else
    btReducedVector p(x.size());
    for (int i = 0; i < m_projections.size(); ++i)
    {
        p += (m_projections[i].dot(x) * m_projections[i]);
    }
    for (int i = 0; i < p.m_indices.size(); ++i)
    {
        x[p.m_indices[i]] -= p.m_vecs[i];
    }
#endif
}

void btDeformableContactProjection::setProjection()
{
#ifndef USE_MGS
    BT_PROFILE("btDeformableContactProjection::setProjection");
    btAlignedObjectArray<btVector3> units;
    units.push_back(btVector3(1,0,0));
    units.push_back(btVector3(0,1,0));
    units.push_back(btVector3(0,0,1));
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        if (!psb->isActive())
        {
            continue;
        }
        for (int j = 0; j < m_staticConstraints[i].size(); ++j)
        {
            int index = m_staticConstraints[i][j].m_node->index;
            m_staticConstraints[i][j].m_node->m_penetration = SIMD_INFINITY;
            if (m_projectionsDict.find(index) == NULL)
            {
                m_projectionsDict.insert(index, units);
            }
            else
            {
                btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
                for (int k = 0; k < 3; ++k)
                {
                    projections.push_back(units[k]);
                }
            }
        }
        for (int j = 0; j < m_nodeAnchorConstraints[i].size(); ++j)
        {
            int index = m_nodeAnchorConstraints[i][j].m_anchor->m_node->index;
            m_nodeAnchorConstraints[i][j].m_anchor->m_node->m_penetration = SIMD_INFINITY;
            if (m_projectionsDict.find(index) == NULL)
            {
                m_projectionsDict.insert(index, units);
            }
            else
            {
                btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
                for (int k = 0; k < 3; ++k)
                {
                    projections.push_back(units[k]);
                }
            }
        }
        for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
        {
            int index = m_nodeRigidConstraints[i][j].m_node->index;
            m_nodeRigidConstraints[i][j].m_node->m_penetration = -m_nodeRigidConstraints[i][j].getContact()->m_cti.m_offset;
            if (m_nodeRigidConstraints[i][j].m_static)
            {
                if (m_projectionsDict.find(index) == NULL)
                {
                    m_projectionsDict.insert(index, units);
                }
                else
                {
                    btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
                    for (int k = 0; k < 3; ++k)
                    {
                        projections.push_back(units[k]);
                    }
                }
            }
            else
            {
                if (m_projectionsDict.find(index) == NULL)
                {
                    btAlignedObjectArray<btVector3> projections;
                    projections.push_back(m_nodeRigidConstraints[i][j].m_normal);
                    m_projectionsDict.insert(index, projections);
                }
                else
                {
                    btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
                    projections.push_back(m_nodeRigidConstraints[i][j].m_normal);
                }
            }
        }
        for (int j = 0; j < m_faceRigidConstraints[i].size(); ++j)
        {
            const btSoftBody::Face* face = m_faceRigidConstraints[i][j].m_face;
            btScalar penetration = -m_faceRigidConstraints[i][j].getContact()->m_cti.m_offset;
            for (int k = 0; k < 3; ++k)
            {
                face->m_n[k]->m_penetration = btMax(face->m_n[k]->m_penetration, penetration);
            }
            for (int k = 0; k < 3; ++k)
            {
                btSoftBody::Node* node = face->m_n[k];
                node->m_penetration = true;
                int index = node->index;
                if (m_faceRigidConstraints[i][j].m_static)
                {
                    if (m_projectionsDict.find(index) == NULL)
                    {
                        m_projectionsDict.insert(index, units);
                    }
                    else
                    {
                        btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
                        for (int k = 0; k < 3; ++k)
                        {
                            projections.push_back(units[k]);
                        }
                    }
                }
                else
                {
                    if (m_projectionsDict.find(index) == NULL)
                    {
                        btAlignedObjectArray<btVector3> projections;
                        projections.push_back(m_faceRigidConstraints[i][j].m_normal);
                        m_projectionsDict.insert(index, projections);
                    }
                    else
                    {
                        btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
                        projections.push_back(m_faceRigidConstraints[i][j].m_normal);
                    }
                }
            }
        }
    }
#else
    int dof = 0;
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        dof += m_softBodies[i]->m_nodes.size();
    }
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        if (!psb->isActive())
        {
            continue;
        }
        for (int j = 0; j < m_staticConstraints[i].size(); ++j)
        {
            int index = m_staticConstraints[i][j].m_node->index;
            m_staticConstraints[i][j].m_node->m_penetration = SIMD_INFINITY;
            btAlignedObjectArray<int> indices;
            btAlignedObjectArray<btVector3> vecs1,vecs2,vecs3;
            indices.push_back(index);
            vecs1.push_back(btVector3(1,0,0));
            vecs2.push_back(btVector3(0,1,0));
            vecs3.push_back(btVector3(0,0,1));
            m_projections.push_back(btReducedVector(dof, indices, vecs1));
            m_projections.push_back(btReducedVector(dof, indices, vecs2));
            m_projections.push_back(btReducedVector(dof, indices, vecs3));
        }
        
        for (int j = 0; j < m_nodeAnchorConstraints[i].size(); ++j)
        {
            int index = m_nodeAnchorConstraints[i][j].m_anchor->m_node->index;
            m_nodeAnchorConstraints[i][j].m_anchor->m_node->m_penetration = SIMD_INFINITY;
            btAlignedObjectArray<int> indices;
            btAlignedObjectArray<btVector3> vecs1,vecs2,vecs3;
            indices.push_back(index);
            vecs1.push_back(btVector3(1,0,0));
            vecs2.push_back(btVector3(0,1,0));
            vecs3.push_back(btVector3(0,0,1));
            m_projections.push_back(btReducedVector(dof, indices, vecs1));
            m_projections.push_back(btReducedVector(dof, indices, vecs2));
            m_projections.push_back(btReducedVector(dof, indices, vecs3));
        }
        for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
        {
            int index = m_nodeRigidConstraints[i][j].m_node->index;
            m_nodeRigidConstraints[i][j].m_node->m_penetration = -m_nodeRigidConstraints[i][j].getContact()->m_cti.m_offset;
            btAlignedObjectArray<int> indices;
            indices.push_back(index);
            btAlignedObjectArray<btVector3> vecs1,vecs2,vecs3;
            if (m_nodeRigidConstraints[i][j].m_static)
            {
                vecs1.push_back(btVector3(1,0,0));
                vecs2.push_back(btVector3(0,1,0));
                vecs3.push_back(btVector3(0,0,1));
                m_projections.push_back(btReducedVector(dof, indices, vecs1));
                m_projections.push_back(btReducedVector(dof, indices, vecs2));
                m_projections.push_back(btReducedVector(dof, indices, vecs3));
            }
            else
            {
                vecs1.push_back(m_nodeRigidConstraints[i][j].m_normal);
                m_projections.push_back(btReducedVector(dof, indices, vecs1));
            }
        }
        for (int j = 0; j < m_faceRigidConstraints[i].size(); ++j)
        {
            const btSoftBody::Face* face = m_faceRigidConstraints[i][j].m_face;
			btVector3 bary = m_faceRigidConstraints[i][j].getContact()->m_bary;
            btScalar penetration = -m_faceRigidConstraints[i][j].getContact()->m_cti.m_offset;
            for (int k = 0; k < 3; ++k)
            {
                face->m_n[k]->m_penetration = btMax(face->m_n[k]->m_penetration, penetration);
            }
			if (m_faceRigidConstraints[i][j].m_static)
			{
				for (int l = 0; l < 3; ++l)
				{
					
					btReducedVector rv(dof);
					for (int k = 0; k < 3; ++k)
					{
						rv.m_indices.push_back(face->m_n[k]->index);
						btVector3 v(0,0,0);
						v[l] = bary[k];
						rv.m_vecs.push_back(v);
                        rv.sort();
					}
					m_projections.push_back(rv);
				}
			}
			else
			{
				btReducedVector rv(dof);
				for (int k = 0; k < 3; ++k)
				{
					rv.m_indices.push_back(face->m_n[k]->index);
					rv.m_vecs.push_back(bary[k] * m_faceRigidConstraints[i][j].m_normal);
                    rv.sort();
				}
				m_projections.push_back(rv);
			}
		}
    }
    btModifiedGramSchmidt<btReducedVector> mgs(m_projections);
    mgs.solve();
    m_projections = mgs.m_out;
#endif
}

void btDeformableContactProjection::checkConstraints(const TVStack& x)
{
    for (int i = 0; i < m_lagrangeMultipliers.size(); ++i)
    {
        btVector3 d(0,0,0);
        const LagrangeMultiplier& lm = m_lagrangeMultipliers[i];
        for (int j = 0; j < lm.m_num_constraints; ++j)
        {
            for (int k = 0; k < lm.m_num_nodes; ++k)
            {
                d[j] += lm.m_weights[k] * x[lm.m_indices[k]].dot(lm.m_dirs[j]);
            }
        }
        printf("d = %f, %f, %f\n",d[0],d[1],d[2]);
    }
}

void btDeformableContactProjection::setLagrangeMultiplier()
{
    for (int i = 0; i < m_softBodies.size(); ++i)
    {
        btSoftBody* psb = m_softBodies[i];
        if (!psb->isActive())
        {
            continue;
        }
        for (int j = 0; j < m_staticConstraints[i].size(); ++j)
        {
            int index = m_staticConstraints[i][j].m_node->index;
            m_staticConstraints[i][j].m_node->m_penetration = SIMD_INFINITY;
            LagrangeMultiplier lm;
            lm.m_num_nodes = 1;
            lm.m_indices[0] = index;
            lm.m_weights[0] = 1.0;
            lm.m_num_constraints = 3;
            lm.m_dirs[0] = btVector3(1,0,0);
            lm.m_dirs[1] = btVector3(0,1,0);
            lm.m_dirs[2] = btVector3(0,0,1);
            m_lagrangeMultipliers.push_back(lm);
        }
        for (int j = 0; j < m_nodeAnchorConstraints[i].size(); ++j)
        {
            int index = m_nodeAnchorConstraints[i][j].m_anchor->m_node->index;
            m_nodeAnchorConstraints[i][j].m_anchor->m_node->m_penetration = SIMD_INFINITY;
            LagrangeMultiplier lm;
            lm.m_num_nodes = 1;
            lm.m_indices[0] = index;
            lm.m_weights[0] = 1.0;
            lm.m_num_constraints = 3;
            lm.m_dirs[0] = btVector3(1,0,0);
            lm.m_dirs[1] = btVector3(0,1,0);
            lm.m_dirs[2] = btVector3(0,0,1);
            m_lagrangeMultipliers.push_back(lm);
        }
        for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
        {
            int index = m_nodeRigidConstraints[i][j].m_node->index;
            m_nodeRigidConstraints[i][j].m_node->m_penetration = -m_nodeRigidConstraints[i][j].getContact()->m_cti.m_offset;
            LagrangeMultiplier lm;
            lm.m_num_nodes = 1;
            lm.m_indices[0] = index;
            lm.m_weights[0] = 1.0;
            if (m_nodeRigidConstraints[i][j].m_static)
            {
                lm.m_num_constraints = 3;
                lm.m_dirs[0] = btVector3(1,0,0);
                lm.m_dirs[1] = btVector3(0,1,0);
                lm.m_dirs[2] = btVector3(0,0,1);
            }
            else
            {
                lm.m_num_constraints = 1;
                lm.m_dirs[0] = m_nodeRigidConstraints[i][j].m_normal;
            }
            m_lagrangeMultipliers.push_back(lm);
        }
        for (int j = 0; j < m_faceRigidConstraints[i].size(); ++j)
        {
            const btSoftBody::Face* face = m_faceRigidConstraints[i][j].m_face;
			
            btVector3 bary = m_faceRigidConstraints[i][j].getContact()->m_bary;
            btScalar penetration = -m_faceRigidConstraints[i][j].getContact()->m_cti.m_offset;
			LagrangeMultiplier lm;
			lm.m_num_nodes = 3;
			for (int k = 0; k<3; ++k)
			{
				face->m_n[k]->m_penetration = btMax(face->m_n[k]->m_penetration, penetration);
				lm.m_indices[k] = face->m_n[k]->index;
				lm.m_weights[k] = bary[k];
			}
            if (m_faceRigidConstraints[i][j].m_static)
            {
				lm.m_num_constraints = 3;
				lm.m_dirs[0] = btVector3(1,0,0);
				lm.m_dirs[1] = btVector3(0,1,0);
				lm.m_dirs[2] = btVector3(0,0,1);
			}
			else
			{
				lm.m_num_constraints = 1;
				lm.m_dirs[0] = m_faceRigidConstraints[i][j].m_normal;
			}
            m_lagrangeMultipliers.push_back(lm);
		}
	}
}

//
void btDeformableContactProjection::applyDynamicFriction(TVStack& f)
{
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
		{
			const btDeformableNodeRigidContactConstraint& constraint = m_nodeRigidConstraints[i][j];
			const btSoftBody::Node* node = constraint.m_node;
			if (node->m_im != 0)
			{
				int index = node->index;
				f[index] += constraint.getDv(node)* (1./node->m_im);
			}
		}
		for (int j = 0; j < m_faceRigidConstraints[i].size(); ++j)
		{
			const btDeformableFaceRigidContactConstraint& constraint = m_faceRigidConstraints[i][j];
			const btSoftBody::Face* face = constraint.getContact()->m_face;
			for (int k = 0; k < 3; ++k)
			{
				const btSoftBody::Node* node = face->m_n[k];
				if (node->m_im != 0)
				{
					int index = node->index;
					f[index] += constraint.getDv(node)* (1./node->m_im);
				}
			}
		}
		for (int j = 0; j < m_deformableConstraints[i].size(); ++j)
		{
			const btDeformableFaceNodeContactConstraint& constraint = m_deformableConstraints[i][j];
			const btSoftBody::Face* face = constraint.getContact()->m_face;
			const btSoftBody::Node* node = constraint.getContact()->m_node;
			if (node->m_im != 0)
			{
				int index = node->index;
				f[index] += constraint.getDv(node)* (1./node->m_im);
			}
			for (int k = 0; k < 3; ++k)
			{
				const btSoftBody::Node* node = face->m_n[k];
				if (node->m_im != 0)
				{
					int index = node->index;
					f[index] += constraint.getDv(node)* (1./node->m_im);
				}
			}
		}
	}
}

void btDeformableContactProjection::reinitialize(bool nodeUpdated)
{
	int N = m_softBodies.size();
	if (nodeUpdated)
	{
		m_staticConstraints.resize(N);
		m_nodeAnchorConstraints.resize(N);
		m_nodeRigidConstraints.resize(N);
		m_faceRigidConstraints.resize(N);
		m_deformableConstraints.resize(N);
		
	}
	for (int i = 0 ; i < N; ++i)
	{
		m_staticConstraints[i].clear();
		m_nodeAnchorConstraints[i].clear();
		m_nodeRigidConstraints[i].clear();
		m_faceRigidConstraints[i].clear();
		m_deformableConstraints[i].clear();
	}
#ifndef USE_MGS
    m_projectionsDict.clear();
#else
    m_projections.clear();
#endif
    m_lagrangeMultipliers.clear();
}



