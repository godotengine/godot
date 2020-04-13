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
btScalar btDeformableContactProjection::update(btCollisionObject** deformableBodies,int numDeformableBodies)
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
				btScalar localResidualSquare = constraint.solveConstraint();
				residualSquare = btMax(residualSquare, localResidualSquare);
			}
			for (int k = 0; k < m_nodeAnchorConstraints[j].size(); ++k)
			{
				btDeformableNodeAnchorConstraint& constraint = m_nodeAnchorConstraints[j][k];
				btScalar localResidualSquare = constraint.solveConstraint();
				residualSquare = btMax(residualSquare, localResidualSquare);
			}
			for (int k = 0; k < m_faceRigidConstraints[j].size(); ++k)
			{
				btDeformableFaceRigidContactConstraint& constraint = m_faceRigidConstraints[j][k];
				btScalar localResidualSquare = constraint.solveConstraint();
				residualSquare = btMax(residualSquare, localResidualSquare);
			}
			for (int k = 0; k < m_deformableConstraints[j].size(); ++k)
			{
				btDeformableFaceNodeContactConstraint& constraint = m_deformableConstraints[j][k];
				btScalar localResidualSquare = constraint.solveConstraint();
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

btScalar btDeformableContactProjection::solveSplitImpulse(const btContactSolverInfo& infoGlobal)
{
	btScalar residualSquare = 0;
	for (int i = 0; i < m_softBodies.size(); ++i)
	{
		// node constraints
		for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
		{
			btDeformableNodeRigidContactConstraint& constraint = m_nodeRigidConstraints[i][j];
			btScalar localResidualSquare = constraint.solveSplitImpulse(infoGlobal);
			residualSquare = btMax(residualSquare, localResidualSquare);
		}
		// anchor constraints
		for (int j = 0; j < m_nodeAnchorConstraints[i].size(); ++j)
		{
			btDeformableNodeAnchorConstraint& constraint = m_nodeAnchorConstraints[i][j];
			btScalar localResidualSquare = constraint.solveSplitImpulse(infoGlobal);
			residualSquare = btMax(residualSquare, localResidualSquare);
		}
		// face constraints
		for (int j = 0; j < m_faceRigidConstraints[i].size(); ++j)
		{
			btDeformableFaceRigidContactConstraint& constraint = m_faceRigidConstraints[i][j];
			btScalar localResidualSquare = constraint.solveSplitImpulse(infoGlobal);
			residualSquare = btMax(residualSquare, localResidualSquare);
		}

	}
	return residualSquare;
}

void btDeformableContactProjection::setConstraints()
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
				btDeformableStaticConstraint static_constraint(&psb->m_nodes[j]);
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
			btDeformableNodeAnchorConstraint constraint(anchor);
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
			btDeformableNodeRigidContactConstraint constraint(contact);
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
			btDeformableFaceRigidContactConstraint constraint(contact);
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
		
		// set Deformable Face vs. Deformable Node constraint
		for (int j = 0; j < psb->m_faceNodeContacts.size(); ++j)
		{
			const btSoftBody::DeformableFaceNodeContact& contact = psb->m_faceNodeContacts[j];

			btDeformableFaceNodeContactConstraint constraint(contact);
			btVector3 va = constraint.getVa();
			btVector3 vb = constraint.getVb();
			const btVector3 vr = vb - va;
			const btScalar dn = btDot(vr, contact.m_normal);
			if (dn > -SIMD_EPSILON)
			{
				m_deformableConstraints[i].push_back(constraint);
			}
		}
	}
}

void btDeformableContactProjection::project(TVStack& x)
{
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
}

void btDeformableContactProjection::setProjection()
{
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
			for (int k = 0; k < 3; ++k)
			{
				const btSoftBody::Node* node = face->m_n[k];
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
		for (int j = 0; j < m_deformableConstraints[i].size(); ++j)
		{
			const btSoftBody::Face* face = m_deformableConstraints[i][j].m_face;
			for (int k = 0; k < 3; ++k)
			{
				const btSoftBody::Node* node = face->m_n[k];
				int index = node->index;
				if (m_deformableConstraints[i][j].m_static)
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
						projections.push_back(m_deformableConstraints[i][j].m_normal);
						m_projectionsDict.insert(index, projections);
					}
					else
					{
						btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
						projections.push_back(m_deformableConstraints[i][j].m_normal);
					}
				}
			}
			
			const btSoftBody::Node* node = m_deformableConstraints[i][j].m_node;
			int index = node->index;
			if (m_deformableConstraints[i][j].m_static)
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
					projections.push_back(m_deformableConstraints[i][j].m_normal);
					m_projectionsDict.insert(index, projections);
				}
				else
				{
					btAlignedObjectArray<btVector3>& projections = *m_projectionsDict[index];
					projections.push_back(m_deformableConstraints[i][j].m_normal);
				}
			}
		}
	}
}


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
	m_projectionsDict.clear();
}



