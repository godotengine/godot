#include "btReducedDeformableBodySolver.h"
#include "btReducedDeformableBody.h"

btReducedDeformableBodySolver::btReducedDeformableBodySolver()
{
  m_ascendOrder = true;
  m_reducedSolver = true;
  m_dampingAlpha = 0;
  m_dampingBeta = 0;
  m_gravity = btVector3(0, 0, 0);
}

void btReducedDeformableBodySolver::setGravity(const btVector3& gravity)
{
  m_gravity = gravity;
}

void btReducedDeformableBodySolver::reinitialize(const btAlignedObjectArray<btSoftBody*>& bodies, btScalar dt)
{
  m_softBodies.copyFromArray(bodies);
	bool nodeUpdated = updateNodes();

	if (nodeUpdated)
	{
		m_dv.resize(m_numNodes, btVector3(0, 0, 0));
		m_ddv.resize(m_numNodes, btVector3(0, 0, 0));
		m_residual.resize(m_numNodes, btVector3(0, 0, 0));
		m_backupVelocity.resize(m_numNodes, btVector3(0, 0, 0));
	}

	// need to setZero here as resize only set value for newly allocated items
	for (int i = 0; i < m_numNodes; ++i)
	{
		m_dv[i].setZero();
		m_ddv[i].setZero();
		m_residual[i].setZero();
	}

	if (dt > 0)
	{
		m_dt = dt;
	}
	m_objective->reinitialize(nodeUpdated, dt);

  int N = bodies.size();
	if (nodeUpdated)
	{
		m_staticConstraints.resize(N);
		m_nodeRigidConstraints.resize(N);
		// m_faceRigidConstraints.resize(N);
	}
	for (int i = 0; i < N; ++i)
	{
		m_staticConstraints[i].clear();
		m_nodeRigidConstraints[i].clear();
		// m_faceRigidConstraints[i].clear();
	}

  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);
    rsb->m_contactNodesList.clear();
  }

  // set node index offsets
  int sum = 0;
  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);
    rsb->m_nodeIndexOffset = sum;
    sum += rsb->m_nodes.size();
  }

	btDeformableBodySolver::updateSoftBodies();
}

void btReducedDeformableBodySolver::predictMotion(btScalar solverdt)
{
  applyExplicitForce(solverdt);

  // predict new mesh location
  predictReduceDeformableMotion(solverdt);

  //TODO: check if there is anything missed from btDeformableBodySolver::predictDeformableMotion
}

void btReducedDeformableBodySolver::predictReduceDeformableMotion(btScalar solverdt)
{
  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);
    if (!rsb->isActive())
    {
      continue;
    }

    // clear contacts variables
		rsb->m_nodeRigidContacts.resize(0);
		rsb->m_faceRigidContacts.resize(0);
		rsb->m_faceNodeContacts.resize(0);
    
    // calculate inverse mass matrix for all nodes
    for (int j = 0; j < rsb->m_nodes.size(); ++j)
    {
      if (rsb->m_nodes[j].m_im > 0)
      {
        rsb->m_nodes[j].m_effectiveMass_inv = rsb->m_nodes[j].m_effectiveMass.inverse();
      }
    }

    // rigid motion: t, R at time^*
    rsb->predictIntegratedTransform(solverdt, rsb->getInterpolationWorldTransform());

    // update reduced dofs at time^*
    // rsb->updateReducedDofs(solverdt);

    // update local moment arm at time^*
    // rsb->updateLocalMomentArm();
    // rsb->updateExternalForceProjectMatrix(true);

    // predict full space velocity at time^* (needed for constraints)
    rsb->mapToFullVelocity(rsb->getInterpolationWorldTransform());

    // update full space nodal position at time^*
    rsb->mapToFullPosition(rsb->getInterpolationWorldTransform());

    // update bounding box
    rsb->updateBounds();

    // update tree
    rsb->updateNodeTree(true, true);
    if (!rsb->m_fdbvt.empty())
    {
      rsb->updateFaceTree(true, true);
    }
  }
}

void btReducedDeformableBodySolver::applyExplicitForce(btScalar solverdt)
{
  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);

    // apply gravity to the rigid frame, get m_linearVelocity at time^*
    rsb->applyRigidGravity(m_gravity, solverdt);

    if (!rsb->isReducedModesOFF())
    {
      // add internal force (elastic force & damping force)
      rsb->applyReducedElasticForce(rsb->m_reducedDofsBuffer);
      rsb->applyReducedDampingForce(rsb->m_reducedVelocityBuffer);

      // get reduced velocity at time^* 
      rsb->updateReducedVelocity(solverdt);
    }

    // apply damping (no need at this point)
    // rsb->applyDamping(solverdt);
  }
}

void btReducedDeformableBodySolver::applyTransforms(btScalar timeStep)
{
  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);

    // rigid motion
    rsb->proceedToTransform(timeStep, true);

    if (!rsb->isReducedModesOFF())
    {
      // update reduced dofs for time^n+1
      rsb->updateReducedDofs(timeStep);

      // update local moment arm for time^n+1
      rsb->updateLocalMomentArm();
      rsb->updateExternalForceProjectMatrix(true);
    }

    // update mesh nodal positions for time^n+1
    rsb->mapToFullPosition(rsb->getRigidTransform());

    // update mesh nodal velocity
    rsb->mapToFullVelocity(rsb->getRigidTransform());

    // end of time step clean up and update
    rsb->endOfTimeStepZeroing();

    // update the rendering mesh
    rsb->interpolateRenderMesh();
  }
}

void btReducedDeformableBodySolver::setConstraints(const btContactSolverInfo& infoGlobal)
{
  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);
    if (!rsb->isActive())
		{
			continue;
		}

    // set fixed constraints
    for (int j = 0; j < rsb->m_fixedNodes.size(); ++j)
		{
      int i_node = rsb->m_fixedNodes[j];
			if (rsb->m_nodes[i_node].m_im == 0)
			{
        for (int k = 0; k < 3; ++k)
        {
          btVector3 dir(0, 0, 0);
          dir[k] = 1;
          btReducedDeformableStaticConstraint static_constraint(rsb, &rsb->m_nodes[i_node], rsb->getRelativePos(i_node), rsb->m_x0[i_node], dir, infoGlobal, m_dt);
          m_staticConstraints[i].push_back(static_constraint);
        }
			}
		}
    btAssert(rsb->m_fixedNodes.size() * 3 == m_staticConstraints[i].size());

    // set Deformable Node vs. Rigid constraint
		for (int j = 0; j < rsb->m_nodeRigidContacts.size(); ++j)
		{
			const btSoftBody::DeformableNodeRigidContact& contact = rsb->m_nodeRigidContacts[j];
			// skip fixed points
			if (contact.m_node->m_im == 0)
			{
				continue;
			}
			btReducedDeformableNodeRigidContactConstraint constraint(rsb, contact, infoGlobal, m_dt);
			m_nodeRigidConstraints[i].push_back(constraint);
      rsb->m_contactNodesList.push_back(contact.m_node->index - rsb->m_nodeIndexOffset);
		}
    // std::cout << "contact node list size: " << rsb->m_contactNodesList.size() << "\n";
    // std::cout << "#contact nodes: " << m_nodeRigidConstraints[i].size() << "\n";

  }
}

btScalar btReducedDeformableBodySolver::solveContactConstraints(btCollisionObject** deformableBodies, int numDeformableBodies, const btContactSolverInfo& infoGlobal)
{
  btScalar residualSquare = 0;

  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btAlignedObjectArray<int> m_orderNonContactConstraintPool;
    btAlignedObjectArray<int> m_orderContactConstraintPool;

    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);

    // shuffle the order of applying constraint
    m_orderNonContactConstraintPool.resize(m_staticConstraints[i].size());
    m_orderContactConstraintPool.resize(m_nodeRigidConstraints[i].size());
    if (infoGlobal.m_solverMode & SOLVER_RANDMIZE_ORDER)
    {
      // fixed constraint order
      for (int j = 0; j < m_staticConstraints[i].size(); ++j)
      {
        m_orderNonContactConstraintPool[j] = m_ascendOrder ? j : m_staticConstraints[i].size() - 1 - j;
      }
      // contact constraint order
      for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
      {
        m_orderContactConstraintPool[j] = m_ascendOrder ? j : m_nodeRigidConstraints[i].size() - 1 - j;
      }

      m_ascendOrder = m_ascendOrder ? false : true;
    }
    else
    {
      for (int j = 0; j < m_staticConstraints[i].size(); ++j)
      {
        m_orderNonContactConstraintPool[j] = j;
      }
      // contact constraint order
      for (int j = 0; j < m_nodeRigidConstraints[i].size(); ++j)
      {
        m_orderContactConstraintPool[j] = j;
      }
    }

    // handle fixed constraint
    for (int k = 0; k < m_staticConstraints[i].size(); ++k)
    {
      btReducedDeformableStaticConstraint& constraint = m_staticConstraints[i][m_orderNonContactConstraintPool[k]];
      btScalar localResidualSquare = constraint.solveConstraint(infoGlobal);
      residualSquare = btMax(residualSquare, localResidualSquare);
    }

    // handle contact constraint

    // node vs rigid contact
    // std::cout << "!!#contact_nodes: " << m_nodeRigidConstraints[i].size() << '\n';
    for (int k = 0; k < m_nodeRigidConstraints[i].size(); ++k)
    {
      btReducedDeformableNodeRigidContactConstraint& constraint = m_nodeRigidConstraints[i][m_orderContactConstraintPool[k]];
      btScalar localResidualSquare = constraint.solveConstraint(infoGlobal);
      residualSquare = btMax(residualSquare, localResidualSquare);
    }

    // face vs rigid contact
    // for (int k = 0; k < m_faceRigidConstraints[i].size(); ++k)
    // {
    // 	btReducedDeformableFaceRigidContactConstraint& constraint = m_faceRigidConstraints[i][k];
    // 	btScalar localResidualSquare = constraint.solveConstraint(infoGlobal);
    // 	residualSquare = btMax(residualSquare, localResidualSquare);
    // }
  }

  
	return residualSquare;
}

void btReducedDeformableBodySolver::deformableBodyInternalWriteBack()
{
  // reduced deformable update
  for (int i = 0; i < m_softBodies.size(); ++i)
  {
    btReducedDeformableBody* rsb = static_cast<btReducedDeformableBody*>(m_softBodies[i]);
    rsb->applyInternalVelocityChanges();
  }
  m_ascendOrder = true;
}