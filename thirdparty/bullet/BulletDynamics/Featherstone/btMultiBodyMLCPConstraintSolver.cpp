/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2018 Google Inc. http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "BulletDynamics/Featherstone/btMultiBodyMLCPConstraintSolver.h"

#include "BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraint.h"
#include "BulletDynamics/MLCPSolvers/btMLCPSolverInterface.h"

#define DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS

static bool interleaveContactAndFriction1 = false;

struct btJointNode1
{
	int jointIndex;          // pointer to enclosing dxJoint object
	int otherBodyIndex;      // *other* body this joint is connected to
	int nextJointNodeIndex;  //-1 for null
	int constraintRowIndex;
};

// Helper function to compute a delta velocity in the constraint space.
static btScalar computeDeltaVelocityInConstraintSpace(
	const btVector3& angularDeltaVelocity,
	const btVector3& contactNormal,
	btScalar invMass,
	const btVector3& angularJacobian,
	const btVector3& linearJacobian)
{
	return angularDeltaVelocity.dot(angularJacobian) + contactNormal.dot(linearJacobian) * invMass;
}

// Faster version of computeDeltaVelocityInConstraintSpace that can be used when contactNormal and linearJacobian are
// identical.
static btScalar computeDeltaVelocityInConstraintSpace(
	const btVector3& angularDeltaVelocity,
	btScalar invMass,
	const btVector3& angularJacobian)
{
	return angularDeltaVelocity.dot(angularJacobian) + invMass;
}

// Helper function to compute a delta velocity in the constraint space.
static btScalar computeDeltaVelocityInConstraintSpace(const btScalar* deltaVelocity, const btScalar* jacobian, int size)
{
	btScalar result = 0;
	for (int i = 0; i < size; ++i)
		result += deltaVelocity[i] * jacobian[i];

	return result;
}

static btScalar computeConstraintMatrixDiagElementMultiBody(
	const btAlignedObjectArray<btSolverBody>& solverBodyPool,
	const btMultiBodyJacobianData& data,
	const btMultiBodySolverConstraint& constraint)
{
	btScalar ret = 0;

	const btMultiBody* multiBodyA = constraint.m_multiBodyA;
	const btMultiBody* multiBodyB = constraint.m_multiBodyB;

	if (multiBodyA)
	{
		const btScalar* jacA = &data.m_jacobians[constraint.m_jacAindex];
		const btScalar* deltaA = &data.m_deltaVelocitiesUnitImpulse[constraint.m_jacAindex];
		const int ndofA = multiBodyA->getNumDofs() + 6;
		ret += computeDeltaVelocityInConstraintSpace(deltaA, jacA, ndofA);
	}
	else
	{
		const int solverBodyIdA = constraint.m_solverBodyIdA;
		btAssert(solverBodyIdA != -1);
		const btSolverBody* solverBodyA = &solverBodyPool[solverBodyIdA];
		const btScalar invMassA = solverBodyA->m_originalBody ? solverBodyA->m_originalBody->getInvMass() : 0.0;
		ret += computeDeltaVelocityInConstraintSpace(
			constraint.m_relpos1CrossNormal,
			invMassA,
			constraint.m_angularComponentA);
	}

	if (multiBodyB)
	{
		const btScalar* jacB = &data.m_jacobians[constraint.m_jacBindex];
		const btScalar* deltaB = &data.m_deltaVelocitiesUnitImpulse[constraint.m_jacBindex];
		const int ndofB = multiBodyB->getNumDofs() + 6;
		ret += computeDeltaVelocityInConstraintSpace(deltaB, jacB, ndofB);
	}
	else
	{
		const int solverBodyIdB = constraint.m_solverBodyIdB;
		btAssert(solverBodyIdB != -1);
		const btSolverBody* solverBodyB = &solverBodyPool[solverBodyIdB];
		const btScalar invMassB = solverBodyB->m_originalBody ? solverBodyB->m_originalBody->getInvMass() : 0.0;
		ret += computeDeltaVelocityInConstraintSpace(
			constraint.m_relpos2CrossNormal,
			invMassB,
			constraint.m_angularComponentB);
	}

	return ret;
}

static btScalar computeConstraintMatrixOffDiagElementMultiBody(
	const btAlignedObjectArray<btSolverBody>& solverBodyPool,
	const btMultiBodyJacobianData& data,
	const btMultiBodySolverConstraint& constraint,
	const btMultiBodySolverConstraint& offDiagConstraint)
{
	btScalar offDiagA = btScalar(0);

	const btMultiBody* multiBodyA = constraint.m_multiBodyA;
	const btMultiBody* multiBodyB = constraint.m_multiBodyB;
	const btMultiBody* offDiagMultiBodyA = offDiagConstraint.m_multiBodyA;
	const btMultiBody* offDiagMultiBodyB = offDiagConstraint.m_multiBodyB;

	// Assumed at least one system is multibody
	btAssert(multiBodyA || multiBodyB);
	btAssert(offDiagMultiBodyA || offDiagMultiBodyB);

	if (offDiagMultiBodyA)
	{
		const btScalar* offDiagJacA = &data.m_jacobians[offDiagConstraint.m_jacAindex];

		if (offDiagMultiBodyA == multiBodyA)
		{
			const int ndofA = multiBodyA->getNumDofs() + 6;
			const btScalar* deltaA = &data.m_deltaVelocitiesUnitImpulse[constraint.m_jacAindex];
			offDiagA += computeDeltaVelocityInConstraintSpace(deltaA, offDiagJacA, ndofA);
		}
		else if (offDiagMultiBodyA == multiBodyB)
		{
			const int ndofB = multiBodyB->getNumDofs() + 6;
			const btScalar* deltaB = &data.m_deltaVelocitiesUnitImpulse[constraint.m_jacBindex];
			offDiagA += computeDeltaVelocityInConstraintSpace(deltaB, offDiagJacA, ndofB);
		}
	}
	else
	{
		const int solverBodyIdA = constraint.m_solverBodyIdA;
		const int solverBodyIdB = constraint.m_solverBodyIdB;

		const int offDiagSolverBodyIdA = offDiagConstraint.m_solverBodyIdA;
		btAssert(offDiagSolverBodyIdA != -1);

		if (offDiagSolverBodyIdA == solverBodyIdA)
		{
			btAssert(solverBodyIdA != -1);
			const btSolverBody* solverBodyA = &solverBodyPool[solverBodyIdA];
			const btScalar invMassA = solverBodyA->m_originalBody ? solverBodyA->m_originalBody->getInvMass() : 0.0;
			offDiagA += computeDeltaVelocityInConstraintSpace(
				offDiagConstraint.m_relpos1CrossNormal,
				offDiagConstraint.m_contactNormal1,
				invMassA, constraint.m_angularComponentA,
				constraint.m_contactNormal1);
		}
		else if (offDiagSolverBodyIdA == solverBodyIdB)
		{
			btAssert(solverBodyIdB != -1);
			const btSolverBody* solverBodyB = &solverBodyPool[solverBodyIdB];
			const btScalar invMassB = solverBodyB->m_originalBody ? solverBodyB->m_originalBody->getInvMass() : 0.0;
			offDiagA += computeDeltaVelocityInConstraintSpace(
				offDiagConstraint.m_relpos1CrossNormal,
				offDiagConstraint.m_contactNormal1,
				invMassB,
				constraint.m_angularComponentB,
				constraint.m_contactNormal2);
		}
	}

	if (offDiagMultiBodyB)
	{
		const btScalar* offDiagJacB = &data.m_jacobians[offDiagConstraint.m_jacBindex];

		if (offDiagMultiBodyB == multiBodyA)
		{
			const int ndofA = multiBodyA->getNumDofs() + 6;
			const btScalar* deltaA = &data.m_deltaVelocitiesUnitImpulse[constraint.m_jacAindex];
			offDiagA += computeDeltaVelocityInConstraintSpace(deltaA, offDiagJacB, ndofA);
		}
		else if (offDiagMultiBodyB == multiBodyB)
		{
			const int ndofB = multiBodyB->getNumDofs() + 6;
			const btScalar* deltaB = &data.m_deltaVelocitiesUnitImpulse[constraint.m_jacBindex];
			offDiagA += computeDeltaVelocityInConstraintSpace(deltaB, offDiagJacB, ndofB);
		}
	}
	else
	{
		const int solverBodyIdA = constraint.m_solverBodyIdA;
		const int solverBodyIdB = constraint.m_solverBodyIdB;

		const int offDiagSolverBodyIdB = offDiagConstraint.m_solverBodyIdB;
		btAssert(offDiagSolverBodyIdB != -1);

		if (offDiagSolverBodyIdB == solverBodyIdA)
		{
			btAssert(solverBodyIdA != -1);
			const btSolverBody* solverBodyA = &solverBodyPool[solverBodyIdA];
			const btScalar invMassA = solverBodyA->m_originalBody ? solverBodyA->m_originalBody->getInvMass() : 0.0;
			offDiagA += computeDeltaVelocityInConstraintSpace(
				offDiagConstraint.m_relpos2CrossNormal,
				offDiagConstraint.m_contactNormal2,
				invMassA, constraint.m_angularComponentA,
				constraint.m_contactNormal1);
		}
		else if (offDiagSolverBodyIdB == solverBodyIdB)
		{
			btAssert(solverBodyIdB != -1);
			const btSolverBody* solverBodyB = &solverBodyPool[solverBodyIdB];
			const btScalar invMassB = solverBodyB->m_originalBody ? solverBodyB->m_originalBody->getInvMass() : 0.0;
			offDiagA += computeDeltaVelocityInConstraintSpace(
				offDiagConstraint.m_relpos2CrossNormal,
				offDiagConstraint.m_contactNormal2,
				invMassB, constraint.m_angularComponentB,
				constraint.m_contactNormal2);
		}
	}

	return offDiagA;
}

void btMultiBodyMLCPConstraintSolver::createMLCPFast(const btContactSolverInfo& infoGlobal)
{
	createMLCPFastRigidBody(infoGlobal);
	createMLCPFastMultiBody(infoGlobal);
}

void btMultiBodyMLCPConstraintSolver::createMLCPFastRigidBody(const btContactSolverInfo& infoGlobal)
{
	int numContactRows = interleaveContactAndFriction1 ? 3 : 1;

	int numConstraintRows = m_allConstraintPtrArray.size();

	if (numConstraintRows == 0)
		return;

	int n = numConstraintRows;
	{
		BT_PROFILE("init b (rhs)");
		m_b.resize(numConstraintRows);
		m_bSplit.resize(numConstraintRows);
		m_b.setZero();
		m_bSplit.setZero();
		for (int i = 0; i < numConstraintRows; i++)
		{
			btScalar jacDiag = m_allConstraintPtrArray[i]->m_jacDiagABInv;
			if (!btFuzzyZero(jacDiag))
			{
				btScalar rhs = m_allConstraintPtrArray[i]->m_rhs;
				btScalar rhsPenetration = m_allConstraintPtrArray[i]->m_rhsPenetration;
				m_b[i] = rhs / jacDiag;
				m_bSplit[i] = rhsPenetration / jacDiag;
			}
		}
	}

	//	btScalar* w = 0;
	//	int nub = 0;

	m_lo.resize(numConstraintRows);
	m_hi.resize(numConstraintRows);

	{
		BT_PROFILE("init lo/ho");

		for (int i = 0; i < numConstraintRows; i++)
		{
			if (0)  //m_limitDependencies[i]>=0)
			{
				m_lo[i] = -BT_INFINITY;
				m_hi[i] = BT_INFINITY;
			}
			else
			{
				m_lo[i] = m_allConstraintPtrArray[i]->m_lowerLimit;
				m_hi[i] = m_allConstraintPtrArray[i]->m_upperLimit;
			}
		}
	}

	//
	int m = m_allConstraintPtrArray.size();

	int numBodies = m_tmpSolverBodyPool.size();
	btAlignedObjectArray<int> bodyJointNodeArray;
	{
		BT_PROFILE("bodyJointNodeArray.resize");
		bodyJointNodeArray.resize(numBodies, -1);
	}
	btAlignedObjectArray<btJointNode1> jointNodeArray;
	{
		BT_PROFILE("jointNodeArray.reserve");
		jointNodeArray.reserve(2 * m_allConstraintPtrArray.size());
	}

	btMatrixXu& J3 = m_scratchJ3;
	{
		BT_PROFILE("J3.resize");
		J3.resize(2 * m, 8);
	}
	btMatrixXu& JinvM3 = m_scratchJInvM3;
	{
		BT_PROFILE("JinvM3.resize/setZero");

		JinvM3.resize(2 * m, 8);
		JinvM3.setZero();
		J3.setZero();
	}
	int cur = 0;
	int rowOffset = 0;
	btAlignedObjectArray<int>& ofs = m_scratchOfs;
	{
		BT_PROFILE("ofs resize");
		ofs.resize(0);
		ofs.resizeNoInitialize(m_allConstraintPtrArray.size());
	}
	{
		BT_PROFILE("Compute J and JinvM");
		int c = 0;

		int numRows = 0;

		for (int i = 0; i < m_allConstraintPtrArray.size(); i += numRows, c++)
		{
			ofs[c] = rowOffset;
			int sbA = m_allConstraintPtrArray[i]->m_solverBodyIdA;
			int sbB = m_allConstraintPtrArray[i]->m_solverBodyIdB;
			btRigidBody* orgBodyA = m_tmpSolverBodyPool[sbA].m_originalBody;
			btRigidBody* orgBodyB = m_tmpSolverBodyPool[sbB].m_originalBody;

			numRows = i < m_tmpSolverNonContactConstraintPool.size() ? m_tmpConstraintSizesPool[c].m_numConstraintRows : numContactRows;
			if (orgBodyA)
			{
				{
					int slotA = -1;
					//find free jointNode slot for sbA
					slotA = jointNodeArray.size();
					jointNodeArray.expand();  //NonInitializing();
					int prevSlot = bodyJointNodeArray[sbA];
					bodyJointNodeArray[sbA] = slotA;
					jointNodeArray[slotA].nextJointNodeIndex = prevSlot;
					jointNodeArray[slotA].jointIndex = c;
					jointNodeArray[slotA].constraintRowIndex = i;
					jointNodeArray[slotA].otherBodyIndex = orgBodyB ? sbB : -1;
				}
				for (int row = 0; row < numRows; row++, cur++)
				{
					btVector3 normalInvMass = m_allConstraintPtrArray[i + row]->m_contactNormal1 * orgBodyA->getInvMass();
					btVector3 relPosCrossNormalInvInertia = m_allConstraintPtrArray[i + row]->m_relpos1CrossNormal * orgBodyA->getInvInertiaTensorWorld();

					for (int r = 0; r < 3; r++)
					{
						J3.setElem(cur, r, m_allConstraintPtrArray[i + row]->m_contactNormal1[r]);
						J3.setElem(cur, r + 4, m_allConstraintPtrArray[i + row]->m_relpos1CrossNormal[r]);
						JinvM3.setElem(cur, r, normalInvMass[r]);
						JinvM3.setElem(cur, r + 4, relPosCrossNormalInvInertia[r]);
					}
					J3.setElem(cur, 3, 0);
					JinvM3.setElem(cur, 3, 0);
					J3.setElem(cur, 7, 0);
					JinvM3.setElem(cur, 7, 0);
				}
			}
			else
			{
				cur += numRows;
			}
			if (orgBodyB)
			{
				{
					int slotB = -1;
					//find free jointNode slot for sbA
					slotB = jointNodeArray.size();
					jointNodeArray.expand();  //NonInitializing();
					int prevSlot = bodyJointNodeArray[sbB];
					bodyJointNodeArray[sbB] = slotB;
					jointNodeArray[slotB].nextJointNodeIndex = prevSlot;
					jointNodeArray[slotB].jointIndex = c;
					jointNodeArray[slotB].otherBodyIndex = orgBodyA ? sbA : -1;
					jointNodeArray[slotB].constraintRowIndex = i;
				}

				for (int row = 0; row < numRows; row++, cur++)
				{
					btVector3 normalInvMassB = m_allConstraintPtrArray[i + row]->m_contactNormal2 * orgBodyB->getInvMass();
					btVector3 relPosInvInertiaB = m_allConstraintPtrArray[i + row]->m_relpos2CrossNormal * orgBodyB->getInvInertiaTensorWorld();

					for (int r = 0; r < 3; r++)
					{
						J3.setElem(cur, r, m_allConstraintPtrArray[i + row]->m_contactNormal2[r]);
						J3.setElem(cur, r + 4, m_allConstraintPtrArray[i + row]->m_relpos2CrossNormal[r]);
						JinvM3.setElem(cur, r, normalInvMassB[r]);
						JinvM3.setElem(cur, r + 4, relPosInvInertiaB[r]);
					}
					J3.setElem(cur, 3, 0);
					JinvM3.setElem(cur, 3, 0);
					J3.setElem(cur, 7, 0);
					JinvM3.setElem(cur, 7, 0);
				}
			}
			else
			{
				cur += numRows;
			}
			rowOffset += numRows;
		}
	}

	//compute JinvM = J*invM.
	const btScalar* JinvM = JinvM3.getBufferPointer();

	const btScalar* Jptr = J3.getBufferPointer();
	{
		BT_PROFILE("m_A.resize");
		m_A.resize(n, n);
	}

	{
		BT_PROFILE("m_A.setZero");
		m_A.setZero();
	}
	int c = 0;
	{
		int numRows = 0;
		BT_PROFILE("Compute A");
		for (int i = 0; i < m_allConstraintPtrArray.size(); i += numRows, c++)
		{
			int row__ = ofs[c];
			int sbA = m_allConstraintPtrArray[i]->m_solverBodyIdA;
			int sbB = m_allConstraintPtrArray[i]->m_solverBodyIdB;
			//	btRigidBody* orgBodyA = m_tmpSolverBodyPool[sbA].m_originalBody;
			//	btRigidBody* orgBodyB = m_tmpSolverBodyPool[sbB].m_originalBody;

			numRows = i < m_tmpSolverNonContactConstraintPool.size() ? m_tmpConstraintSizesPool[c].m_numConstraintRows : numContactRows;

			const btScalar* JinvMrow = JinvM + 2 * 8 * (size_t)row__;

			{
				int startJointNodeA = bodyJointNodeArray[sbA];
				while (startJointNodeA >= 0)
				{
					int j0 = jointNodeArray[startJointNodeA].jointIndex;
					int cr0 = jointNodeArray[startJointNodeA].constraintRowIndex;
					if (j0 < c)
					{
						int numRowsOther = cr0 < m_tmpSolverNonContactConstraintPool.size() ? m_tmpConstraintSizesPool[j0].m_numConstraintRows : numContactRows;
						size_t ofsother = (m_allConstraintPtrArray[cr0]->m_solverBodyIdB == sbA) ? 8 * numRowsOther : 0;
						//printf("%d joint i %d and j0: %d: ",count++,i,j0);
						m_A.multiplyAdd2_p8r(JinvMrow,
											 Jptr + 2 * 8 * (size_t)ofs[j0] + ofsother, numRows, numRowsOther, row__, ofs[j0]);
					}
					startJointNodeA = jointNodeArray[startJointNodeA].nextJointNodeIndex;
				}
			}

			{
				int startJointNodeB = bodyJointNodeArray[sbB];
				while (startJointNodeB >= 0)
				{
					int j1 = jointNodeArray[startJointNodeB].jointIndex;
					int cj1 = jointNodeArray[startJointNodeB].constraintRowIndex;

					if (j1 < c)
					{
						int numRowsOther = cj1 < m_tmpSolverNonContactConstraintPool.size() ? m_tmpConstraintSizesPool[j1].m_numConstraintRows : numContactRows;
						size_t ofsother = (m_allConstraintPtrArray[cj1]->m_solverBodyIdB == sbB) ? 8 * numRowsOther : 0;
						m_A.multiplyAdd2_p8r(JinvMrow + 8 * (size_t)numRows,
											 Jptr + 2 * 8 * (size_t)ofs[j1] + ofsother, numRows, numRowsOther, row__, ofs[j1]);
					}
					startJointNodeB = jointNodeArray[startJointNodeB].nextJointNodeIndex;
				}
			}
		}

		{
			BT_PROFILE("compute diagonal");
			// compute diagonal blocks of m_A

			int row__ = 0;
			int numJointRows = m_allConstraintPtrArray.size();

			int jj = 0;
			for (; row__ < numJointRows;)
			{
				//int sbA = m_allConstraintPtrArray[row__]->m_solverBodyIdA;
				int sbB = m_allConstraintPtrArray[row__]->m_solverBodyIdB;
				//	btRigidBody* orgBodyA = m_tmpSolverBodyPool[sbA].m_originalBody;
				btRigidBody* orgBodyB = m_tmpSolverBodyPool[sbB].m_originalBody;

				const unsigned int infom = row__ < m_tmpSolverNonContactConstraintPool.size() ? m_tmpConstraintSizesPool[jj].m_numConstraintRows : numContactRows;

				const btScalar* JinvMrow = JinvM + 2 * 8 * (size_t)row__;
				const btScalar* Jrow = Jptr + 2 * 8 * (size_t)row__;
				m_A.multiply2_p8r(JinvMrow, Jrow, infom, infom, row__, row__);
				if (orgBodyB)
				{
					m_A.multiplyAdd2_p8r(JinvMrow + 8 * (size_t)infom, Jrow + 8 * (size_t)infom, infom, infom, row__, row__);
				}
				row__ += infom;
				jj++;
			}
		}
	}

	if (1)
	{
		// add cfm to the diagonal of m_A
		for (int i = 0; i < m_A.rows(); ++i)
		{
			m_A.setElem(i, i, m_A(i, i) + infoGlobal.m_globalCfm / infoGlobal.m_timeStep);
		}
	}

	///fill the upper triangle of the matrix, to make it symmetric
	{
		BT_PROFILE("fill the upper triangle ");
		m_A.copyLowerToUpperTriangle();
	}

	{
		BT_PROFILE("resize/init x");
		m_x.resize(numConstraintRows);
		m_xSplit.resize(numConstraintRows);

		if (infoGlobal.m_solverMode & SOLVER_USE_WARMSTARTING)
		{
			for (int i = 0; i < m_allConstraintPtrArray.size(); i++)
			{
				const btSolverConstraint& c = *m_allConstraintPtrArray[i];
				m_x[i] = c.m_appliedImpulse;
				m_xSplit[i] = c.m_appliedPushImpulse;
			}
		}
		else
		{
			m_x.setZero();
			m_xSplit.setZero();
		}
	}
}

void btMultiBodyMLCPConstraintSolver::createMLCPFastMultiBody(const btContactSolverInfo& infoGlobal)
{
	const int multiBodyNumConstraints = m_multiBodyAllConstraintPtrArray.size();

	if (multiBodyNumConstraints == 0)
		return;

	// 1. Compute b
	{
		BT_PROFILE("init b (rhs)");

		m_multiBodyB.resize(multiBodyNumConstraints);
		m_multiBodyB.setZero();

		for (int i = 0; i < multiBodyNumConstraints; ++i)
		{
			const btMultiBodySolverConstraint& constraint = *m_multiBodyAllConstraintPtrArray[i];
			const btScalar jacDiag = constraint.m_jacDiagABInv;

			if (!btFuzzyZero(jacDiag))
			{
				// Note that rhsPenetration is currently always zero because the split impulse hasn't been implemented for multibody yet.
				const btScalar rhs = constraint.m_rhs;
				m_multiBodyB[i] = rhs / jacDiag;
			}
		}
	}

	// 2. Compute lo and hi
	{
		BT_PROFILE("init lo/ho");

		m_multiBodyLo.resize(multiBodyNumConstraints);
		m_multiBodyHi.resize(multiBodyNumConstraints);

		for (int i = 0; i < multiBodyNumConstraints; ++i)
		{
			const btMultiBodySolverConstraint& constraint = *m_multiBodyAllConstraintPtrArray[i];
			m_multiBodyLo[i] = constraint.m_lowerLimit;
			m_multiBodyHi[i] = constraint.m_upperLimit;
		}
	}

	// 3. Construct A matrix by using the impulse testing
	{
		BT_PROFILE("Compute A");

		{
			BT_PROFILE("m_A.resize");
			m_multiBodyA.resize(multiBodyNumConstraints, multiBodyNumConstraints);
		}

		for (int i = 0; i < multiBodyNumConstraints; ++i)
		{
			// Compute the diagonal of A, which is A(i, i)
			const btMultiBodySolverConstraint& constraint = *m_multiBodyAllConstraintPtrArray[i];
			const btScalar diagA = computeConstraintMatrixDiagElementMultiBody(m_tmpSolverBodyPool, m_data, constraint);
			m_multiBodyA.setElem(i, i, diagA);

			// Computes the off-diagonals of A:
			//   a. The rest of i-th row of A, from A(i, i+1) to A(i, n)
			//   b. The rest of i-th column of A, from A(i+1, i) to A(n, i)
			for (int j = i + 1; j < multiBodyNumConstraints; ++j)
			{
				const btMultiBodySolverConstraint& offDiagConstraint = *m_multiBodyAllConstraintPtrArray[j];
				const btScalar offDiagA = computeConstraintMatrixOffDiagElementMultiBody(m_tmpSolverBodyPool, m_data, constraint, offDiagConstraint);

				// Set the off-diagonal values of A. Note that A is symmetric.
				m_multiBodyA.setElem(i, j, offDiagA);
				m_multiBodyA.setElem(j, i, offDiagA);
			}
		}
	}

	// Add CFM to the diagonal of m_A
	for (int i = 0; i < m_multiBodyA.rows(); ++i)
	{
		m_multiBodyA.setElem(i, i, m_multiBodyA(i, i) + infoGlobal.m_globalCfm / infoGlobal.m_timeStep);
	}

	// 4. Initialize x
	{
		BT_PROFILE("resize/init x");

		m_multiBodyX.resize(multiBodyNumConstraints);

		if (infoGlobal.m_solverMode & SOLVER_USE_WARMSTARTING)
		{
			for (int i = 0; i < multiBodyNumConstraints; ++i)
			{
				const btMultiBodySolverConstraint& constraint = *m_multiBodyAllConstraintPtrArray[i];
				m_multiBodyX[i] = constraint.m_appliedImpulse;
			}
		}
		else
		{
			m_multiBodyX.setZero();
		}
	}
}

bool btMultiBodyMLCPConstraintSolver::solveMLCP(const btContactSolverInfo& infoGlobal)
{
	bool result = true;

	if (m_A.rows() != 0)
	{
		// If using split impulse, we solve 2 separate (M)LCPs
		if (infoGlobal.m_splitImpulse)
		{
			const btMatrixXu Acopy = m_A;
			const btAlignedObjectArray<int> limitDependenciesCopy = m_limitDependencies;
			// TODO(JS): Do we really need these copies when solveMLCP takes them as const?

			result = m_solver->solveMLCP(m_A, m_b, m_x, m_lo, m_hi, m_limitDependencies, infoGlobal.m_numIterations);
			if (result)
				result = m_solver->solveMLCP(Acopy, m_bSplit, m_xSplit, m_lo, m_hi, limitDependenciesCopy, infoGlobal.m_numIterations);
		}
		else
		{
			result = m_solver->solveMLCP(m_A, m_b, m_x, m_lo, m_hi, m_limitDependencies, infoGlobal.m_numIterations);
		}
	}

	if (!result)
		return false;

	if (m_multiBodyA.rows() != 0)
	{
		result = m_solver->solveMLCP(m_multiBodyA, m_multiBodyB, m_multiBodyX, m_multiBodyLo, m_multiBodyHi, m_multiBodyLimitDependencies, infoGlobal.m_numIterations);
	}

	return result;
}

btScalar btMultiBodyMLCPConstraintSolver::solveGroupCacheFriendlySetup(
	btCollisionObject** bodies,
	int numBodies,
	btPersistentManifold** manifoldPtr,
	int numManifolds,
	btTypedConstraint** constraints,
	int numConstraints,
	const btContactSolverInfo& infoGlobal,
	btIDebugDraw* debugDrawer)
{
	// 1. Setup for rigid-bodies
	btMultiBodyConstraintSolver::solveGroupCacheFriendlySetup(
		bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);

	// 2. Setup for multi-bodies
	//   a. Collect all different kinds of constraint as pointers into one array, m_allConstraintPtrArray
	//   b. Set the index array for frictional contact constraints, m_limitDependencies
	{
		BT_PROFILE("gather constraint data");

		int dindex = 0;

		const int numRigidBodyConstraints = m_tmpSolverNonContactConstraintPool.size() + m_tmpSolverContactConstraintPool.size() + m_tmpSolverContactFrictionConstraintPool.size();
		const int numMultiBodyConstraints = m_multiBodyNonContactConstraints.size() + m_multiBodyNormalContactConstraints.size() + m_multiBodyFrictionContactConstraints.size();

		m_allConstraintPtrArray.resize(0);
		m_multiBodyAllConstraintPtrArray.resize(0);

		// i. Setup for rigid bodies

		m_limitDependencies.resize(numRigidBodyConstraints);

		for (int i = 0; i < m_tmpSolverNonContactConstraintPool.size(); ++i)
		{
			m_allConstraintPtrArray.push_back(&m_tmpSolverNonContactConstraintPool[i]);
			m_limitDependencies[dindex++] = -1;
		}

		int firstContactConstraintOffset = dindex;

		// The btSequentialImpulseConstraintSolver moves all friction constraints at the very end, we can also interleave them instead
		if (interleaveContactAndFriction1)
		{
			for (int i = 0; i < m_tmpSolverContactConstraintPool.size(); i++)
			{
				const int numFrictionPerContact = m_tmpSolverContactConstraintPool.size() == m_tmpSolverContactFrictionConstraintPool.size() ? 1 : 2;

				m_allConstraintPtrArray.push_back(&m_tmpSolverContactConstraintPool[i]);
				m_limitDependencies[dindex++] = -1;
				m_allConstraintPtrArray.push_back(&m_tmpSolverContactFrictionConstraintPool[i * numFrictionPerContact]);
				int findex = (m_tmpSolverContactFrictionConstraintPool[i * numFrictionPerContact].m_frictionIndex * (1 + numFrictionPerContact));
				m_limitDependencies[dindex++] = findex + firstContactConstraintOffset;
				if (numFrictionPerContact == 2)
				{
					m_allConstraintPtrArray.push_back(&m_tmpSolverContactFrictionConstraintPool[i * numFrictionPerContact + 1]);
					m_limitDependencies[dindex++] = findex + firstContactConstraintOffset;
				}
			}
		}
		else
		{
			for (int i = 0; i < m_tmpSolverContactConstraintPool.size(); i++)
			{
				m_allConstraintPtrArray.push_back(&m_tmpSolverContactConstraintPool[i]);
				m_limitDependencies[dindex++] = -1;
			}
			for (int i = 0; i < m_tmpSolverContactFrictionConstraintPool.size(); i++)
			{
				m_allConstraintPtrArray.push_back(&m_tmpSolverContactFrictionConstraintPool[i]);
				m_limitDependencies[dindex++] = m_tmpSolverContactFrictionConstraintPool[i].m_frictionIndex + firstContactConstraintOffset;
			}
		}

		if (!m_allConstraintPtrArray.size())
		{
			m_A.resize(0, 0);
			m_b.resize(0);
			m_x.resize(0);
			m_lo.resize(0);
			m_hi.resize(0);
		}

		// ii. Setup for multibodies

		dindex = 0;

		m_multiBodyLimitDependencies.resize(numMultiBodyConstraints);

		for (int i = 0; i < m_multiBodyNonContactConstraints.size(); ++i)
		{
			m_multiBodyAllConstraintPtrArray.push_back(&m_multiBodyNonContactConstraints[i]);
			m_multiBodyLimitDependencies[dindex++] = -1;
		}

		firstContactConstraintOffset = dindex;

		// The btSequentialImpulseConstraintSolver moves all friction constraints at the very end, we can also interleave them instead
		if (interleaveContactAndFriction1)
		{
			for (int i = 0; i < m_multiBodyNormalContactConstraints.size(); ++i)
			{
				const int numtiBodyNumFrictionPerContact = m_multiBodyNormalContactConstraints.size() == m_multiBodyFrictionContactConstraints.size() ? 1 : 2;

				m_multiBodyAllConstraintPtrArray.push_back(&m_multiBodyNormalContactConstraints[i]);
				m_multiBodyLimitDependencies[dindex++] = -1;

				btMultiBodySolverConstraint& frictionContactConstraint1 = m_multiBodyFrictionContactConstraints[i * numtiBodyNumFrictionPerContact];
				m_multiBodyAllConstraintPtrArray.push_back(&frictionContactConstraint1);

				const int findex = (frictionContactConstraint1.m_frictionIndex * (1 + numtiBodyNumFrictionPerContact)) + firstContactConstraintOffset;

				m_multiBodyLimitDependencies[dindex++] = findex;

				if (numtiBodyNumFrictionPerContact == 2)
				{
					btMultiBodySolverConstraint& frictionContactConstraint2 = m_multiBodyFrictionContactConstraints[i * numtiBodyNumFrictionPerContact + 1];
					m_multiBodyAllConstraintPtrArray.push_back(&frictionContactConstraint2);

					m_multiBodyLimitDependencies[dindex++] = findex;
				}
			}
		}
		else
		{
			for (int i = 0; i < m_multiBodyNormalContactConstraints.size(); ++i)
			{
				m_multiBodyAllConstraintPtrArray.push_back(&m_multiBodyNormalContactConstraints[i]);
				m_multiBodyLimitDependencies[dindex++] = -1;
			}
			for (int i = 0; i < m_multiBodyFrictionContactConstraints.size(); ++i)
			{
				m_multiBodyAllConstraintPtrArray.push_back(&m_multiBodyFrictionContactConstraints[i]);
				m_multiBodyLimitDependencies[dindex++] = m_multiBodyFrictionContactConstraints[i].m_frictionIndex + firstContactConstraintOffset;
			}
		}

		if (!m_multiBodyAllConstraintPtrArray.size())
		{
			m_multiBodyA.resize(0, 0);
			m_multiBodyB.resize(0);
			m_multiBodyX.resize(0);
			m_multiBodyLo.resize(0);
			m_multiBodyHi.resize(0);
		}
	}

	// Construct MLCP terms
	{
		BT_PROFILE("createMLCPFast");
		createMLCPFast(infoGlobal);
	}

	return btScalar(0);
}

btScalar btMultiBodyMLCPConstraintSolver::solveGroupCacheFriendlyIterations(btCollisionObject** bodies, int numBodies, btPersistentManifold** manifoldPtr, int numManifolds, btTypedConstraint** constraints, int numConstraints, const btContactSolverInfo& infoGlobal, btIDebugDraw* debugDrawer)
{
	bool result = true;
	{
		BT_PROFILE("solveMLCP");
		result = solveMLCP(infoGlobal);
	}

	// Fallback to btSequentialImpulseConstraintSolver::solveGroupCacheFriendlyIterations if the solution isn't valid.
	if (!result)
	{
		m_fallback++;
		return btMultiBodyConstraintSolver::solveGroupCacheFriendlyIterations(bodies, numBodies, manifoldPtr, numManifolds, constraints, numConstraints, infoGlobal, debugDrawer);
	}

	{
		BT_PROFILE("process MLCP results");

		for (int i = 0; i < m_allConstraintPtrArray.size(); ++i)
		{
			const btSolverConstraint& c = *m_allConstraintPtrArray[i];

			const btScalar deltaImpulse = m_x[i] - c.m_appliedImpulse;
			c.m_appliedImpulse = m_x[i];

			int sbA = c.m_solverBodyIdA;
			int sbB = c.m_solverBodyIdB;

			btSolverBody& solverBodyA = m_tmpSolverBodyPool[sbA];
			btSolverBody& solverBodyB = m_tmpSolverBodyPool[sbB];

			solverBodyA.internalApplyImpulse(c.m_contactNormal1 * solverBodyA.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
			solverBodyB.internalApplyImpulse(c.m_contactNormal2 * solverBodyB.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);

			if (infoGlobal.m_splitImpulse)
			{
				const btScalar deltaPushImpulse = m_xSplit[i] - c.m_appliedPushImpulse;
				solverBodyA.internalApplyPushImpulse(c.m_contactNormal1 * solverBodyA.internalGetInvMass(), c.m_angularComponentA, deltaPushImpulse);
				solverBodyB.internalApplyPushImpulse(c.m_contactNormal2 * solverBodyB.internalGetInvMass(), c.m_angularComponentB, deltaPushImpulse);
				c.m_appliedPushImpulse = m_xSplit[i];
			}
		}

		for (int i = 0; i < m_multiBodyAllConstraintPtrArray.size(); ++i)
		{
			btMultiBodySolverConstraint& c = *m_multiBodyAllConstraintPtrArray[i];

			const btScalar deltaImpulse = m_multiBodyX[i] - c.m_appliedImpulse;
			c.m_appliedImpulse = m_multiBodyX[i];

			btMultiBody* multiBodyA = c.m_multiBodyA;
			if (multiBodyA)
			{
				const int ndofA = multiBodyA->getNumDofs() + 6;
				applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacAindex], deltaImpulse, c.m_deltaVelAindex, ndofA);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
				//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
				//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
				multiBodyA->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacAindex], deltaImpulse);
#endif  // DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
			}
			else
			{
				const int sbA = c.m_solverBodyIdA;
				btSolverBody& solverBodyA = m_tmpSolverBodyPool[sbA];
				solverBodyA.internalApplyImpulse(c.m_contactNormal1 * solverBodyA.internalGetInvMass(), c.m_angularComponentA, deltaImpulse);
			}

			btMultiBody* multiBodyB = c.m_multiBodyB;
			if (multiBodyB)
			{
				const int ndofB = multiBodyB->getNumDofs() + 6;
				applyDeltaVee(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacBindex], deltaImpulse, c.m_deltaVelBindex, ndofB);
#ifdef DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
				//note: update of the actual velocities (below) in the multibody does not have to happen now since m_deltaVelocities can be applied after all iterations
				//it would make the multibody solver more like the regular one with m_deltaVelocities being equivalent to btSolverBody::m_deltaLinearVelocity/m_deltaAngularVelocity
				multiBodyB->applyDeltaVeeMultiDof2(&m_data.m_deltaVelocitiesUnitImpulse[c.m_jacBindex], deltaImpulse);
#endif  // DIRECTLY_UPDATE_VELOCITY_DURING_SOLVER_ITERATIONS
			}
			else
			{
				const int sbB = c.m_solverBodyIdB;
				btSolverBody& solverBodyB = m_tmpSolverBodyPool[sbB];
				solverBodyB.internalApplyImpulse(c.m_contactNormal2 * solverBodyB.internalGetInvMass(), c.m_angularComponentB, deltaImpulse);
			}
		}
	}

	return btScalar(0);
}

btMultiBodyMLCPConstraintSolver::btMultiBodyMLCPConstraintSolver(btMLCPSolverInterface* solver)
	: m_solver(solver), m_fallback(0)
{
	// Do nothing
}

btMultiBodyMLCPConstraintSolver::~btMultiBodyMLCPConstraintSolver()
{
	// Do nothing
}

void btMultiBodyMLCPConstraintSolver::setMLCPSolver(btMLCPSolverInterface* solver)
{
	m_solver = solver;
}

int btMultiBodyMLCPConstraintSolver::getNumFallbacks() const
{
	return m_fallback;
}

void btMultiBodyMLCPConstraintSolver::setNumFallbacks(int num)
{
	m_fallback = num;
}

btConstraintSolverType btMultiBodyMLCPConstraintSolver::getSolverType() const
{
	return BT_MLCP_SOLVER;
}
