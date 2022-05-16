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

#ifndef BT_DEFORMABLE_BODY_SOLVERS_H
#define BT_DEFORMABLE_BODY_SOLVERS_H

#include "btSoftBodySolvers.h"
#include "btDeformableBackwardEulerObjective.h"
#include "btDeformableMultiBodyDynamicsWorld.h"
#include "BulletDynamics/Featherstone/btMultiBodyLinkCollider.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraint.h"
#include "btConjugateResidual.h"
#include "btConjugateGradient.h"
struct btCollisionObjectWrapper;
// class btDeformableBackwardEulerObjective;
// class btDeformableMultiBodyDynamicsWorld;

class btDeformableBodySolver : public btSoftBodySolver
{
	typedef btAlignedObjectArray<btVector3> TVStack;

protected:
	int m_numNodes;                                                // total number of deformable body nodes
	TVStack m_dv;                                                  // v_{n+1} - v_n
	TVStack m_backup_dv;                                           // backed up dv
	TVStack m_ddv;                                                 // incremental dv
	TVStack m_residual;                                            // rhs of the linear solve
	btAlignedObjectArray<btSoftBody*> m_softBodies;                // all deformable bodies
	TVStack m_backupVelocity;                                      // backed up v, equals v_n for implicit, equals v_{n+1}^* for explicit
	btScalar m_dt;                                                 // dt
	btConjugateGradient<btDeformableBackwardEulerObjective> m_cg;  // CG solver
	btConjugateResidual<btDeformableBackwardEulerObjective> m_cr;  // CR solver
	bool m_implicit;                                               // use implicit scheme if true, explicit scheme if false
	int m_maxNewtonIterations;                                     // max number of newton iterations
	btScalar m_newtonTolerance;                                    // stop newton iterations if f(x) < m_newtonTolerance
	bool m_lineSearch;                                             // If true, use newton's method with line search under implicit scheme
	bool m_reducedSolver;																					 // flag for reduced soft body solver
public:
	// handles data related to objective function
	btDeformableBackwardEulerObjective* m_objective;
	bool m_useProjection;

	btDeformableBodySolver();

	virtual ~btDeformableBodySolver();

	virtual SolverTypes getSolverType() const
	{
		return DEFORMABLE_SOLVER;
	}

	// update soft body normals
	virtual void updateSoftBodies();

	virtual btScalar solveContactConstraints(btCollisionObject** deformableBodies, int numDeformableBodies, const btContactSolverInfo& infoGlobal);

	// solve the momentum equation
	virtual void solveDeformableConstraints(btScalar solverdt);

	// set gravity (get from deformable world)
	virtual void setGravity(const btVector3& gravity)
	{
		// for full deformable object, we don't store gravity in the solver
		// this function is overriden in the reduced deformable object
	}

	// resize/clear data structures
	virtual void reinitialize(const btAlignedObjectArray<btSoftBody*>& softBodies, btScalar dt);

	// set up contact constraints
	virtual void setConstraints(const btContactSolverInfo& infoGlobal);

	// add in elastic forces and gravity to obtain v_{n+1}^* and calls predictDeformableMotion
	virtual void predictMotion(btScalar solverdt);

	// move to temporary position x_{n+1}^* = x_n + dt * v_{n+1}^*
	// x_{n+1}^* is stored in m_q
	void predictDeformableMotion(btSoftBody* psb, btScalar dt);

	// save the current velocity to m_backupVelocity
	void backupVelocity();

	// set m_dv and m_backupVelocity to desired value to prepare for momentum solve
	virtual void setupDeformableSolve(bool implicit);

	// set the current velocity to that backed up in m_backupVelocity
	void revertVelocity();

	// set velocity to m_dv + m_backupVelocity
	void updateVelocity();

	// update the node count
	bool updateNodes();

	// calculate the change in dv resulting from the momentum solve
	void computeStep(TVStack& ddv, const TVStack& residual);

	// calculate the change in dv resulting from the momentum solve when line search is turned on
	btScalar computeDescentStep(TVStack& ddv, const TVStack& residual, bool verbose = false);

	virtual void copySoftBodyToVertexBuffer(const btSoftBody* const softBody, btVertexBufferDescriptor* vertexBuffer) {}

	// process collision between deformable and rigid
	virtual void processCollision(btSoftBody* softBody, const btCollisionObjectWrapper* collisionObjectWrap)
	{
		softBody->defaultCollisionHandler(collisionObjectWrap);
	}

	// process collision between deformable and deformable
	virtual void processCollision(btSoftBody* softBody, btSoftBody* otherSoftBody)
	{
		softBody->defaultCollisionHandler(otherSoftBody);
	}

	// If true, implicit time stepping scheme is used.
	// Otherwise, explicit time stepping scheme is used
	void setImplicit(bool implicit);

	// If true, newton's method with line search is used when implicit time stepping scheme is turned on
	void setLineSearch(bool lineSearch);

	// set temporary position x^* = x_n + dt * v
	// update the deformation gradient at position x^*
	void updateState();

	// set dv = dv + scale * ddv
	void updateDv(btScalar scale = 1);

	// set temporary position x^* = x_n + dt * v^*
	void updateTempPosition();

	// save the current dv to m_backup_dv;
	void backupDv();

	// set dv to the backed-up value
	void revertDv();

	// set dv = dv + scale * ddv
	// set v^* = v_n + dv
	// set temporary position x^* = x_n + dt * v^*
	// update the deformation gradient at position x^*
	void updateEnergy(btScalar scale);

	// calculates the appropriately scaled kinetic energy in the system, which is
	// 1/2 * dv^T * M * dv
	// used in line search
	btScalar kineticEnergy();

	// add explicit force to the velocity in the objective class
	virtual void applyExplicitForce();

	// execute position/velocity update and apply anchor constraints in the integrateTransforms from the Dynamics world
	virtual void applyTransforms(btScalar timeStep);

	virtual void setStrainLimiting(bool opt)
	{
		m_objective->m_projection.m_useStrainLimiting = opt;
	}

	virtual void setPreconditioner(int opt)
	{
		switch (opt)
		{
			case btDeformableBackwardEulerObjective::Mass_preconditioner:
				m_objective->m_preconditioner = m_objective->m_massPreconditioner;
				break;

			case btDeformableBackwardEulerObjective::KKT_preconditioner:
				m_objective->m_preconditioner = m_objective->m_KKTPreconditioner;
				break;
			
			default:
				btAssert(false);
				break;
		}
	}

	virtual btAlignedObjectArray<btDeformableLagrangianForce*>* getLagrangianForceArray()
	{
		return &(m_objective->m_lf);
	}

	virtual const btAlignedObjectArray<btSoftBody::Node*>* getIndices()
	{
		return m_objective->getIndices();
	}

	virtual void setProjection()
	{
		m_objective->m_projection.setProjection();
	}

	virtual void setLagrangeMultiplier()
	{
		m_objective->m_projection.setLagrangeMultiplier();
	}

	virtual bool isReducedSolver()
	{
		return m_reducedSolver;
	}
	
	virtual void deformableBodyInternalWriteBack() {}

	// unused functions
	virtual void optimize(btAlignedObjectArray<btSoftBody*>& softBodies, bool forceUpdate = false) {}
	virtual void solveConstraints(btScalar dt) {}
	virtual bool checkInitialized() { return true; }
	virtual void copyBackToSoftBodies(bool bMove = true) {}
};

#endif /* btDeformableBodySolver_h */
