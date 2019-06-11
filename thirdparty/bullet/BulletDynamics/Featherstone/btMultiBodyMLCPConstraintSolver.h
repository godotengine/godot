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

#ifndef BT_MULTIBODY_MLCP_CONSTRAINT_SOLVER_H
#define BT_MULTIBODY_MLCP_CONSTRAINT_SOLVER_H

#include "LinearMath/btMatrixX.h"
#include "LinearMath/btThreads.h"
#include "BulletDynamics/Featherstone/btMultiBodyConstraintSolver.h"

class btMLCPSolverInterface;
class btMultiBody;

class btMultiBodyMLCPConstraintSolver : public btMultiBodyConstraintSolver
{
protected:
	/// \name MLCP Formulation for Rigid Bodies
	/// \{

	/// A matrix in the MLCP formulation
	btMatrixXu m_A;

	/// b vector in the MLCP formulation.
	btVectorXu m_b;

	/// Constraint impulse, which is an output of MLCP solving.
	btVectorXu m_x;

	/// Lower bound of constraint impulse, \c m_x.
	btVectorXu m_lo;

	/// Upper bound of constraint impulse, \c m_x.
	btVectorXu m_hi;

	/// \}

	/// \name Cache Variables for Split Impulse for Rigid Bodies
	/// When using 'split impulse' we solve two separate (M)LCPs
	/// \{

	/// Split impulse Cache vector corresponding to \c m_b.
	btVectorXu m_bSplit;

	/// Split impulse cache vector corresponding to \c m_x.
	btVectorXu m_xSplit;

	/// \}

	/// \name MLCP Formulation for Multibodies
	/// \{

	/// A matrix in the MLCP formulation
	btMatrixXu m_multiBodyA;

	/// b vector in the MLCP formulation.
	btVectorXu m_multiBodyB;

	/// Constraint impulse, which is an output of MLCP solving.
	btVectorXu m_multiBodyX;

	/// Lower bound of constraint impulse, \c m_x.
	btVectorXu m_multiBodyLo;

	/// Upper bound of constraint impulse, \c m_x.
	btVectorXu m_multiBodyHi;

	/// \}

	/// Indices of normal contact constraint associated with frictional contact constraint for rigid bodies.
	///
	/// This is used by the MLCP solver to update the upper bounds of frictional contact impulse given intermediate
	/// normal contact impulse. For example, i-th element represents the index of a normal constraint that is
	/// accosiated with i-th frictional contact constraint if i-th constraint is a frictional contact constraint.
	/// Otherwise, -1.
	btAlignedObjectArray<int> m_limitDependencies;

	/// Indices of normal contact constraint associated with frictional contact constraint for multibodies.
	///
	/// This is used by the MLCP solver to update the upper bounds of frictional contact impulse given intermediate
	/// normal contact impulse. For example, i-th element represents the index of a normal constraint that is
	/// accosiated with i-th frictional contact constraint if i-th constraint is a frictional contact constraint.
	/// Otherwise, -1.
	btAlignedObjectArray<int> m_multiBodyLimitDependencies;

	/// Array of all the rigid body constraints
	btAlignedObjectArray<btSolverConstraint*> m_allConstraintPtrArray;

	/// Array of all the multibody constraints
	btAlignedObjectArray<btMultiBodySolverConstraint*> m_multiBodyAllConstraintPtrArray;

	/// MLCP solver
	btMLCPSolverInterface* m_solver;

	/// Count of fallbacks of using btSequentialImpulseConstraintSolver, which happens when the MLCP solver fails.
	int m_fallback;

	/// \name MLCP Scratch Variables
	/// The following scratch variables are not stateful -- contents are cleared prior to each use.
	/// They are only cached here to avoid extra memory allocations and deallocations and to ensure
	/// that multiple instances of the solver can be run in parallel.
	///
	/// \{

	/// Cache variable for constraint Jacobian matrix.
	btMatrixXu m_scratchJ3;

	/// Cache variable for constraint Jacobian times inverse mass matrix.
	btMatrixXu m_scratchJInvM3;

	/// Cache variable for offsets.
	btAlignedObjectArray<int> m_scratchOfs;

	/// \}

	/// Constructs MLCP terms, which are \c m_A, \c m_b, \c m_lo, and \c m_hi.
	virtual void createMLCPFast(const btContactSolverInfo& infoGlobal);

	/// Constructs MLCP terms for constraints of two rigid bodies
	void createMLCPFastRigidBody(const btContactSolverInfo& infoGlobal);

	/// Constructs MLCP terms for constraints of two multi-bodies or one rigid body and one multibody
	void createMLCPFastMultiBody(const btContactSolverInfo& infoGlobal);

	/// Solves MLCP and returns the success
	virtual bool solveMLCP(const btContactSolverInfo& infoGlobal);

	// Documentation inherited
	btScalar solveGroupCacheFriendlySetup(
		btCollisionObject** bodies,
		int numBodies,
		btPersistentManifold** manifoldPtr,
		int numManifolds,
		btTypedConstraint** constraints,
		int numConstraints,
		const btContactSolverInfo& infoGlobal,
		btIDebugDraw* debugDrawer) BT_OVERRIDE;

	// Documentation inherited
	btScalar solveGroupCacheFriendlyIterations(
		btCollisionObject** bodies,
		int numBodies,
		btPersistentManifold** manifoldPtr,
		int numManifolds,
		btTypedConstraint** constraints,
		int numConstraints,
		const btContactSolverInfo& infoGlobal,
		btIDebugDraw* debugDrawer) ;

public:
	BT_DECLARE_ALIGNED_ALLOCATOR()

	/// Constructor
	///
	/// \param[in] solver MLCP solver. Assumed it's not null.
	/// \param[in] maxLCPSize Maximum size of LCP to solve using MLCP solver. If the MLCP size exceeds this number, sequaltial impulse method will be used.
	explicit btMultiBodyMLCPConstraintSolver(btMLCPSolverInterface* solver);

	/// Destructor
	virtual ~btMultiBodyMLCPConstraintSolver();

	/// Sets MLCP solver. Assumed it's not null.
	void setMLCPSolver(btMLCPSolverInterface* solver);

	/// Returns the number of fallbacks of using btSequentialImpulseConstraintSolver, which happens when the MLCP
	/// solver fails.
	int getNumFallbacks() const;

	/// Sets the number of fallbacks. This function may be used to reset the number to zero.
	void setNumFallbacks(int num);

	/// Returns the constraint solver type.
	virtual btConstraintSolverType getSolverType() const;
};

#endif  // BT_MULTIBODY_MLCP_CONSTRAINT_SOLVER_H
