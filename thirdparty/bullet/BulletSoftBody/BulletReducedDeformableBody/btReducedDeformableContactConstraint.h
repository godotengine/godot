#include "../btDeformableContactConstraint.h"
#include "btReducedDeformableBody.h"

// ================= static constraints ===================
class btReducedDeformableStaticConstraint : public btDeformableStaticConstraint
{
 public:
  btReducedDeformableBody* m_rsb;
  btScalar m_dt;
  btVector3 m_ri;
  btVector3 m_targetPos;
  btVector3 m_impulseDirection;
  btMatrix3x3 m_impulseFactorMatrix;
  btScalar m_impulseFactor;
  btScalar m_rhs;
  btScalar m_appliedImpulse;
  btScalar m_erp;

  btReducedDeformableStaticConstraint(btReducedDeformableBody* rsb, 
                                      btSoftBody::Node* node,
                                      const btVector3& ri,
                                      const btVector3& x0,
                                      const btVector3& dir,
                                      const btContactSolverInfo& infoGlobal,
                                      btScalar dt);
	// btReducedDeformableStaticConstraint(const btReducedDeformableStaticConstraint& other);
  btReducedDeformableStaticConstraint() {}
  virtual ~btReducedDeformableStaticConstraint() {}

  virtual btScalar solveConstraint(const btContactSolverInfo& infoGlobal);
  
  // this calls reduced deformable body's applyFullSpaceImpulse
  virtual void applyImpulse(const btVector3& impulse);

  btVector3 getDeltaVa() const;

  // virtual void applySplitImpulse(const btVector3& impulse) {}
};

// ================= base contact constraints ===================
class btReducedDeformableRigidContactConstraint : public btDeformableRigidContactConstraint
{
 public:
  bool m_collideStatic;     // flag for collision with static object
  bool m_collideMultibody;  // flag for collision with multibody

  int m_nodeQueryIndex;
  int m_solverBodyId;       // for debugging

  btReducedDeformableBody* m_rsb;
  btSolverBody* m_solverBody;
  btScalar m_dt;

  btScalar m_appliedNormalImpulse;
  btScalar m_appliedTangentImpulse;
  btScalar m_appliedTangentImpulse2;
  btScalar m_normalImpulseFactor;
  btScalar m_tangentImpulseFactor;
  btScalar m_tangentImpulseFactor2;
  btScalar m_tangentImpulseFactorInv;
  btScalar m_tangentImpulseFactorInv2;
  btScalar m_rhs;
  btScalar m_rhs_tangent;
  btScalar m_rhs_tangent2;
  
  btScalar m_cfm;
  btScalar m_cfm_friction;
  btScalar m_erp;
  btScalar m_erp_friction;
  btScalar m_friction;

  btVector3 m_contactNormalA;     // surface normal for rigid body (opposite direction as impulse)
  btVector3 m_contactNormalB;     // surface normal for reduced deformable body (opposite direction as impulse)
  btVector3 m_contactTangent;     // tangential direction of the relative velocity
  btVector3 m_contactTangent2;    // 2nd tangential direction of the relative velocity
  btVector3 m_relPosA;            // relative position of the contact point for A (rigid)
  btVector3 m_relPosB;            // relative position of the contact point for B
  btMatrix3x3 m_impulseFactor;    // total impulse matrix

  btVector3 m_bufferVelocityA;    // velocity at the beginning of the iteration
  btVector3 m_bufferVelocityB;
  btVector3 m_linearComponentNormal;    // linear components for the solver body
  btVector3 m_angularComponentNormal;   // angular components for the solver body
  // since 2nd contact direction only applies to multibody, these components will never be used
  btVector3 m_linearComponentTangent;
  btVector3 m_angularComponentTangent;

  btReducedDeformableRigidContactConstraint(btReducedDeformableBody* rsb, 
                                            const btSoftBody::DeformableRigidContact& c, 
                                            const btContactSolverInfo& infoGlobal,
                                            btScalar dt);
	// btReducedDeformableRigidContactConstraint(const btReducedDeformableRigidContactConstraint& other);
  btReducedDeformableRigidContactConstraint() {}
  virtual ~btReducedDeformableRigidContactConstraint() {}

  void setSolverBody(const int bodyId, btSolverBody& solver_body);

  virtual void warmStarting() {}

  virtual btScalar solveConstraint(const btContactSolverInfo& infoGlobal);

  void calculateTangentialImpulse(btScalar& deltaImpulse_tangent, 
                                  btScalar& appliedImpulse, 
                                  const btScalar rhs_tangent,
                                  const btScalar tangentImpulseFactorInv,
                                  const btVector3& tangent,
                                  const btScalar lower_limit,
                                  const btScalar upper_limit,
                                  const btVector3& deltaV_rel);

  virtual void applyImpulse(const btVector3& impulse) {}

  virtual void applySplitImpulse(const btVector3& impulse) {} // TODO: may need later

  virtual btVector3 getVa() const;
  virtual btVector3 getDeltaVa() const = 0;
  virtual btVector3 getDeltaVb() const = 0;
};

// ================= node vs rigid constraints ===================
class btReducedDeformableNodeRigidContactConstraint : public btReducedDeformableRigidContactConstraint
{
 public:
  btSoftBody::Node* m_node;

  btReducedDeformableNodeRigidContactConstraint(btReducedDeformableBody* rsb, 
                                                const btSoftBody::DeformableNodeRigidContact& contact, 
                                                const btContactSolverInfo& infoGlobal,
                                                btScalar dt);
	// btReducedDeformableNodeRigidContactConstraint(const btReducedDeformableNodeRigidContactConstraint& other);
  btReducedDeformableNodeRigidContactConstraint() {}
  virtual ~btReducedDeformableNodeRigidContactConstraint() {}

  virtual void warmStarting();

  // get the velocity of the deformable node in contact
	virtual btVector3 getVb() const;

  // get the velocity change of the rigid body
  virtual btVector3 getDeltaVa() const;

  // get velocity change of the node in contat
  virtual btVector3 getDeltaVb() const;

	// get the split impulse velocity of the deformable face at the contact point
	virtual btVector3 getSplitVb() const;

	// get the velocity change of the input soft body node in the constraint
	virtual btVector3 getDv(const btSoftBody::Node*) const;

	// cast the contact to the desired type
	const btSoftBody::DeformableNodeRigidContact* getContact() const
	{
		return static_cast<const btSoftBody::DeformableNodeRigidContact*>(m_contact);
	}
  
  // this calls reduced deformable body's applyFullSpaceImpulse
  virtual void applyImpulse(const btVector3& impulse);
};

// ================= face vs rigid constraints ===================
class btReducedDeformableFaceRigidContactConstraint : public btReducedDeformableRigidContactConstraint
{
 public:
  btSoftBody::Face* m_face;
	bool m_useStrainLimiting;

  btReducedDeformableFaceRigidContactConstraint(btReducedDeformableBody* rsb, 
                                                const btSoftBody::DeformableFaceRigidContact& contact, 
                                                const btContactSolverInfo& infoGlobal,
                                                btScalar dt, 
                                                bool useStrainLimiting);
	// btReducedDeformableFaceRigidContactConstraint(const btReducedDeformableFaceRigidContactConstraint& other);
  btReducedDeformableFaceRigidContactConstraint() {}
  virtual ~btReducedDeformableFaceRigidContactConstraint() {}

  // get the velocity of the deformable face at the contact point
	virtual btVector3 getVb() const;

	// get the split impulse velocity of the deformable face at the contact point
	virtual btVector3 getSplitVb() const;

	// get the velocity change of the input soft body node in the constraint
	virtual btVector3 getDv(const btSoftBody::Node*) const;

	// cast the contact to the desired type
	const btSoftBody::DeformableFaceRigidContact* getContact() const
	{
		return static_cast<const btSoftBody::DeformableFaceRigidContact*>(m_contact);
	}

  // this calls reduced deformable body's applyFullSpaceImpulse
  virtual void applyImpulse(const btVector3& impulse);
};