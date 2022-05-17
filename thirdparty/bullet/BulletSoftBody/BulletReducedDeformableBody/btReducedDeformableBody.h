#ifndef BT_REDUCED_SOFT_BODY_H
#define BT_REDUCED_SOFT_BODY_H

#include "../btSoftBody.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btMatrix3x3.h"
#include "LinearMath/btTransform.h"

// Reduced deformable body is a simplified deformable object embedded in a rigid frame.
class btReducedDeformableBody : public btSoftBody
{
 public:
  //
  //  Typedefs
  //
  typedef btAlignedObjectArray<btVector3> TVStack;
  // typedef btAlignedObjectArray<btMatrix3x3> tBlockDiagMatrix;
  typedef btAlignedObjectArray<btScalar> tDenseArray;
  typedef btAlignedObjectArray<btAlignedObjectArray<btScalar> > tDenseMatrix;

 private:
  // flag to turn off the reduced modes
  bool m_rigidOnly;

  // Flags for transform. Once transform is applied, users cannot scale the mesh or change its total mass.
  bool m_transform_lock;

  // scaling factors
  btScalar m_rhoScale;         // mass density scale
  btScalar m_ksScale;          // stiffness scale

  // projection matrix
  tDenseMatrix m_projPA;        // Eqn. 4.11 from Rahul Sheth's thesis
  tDenseMatrix m_projCq;
  tDenseArray m_STP;
  tDenseArray m_MrInvSTP;

  TVStack m_localMomentArm; // Sq + x0

  btVector3 m_internalDeltaLinearVelocity;
  btVector3 m_internalDeltaAngularVelocity;
  tDenseArray m_internalDeltaReducedVelocity;
  
  btVector3 m_linearVelocityFromReduced;  // contribution to the linear velocity from reduced velocity
  btVector3 m_angularVelocityFromReduced; // contribution to the angular velocity from reduced velocity
  btVector3 m_internalDeltaAngularVelocityFromReduced;

 protected:
  // rigid frame
  btScalar m_mass;          // total mass of the rigid frame
  btScalar m_inverseMass;   // inverse of the total mass of the rigid frame
  btVector3 m_linearVelocity;
  btVector3 m_angularVelocity;
  btScalar m_linearDamping;    // linear damping coefficient
  btScalar m_angularDamping;    // angular damping coefficient
  btVector3 m_linearFactor;
  btVector3 m_angularFactor;
  // btVector3 m_invInertiaLocal;
  btMatrix3x3 m_invInertiaLocal;
  btTransform m_rigidTransformWorld;
  btMatrix3x3 m_invInertiaTensorWorldInitial;
  btMatrix3x3 m_invInertiaTensorWorld;
  btMatrix3x3 m_interpolateInvInertiaTensorWorld;
  btVector3 m_initialCoM;  // initial center of mass (original of the m_rigidTransformWorld)

  // damping
  btScalar m_dampingAlpha;
  btScalar m_dampingBeta;

 public:
  //
  //  Fields
  //

  // reduced space
  int m_nReduced;
  int m_nFull;
  tDenseMatrix m_modes;														// modes of the reduced deformable model. Each inner array is a mode, outer array size = n_modes
  tDenseArray m_reducedDofs;				   // Reduced degree of freedom
  tDenseArray m_reducedDofsBuffer;     // Reduced degree of freedom at t^n
  tDenseArray m_reducedVelocity;		   // Reduced velocity array
  tDenseArray m_reducedVelocityBuffer; // Reduced velocity array at t^n
  tDenseArray m_reducedForceExternal;          // reduced external force
  tDenseArray m_reducedForceElastic;           // reduced internal elastic force
  tDenseArray m_reducedForceDamping;           // reduced internal damping force
  tDenseArray m_eigenvalues;		// eigenvalues of the reduce deformable model
  tDenseArray m_Kr;	// reduced stiffness matrix
  
  // full space
  TVStack m_x0;					     				 // Rest position
  tDenseArray m_nodalMass;           // Mass on each node
  btAlignedObjectArray<int> m_fixedNodes; // index of the fixed nodes
  int m_nodeIndexOffset;             // offset of the node index needed for contact solver when there are multiple reduced deformable body in the world.

  // contacts
  btAlignedObjectArray<int> m_contactNodesList;

  //
  // Api
  //
  btReducedDeformableBody(btSoftBodyWorldInfo* worldInfo, int node_count, const btVector3* x, const btScalar* m);

  ~btReducedDeformableBody() {}

  //
  // initializing helpers
  //
  void internalInitialization();

  void setReducedModes(int num_modes, int full_size);

  void setMassProps(const tDenseArray& mass_array);

  void setInertiaProps();

  void setRigidVelocity(const btVector3& v);

  void setRigidAngularVelocity(const btVector3& omega);

  void setStiffnessScale(const btScalar ks);

  void setMassScale(const btScalar rho);

  void setFixedNodes(const int n_node);

  void setDamping(const btScalar alpha, const btScalar beta);

  void disableReducedModes(const bool rigid_only);

  virtual void setTotalMass(btScalar mass, bool fromfaces = false);

  //
  // various internal updates
  //
  virtual void transformTo(const btTransform& trs);
  virtual void transform(const btTransform& trs);
  // caution: 
  // need to use scale before using transform, because the scale is performed in the local frame 
  // (i.e., may have some rotation already, but the m_rigidTransformWorld doesn't have this info)
  virtual void scale(const btVector3& scl);

 private:
  void updateRestNodalPositions();

  void updateInitialInertiaTensor(const btMatrix3x3& rotation);

  void updateLocalInertiaTensorFromNodes();

  void updateInertiaTensor();

  void updateModesByRotation(const btMatrix3x3& rotation);
 
 public:
  void updateLocalMomentArm();

  void predictIntegratedTransform(btScalar dt, btTransform& predictedTransform);

  // update the external force projection matrix 
  void updateExternalForceProjectMatrix(bool initialized);

  void endOfTimeStepZeroing();

  void applyInternalVelocityChanges();

  //
  // position and velocity update related
  //

  // compute reduced degree of freedoms
  void updateReducedDofs(btScalar solverdt);

  // compute reduced velocity update (for explicit time stepping)
  void updateReducedVelocity(btScalar solverdt);

  // map to full degree of freedoms
  void mapToFullPosition(const btTransform& ref_trans);

  // compute full space velocity from the reduced velocity
  void mapToFullVelocity(const btTransform& ref_trans);

  // compute total angular momentum
  const btVector3 computeTotalAngularMomentum() const;

  // get a single node's full space velocity from the reduced velocity
  const btVector3 computeNodeFullVelocity(const btTransform& ref_trans, int n_node) const;

  // get a single node's all delta velocity
  const btVector3 internalComputeNodeDeltaVelocity(const btTransform& ref_trans, int n_node) const;

  //
  // rigid motion related
  //
  void applyDamping(btScalar timeStep);

  void applyCentralImpulse(const btVector3& impulse);

  void applyTorqueImpulse(const btVector3& torque);

  void proceedToTransform(btScalar dt, bool end_of_time_step);

  //
  // force related
  //

  // apply impulse to the rigid frame
  void internalApplyRigidImpulse(const btVector3& impulse, const btVector3& rel_pos);

  // apply impulse to nodes in the full space
  void internalApplyFullSpaceImpulse(const btVector3& impulse, const btVector3& rel_pos, int n_node, btScalar dt);

  // apply nodal external force in the full space
  void applyFullSpaceNodalForce(const btVector3& f_ext, int n_node);

  // apply gravity to the rigid frame
  void applyRigidGravity(const btVector3& gravity, btScalar dt);

  // apply reduced elastic force
  void applyReducedElasticForce(const tDenseArray& reduce_dofs);

  // apply reduced damping force
  void applyReducedDampingForce(const tDenseArray& reduce_vel);

  // calculate the impulse factor
  virtual btMatrix3x3 getImpulseFactor(int n_node);

  // get relative position from a node to the CoM of the rigid frame
  btVector3 getRelativePos(int n_node);

  //
  // accessors
  //
  bool isReducedModesOFF() const;
  btScalar getTotalMass() const;
  btTransform& getRigidTransform();
  const btVector3& getLinearVelocity() const;
	const btVector3& getAngularVelocity() const;

  #if defined(BT_CLAMP_VELOCITY_TO) && BT_CLAMP_VELOCITY_TO > 0
  void clampVelocity(btVector3& v) const {
      v.setX(
          fmax(-BT_CLAMP_VELOCITY_TO,
                fmin(BT_CLAMP_VELOCITY_TO, v.getX()))
      );
      v.setY(
          fmax(-BT_CLAMP_VELOCITY_TO,
                fmin(BT_CLAMP_VELOCITY_TO, v.getY()))
      );
      v.setZ(
          fmax(-BT_CLAMP_VELOCITY_TO,
                fmin(BT_CLAMP_VELOCITY_TO, v.getZ()))
      );
  }
  #endif
};

#endif // BT_REDUCED_SOFT_BODY_H