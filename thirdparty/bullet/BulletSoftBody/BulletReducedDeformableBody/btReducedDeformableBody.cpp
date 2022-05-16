#include "btReducedDeformableBody.h"
#include "../btSoftBodyInternals.h"
#include "btReducedDeformableBodyHelpers.h"
#include "LinearMath/btTransformUtil.h"
#include <iostream>
#include <fstream>

btReducedDeformableBody::btReducedDeformableBody(btSoftBodyWorldInfo* worldInfo, int node_count, const btVector3* x, const btScalar* m)
 : btSoftBody(worldInfo, node_count, x, m), m_rigidOnly(false)
{
  // reduced deformable
  m_reducedModel = true;
  m_nReduced = 0;
  m_nFull = 0;
  m_nodeIndexOffset = 0;

  m_transform_lock = false;
  m_ksScale = 1.0;
  m_rhoScale = 1.0;

  // rigid motion
  m_linearVelocity.setZero();
	m_angularVelocity.setZero();
  m_internalDeltaLinearVelocity.setZero();
  m_internalDeltaAngularVelocity.setZero();
  m_angularVelocityFromReduced.setZero();
  m_internalDeltaAngularVelocityFromReduced.setZero();
	m_angularFactor.setValue(1, 1, 1);
	m_linearFactor.setValue(1, 1, 1);
  // m_invInertiaLocal.setValue(1, 1, 1);
  m_invInertiaLocal.setIdentity();
  m_mass = 0.0;
  m_inverseMass = 0.0;

  m_linearDamping = 0;
  m_angularDamping = 0;

  // Rayleigh damping
  m_dampingAlpha = 0;
  m_dampingBeta = 0;

  m_rigidTransformWorld.setIdentity();
}

void btReducedDeformableBody::setReducedModes(int num_modes, int full_size)
{
  m_nReduced = num_modes;
  m_nFull = full_size;
  m_reducedDofs.resize(m_nReduced, 0);
  m_reducedDofsBuffer.resize(m_nReduced, 0);
  m_reducedVelocity.resize(m_nReduced, 0);
  m_reducedVelocityBuffer.resize(m_nReduced, 0);
  m_reducedForceElastic.resize(m_nReduced, 0);
  m_reducedForceDamping.resize(m_nReduced, 0);
  m_reducedForceExternal.resize(m_nReduced, 0);
  m_internalDeltaReducedVelocity.resize(m_nReduced, 0);
  m_nodalMass.resize(full_size, 0);
  m_localMomentArm.resize(m_nFull);
}

void btReducedDeformableBody::setMassProps(const tDenseArray& mass_array)
{
  btScalar total_mass = 0;
  btVector3 CoM(0, 0, 0);
	for (int i = 0; i < m_nFull; ++i)
	{
		m_nodalMass[i] = m_rhoScale * mass_array[i];
		m_nodes[i].m_im = mass_array[i] > 0 ? 1.0 / (m_rhoScale * mass_array[i]) : 0;
		total_mass += m_rhoScale * mass_array[i];

    CoM += m_nodalMass[i] * m_nodes[i].m_x;
	}
  // total rigid body mass
  m_mass = total_mass;
  m_inverseMass = total_mass > 0 ? 1.0 / total_mass : 0;
  // original CoM
  m_initialCoM = CoM / total_mass;
}

void btReducedDeformableBody::setInertiaProps()
{
  // make sure the initial CoM is at the origin (0,0,0)
  // for (int i = 0; i < m_nFull; ++i)
  // {
  //   m_nodes[i].m_x -= m_initialCoM;
  // }
  // m_initialCoM.setZero();
  m_rigidTransformWorld.setOrigin(m_initialCoM);
  m_interpolationWorldTransform = m_rigidTransformWorld;
  
  updateLocalInertiaTensorFromNodes();

  // update world inertia tensor
  btMatrix3x3 rotation;
  rotation.setIdentity();
  updateInitialInertiaTensor(rotation);
  updateInertiaTensor();
  m_interpolateInvInertiaTensorWorld = m_invInertiaTensorWorld;
}

void btReducedDeformableBody::setRigidVelocity(const btVector3& v)
{
  m_linearVelocity = v;
}

void btReducedDeformableBody::setRigidAngularVelocity(const btVector3& omega)
{
  m_angularVelocity = omega;
}

void btReducedDeformableBody::setStiffnessScale(const btScalar ks)
{
  m_ksScale = ks;
}

void btReducedDeformableBody::setMassScale(const btScalar rho)
{
  m_rhoScale = rho;
}

void btReducedDeformableBody::setFixedNodes(const int n_node)
{
  m_fixedNodes.push_back(n_node);
  m_nodes[n_node].m_im = 0;   // set inverse mass to be zero for the constraint solver.
}

void btReducedDeformableBody::setDamping(const btScalar alpha, const btScalar beta)
{
  m_dampingAlpha = alpha;
  m_dampingBeta = beta;
}

void btReducedDeformableBody::internalInitialization()
{
  // zeroing
  endOfTimeStepZeroing();
  // initialize rest position
  updateRestNodalPositions();
  // initialize local nodal moment arm form the CoM
  updateLocalMomentArm();
  // initialize projection matrix
  updateExternalForceProjectMatrix(false);
}

void btReducedDeformableBody::updateLocalMomentArm()
{
  TVStack delta_x;
  delta_x.resize(m_nFull);

  for (int i = 0; i < m_nFull; ++i)
  {
    for (int k = 0; k < 3; ++k)
    {
      // compute displacement
      delta_x[i][k] = 0;
      for (int j = 0; j < m_nReduced; ++j) 
      {
        delta_x[i][k] += m_modes[j][3 * i + k] * m_reducedDofs[j];
      }
    }
    // get new moment arm Sq + x0
    m_localMomentArm[i] = m_x0[i] - m_initialCoM + delta_x[i];
  }
}

void btReducedDeformableBody::updateExternalForceProjectMatrix(bool initialized)
{
  // if not initialized, need to compute both P_A and Cq
  // otherwise, only need to udpate Cq
  if (!initialized)
  {
    // resize
    m_projPA.resize(m_nReduced);
    m_projCq.resize(m_nReduced);

    m_STP.resize(m_nReduced);
    m_MrInvSTP.resize(m_nReduced);

    // P_A
    for (int r = 0; r < m_nReduced; ++r)
    {
      m_projPA[r].resize(3 * m_nFull, 0);
      for (int i = 0; i < m_nFull; ++i)
      {
        btMatrix3x3 mass_scaled_i = Diagonal(1) - Diagonal(m_nodalMass[i] / m_mass);
        btVector3 s_ri(m_modes[r][3 * i], m_modes[r][3 * i + 1], m_modes[r][3 * i + 2]);
        btVector3 prod_i = mass_scaled_i * s_ri;

        for (int k = 0; k < 3; ++k)
          m_projPA[r][3 * i + k] = prod_i[k];

        // btScalar ratio = m_nodalMass[i] / m_mass;
        // m_projPA[r] += btVector3(- m_modes[r][3 * i] * ratio,
        //                          - m_modes[r][3 * i + 1] * ratio,
        //                          - m_modes[r][3 * i + 2] * ratio);
      }
    }
  }

  // C(q) is updated once per position update
  for (int r = 0; r < m_nReduced; ++r)
  {
  	m_projCq[r].resize(3 * m_nFull, 0);
    for (int i = 0; i < m_nFull; ++i)
    {
      btMatrix3x3 r_star = Cross(m_localMomentArm[i]);
      btVector3 s_ri(m_modes[r][3 * i], m_modes[r][3 * i + 1], m_modes[r][3 * i + 2]);
      btVector3 prod_i = r_star * m_invInertiaTensorWorld * r_star * s_ri;

      for (int k = 0; k < 3; ++k)
        m_projCq[r][3 * i + k] = m_nodalMass[i] * prod_i[k];

      // btVector3 si(m_modes[r][3 * i], m_modes[r][3 * i + 1], m_modes[r][3 * i + 2]);
      // m_projCq[r] += m_nodalMass[i] * si.cross(m_localMomentArm[i]);
    }
  }
}

void btReducedDeformableBody::endOfTimeStepZeroing()
{
  for (int i = 0; i < m_nReduced; ++i)
  {
    m_reducedForceElastic[i] = 0;
    m_reducedForceDamping[i] = 0;
    m_reducedForceExternal[i] = 0;
    m_internalDeltaReducedVelocity[i] = 0;
    m_reducedDofsBuffer[i] = m_reducedDofs[i];
    m_reducedVelocityBuffer[i] = m_reducedVelocity[i];
  }
  // std::cout << "zeroed!\n";
}

void btReducedDeformableBody::applyInternalVelocityChanges()
{
  m_linearVelocity += m_internalDeltaLinearVelocity;
  m_angularVelocity += m_internalDeltaAngularVelocity;
  m_internalDeltaLinearVelocity.setZero();
  m_internalDeltaAngularVelocity.setZero();
  for (int r = 0; r < m_nReduced; ++r)
  {
    m_reducedVelocity[r] += m_internalDeltaReducedVelocity[r];
    m_internalDeltaReducedVelocity[r] = 0;
  }
}

void btReducedDeformableBody::predictIntegratedTransform(btScalar dt, btTransform& predictedTransform)
{
	btTransformUtil::integrateTransform(m_rigidTransformWorld, m_linearVelocity, m_angularVelocity, dt, predictedTransform);
}

void btReducedDeformableBody::updateReducedDofs(btScalar solverdt)
{
  for (int r = 0; r < m_nReduced; ++r)
  { 
    m_reducedDofs[r] = m_reducedDofsBuffer[r] + solverdt * m_reducedVelocity[r];
  }
}

void btReducedDeformableBody::mapToFullPosition(const btTransform& ref_trans)
{
  btVector3 origin = ref_trans.getOrigin();
  btMatrix3x3 rotation = ref_trans.getBasis();
  

  for (int i = 0; i < m_nFull; ++i)
  {
    m_nodes[i].m_x = rotation * m_localMomentArm[i] + origin;
    m_nodes[i].m_q = m_nodes[i].m_x;
  }
}

void btReducedDeformableBody::updateReducedVelocity(btScalar solverdt)
{
  // update reduced velocity
  for (int r = 0; r < m_nReduced; ++r)
  {
    // the reduced mass is always identity!
    btScalar delta_v = 0;
    delta_v = solverdt * (m_reducedForceElastic[r] + m_reducedForceDamping[r]);
    // delta_v = solverdt * (m_reducedForceElastic[r] + m_reducedForceDamping[r] + m_reducedForceExternal[r]);
    m_reducedVelocity[r] = m_reducedVelocityBuffer[r] + delta_v;
  }
}

void btReducedDeformableBody::mapToFullVelocity(const btTransform& ref_trans)
{
  // compute the reduced contribution to the angular velocity
  // btVector3 sum_linear(0, 0, 0);
  // btVector3 sum_angular(0, 0, 0);
  // m_linearVelocityFromReduced.setZero();
  // m_angularVelocityFromReduced.setZero();
  // for (int i = 0; i < m_nFull; ++i)
  // {
  //   btVector3 r_com = ref_trans.getBasis() * m_localMomentArm[i];
  //   btMatrix3x3 r_star = Cross(r_com);

  //   btVector3 v_from_reduced(0, 0, 0);
  //   for (int k = 0; k < 3; ++k)
  //   {
  //     for (int r = 0; r < m_nReduced; ++r)
  //     {
  //       v_from_reduced[k] += m_modes[r][3 * i + k] * m_reducedVelocity[r];
  //     }
  //   }

  //   btVector3 delta_linear = m_nodalMass[i] * v_from_reduced;
  //   btVector3 delta_angular = m_nodalMass[i] * (r_star * ref_trans.getBasis() * v_from_reduced);
  //   sum_linear += delta_linear;
  //   sum_angular += delta_angular;
  //   // std::cout << "delta_linear: " << delta_linear[0] << "\t" << delta_linear[1] << "\t" << delta_linear[2] << "\n";
  //   // std::cout << "delta_angular: " << delta_angular[0] << "\t" << delta_angular[1] << "\t" << delta_angular[2] << "\n";
  //   // std::cout << "sum_linear: " << sum_linear[0] << "\t" << sum_linear[1] << "\t" << sum_linear[2] << "\n";
  //   // std::cout << "sum_angular: " << sum_angular[0] << "\t" << sum_angular[1] << "\t" << sum_angular[2] << "\n";
  // }
  // m_linearVelocityFromReduced = 1.0 / m_mass * (ref_trans.getBasis() * sum_linear);
  // m_angularVelocityFromReduced = m_interpolateInvInertiaTensorWorld * sum_angular;

  // m_linearVelocity -= m_linearVelocityFromReduced;
  // m_angularVelocity -= m_angularVelocityFromReduced;

  for (int i = 0; i < m_nFull; ++i)
  {
    m_nodes[i].m_v = computeNodeFullVelocity(ref_trans, i);
  }
}

const btVector3 btReducedDeformableBody::computeTotalAngularMomentum() const
{
  btVector3 L_rigid = m_invInertiaTensorWorld.inverse() * m_angularVelocity;
  btVector3 L_reduced(0, 0, 0);
  btMatrix3x3 omega_prime_star = Cross(m_angularVelocityFromReduced);

  for (int i = 0; i < m_nFull; ++i)
  {
    btVector3 r_com = m_rigidTransformWorld.getBasis() * m_localMomentArm[i];
    btMatrix3x3 r_star = Cross(r_com);

    btVector3 v_from_reduced(0, 0, 0);
    for (int k = 0; k < 3; ++k)
    {
      for (int r = 0; r < m_nReduced; ++r)
      {
        v_from_reduced[k] += m_modes[r][3 * i + k] * m_reducedVelocity[r];
      }
    }

    L_reduced += m_nodalMass[i] * (r_star * (m_rigidTransformWorld.getBasis() * v_from_reduced - omega_prime_star * r_com));
    // L_reduced += m_nodalMass[i] * (r_star * (m_rigidTransformWorld.getBasis() * v_from_reduced));
  }
  return L_rigid + L_reduced;
}

const btVector3 btReducedDeformableBody::computeNodeFullVelocity(const btTransform& ref_trans, int n_node) const
{
  btVector3 v_from_reduced(0, 0, 0);
  btVector3 r_com = ref_trans.getBasis() * m_localMomentArm[n_node];
  // compute velocity contributed by the reduced velocity
  for (int k = 0; k < 3; ++k)
  {
    for (int r = 0; r < m_nReduced; ++r)
    {
      v_from_reduced[k] += m_modes[r][3 * n_node + k] * m_reducedVelocity[r];
    }
  }
  // get new velocity
  btVector3 vel = m_angularVelocity.cross(r_com) + 
                  ref_trans.getBasis() * v_from_reduced +
                  m_linearVelocity;
  return vel;
}

const btVector3 btReducedDeformableBody::internalComputeNodeDeltaVelocity(const btTransform& ref_trans, int n_node) const
{
  btVector3 deltaV_from_reduced(0, 0, 0);
  btVector3 r_com = ref_trans.getBasis() * m_localMomentArm[n_node];

  // compute velocity contributed by the reduced velocity
  for (int k = 0; k < 3; ++k)
  {
    for (int r = 0; r < m_nReduced; ++r)
    {
      deltaV_from_reduced[k] += m_modes[r][3 * n_node + k] * m_internalDeltaReducedVelocity[r];
    }
  }

  // get delta velocity
  btVector3 deltaV = m_internalDeltaAngularVelocity.cross(r_com) + 
                     ref_trans.getBasis() * deltaV_from_reduced +
                     m_internalDeltaLinearVelocity;
  return deltaV;
}

void btReducedDeformableBody::proceedToTransform(btScalar dt, bool end_of_time_step)
{
  btTransformUtil::integrateTransform(m_rigidTransformWorld, m_linearVelocity, m_angularVelocity, dt, m_interpolationWorldTransform);
  updateInertiaTensor();
  // m_interpolateInvInertiaTensorWorld = m_interpolationWorldTransform.getBasis().scaled(m_invInertiaLocal) * m_interpolationWorldTransform.getBasis().transpose();
  m_rigidTransformWorld = m_interpolationWorldTransform;
  m_invInertiaTensorWorld = m_interpolateInvInertiaTensorWorld;
}

void btReducedDeformableBody::transformTo(const btTransform& trs)
{
	btTransform current_transform = getRigidTransform();
	btTransform new_transform(trs.getBasis() * current_transform.getBasis().transpose(),
                            trs.getOrigin() - current_transform.getOrigin());
  transform(new_transform);
}

void btReducedDeformableBody::transform(const btTransform& trs)
{
  m_transform_lock = true;

  // transform mesh
  {
    const btScalar margin = getCollisionShape()->getMargin();
    ATTRIBUTE_ALIGNED16(btDbvtVolume)
    vol;

    btVector3 CoM = m_rigidTransformWorld.getOrigin();
    btVector3 translation = trs.getOrigin();
    btMatrix3x3 rotation = trs.getBasis();

    for (int i = 0; i < m_nodes.size(); ++i)
    {
      Node& n = m_nodes[i];
      n.m_x = rotation * (n.m_x - CoM) + CoM + translation;
      n.m_q = rotation * (n.m_q - CoM) + CoM + translation;
      n.m_n = rotation * n.m_n;
      vol = btDbvtVolume::FromCR(n.m_x, margin);

      m_ndbvt.update(n.m_leaf, vol);
    }
    updateNormals();
    updateBounds();
    updateConstants();
  }

  // update modes
  updateModesByRotation(trs.getBasis());

  // update inertia tensor
  updateInitialInertiaTensor(trs.getBasis());
  updateInertiaTensor();
  m_interpolateInvInertiaTensorWorld = m_invInertiaTensorWorld;
  
  // update rigid frame (No need to update the rotation. Nodes have already been updated.)
  m_rigidTransformWorld.setOrigin(m_initialCoM + trs.getOrigin());
  m_interpolationWorldTransform = m_rigidTransformWorld;
  m_initialCoM = m_rigidTransformWorld.getOrigin();

  internalInitialization();
}

void btReducedDeformableBody::scale(const btVector3& scl)
{
  // Scaling the mesh after transform is applied is not allowed
  btAssert(!m_transform_lock);

  // scale the mesh
  {
    const btScalar margin = getCollisionShape()->getMargin();
    ATTRIBUTE_ALIGNED16(btDbvtVolume)
    vol;

    btVector3 CoM = m_rigidTransformWorld.getOrigin();

    for (int i = 0; i < m_nodes.size(); ++i)
    {
      Node& n = m_nodes[i];
      n.m_x = (n.m_x - CoM) * scl + CoM;
      n.m_q = (n.m_q - CoM) * scl + CoM;
      vol = btDbvtVolume::FromCR(n.m_x, margin);
      m_ndbvt.update(n.m_leaf, vol);
    }
    updateNormals();
    updateBounds();
    updateConstants();
    initializeDmInverse();
  }

  // update inertia tensor
  updateLocalInertiaTensorFromNodes();

  btMatrix3x3 id;
  id.setIdentity();
  updateInitialInertiaTensor(id);   // there is no rotation, but the local inertia tensor has changed
  updateInertiaTensor();
  m_interpolateInvInertiaTensorWorld = m_invInertiaTensorWorld;

  internalInitialization();
}

void btReducedDeformableBody::setTotalMass(btScalar mass, bool fromfaces)
{
  // Changing the total mass after transform is applied is not allowed
  btAssert(!m_transform_lock);

  btScalar scale_ratio = mass / m_mass;

  // update nodal mass
  for (int i = 0; i < m_nFull; ++i)
  {
    m_nodalMass[i] *= scale_ratio;
  }
  m_mass = mass;
  m_inverseMass = mass > 0 ? 1.0 / mass : 0;

  // update inertia tensors
  updateLocalInertiaTensorFromNodes();

  btMatrix3x3 id;
  id.setIdentity();
  updateInitialInertiaTensor(id);   // there is no rotation, but the local inertia tensor has changed
  updateInertiaTensor();
  m_interpolateInvInertiaTensorWorld = m_invInertiaTensorWorld;

  internalInitialization();
}

void btReducedDeformableBody::updateRestNodalPositions()
{
  // update reset nodal position
  m_x0.resize(m_nFull);
  for (int i = 0; i < m_nFull; ++i)
  {
    m_x0[i] = m_nodes[i].m_x;
  }
}

// reference notes:
// https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-07-dynamics-fall-2009/lecture-notes/MIT16_07F09_Lec26.pdf
void btReducedDeformableBody::updateLocalInertiaTensorFromNodes()
{
  btMatrix3x3 inertia_tensor;
  inertia_tensor.setZero();

  for (int p = 0; p < m_nFull; ++p)
  {
    btMatrix3x3 particle_inertia;
    particle_inertia.setZero();

    btVector3 r = m_nodes[p].m_x - m_initialCoM;

    particle_inertia[0][0] = m_nodalMass[p] * (r[1] * r[1] + r[2] * r[2]);
    particle_inertia[1][1] = m_nodalMass[p] * (r[0] * r[0] + r[2] * r[2]);
    particle_inertia[2][2] = m_nodalMass[p] * (r[0] * r[0] + r[1] * r[1]);

    particle_inertia[0][1] = - m_nodalMass[p] * (r[0] * r[1]);
    particle_inertia[0][2] = - m_nodalMass[p] * (r[0] * r[2]);
    particle_inertia[1][2] = - m_nodalMass[p] * (r[1] * r[2]);

    particle_inertia[1][0] = particle_inertia[0][1];
    particle_inertia[2][0] = particle_inertia[0][2];
    particle_inertia[2][1] = particle_inertia[1][2];

    inertia_tensor += particle_inertia;
  }
  m_invInertiaLocal = inertia_tensor.inverse();
}

void btReducedDeformableBody::updateInitialInertiaTensor(const btMatrix3x3& rotation)
{
  // m_invInertiaTensorWorldInitial = rotation.scaled(m_invInertiaLocal) * rotation.transpose();
  m_invInertiaTensorWorldInitial = rotation * m_invInertiaLocal * rotation.transpose();
}

void btReducedDeformableBody::updateModesByRotation(const btMatrix3x3& rotation)
{
  for (int r = 0; r < m_nReduced; ++r)
  {
    for (int i = 0; i < m_nFull; ++i)
    {
      btVector3 nodal_disp(m_modes[r][3 * i], m_modes[r][3 * i + 1], m_modes[r][3 * i + 2]);
      nodal_disp = rotation * nodal_disp;

      for (int k = 0; k < 3; ++k)
      {
        m_modes[r][3 * i + k] = nodal_disp[k];
      }
    }
  }
}

void btReducedDeformableBody::updateInertiaTensor()
{
	m_invInertiaTensorWorld = m_rigidTransformWorld.getBasis() * m_invInertiaTensorWorldInitial * m_rigidTransformWorld.getBasis().transpose();
}

void btReducedDeformableBody::applyDamping(btScalar timeStep)
{
  m_linearVelocity *= btScalar(1) - m_linearDamping;
  m_angularDamping *= btScalar(1) - m_angularDamping;
}

void btReducedDeformableBody::applyCentralImpulse(const btVector3& impulse)
{
  m_linearVelocity += impulse * m_linearFactor * m_inverseMass;
  #if defined(BT_CLAMP_VELOCITY_TO) && BT_CLAMP_VELOCITY_TO > 0
  clampVelocity(m_linearVelocity);
  #endif
}

void btReducedDeformableBody::applyTorqueImpulse(const btVector3& torque)
{
  m_angularVelocity += m_interpolateInvInertiaTensorWorld * torque * m_angularFactor;
  #if defined(BT_CLAMP_VELOCITY_TO) && BT_CLAMP_VELOCITY_TO > 0
  clampVelocity(m_angularVelocity);
  #endif
}

void btReducedDeformableBody::internalApplyRigidImpulse(const btVector3& impulse, const btVector3& rel_pos)
{
  if (m_inverseMass == btScalar(0.))
  {
    std::cout << "something went wrong...probably didn't initialize?\n";
    btAssert(false);
  }
  // delta linear velocity
  m_internalDeltaLinearVelocity += impulse * m_linearFactor * m_inverseMass;
  // delta angular velocity
  btVector3 torque = rel_pos.cross(impulse * m_linearFactor);
  m_internalDeltaAngularVelocity += m_interpolateInvInertiaTensorWorld * torque * m_angularFactor;
}

btVector3 btReducedDeformableBody::getRelativePos(int n_node)
{
  btMatrix3x3 rotation = m_interpolationWorldTransform.getBasis();
  btVector3 ri = rotation * m_localMomentArm[n_node];
  return ri;
}

btMatrix3x3 btReducedDeformableBody::getImpulseFactor(int n_node)
{
  // relative position
  btMatrix3x3 rotation = m_interpolationWorldTransform.getBasis();
  btVector3 ri = rotation * m_localMomentArm[n_node];
  btMatrix3x3 ri_skew = Cross(ri);

  // calculate impulse factor
  // rigid part
  btScalar inv_mass = m_nodalMass[n_node] > btScalar(0) ? btScalar(1) / m_mass : btScalar(0);
  btMatrix3x3 K1 = Diagonal(inv_mass);
  K1 -= ri_skew * m_interpolateInvInertiaTensorWorld * ri_skew;

  // reduced deformable part
  btMatrix3x3 SA;
  SA.setZero();
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int r = 0; r < m_nReduced; ++r)
      {
        SA[i][j] += m_modes[r][3 * n_node + i] * (m_projPA[r][3 * n_node + j] + m_projCq[r][3 * n_node + j]);
      }
    }
  }
  btMatrix3x3 RSARinv = rotation * SA * rotation.transpose();


  TVStack omega_helper; // Sum_i m_i r*_i R S_i
  omega_helper.resize(m_nReduced);
  for (int r = 0; r < m_nReduced; ++r)
  {
    omega_helper[r].setZero();
    for (int i = 0; i < m_nFull; ++i)
    {
      btMatrix3x3 mi_rstar_i = rotation * Cross(m_localMomentArm[i]) * m_nodalMass[i];
      btVector3 s_ri(m_modes[r][3 * i], m_modes[r][3 * i + 1], m_modes[r][3 * i + 2]);
      omega_helper[r] += mi_rstar_i * rotation * s_ri;
    }
  }

  btMatrix3x3 sum_multiply_A;
  sum_multiply_A.setZero();
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int r = 0; r < m_nReduced; ++r)
      {
        sum_multiply_A[i][j] += omega_helper[r][i] * (m_projPA[r][3 * n_node + j] + m_projCq[r][3 * n_node + j]);
      }
    }
  }

  btMatrix3x3 K2 = RSARinv + ri_skew * m_interpolateInvInertiaTensorWorld * sum_multiply_A * rotation.transpose();

  return m_rigidOnly ? K1 : K1 + K2;
}

void btReducedDeformableBody::internalApplyFullSpaceImpulse(const btVector3& impulse, const btVector3& rel_pos, int n_node, btScalar dt)
{
  if (!m_rigidOnly)
  {
    // apply impulse force
    applyFullSpaceNodalForce(impulse / dt, n_node);

    // update delta damping force
    tDenseArray reduced_vel_tmp;
    reduced_vel_tmp.resize(m_nReduced);
    for (int r = 0; r < m_nReduced; ++r)
    {
      reduced_vel_tmp[r] = m_reducedVelocity[r] + m_internalDeltaReducedVelocity[r];
    }
    applyReducedDampingForce(reduced_vel_tmp);
    // applyReducedDampingForce(m_internalDeltaReducedVelocity);

    // delta reduced velocity
    for (int r = 0; r < m_nReduced; ++r)
    {
      // The reduced mass is always identity!
      m_internalDeltaReducedVelocity[r] += dt * (m_reducedForceDamping[r] + m_reducedForceExternal[r]);
    }
  }

  internalApplyRigidImpulse(impulse, rel_pos);
}

void btReducedDeformableBody::applyFullSpaceNodalForce(const btVector3& f_ext, int n_node)
{
  // f_local = R^-1 * f_ext //TODO: interpoalted transfrom
  // btVector3 f_local = m_rigidTransformWorld.getBasis().transpose() * f_ext;
  btVector3 f_local = m_interpolationWorldTransform.getBasis().transpose() * f_ext;

  // f_ext_r = [S^T * P]_{n_node} * f_local
  tDenseArray f_ext_r;
  f_ext_r.resize(m_nReduced, 0);
  for (int r = 0; r < m_nReduced; ++r)
  {
    m_reducedForceExternal[r] = 0;
    for (int k = 0; k < 3; ++k)
    {
      f_ext_r[r] += (m_projPA[r][3 * n_node + k] + m_projCq[r][3 * n_node + k]) * f_local[k];
    }

    m_reducedForceExternal[r] += f_ext_r[r];
  }
}

void btReducedDeformableBody::applyRigidGravity(const btVector3& gravity, btScalar dt)
{
  // update rigid frame velocity
  m_linearVelocity += dt * gravity;
}

void btReducedDeformableBody::applyReducedElasticForce(const tDenseArray& reduce_dofs)
{
  for (int r = 0; r < m_nReduced; ++r) 
  {
    m_reducedForceElastic[r] = - m_ksScale * m_Kr[r] * reduce_dofs[r];
  }
}

void btReducedDeformableBody::applyReducedDampingForce(const tDenseArray& reduce_vel)
{
  for (int r = 0; r < m_nReduced; ++r) 
  {
    m_reducedForceDamping[r] = - m_dampingBeta * m_ksScale * m_Kr[r] * reduce_vel[r];
  }
}

btScalar btReducedDeformableBody::getTotalMass() const
{
  return m_mass;
}

btTransform& btReducedDeformableBody::getRigidTransform()
{
  return m_rigidTransformWorld;
}

const btVector3& btReducedDeformableBody::getLinearVelocity() const
{
  return m_linearVelocity;
}

const btVector3& btReducedDeformableBody::getAngularVelocity() const
{
  return m_angularVelocity;
}

void btReducedDeformableBody::disableReducedModes(const bool rigid_only)
{
  m_rigidOnly = rigid_only;
}

bool btReducedDeformableBody::isReducedModesOFF() const
{
  return m_rigidOnly;
}