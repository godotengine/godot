/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_RIGIDBODY_H
#define BT_RIGIDBODY_H

#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btTransform.h"
#include "BulletCollision/BroadphaseCollision/btBroadphaseProxy.h"
#include "BulletCollision/CollisionDispatch/btCollisionObject.h"

class btCollisionShape;
class btMotionState;
class btTypedConstraint;

extern btScalar gDeactivationTime;
extern bool gDisableDeactivation;

#ifdef BT_USE_DOUBLE_PRECISION
#define btRigidBodyData btRigidBodyDoubleData
#define btRigidBodyDataName "btRigidBodyDoubleData"
#else
#define btRigidBodyData btRigidBodyFloatData
#define btRigidBodyDataName "btRigidBodyFloatData"
#endif  //BT_USE_DOUBLE_PRECISION

enum btRigidBodyFlags
{
	BT_DISABLE_WORLD_GRAVITY = 1,
	///BT_ENABLE_GYROPSCOPIC_FORCE flags is enabled by default in Bullet 2.83 and onwards.
	///and it BT_ENABLE_GYROPSCOPIC_FORCE becomes equivalent to BT_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_BODY
	///See Demos/GyroscopicDemo and computeGyroscopicImpulseImplicit
	BT_ENABLE_GYROSCOPIC_FORCE_EXPLICIT = 2,
	BT_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_WORLD = 4,
	BT_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_BODY = 8,
	BT_ENABLE_GYROPSCOPIC_FORCE = BT_ENABLE_GYROSCOPIC_FORCE_IMPLICIT_BODY,
};

///The btRigidBody is the main class for rigid body objects. It is derived from btCollisionObject, so it keeps a pointer to a btCollisionShape.
///It is recommended for performance and memory use to share btCollisionShape objects whenever possible.
///There are 3 types of rigid bodies:
///- A) Dynamic rigid bodies, with positive mass. Motion is controlled by rigid body dynamics.
///- B) Fixed objects with zero mass. They are not moving (basically collision objects)
///- C) Kinematic objects, which are objects without mass, but the user can move them. There is one-way interaction, and Bullet calculates a velocity based on the timestep and previous and current world transform.
///Bullet automatically deactivates dynamic rigid bodies, when the velocity is below a threshold for a given time.
///Deactivated (sleeping) rigid bodies don't take any processing time, except a minor broadphase collision detection impact (to allow active objects to activate/wake up sleeping objects)
class btRigidBody : public btCollisionObject
{
	btMatrix3x3 m_invInertiaTensorWorld;
	btVector3 m_linearVelocity;
	btVector3 m_angularVelocity;
	btScalar m_inverseMass;
	btVector3 m_linearFactor;

	btVector3 m_gravity;
	btVector3 m_gravity_acceleration;
	btVector3 m_invInertiaLocal;
	btVector3 m_totalForce;
	btVector3 m_totalTorque;

	btScalar m_linearDamping;
	btScalar m_angularDamping;

	bool m_additionalDamping;
	btScalar m_additionalDampingFactor;
	btScalar m_additionalLinearDampingThresholdSqr;
	btScalar m_additionalAngularDampingThresholdSqr;
	btScalar m_additionalAngularDampingFactor;

	btScalar m_linearSleepingThreshold;
	btScalar m_angularSleepingThreshold;

	//m_optionalMotionState allows to automatic synchronize the world transform for active objects
	btMotionState* m_optionalMotionState;

	//keep track of typed constraints referencing this rigid body, to disable collision between linked bodies
	btAlignedObjectArray<btTypedConstraint*> m_constraintRefs;

	int m_rigidbodyFlags;

	int m_debugBodyId;

protected:
	ATTRIBUTE_ALIGNED16(btVector3 m_deltaLinearVelocity);
	btVector3 m_deltaAngularVelocity;
	btVector3 m_angularFactor;
	btVector3 m_invMass;
	btVector3 m_pushVelocity;
	btVector3 m_turnVelocity;

public:
	///The btRigidBodyConstructionInfo structure provides information to create a rigid body. Setting mass to zero creates a fixed (non-dynamic) rigid body.
	///For dynamic objects, you can use the collision shape to approximate the local inertia tensor, otherwise use the zero vector (default argument)
	///You can use the motion state to synchronize the world transform between physics and graphics objects.
	///And if the motion state is provided, the rigid body will initialize its initial world transform from the motion state,
	///m_startWorldTransform is only used when you don't provide a motion state.
	struct btRigidBodyConstructionInfo
	{
		btScalar m_mass;

		///When a motionState is provided, the rigid body will initialize its world transform from the motion state
		///In this case, m_startWorldTransform is ignored.
		btMotionState* m_motionState;
		btTransform m_startWorldTransform;

		btCollisionShape* m_collisionShape;
		btVector3 m_localInertia;
		btScalar m_linearDamping;
		btScalar m_angularDamping;

		///best simulation results when friction is non-zero
		btScalar m_friction;
		///the m_rollingFriction prevents rounded shapes, such as spheres, cylinders and capsules from rolling forever.
		///See Bullet/Demos/RollingFrictionDemo for usage
		btScalar m_rollingFriction;
		btScalar m_spinningFriction;  //torsional friction around contact normal

		///best simulation results using zero restitution.
		btScalar m_restitution;

		btScalar m_linearSleepingThreshold;
		btScalar m_angularSleepingThreshold;

		//Additional damping can help avoiding lowpass jitter motion, help stability for ragdolls etc.
		//Such damping is undesirable, so once the overall simulation quality of the rigid body dynamics system has improved, this should become obsolete
		bool m_additionalDamping;
		btScalar m_additionalDampingFactor;
		btScalar m_additionalLinearDampingThresholdSqr;
		btScalar m_additionalAngularDampingThresholdSqr;
		btScalar m_additionalAngularDampingFactor;

		btRigidBodyConstructionInfo(btScalar mass, btMotionState* motionState, btCollisionShape* collisionShape, const btVector3& localInertia = btVector3(0, 0, 0)) : m_mass(mass),
																																									   m_motionState(motionState),
																																									   m_collisionShape(collisionShape),
																																									   m_localInertia(localInertia),
																																									   m_linearDamping(btScalar(0.)),
																																									   m_angularDamping(btScalar(0.)),
																																									   m_friction(btScalar(0.5)),
																																									   m_rollingFriction(btScalar(0)),
																																									   m_spinningFriction(btScalar(0)),
																																									   m_restitution(btScalar(0.)),
																																									   m_linearSleepingThreshold(btScalar(0.8)),
																																									   m_angularSleepingThreshold(btScalar(1.f)),
																																									   m_additionalDamping(false),
																																									   m_additionalDampingFactor(btScalar(0.005)),
																																									   m_additionalLinearDampingThresholdSqr(btScalar(0.01)),
																																									   m_additionalAngularDampingThresholdSqr(btScalar(0.01)),
																																									   m_additionalAngularDampingFactor(btScalar(0.01))
		{
			m_startWorldTransform.setIdentity();
		}
	};

	///btRigidBody constructor using construction info
	btRigidBody(const btRigidBodyConstructionInfo& constructionInfo);

	///btRigidBody constructor for backwards compatibility.
	///To specify friction (etc) during rigid body construction, please use the other constructor (using btRigidBodyConstructionInfo)
	btRigidBody(btScalar mass, btMotionState* motionState, btCollisionShape* collisionShape, const btVector3& localInertia = btVector3(0, 0, 0));

	virtual ~btRigidBody()
	{
		//No constraints should point to this rigidbody
		//Remove constraints from the dynamics world before you delete the related rigidbodies.
		btAssert(m_constraintRefs.size() == 0);
	}

protected:
	///setupRigidBody is only used internally by the constructor
	void setupRigidBody(const btRigidBodyConstructionInfo& constructionInfo);

public:
	void proceedToTransform(const btTransform& newTrans);

	///to keep collision detection and dynamics separate we don't store a rigidbody pointer
	///but a rigidbody is derived from btCollisionObject, so we can safely perform an upcast
	static const btRigidBody* upcast(const btCollisionObject* colObj)
	{
		if (colObj->getInternalType() & btCollisionObject::CO_RIGID_BODY)
			return (const btRigidBody*)colObj;
		return 0;
	}
	static btRigidBody* upcast(btCollisionObject* colObj)
	{
		if (colObj->getInternalType() & btCollisionObject::CO_RIGID_BODY)
			return (btRigidBody*)colObj;
		return 0;
	}

	/// continuous collision detection needs prediction
	void predictIntegratedTransform(btScalar step, btTransform& predictedTransform);

	void saveKinematicState(btScalar step);

	void applyGravity();

	void setGravity(const btVector3& acceleration);

	const btVector3& getGravity() const
	{
		return m_gravity_acceleration;
	}

	void setDamping(btScalar lin_damping, btScalar ang_damping);

	btScalar getLinearDamping() const
	{
		return m_linearDamping;
	}

	btScalar getAngularDamping() const
	{
		return m_angularDamping;
	}

	btScalar getLinearSleepingThreshold() const
	{
		return m_linearSleepingThreshold;
	}

	btScalar getAngularSleepingThreshold() const
	{
		return m_angularSleepingThreshold;
	}

	void applyDamping(btScalar timeStep);

	SIMD_FORCE_INLINE const btCollisionShape* getCollisionShape() const
	{
		return m_collisionShape;
	}

	SIMD_FORCE_INLINE btCollisionShape* getCollisionShape()
	{
		return m_collisionShape;
	}

	void setMassProps(btScalar mass, const btVector3& inertia);

	const btVector3& getLinearFactor() const
	{
		return m_linearFactor;
	}
	void setLinearFactor(const btVector3& linearFactor)
	{
		m_linearFactor = linearFactor;
		m_invMass = m_linearFactor * m_inverseMass;
	}
	btScalar getInvMass() const { return m_inverseMass; }
	const btMatrix3x3& getInvInertiaTensorWorld() const
	{
		return m_invInertiaTensorWorld;
	}

	void integrateVelocities(btScalar step);

	void setCenterOfMassTransform(const btTransform& xform);

	void applyCentralForce(const btVector3& force)
	{
		m_totalForce += force * m_linearFactor;
	}

	const btVector3& getTotalForce() const
	{
		return m_totalForce;
	};

	const btVector3& getTotalTorque() const
	{
		return m_totalTorque;
	};

	const btVector3& getInvInertiaDiagLocal() const
	{
		return m_invInertiaLocal;
	};

	void setInvInertiaDiagLocal(const btVector3& diagInvInertia)
	{
		m_invInertiaLocal = diagInvInertia;
	}

	void setSleepingThresholds(btScalar linear, btScalar angular)
	{
		m_linearSleepingThreshold = linear;
		m_angularSleepingThreshold = angular;
	}

	void applyTorque(const btVector3& torque)
	{
		m_totalTorque += torque * m_angularFactor;
	}

	void applyForce(const btVector3& force, const btVector3& rel_pos)
	{
		applyCentralForce(force);
		applyTorque(rel_pos.cross(force * m_linearFactor));
	}

	void applyCentralImpulse(const btVector3& impulse)
	{
		m_linearVelocity += impulse * m_linearFactor * m_inverseMass;
	}

	void applyTorqueImpulse(const btVector3& torque)
	{
		m_angularVelocity += m_invInertiaTensorWorld * torque * m_angularFactor;
	}

	void applyImpulse(const btVector3& impulse, const btVector3& rel_pos)
	{
		if (m_inverseMass != btScalar(0.))
		{
			applyCentralImpulse(impulse);
			if (m_angularFactor)
			{
				applyTorqueImpulse(rel_pos.cross(impulse * m_linearFactor));
			}
		}
	}

	void clearForces()
	{
		m_totalForce.setValue(btScalar(0.0), btScalar(0.0), btScalar(0.0));
		m_totalTorque.setValue(btScalar(0.0), btScalar(0.0), btScalar(0.0));
	}

	void updateInertiaTensor();

	const btVector3& getCenterOfMassPosition() const
	{
		return m_worldTransform.getOrigin();
	}
	btQuaternion getOrientation() const;

	const btTransform& getCenterOfMassTransform() const
	{
		return m_worldTransform;
	}
	const btVector3& getLinearVelocity() const
	{
		return m_linearVelocity;
	}
	const btVector3& getAngularVelocity() const
	{
		return m_angularVelocity;
	}

	inline void setLinearVelocity(const btVector3& lin_vel)
	{
		m_updateRevision++;
		m_linearVelocity = lin_vel;
	}

	inline void setAngularVelocity(const btVector3& ang_vel)
	{
		m_updateRevision++;
		m_angularVelocity = ang_vel;
	}

	btVector3 getVelocityInLocalPoint(const btVector3& rel_pos) const
	{
		//we also calculate lin/ang velocity for kinematic objects
		return m_linearVelocity + m_angularVelocity.cross(rel_pos);

		//for kinematic objects, we could also use use:
		//		return 	(m_worldTransform(rel_pos) - m_interpolationWorldTransform(rel_pos)) / m_kinematicTimeStep;
	}

	void translate(const btVector3& v)
	{
		m_worldTransform.getOrigin() += v;
	}

	void getAabb(btVector3& aabbMin, btVector3& aabbMax) const;

	SIMD_FORCE_INLINE btScalar computeImpulseDenominator(const btVector3& pos, const btVector3& normal) const
	{
		btVector3 r0 = pos - getCenterOfMassPosition();

		btVector3 c0 = (r0).cross(normal);

		btVector3 vec = (c0 * getInvInertiaTensorWorld()).cross(r0);

		return m_inverseMass + normal.dot(vec);
	}

	SIMD_FORCE_INLINE btScalar computeAngularImpulseDenominator(const btVector3& axis) const
	{
		btVector3 vec = axis * getInvInertiaTensorWorld();
		return axis.dot(vec);
	}

	SIMD_FORCE_INLINE void updateDeactivation(btScalar timeStep)
	{
		if ((getActivationState() == ISLAND_SLEEPING) || (getActivationState() == DISABLE_DEACTIVATION))
			return;

		if ((getLinearVelocity().length2() < m_linearSleepingThreshold * m_linearSleepingThreshold) &&
			(getAngularVelocity().length2() < m_angularSleepingThreshold * m_angularSleepingThreshold))
		{
			m_deactivationTime += timeStep;
		}
		else
		{
			m_deactivationTime = btScalar(0.);
			setActivationState(0);
		}
	}

	SIMD_FORCE_INLINE bool wantsSleeping()
	{
		if (getActivationState() == DISABLE_DEACTIVATION)
			return false;

		//disable deactivation
		if (gDisableDeactivation || (gDeactivationTime == btScalar(0.)))
			return false;

		if ((getActivationState() == ISLAND_SLEEPING) || (getActivationState() == WANTS_DEACTIVATION))
			return true;

		if (m_deactivationTime > gDeactivationTime)
		{
			return true;
		}
		return false;
	}

	const btBroadphaseProxy* getBroadphaseProxy() const
	{
		return m_broadphaseHandle;
	}
	btBroadphaseProxy* getBroadphaseProxy()
	{
		return m_broadphaseHandle;
	}
	void setNewBroadphaseProxy(btBroadphaseProxy* broadphaseProxy)
	{
		m_broadphaseHandle = broadphaseProxy;
	}

	//btMotionState allows to automatic synchronize the world transform for active objects
	btMotionState* getMotionState()
	{
		return m_optionalMotionState;
	}
	const btMotionState* getMotionState() const
	{
		return m_optionalMotionState;
	}
	void setMotionState(btMotionState* motionState)
	{
		m_optionalMotionState = motionState;
		if (m_optionalMotionState)
			motionState->getWorldTransform(m_worldTransform);
	}

	//for experimental overriding of friction/contact solver func
	int m_contactSolverType;
	int m_frictionSolverType;

	void setAngularFactor(const btVector3& angFac)
	{
		m_updateRevision++;
		m_angularFactor = angFac;
	}

	void setAngularFactor(btScalar angFac)
	{
		m_updateRevision++;
		m_angularFactor.setValue(angFac, angFac, angFac);
	}
	const btVector3& getAngularFactor() const
	{
		return m_angularFactor;
	}

	//is this rigidbody added to a btCollisionWorld/btDynamicsWorld/btBroadphase?
	bool isInWorld() const
	{
		return (getBroadphaseProxy() != 0);
	}

	void addConstraintRef(btTypedConstraint* c);
	void removeConstraintRef(btTypedConstraint* c);

	btTypedConstraint* getConstraintRef(int index)
	{
		return m_constraintRefs[index];
	}

	int getNumConstraintRefs() const
	{
		return m_constraintRefs.size();
	}

	void setFlags(int flags)
	{
		m_rigidbodyFlags = flags;
	}

	int getFlags() const
	{
		return m_rigidbodyFlags;
	}

	///perform implicit force computation in world space
	btVector3 computeGyroscopicImpulseImplicit_World(btScalar dt) const;

	///perform implicit force computation in body space (inertial frame)
	btVector3 computeGyroscopicImpulseImplicit_Body(btScalar step) const;

	///explicit version is best avoided, it gains energy
	btVector3 computeGyroscopicForceExplicit(btScalar maxGyroscopicForce) const;
	btVector3 getLocalInertia() const;

	///////////////////////////////////////////////

	virtual int calculateSerializeBufferSize() const;

	///fills the dataBuffer and returns the struct name (and 0 on failure)
	virtual const char* serialize(void* dataBuffer, class btSerializer* serializer) const;

	virtual void serializeSingleObject(class btSerializer* serializer) const;
};

//@todo add m_optionalMotionState and m_constraintRefs to btRigidBodyData
///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btRigidBodyFloatData
{
	btCollisionObjectFloatData m_collisionObjectData;
	btMatrix3x3FloatData m_invInertiaTensorWorld;
	btVector3FloatData m_linearVelocity;
	btVector3FloatData m_angularVelocity;
	btVector3FloatData m_angularFactor;
	btVector3FloatData m_linearFactor;
	btVector3FloatData m_gravity;
	btVector3FloatData m_gravity_acceleration;
	btVector3FloatData m_invInertiaLocal;
	btVector3FloatData m_totalForce;
	btVector3FloatData m_totalTorque;
	float m_inverseMass;
	float m_linearDamping;
	float m_angularDamping;
	float m_additionalDampingFactor;
	float m_additionalLinearDampingThresholdSqr;
	float m_additionalAngularDampingThresholdSqr;
	float m_additionalAngularDampingFactor;
	float m_linearSleepingThreshold;
	float m_angularSleepingThreshold;
	int m_additionalDamping;
};

///do not change those serialization structures, it requires an updated sBulletDNAstr/sBulletDNAstr64
struct btRigidBodyDoubleData
{
	btCollisionObjectDoubleData m_collisionObjectData;
	btMatrix3x3DoubleData m_invInertiaTensorWorld;
	btVector3DoubleData m_linearVelocity;
	btVector3DoubleData m_angularVelocity;
	btVector3DoubleData m_angularFactor;
	btVector3DoubleData m_linearFactor;
	btVector3DoubleData m_gravity;
	btVector3DoubleData m_gravity_acceleration;
	btVector3DoubleData m_invInertiaLocal;
	btVector3DoubleData m_totalForce;
	btVector3DoubleData m_totalTorque;
	double m_inverseMass;
	double m_linearDamping;
	double m_angularDamping;
	double m_additionalDampingFactor;
	double m_additionalLinearDampingThresholdSqr;
	double m_additionalAngularDampingThresholdSqr;
	double m_additionalAngularDampingFactor;
	double m_linearSleepingThreshold;
	double m_angularSleepingThreshold;
	int m_additionalDamping;
	char m_padding[4];
};

#endif  //BT_RIGIDBODY_H
