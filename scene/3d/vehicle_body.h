#ifndef VEHICLE_BODY_H
#define VEHICLE_BODY_H

#include "scene/3d/physics_body.h"

class VehicleBody;

class VehicleWheel : public Spatial {

	OBJ_TYPE(VehicleWheel,Spatial);

friend class VehicleBody;


	Transform m_worldTransform;
	Transform local_xform;


	Vector3	m_chassisConnectionPointCS; //const
	Vector3	m_wheelDirectionCS;//const
	Vector3	m_wheelAxleCS; // const or modified by steering

	real_t m_suspensionRestLength;
	real_t m_maxSuspensionTravelCm;
	real_t m_wheelRadius;

	real_t m_suspensionStiffness;
	real_t m_wheelsDampingCompression;
	real_t m_wheelsDampingRelaxation;
	real_t m_frictionSlip;
	real_t m_maxSuspensionForce;
	bool m_bIsFrontWheel;

	VehicleBody *body;

//	btVector3	m_wheelAxleCS; // const or modified by steering ?

	real_t	m_steering;
	real_t	m_rotation;
	real_t	m_deltaRotation;
	real_t	m_rollInfluence;
	real_t	m_engineForce;
	real_t	m_brake;

	real_t	m_clippedInvContactDotSuspension;
	real_t	m_suspensionRelativeVelocity;
	//calculated by suspension
	real_t	m_wheelsSuspensionForce;
	real_t	m_skidInfo;


	struct RaycastInfo {
		//set by raycaster
		Vector3	m_contactNormalWS;//contactnormal
		Vector3	m_contactPointWS;//raycast hitpoint
		real_t	m_suspensionLength;
		Vector3	m_hardPointWS;//raycast starting point
		Vector3	m_wheelDirectionWS; //direction in worldspace
		Vector3	m_wheelAxleWS; // axle in worldspace
		bool m_isInContact;
		PhysicsBody* m_groundObject; //could be general void* ptr
	} m_raycastInfo;

	void _update(PhysicsDirectBodyState *s);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:


	VehicleWheel();

};


class VehicleBody : public PhysicsBody {

	OBJ_TYPE(VehicleBody,PhysicsBody);

	real_t mass;
	real_t friction;

	Vector3 linear_velocity;
	Vector3  angular_velocity;
	bool ccd;

	real_t	m_pitchControl;
	real_t	m_steeringValue;
	real_t  m_currentVehicleSpeedKmHour;

	Set<RID> exclude;

	Vector<Vector3>	m_forwardWS;
	Vector<Vector3>	m_axle;
	Vector<real_t>	m_forwardImpulse;
	Vector<real_t>	m_sideImpulse;

	struct btVehicleWheelContactPoint {
		PhysicsDirectBodyState *m_s;
		PhysicsBody* m_body1;
		Vector3	m_frictionPositionWorld;
		Vector3	m_frictionDirectionWorld;
		real_t	m_jacDiagABInv;
		real_t	m_maxImpulse;


		btVehicleWheelContactPoint(PhysicsDirectBodyState *s,PhysicsBody* body1,const Vector3& frictionPosWorld,const Vector3& frictionDirectionWorld, real_t maxImpulse);
	};

	void _resolve_single_bilateral(PhysicsDirectBodyState *s, const Vector3& pos1, PhysicsBody* body2, const Vector3& pos2, const Vector3& normal, real_t& impulse);
	real_t _calc_rolling_friction(btVehicleWheelContactPoint& contactPoint);

	void _update_friction(PhysicsDirectBodyState *s);
	void _update_suspension(PhysicsDirectBodyState *s);
	real_t _ray_cast(int p_idx,PhysicsDirectBodyState *s);
	void _update_wheel_transform(VehicleWheel& wheel ,PhysicsDirectBodyState *s);
	void _update_wheel(int p_idx,PhysicsDirectBodyState *s);



friend class VehicleWheel;
	Vector<VehicleWheel*> wheels;

	static void _bind_methods();

	void _direct_state_changed(Object *p_state);
public:


	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_friction(real_t p_friction);
	real_t get_friction() const;


	VehicleBody();
};

#endif // VEHICLE_BODY_H
