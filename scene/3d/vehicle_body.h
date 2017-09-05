/*************************************************************************/
/*  vehicle_body.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#ifndef VEHICLE_BODY_H
#define VEHICLE_BODY_H

#include "scene/3d/physics_body.h"

class VehicleBody;

class VehicleWheel : public Spatial {

	GDCLASS(VehicleWheel, Spatial);

	friend class VehicleBody;

	Transform m_worldTransform;
	Transform local_xform;
	bool engine_traction;
	bool steers;

	Vector3 m_chassisConnectionPointCS; //const
	Vector3 m_wheelDirectionCS; //const
	Vector3 m_wheelAxleCS; // const or modified by steering

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

	//btVector3	m_wheelAxleCS; // const or modified by steering ?

	real_t m_steering;
	real_t m_rotation;
	real_t m_deltaRotation;
	real_t m_rollInfluence;
	//real_t	m_engineForce;
	real_t m_brake;

	real_t m_clippedInvContactDotSuspension;
	real_t m_suspensionRelativeVelocity;
	//calculated by suspension
	real_t m_wheelsSuspensionForce;
	real_t m_skidInfo;

	struct RaycastInfo {
		//set by raycaster
		Vector3 m_contactNormalWS; //contactnormal
		Vector3 m_contactPointWS; //raycast hitpoint
		real_t m_suspensionLength;
		Vector3 m_hardPointWS; //raycast starting point
		Vector3 m_wheelDirectionWS; //direction in worldspace
		Vector3 m_wheelAxleWS; // axle in worldspace
		bool m_isInContact;
		PhysicsBody *m_groundObject; //could be general void* ptr
	} m_raycastInfo;

	void _update(PhysicsDirectBodyState *s);

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_radius(float p_radius);
	float get_radius() const;

	void set_suspension_rest_length(float p_length);
	float get_suspension_rest_length() const;

	void set_suspension_travel(float p_length);
	float get_suspension_travel() const;

	void set_suspension_stiffness(float p_value);
	float get_suspension_stiffness() const;

	void set_suspension_max_force(float p_value);
	float get_suspension_max_force() const;

	void set_damping_compression(float p_value);
	float get_damping_compression() const;

	void set_damping_relaxation(float p_value);
	float get_damping_relaxation() const;

	void set_friction_slip(float p_value);
	float get_friction_slip() const;

	void set_use_as_traction(bool p_enable);
	bool is_used_as_traction() const;

	void set_use_as_steering(bool p_enabled);
	bool is_used_as_steering() const;

	bool is_in_contact() const;

	void set_roll_influence(float p_value);
	float get_roll_influence() const;

	String get_configuration_warning() const;

	VehicleWheel();
};

class VehicleBody : public PhysicsBody {

	GDCLASS(VehicleBody, PhysicsBody);

	real_t mass;
	real_t friction;

	float engine_force;
	float brake;

	Vector3 linear_velocity;
	Vector3 angular_velocity;
	bool ccd;

	real_t m_pitchControl;
	real_t m_steeringValue;
	real_t m_currentVehicleSpeedKmHour;

	Set<RID> exclude;

	Vector<Vector3> m_forwardWS;
	Vector<Vector3> m_axle;
	Vector<real_t> m_forwardImpulse;
	Vector<real_t> m_sideImpulse;

	struct btVehicleWheelContactPoint {
		PhysicsDirectBodyState *m_s;
		PhysicsBody *m_body1;
		Vector3 m_frictionPositionWorld;
		Vector3 m_frictionDirectionWorld;
		real_t m_jacDiagABInv;
		real_t m_maxImpulse;

		btVehicleWheelContactPoint(PhysicsDirectBodyState *s, PhysicsBody *body1, const Vector3 &frictionPosWorld, const Vector3 &frictionDirectionWorld, real_t maxImpulse);
	};

	void _resolve_single_bilateral(PhysicsDirectBodyState *s, const Vector3 &pos1, PhysicsBody *body2, const Vector3 &pos2, const Vector3 &normal, real_t &impulse);
	real_t _calc_rolling_friction(btVehicleWheelContactPoint &contactPoint);

	void _update_friction(PhysicsDirectBodyState *s);
	void _update_suspension(PhysicsDirectBodyState *s);
	real_t _ray_cast(int p_idx, PhysicsDirectBodyState *s);
	void _update_wheel_transform(VehicleWheel &wheel, PhysicsDirectBodyState *s);
	void _update_wheel(int p_idx, PhysicsDirectBodyState *s);

	friend class VehicleWheel;
	Vector<VehicleWheel *> wheels;

	static void _bind_methods();

	void _direct_state_changed(Object *p_state);

public:
	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_engine_force(float p_engine_force);
	float get_engine_force() const;

	void set_brake(float p_brake);
	float get_brake() const;

	void set_steering(float p_steering);
	float get_steering() const;

	Vector3 get_linear_velocity() const;

	VehicleBody();
};

#endif // VEHICLE_BODY_H
