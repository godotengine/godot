/*************************************************************************/
/*  vehicle_body.cpp                                                     */
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
#include "vehicle_body.h"

#define ROLLING_INFLUENCE_FIX

class btVehicleJacobianEntry {
public:
	Vector3 m_linearJointAxis;
	Vector3 m_aJ;
	Vector3 m_bJ;
	Vector3 m_0MinvJt;
	Vector3 m_1MinvJt;
	//Optimization: can be stored in the w/last component of one of the vectors
	real_t m_Adiag;

	real_t getDiagonal() const { return m_Adiag; }

	btVehicleJacobianEntry(){};
	//constraint between two different rigidbodies
	btVehicleJacobianEntry(
			const Basis &world2A,
			const Basis &world2B,
			const Vector3 &rel_pos1,
			const Vector3 &rel_pos2,
			const Vector3 &jointAxis,
			const Vector3 &inertiaInvA,
			const real_t massInvA,
			const Vector3 &inertiaInvB,
			const real_t massInvB) :
			m_linearJointAxis(jointAxis) {
		m_aJ = world2A.xform(rel_pos1.cross(m_linearJointAxis));
		m_bJ = world2B.xform(rel_pos2.cross(-m_linearJointAxis));
		m_0MinvJt = inertiaInvA * m_aJ;
		m_1MinvJt = inertiaInvB * m_bJ;
		m_Adiag = massInvA + m_0MinvJt.dot(m_aJ) + massInvB + m_1MinvJt.dot(m_bJ);

		//btAssert(m_Adiag > real_t(0.0));
	}

	real_t getRelativeVelocity(const Vector3 &linvelA, const Vector3 &angvelA, const Vector3 &linvelB, const Vector3 &angvelB) {
		Vector3 linrel = linvelA - linvelB;
		Vector3 angvela = angvelA * m_aJ;
		Vector3 angvelb = angvelB * m_bJ;
		linrel *= m_linearJointAxis;
		angvela += angvelb;
		angvela += linrel;
		real_t rel_vel2 = angvela[0] + angvela[1] + angvela[2];
		return rel_vel2 + CMP_EPSILON;
	}
};

void VehicleWheel::_notification(int p_what) {

	if (p_what == NOTIFICATION_ENTER_TREE) {

		VehicleBody *cb = Object::cast_to<VehicleBody>(get_parent());
		if (!cb)
			return;
		body = cb;
		local_xform = get_transform();
		cb->wheels.push_back(this);

		m_chassisConnectionPointCS = get_transform().origin;
		m_wheelDirectionCS = -get_transform().basis.get_axis(Vector3::AXIS_Y).normalized();
		m_wheelAxleCS = get_transform().basis.get_axis(Vector3::AXIS_X).normalized();
	}
	if (p_what == NOTIFICATION_EXIT_TREE) {

		VehicleBody *cb = Object::cast_to<VehicleBody>(get_parent());
		if (!cb)
			return;
		cb->wheels.erase(this);
		body = NULL;
	}
}

String VehicleWheel::get_configuration_warning() const {
	if (!Object::cast_to<VehicleBody>(get_parent())) {
		return TTR("VehicleWheel serves to provide a wheel system to a VehicleBody. Please use it as a child of a VehicleBody.");
	}

	return String();
}

void VehicleWheel::_update(PhysicsDirectBodyState *s) {

	if (m_raycastInfo.m_isInContact)

	{
		real_t project = m_raycastInfo.m_contactNormalWS.dot(m_raycastInfo.m_wheelDirectionWS);
		Vector3 chassis_velocity_at_contactPoint;
		Vector3 relpos = m_raycastInfo.m_contactPointWS - s->get_transform().origin;

		chassis_velocity_at_contactPoint = s->get_linear_velocity() +
										   (s->get_angular_velocity()).cross(relpos); // * mPos);

		real_t projVel = m_raycastInfo.m_contactNormalWS.dot(chassis_velocity_at_contactPoint);
		if (project >= real_t(-0.1)) {
			m_suspensionRelativeVelocity = real_t(0.0);
			m_clippedInvContactDotSuspension = real_t(1.0) / real_t(0.1);
		} else {
			real_t inv = real_t(-1.) / project;
			m_suspensionRelativeVelocity = projVel * inv;
			m_clippedInvContactDotSuspension = inv;
		}

	}

	else // Not in contact : position wheel in a nice (rest length) position
	{
		m_raycastInfo.m_suspensionLength = m_suspensionRestLength;
		m_suspensionRelativeVelocity = real_t(0.0);
		m_raycastInfo.m_contactNormalWS = -m_raycastInfo.m_wheelDirectionWS;
		m_clippedInvContactDotSuspension = real_t(1.0);
	}
}

void VehicleWheel::set_radius(float p_radius) {

	m_wheelRadius = p_radius;
	update_gizmo();
}

float VehicleWheel::get_radius() const {

	return m_wheelRadius;
}

void VehicleWheel::set_suspension_rest_length(float p_length) {

	m_suspensionRestLength = p_length;
	update_gizmo();
}
float VehicleWheel::get_suspension_rest_length() const {

	return m_suspensionRestLength;
}

void VehicleWheel::set_suspension_travel(float p_length) {

	m_maxSuspensionTravelCm = p_length / 0.01;
}
float VehicleWheel::get_suspension_travel() const {

	return m_maxSuspensionTravelCm * 0.01;
}

void VehicleWheel::set_suspension_stiffness(float p_value) {

	m_suspensionStiffness = p_value;
}
float VehicleWheel::get_suspension_stiffness() const {

	return m_suspensionStiffness;
}

void VehicleWheel::set_suspension_max_force(float p_value) {

	m_maxSuspensionForce = p_value;
}
float VehicleWheel::get_suspension_max_force() const {

	return m_maxSuspensionForce;
}

void VehicleWheel::set_damping_compression(float p_value) {

	m_wheelsDampingCompression = p_value;
}
float VehicleWheel::get_damping_compression() const {

	return m_wheelsDampingCompression;
}

void VehicleWheel::set_damping_relaxation(float p_value) {

	m_wheelsDampingRelaxation = p_value;
}
float VehicleWheel::get_damping_relaxation() const {

	return m_wheelsDampingRelaxation;
}

void VehicleWheel::set_friction_slip(float p_value) {

	m_frictionSlip = p_value;
}
float VehicleWheel::get_friction_slip() const {

	return m_frictionSlip;
}

void VehicleWheel::set_roll_influence(float p_value) {
	m_rollInfluence = p_value;
}

float VehicleWheel::get_roll_influence() const {
	return m_rollInfluence;
}

bool VehicleWheel::is_in_contact() const {
	return m_raycastInfo.m_isInContact;
}

void VehicleWheel::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_radius", "length"), &VehicleWheel::set_radius);
	ClassDB::bind_method(D_METHOD("get_radius"), &VehicleWheel::get_radius);

	ClassDB::bind_method(D_METHOD("set_suspension_rest_length", "length"), &VehicleWheel::set_suspension_rest_length);
	ClassDB::bind_method(D_METHOD("get_suspension_rest_length"), &VehicleWheel::get_suspension_rest_length);

	ClassDB::bind_method(D_METHOD("set_suspension_travel", "length"), &VehicleWheel::set_suspension_travel);
	ClassDB::bind_method(D_METHOD("get_suspension_travel"), &VehicleWheel::get_suspension_travel);

	ClassDB::bind_method(D_METHOD("set_suspension_stiffness", "length"), &VehicleWheel::set_suspension_stiffness);
	ClassDB::bind_method(D_METHOD("get_suspension_stiffness"), &VehicleWheel::get_suspension_stiffness);

	ClassDB::bind_method(D_METHOD("set_suspension_max_force", "length"), &VehicleWheel::set_suspension_max_force);
	ClassDB::bind_method(D_METHOD("get_suspension_max_force"), &VehicleWheel::get_suspension_max_force);

	ClassDB::bind_method(D_METHOD("set_damping_compression", "length"), &VehicleWheel::set_damping_compression);
	ClassDB::bind_method(D_METHOD("get_damping_compression"), &VehicleWheel::get_damping_compression);

	ClassDB::bind_method(D_METHOD("set_damping_relaxation", "length"), &VehicleWheel::set_damping_relaxation);
	ClassDB::bind_method(D_METHOD("get_damping_relaxation"), &VehicleWheel::get_damping_relaxation);

	ClassDB::bind_method(D_METHOD("set_use_as_traction", "enable"), &VehicleWheel::set_use_as_traction);
	ClassDB::bind_method(D_METHOD("is_used_as_traction"), &VehicleWheel::is_used_as_traction);

	ClassDB::bind_method(D_METHOD("set_use_as_steering", "enable"), &VehicleWheel::set_use_as_steering);
	ClassDB::bind_method(D_METHOD("is_used_as_steering"), &VehicleWheel::is_used_as_steering);

	ClassDB::bind_method(D_METHOD("set_friction_slip", "length"), &VehicleWheel::set_friction_slip);
	ClassDB::bind_method(D_METHOD("get_friction_slip"), &VehicleWheel::get_friction_slip);

	ClassDB::bind_method(D_METHOD("is_in_contact"), &VehicleWheel::is_in_contact);

	ClassDB::bind_method(D_METHOD("set_roll_influence", "roll_influence"), &VehicleWheel::set_roll_influence);
	ClassDB::bind_method(D_METHOD("get_roll_influence"), &VehicleWheel::get_roll_influence);

	ClassDB::bind_method(D_METHOD("get_skidinfo"), &VehicleWheel::get_skidinfo);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_as_traction"), "set_use_as_traction", "is_used_as_traction");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "use_as_steering"), "set_use_as_steering", "is_used_as_steering");
	ADD_GROUP("Wheel", "wheel_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "wheel_roll_influence"), "set_roll_influence", "get_roll_influence");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "wheel_radius"), "set_radius", "get_radius");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "wheel_rest_length"), "set_suspension_rest_length", "get_suspension_rest_length");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "wheel_friction_slip"), "set_friction_slip", "get_friction_slip");
	ADD_GROUP("Suspension", "suspension_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "suspension_travel"), "set_suspension_travel", "get_suspension_travel");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "suspension_stiffness"), "set_suspension_stiffness", "get_suspension_stiffness");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "suspension_max_force"), "set_suspension_max_force", "get_suspension_max_force");
	ADD_GROUP("Damping", "damping_");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "damping_compression"), "set_damping_compression", "get_damping_compression");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "damping_relaxation"), "set_damping_relaxation", "get_damping_relaxation");
}

void VehicleWheel::set_use_as_traction(bool p_enable) {

	engine_traction = p_enable;
}

bool VehicleWheel::is_used_as_traction() const {

	return engine_traction;
}

void VehicleWheel::set_use_as_steering(bool p_enabled) {

	steers = p_enabled;
}

bool VehicleWheel::is_used_as_steering() const {

	return steers;
}

float VehicleWheel::get_skidinfo() const {

	return m_skidInfo;
}

VehicleWheel::VehicleWheel() {

	steers = false;
	engine_traction = false;

	m_steering = real_t(0.);
	//m_engineForce = real_t(0.);
	m_rotation = real_t(0.);
	m_deltaRotation = real_t(0.);
	m_brake = real_t(0.);
	m_rollInfluence = real_t(0.1);

	m_suspensionRestLength = 0.15;
	m_wheelRadius = 0.5; //0.28;
	m_suspensionStiffness = 5.88;
	m_wheelsDampingCompression = 0.83;
	m_wheelsDampingRelaxation = 0.88;
	m_frictionSlip = 10.5;
	m_bIsFrontWheel = false;
	m_maxSuspensionTravelCm = 500;
	m_maxSuspensionForce = 6000;

	m_suspensionRelativeVelocity = 0;
	m_clippedInvContactDotSuspension = 1.0;
	m_raycastInfo.m_isInContact = false;

	body = NULL;
}

void VehicleBody::_update_wheel_transform(VehicleWheel &wheel, PhysicsDirectBodyState *s) {

	wheel.m_raycastInfo.m_isInContact = false;

	Transform chassisTrans = s->get_transform();
	/*
	if (interpolatedTransform && (getRigidBody()->getMotionState())) {
		getRigidBody()->getMotionState()->getWorldTransform(chassisTrans);
	}
	*/

	wheel.m_raycastInfo.m_hardPointWS = chassisTrans.xform(wheel.m_chassisConnectionPointCS);
	//wheel.m_raycastInfo.m_hardPointWS+=s->get_linear_velocity()*s->get_step();
	wheel.m_raycastInfo.m_wheelDirectionWS = chassisTrans.get_basis().xform(wheel.m_wheelDirectionCS).normalized();
	wheel.m_raycastInfo.m_wheelAxleWS = chassisTrans.get_basis().xform(wheel.m_wheelAxleCS).normalized();
}

void VehicleBody::_update_wheel(int p_idx, PhysicsDirectBodyState *s) {

	VehicleWheel &wheel = *wheels[p_idx];
	_update_wheel_transform(wheel, s);

	Vector3 up = -wheel.m_raycastInfo.m_wheelDirectionWS;
	const Vector3 &right = wheel.m_raycastInfo.m_wheelAxleWS;
	Vector3 fwd = up.cross(right);
	fwd = fwd.normalized();
	//up = right.cross(fwd);
	//up.normalize();

	//rotate around steering over de wheelAxleWS
	real_t steering = wheel.steers ? m_steeringValue : 0.0;
	//print_line(itos(p_idx)+": "+rtos(steering));

	Basis steeringMat(up, steering);

	Basis rotatingMat(right, -wheel.m_rotation);

	/*
	if (p_idx==1)
		print_line("steeringMat " +steeringMat);
	*/

	Basis basis2(
			right[0], up[0], fwd[0],
			right[1], up[1], fwd[1],
			right[2], up[2], fwd[2]);

	wheel.m_worldTransform.set_basis(steeringMat * rotatingMat * basis2);
	//wheel.m_worldTransform.set_basis(basis2 * (steeringMat * rotatingMat));
	wheel.m_worldTransform.set_origin(
			wheel.m_raycastInfo.m_hardPointWS + wheel.m_raycastInfo.m_wheelDirectionWS * wheel.m_raycastInfo.m_suspensionLength);
}

real_t VehicleBody::_ray_cast(int p_idx, PhysicsDirectBodyState *s) {

	VehicleWheel &wheel = *wheels[p_idx];

	_update_wheel_transform(wheel, s);

	real_t depth = -1;

	real_t raylen = wheel.m_suspensionRestLength + wheel.m_wheelRadius;

	Vector3 rayvector = wheel.m_raycastInfo.m_wheelDirectionWS * (raylen);
	Vector3 source = wheel.m_raycastInfo.m_hardPointWS;
	wheel.m_raycastInfo.m_contactPointWS = source + rayvector;
	const Vector3 &target = wheel.m_raycastInfo.m_contactPointWS;
	source -= wheel.m_wheelRadius * wheel.m_raycastInfo.m_wheelDirectionWS;

	real_t param = real_t(0.);

	PhysicsDirectSpaceState::RayResult rr;

	PhysicsDirectSpaceState *ss = s->get_space_state();

	bool col = ss->intersect_ray(source, target, rr, exclude);

	wheel.m_raycastInfo.m_groundObject = 0;

	if (col) {
		//print_line("WHEEL "+itos(p_idx)+" FROM "+source+" TO: "+target);
		//print_line("WHEEL "+itos(p_idx)+" COLLIDE? "+itos(col));
		param = source.distance_to(rr.position) / source.distance_to(target);
		depth = raylen * param;
		wheel.m_raycastInfo.m_contactNormalWS = rr.normal;

		wheel.m_raycastInfo.m_isInContact = true;
		if (rr.collider)
			wheel.m_raycastInfo.m_groundObject = Object::cast_to<PhysicsBody>(rr.collider);

		real_t hitDistance = param * raylen;
		wheel.m_raycastInfo.m_suspensionLength = hitDistance - wheel.m_wheelRadius;
		//clamp on max suspension travel

		real_t minSuspensionLength = wheel.m_suspensionRestLength - wheel.m_maxSuspensionTravelCm * real_t(0.01);
		real_t maxSuspensionLength = wheel.m_suspensionRestLength + wheel.m_maxSuspensionTravelCm * real_t(0.01);
		if (wheel.m_raycastInfo.m_suspensionLength < minSuspensionLength) {
			wheel.m_raycastInfo.m_suspensionLength = minSuspensionLength;
		}
		if (wheel.m_raycastInfo.m_suspensionLength > maxSuspensionLength) {
			wheel.m_raycastInfo.m_suspensionLength = maxSuspensionLength;
		}

		wheel.m_raycastInfo.m_contactPointWS = rr.position;

		real_t denominator = wheel.m_raycastInfo.m_contactNormalWS.dot(wheel.m_raycastInfo.m_wheelDirectionWS);

		Vector3 chassis_velocity_at_contactPoint;
		//Vector3 relpos = wheel.m_raycastInfo.m_contactPointWS-getRigidBody()->getCenterOfMassPosition();

		//chassis_velocity_at_contactPoint = getRigidBody()->getVelocityInLocalPoint(relpos);

		chassis_velocity_at_contactPoint = s->get_linear_velocity() +
										   (s->get_angular_velocity()).cross(wheel.m_raycastInfo.m_contactPointWS - s->get_transform().origin); // * mPos);

		real_t projVel = wheel.m_raycastInfo.m_contactNormalWS.dot(chassis_velocity_at_contactPoint);

		if (denominator >= real_t(-0.1)) {
			wheel.m_suspensionRelativeVelocity = real_t(0.0);
			wheel.m_clippedInvContactDotSuspension = real_t(1.0) / real_t(0.1);
		} else {
			real_t inv = real_t(-1.) / denominator;
			wheel.m_suspensionRelativeVelocity = projVel * inv;
			wheel.m_clippedInvContactDotSuspension = inv;
		}

	} else {
		wheel.m_raycastInfo.m_isInContact = false;
		//put wheel info as in rest position
		wheel.m_raycastInfo.m_suspensionLength = wheel.m_suspensionRestLength;
		wheel.m_suspensionRelativeVelocity = real_t(0.0);
		wheel.m_raycastInfo.m_contactNormalWS = -wheel.m_raycastInfo.m_wheelDirectionWS;
		wheel.m_clippedInvContactDotSuspension = real_t(1.0);
	}

	return depth;
}

void VehicleBody::_update_suspension(PhysicsDirectBodyState *s) {

	real_t chassisMass = mass;

	for (int w_it = 0; w_it < wheels.size(); w_it++) {
		VehicleWheel &wheel_info = *wheels[w_it];

		if (wheel_info.m_raycastInfo.m_isInContact) {
			real_t force;
			//Spring
			{
				real_t susp_length = wheel_info.m_suspensionRestLength;
				real_t current_length = wheel_info.m_raycastInfo.m_suspensionLength;

				real_t length_diff = (susp_length - current_length);

				force = wheel_info.m_suspensionStiffness * length_diff * wheel_info.m_clippedInvContactDotSuspension;
			}

			// Damper
			{
				real_t projected_rel_vel = wheel_info.m_suspensionRelativeVelocity;
				{
					real_t susp_damping;
					if (projected_rel_vel < real_t(0.0)) {
						susp_damping = wheel_info.m_wheelsDampingCompression;
					} else {
						susp_damping = wheel_info.m_wheelsDampingRelaxation;
					}
					force -= susp_damping * projected_rel_vel;
				}
			}

			// RESULT
			wheel_info.m_wheelsSuspensionForce = force * chassisMass;
			if (wheel_info.m_wheelsSuspensionForce < real_t(0.)) {
				wheel_info.m_wheelsSuspensionForce = real_t(0.);
			}
		} else {
			wheel_info.m_wheelsSuspensionForce = real_t(0.0);
		}
	}
}

//bilateral constraint between two dynamic objects
void VehicleBody::_resolve_single_bilateral(PhysicsDirectBodyState *s, const Vector3 &pos1,
		PhysicsBody *body2, const Vector3 &pos2, const Vector3 &normal, real_t &impulse) {

	real_t normalLenSqr = normal.length_squared();
	//ERR_FAIL_COND( normalLenSqr < real_t(1.1));

	if (normalLenSqr > real_t(1.1)) {
		impulse = real_t(0.);
		return;
	}

	Vector3 rel_pos1 = pos1 - s->get_transform().origin;
	Vector3 rel_pos2;
	if (body2)
		rel_pos2 = pos2 - body2->get_global_transform().origin;
	//this jacobian entry could be re-used for all iterations

	Vector3 vel1 = s->get_linear_velocity() + (s->get_angular_velocity()).cross(rel_pos1); // * mPos);
	Vector3 vel2;

	if (body2)
		vel2 = body2->get_linear_velocity() + body2->get_angular_velocity().cross(rel_pos2);

	Vector3 vel = vel1 - vel2;

	Basis b2trans;
	float b2invmass = 0;
	Vector3 b2lv;
	Vector3 b2av;
	Vector3 b2invinertia; //todo

	if (body2) {
		b2trans = body2->get_global_transform().basis.transposed();
		b2invmass = body2->get_inverse_mass();
		b2lv = body2->get_linear_velocity();
		b2av = body2->get_angular_velocity();
	}

	btVehicleJacobianEntry jac(s->get_transform().basis.transposed(),
			b2trans,
			rel_pos1,
			rel_pos2,
			normal,
			s->get_inverse_inertia_tensor().get_main_diagonal(),
			1.0 / mass,
			b2invinertia,
			b2invmass);

	// FIXME: rel_vel assignment here is overwritten by the following assignment.
	// What seemes to be intented in the next next assignment is: rel_vel = normal.dot(rel_vel);
	// Investigate why.
	real_t rel_vel = jac.getRelativeVelocity(
			s->get_linear_velocity(),
			s->get_transform().basis.transposed().xform(s->get_angular_velocity()),
			b2lv,
			b2trans.xform(b2av));

	rel_vel = normal.dot(vel);

	//TODO: move this into proper structure
	real_t contactDamping = real_t(0.4);
#define ONLY_USE_LINEAR_MASS
#ifdef ONLY_USE_LINEAR_MASS
	real_t massTerm = real_t(1.) / ((1.0 / mass) + b2invmass);
	impulse = -contactDamping * rel_vel * massTerm;
#else
	real_t velocityImpulse = -contactDamping * rel_vel * jacDiagABInv;
	impulse = velocityImpulse;
#endif
}

VehicleBody::btVehicleWheelContactPoint::btVehicleWheelContactPoint(PhysicsDirectBodyState *s, PhysicsBody *body1, const Vector3 &frictionPosWorld, const Vector3 &frictionDirectionWorld, real_t maxImpulse) :
		m_s(s),
		m_body1(body1),
		m_frictionPositionWorld(frictionPosWorld),
		m_frictionDirectionWorld(frictionDirectionWorld),
		m_maxImpulse(maxImpulse) {
	float denom0 = 0;
	float denom1 = 0;

	{
		Vector3 r0 = frictionPosWorld - s->get_transform().origin;
		Vector3 c0 = (r0).cross(frictionDirectionWorld);
		Vector3 vec = s->get_inverse_inertia_tensor().xform_inv(c0).cross(r0);
		denom0 = s->get_inverse_mass() + frictionDirectionWorld.dot(vec);
	}

	/* TODO: Why is this code unused?
	if (body1) {

		Vector3 r0 = frictionPosWorld - body1->get_global_transform().origin;
		Vector3 c0 = (r0).cross(frictionDirectionWorld);
		Vector3 vec = s->get_inverse_inertia_tensor().xform_inv(c0).cross(r0);
		//denom1= body1->get_inverse_mass() + frictionDirectionWorld.dot(vec);

	}
	*/

	real_t relaxation = 1.f;
	m_jacDiagABInv = relaxation / (denom0 + denom1);
}

real_t VehicleBody::_calc_rolling_friction(btVehicleWheelContactPoint &contactPoint) {

	real_t j1 = 0.f;

	const Vector3 &contactPosWorld = contactPoint.m_frictionPositionWorld;

	Vector3 rel_pos1 = contactPosWorld - contactPoint.m_s->get_transform().origin;
	Vector3 rel_pos2;
	if (contactPoint.m_body1)
		rel_pos2 = contactPosWorld - contactPoint.m_body1->get_global_transform().origin;

	real_t maxImpulse = contactPoint.m_maxImpulse;

	Vector3 vel1 = contactPoint.m_s->get_linear_velocity() + (contactPoint.m_s->get_angular_velocity()).cross(rel_pos1); // * mPos);

	Vector3 vel2;
	if (contactPoint.m_body1) {
		vel2 = contactPoint.m_body1->get_linear_velocity() + contactPoint.m_body1->get_angular_velocity().cross(rel_pos2);
	}

	Vector3 vel = vel1 - vel2;

	real_t vrel = contactPoint.m_frictionDirectionWorld.dot(vel);

	// calculate j that moves us to zero relative velocity
	j1 = -vrel * contactPoint.m_jacDiagABInv;

	return CLAMP(j1, -maxImpulse, maxImpulse);
}

static const real_t sideFrictionStiffness2 = real_t(1.0);
void VehicleBody::_update_friction(PhysicsDirectBodyState *s) {

	//calculate the impulse, so that the wheels don't move sidewards
	int numWheel = wheels.size();
	if (!numWheel)
		return;

	m_forwardWS.resize(numWheel);
	m_axle.resize(numWheel);
	m_forwardImpulse.resize(numWheel);
	m_sideImpulse.resize(numWheel);

	int numWheelsOnGround = 0;

	//collapse all those loops into one!
	for (int i = 0; i < wheels.size(); i++) {
		VehicleWheel &wheelInfo = *wheels[i];
		if (wheelInfo.m_raycastInfo.m_isInContact)
			numWheelsOnGround++;
		m_sideImpulse[i] = real_t(0.);
		m_forwardImpulse[i] = real_t(0.);
	}

	{

		for (int i = 0; i < wheels.size(); i++) {

			VehicleWheel &wheelInfo = *wheels[i];

			if (wheelInfo.m_raycastInfo.m_isInContact) {

				//const btTransform& wheelTrans = getWheelTransformWS( i );

				Basis wheelBasis0 = wheelInfo.m_worldTransform.basis; //get_global_transform().basis;

				m_axle[i] = wheelBasis0.get_axis(Vector3::AXIS_X);
				//m_axle[i] = wheelInfo.m_raycastInfo.m_wheelAxleWS;

				const Vector3 &surfNormalWS = wheelInfo.m_raycastInfo.m_contactNormalWS;
				real_t proj = m_axle[i].dot(surfNormalWS);
				m_axle[i] -= surfNormalWS * proj;
				m_axle[i] = m_axle[i].normalized();

				m_forwardWS[i] = surfNormalWS.cross(m_axle[i]);
				m_forwardWS[i].normalize();

				_resolve_single_bilateral(s, wheelInfo.m_raycastInfo.m_contactPointWS,
						wheelInfo.m_raycastInfo.m_groundObject, wheelInfo.m_raycastInfo.m_contactPointWS,
						m_axle[i], m_sideImpulse[i]);

				m_sideImpulse[i] *= sideFrictionStiffness2;
			}
		}
	}

	real_t sideFactor = real_t(1.);
	real_t fwdFactor = 0.5;

	bool sliding = false;
	{
		for (int wheel = 0; wheel < wheels.size(); wheel++) {
			VehicleWheel &wheelInfo = *wheels[wheel];

			//class btRigidBody* groundObject = (class btRigidBody*) wheelInfo.m_raycastInfo.m_groundObject;

			real_t rollingFriction = 0.f;

			if (wheelInfo.m_raycastInfo.m_isInContact) {
				if (engine_force != 0.f) {
					rollingFriction = -engine_force * s->get_step();
				} else {
					real_t defaultRollingFrictionImpulse = 0.f;
					float cbrake = MAX(wheelInfo.m_brake, brake);
					real_t maxImpulse = cbrake ? cbrake : defaultRollingFrictionImpulse;
					btVehicleWheelContactPoint contactPt(s, wheelInfo.m_raycastInfo.m_groundObject, wheelInfo.m_raycastInfo.m_contactPointWS, m_forwardWS[wheel], maxImpulse);
					rollingFriction = _calc_rolling_friction(contactPt);
				}
			}

			//switch between active rolling (throttle), braking and non-active rolling friction (no throttle/break)

			m_forwardImpulse[wheel] = real_t(0.);
			wheelInfo.m_skidInfo = real_t(1.);

			if (wheelInfo.m_raycastInfo.m_isInContact) {
				wheelInfo.m_skidInfo = real_t(1.);

				real_t maximp = wheelInfo.m_wheelsSuspensionForce * s->get_step() * wheelInfo.m_frictionSlip;
				real_t maximpSide = maximp;

				real_t maximpSquared = maximp * maximpSide;

				m_forwardImpulse[wheel] = rollingFriction; //wheelInfo.m_engineForce* timeStep;

				real_t x = (m_forwardImpulse[wheel]) * fwdFactor;
				real_t y = (m_sideImpulse[wheel]) * sideFactor;

				real_t impulseSquared = (x * x + y * y);

				if (impulseSquared > maximpSquared) {
					sliding = true;

					real_t factor = maximp / Math::sqrt(impulseSquared);

					wheelInfo.m_skidInfo *= factor;
				}
			}
		}
	}

	if (sliding) {
		for (int wheel = 0; wheel < wheels.size(); wheel++) {
			if (m_sideImpulse[wheel] != real_t(0.)) {
				if (wheels[wheel]->m_skidInfo < real_t(1.)) {
					m_forwardImpulse[wheel] *= wheels[wheel]->m_skidInfo;
					m_sideImpulse[wheel] *= wheels[wheel]->m_skidInfo;
				}
			}
		}
	}

	// apply the impulses
	{
		for (int wheel = 0; wheel < wheels.size(); wheel++) {
			VehicleWheel &wheelInfo = *wheels[wheel];

			Vector3 rel_pos = wheelInfo.m_raycastInfo.m_contactPointWS -
							  s->get_transform().origin;

			if (m_forwardImpulse[wheel] != real_t(0.)) {
				s->apply_impulse(rel_pos, m_forwardWS[wheel] * (m_forwardImpulse[wheel]));
			}
			if (m_sideImpulse[wheel] != real_t(0.)) {
				PhysicsBody *groundObject = wheelInfo.m_raycastInfo.m_groundObject;

				Vector3 rel_pos2;
				if (groundObject) {
					rel_pos2 = wheelInfo.m_raycastInfo.m_contactPointWS - groundObject->get_global_transform().origin;
				}

				Vector3 sideImp = m_axle[wheel] * m_sideImpulse[wheel];

#if defined ROLLING_INFLUENCE_FIX // fix. It only worked if car's up was along Y - VT.
				Vector3 vChassisWorldUp = s->get_transform().basis.transposed()[1]; //getRigidBody()->getCenterOfMassTransform().getBasis().getColumn(m_indexUpAxis);
				rel_pos -= vChassisWorldUp * (vChassisWorldUp.dot(rel_pos) * (1.f - wheelInfo.m_rollInfluence));
#else
				rel_pos[1] *= wheelInfo.m_rollInfluence; //?
#endif
				s->apply_impulse(rel_pos, sideImp);

				//apply friction impulse on the ground
				//todo
				//groundObject->applyImpulse(-sideImp,rel_pos2);
			}
		}
	}
}

void VehicleBody::_direct_state_changed(Object *p_state) {

	PhysicsDirectBodyState *s = Object::cast_to<PhysicsDirectBodyState>(p_state);

	set_ignore_transform_notification(true);
	set_global_transform(s->get_transform());
	set_ignore_transform_notification(false);

	float step = s->get_step();

	for (int i = 0; i < wheels.size(); i++) {

		_update_wheel(i, s);
	}

	for (int i = 0; i < wheels.size(); i++) {

		_ray_cast(i, s);
		wheels[i]->set_transform(s->get_transform().inverse() * wheels[i]->m_worldTransform);
	}

	_update_suspension(s);

	for (int i = 0; i < wheels.size(); i++) {

		//apply suspension force
		VehicleWheel &wheel = *wheels[i];

		real_t suspensionForce = wheel.m_wheelsSuspensionForce;

		if (suspensionForce > wheel.m_maxSuspensionForce) {
			suspensionForce = wheel.m_maxSuspensionForce;
		}
		Vector3 impulse = wheel.m_raycastInfo.m_contactNormalWS * suspensionForce * step;
		Vector3 relpos = wheel.m_raycastInfo.m_contactPointWS - s->get_transform().origin;

		s->apply_impulse(relpos, impulse);
		//getRigidBody()->applyImpulse(impulse, relpos);
	}

	_update_friction(s);

	for (int i = 0; i < wheels.size(); i++) {
		VehicleWheel &wheel = *wheels[i];
		Vector3 relpos = wheel.m_raycastInfo.m_hardPointWS - s->get_transform().origin;
		Vector3 vel = s->get_linear_velocity() + (s->get_angular_velocity()).cross(relpos); // * mPos);

		if (wheel.m_raycastInfo.m_isInContact) {
			const Transform &chassisWorldTransform = s->get_transform();

			Vector3 fwd(
					chassisWorldTransform.basis[0][Vector3::AXIS_Z],
					chassisWorldTransform.basis[1][Vector3::AXIS_Z],
					chassisWorldTransform.basis[2][Vector3::AXIS_Z]);

			real_t proj = fwd.dot(wheel.m_raycastInfo.m_contactNormalWS);
			fwd -= wheel.m_raycastInfo.m_contactNormalWS * proj;

			real_t proj2 = fwd.dot(vel);

			wheel.m_deltaRotation = (proj2 * step) / (wheel.m_wheelRadius);
			wheel.m_rotation += wheel.m_deltaRotation;

		} else {
			wheel.m_rotation += wheel.m_deltaRotation;
		}

		wheel.m_deltaRotation *= real_t(0.99); //damping of rotation when not in contact
	}
	linear_velocity = s->get_linear_velocity();
}

void VehicleBody::set_mass(real_t p_mass) {

	mass = p_mass;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_MASS, mass);
}

real_t VehicleBody::get_mass() const {

	return mass;
}

void VehicleBody::set_friction(real_t p_friction) {

	friction = p_friction;
	PhysicsServer::get_singleton()->body_set_param(get_rid(), PhysicsServer::BODY_PARAM_FRICTION, friction);
}

real_t VehicleBody::get_friction() const {

	return friction;
}

void VehicleBody::set_engine_force(float p_engine_force) {

	engine_force = p_engine_force;
}

float VehicleBody::get_engine_force() const {

	return engine_force;
}

void VehicleBody::set_brake(float p_brake) {

	brake = p_brake;
}
float VehicleBody::get_brake() const {

	return brake;
}

void VehicleBody::set_steering(float p_steering) {

	m_steeringValue = p_steering;
}
float VehicleBody::get_steering() const {

	return m_steeringValue;
}

Vector3 VehicleBody::get_linear_velocity() const {
	return linear_velocity;
}

void VehicleBody::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_mass", "mass"), &VehicleBody::set_mass);
	ClassDB::bind_method(D_METHOD("get_mass"), &VehicleBody::get_mass);

	ClassDB::bind_method(D_METHOD("set_friction", "friction"), &VehicleBody::set_friction);
	ClassDB::bind_method(D_METHOD("get_friction"), &VehicleBody::get_friction);

	ClassDB::bind_method(D_METHOD("set_engine_force", "engine_force"), &VehicleBody::set_engine_force);
	ClassDB::bind_method(D_METHOD("get_engine_force"), &VehicleBody::get_engine_force);

	ClassDB::bind_method(D_METHOD("set_brake", "brake"), &VehicleBody::set_brake);
	ClassDB::bind_method(D_METHOD("get_brake"), &VehicleBody::get_brake);

	ClassDB::bind_method(D_METHOD("set_steering", "steering"), &VehicleBody::set_steering);
	ClassDB::bind_method(D_METHOD("get_steering"), &VehicleBody::get_steering);

	ClassDB::bind_method(D_METHOD("get_linear_velocity"), &VehicleBody::get_linear_velocity);

	ClassDB::bind_method(D_METHOD("_direct_state_changed"), &VehicleBody::_direct_state_changed);

	ADD_GROUP("Motion", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "engine_force", PROPERTY_HINT_RANGE, "0.00,1024.0,0.01"), "set_engine_force", "get_engine_force");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "brake", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_brake", "get_brake");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "steering", PROPERTY_HINT_RANGE, "-180,180.0,0.01"), "set_steering", "get_steering");
	ADD_GROUP("Mass", "");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "mass", PROPERTY_HINT_RANGE, "0.01,65536,0.01"), "set_mass", "get_mass");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "friction", PROPERTY_HINT_RANGE, "0.01,1,0.01"), "set_friction", "get_friction");
}

VehicleBody::VehicleBody() :
		PhysicsBody(PhysicsServer::BODY_MODE_RIGID) {

	m_pitchControl = 0;
	m_currentVehicleSpeedKmHour = real_t(0.);
	m_steeringValue = real_t(0.);

	engine_force = 0;
	brake = 0;

	friction = 1;

	ccd = false;

	exclude.insert(get_rid());
	PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(), this, "_direct_state_changed");

	set_mass(40);
}
