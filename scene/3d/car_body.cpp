/*************************************************************************/
/*  car_body.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "car_body.h"

#define DEG2RADMUL (Math_PI/180.0)
#define RAD2DEGMUL (180.0/Math_PI)

void CarWheel::_notification(int p_what) {


	if (p_what==NOTIFICATION_ENTER_SCENE) {

		if (!get_parent())
			return;
		CarBody *cb = get_parent()->cast_to<CarBody>();
		if (!cb)
			return;
		body=cb;
		local_xform=get_transform();
		cb->wheels.push_back(this);
	}
	if (p_what==NOTIFICATION_EXIT_SCENE) {

		if (!get_parent())
			return;
		CarBody *cb = get_parent()->cast_to<CarBody>();
		if (!cb)
			return;
		cb->wheels.erase(this);
		body=NULL;
	}
}

void CarWheel::set_side_friction(real_t p_friction) {

	side_friction=p_friction;
}
void CarWheel::set_forward_friction(real_t p_friction)  {

	forward_friction=p_friction;
}
void CarWheel::set_travel(real_t p_travel) {

	travel=p_travel;
	update_gizmo();

}
void CarWheel::set_radius(real_t p_radius) {

	radius=p_radius;
	update_gizmo();

}
void CarWheel::set_resting_frac(real_t p_frac) {

	resting_frac=p_frac;
}
void CarWheel::set_damping_frac(real_t p_frac) {

	damping_frac=p_frac;
}
void CarWheel::set_num_rays(real_t p_rays) {

	num_rays=p_rays;
}

real_t CarWheel::get_side_friction() const{

	return side_friction;
}
real_t CarWheel::get_forward_friction() const{

	return forward_friction;
}
real_t CarWheel::get_travel() const{

	return travel;
}
real_t CarWheel::get_radius() const{

	return radius;
}
real_t CarWheel::get_resting_frac() const{

	return resting_frac;
}
real_t CarWheel::get_damping_frac() const{

	return damping_frac;
}

int CarWheel::get_num_rays() const{

	return num_rays;
}


void CarWheel::update(real_t dt) {


	if (dt <= 0.0f)
		return;

	float origAngVel = angVel;

	if (locked)
	{
		angVel = 0;
		torque = 0;
	}
	else
	{

		float wheelMass = 0.03f * body->mass;
		float inertia = 0.5f * (radius * radius) * wheelMass;

		angVel += torque * dt / inertia;
		torque = 0;

		// prevent friction from reversing dir - todo do this better
		// by limiting the torque
		if (((origAngVel > angVelForGrip) && (angVel < angVelForGrip)) ||
				((origAngVel < angVelForGrip) && (angVel > angVelForGrip)))
			angVel = angVelForGrip;

		angVel += driveTorque * dt / inertia;
		driveTorque = 0;

		float maxAngVel = 200;
		print_line("angvel: "+rtos(angVel));
		angVel = CLAMP(angVel, -maxAngVel, maxAngVel);

		axisAngle += Math::rad2deg(dt * angVel);
	}
}

bool CarWheel::add_forces(PhysicsDirectBodyState *s) {


	Vector3 force;

	PhysicsDirectSpaceState *space = s->get_space_state();

	Transform world = s->get_transform() * local_xform;

	// OpenGl has differnet row/column order for matrixes than XNA has ..
	//Vector3 wheelFwd = world.get_basis().get_axis(Vector3::AXIS_Z);
	//Vector3 wheelFwd = RotationMatrix(mSteerAngle, worldAxis) * carBody.GetOrientation().GetCol(0);
	Vector3 wheelUp = world.get_basis().get_axis(Vector3::AXIS_Y);
	Vector3 wheelFwd = Matrix3(wheelUp,Math::deg2rad(steerAngle)).xform( world.get_basis().get_axis(Vector3::AXIS_Z) );
	Vector3 wheelLeft = -wheelUp.cross(wheelFwd).normalized();
	Vector3 worldPos = world.origin;
	Vector3 worldAxis = wheelUp;

	// start of ray
	float rayLen = 2.0f * radius + travel;
	Vector3 wheelRayEnd = worldPos - radius * worldAxis;
	Vector3 wheelRayBegin = wheelRayEnd + rayLen * worldAxis;
	//wheelRayEnd = -rayLen * worldAxis;

	//Assert(PhysicsSystem.CurrentPhysicsSystem);


	///Assert(collSystem);
 ///
	const int maxNumRays = 32;

	int numRaysUse = MIN(num_rays, maxNumRays);

	// adjust the start position of the ray - divide the wheel into numRays+2
	// rays, but don't use the first/last.
	float deltaFwd = (2.0f * radius) / (numRaysUse + 1);
	float deltaFwdStart = deltaFwd;

	float fracs[maxNumRays];
	Vector3 segmentEnds[maxNumRays];
	Vector3 groundPositions[maxNumRays];
	Vector3 groundNormals[maxNumRays];


	lastOnFloor = false;
	int bestIRay = 0;
	int iRay;


	for (iRay = 0; iRay < numRaysUse; ++iRay)
	{
		fracs[iRay] = 1e20;
		// work out the offset relative to the middle ray
		float distFwd = (deltaFwdStart + iRay * deltaFwd) - radius;
		float zOffset = radius * (1.0f - (float)Math::cos( Math::deg2rad( 90.0f * (distFwd / radius))));

		segmentEnds[iRay] = wheelRayEnd + distFwd * wheelFwd + zOffset * wheelUp;


		PhysicsDirectSpaceState::RayResult rr;

		bool collided = space->intersect_ray(wheelRayBegin,segmentEnds[iRay],rr,body->exclude);


		if (collided){
			lastOnFloor = true;
			groundPositions[iRay]=rr.position;
			groundNormals[iRay]=rr.normal;
			fracs[iRay] = ((wheelRayBegin-rr.position).length() /  (wheelRayBegin-wheelRayEnd).length());
			if (fracs[iRay] < fracs[bestIRay])
				bestIRay = iRay;
		}
	}


	if (!lastOnFloor)
		return false;

	//Assert(bestIRay < numRays);

	// use the best one
	Vector3 groundPos = groundPositions[bestIRay];
	float frac = fracs[bestIRay];

	//  const Vector3 groundNormal = (worldPos - segments[bestIRay].GetEnd()).NormaliseSafe();
	//  const Vector3 groundNormal = groundNormals[bestIRay];


	Vector3 groundNormal = worldAxis;

	if (numRaysUse > 1)
	{
		for (iRay = 0; iRay < numRaysUse; ++iRay)
		{
			if (fracs[iRay] <= 1.0f)
			{
				groundNormal += (1.0f - fracs[iRay]) * (worldPos - segmentEnds[iRay]);
			}
		}

		groundNormal.normalize();

	}
	else
	{
		groundNormal = groundNormals[bestIRay];
	}



	float spring = (body->mass/body->wheels.size()) * s->get_total_gravity().length() / (resting_frac * travel);

	float displacement = rayLen * (1.0f - frac);
	displacement = CLAMP(displacement, 0, travel);



	float displacementForceMag = displacement * spring;

	// reduce force when suspension is par to ground
	displacementForceMag *= groundNormals[bestIRay].dot(worldAxis);

	// apply damping
	float damping = 2.0f * (float)Math::sqrt(spring * body->mass);
	damping /= body->wheels.size(); // assume wheels act together
	damping *= damping_frac;  // a bit bouncy

	float upSpeed = (displacement - lastDisplacement) / s->get_step();

	float dampingForceMag = upSpeed * damping;

	float totalForceMag = displacementForceMag + dampingForceMag;

	if (totalForceMag < 0.0f) totalForceMag = 0.0f;

	Vector3 extraForce = totalForceMag * worldAxis;


	force += extraForce;
	// side-slip friction and drive force. Work out wheel- and floor-relative coordinate frame
	Vector3 groundUp = groundNormal;
	Vector3 groundLeft = groundNormal.cross(wheelFwd).normalized();

	Vector3 groundFwd = groundLeft.cross(groundUp);

	Vector3 wheelPointVel = s->get_linear_velocity() +
			(s->get_angular_velocity()).cross(s->get_transform().basis.xform(local_xform.origin));// * mPos);

	Vector3 rimVel = -angVel * wheelLeft.cross(groundPos - worldPos);
	wheelPointVel += rimVel;

	// if sitting on another body then adjust for its velocity.
	/*if (worldBody != null)
	{
		Vector3 worldVel = worldBody.Velocity +
				Vector3.Cross(worldBody.AngularVelocity, groundPos - worldBody.Position);

		wheelPointVel -= worldVel;
	}*/

	// sideways forces
	float noslipVel = 0.2f;
	float slipVel = 0.4f;
	float slipFactor = 0.7f;

	float smallVel = 3;
	float friction = side_friction;

	float sideVel = wheelPointVel.dot(groundLeft);

	if ((sideVel > slipVel) || (sideVel < -slipVel))
		friction *= slipFactor;
	else
		if ((sideVel > noslipVel) || (sideVel < -noslipVel))
			friction *= 1.0f - (1.0f - slipFactor) * (Math::absf(sideVel) - noslipVel) / (slipVel - noslipVel);

	if (sideVel < 0.0f)
		friction *= -1.0f;

	if (Math::absf(sideVel) < smallVel)
		friction *= Math::absf(sideVel) / smallVel;

	float sideForce = -friction * totalForceMag;

	extraForce = sideForce * groundLeft;
	force += extraForce;
	// fwd/back forces
	friction = forward_friction;
	float fwdVel = wheelPointVel.dot(groundFwd);

	if ((fwdVel > slipVel) || (fwdVel < -slipVel))
		friction *= slipFactor;
	else
		if ((fwdVel > noslipVel) || (fwdVel < -noslipVel))
			friction *= 1.0f - (1.0f - slipFactor) * (Math::absf(fwdVel) - noslipVel) / (slipVel - noslipVel);

	if (fwdVel < 0.0f)
		friction *= -1.0f;

	if (Math::absf(fwdVel) < smallVel)
		friction *= Math::absf(fwdVel) / smallVel;

	float fwdForce = -friction * totalForceMag;

	extraForce = fwdForce * groundFwd;
	force += extraForce;


	//if (!force.IsSensible())
	//{
	//  TRACE_FILE_IF(ONCE_1)
	//    TRACE("Bad force in car wheel\n");
	//  return true;
	//}

	// fwd force also spins the wheel
	Vector3 wheelCentreVel = s->get_linear_velocity() +
			(s->get_angular_velocity()).cross(s->get_transform().basis.xform(local_xform.origin));

	angVelForGrip = wheelCentreVel.dot(groundFwd) / radius;
	torque += -fwdForce * radius;

	// add force to car
//	carBody.AddWorldForce(force, groundPos);

	s->add_force(force,(groundPos-s->get_transform().origin));

	// add force to the world
	/*
	if (worldBody != null && !worldBody.Immovable)
	{
		// todo get the position in the right place...
		// also limit the velocity that this force can produce by looking at the
		// mass/inertia of the other object
		float maxOtherBodyAcc = 500.0f;
		float maxOtherBodyForce = maxOtherBodyAcc * worldBody.Mass;

		if (force.LengthSquared() > (maxOtherBodyForce * maxOtherBodyForce))
			force *= maxOtherBodyForce / force.Length();

		worldBody.AddWorldForce(-force, groundPos);
	}*/

	Transform wheel_xf = local_xform;
	wheel_xf.origin += wheelUp * displacement;
	wheel_xf.basis = wheel_xf.basis * Matrix3(Vector3(0,1,0),Math::deg2rad(steerAngle));
	//wheel_xf.basis = wheel_xf.basis * Matrix3(wheel_xf.basis[0],-Math::deg2rad(axisAngle));

	set_transform(wheel_xf);
	lastDisplacement=displacement;
	return true;

}

void CarWheel::set_type_drive(bool p_enable) {

	type_drive=p_enable;
}

bool CarWheel::is_type_drive() const {

	return type_drive;
}

void CarWheel::set_type_steer(bool p_enable) {

	type_steer=p_enable;
}

bool CarWheel::is_type_steer() const {

	return type_steer;
}


void CarWheel::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_side_friction","friction"),&CarWheel::set_side_friction);
	ObjectTypeDB::bind_method(_MD("set_forward_friction","friction"),&CarWheel::set_forward_friction);
	ObjectTypeDB::bind_method(_MD("set_travel","distance"),&CarWheel::set_travel);
	ObjectTypeDB::bind_method(_MD("set_radius","radius"),&CarWheel::set_radius);
	ObjectTypeDB::bind_method(_MD("set_resting_frac","frac"),&CarWheel::set_resting_frac);
	ObjectTypeDB::bind_method(_MD("set_damping_frac","frac"),&CarWheel::set_damping_frac);
	ObjectTypeDB::bind_method(_MD("set_num_rays","amount"),&CarWheel::set_num_rays);

	ObjectTypeDB::bind_method(_MD("get_side_friction"),&CarWheel::get_side_friction);
	ObjectTypeDB::bind_method(_MD("get_forward_friction"),&CarWheel::get_forward_friction);
	ObjectTypeDB::bind_method(_MD("get_travel"),&CarWheel::get_travel);
	ObjectTypeDB::bind_method(_MD("get_radius"),&CarWheel::get_radius);
	ObjectTypeDB::bind_method(_MD("get_resting_frac"),&CarWheel::get_resting_frac);
	ObjectTypeDB::bind_method(_MD("get_damping_frac"),&CarWheel::get_damping_frac);
	ObjectTypeDB::bind_method(_MD("get_num_rays"),&CarWheel::get_num_rays);

	ObjectTypeDB::bind_method(_MD("set_type_drive","enable"),&CarWheel::set_type_drive);
	ObjectTypeDB::bind_method(_MD("is_type_drive"),&CarWheel::is_type_drive);

	ObjectTypeDB::bind_method(_MD("set_type_steer","enable"),&CarWheel::set_type_steer);
	ObjectTypeDB::bind_method(_MD("is_type_steer"),&CarWheel::is_type_steer);

	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"type/drive"),_SCS("set_type_drive"),_SCS("is_type_drive"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"type/steer"),_SCS("set_type_steer"),_SCS("is_type_steer"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/side_friction",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_side_friction"),_SCS("get_side_friction"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/forward_friction",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_forward_friction"),_SCS("get_forward_friction"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/travel",PROPERTY_HINT_RANGE,"0.01,1024,0.01"),_SCS("set_travel"),_SCS("get_travel"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/radius",PROPERTY_HINT_RANGE,"0.01,1024,0.01"),_SCS("set_radius"),_SCS("get_radius"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/resting_frac",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_resting_frac"),_SCS("get_resting_frac"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/damping_frac",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_damping_frac"),_SCS("get_damping_frac"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/num_rays",PROPERTY_HINT_RANGE,"1,32,1"),_SCS("set_num_rays"),_SCS("get_num_rays"));

}

CarWheel::CarWheel() {

	side_friction=4.7;
	forward_friction=5.0;
	travel=0.2;
	radius=0.4;
	resting_frac=0.45;
	damping_frac=0.3;
	num_rays=1;

	angVel = 0.0f;
	steerAngle = 0.0f;
	torque = 0.0f;
	driveTorque = 0.0f;
	axisAngle = 0.0f;
	upSpeed = 0.0f;
	locked = false;
	lastDisplacement = 0.0f;
	lastOnFloor = false;
	angVelForGrip = 0.0f;
	angVelForGrip=0;

	type_drive=false;
	type_steer=false;

}

///


void CarBody::set_max_steer_angle(real_t p_angle)  {

	max_steer_angle=p_angle;
}
void CarBody::set_steer_rate(real_t p_rate) {

	steer_rate=p_rate;
}
void CarBody::set_drive_torque(real_t p_torque) {

	drive_torque=p_torque;
}

real_t CarBody::get_max_steer_angle() const{

	return max_steer_angle;
}
real_t CarBody::get_steer_rate() const{

	return steer_rate;
}
real_t CarBody::get_drive_torque() const{

	return drive_torque;
}


void CarBody::set_target_steering(float p_steering) {

	target_steering=p_steering;
}

void CarBody::set_target_accelerate(float p_accelerate) {
	target_accelerate=p_accelerate;
}

void CarBody::set_hand_brake(float p_amont) {

	hand_brake=p_amont;
}

real_t CarBody::get_target_steering() const {

	return target_steering;
}
real_t CarBody::get_target_accelerate() const {

	return target_accelerate;
}
real_t CarBody::get_hand_brake() const {

	return hand_brake;
}


void CarBody::_direct_state_changed(Object *p_state) {

	PhysicsDirectBodyState *state=p_state->cast_to<PhysicsDirectBodyState>();

	float dt = state->get_step();
	AABB aabb;
	int drive_total=0;
	for(int i=0;i<wheels.size();i++) {
		CarWheel *w=wheels[i];
		if (i==0) {
			aabb.pos=w->local_xform.origin;
		} else {
			aabb.expand_to(w->local_xform.origin);
		}
		if (w->type_drive)
			drive_total++;

	}
	// control inputs
	float deltaAccelerate = dt * 4.0f;

	float dAccelerate = target_accelerate - accelerate;
	dAccelerate = CLAMP(dAccelerate, -deltaAccelerate, deltaAccelerate);
	accelerate += dAccelerate;

	float deltaSteering = dt * steer_rate;
	float dSteering = target_steering - steering;
	dSteering = CLAMP(dSteering, -deltaSteering, deltaSteering);
	steering += dSteering;

	// apply these inputs
	float maxTorque = drive_torque;

	float torque_div = drive_total/2;
	if (torque_div>0)
		maxTorque/=torque_div;


	float alpha = ABS(max_steer_angle * steering);
	float angleSgn = steering > 0.0f ? 1.0f : -1.0f;

	int wheels_on_floor=0;

	for(int i=0;i<wheels.size();i++) {

		CarWheel *w=wheels[i];
		if (w->type_drive)
			w->driveTorque+=maxTorque * accelerate;
		w->locked = !w->type_steer && (hand_brake > 0.5f);

		if (w->type_steer) {
			//steering

			bool inner = (steering > 0 && w->local_xform.origin.x > 0) || (steering < 0 && w->local_xform.origin.x < 0);

			if (inner || alpha==0.0) {

				w->steerAngle = (angleSgn * alpha);
			} else {
				float dx = aabb.size.z;
				float dy = aabb.size.x;

				float beta = Math::atan2(dy, dx + (dy / (float)Math::tan(Math::deg2rad(alpha))));
				beta = Math::rad2deg(beta);
				w->steerAngle = (angleSgn * beta);

			}
		}

		if (w->add_forces(state))
			wheels_on_floor++;
		w->update(dt);


	}

	print_line("onfloor: "+itos(wheels_on_floor));


	set_ignore_transform_notification(true);
	set_global_transform(state->get_transform());
	linear_velocity=state->get_linear_velocity();
	angular_velocity=state->get_angular_velocity();
	//active=!state->is_sleeping();
	//if (get_script_instance())
	//	get_script_instance()->call("_integrate_forces",state);
	set_ignore_transform_notification(false);


}

void CarBody::set_mass(real_t p_mass) {

	mass=p_mass;
	PhysicsServer::get_singleton()->body_set_param(get_rid(),PhysicsServer::BODY_PARAM_MASS,mass);
}

real_t CarBody::get_mass() const{

	return mass;
}


void CarBody::set_friction(real_t p_friction) {

	friction=p_friction;
	PhysicsServer::get_singleton()->body_set_param(get_rid(),PhysicsServer::BODY_PARAM_FRICTION,friction);
}

real_t CarBody::get_friction() const{

	return friction;
}


void CarBody::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_max_steer_angle","value"),&CarBody::set_max_steer_angle);
	ObjectTypeDB::bind_method(_MD("set_steer_rate","rate"),&CarBody::set_steer_rate);
	ObjectTypeDB::bind_method(_MD("set_drive_torque","value"),&CarBody::set_drive_torque);

	ObjectTypeDB::bind_method(_MD("get_max_steer_angle"),&CarBody::get_max_steer_angle);
	ObjectTypeDB::bind_method(_MD("get_steer_rate"),&CarBody::get_steer_rate);
	ObjectTypeDB::bind_method(_MD("get_drive_torque"),&CarBody::get_drive_torque);

	ObjectTypeDB::bind_method(_MD("set_target_steering","amount"),&CarBody::set_target_steering);
	ObjectTypeDB::bind_method(_MD("set_target_accelerate","amount"),&CarBody::set_target_accelerate);
	ObjectTypeDB::bind_method(_MD("set_hand_brake","amount"),&CarBody::set_hand_brake);

	ObjectTypeDB::bind_method(_MD("get_target_steering"),&CarBody::get_target_steering);
	ObjectTypeDB::bind_method(_MD("get_target_accelerate"),&CarBody::get_target_accelerate);
	ObjectTypeDB::bind_method(_MD("get_hand_brake"),&CarBody::get_hand_brake);

	ObjectTypeDB::bind_method(_MD("set_mass","mass"),&CarBody::set_mass);
	ObjectTypeDB::bind_method(_MD("get_mass"),&CarBody::get_mass);

	ObjectTypeDB::bind_method(_MD("set_friction","friction"),&CarBody::set_friction);
	ObjectTypeDB::bind_method(_MD("get_friction"),&CarBody::get_friction);

	ObjectTypeDB::bind_method(_MD("_direct_state_changed"),&CarBody::_direct_state_changed);

	ADD_PROPERTY( PropertyInfo(Variant::REAL,"body/mass",PROPERTY_HINT_RANGE,"0.01,65536,0.01"),_SCS("set_mass"),_SCS("get_mass"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"body/friction",PROPERTY_HINT_RANGE,"0.01,1,0.01"),_SCS("set_friction"),_SCS("get_friction"));

	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/max_steer_angle",PROPERTY_HINT_RANGE,"1,90,1"),_SCS("set_max_steer_angle"),_SCS("get_max_steer_angle"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/drive_torque",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_drive_torque"),_SCS("get_drive_torque"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"config/steer_rate",PROPERTY_HINT_RANGE,"0.01,64,0.01"),_SCS("set_steer_rate"),_SCS("get_steer_rate"));

	ADD_PROPERTY( PropertyInfo(Variant::REAL,"drive/target_steering",PROPERTY_HINT_RANGE,"-1,1,0.01"),_SCS("set_target_steering"),_SCS("get_target_steering"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"drive/target_accelerate",PROPERTY_HINT_RANGE,"-1,1,0.01"),_SCS("set_target_accelerate"),_SCS("get_target_accelerate"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"drive/hand_brake",PROPERTY_HINT_RANGE,"0,1,0.01"),_SCS("set_hand_brake"),_SCS("get_hand_brake"));

}

CarBody::CarBody() : PhysicsBody(PhysicsServer::BODY_MODE_RIGID) {

	forward_drive=true;
	backward_drive=true;
	max_steer_angle=30;
	steer_rate=1;
	drive_torque=520;

	target_steering=0;
	target_accelerate=0;
	hand_brake=0;

	steering=0;
	accelerate=0;

	mass=1;
	friction=1;

	ccd=false;
//	can_sleep=true;




	exclude.insert(get_rid());
	PhysicsServer::get_singleton()->body_set_force_integration_callback(get_rid(),this,"_direct_state_changed");


}
