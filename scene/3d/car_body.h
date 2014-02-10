/*************************************************************************/
/*  car_body.h                                                           */
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
#ifndef CAR_BODY_H
#define CAR_BODY_H

#include "scene/3d/physics_body.h"


class CarBody;

class CarWheel : public Spatial {

	OBJ_TYPE(CarWheel,Spatial);	

friend class CarBody;
	real_t side_friction;
	real_t forward_friction;
	real_t travel;
	real_t radius;
	real_t resting_frac;
	real_t damping_frac;
	int num_rays;
	Transform local_xform;

	CarBody *body;

	float angVel;
	float steerAngle;
	float torque;
	float driveTorque;
	float axisAngle;
	float upSpeed; // speed relative to the car
	bool locked;
	// last frame stuff
	float lastDisplacement;
	float angVelForGrip;
	bool lastOnFloor;

	bool type_drive;
	bool type_steer;


protected:
	void update(real_t dt);
	bool add_forces(PhysicsDirectBodyState *s);
	void _notification(int p_what);
	static void _bind_methods();

public:

	void set_side_friction(real_t p_friction);
	void set_forward_friction(real_t p_friction);
	void set_travel(real_t p_travel);
	void set_radius(real_t p_radius);
	void set_resting_frac(real_t p_frac);
	void set_damping_frac(real_t p_frac);
	void set_num_rays(real_t p_rays);

	real_t get_side_friction() const;
	real_t get_forward_friction() const;
	real_t get_travel() const;
	real_t get_radius() const;
	real_t get_resting_frac() const;
	real_t get_damping_frac() const;
	int get_num_rays() const;

	void set_type_drive(bool p_enable);
	bool is_type_drive() const;

	void set_type_steer(bool p_enable);
	bool is_type_steer() const;

	CarWheel();

};



class CarBody : public PhysicsBody {

	OBJ_TYPE(CarBody,PhysicsBody);

	real_t mass;
	real_t friction;

	Vector3 linear_velocity;
	Vector3  angular_velocity;
	bool ccd;

	real_t max_steer_angle;
	real_t steer_rate;
	int wheel_num_rays;
	real_t drive_torque;

// control stuff
	real_t target_steering;
	real_t target_accelerate;

	bool forward_drive;
	bool backward_drive;

	real_t steering;
	real_t accelerate;
	real_t hand_brake;
	Set<RID> exclude;


friend class CarWheel;
	Vector<CarWheel*> wheels;

	static void _bind_methods();

	void _direct_state_changed(Object *p_state);
public:


	void set_mass(real_t p_mass);
	real_t get_mass() const;

	void set_friction(real_t p_friction);
	real_t get_friction() const;

	void set_max_steer_angle(real_t p_angle);
	void set_steer_rate(real_t p_rate);
	void set_drive_torque(real_t p_torque);

	real_t get_max_steer_angle() const;
	real_t get_steer_rate() const;
	real_t get_drive_torque() const;


	void set_target_steering(float p_steering);
	void set_target_accelerate(float p_accelerate);
	void set_hand_brake(float p_amont);

	real_t get_target_steering() const;
	real_t get_target_accelerate() const;
	real_t get_hand_brake() const;


	CarBody();
};

#endif // CAR_BODY_H
