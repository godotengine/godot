/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Sensor Fusion */

#ifndef FUSION_H
#define FUSION_H

#include "omath.h"

#define FF_USE_GRAVITY 1

typedef struct {
	int state;

	quatf orient;   // orientation
	vec3f accel;    // acceleration
	vec3f ang_vel;  // angular velocity
	vec3f mag;      // magnetometer
	vec3f raw_mag;  // raw magnetometer values

	int iterations;
	float time;

	int flags;

	// filter queues for magnetometer, accelerometers and angular velocity
	filter_queue mag_fq, accel_fq, ang_vel_fq;

	// gravity correction
	int device_level_count;
	float grav_error_angle;
	vec3f grav_error_axis;
	float grav_gain; // amount of correction
} fusion;

void ofusion_init(fusion* me);
void ofusion_update(fusion* me, float dt, const vec3f* ang_vel, const vec3f* accel, const vec3f* mag_field);

#endif
