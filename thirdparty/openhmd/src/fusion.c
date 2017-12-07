/*
 * OpenHMD - Free and Open Source API and drivers for immersive technology.
 * Copyright (C) 2013 Fredrik Hultin.
 * Copyright (C) 2013 Jakob Bornecrantz.
 * Distributed under the Boost 1.0 licence, see LICENSE for full text.
 */

/* Sensor Fusion Implementation */

#include <string.h>
#include "openhmdi.h"

void ofusion_init(fusion* me)
{
	memset(me, 0, sizeof(fusion));
	me->orient.w = 1.0f;

	ofq_init(&me->mag_fq, 20);
	ofq_init(&me->accel_fq, 20);
	ofq_init(&me->ang_vel_fq, 20);

	me->flags = FF_USE_GRAVITY;
	me->grav_gain = 0.05f;
}

void ofusion_update(fusion* me, float dt, const vec3f* ang_vel, const vec3f* accel, const vec3f* mag)
{
	me->ang_vel = *ang_vel;
	me->accel = *accel;
	me->raw_mag = *mag;

	me->mag = *mag;

	vec3f world_accel;
	oquatf_get_rotated(&me->orient, accel, &world_accel);

	me->iterations += 1;
	me->time += dt;

	ofq_add(&me->mag_fq, mag);
	ofq_add(&me->accel_fq, &world_accel);
	ofq_add(&me->ang_vel_fq, ang_vel);

	float ang_vel_length = ovec3f_get_length(ang_vel);

	if(ang_vel_length > 0.0001f){
		vec3f rot_axis =
			{{ ang_vel->x / ang_vel_length, ang_vel->y / ang_vel_length, ang_vel->z / ang_vel_length }};

		float rot_angle = ang_vel_length * dt;

		quatf delta_orient;
		oquatf_init_axis(&delta_orient, &rot_axis, rot_angle);

		oquatf_mult_me(&me->orient, &delta_orient);
	}

	// gravity correction
	if(me->flags & FF_USE_GRAVITY){
		const float gravity_tolerance = .4f, ang_vel_tolerance = .1f;
		const float min_tilt_error = 0.05f, max_tilt_error = 0.01f;

		// if the device is within tolerance levels, count this as the device is level and add to the counter
		// otherwise reset the counter and start over

		me->device_level_count =
			fabsf(ovec3f_get_length(accel) - 9.82f) < gravity_tolerance * 2.0f && ang_vel_length < ang_vel_tolerance
			? me->device_level_count + 1 : 0;

		// device has been level for long enough, grab mean from the accelerometer filter queue (last n values)
		// and use for correction

		if(me->device_level_count > 50){
			me->device_level_count = 0;

			vec3f accel_mean;
			ofq_get_mean(&me->accel_fq, &accel_mean);
			if (ovec3f_get_length(&accel_mean) - 9.82f < gravity_tolerance)
			{
				// Calculate a cross product between what the device
				// thinks is up and what gravity indicates is down.
				// The values are optimized of what we would get out
				// from the cross product.
				vec3f tilt = {{accel_mean.z, 0, -accel_mean.x}};

				ovec3f_normalize_me(&tilt);
				ovec3f_normalize_me(&accel_mean);

				vec3f up = {{0, 1.0f, 0}};
				float tilt_angle = ovec3f_get_angle(&up, &accel_mean);

				if(tilt_angle > max_tilt_error){
					me->grav_error_angle = tilt_angle;
					me->grav_error_axis = tilt;
				}
			}
		}

		// perform gravity tilt correction
		if(me->grav_error_angle > min_tilt_error){
			float use_angle;
			// if less than 2000 iterations have passed, set the up axis to the correction value outright
			if(me->iterations < 2000){
				use_angle = -me->grav_error_angle;
				me->grav_error_angle = 0;
			}

			// otherwise try to correct
			else {
				use_angle = -me->grav_gain * me->grav_error_angle * 0.005f * (5.0f * ang_vel_length + 1.0f);
				me->grav_error_angle += use_angle;
			}

			// perform the correction
			quatf corr_quat, old_orient;
			oquatf_init_axis(&corr_quat, &me->grav_error_axis, use_angle);
			old_orient = me->orient;

			oquatf_mult(&corr_quat, &old_orient, &me->orient);
		}
	}

	// mitigate drift due to floating point
	// inprecision with quat multiplication.
	oquatf_normalize_me(&me->orient);
}
