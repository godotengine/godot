/*************************************************************************/
/*  main_timer_sync.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef MAIN_TIMER_SYNC_H
#define MAIN_TIMER_SYNC_H

#include "core/engine.h"

struct MainFrameTime {
	float idle_step; // time to advance idles for (argument to process())
	int physics_steps; // number of times to iterate the physics engine
	float interpolation_fraction; // fraction through the current physics tick

	void clamp_idle(float min_idle_step, float max_idle_step);
};

class MainTimerSync {
	// wall clock time measured on the main thread
	uint64_t last_cpu_ticks_usec = 0;
	uint64_t current_cpu_ticks_usec = 0;

	// logical game time since last physics timestep
	float time_accum = 0;

	// current difference between wall clock time and reported sum of idle_steps
	float time_deficit = 0;

	// number of frames back for keeping accumulated physics steps roughly constant.
	// value of 12 chosen because that is what is required to make 144 Hz monitors
	// behave well with 60 Hz physics updates. The only worse commonly available refresh
	// would be 85, requiring CONTROL_STEPS = 17.
	static const int CONTROL_STEPS = 12;

	// sum of physics steps done over the last (i+1) frames
	int accumulated_physics_steps[CONTROL_STEPS];

	// typical value for accumulated_physics_steps[i] is either this or this plus one
	int typical_physics_steps[CONTROL_STEPS];

	int fixed_fps = 0;

protected:
	// returns the fraction of p_frame_slice required for the timer to overshoot
	// before advance_core considers changing the physics_steps return from
	// the typical values as defined by typical_physics_steps
	float get_physics_jitter_fix();

	// gets our best bet for the average number of physics steps per render frame
	// return value: number of frames back this data is consistent
	int get_average_physics_steps(float &p_min, float &p_max);

	// advance physics clock by p_idle_step, return appropriate number of steps to simulate
	MainFrameTime advance_core(float p_frame_slice, int p_iterations_per_second, float p_idle_step);

	// calls advance_core, keeps track of deficit it adds to animaption_step, make sure the deficit sum stays close to zero
	MainFrameTime advance_checked(float p_frame_slice, int p_iterations_per_second, float p_idle_step);

	// determine wall clock step since last iteration
	float get_cpu_idle_step();

public:
	MainTimerSync();

	// start the clock
	void init(uint64_t p_cpu_ticks_usec);
	// set measured wall clock time
	void set_cpu_ticks_usec(uint64_t p_cpu_ticks_usec);
	//set fixed fps
	void set_fixed_fps(int p_fixed_fps);

	// advance one frame, return timesteps to take
	MainFrameTime advance(float p_frame_slice, int p_iterations_per_second);
};

#endif // MAIN_TIMER_SYNC_H
