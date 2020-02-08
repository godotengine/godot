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

// Used to return timing information to main::iteration
struct MainFrameTime {
	MainFrameTime();

	// time to advance idles for (argument to process())
	// timescale has been applied
	float scaled_frame_delta;

	// number of times to iterate the physics engine
	int physics_steps;
	// delta to pass to _physics_process (except for variable steps, see below)
	float physics_fixed_step_delta;

	// for semi fixed and frame based methods,
	// the last physics step can optionally have  variable delta to pass
	// to physics engine
	bool physics_variable_step; // is the last physics step variable?
	float physics_variable_step_delta; // if so, what is the delta on this variable step

	// logical fraction through the current physics tick at the time of the frame
	// useful for fixed timestep interpolation
	float interpolation_fraction;

	void clamp_frame_delta(float p_min, float p_max);
};

/////////////////////////////////////////////////////////

class MainTimerSync {

	// we include the timestep methods as nested classes
	// as they do not need to be visible outside MainTimerSync
	class Timestep_Base {
	public:
		virtual void advance(MainFrameTime &r_mft, float p_delta, float p_sec_per_tick, float p_iterations_per_second) = 0;
		virtual ~Timestep_Base() {}
	};

	class Timestep_JitterFix : public Timestep_Base {
		// logical game time since last physics timestep
		float time_accum;

		// current difference between wall clock time and reported sum of idle_steps
		float time_deficit;

		// number of frames back for keeping accumulated physics steps roughly constant.
		// value of 12 chosen because that is what is required to make 144 Hz monitors
		// behave well with 60 Hz physics updates. The only worse commonly available refresh
		// would be 85, requiring CONTROL_STEPS = 17.
		static const int CONTROL_STEPS = 12;

		// sum of physics steps done over the last (i+1) frames
		int accumulated_physics_steps[CONTROL_STEPS];

		// typical value for accumulated_physics_steps[i] is either this or this plus one
		int typical_physics_steps[CONTROL_STEPS];

	protected:
		// returns the fraction of p_frame_slice required for the timer to overshoot
		// before advance_core considers changing the physics_steps return from
		// the typical values as defined by typical_physics_steps
		float get_physics_jitter_fix();

		// gets our best bet for the average number of physics steps per render frame
		// return value: number of frames back this data is consistent
		int get_average_physics_steps(float &p_min, float &p_max);

		// advance physics clock by p_idle_step, return appropriate number of steps to simulate
		void advance_core(MainFrameTime &r_mft, float p_frame_slice, float p_iterations_per_second, float p_idle_step);

	public:
		Timestep_JitterFix();

		// advance one frame, return timesteps to take
		virtual void advance(MainFrameTime &r_mft, float p_idle_step, float p_frame_slice, float p_iterations_per_second);
	};

	class Timestep_SemiFixed : public Timestep_Base {
	public:
		// advance one frame, return timesteps to take
		virtual void advance(MainFrameTime &r_mft, float p_delta, float p_sec_per_tick, float p_iterations_per_second);
	};

	// reference fixed timestep implementation
	class Timestep_Fixed : public Timestep_Base {
		float time_left_over;

	public:
		Timestep_Fixed();

		// advance one frame, return timesteps to take
		virtual void advance(MainFrameTime &r_mft, float p_delta, float p_sec_per_tick, float p_iterations_per_second);
	};

	// wall clock time measured on the main thread
	uint64_t last_cpu_ticks_usec;
	uint64_t current_cpu_ticks_usec;

	int fixed_fps;

	// whether to stretch the physics ticks when using global time_scale
	bool stretch_ticks;

	// the currently selected timestep method
	Timestep_Base *method;

	Timestep_JitterFix ts_jitter_fix;
	Timestep_Fixed ts_fixed;
	Timestep_SemiFixed ts_semi_fixed;

	// determine wall clock step since last iteration
	float get_cpu_idle_step();

public:
	MainTimerSync();

	// start the clock
	void init(uint64_t p_cpu_ticks_usec);

	// advance one frame, return timesteps to take in r_mft
	void advance(MainFrameTime &r_mft, int p_iterations_per_second, uint64_t p_cpu_ticks_usec, int p_fixed_fps);
};

#endif // MAIN_TIMER_SYNC_H
