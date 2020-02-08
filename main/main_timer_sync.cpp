/*************************************************************************/
/*  main_timer_sync.cpp                                                  */
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

#include "main_timer_sync.h"
#include "core/project_settings.h"

/////////////////////////////////////////////////////////////////

void MainFrameTime::clamp_frame_delta(float p_min, float p_max) {
	if (scaled_frame_delta < p_min) {
		scaled_frame_delta = p_min;
	} else if (scaled_frame_delta > p_max) {
		scaled_frame_delta = p_max;
	}
}

MainFrameTime::MainFrameTime() {
	// initialize the timing info with some sensible values, no matter what timestep method is used
	scaled_frame_delta = 0.0f;
	physics_steps = 1;
	physics_fixed_step_delta = 0.0f;
	physics_variable_step = false;
	physics_variable_step_delta = 0.0f;
	interpolation_fraction = 0.0f;
}

/////////////////////////////////////////////////////////////////

MainTimerSync::MainTimerSync() :
		last_cpu_ticks_usec(0),
		current_cpu_ticks_usec(0),
		fixed_fps(0) {
	stretch_ticks = true;
	method = 0;
}

// start the clock
void MainTimerSync::init(uint64_t p_cpu_ticks_usec) {
	current_cpu_ticks_usec = last_cpu_ticks_usec = p_cpu_ticks_usec;

	// Just read the project settings once in init rather than every frame.
	// Note that this is assuming that ticking is not desired within the editor.
	// If ticking is to be changed within the editor, MainTimerSync will need to be
	// informed somehow (e.g. calling this function from here onwards).
	// This is not requested from the ProjectSettings every frame for efficiency,
	// primarily because the timestep method is specified as a string (for future expandability
	// and backward compatibility)

	// which timestep method should we use?
	String sz = ProjectSettings::get_singleton()->get("physics/common/timestep/method");

	// default
	method = &ts_jitter_fix;

	if (sz == "Jitter Fix") {
		method = &ts_jitter_fix;
	}

	if (sz == "Fixed") {
		method = &ts_fixed;
	}

	if (sz == "Semi Fixed") {
		method = &ts_semi_fixed;
	}

	// whether to stretch the physics ticks when using global time_scale
	stretch_ticks = Engine::get_singleton()->get_physics_stretch_ticks();
}

// determine wall clock step since last iteration
float MainTimerSync::get_cpu_idle_step() {
	uint64_t delta = current_cpu_ticks_usec - last_cpu_ticks_usec;
	last_cpu_ticks_usec = current_cpu_ticks_usec;

	// add delta smoothing here... NYI

	// return as a float in seconds
	return delta / 1000000.0;
}

// advance one frame, return timesteps to take
void MainTimerSync::advance(MainFrameTime &r_mft, int p_iterations_per_second, uint64_t p_cpu_ticks_usec, int p_fixed_fps) {

	// set measured wall clock time
	current_cpu_ticks_usec = p_cpu_ticks_usec;
	fixed_fps = p_fixed_fps;

	// safety for divide by zero, might not be needed
	if (p_iterations_per_second <= 0)
		p_iterations_per_second = 1;

	// convert p_iterations_per_second to a float because we may need to adjust according to timescale
	float ticks_per_sec = p_iterations_per_second;
	float frame_slice_orig = 1.0f / ticks_per_sec;

	// common to all methods
	float delta;
	if (fixed_fps <= 0) {
		delta = get_cpu_idle_step();
	} else {
		delta = 1.0f / fixed_fps;
	}

	// handle global timescale as part of the physics ticking because we may want to adjust the number
	// of physics ticks
	float time_scale = Engine::get_singleton()->get_time_scale();

	// if time scaling is active
	if (time_scale != 1.0f) {
		// adjust the delta according to the timescale
		delta *= time_scale;

		// Tick stretching....
		// If in legacy mode, stretch ticks so that the same number occur in the frame as with no time_scale applied. This will
		// give unpredictable physics results at different timescales.

		// Alteratively, not stretching ticks will result in more (or less) ticks taking place in the frame according to the timescale.
		// This will give consistent physics (just sped up or slowed down) but will suffer from judder at low timescales when using
		// fixed timestep unless interpolation is used.
		if (stretch_ticks) {
			// prevent divide by zero
			// this is just some arbitrary epsilon to prevent divide by zero (can be changed)
			if (time_scale < 0.0001f)
				time_scale = 0.0001f;

			ticks_per_sec *= 1.0f / time_scale;
		}
	}

	float frame_slice_scaled = 1.0f / ticks_per_sec;

	//  should never happen, but just in case
	if (!method) {
		WARN_PRINT("MainTimerSync - Must call init() before calling advance()");
		method = &ts_jitter_fix;
	}

	// use the currently selected method to do the timestepping
	method->advance(r_mft, delta, frame_slice_scaled, ticks_per_sec);

	// limit the number of physics steps to prevent runaway physics
	static const int max_physics_steps = 8;
	if (fixed_fps == -1 && r_mft.physics_steps > max_physics_steps) {
		// this must use the SCALED frame_slice, because at this stage number of ticks is
		// dependent on the scaled frame_slice (as the overall delta is also scaled)
		r_mft.scaled_frame_delta -= (r_mft.physics_steps - max_physics_steps) * frame_slice_scaled;
		r_mft.physics_steps = max_physics_steps;
	}

	// return the actual used physics step delta, because this
	// may have changed because of time_scale
	if (stretch_ticks) {
		r_mft.physics_fixed_step_delta = frame_slice_scaled;
	} else {
		// retain original tick delta (e.g. deterministic bullet time)
		r_mft.physics_fixed_step_delta = frame_slice_orig;

		// variable time
		if (r_mft.physics_variable_step) {
			// in this special case, the variable step delta is stored as a fraction of the SCALED frame slice,
			// so needs to have this scale removed and rescaled to the original frame slice

			// fraction through the whole tick the variable tick is
			float f = r_mft.physics_variable_step_delta / frame_slice_scaled;

			// rescale to match the original tick size
			r_mft.physics_variable_step_delta = f * frame_slice_orig;
		}
	}
}

/////////////////////////////////////////////////////////////////

// advance one frame, return timesteps to take
void MainTimerSync::Timestep_JitterFix::advance(MainFrameTime &r_mft, float p_idle_step, float p_frame_slice, float p_iterations_per_second) {
	// calls advance_core, keeps track of deficit it adds to animaption_step, make sure the deficit sum stays close to zero

	// compensate for last deficit
	p_idle_step += time_deficit;

	advance_core(r_mft, p_frame_slice, p_iterations_per_second, p_idle_step);

	// we will do some clamping on r_mft.frame_delta and need to sync those changes to time_accum,
	// that's easiest if we just remember their fixed difference now
	const double idle_minus_accum = r_mft.scaled_frame_delta - time_accum;

	// first, least important clamping: keep r_mft.frame_delta consistent with typical_physics_steps.
	// this smoothes out the idle steps and culls small but quick variations.
	{
		float min_average_physics_steps, max_average_physics_steps;
		int consistent_steps = get_average_physics_steps(min_average_physics_steps, max_average_physics_steps);
		if (consistent_steps > 3) {
			r_mft.clamp_frame_delta(min_average_physics_steps * p_frame_slice, max_average_physics_steps * p_frame_slice);
		}
	}

	// second clamping: keep abs(time_deficit) < jitter_fix * frame_slise
	float max_clock_deviation = get_physics_jitter_fix() * p_frame_slice;
	r_mft.clamp_frame_delta(p_idle_step - max_clock_deviation, p_idle_step + max_clock_deviation);

	// last clamping: make sure time_accum is between 0 and p_frame_slice for consistency between physics and idle
	r_mft.clamp_frame_delta(idle_minus_accum, idle_minus_accum + p_frame_slice);

	// restore time_accum
	time_accum = r_mft.scaled_frame_delta - idle_minus_accum;

	// track deficit
	time_deficit = p_idle_step - r_mft.scaled_frame_delta;

	// we will try and work out what is the interpolation fraction
	// note this is assuming jitter fix is completely turned off when set to 0.0. Is it?
	r_mft.interpolation_fraction = time_accum / (1.0f / p_iterations_per_second);
}

MainTimerSync::Timestep_JitterFix::Timestep_JitterFix() :
		time_accum(0),
		time_deficit(0) {
	for (int i = CONTROL_STEPS - 1; i >= 0; --i) {
		typical_physics_steps[i] = i;
		accumulated_physics_steps[i] = i;
	}
}

// returns the fraction of p_frame_slice required for the timer to overshoot
// before advance_core considers changing the physics_steps return from
// the typical values as defined by typical_physics_steps
float MainTimerSync::Timestep_JitterFix::get_physics_jitter_fix() {
	return Engine::get_singleton()->get_physics_jitter_fix();
}

// gets our best bet for the average number of physics steps per render frame
// return value: number of frames back this data is consistent
int MainTimerSync::Timestep_JitterFix::get_average_physics_steps(float &p_min, float &p_max) {
	p_min = typical_physics_steps[0];
	p_max = p_min + 1;

	for (int i = 1; i < CONTROL_STEPS; ++i) {
		const float typical_lower = typical_physics_steps[i];
		const float current_min = typical_lower / (i + 1);
		if (current_min > p_max)
			return i; // bail out of further restrictions would void the interval
		else if (current_min > p_min)
			p_min = current_min;
		const float current_max = (typical_lower + 1) / (i + 1);
		if (current_max < p_min)
			return i;
		else if (current_max < p_max)
			p_max = current_max;
	}

	return CONTROL_STEPS;
}

// advance physics clock by p_idle_step, return appropriate number of steps to simulate
void MainTimerSync::Timestep_JitterFix::advance_core(MainFrameTime &r_mft, float p_frame_slice, float p_iterations_per_second, float p_idle_step) {
	r_mft.scaled_frame_delta = p_idle_step;

	// simple determination of number of physics iteration
	time_accum += r_mft.scaled_frame_delta;
	r_mft.physics_steps = floor(time_accum * p_iterations_per_second);

	int min_typical_steps = typical_physics_steps[0];
	int max_typical_steps = min_typical_steps + 1;

	// given the past recorded steps and typical steps to match, calculate bounds for this
	// step to be typical
	bool update_typical = false;

	for (int i = 0; i < CONTROL_STEPS - 1; ++i) {
		int steps_left_to_match_typical = typical_physics_steps[i + 1] - accumulated_physics_steps[i];
		if (steps_left_to_match_typical > max_typical_steps ||
				steps_left_to_match_typical + 1 < min_typical_steps) {
			update_typical = true;
			break;
		}

		if (steps_left_to_match_typical > min_typical_steps)
			min_typical_steps = steps_left_to_match_typical;
		if (steps_left_to_match_typical + 1 < max_typical_steps)
			max_typical_steps = steps_left_to_match_typical + 1;
	}

	// try to keep it consistent with previous iterations
	if (r_mft.physics_steps < min_typical_steps) {
		const int max_possible_steps = floor((time_accum)*p_iterations_per_second + get_physics_jitter_fix());
		if (max_possible_steps < min_typical_steps) {
			r_mft.physics_steps = max_possible_steps;
			update_typical = true;
		} else {
			r_mft.physics_steps = min_typical_steps;
		}
	} else if (r_mft.physics_steps > max_typical_steps) {
		const int min_possible_steps = floor((time_accum)*p_iterations_per_second - get_physics_jitter_fix());
		if (min_possible_steps > max_typical_steps) {
			r_mft.physics_steps = min_possible_steps;
			update_typical = true;
		} else {
			r_mft.physics_steps = max_typical_steps;
		}
	}

	time_accum -= r_mft.physics_steps * p_frame_slice;

	// keep track of accumulated step counts
	for (int i = CONTROL_STEPS - 2; i >= 0; --i) {
		accumulated_physics_steps[i + 1] = accumulated_physics_steps[i] + r_mft.physics_steps;
	}
	accumulated_physics_steps[0] = r_mft.physics_steps;

	if (update_typical) {
		for (int i = CONTROL_STEPS - 1; i >= 0; --i) {
			if (typical_physics_steps[i] > accumulated_physics_steps[i]) {
				typical_physics_steps[i] = accumulated_physics_steps[i];
			} else if (typical_physics_steps[i] < accumulated_physics_steps[i] - 1) {
				typical_physics_steps[i] = accumulated_physics_steps[i] - 1;
			}
		}
	}
}

/////////////////////////////////////////////////////////////////

void MainTimerSync::Timestep_SemiFixed::advance(MainFrameTime &r_mft, float p_delta, float p_sec_per_tick, float p_iterations_per_second) {

	r_mft.scaled_frame_delta = p_delta;
	float time_available = p_delta;

	r_mft.physics_steps = floor(time_available * p_iterations_per_second);
	time_available -= r_mft.physics_steps * p_sec_per_tick;

	// if there is more than a certain amount leftover, have an extra physics tick
	if (time_available <= 0.0f)
		return;

	r_mft.physics_steps += 1;
	r_mft.physics_variable_step = true;
	r_mft.physics_variable_step_delta = time_available;
}

/////////////////////////////////////////////////////////////////

MainTimerSync::Timestep_Fixed::Timestep_Fixed() {
	time_left_over = 0.0f;
}

// Simple reference implementation of fixed timestep
void MainTimerSync::Timestep_Fixed::advance(MainFrameTime &r_mft, float p_delta, float p_sec_per_tick, float p_iterations_per_second) {

	r_mft.scaled_frame_delta = p_delta;

	float time_available = time_left_over + p_delta;

	r_mft.physics_steps = floor(time_available * p_iterations_per_second);

	time_left_over = time_available - (r_mft.physics_steps * p_sec_per_tick);

	r_mft.interpolation_fraction = time_left_over / p_sec_per_tick;
}
