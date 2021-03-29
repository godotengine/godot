/*************************************************************************/
/*  main_timer_sync.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

void MainFrameTime::clamp_process_step(float min_process_step, float max_process_step) {
	if (process_step < min_process_step) {
		process_step = min_process_step;
	} else if (process_step > max_process_step) {
		process_step = max_process_step;
	}
}

/////////////////////////////////

// returns the fraction of p_physics_step required for the timer to overshoot
// before advance_core considers changing the physics_steps return from
// the typical values as defined by typical_physics_steps
float MainTimerSync::get_physics_jitter_fix() {
	return Engine::get_singleton()->get_physics_jitter_fix();
}

// gets our best bet for the average number of physics steps per render frame
// return value: number of frames back this data is consistent
int MainTimerSync::get_average_physics_steps(float &p_min, float &p_max) {
	p_min = typical_physics_steps[0];
	p_max = p_min + 1;

	for (int i = 1; i < CONTROL_STEPS; ++i) {
		const float typical_lower = typical_physics_steps[i];
		const float current_min = typical_lower / (i + 1);
		if (current_min > p_max) {
			return i; // bail out of further restrictions would void the interval
		} else if (current_min > p_min) {
			p_min = current_min;
		}
		const float current_max = (typical_lower + 1) / (i + 1);
		if (current_max < p_min) {
			return i;
		} else if (current_max < p_max) {
			p_max = current_max;
		}
	}

	return CONTROL_STEPS;
}

// advance physics clock by p_process_step, return appropriate number of steps to simulate
MainFrameTime MainTimerSync::advance_core(float p_physics_step, int p_physics_fps, float p_process_step) {
	MainFrameTime ret;

	ret.process_step = p_process_step;

	// simple determination of number of physics iteration
	time_accum += ret.process_step;
	ret.physics_steps = floor(time_accum * p_physics_fps);

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

		if (steps_left_to_match_typical > min_typical_steps) {
			min_typical_steps = steps_left_to_match_typical;
		}
		if (steps_left_to_match_typical + 1 < max_typical_steps) {
			max_typical_steps = steps_left_to_match_typical + 1;
		}
	}

	// try to keep it consistent with previous iterations
	if (ret.physics_steps < min_typical_steps) {
		const int max_possible_steps = floor((time_accum)*p_physics_fps + get_physics_jitter_fix());
		if (max_possible_steps < min_typical_steps) {
			ret.physics_steps = max_possible_steps;
			update_typical = true;
		} else {
			ret.physics_steps = min_typical_steps;
		}
	} else if (ret.physics_steps > max_typical_steps) {
		const int min_possible_steps = floor((time_accum)*p_physics_fps - get_physics_jitter_fix());
		if (min_possible_steps > max_typical_steps) {
			ret.physics_steps = min_possible_steps;
			update_typical = true;
		} else {
			ret.physics_steps = max_typical_steps;
		}
	}

	time_accum -= ret.physics_steps * p_physics_step;

	// keep track of accumulated step counts
	for (int i = CONTROL_STEPS - 2; i >= 0; --i) {
		accumulated_physics_steps[i + 1] = accumulated_physics_steps[i] + ret.physics_steps;
	}
	accumulated_physics_steps[0] = ret.physics_steps;

	if (update_typical) {
		for (int i = CONTROL_STEPS - 1; i >= 0; --i) {
			if (typical_physics_steps[i] > accumulated_physics_steps[i]) {
				typical_physics_steps[i] = accumulated_physics_steps[i];
			} else if (typical_physics_steps[i] < accumulated_physics_steps[i] - 1) {
				typical_physics_steps[i] = accumulated_physics_steps[i] - 1;
			}
		}
	}

	return ret;
}

// calls advance_core, keeps track of deficit it adds to animaption_step, make sure the deficit sum stays close to zero
MainFrameTime MainTimerSync::advance_checked(float p_physics_step, int p_physics_fps, float p_process_step) {
	if (fixed_fps != -1) {
		p_process_step = 1.0 / fixed_fps;
	}

	// compensate for last deficit
	p_process_step += time_deficit;

	MainFrameTime ret = advance_core(p_physics_step, p_physics_fps, p_process_step);

	// we will do some clamping on ret.process_step and need to sync those changes to time_accum,
	// that's easiest if we just remember their fixed difference now
	const double process_minus_accum = ret.process_step - time_accum;

	// first, least important clamping: keep ret.process_step consistent with typical_physics_steps.
	// this smoothes out the process steps and culls small but quick variations.
	{
		float min_average_physics_steps, max_average_physics_steps;
		int consistent_steps = get_average_physics_steps(min_average_physics_steps, max_average_physics_steps);
		if (consistent_steps > 3) {
			ret.clamp_process_step(min_average_physics_steps * p_physics_step, max_average_physics_steps * p_physics_step);
		}
	}

	// second clamping: keep abs(time_deficit) < jitter_fix * frame_slise
	float max_clock_deviation = get_physics_jitter_fix() * p_physics_step;
	ret.clamp_process_step(p_process_step - max_clock_deviation, p_process_step + max_clock_deviation);

	// last clamping: make sure time_accum is between 0 and p_physics_step for consistency between physics and process
	ret.clamp_process_step(process_minus_accum, process_minus_accum + p_physics_step);

	// restore time_accum
	time_accum = ret.process_step - process_minus_accum;

	// track deficit
	time_deficit = p_process_step - ret.process_step;

	// p_physics_step is 1.0 / iterations_per_sec
	// i.e. the time in seconds taken by a physics tick
	ret.interpolation_fraction = time_accum / p_physics_step;

	return ret;
}

// determine wall clock step since last iteration
float MainTimerSync::get_cpu_process_step() {
	uint64_t cpu_ticks_elapsed = current_cpu_ticks_usec - last_cpu_ticks_usec;
	last_cpu_ticks_usec = current_cpu_ticks_usec;

	return cpu_ticks_elapsed / 1000000.0;
}

MainTimerSync::MainTimerSync() {
	for (int i = CONTROL_STEPS - 1; i >= 0; --i) {
		typical_physics_steps[i] = i;
		accumulated_physics_steps[i] = i;
	}
}

// start the clock
void MainTimerSync::init(uint64_t p_cpu_ticks_usec) {
	current_cpu_ticks_usec = last_cpu_ticks_usec = p_cpu_ticks_usec;
}

// set measured wall clock time
void MainTimerSync::set_cpu_ticks_usec(uint64_t p_cpu_ticks_usec) {
	current_cpu_ticks_usec = p_cpu_ticks_usec;
}

void MainTimerSync::set_fixed_fps(int p_fixed_fps) {
	fixed_fps = p_fixed_fps;
}

// advance one physics frame, return timesteps to take
MainFrameTime MainTimerSync::advance(float p_physics_step, int p_physics_fps) {
	float cpu_process_step = get_cpu_process_step();

	return advance_checked(p_physics_step, p_physics_fps, cpu_process_step);
}
