/**************************************************************************/
/*  main_timer_sync.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "main_timer_sync.h"

#include "core/math/math_funcs.h"
#include "core/os/os.h"

void MainFrameTime::clamp_idle(float min_idle_step, float max_idle_step) {
	if (idle_step < min_idle_step) {
		idle_step = min_idle_step;
	} else if (idle_step > max_idle_step) {
		idle_step = max_idle_step;
	}
}

/////////////////////////////////

void MainTimerSync::DeltaSmoother::update_refresh_rate_estimator(int64_t p_delta) {
	// the calling code should prevent 0 or negative values of delta
	// (preventing divide by zero)

	// note that if the estimate gets locked, and something external changes this
	// (e.g. user changes to non-vsync in the OS), then the results may be less than ideal,
	// but usually it will detect this via the FPS measurement and not attempt smoothing.
	// This should be a rare occurrence anyway, and will be cured next time user restarts game.
	if (_estimate_locked) {
		return;
	}

	// First average the delta over NUM_READINGS
	_estimator_total_delta += p_delta;
	_estimator_delta_readings++;

	const int NUM_READINGS = 60;

	if (_estimator_delta_readings < NUM_READINGS) {
		return;
	}

	// use average
	p_delta = _estimator_total_delta / NUM_READINGS;

	// reset the averager for next time
	_estimator_delta_readings = 0;
	_estimator_total_delta = 0;

	///////////////////////////////

	int fps = Math::round(1000000.0 / p_delta);

	// initial estimation, to speed up converging, special case we will estimate the refresh rate
	// from the first average FPS reading
	if (_estimated_fps == 0) {
		// below 50 might be chugging loading stuff, or else
		// dropping loads of frames, so the estimate will be inaccurate
		if (fps >= 50) {
			_estimated_fps = fps;
#ifdef GODOT_DEBUG_DELTA_SMOOTHER
			print_line("initial guess (average measured) refresh rate: " + itos(fps));
#endif
		} else {
			// can't get started until above 50
			return;
		}
	}

	// we hit our exact estimated refresh rate.
	// increase our confidence in the estimate.
	if (fps == _estimated_fps) {
		// note that each hit is an average of NUM_READINGS frames
		_hits_at_estimated++;

		if (_estimate_complete && _hits_at_estimated == 20) {
			_estimate_locked = true;
#ifdef GODOT_DEBUG_DELTA_SMOOTHER
			print_line("estimate LOCKED at " + itos(_estimated_fps) + " fps");
#endif
			return;
		}

		// if we are getting pretty confident in this estimate, decide it is complete
		// (it can still be increased later, and possibly lowered but only for a short time)
		if ((!_estimate_complete) && (_hits_at_estimated > 2)) {
			// when the estimate is complete we turn on smoothing
			if (_estimated_fps) {
				_estimate_complete = true;
				_vsync_delta = 1000000 / _estimated_fps;

#ifdef GODOT_DEBUG_DELTA_SMOOTHER
				print_line("estimate complete. vsync_delta " + itos(_vsync_delta) + ", fps " + itos(_estimated_fps));
#endif
			}
		}

#ifdef GODOT_DEBUG_DELTA_SMOOTHER
		if ((_hits_at_estimated % (400 / NUM_READINGS)) == 0) {
			String sz = "hits at estimated : " + itos(_hits_at_estimated) + ", above : " + itos(_hits_above_estimated) + "( " + itos(_hits_one_above_estimated) + " ), below : " + itos(_hits_below_estimated) + " (" + itos(_hits_one_below_estimated) + " )";

			print_line(sz);
		}
#endif

		return;
	}

	const int SIGNIFICANCE_UP = 1;
	const int SIGNIFICANCE_DOWN = 2;

	// we are not usually interested in slowing the estimate
	// but we may have overshot, so make it possible to reduce
	if (fps < _estimated_fps) {
		// micro changes
		if (fps == (_estimated_fps - 1)) {
			_hits_one_below_estimated++;

			if ((_hits_one_below_estimated > _hits_at_estimated) && (_hits_one_below_estimated > SIGNIFICANCE_DOWN)) {
				_estimated_fps--;
				made_new_estimate();
			}

			return;
		} else {
			_hits_below_estimated++;

			// don't allow large lowering if we are established at a refresh rate, as it will probably be dropped frames
			bool established = _estimate_complete && (_hits_at_estimated > 10);

			// macro changes
			// note there is a large barrier to macro lowering. That is because it is more likely to be dropped frames
			// than mis-estimation of the refresh rate.
			if (!established) {
				if (((_hits_below_estimated / 8) > _hits_at_estimated) && (_hits_below_estimated > SIGNIFICANCE_DOWN)) {
					// decrease the estimate
					_estimated_fps--;
					made_new_estimate();
				}
			}

			return;
		}
	}

	// Changes increasing the estimate.
	// micro changes
	if (fps == (_estimated_fps + 1)) {
		_hits_one_above_estimated++;

		if ((_hits_one_above_estimated > _hits_at_estimated) && (_hits_one_above_estimated > SIGNIFICANCE_UP)) {
			_estimated_fps++;
			made_new_estimate();
		}
		return;
	} else {
		_hits_above_estimated++;

		// macro changes
		if ((_hits_above_estimated > _hits_at_estimated) && (_hits_above_estimated > SIGNIFICANCE_UP)) {
			// increase the estimate
			int change = fps - _estimated_fps;
			change /= 2;
			change = MAX(1, change);

			_estimated_fps += change;
			made_new_estimate();
		}
		return;
	}
}

bool MainTimerSync::DeltaSmoother::fps_allows_smoothing(int64_t p_delta) {
	_measurement_time += p_delta;
	_measurement_frame_count++;

	if (_measurement_frame_count == _measurement_end_frame) {
		// only switch on or off if the estimate is complete
		if (_estimate_complete) {
			int64_t time_passed = _measurement_time - _measurement_start_time;

			// average delta
			time_passed /= MEASURE_FPS_OVER_NUM_FRAMES;

			// estimate fps
			if (time_passed) {
				double fps = 1000000.0 / time_passed;
				double ratio = fps / (double)_estimated_fps;

				//print_line("ratio : " + String(Variant(ratio)));

				if ((ratio > 0.95) && (ratio < 1.05)) {
					_measurement_allows_smoothing = true;
				} else {
					_measurement_allows_smoothing = false;
				}
			}
		} // estimate complete

		// new start time for next iteration
		_measurement_start_time = _measurement_time;
		_measurement_end_frame += MEASURE_FPS_OVER_NUM_FRAMES;
	}

	return _measurement_allows_smoothing;
}

int64_t MainTimerSync::DeltaSmoother::smooth_delta(int64_t p_delta) {
	// Conditions to disable smoothing.
	// Note that vsync is a request, it cannot be relied on, the OS may override this.
	// If the OS turns vsync on without vsync in the app, smoothing will not be enabled.
	// If the OS turns vsync off with sync enabled in the app, the smoothing must detect this
	// via the error metric and switch off.
	if (!OS::get_singleton()->is_delta_smoothing_enabled() || !OS::get_singleton()->is_vsync_enabled() || Engine::get_singleton()->is_editor_hint()) {
		return p_delta;
	}

	// Very important, ignore long deltas and pass them back unmodified.
	// This is to deal with resuming after suspend for long periods.
	if (p_delta > 1000000) {
		return p_delta;
	}

	// keep a running guesstimate of the FPS, and turn off smoothing if
	// conditions not close to the estimated FPS
	if (!fps_allows_smoothing(p_delta)) {
		return p_delta;
	}

	// we can't cope with negative deltas .. OS bug on some hardware
	// and also very small deltas caused by vsync being off.
	// This could possibly be part of a hiccup, this value isn't fixed in stone...
	if (p_delta < 1000) {
		return p_delta;
	}

	// note still some vsync off will still get through to this point...
	// and we need to cope with it by not converging the estimator / and / or not smoothing
	update_refresh_rate_estimator(p_delta);

	// no smoothing until we know what the refresh rate is
	if (!_estimate_complete) {
		return p_delta;
	}

	// accumulate the time we have available to use
	_leftover_time += p_delta;

	// how many vsyncs units can we fit?
	int64_t units = _leftover_time / _vsync_delta;

	// a delta must include minimum 1 vsync
	// (if it is less than that, it is either random error or we are no longer running at the vsync rate,
	// in which case we should switch off delta smoothing, or re-estimate the refresh rate)
	units = MAX(units, 1);

	_leftover_time -= units * _vsync_delta;
	// print_line("units " + itos(units) + ", leftover " + itos(_leftover_time/1000) + " ms");

	return units * _vsync_delta;
}

/////////////////////////////////////

// returns the fraction of p_frame_slice required for the timer to overshoot
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
			return i; // bail out if further restrictions would void the interval
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

// advance physics clock by p_idle_step, return appropriate number of steps to simulate
MainFrameTime MainTimerSync::advance_core(float p_frame_slice, int p_iterations_per_second, float p_idle_step) {
	MainFrameTime ret;

	ret.idle_step = p_idle_step;

	// simple determination of number of physics iteration
	time_accum += ret.idle_step;
	ret.physics_steps = floor(time_accum * p_iterations_per_second);

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

#ifdef DEBUG_ENABLED
	if (max_typical_steps < 0) {
		WARN_PRINT_ONCE("`max_typical_steps` is negative. This could hint at an engine bug or system timer misconfiguration.");
	}
#endif

	// try to keep it consistent with previous iterations
	if (ret.physics_steps < min_typical_steps) {
		const int max_possible_steps = floor((time_accum)*p_iterations_per_second + get_physics_jitter_fix());
		if (max_possible_steps < min_typical_steps) {
			ret.physics_steps = max_possible_steps;
			update_typical = true;
		} else {
			ret.physics_steps = min_typical_steps;
		}
	} else if (ret.physics_steps > max_typical_steps) {
		const int min_possible_steps = floor((time_accum)*p_iterations_per_second - get_physics_jitter_fix());
		if (min_possible_steps > max_typical_steps) {
			ret.physics_steps = min_possible_steps;
			update_typical = true;
		} else {
			ret.physics_steps = max_typical_steps;
		}
	}

	if (ret.physics_steps < 0) {
		ret.physics_steps = 0;
	}

	time_accum -= ret.physics_steps * p_frame_slice;

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
MainFrameTime MainTimerSync::advance_checked(float p_frame_slice, int p_iterations_per_second, float p_idle_step) {
	if (fixed_fps != -1) {
		p_idle_step = 1.0 / fixed_fps;
	}

	float min_output_step = p_idle_step / 8;
	min_output_step = MAX(min_output_step, 1E-6);

	// compensate for last deficit
	p_idle_step += time_deficit;

	MainFrameTime ret = advance_core(p_frame_slice, p_iterations_per_second, p_idle_step);

	// we will do some clamping on ret.idle_step and need to sync those changes to time_accum,
	// that's easiest if we just remember their fixed difference now
	const double idle_minus_accum = ret.idle_step - time_accum;

	// first, least important clamping: keep ret.idle_step consistent with typical_physics_steps.
	// this smoothes out the idle steps and culls small but quick variations.
	{
		float min_average_physics_steps, max_average_physics_steps;
		int consistent_steps = get_average_physics_steps(min_average_physics_steps, max_average_physics_steps);
		if (consistent_steps > 3) {
			ret.clamp_idle(min_average_physics_steps * p_frame_slice, max_average_physics_steps * p_frame_slice);
		}
	}

	// second clamping: keep abs(time_deficit) < jitter_fix * frame_slise
	float max_clock_deviation = get_physics_jitter_fix() * p_frame_slice;
	ret.clamp_idle(p_idle_step - max_clock_deviation, p_idle_step + max_clock_deviation);

	// last clamping: make sure time_accum is between 0 and p_frame_slice for consistency between physics and idle
	ret.clamp_idle(idle_minus_accum, idle_minus_accum + p_frame_slice);

	// all the operations above may have turned ret.idle_step negative or zero, keep a minimal value
	if (ret.idle_step < min_output_step) {
		ret.idle_step = min_output_step;
	}

	// restore time_accum
	time_accum = ret.idle_step - idle_minus_accum;

	// forcing ret.idle_step to be positive may trigger a violation of the
	// promise that time_accum is between 0 and p_frame_slice
#ifdef DEBUG_ENABLED
	if (time_accum < -1E-7) {
		WARN_PRINT_ONCE("Intermediate value of `time_accum` is negative. This could hint at an engine bug or system timer misconfiguration.");
	}
#endif

	if (time_accum > p_frame_slice) {
		const int extra_physics_steps = floor(time_accum * p_iterations_per_second);
		time_accum -= extra_physics_steps * p_frame_slice;
		ret.physics_steps += extra_physics_steps;
	}

#ifdef DEBUG_ENABLED
	if (time_accum < -1E-7) {
		WARN_PRINT_ONCE("Final value of `time_accum` is negative. It should always be between 0 and `p_physics_step`. This hints at an engine bug.");
	}
	if (time_accum > p_frame_slice + 1E-7) {
		WARN_PRINT_ONCE("Final value of `time_accum` is larger than `p_frame_slice`. It should always be between 0 and `p_frame_slice`. This hints at an engine bug.");
	}
#endif

	// track deficit
	time_deficit = p_idle_step - ret.idle_step;

	// p_frame_slice is 1.0 / iterations_per_sec
	// i.e. the time in seconds taken by a physics tick
	ret.interpolation_fraction = time_accum / p_frame_slice;

	return ret;
}

// determine wall clock step since last iteration
float MainTimerSync::get_cpu_idle_step() {
	uint64_t cpu_ticks_elapsed = current_cpu_ticks_usec - last_cpu_ticks_usec;
	last_cpu_ticks_usec = current_cpu_ticks_usec;

	cpu_ticks_elapsed = _delta_smoother.smooth_delta(cpu_ticks_elapsed);

	return cpu_ticks_elapsed / 1000000.0;
}

MainTimerSync::MainTimerSync() :
		last_cpu_ticks_usec(0),
		current_cpu_ticks_usec(0),
		time_accum(0),
		time_deficit(0),
		fixed_fps(0) {
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

// advance one frame, return timesteps to take
MainFrameTime MainTimerSync::advance(float p_frame_slice, int p_iterations_per_second) {
	float cpu_idle_step = get_cpu_idle_step();

	MainFrameTime mft = advance_checked(p_frame_slice, p_iterations_per_second, cpu_idle_step);

	// Now backcalculate the logical timing of the first physics tick.
	// This is used for processing input.
	// It is approximate, but should be fine for input.
	mft.usec_per_tick = 1000000 / p_iterations_per_second;
	uint64_t leftover_usec = mft.interpolation_fraction * mft.usec_per_tick;

	// Note we are using the ACTUAL CPU time for this estimate,
	// NOT the smoothed accumulated time.
	// This is because the input timestamps are measured in realtime,
	// and smoothed time / realtime can get out of sync.
	mft.first_physics_tick_logical_time_usecs = current_cpu_ticks_usec;
	mft.first_physics_tick_logical_time_usecs -= (mft.physics_steps * mft.usec_per_tick) + leftover_usec;

	return mft;
}
