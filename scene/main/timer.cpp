/*************************************************************************/
/*  timer.cpp                                                            */
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
#include "timer.h"

#include "engine.h"

void Timer::_notification(int p_what) {

	switch (p_what) {

		case NOTIFICATION_READY: {

			if (autostart) {
#ifdef TOOLS_ENABLED
				if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root() && (get_tree()->get_edited_scene_root() == this || get_tree()->get_edited_scene_root()->is_a_parent_of(this)))
					break;
#endif
				start();
				autostart = false;
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (timer_process_mode == TIMER_PROCESS_PHYSICS || !is_processing_internal())
				return;
			time_left -= get_process_delta_time();

			if (time_left < 0) {
				if (repeat)
					time_left += time_interval;
				else
					stop();

				emit_signal("timeout");
			}

		} break;
		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (timer_process_mode == TIMER_PROCESS_IDLE || !is_physics_processing_internal())
				return;
			time_left -= get_physics_process_delta_time();

			if (time_left < 0) {
				if (repeat)
					time_left += time_interval;
				else
					stop();
				emit_signal("timeout");
			}

		} break;
	}
}

void Timer::set_time_interval(float p_time) {
	ERR_EXPLAIN("time should be greater than zero.");
	ERR_FAIL_COND(p_time <= 0);
	time_interval = p_time;
}
float Timer::get_time_interval() const {

	return time_interval;
}

void Timer::set_repeat(bool p_repeat) {

	repeat = p_repeat;
}
bool Timer::is_repeat() const {

	return repeat;
}

void Timer::set_autostart(bool p_start) {

	autostart = p_start;
}
bool Timer::has_autostart() const {

	return autostart;
}

void Timer::start() {
	time_left = time_interval;
	_set_process(true);
}

void Timer::stop() {
	time_left = -1;
	_set_process(false);
	autostart = false;
}

void Timer::set_paused(bool p_paused) {
	if (paused == p_paused)
		return;

	paused = p_paused;
	_set_process(processing);
}

bool Timer::is_paused() const {
	return paused;
}

bool Timer::is_stopped() const {
	return get_time_left() <= 0;
}

float Timer::get_time_left() const {

	return time_left > 0 ? time_left : 0;
}

void Timer::set_timer_process_mode(TimerProcessMode p_mode) {

	if (timer_process_mode == p_mode)
		return;

	switch (timer_process_mode) {
		case TIMER_PROCESS_PHYSICS:
			if (is_physics_processing_internal()) {
				set_physics_process_internal(false);
				set_process_internal(true);
			}
			break;
		case TIMER_PROCESS_IDLE:
			if (is_processing_internal()) {
				set_process_internal(false);
				set_physics_process_internal(true);
			}
			break;
	}
	timer_process_mode = p_mode;
}

Timer::TimerProcessMode Timer::get_timer_process_mode() const {

	return timer_process_mode;
}

void Timer::_set_process(bool p_process, bool p_force) {
	switch (timer_process_mode) {
		case TIMER_PROCESS_PHYSICS: set_physics_process_internal(p_process && !paused); break;
		case TIMER_PROCESS_IDLE: set_process_internal(p_process && !paused); break;
	}
	processing = p_process;
}

void Timer::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_time_interval", "time_sec"), &Timer::set_time_interval);
	ClassDB::bind_method(D_METHOD("get_time_interval"), &Timer::get_time_interval);

	ClassDB::bind_method(D_METHOD("set_repeat", "enable"), &Timer::set_repeat);
	ClassDB::bind_method(D_METHOD("is_repeat"), &Timer::is_repeat);

	ClassDB::bind_method(D_METHOD("set_autostart", "enable"), &Timer::set_autostart);
	ClassDB::bind_method(D_METHOD("has_autostart"), &Timer::has_autostart);

	ClassDB::bind_method(D_METHOD("start"), &Timer::start);
	ClassDB::bind_method(D_METHOD("stop"), &Timer::stop);

	ClassDB::bind_method(D_METHOD("set_paused", "paused"), &Timer::set_paused);
	ClassDB::bind_method(D_METHOD("is_paused"), &Timer::is_paused);

	ClassDB::bind_method(D_METHOD("is_stopped"), &Timer::is_stopped);

	ClassDB::bind_method(D_METHOD("get_time_left"), &Timer::get_time_left);

	ClassDB::bind_method(D_METHOD("set_timer_process_mode", "mode"), &Timer::set_timer_process_mode);
	ClassDB::bind_method(D_METHOD("get_timer_process_mode"), &Timer::get_timer_process_mode);

	ADD_SIGNAL(MethodInfo("timeout"));

	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_timer_process_mode", "get_timer_process_mode");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "time_interval", PROPERTY_HINT_EXP_RANGE, "0.01,4096,0.01"), "set_time_interval", "get_time_interval");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "repeat"), "set_repeat", "is_repeat");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "autostart"), "set_autostart", "has_autostart");

	BIND_ENUM_CONSTANT(TIMER_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(TIMER_PROCESS_IDLE);
}

Timer::Timer() {
	timer_process_mode = TIMER_PROCESS_IDLE;
	autostart = false;
	time_interval = 1;
	repeat = true;
	time_left = -1;
	processing = false;
	paused = false;
}
