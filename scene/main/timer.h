/*************************************************************************/
/*  timer.h                                                              */
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

#ifndef TIMER_H
#define TIMER_H

#include "scene/main/node.h"

class Timer : public Node {
	GDCLASS(Timer, Node);

	float wait_time;
	bool one_shot;
	bool autostart;
	bool processing;
	bool paused;

	double time_left;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	enum TimerProcessMode {
		TIMER_PROCESS_PHYSICS,
		TIMER_PROCESS_IDLE,
	};

	void set_wait_time(float p_time);
	float get_wait_time() const;

	void set_one_shot(bool p_one_shot);
	bool is_one_shot() const;

	void set_autostart(bool p_start);
	bool has_autostart() const;

	void start(float p_time = -1);
	void stop();

	void set_paused(bool p_paused);
	bool is_paused() const;

	bool is_stopped() const;

	float get_time_left() const;

	String get_configuration_warning() const;

	void set_timer_process_mode(TimerProcessMode p_mode);
	TimerProcessMode get_timer_process_mode() const;
	Timer();

private:
	TimerProcessMode timer_process_mode;
	void _set_process(bool p_process, bool p_force = false);
};

VARIANT_ENUM_CAST(Timer::TimerProcessMode);

#endif // TIMER_H
