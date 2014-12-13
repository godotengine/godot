/*************************************************************************/
/*  timer.cpp                                                            */
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
#include "timer.h"


void Timer::_notification(int p_what) {

	switch(p_what) {


		case NOTIFICATION_READY: {

			if (autostart) {
#ifdef TOOLS_ENABLED
				if (get_tree()->is_editor_hint() && get_tree()->get_edited_scene_root() && (get_tree()->get_edited_scene_root()==this || get_tree()->get_edited_scene_root()->is_a_parent_of(this)))
					break;
#endif
				start();
			}
		} break;
		case NOTIFICATION_PROCESS: {

			time_left -= get_process_delta_time();

			if (time_left<0) {
				if (!one_shot)
					time_left=wait_time+time_left;
				else
					stop();

				emit_signal("timeout");
			}

		} break;
	}
}



void Timer::set_wait_time(float p_time) {

	ERR_EXPLAIN("time should be greater than zero.");
	ERR_FAIL_COND(p_time<=0);
	wait_time=p_time;

}
float Timer::get_wait_time() const {

	return wait_time;
}

void Timer::set_one_shot(bool p_one_shot) {

	one_shot=p_one_shot;
}
bool Timer::is_one_shot() const {

	return one_shot;
}

void Timer::set_autostart(bool p_start) {

	autostart=p_start;
}
bool Timer::has_autostart() const {

	return autostart;
}

void Timer::start() {

	time_left=wait_time;	
	set_process(true);	
}

void Timer::stop() {
	time_left=-1;
	set_process(false);
	autostart=false;
}

float Timer::get_time_left() const {

	return time_left >0 ? time_left : 0;
}


void Timer::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_wait_time","time_sec"),&Timer::set_wait_time);
	ObjectTypeDB::bind_method(_MD("get_wait_time"),&Timer::get_wait_time);

	ObjectTypeDB::bind_method(_MD("set_one_shot","enable"),&Timer::set_one_shot);
	ObjectTypeDB::bind_method(_MD("is_one_shot"),&Timer::is_one_shot);

	ObjectTypeDB::bind_method(_MD("set_autostart","enable"),&Timer::set_autostart);
	ObjectTypeDB::bind_method(_MD("has_autostart"),&Timer::has_autostart);

	ObjectTypeDB::bind_method(_MD("start"),&Timer::start);
	ObjectTypeDB::bind_method(_MD("stop"),&Timer::stop);

	ObjectTypeDB::bind_method(_MD("get_time_left"),&Timer::get_time_left);

	ADD_SIGNAL( MethodInfo("timeout") );

	ADD_PROPERTY( PropertyInfo(Variant::REAL, "wait_time", PROPERTY_HINT_EXP_RANGE, "0.01,4096,0.01" ), _SCS("set_wait_time"), _SCS("get_wait_time") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "one_shot" ), _SCS("set_one_shot"), _SCS("is_one_shot") );
	ADD_PROPERTY( PropertyInfo(Variant::BOOL, "autostart" ), _SCS("set_autostart"), _SCS("has_autostart") );

}

Timer::Timer() {


	autostart=false;
	wait_time=1;
	one_shot=false;
	time_left=-1;
}
