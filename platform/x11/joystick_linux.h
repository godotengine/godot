/*************************************************************************/
/*  joystick_linux.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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

//author: Andreas Haas <hondres,  liugam3@gmail.com>
#ifndef JOYSTICK_LINUX_H
#define JOYSTICK_LINUX_H
#ifdef JOYDEV_ENABLED
#include "main/input_default.h"
#include "os/thread.h"
#include "os/mutex.h"

struct input_absinfo;

class joystick_linux
{
public:
	joystick_linux(InputDefault *in);
	~joystick_linux();
	uint32_t process_joysticks(uint32_t p_event_id);
private:

	enum {
		JOYSTICKS_MAX = 16,
		MAX_ABS = 63,
		MAX_KEY = 767,   // Hack because <linux/input.h> can't be included here
	};

	struct Joystick {
		InputDefault::JoyAxis curr_axis[MAX_ABS];
		int key_map[MAX_KEY];
		int abs_map[MAX_ABS];
		int dpad;
		int fd;

		String devpath;
		input_absinfo *abs_info[MAX_ABS];

		bool force_feedback;
		int ff_effect_id;
		uint64_t ff_effect_timestamp;

		Joystick();
		~Joystick();
		void reset();
	};

	bool exit_udev;
	Mutex *joy_mutex;
	Thread *joy_thread;
	InputDefault *input;
	Joystick joysticks[JOYSTICKS_MAX];
	Vector<String> attached_devices;

	static void joy_thread_func(void *p_user);

	int get_joy_from_path(String path) const;
	int get_free_joy_slot() const;

	void setup_joystick_properties(int p_id);
	void close_joystick(int p_id = -1);
#ifdef UDEV_ENABLED
	void enumerate_joysticks(struct udev *_udev);
	void monitor_joysticks(struct udev *_udev);
#endif
	void monitor_joysticks();
	void run_joystick_thread();
	void open_joystick(const char* path);

	void joystick_vibration_start(int p_id, float p_weak_magnitude, float p_strong_magnitude, float p_duration, uint64_t p_timestamp);
	void joystick_vibration_stop(int p_id, uint64_t p_timestamp);

	InputDefault::JoyAxis axis_correct(const input_absinfo *abs, int value) const;
};

#endif
#endif // JOYSTICK_LINUX_H
