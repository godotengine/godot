/*************************************************************************/
/*  joypad_linux.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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
#ifndef JOYPAD_LINUX_H
#define JOYPAD_LINUX_H

#ifdef JOYDEV_ENABLED
#include "core/os/mutex.h"
#include "core/os/thread.h"
#include "main/input_default.h"

struct input_absinfo;

class JoypadLinux {
public:
	JoypadLinux(InputDefault *in);
	~JoypadLinux();
	void process_joypads();

private:
	enum {
		JOYPADS_MAX = 16,
		MAX_ABS = 63,
		MAX_KEY = 767, // Hack because <linux/input.h> can't be included here
	};

	struct JoypadEvent {
		uint16_t type;
		uint16_t code;
		int32_t value;
	};

	struct Joypad {
		float curr_axis[MAX_ABS];
		int key_map[MAX_KEY];
		int abs_map[MAX_ABS];
		int dpad;
		int fd;

		String devpath;
		input_absinfo *abs_info[MAX_ABS];

		bool force_feedback;
		int ff_effect_id;
		uint64_t ff_effect_timestamp;

		LocalVector<JoypadEvent> events;

		Joypad();
		~Joypad();
		void reset();
	};

#ifdef UDEV_ENABLED
	bool use_udev;
#endif
	InputDefault *input;

	SafeFlag monitor_joypads_exit;
	SafeFlag joypad_events_exit;
	Thread monitor_joypads_thread;
	Thread joypad_events_thread;

	Joypad joypads[JOYPADS_MAX];
	Mutex joypads_mutex[JOYPADS_MAX];

	Vector<String> attached_devices;

	static void monitor_joypads_thread_func(void *p_user);
	void monitor_joypads_thread_run();

	void open_joypad(const char *p_path);
	void setup_joypad_properties(Joypad &p_joypad);

	void close_joypads();
	void close_joypad(const char *p_devpath);
	void close_joypad(Joypad &p_joypad, int p_id);

#ifdef UDEV_ENABLED
	void enumerate_joypads(struct udev *p_udev);
	void monitor_joypads(struct udev *p_udev);
#endif
	void monitor_joypads();

	void joypad_vibration_start(Joypad &p_joypad, float p_weak_magnitude, float p_strong_magnitude, float p_duration, uint64_t p_timestamp);
	void joypad_vibration_stop(Joypad &p_joypad, uint64_t p_timestamp);

	static void joypad_events_thread_func(void *p_user);
	void joypad_events_thread_run();

	float axis_correct(const input_absinfo *p_abs, int p_value) const;
};

#endif
#endif // JOYPAD_LINUX_H
