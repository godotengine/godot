/*************************************************************************/
/*  engine.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#ifndef ENGINE_H
#define ENGINE_H

#include "list.h"
#include "os/main_loop.h"
#include "ustring.h"
#include "vector.h"

class Engine {

	friend class Main;

	String _custom_level;
	uint64_t frames_drawn;
	uint32_t _frame_delay;

	int ips;
	float _fps;
	int _target_fps;
	float _time_scale;
	bool _pixel_snap;
	uint64_t _fixed_frames;
	uint64_t _idle_frames;
	bool _in_fixed;

	static Engine *singleton;

public:
	static Engine *get_singleton();

	virtual void set_iterations_per_second(int p_ips);
	virtual int get_iterations_per_second() const;

	virtual void set_target_fps(int p_fps);
	virtual float get_target_fps() const;

	virtual float get_frames_per_second() const { return _fps; }

	String get_custom_level() const { return _custom_level; }

	uint64_t get_frames_drawn();

	uint64_t get_fixed_frames() const { return _fixed_frames; }
	uint64_t get_idle_frames() const { return _idle_frames; }
	bool is_in_fixed_frame() const { return _in_fixed; }

	void set_time_scale(float p_scale);
	float get_time_scale() const;

	void set_frame_delay(uint32_t p_msec);
	uint32_t get_frame_delay() const;

	_FORCE_INLINE_ bool get_use_pixel_snap() const { return _pixel_snap; }

	Dictionary get_version_info() const;

	Engine();
};

#endif // ENGINE_H
