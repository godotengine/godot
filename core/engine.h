/*************************************************************************/
/*  engine.h                                                             */
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

#ifndef ENGINE_H
#define ENGINE_H

#include "core/list.h"
#include "core/os/main_loop.h"
#include "core/ustring.h"
#include "core/vector.h"

class Engine {
public:
	struct Singleton {
		StringName name;
		Object *ptr;
		Singleton(const StringName &p_name = StringName(), Object *p_ptr = nullptr);
	};

private:
	friend class Main;

	uint64_t frames_drawn;
	uint32_t _frame_delay;
	uint64_t _frame_ticks;
	float _frame_step;

	int ips;
	float physics_jitter_fix;
	float _fps;
	int _target_fps;
	float _time_scale;
	bool _gpu_pixel_snap;
	uint64_t _physics_frames;
	float _physics_interpolation_fraction;
	bool _portals_active;
	bool _occlusion_culling_active;

	uint64_t _idle_frames;
	bool _in_physics;

	List<Singleton> singletons;
	Map<StringName, Object *> singleton_ptrs;

	bool editor_hint;

	static Engine *singleton;

public:
	static Engine *get_singleton();

	virtual void set_iterations_per_second(int p_ips);
	virtual int get_iterations_per_second() const;

	void set_physics_jitter_fix(float p_threshold);
	float get_physics_jitter_fix() const;

	virtual void set_target_fps(int p_fps);
	virtual int get_target_fps() const;

	virtual float get_frames_per_second() const { return _fps; }

	uint64_t get_frames_drawn();

	uint64_t get_physics_frames() const { return _physics_frames; }
	uint64_t get_idle_frames() const { return _idle_frames; }
	bool is_in_physics_frame() const { return _in_physics; }
	uint64_t get_idle_frame_ticks() const { return _frame_ticks; }
	float get_idle_frame_step() const { return _frame_step; }
	float get_physics_interpolation_fraction() const { return _physics_interpolation_fraction; }

	void set_time_scale(float p_scale);
	float get_time_scale() const;

	void set_frame_delay(uint32_t p_msec);
	uint32_t get_frame_delay() const;

	void set_print_error_messages(bool p_enabled);
	bool is_printing_error_messages() const;

	void add_singleton(const Singleton &p_singleton);
	void get_singletons(List<Singleton> *p_singletons);
	bool has_singleton(const String &p_name) const;
	Object *get_singleton_object(const String &p_name) const;

	_FORCE_INLINE_ bool get_use_gpu_pixel_snap() const { return _gpu_pixel_snap; }
	bool are_portals_active() const { return _portals_active; }
	void set_portals_active(bool p_active);

#ifdef TOOLS_ENABLED
	_FORCE_INLINE_ void set_editor_hint(bool p_enabled) { editor_hint = p_enabled; }
	_FORCE_INLINE_ bool is_editor_hint() const { return editor_hint; }
#else
	_FORCE_INLINE_ void set_editor_hint(bool p_enabled) {}
	_FORCE_INLINE_ bool is_editor_hint() const { return false; }
#endif

	Dictionary get_version_info() const;
	Dictionary get_author_info() const;
	Array get_copyright_info() const;
	Dictionary get_donor_info() const;
	Dictionary get_license_info() const;
	String get_license_text() const;

	Engine();
	virtual ~Engine() {}
};

#endif // ENGINE_H
