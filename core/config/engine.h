/*************************************************************************/
/*  engine.h                                                             */
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

#ifndef ENGINE_H
#define ENGINE_H

#include "core/os/main_loop.h"
#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "core/templates/vector.h"

class Engine {
public:
	struct Singleton {
		StringName name;
		Object *ptr;
		StringName class_name; //used for binding generation hinting
		bool user_created = false;
		Singleton(const StringName &p_name = StringName(), Object *p_ptr = nullptr, const StringName &p_class_name = StringName());
	};

private:
	friend class Main;

	uint64_t frames_drawn = 0;
	uint32_t _frame_delay = 0;
	uint64_t _frame_ticks = 0;
	double _process_step = 0;

	int ips = 60;
	double physics_jitter_fix = 0.5;
	double _fps = 1;
	int _target_fps = 0;
	double _time_scale = 1.0;
	uint64_t _physics_frames = 0;
	double _physics_interpolation_fraction = 0.0f;
	bool abort_on_gpu_errors = false;
	bool use_validation_layers = false;
	int32_t gpu_idx = -1;

	uint64_t _process_frames = 0;
	bool _in_physics = false;

	List<Singleton> singletons;
	Map<StringName, Object *> singleton_ptrs;

	bool editor_hint = false;

	static Engine *singleton;

	String shader_cache_path;

public:
	static Engine *get_singleton();

	virtual void set_physics_ticks_per_second(int p_ips);
	virtual int get_physics_ticks_per_second() const;

	void set_physics_jitter_fix(double p_threshold);
	double get_physics_jitter_fix() const;

	virtual void set_target_fps(int p_fps);
	virtual int get_target_fps() const;

	virtual double get_frames_per_second() const { return _fps; }

	uint64_t get_frames_drawn();

	uint64_t get_physics_frames() const { return _physics_frames; }
	uint64_t get_process_frames() const { return _process_frames; }
	bool is_in_physics_frame() const { return _in_physics; }
	uint64_t get_frame_ticks() const { return _frame_ticks; }
	double get_process_step() const { return _process_step; }
	double get_physics_interpolation_fraction() const { return _physics_interpolation_fraction; }

	void set_time_scale(double p_scale);
	double get_time_scale() const;

	void set_print_error_messages(bool p_enabled);
	bool is_printing_error_messages() const;

	void set_frame_delay(uint32_t p_msec);
	uint32_t get_frame_delay() const;

	void add_singleton(const Singleton &p_singleton);
	void get_singletons(List<Singleton> *p_singletons);
	bool has_singleton(const StringName &p_name) const;
	Object *get_singleton_object(const StringName &p_name) const;
	void remove_singleton(const StringName &p_name);
	bool is_singleton_user_created(const StringName &p_name) const;

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

	void set_shader_cache_path(const String &p_path);
	String get_shader_cache_path() const;

	bool is_abort_on_gpu_errors_enabled() const;
	bool is_validation_layers_enabled() const;
	int32_t get_gpu_index() const;

	Engine();
	virtual ~Engine() {}
};

#endif // ENGINE_H
