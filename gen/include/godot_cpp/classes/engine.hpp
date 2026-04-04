/**************************************************************************/
/*  engine.hpp                                                            */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class MainLoop;
class ScriptBacktrace;
class ScriptLanguage;
class StringName;

class Engine : public Object {
	GDEXTENSION_CLASS(Engine, Object)

	static Engine *singleton;

public:
	static Engine *get_singleton();

	void set_physics_ticks_per_second(int32_t p_physics_ticks_per_second);
	int32_t get_physics_ticks_per_second() const;
	void set_max_physics_steps_per_frame(int32_t p_max_physics_steps);
	int32_t get_max_physics_steps_per_frame() const;
	void set_physics_jitter_fix(double p_physics_jitter_fix);
	double get_physics_jitter_fix() const;
	double get_physics_interpolation_fraction() const;
	void set_max_fps(int32_t p_max_fps);
	int32_t get_max_fps() const;
	void set_time_scale(double p_time_scale);
	double get_time_scale();
	int32_t get_frames_drawn();
	double get_frames_per_second() const;
	uint64_t get_physics_frames() const;
	uint64_t get_process_frames() const;
	MainLoop *get_main_loop() const;
	Dictionary get_version_info() const;
	Dictionary get_author_info() const;
	TypedArray<Dictionary> get_copyright_info() const;
	Dictionary get_donor_info() const;
	Dictionary get_license_info() const;
	String get_license_text() const;
	String get_architecture_name() const;
	bool is_in_physics_frame() const;
	bool has_singleton(const StringName &p_name) const;
	Object *get_singleton(const StringName &p_name) const;
	void register_singleton(const StringName &p_name, Object *p_instance);
	void unregister_singleton(const StringName &p_name);
	PackedStringArray get_singleton_list() const;
	Error register_script_language(ScriptLanguage *p_language);
	Error unregister_script_language(ScriptLanguage *p_language);
	int32_t get_script_language_count();
	ScriptLanguage *get_script_language(int32_t p_index) const;
	TypedArray<Ref<ScriptBacktrace>> capture_script_backtraces(bool p_include_variables = false) const;
	bool is_editor_hint() const;
	bool is_embedded_in_editor() const;
	String get_write_movie_path() const;
	void set_print_to_stdout(bool p_enabled);
	bool is_printing_to_stdout() const;
	void set_print_error_messages(bool p_enabled);
	bool is_printing_error_messages() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~Engine();

public:
};

} // namespace godot

