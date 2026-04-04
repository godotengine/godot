/**************************************************************************/
/*  undo_redo.hpp                                                         */
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

#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;
class StringName;
class Variant;

class UndoRedo : public Object {
	GDEXTENSION_CLASS(UndoRedo, Object)

public:
	enum MergeMode {
		MERGE_DISABLE = 0,
		MERGE_ENDS = 1,
		MERGE_ALL = 2,
	};

	void create_action(const String &p_name, UndoRedo::MergeMode p_merge_mode = (UndoRedo::MergeMode)0, bool p_backward_undo_ops = false);
	void commit_action(bool p_execute = true);
	bool is_committing_action() const;
	void add_do_method(const Callable &p_callable);
	void add_undo_method(const Callable &p_callable);
	void add_do_property(Object *p_object, const StringName &p_property, const Variant &p_value);
	void add_undo_property(Object *p_object, const StringName &p_property, const Variant &p_value);
	void add_do_reference(Object *p_object);
	void add_undo_reference(Object *p_object);
	void start_force_keep_in_merge_ends();
	void end_force_keep_in_merge_ends();
	int32_t get_history_count();
	int32_t get_current_action();
	String get_action_name(int32_t p_id);
	void clear_history(bool p_increase_version = true);
	String get_current_action_name() const;
	bool has_undo() const;
	bool has_redo() const;
	uint64_t get_version() const;
	void set_max_steps(int32_t p_max_steps);
	int32_t get_max_steps() const;
	bool redo();
	bool undo();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(UndoRedo::MergeMode);

