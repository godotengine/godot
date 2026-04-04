/**************************************************************************/
/*  editor_undo_redo_manager.hpp                                          */
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

#include <godot_cpp/classes/undo_redo.hpp>
#include <godot_cpp/core/object.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;
class StringName;
class Variant;

class EditorUndoRedoManager : public Object {
	GDEXTENSION_CLASS(EditorUndoRedoManager, Object)

public:
	enum SpecialHistory {
		GLOBAL_HISTORY = 0,
		REMOTE_HISTORY = -9,
		INVALID_HISTORY = -99,
	};

	void create_action(const String &p_name, UndoRedo::MergeMode p_merge_mode = (UndoRedo::MergeMode)0, Object *p_custom_context = nullptr, bool p_backward_undo_ops = false, bool p_mark_unsaved = true);
	void commit_action(bool p_execute = true);
	bool is_committing_action() const;
	void force_fixed_history();

private:
	void add_do_method_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	void add_do_method(Object *p_object, const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 2 + sizeof...(Args)> variant_args{{ Variant(p_object), Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 2 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		add_do_method_internal(call_args.data(), variant_args.size());
	}

private:
	void add_undo_method_internal(const Variant **p_args, GDExtensionInt p_arg_count);

public:
	template <typename... Args>
	void add_undo_method(Object *p_object, const StringName &p_method, const Args &...p_args) {
		std::array<Variant, 2 + sizeof...(Args)> variant_args{{ Variant(p_object), Variant(p_method), Variant(p_args)... }};
		std::array<const Variant *, 2 + sizeof...(Args)> call_args;
		for (size_t i = 0; i < variant_args.size(); i++) {
			call_args[i] = &variant_args[i];
		}
		add_undo_method_internal(call_args.data(), variant_args.size());
	}
	void add_do_property(Object *p_object, const StringName &p_property, const Variant &p_value);
	void add_undo_property(Object *p_object, const StringName &p_property, const Variant &p_value);
	void add_do_reference(Object *p_object);
	void add_undo_reference(Object *p_object);
	int32_t get_object_history_id(Object *p_object) const;
	UndoRedo *get_history_undo_redo(int32_t p_id) const;
	void clear_history(int32_t p_id = -99, bool p_increase_version = true);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(EditorUndoRedoManager::SpecialHistory);

