/*************************************************************************/
/*  undo_redo.h                                                          */
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

#ifndef UNDO_REDO_H
#define UNDO_REDO_H

#include "core/object/class_db.h"
#include "core/object/ref_counted.h"

class UndoRedo : public Object {
	GDCLASS(UndoRedo, Object);
	OBJ_SAVE_TYPE(UndoRedo);

public:
	enum MergeMode {
		MERGE_DISABLE,
		MERGE_ENDS,
		MERGE_ALL
	};

	typedef void (*CommitNotifyCallback)(void *p_ud, const String &p_name);
	Variant _add_do_method(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Variant _add_undo_method(const Variant **p_args, int p_argcount, Callable::CallError &r_error);

	typedef void (*MethodNotifyCallback)(void *p_ud, Object *p_base, const StringName &p_name, VARIANT_ARG_DECLARE);
	typedef void (*PropertyNotifyCallback)(void *p_ud, Object *p_base, const StringName &p_property, const Variant &p_value);

private:
	struct Operation {
		enum Type {
			TYPE_METHOD,
			TYPE_PROPERTY,
			TYPE_REFERENCE
		};

		Type type;
		bool force_keep_in_merge_ends;
		Ref<RefCounted> ref;
		ObjectID object;
		StringName name;
		Variant args[VARIANT_ARG_MAX];
	};

	struct Action {
		String name;
		List<Operation> do_ops;
		List<Operation> undo_ops;
		uint64_t last_tick;
	};

	Vector<Action> actions;
	int current_action = -1;
	bool force_keep_in_merge_ends = false;
	int action_level = 0;
	MergeMode merge_mode = MERGE_DISABLE;
	bool merging = false;
	uint64_t version = 1;

	void _pop_history_tail();
	void _process_operation_list(List<Operation>::Element *E);
	void _discard_redo();
	bool _redo(bool p_execute);

	CommitNotifyCallback callback = nullptr;
	void *callback_ud = nullptr;
	void *method_callbck_ud = nullptr;
	void *prop_callback_ud = nullptr;

	MethodNotifyCallback method_callback = nullptr;
	PropertyNotifyCallback property_callback = nullptr;

	int committing = 0;

protected:
	static void _bind_methods();

public:
	void create_action(const String &p_name = "", MergeMode p_mode = MERGE_DISABLE);

	void add_do_method(Object *p_object, const StringName &p_method, VARIANT_ARG_LIST);
	void add_undo_method(Object *p_object, const StringName &p_method, VARIANT_ARG_LIST);
	void add_do_property(Object *p_object, const StringName &p_property, const Variant &p_value);
	void add_undo_property(Object *p_object, const StringName &p_property, const Variant &p_value);
	void add_do_reference(Object *p_object);
	void add_undo_reference(Object *p_object);

	void start_force_keep_in_merge_ends();
	void end_force_keep_in_merge_ends();

	bool is_committing_action() const;
	void commit_action(bool p_execute = true);

	bool redo();
	bool undo();
	String get_current_action_name() const;

	int get_history_count();
	int get_current_action();
	String get_action_name(int p_id);
	void clear_history(bool p_increase_version = true);

	bool has_undo() const;
	bool has_redo() const;

	uint64_t get_version() const;

	void set_commit_notify_callback(CommitNotifyCallback p_callback, void *p_ud);

	void set_method_notify_callback(MethodNotifyCallback p_method_callback, void *p_ud);
	void set_property_notify_callback(PropertyNotifyCallback p_property_callback, void *p_ud);

	UndoRedo() {}
	~UndoRedo();
};

VARIANT_ENUM_CAST(UndoRedo::MergeMode);

#endif // UNDO_REDO_H
