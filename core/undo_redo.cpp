/*************************************************************************/
/*  undo_redo.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "undo_redo.h"

#include "core/os/os.h"

void UndoRedo::_discard_redo() {
	if (current_action == actions.size() - 1) {
		return;
	}

	for (int i = current_action + 1; i < actions.size(); i++) {
		for (List<Operation>::Element *E = actions.write[i].do_ops.front(); E; E = E->next()) {
			if (E->get().type == Operation::TYPE_REFERENCE) {
				Object *obj = ObjectDB::get_instance(E->get().object);
				if (obj) {
					memdelete(obj);
				}
			}
		}
		//ERASE do data
	}

	actions.resize(current_action + 1);
}

void UndoRedo::create_action(const String &p_name, MergeMode p_mode) {
	uint32_t ticks = OS::get_singleton()->get_ticks_msec();

	if (action_level == 0) {
		_discard_redo();

		// Check if the merge operation is valid
		if (p_mode != MERGE_DISABLE && actions.size() && actions[actions.size() - 1].name == p_name && actions[actions.size() - 1].last_tick + 800 > ticks) {
			current_action = actions.size() - 2;

			if (p_mode == MERGE_ENDS) {
				// Clear all do ops from last action, and delete all object references
				List<Operation>::Element *E = actions.write[current_action + 1].do_ops.front();

				while (E) {
					if (E->get().type == Operation::TYPE_REFERENCE) {
						Object *obj = ObjectDB::get_instance(E->get().object);

						if (obj) {
							memdelete(obj);
						}
					}

					E = E->next();
					actions.write[current_action + 1].do_ops.pop_front();
				}
			}

			actions.write[actions.size() - 1].last_tick = ticks;

			merge_mode = p_mode;
			merging = true;
		} else {
			Action new_action;
			new_action.name = p_name;
			new_action.last_tick = ticks;
			actions.push_back(new_action);

			merge_mode = MERGE_DISABLE;
		}
	}

	action_level++;
}

void UndoRedo::add_do_method(Object *p_object, const StringName &p_method, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());
	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object)) {
		do_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));
	}

	do_op.type = Operation::TYPE_METHOD;
	do_op.name = p_method;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		do_op.args[i] = *argptr[i];
	}
	actions.write[current_action + 1].do_ops.push_back(do_op);
}

void UndoRedo::add_undo_method(Object *p_object, const StringName &p_method, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());

	// No undo if the merge mode is MERGE_ENDS
	if (merge_mode == MERGE_ENDS) {
		return;
	}

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object)) {
		undo_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));
	}

	undo_op.type = Operation::TYPE_METHOD;
	undo_op.name = p_method;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		undo_op.args[i] = *argptr[i];
	}
	actions.write[current_action + 1].undo_ops.push_back(undo_op);
}

void UndoRedo::add_do_property(Object *p_object, const StringName &p_property, const Variant &p_value) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());
	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object)) {
		do_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));
	}

	do_op.type = Operation::TYPE_PROPERTY;
	do_op.name = p_property;
	do_op.args[0] = p_value;
	actions.write[current_action + 1].do_ops.push_back(do_op);
}

void UndoRedo::add_undo_property(Object *p_object, const StringName &p_property, const Variant &p_value) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());

	// No undo if the merge mode is MERGE_ENDS
	if (merge_mode == MERGE_ENDS) {
		return;
	}

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object)) {
		undo_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));
	}

	undo_op.type = Operation::TYPE_PROPERTY;
	undo_op.name = p_property;
	undo_op.args[0] = p_value;
	actions.write[current_action + 1].undo_ops.push_back(undo_op);
}

void UndoRedo::add_do_reference(Object *p_object) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());
	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object)) {
		do_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));
	}

	do_op.type = Operation::TYPE_REFERENCE;
	actions.write[current_action + 1].do_ops.push_back(do_op);
}

void UndoRedo::add_undo_reference(Object *p_object) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());

	// No undo if the merge mode is MERGE_ENDS
	if (merge_mode == MERGE_ENDS) {
		return;
	}

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object)) {
		undo_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));
	}

	undo_op.type = Operation::TYPE_REFERENCE;
	actions.write[current_action + 1].undo_ops.push_back(undo_op);
}

void UndoRedo::_pop_history_tail() {
	_discard_redo();

	if (!actions.size()) {
		return;
	}

	for (List<Operation>::Element *E = actions.write[0].undo_ops.front(); E; E = E->next()) {
		if (E->get().type == Operation::TYPE_REFERENCE) {
			Object *obj = ObjectDB::get_instance(E->get().object);
			if (obj) {
				memdelete(obj);
			}
		}
	}

	actions.remove(0);
	if (current_action >= 0) {
		current_action--;
	}
}

bool UndoRedo::is_committing_action() const {
	return committing > 0;
}

void UndoRedo::commit_action() {
	ERR_FAIL_COND(action_level <= 0);
	action_level--;
	if (action_level > 0) {
		return; //still nested
	}

	if (merging) {
		version--;
		merging = false;
	}

	committing++;
	redo(); // perform action
	committing--;
	if (callback && actions.size() > 0) {
		callback(callback_ud, actions[actions.size() - 1].name);
	}
}

void UndoRedo::_process_operation_list(List<Operation>::Element *E) {
	for (; E; E = E->next()) {
		Operation &op = E->get();

		Object *obj = ObjectDB::get_instance(op.object);
		if (!obj) { //may have been deleted and this is fine
			continue;
		}

		switch (op.type) {
			case Operation::TYPE_METHOD: {
				Vector<const Variant *> argptrs;
				argptrs.resize(VARIANT_ARG_MAX);
				int argc = 0;

				for (int i = 0; i < VARIANT_ARG_MAX; i++) {
					if (op.args[i].get_type() == Variant::NIL) {
						break;
					}
					argptrs.write[i] = &op.args[i];
					argc++;
				}
				argptrs.resize(argc);

				Callable::CallError ce;
				obj->call(op.name, (const Variant **)argptrs.ptr(), argc, ce);
				if (ce.error != Callable::CallError::CALL_OK) {
					ERR_PRINT("Error calling method from signal '" + String(op.name) + "': " + Variant::get_call_error_text(obj, op.name, (const Variant **)argptrs.ptr(), argc, ce));
				}
#ifdef TOOLS_ENABLED
				Resource *res = Object::cast_to<Resource>(obj);
				if (res) {
					res->set_edited(true);
				}

#endif

				if (method_callback) {
					method_callback(method_callbck_ud, obj, op.name, VARIANT_ARGS_FROM_ARRAY(op.args));
				}
			} break;
			case Operation::TYPE_PROPERTY: {
				obj->set(op.name, op.args[0]);
#ifdef TOOLS_ENABLED
				Resource *res = Object::cast_to<Resource>(obj);
				if (res) {
					res->set_edited(true);
				}
#endif
				if (property_callback) {
					property_callback(prop_callback_ud, obj, op.name, op.args[0]);
				}
			} break;
			case Operation::TYPE_REFERENCE: {
				//do nothing
			} break;
		}
	}
}

bool UndoRedo::redo() {
	ERR_FAIL_COND_V(action_level > 0, false);

	if ((current_action + 1) >= actions.size()) {
		return false; //nothing to redo
	}

	current_action++;

	_process_operation_list(actions.write[current_action].do_ops.front());
	version++;
	emit_signal("version_changed");

	return true;
}

bool UndoRedo::undo() {
	ERR_FAIL_COND_V(action_level > 0, false);
	if (current_action < 0) {
		return false; //nothing to redo
	}
	_process_operation_list(actions.write[current_action].undo_ops.front());
	current_action--;
	version--;
	emit_signal("version_changed");

	return true;
}

void UndoRedo::clear_history(bool p_increase_version) {
	ERR_FAIL_COND(action_level > 0);
	_discard_redo();

	while (actions.size()) {
		_pop_history_tail();
	}

	if (p_increase_version) {
		version++;
		emit_signal("version_changed");
	}
}

String UndoRedo::get_current_action_name() const {
	ERR_FAIL_COND_V(action_level > 0, "");
	if (current_action < 0) {
		return "";
	}
	return actions[current_action].name;
}

bool UndoRedo::has_undo() {
	return current_action >= 0;
}

bool UndoRedo::has_redo() {
	return (current_action + 1) < actions.size();
}

uint64_t UndoRedo::get_version() const {
	return version;
}

void UndoRedo::set_commit_notify_callback(CommitNotifyCallback p_callback, void *p_ud) {
	callback = p_callback;
	callback_ud = p_ud;
}

void UndoRedo::set_method_notify_callback(MethodNotifyCallback p_method_callback, void *p_ud) {
	method_callback = p_method_callback;
	method_callbck_ud = p_ud;
}

void UndoRedo::set_property_notify_callback(PropertyNotifyCallback p_property_callback, void *p_ud) {
	property_callback = p_property_callback;
	prop_callback_ud = p_ud;
}

UndoRedo::~UndoRedo() {
	clear_history();
}

Variant UndoRedo::_add_do_method(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}

	if (p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING_NAME;
		return Variant();
	}

	r_error.error = Callable::CallError::CALL_OK;

	Object *object = *p_args[0];
	StringName method = *p_args[1];

	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(VARIANT_ARG_MAX, p_argcount - 2); ++i) {
		v[i] = *p_args[i + 2];
	}

	static_assert(VARIANT_ARG_MAX == 5, "This code needs to be updated if VARIANT_ARG_MAX != 5");
	add_do_method(object, method, v[0], v[1], v[2], v[3], v[4]);
	return Variant();
}

Variant UndoRedo::_add_undo_method(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}

	if (p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING_NAME;
		return Variant();
	}

	r_error.error = Callable::CallError::CALL_OK;

	Object *object = *p_args[0];
	StringName method = *p_args[1];

	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(VARIANT_ARG_MAX, p_argcount - 2); ++i) {
		v[i] = *p_args[i + 2];
	}

	static_assert(VARIANT_ARG_MAX == 5, "This code needs to be updated if VARIANT_ARG_MAX != 5");
	add_undo_method(object, method, v[0], v[1], v[2], v[3], v[4]);
	return Variant();
}

void UndoRedo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_action", "name", "merge_mode"), &UndoRedo::create_action, DEFVAL(MERGE_DISABLE));
	ClassDB::bind_method(D_METHOD("commit_action"), &UndoRedo::commit_action);
	ClassDB::bind_method(D_METHOD("is_committing_action"), &UndoRedo::is_committing_action);

	{
		MethodInfo mi;
		mi.name = "add_do_method";
		mi.arguments.push_back(PropertyInfo(Variant::OBJECT, "object"));
		mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "add_do_method", &UndoRedo::_add_do_method, mi, varray(), false);
	}

	{
		MethodInfo mi;
		mi.name = "add_undo_method";
		mi.arguments.push_back(PropertyInfo(Variant::OBJECT, "object"));
		mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "add_undo_method", &UndoRedo::_add_undo_method, mi, varray(), false);
	}

	ClassDB::bind_method(D_METHOD("add_do_property", "object", "property", "value"), &UndoRedo::add_do_property);
	ClassDB::bind_method(D_METHOD("add_undo_property", "object", "property", "value"), &UndoRedo::add_undo_property);
	ClassDB::bind_method(D_METHOD("add_do_reference", "object"), &UndoRedo::add_do_reference);
	ClassDB::bind_method(D_METHOD("add_undo_reference", "object"), &UndoRedo::add_undo_reference);
	ClassDB::bind_method(D_METHOD("clear_history", "increase_version"), &UndoRedo::clear_history, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("get_current_action_name"), &UndoRedo::get_current_action_name);
	ClassDB::bind_method(D_METHOD("has_undo"), &UndoRedo::has_undo);
	ClassDB::bind_method(D_METHOD("has_redo"), &UndoRedo::has_redo);
	ClassDB::bind_method(D_METHOD("get_version"), &UndoRedo::get_version);
	ClassDB::bind_method(D_METHOD("redo"), &UndoRedo::redo);
	ClassDB::bind_method(D_METHOD("undo"), &UndoRedo::undo);

	ADD_SIGNAL(MethodInfo("version_changed"));

	BIND_ENUM_CONSTANT(MERGE_DISABLE);
	BIND_ENUM_CONSTANT(MERGE_ENDS);
	BIND_ENUM_CONSTANT(MERGE_ALL);
}
