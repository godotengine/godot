/*************************************************************************/
/*  undo_redo.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#include "undo_redo.h"

#include "os/os.h"

void UndoRedo::_discard_redo() {

	if (current_action == actions.size() - 1)
		return;

	for (int i = current_action + 1; i < actions.size(); i++) {

		for (List<Operation>::Element *E = actions[i].do_ops.front(); E; E = E->next()) {

			if (E->get().type == Operation::TYPE_REFERENCE) {

				Object *obj = ObjectDB::get_instance(E->get().object);
				if (obj)
					memdelete(obj);
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
				List<Operation>::Element *E = actions[current_action + 1].do_ops.front();

				while (E) {

					if (E->get().type == Operation::TYPE_REFERENCE) {

						Object *obj = ObjectDB::get_instance(E->get().object);

						if (obj)
							memdelete(obj);
					}

					E = E->next();
					actions[current_action + 1].do_ops.pop_front();
				}
			}

			actions[actions.size() - 1].last_tick = ticks;

			merge_mode = p_mode;

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

void UndoRedo::add_do_method(Object *p_object, const String &p_method, VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());
	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object))
		do_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));

	do_op.type = Operation::TYPE_METHOD;
	do_op.name = p_method;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		do_op.args[i] = *argptr[i];
	}
	actions[current_action + 1].do_ops.push_back(do_op);
}

void UndoRedo::add_undo_method(Object *p_object, const String &p_method, VARIANT_ARG_DECLARE) {

	VARIANT_ARGPTRS
	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());

	// No undo if the merge mode is MERGE_ENDS
	if (merge_mode == MERGE_ENDS)
		return;

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object))
		undo_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));

	undo_op.type = Operation::TYPE_METHOD;
	undo_op.name = p_method;

	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		undo_op.args[i] = *argptr[i];
	}
	actions[current_action + 1].undo_ops.push_back(undo_op);
}
void UndoRedo::add_do_property(Object *p_object, const String &p_property, const Variant &p_value) {

	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());
	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object))
		do_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));

	do_op.type = Operation::TYPE_PROPERTY;
	do_op.name = p_property;
	do_op.args[0] = p_value;
	actions[current_action + 1].do_ops.push_back(do_op);
}
void UndoRedo::add_undo_property(Object *p_object, const String &p_property, const Variant &p_value) {

	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());

	// No undo if the merge mode is MERGE_ENDS
	if (merge_mode == MERGE_ENDS)
		return;

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object))
		undo_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));

	undo_op.type = Operation::TYPE_PROPERTY;
	undo_op.name = p_property;
	undo_op.args[0] = p_value;
	actions[current_action + 1].undo_ops.push_back(undo_op);
}
void UndoRedo::add_do_reference(Object *p_object) {

	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());
	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object))
		do_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));

	do_op.type = Operation::TYPE_REFERENCE;
	actions[current_action + 1].do_ops.push_back(do_op);
}
void UndoRedo::add_undo_reference(Object *p_object) {

	ERR_FAIL_COND(action_level <= 0);
	ERR_FAIL_COND((current_action + 1) >= actions.size());

	// No undo if the merge mode is MERGE_ENDS
	if (merge_mode == MERGE_ENDS)
		return;

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<Resource>(p_object))
		undo_op.resref = Ref<Resource>(Object::cast_to<Resource>(p_object));

	undo_op.type = Operation::TYPE_REFERENCE;
	actions[current_action + 1].undo_ops.push_back(undo_op);
}

void UndoRedo::_pop_history_tail() {

	_discard_redo();

	if (!actions.size())
		return;

	for (List<Operation>::Element *E = actions[0].undo_ops.front(); E; E = E->next()) {

		if (E->get().type == Operation::TYPE_REFERENCE) {

			Object *obj = ObjectDB::get_instance(E->get().object);
			if (obj)
				memdelete(obj);
		}
	}

	actions.remove(0);
	current_action--;
}

void UndoRedo::commit_action() {

	ERR_FAIL_COND(action_level <= 0);
	action_level--;
	if (action_level > 0)
		return; //still nested

	redo(); // perform action

	if (max_steps > 0 && actions.size() > max_steps) {
		//clear early steps

		while (actions.size() > max_steps)
			_pop_history_tail();
	}

	if (callback && actions.size() > 0) {
		callback(callback_ud, actions[actions.size() - 1].name);
	}
}

void UndoRedo::_process_operation_list(List<Operation>::Element *E) {

	for (; E; E = E->next()) {

		Operation &op = E->get();

		Object *obj = ObjectDB::get_instance(op.object);
		if (!obj) {
			//corruption
			clear_history();
			ERR_FAIL_COND(!obj);
		}

		switch (op.type) {

			case Operation::TYPE_METHOD: {

				obj->call(op.name, VARIANT_ARGS_FROM_ARRAY(op.args));
#ifdef TOOLS_ENABLED
				Resource *res = Object::cast_to<Resource>(obj);
				if (res)
					res->set_edited(true);

#endif

				if (method_callback) {
					method_callback(method_callbck_ud, obj, op.name, VARIANT_ARGS_FROM_ARRAY(op.args));
				}
			} break;
			case Operation::TYPE_PROPERTY: {

				obj->set(op.name, op.args[0]);
#ifdef TOOLS_ENABLED
				Resource *res = Object::cast_to<Resource>(obj);
				if (res)
					res->set_edited(true);
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

void UndoRedo::redo() {

	ERR_FAIL_COND(action_level > 0);

	if ((current_action + 1) >= actions.size())
		return; //nothing to redo
	current_action++;

	_process_operation_list(actions[current_action].do_ops.front());
	version++;
}

void UndoRedo::undo() {

	ERR_FAIL_COND(action_level > 0);
	if (current_action < 0)
		return; //nothing to redo
	_process_operation_list(actions[current_action].undo_ops.front());
	current_action--;
	version--;
}

void UndoRedo::clear_history() {

	ERR_FAIL_COND(action_level > 0);
	_discard_redo();

	while (actions.size())
		_pop_history_tail();

	//version++;
}

String UndoRedo::get_current_action_name() const {

	ERR_FAIL_COND_V(action_level > 0, "");
	if (current_action < 0)
		return ""; //nothing to redo
	return actions[current_action].name;
}

void UndoRedo::set_max_steps(int p_max_steps) {

	max_steps = p_max_steps;
}

int UndoRedo::get_max_steps() const {

	return max_steps;
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

UndoRedo::UndoRedo() {

	version = 1;
	action_level = 0;
	current_action = -1;
	max_steps = -1;
	merge_mode = MERGE_DISABLE;
	callback = NULL;
	callback_ud = NULL;

	method_callbck_ud = NULL;
	prop_callback_ud = NULL;
	method_callback = NULL;
	property_callback = NULL;
}

UndoRedo::~UndoRedo() {

	clear_history();
}

Variant UndoRedo::_add_do_method(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (p_argcount < 2) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}

	if (p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING;
		return Variant();
	}

	r_error.error = Variant::CallError::CALL_OK;

	Object *object = *p_args[0];
	String method = *p_args[1];

	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(VARIANT_ARG_MAX, p_argcount - 2); ++i) {

		v[i] = *p_args[i + 2];
	}

	add_do_method(object, method, v[0], v[1], v[2], v[3], v[4]);
	return Variant();
}

Variant UndoRedo::_add_undo_method(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	if (p_argcount < 2) {
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return Variant();
	}

	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}

	if (p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING;
		return Variant();
	}

	r_error.error = Variant::CallError::CALL_OK;

	Object *object = *p_args[0];
	String method = *p_args[1];

	Variant v[VARIANT_ARG_MAX];

	for (int i = 0; i < MIN(VARIANT_ARG_MAX, p_argcount - 2); ++i) {

		v[i] = *p_args[i + 2];
	}

	add_undo_method(object, method, v[0], v[1], v[2], v[3], v[4]);
	return Variant();
}

void UndoRedo::_bind_methods() {

	ClassDB::bind_method(D_METHOD("create_action", "name", "merge_mode"), &UndoRedo::create_action, DEFVAL(MERGE_DISABLE));
	ClassDB::bind_method(D_METHOD("commit_action"), &UndoRedo::commit_action);

	//ClassDB::bind_method(D_METHOD("add_do_method","p_object", "p_method", "VARIANT_ARG_LIST"),&UndoRedo::add_do_method);
	//ClassDB::bind_method(D_METHOD("add_undo_method","p_object", "p_method", "VARIANT_ARG_LIST"),&UndoRedo::add_undo_method);

	{
		MethodInfo mi;
		mi.name = "add_do_method";
		mi.arguments.push_back(PropertyInfo(Variant::OBJECT, "object"));
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "add_do_method", &UndoRedo::_add_do_method, mi);
	}

	{
		MethodInfo mi;
		mi.name = "add_undo_method";
		mi.arguments.push_back(PropertyInfo(Variant::OBJECT, "object"));
		mi.arguments.push_back(PropertyInfo(Variant::STRING, "method"));

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "add_undo_method", &UndoRedo::_add_undo_method, mi);
	}

	ClassDB::bind_method(D_METHOD("add_do_property", "object", "property", "value"), &UndoRedo::add_do_property);
	ClassDB::bind_method(D_METHOD("add_undo_property", "object", "property", "value"), &UndoRedo::add_undo_property);
	ClassDB::bind_method(D_METHOD("add_do_reference", "object"), &UndoRedo::add_do_reference);
	ClassDB::bind_method(D_METHOD("add_undo_reference", "object"), &UndoRedo::add_undo_reference);
	ClassDB::bind_method(D_METHOD("clear_history"), &UndoRedo::clear_history);
	ClassDB::bind_method(D_METHOD("get_current_action_name"), &UndoRedo::get_current_action_name);
	ClassDB::bind_method(D_METHOD("get_version"), &UndoRedo::get_version);
	ClassDB::bind_method(D_METHOD("set_max_steps", "max_steps"), &UndoRedo::set_max_steps);
	ClassDB::bind_method(D_METHOD("get_max_steps"), &UndoRedo::get_max_steps);
	ClassDB::bind_method(D_METHOD("redo"), &UndoRedo::redo);
	ClassDB::bind_method(D_METHOD("undo"), &UndoRedo::undo);

	BIND_ENUM_CONSTANT(MERGE_DISABLE);
	BIND_ENUM_CONSTANT(MERGE_ENDS);
	BIND_ENUM_CONSTANT(MERGE_ALL);
}
