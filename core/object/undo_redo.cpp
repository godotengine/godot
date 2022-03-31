/*************************************************************************/
/*  undo_redo.cpp                                                        */
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

#include "undo_redo.h"

#include "core/io/resource.h"
#include "core/os/os.h"
#include "core/templates/local_vector.h"

void UndoRedo::Operation::delete_reference() {
	if (type != Operation::TYPE_REFERENCE) {
		return;
	}
	if (ref.is_valid()) {
		ref.unref();
	} else {
		Object *obj = ObjectDB::get_instance(object);
		if (obj) {
			memdelete(obj);
		}
	}
}

void UndoRedo::_discard_redo() {
	if (*p_current_action == p_actions->size() - 1) {
		return;
	}

	for (int i = *p_current_action + 1; i < p_actions->size(); i++) {
		for (Operation &E : p_actions->write[i].do_ops) {
			E.delete_reference();
		}
		//ERASE do data
	}

	p_actions->resize(*p_current_action + 1);
}

bool UndoRedo::_redo(bool p_execute) {
	ERR_FAIL_COND_V(action_level > 0, false);

	if ((*p_current_action + 1) >= p_actions->size()) {
		return false; //nothing to redo
	}

	*p_current_action = *p_current_action + 1;
	if (p_execute) {
		_process_operation_list(p_actions->write[*p_current_action].do_ops.front());
	}
	version++;
	emit_signal(SNAME("version_changed"));

	return true;
}

void UndoRedo::create_action(const String &p_name, MergeMode p_mode) {
	uint64_t ticks = OS::get_singleton()->get_ticks_msec();

	if (action_level == 0) {
		discarding_redo = true;
		if (p_mode != MERGE_DISABLE && p_actions->size() && (*p_actions)[p_actions->size() - 1].name == p_name && (*p_actions)[p_actions->size() - 1].last_tick + 800 > ticks) {
			merge_mode = p_mode;
			merging = true;
		} else {
			merge_mode = MERGE_DISABLE;
		}
	}

	action = Action(); // Clear last committed action.
	action.name = p_name;
	action.last_tick = ticks;

	action_level++;

	force_keep_in_merge_ends = false;
}

void UndoRedo::add_do_methodp(Object *p_object, const StringName &p_method, const Variant **p_args, int p_argcount) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());
	set_action_context(p_object);

	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<RefCounted>(p_object)) {
		do_op.ref = Ref<RefCounted>(Object::cast_to<RefCounted>(p_object));
	}

	do_op.type = Operation::TYPE_METHOD;
	do_op.name = p_method;

	for (int i = 0; i < p_argcount; i++) {
		do_op.args.push_back(*p_args[i]);
	}
	action.do_ops.push_back(do_op);
}

void UndoRedo::add_undo_methodp(Object *p_object, const StringName &p_method, const Variant **p_args, int p_argcount) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());
	set_action_context(p_object);

	// No undo if the merge mode is MERGE_ENDS
	if (!force_keep_in_merge_ends && merge_mode == MERGE_ENDS) {
		return;
	}

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<RefCounted>(p_object)) {
		undo_op.ref = Ref<RefCounted>(Object::cast_to<RefCounted>(p_object));
	}

	undo_op.type = Operation::TYPE_METHOD;
	undo_op.force_keep_in_merge_ends = force_keep_in_merge_ends;
	undo_op.name = p_method;

	for (int i = 0; i < p_argcount; i++) {
		undo_op.args.push_back(*p_args[i]);
	}
	action.undo_ops.push_back(undo_op);
}

void UndoRedo::add_do_property(Object *p_object, const StringName &p_property, const Variant &p_value) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());
	set_action_context(p_object);

	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<RefCounted>(p_object)) {
		do_op.ref = Ref<RefCounted>(Object::cast_to<RefCounted>(p_object));
	}

	do_op.type = Operation::TYPE_PROPERTY;
	do_op.name = p_property;
	do_op.args.push_back(p_value);
	action.do_ops.push_back(do_op);
}

void UndoRedo::add_undo_property(Object *p_object, const StringName &p_property, const Variant &p_value) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());
	set_action_context(p_object);

	// No undo if the merge mode is MERGE_ENDS
	if (!force_keep_in_merge_ends && merge_mode == MERGE_ENDS) {
		return;
	}

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<RefCounted>(p_object)) {
		undo_op.ref = Ref<RefCounted>(Object::cast_to<RefCounted>(p_object));
	}

	undo_op.type = Operation::TYPE_PROPERTY;
	undo_op.force_keep_in_merge_ends = force_keep_in_merge_ends;
	undo_op.name = p_property;
	undo_op.args.push_back(p_value);
	action.undo_ops.push_back(undo_op);
}

void UndoRedo::add_do_reference(Object *p_object) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());
	set_action_context(p_object);

	Operation do_op;
	do_op.object = p_object->get_instance_id();
	if (Object::cast_to<RefCounted>(p_object)) {
		do_op.ref = Ref<RefCounted>(Object::cast_to<RefCounted>(p_object));
	}

	do_op.type = Operation::TYPE_REFERENCE;
	action.do_ops.push_back(do_op);
}

void UndoRedo::add_undo_reference(Object *p_object) {
	ERR_FAIL_COND(p_object == nullptr);
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());
	set_action_context(p_object);

	// No undo if the merge mode is MERGE_ENDS
	if (!force_keep_in_merge_ends && merge_mode == MERGE_ENDS) {
		return;
	}

	Operation undo_op;
	undo_op.object = p_object->get_instance_id();
	if (Object::cast_to<RefCounted>(p_object)) {
		undo_op.ref = Ref<RefCounted>(Object::cast_to<RefCounted>(p_object));
	}

	undo_op.type = Operation::TYPE_REFERENCE;
	undo_op.force_keep_in_merge_ends = force_keep_in_merge_ends;
	action.undo_ops.push_back(undo_op);
}

void UndoRedo::start_force_keep_in_merge_ends() {
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());

	force_keep_in_merge_ends = true;
}

void UndoRedo::end_force_keep_in_merge_ends() {
	ERR_FAIL_COND(action_level <= 0);
	//ERR_FAIL_COND((*p_current_action + 1) >= p_actions->size());

	force_keep_in_merge_ends = false;
}

void UndoRedo::_pop_history_tail() { // TODO
	/*_discard_redo();

	if (!actions.size()) {
		return;
	}

	for (Operation &E : actions.write[0].undo_ops) {
		E.delete_reference();
	}

	actions.remove_at(0);
	if (current_action >= 0) {
		current_action--;
	}*/
}

bool UndoRedo::is_committing_action() const {
	return committing > 0;
}

void UndoRedo::commit_action(bool p_execute) {
	ERR_FAIL_COND(action_level <= 0);
	action_level--;
	if (action_level > 0) {
		return; // Still nested.
	}

	if (object_context_id == get_context_id_default()) {
		WARN_PRINT(vformat("Context for action '%s' could not be determined automatically, and the action was lost. The context must be set via set_action_context.", action.name));
	}
	set_editor_context(object_context_id);

	if (discarding_redo) {
		_discard_redo();
	}

	if (merging) {
		*p_current_action = p_actions->size() - 2;

		if (merge_mode == MERGE_ENDS) {
			// Clear all do ops from last action if they are not forced kept.
			LocalVector<List<Operation>::Element *> to_remove;
			for (List<Operation>::Element *E = p_actions->write[*p_current_action + 1].do_ops.front(); E; E = E->next()) {
				if (!E->get().force_keep_in_merge_ends) {
					to_remove.push_back(E);
				}
			}

			for (unsigned int i = 0; i < to_remove.size(); i++) {
				List<Operation>::Element *E = to_remove[i];
				// Delete all object references
				if (E->get().type == Operation::TYPE_REFERENCE) {
					Object *obj = ObjectDB::get_instance(E->get().object);

					if (obj) {
						memdelete(obj);
					}
				}
				E->erase();
			}

			for (List<Operation>::Element *E = p_actions->write[*p_current_action + 1].do_ops.front(); E; E = E->next()) {
				action.do_ops.push_back(E->get());
			}
			for (List<Operation>::Element *E = p_actions->write[*p_current_action + 1].undo_ops.front(); E; E = E->next()) {
				action.undo_ops.push_back(E->get());
			}

			p_actions->remove_at(*p_current_action + 1);
		}

		version--;
		merging = false;
	}

	context_actions[object_context_id].actions.push_back(action);
	object_context_id = get_context_id_default();

	committing++;
	_redo(p_execute); // Perform action.
	committing--;

	if (callback && p_actions->size() > 0) {
		callback(callback_ud, action.name);
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
				int argc = op.args.size();
				Vector<const Variant *> argptrs;
				argptrs.resize(argc);

				for (int i = 0; i < argc; i++) {
					argptrs.write[i] = &op.args[i];
				}

				Callable::CallError ce;
				obj->callp(op.name, (const Variant **)argptrs.ptr(), argc, ce);
				if (ce.error != Callable::CallError::CALL_OK) {
					ERR_PRINT("Error calling UndoRedo method operation '" + String(op.name) + "': " + Variant::get_call_error_text(obj, op.name, (const Variant **)argptrs.ptr(), argc, ce));
				}
#ifdef TOOLS_ENABLED
				Resource *res = Object::cast_to<Resource>(obj);
				if (res) {
					res->set_edited(true);
				}

#endif

				if (method_callback) {
					method_callback(method_callback_ud, obj, op.name, (const Variant **)argptrs.ptr(), argc);
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

void UndoRedo::set_action_context(Object *p_object) {
	if (object_context_id == get_context_id_default()) {
		if (p_object->is_class("Resource")) {
			object_context_id = p_object->get_instance_id();
		} else if (p_object->call("get_edited_resource").get_type() != Variant::NIL) {
			Ref<Resource> r_resource = p_object->call("get_edited_resource");
			object_context_id = r_resource.ptr()->get_instance_id();
		} else if (p_object->is_class("Node") && ClassDB::can_instantiate(p_object->get_class())) {
			if (p_object->get("scene_file_path") == "") {
				p_object = p_object->call("get_owner");
			}
			if (p_object != nullptr) {
				object_context_id = p_object->get_instance_id();
			}
		}
	}
}

void UndoRedo::set_editor_context(ObjectID object_id) {
	if (!context_actions.has(object_id)) {
		context_actions[object_id] = { {}, -1 };
	}
	p_actions = &context_actions[object_id].actions;
	p_current_action = &context_actions[object_id].current_action;
}

bool UndoRedo::redo() {
	return _redo(true);
}

bool UndoRedo::undo() {
	ERR_FAIL_COND_V(action_level > 0, false);
	if (*p_current_action < 0) {
		return false; //nothing to redo
	}
	_process_operation_list(p_actions->write[*p_current_action].undo_ops.front());
	*p_current_action = *p_current_action - 1;
	version--;
	emit_signal(SNAME("version_changed"));

	return true;
}

int UndoRedo::get_history_count() {
	ERR_FAIL_COND_V(action_level > 0, -1);

	//return actions.size(); TODO
	return 0;
}

int UndoRedo::get_current_action() {
	ERR_FAIL_COND_V(action_level > 0, -1);

	//return current_action; TODO
	return -1;
}

String UndoRedo::get_action_name(int p_id) {
	ERR_FAIL_INDEX_V(p_id, p_actions->size(), "");

	//return actions[p_id].name; TODO
	return "";
}

void UndoRedo::clear_history(bool p_increase_version) { // TODO (many functions need parameter specifying undo-context)
	ERR_FAIL_COND(action_level > 0);
	//_discard_redo();

	/*while (actions.size()) { TODO
		_pop_history_tail();
	}*/

	if (p_increase_version) {
		version++;
		emit_signal(SNAME("version_changed"));
	}
}

String UndoRedo::get_current_action_name(ObjectID object_id) const {
	ERR_FAIL_COND_V(action_level > 0, "");
	if (context_actions[object_id].current_action < 0) {
		return "";
	}
	return context_actions[object_id].actions[context_actions[object_id].current_action].name;
}

bool UndoRedo::has_undo(ObjectID object_id) const { // TODO: more friendly way of specifying context
	return context_actions.has(object_id) && context_actions[object_id].current_action >= 0;
}

bool UndoRedo::has_redo(ObjectID object_id) const {
	return context_actions.has(object_id) && context_actions[object_id].current_action + 1 < context_actions[object_id].actions.size();
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
	method_callback_ud = p_ud;
}

void UndoRedo::set_property_notify_callback(PropertyNotifyCallback p_property_callback, void *p_ud) {
	property_callback = p_property_callback;
	prop_callback_ud = p_ud;
}

UndoRedo::~UndoRedo() {
	clear_history();
}

void UndoRedo::_add_do_method(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return;
	}

	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return;
	}

	if (p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING_NAME;
		return;
	}

	r_error.error = Callable::CallError::CALL_OK;

	Object *object = *p_args[0];
	StringName method = *p_args[1];

	add_do_methodp(object, method, p_args + 2, p_argcount - 2);
}

void UndoRedo::_add_undo_method(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 0;
		return;
	}

	if (p_args[0]->get_type() != Variant::OBJECT) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 0;
		r_error.expected = Variant::OBJECT;
		return;
	}

	if (p_args[1]->get_type() != Variant::STRING_NAME && p_args[1]->get_type() != Variant::STRING) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = 1;
		r_error.expected = Variant::STRING_NAME;
		return;
	}

	r_error.error = Callable::CallError::CALL_OK;

	Object *object = *p_args[0];
	StringName method = *p_args[1];

	add_undo_methodp(object, method, p_args + 2, p_argcount - 2);
}

void UndoRedo::_bind_methods() {
	ClassDB::bind_method(D_METHOD("create_action", "name", "merge_mode"), &UndoRedo::create_action, DEFVAL(MERGE_DISABLE));
	ClassDB::bind_method(D_METHOD("commit_action", "execute"), &UndoRedo::commit_action, DEFVAL(true));
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

	ClassDB::bind_method(D_METHOD("start_force_keep_in_merge_ends"), &UndoRedo::start_force_keep_in_merge_ends);
	ClassDB::bind_method(D_METHOD("end_force_keep_in_merge_ends"), &UndoRedo::end_force_keep_in_merge_ends);

	ClassDB::bind_method(D_METHOD("get_history_count"), &UndoRedo::get_history_count);
	ClassDB::bind_method(D_METHOD("get_current_action"), &UndoRedo::get_current_action);
	ClassDB::bind_method(D_METHOD("get_action_name", "id"), &UndoRedo::get_action_name);
	ClassDB::bind_method(D_METHOD("clear_history", "increase_version"), &UndoRedo::clear_history, DEFVAL(true));

	ClassDB::bind_method(D_METHOD("get_current_action_name"), &UndoRedo::get_current_action_name);

	ClassDB::bind_method(D_METHOD("has_undo", "object_id"), &UndoRedo::has_undo);
	ClassDB::bind_method(D_METHOD("has_redo"), &UndoRedo::has_redo);
	ClassDB::bind_method(D_METHOD("get_version"), &UndoRedo::get_version);
	ClassDB::bind_method(D_METHOD("redo"), &UndoRedo::redo);
	ClassDB::bind_method(D_METHOD("undo"), &UndoRedo::undo);

	ADD_SIGNAL(MethodInfo("version_changed"));

	BIND_ENUM_CONSTANT(MERGE_DISABLE);
	BIND_ENUM_CONSTANT(MERGE_ENDS);
	BIND_ENUM_CONSTANT(MERGE_ALL);
}

UndoRedo::UndoRedo() {
	context_actions[ObjectID(uint64_t())] = { {}, -1 };
	p_actions = &context_actions[ObjectID(uint64_t())].actions;
	p_current_action = &context_actions[ObjectID(uint64_t())].current_action;
}
