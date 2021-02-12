/*************************************************************************/
/*  editor_actions.cpp                                                   */
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

#include "editor/editor_actions.h"

void EditorActions::add_action(StringName p_name, const Callable &p_callable) {
	ERR_FAIL_COND_MSG(callables.has(p_name), "The EditorAction '" + String(p_name) + "' already exists. Unable to add it.");
	callables[p_name] = p_callable;
}

void EditorActions::add_action_obj(StringName p_name, const Object *p_object, const StringName &p_method) {
	ERR_FAIL_COND_MSG(callables.has(p_name), "The EditorAction '" + String(p_name) + "' already exists. Unable to add it.");
	callables[p_name] = Callable(p_object, p_method);
}

void EditorActions::remove_action(StringName p_name) {
	ERR_FAIL_COND_MSG(!callables.has(p_name), "The EditorAction '" + String(p_name) + "' does not exist. Unable to remove it.");
	callables.erase(p_name);
}

void EditorActions::get_action_list(List<StringName> *p_list) const {
	//callables.get_key_list(p_list);
}

Callable EditorActions::get_action(StringName p_name) {
	ERR_FAIL_COND_V_MSG(!callables.has(p_name), Callable(), "The EditorAction '" + String(p_name) + "' does not exist. Unable to get it.");
	return callables[p_name];
}

void EditorActions::add_on_action_executing(StringName p_name, const Callable &p_callable, Array params) {
	ERR_FAIL_COND_MSG(callables_on_executing.has(p_name), "Callback for '" + String(p_name) + "' already exists. Unable to add it.");
	callables_on_executing[p_name].push_back(Pair<Callable, Array>(p_callable, params));
}

void EditorActions::add_on_action_executed(StringName p_name, const Callable &p_callable, Array params) {
	ERR_FAIL_COND_MSG(callables_on_executed.has(p_name), "Callback for '" + String(p_name) + "' already exists. Unable to add it.");
	print_line(p_name);
	callables_on_executed[p_name].push_back(Pair<Callable, Array>(p_callable, params));
}

void EditorActions::execute_action(StringName action_name, const Variant **params, int p_argcount) {
	ERR_FAIL_COND_MSG(!callables.has(action_name), "Execute action " + String(action_name) + " not found");
	if (callables_on_executing.has(action_name)) {
		for (List<Pair<Callable, Array>>::Element *E = callables_on_executing[action_name].front(); E; E = E->next()) {
			Array &c_params = E->get().second;
			Vector<Variant> v_arr;
			for (int i = 0; i < c_params.size(); i++) {
				v_arr.push_back(c_params[i]);
			}
			for (int i = 0; i < p_argcount; i++) {
				v_arr.push_back(*params[i]);
			}
			const Variant *co_params = v_arr.ptr();
			E->get().first.call_deferred(&co_params, c_params.size() + p_argcount);
		}
	}
	callables[action_name].call_deferred(params, p_argcount);
	if (callables_on_executed.has(action_name)) {
		for (List<Pair<Callable, Array>>::Element *E = callables_on_executed[action_name].front(); E; E = E->next()) {
			Array &c_params = E->get().second;
			Vector<Variant> v_arr;
			for (int i = 0; i < c_params.size(); i++) {
				v_arr.push_back(c_params[i]);
			}
			for (int i = 0; i < p_argcount; i++) {
				v_arr.push_back(*params[i]);
			}
			const Variant *co_params = v_arr.ptr();
			E->get().first.call_deferred(&co_params, c_params.size() + p_argcount);
		}
	}
}

Variant EditorActions::execute_action_fold(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	const Variant &p_name = *p_args[p_argcount - 1];
	ERR_FAIL_COND_V_MSG(p_name.get_type() != Variant::STRING_NAME, Variant(), "Execute action failed because name was not StringName");
	execute_action(p_name, p_args, p_argcount - 1);
	return Variant();
}

void EditorActions::_execute_action(StringName action_name, Array params) {
	Vector<Variant> v_arr;
	for (int i = 0; i < params.size(); i++) {
		v_arr.push_back(params[i]);
	}
	const Variant *ptr = v_arr.ptr();
	execute_action(action_name, &ptr, v_arr.size());
}

Callable EditorActions::get_execute_callable() {
	return Callable(this, "execute_action_fold");
}

void EditorActions::clear_execute_cb() {
	callables_on_executed.clear();
	callables_on_executing.clear();
}

Array EditorActions::_get_action_list() const {
	Array ret;
	List<StringName> lst;
	get_action_list(&lst);
	for (List<StringName>::Element *E = lst.front(); E; E = E->next()) {
		ret.append(E->get());
	}
	return ret;
}

void EditorActions::_bind_methods() {
	//ClassDB::bind_method(D_METHOD("add_action", "name", "callable"), &EditorActions::add_action);
	//ClassDB::bind_method(D_METHOD("add_action_obj", "name", "object", "method"), &EditorActions::add_action_obj);
	//ClassDB::bind_method(D_METHOD("remove_action", "name"), &EditorActions::remove_action);

	ClassDB::bind_method("get_action_list", &EditorActions::_get_action_list);
	ClassDB::bind_method(D_METHOD("get_action", "name"), &EditorActions::get_action);

	ClassDB::bind_method(D_METHOD("add_on_action_executing", "name", "callable", "params"), &EditorActions::add_on_action_executing);
	ClassDB::bind_method(D_METHOD("add_on_action_executed", "name", "callable", "params"), &EditorActions::add_on_action_executed);

	ClassDB::bind_method(D_METHOD("execute_action", "action", "params"), &EditorActions::_execute_action);

	{
		MethodInfo mi;
		mi.name = "execute_action_fold";
		mi.arguments.push_back(PropertyInfo(Variant::STRING_NAME, "name"));
		mi.arguments.push_back(PropertyInfo(Variant::VARIANT_MAX, "p1"));
		mi.arguments.push_back(PropertyInfo(Variant::VARIANT_MAX, "p2"));
		mi.arguments.push_back(PropertyInfo(Variant::VARIANT_MAX, "p3"));
		mi.arguments.push_back(PropertyInfo(Variant::VARIANT_MAX, "p4"));
		mi.arguments.push_back(PropertyInfo(Variant::VARIANT_MAX, "p5"));

		Vector<Variant> v_arr;
		v_arr.push_back("");
		v_arr.push_back(Variant());
		v_arr.push_back(Variant());
		v_arr.push_back(Variant());
		v_arr.push_back(Variant());
		v_arr.push_back(Variant());

		ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "execute_action_fold", &EditorActions::execute_action_fold, mi, v_arr, false);
	}

	ClassDB::bind_method("clear_execute_cb", &EditorActions::clear_execute_cb);
}

EditorActions::EditorActions() {}
