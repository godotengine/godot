/*************************************************************************/
/*  editor_actions.h                                                     */
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

#ifndef EDITOR_ACTIONS_H
#define EDITOR_ACTIONS_H

#include "core/pair.h"
#include "core/reference.h"

class EditorActions : public Reference {
	GDCLASS(EditorActions, Reference);

	HashMap<String, Callable> callables;

	HashMap<StringName, List<Pair<Callable, Array>>> callables_on_executing;
	HashMap<StringName, List<Pair<Callable, Array>>> callables_on_executed;

protected:
	static void _bind_methods();
	Array _get_action_list() const;

public:
	EditorActions();

	void add_action(StringName p_name, const Callable &p_callable);
	void add_action_obj(StringName p_name, const Object *p_object, const StringName &p_method);
	void remove_action(StringName p_name);

	void get_action_list(List<StringName> *p_list) const;
	Callable get_action(StringName p_name);

	void add_on_action_executing(StringName p_name, const Callable &p_callable, Array params);
	void add_on_action_executed(StringName p_name, const Callable &p_callable, Array params);

	void execute_action(StringName action_name, const Variant **params, int p_argcount);
	Variant execute_action_fold(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Callable get_execute_callable();

	void clear_execute_cb();

protected:
	void _execute_action(StringName action_name, Array params);
};

#endif EDITOR_ACTIONS_H
