/*************************************************************************/
/*  gdscript_function.cpp                                                */
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

#include "gdscript_function.h"

#include "gdscript.h"

const int *GDScriptFunction::get_code() const {
	return _code_ptr;
}

int GDScriptFunction::get_code_size() const {
	return _code_size;
}

Variant GDScriptFunction::get_constant(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, constants.size(), "<errconst>");
	return constants[p_idx];
}

StringName GDScriptFunction::get_global_name(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, global_names.size(), "<errgname>");
	return global_names[p_idx];
}

int GDScriptFunction::get_default_argument_count() const {
	return _default_arg_count;
}

int GDScriptFunction::get_default_argument_addr(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, default_arguments.size(), -1);
	return default_arguments[p_idx];
}

GDScriptDataType GDScriptFunction::get_return_type() const {
	return return_type;
}

GDScriptDataType GDScriptFunction::get_argument_type(int p_idx) const {
	ERR_FAIL_INDEX_V(p_idx, argument_types.size(), GDScriptDataType());
	return argument_types[p_idx];
}

StringName GDScriptFunction::get_name() const {
	return name;
}

int GDScriptFunction::get_max_stack_size() const {
	return _stack_size;
}

struct _GDFKC {
	int order = 0;
	List<int> pos;
};

struct _GDFKCS {
	int order = 0;
	StringName id;
	int pos = 0;

	bool operator<(const _GDFKCS &p_r) const {
		return order < p_r.order;
	}
};

void GDScriptFunction::debug_get_stack_member_state(int p_line, List<Pair<StringName, int>> *r_stackvars) const {
	int oc = 0;
	Map<StringName, _GDFKC> sdmap;
	for (const List<StackDebug>::Element *E = stack_debug.front(); E; E = E->next()) {
		const StackDebug &sd = E->get();
		if (sd.line > p_line) {
			break;
		}

		if (sd.added) {
			if (!sdmap.has(sd.identifier)) {
				_GDFKC d;
				d.order = oc++;
				d.pos.push_back(sd.pos);
				sdmap[sd.identifier] = d;

			} else {
				sdmap[sd.identifier].pos.push_back(sd.pos);
			}
		} else {
			ERR_CONTINUE(!sdmap.has(sd.identifier));

			sdmap[sd.identifier].pos.pop_back();
			if (sdmap[sd.identifier].pos.is_empty()) {
				sdmap.erase(sd.identifier);
			}
		}
	}

	List<_GDFKCS> stackpositions;
	for (Map<StringName, _GDFKC>::Element *E = sdmap.front(); E; E = E->next()) {
		_GDFKCS spp;
		spp.id = E->key();
		spp.order = E->get().order;
		spp.pos = E->get().pos.back()->get();
		stackpositions.push_back(spp);
	}

	stackpositions.sort();

	for (List<_GDFKCS>::Element *E = stackpositions.front(); E; E = E->next()) {
		Pair<StringName, int> p;
		p.first = E->get().id;
		p.second = E->get().pos;
		r_stackvars->push_back(p);
	}
}

GDScriptFunction::GDScriptFunction() {
	name = "<anonymous>";
#ifdef DEBUG_ENABLED
	{
		MutexLock lock(GDScriptLanguage::get_singleton()->lock);
		GDScriptLanguage::get_singleton()->function_list.add(&function_list);
	}
#endif
}

GDScriptFunction::~GDScriptFunction() {
	for (int i = 0; i < lambdas.size(); i++) {
		memdelete(lambdas[i]);
	}

#ifdef DEBUG_ENABLED

	MutexLock lock(GDScriptLanguage::get_singleton()->lock);

	GDScriptLanguage::get_singleton()->function_list.remove(&function_list);
#endif
}

/////////////////////

Variant GDScriptFunctionState::_signal_callback(const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	Variant arg;
	r_error.error = Callable::CallError::CALL_OK;

	if (p_argcount == 0) {
		r_error.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		r_error.argument = 1;
		return Variant();
	} else if (p_argcount == 1) {
		//noooneee
	} else if (p_argcount == 2) {
		arg = *p_args[0];
	} else {
		Array extra_args;
		for (int i = 0; i < p_argcount - 1; i++) {
			extra_args.push_back(*p_args[i]);
		}
		arg = extra_args;
	}

	Ref<GDScriptFunctionState> self = *p_args[p_argcount - 1];

	if (self.is_null()) {
		r_error.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
		r_error.argument = p_argcount - 1;
		r_error.expected = Variant::OBJECT;
		return Variant();
	}

	return resume(arg);
}

bool GDScriptFunctionState::is_valid(bool p_extended_check) const {
	if (function == nullptr) {
		return false;
	}

	if (p_extended_check) {
		MutexLock lock(GDScriptLanguage::get_singleton()->lock);

		// Script gone?
		if (!scripts_list.in_list()) {
			return false;
		}
		// Class instance gone? (if not static function)
		if (state.instance && !instances_list.in_list()) {
			return false;
		}
	}

	return true;
}

Variant GDScriptFunctionState::resume(const Variant &p_arg) {
	ERR_FAIL_COND_V(!function, Variant());
	{
		MutexLock lock(GDScriptLanguage::singleton->lock);

		if (!scripts_list.in_list()) {
#ifdef DEBUG_ENABLED
			ERR_FAIL_V_MSG(Variant(), "Resumed function '" + state.function_name + "()' after await, but script is gone. At script: " + state.script_path + ":" + itos(state.line));
#else
			return Variant();
#endif
		}
		if (state.instance && !instances_list.in_list()) {
#ifdef DEBUG_ENABLED
			ERR_FAIL_V_MSG(Variant(), "Resumed function '" + state.function_name + "()' after await, but class instance is gone. At script: " + state.script_path + ":" + itos(state.line));
#else
			return Variant();
#endif
		}
		// Do these now to avoid locking again after the call
		scripts_list.remove_from_list();
		instances_list.remove_from_list();
	}

	state.result = p_arg;
	Callable::CallError err;
	Variant ret = function->call(nullptr, nullptr, 0, err, &state);

	bool completed = true;

	// If the return value is a GDScriptFunctionState reference,
	// then the function did await again after resuming.
	if (ret.is_ref()) {
		GDScriptFunctionState *gdfs = Object::cast_to<GDScriptFunctionState>(ret);
		if (gdfs && gdfs->function == function) {
			completed = false;
			gdfs->first_state = first_state.is_valid() ? first_state : Ref<GDScriptFunctionState>(this);
		}
	}

	function = nullptr; //cleaned up;
	state.result = Variant();

	if (completed) {
		if (first_state.is_valid()) {
			first_state->emit_signal("completed", ret);
		} else {
			emit_signal("completed", ret);
		}

#ifdef DEBUG_ENABLED
		if (EngineDebugger::is_active()) {
			GDScriptLanguage::get_singleton()->exit_function();
		}
#endif
	}

	return ret;
}

void GDScriptFunctionState::_clear_stack() {
	if (state.stack_size) {
		Variant *stack = (Variant *)state.stack.ptr();
		for (int i = 0; i < state.stack_size; i++) {
			stack[i].~Variant();
		}
		state.stack_size = 0;
	}
}

void GDScriptFunctionState::_bind_methods() {
	ClassDB::bind_method(D_METHOD("resume", "arg"), &GDScriptFunctionState::resume, DEFVAL(Variant()));
	ClassDB::bind_method(D_METHOD("is_valid", "extended_check"), &GDScriptFunctionState::is_valid, DEFVAL(false));
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_signal_callback", &GDScriptFunctionState::_signal_callback, MethodInfo("_signal_callback"));

	ADD_SIGNAL(MethodInfo("completed", PropertyInfo(Variant::NIL, "result", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NIL_IS_VARIANT)));
}

GDScriptFunctionState::GDScriptFunctionState() :
		scripts_list(this),
		instances_list(this) {
}

GDScriptFunctionState::~GDScriptFunctionState() {
	_clear_stack();

	{
		MutexLock lock(GDScriptLanguage::singleton->lock);
		scripts_list.remove_from_list();
		instances_list.remove_from_list();
	}
}
