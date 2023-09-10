/**************************************************************************/
/*  script_instance.cpp                                                   */
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

#include "script_instance.h"

#include "core/object/script_language.h"

Variant ScriptInstance::call_const(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	return callp(p_method, p_args, p_argcount, r_error);
}

void ScriptInstance::get_property_state(List<Pair<StringName, Variant>> &state) {
	List<PropertyInfo> pinfo;
	get_property_list(&pinfo);
	for (const PropertyInfo &E : pinfo) {
		if (E.usage & PROPERTY_USAGE_STORAGE) {
			Pair<StringName, Variant> p;
			p.first = E.name;
			if (get(p.first, p.second)) {
				state.push_back(p);
			}
		}
	}
}

void ScriptInstance::property_set_fallback(const StringName &, const Variant &, bool *r_valid) {
	if (r_valid) {
		*r_valid = false;
	}
}

Variant ScriptInstance::property_get_fallback(const StringName &, bool *r_valid) {
	if (r_valid) {
		*r_valid = false;
	}
	return Variant();
}

const Variant ScriptInstance::get_rpc_config() const {
	return get_script()->get_rpc_config();
}

ScriptInstance::~ScriptInstance() {
}
