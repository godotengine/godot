/*************************************************************************/
/*  method_watcher.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "method_watcher.h"

void MethodWatcher::bind_method(const String &p_name, const Variant &p_return) {
	MethodInfo mi;
	mi.m_return = p_return;
	m_methods.insert(p_name, mi)->get();
}

void MethodWatcher::add_property(const String &p_name, const StringName p_setter, const StringName p_getter) {
	SetGetPair setget;
	setget.first = p_setter;
	setget.second = p_getter;
	m_properties.insert(p_name, setget);
}

const Vector<MethodWatcher::Args> MethodWatcher::get_calls(const String &p_name) const {
	MethodMap::Element *method = m_methods.find(p_name);
	if (method) {
		return method->get().m_calls;
	}
	return Vector<MethodWatcher::Args>();
}

Variant MethodWatcher::get(const Variant &p_key, bool *r_valid) {
	const PropertyMap::Element *prop = m_properties.find(p_key);
	if (prop) {
		const StringName &second = prop->get().second;
		if (second) {
			Variant::CallError ce;
			Variant result = this->call(second, NULL, 0, ce);
			if (ce.error == Variant::CallError::CALL_OK) {
				if (r_valid) {
					*r_valid = true;
				}
				return result;
			}
		}
	}
	return Variant();
}

void MethodWatcher::set(const Variant &p_key, const Variant &p_value, bool *r_valid) {
	const PropertyMap::Element *prop = m_properties.find(p_key);
	if (prop) {
		const StringName &first = prop->get().first;
		if (first) {
			const Variant *args[] = { &p_value };
			Variant::CallError ce;
			call(first, args, 1, ce);
			if (ce.error == Variant::CallError::CALL_OK) {
				if (r_valid) {
					*r_valid = true;
				}
			}
		}
	}
}

Variant MethodWatcher::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	MethodMap::Element *method = m_methods.find(p_method);
	r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
	MethodInfo *info = NULL;
	if (method) {
		info = &method->get();
	} else {
		info = &m_methods.insert(p_method, MethodInfo())->get();
	}
	Args args;
	if (p_argcount > 0) {
		args.resize(p_argcount);
		for (int i = 0; i < p_argcount; i++) {
			args.set(i, *p_args[i]);
		}
	}
	info->m_calls.push_back(args);
	if (info->m_return) {
		Ref<FuncRef> func = info->m_return;
		if (func.is_valid()) {
			return func->call_func(p_args, p_argcount, r_error);
		} else {
			r_error.error = Variant::CallError::CALL_OK;
			return info->m_return;
		}
	}
	return Variant();
}

bool MethodWatcher::has_method(const StringName &p_method) const {
	return m_methods.find(p_method) != NULL;
}
