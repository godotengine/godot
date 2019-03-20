/*************************************************************************/
/*  method_override.cpp                                                  */
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

#include "method_override.h"

void MethodOverride::bind_method(const String &p_name, const Variant &value) {
}

void MethodOverride::add_property(const String &p_name, const StringName setter, const StringName getter) {

}

Variant MethodOverride::getvar(const Variant &p_key, bool *r_valid) const {
	const PropertyMap::Element *prop = m_properties.find(p_key);
	if (prop) {
		MethodOverride proxy(*this);
		const SetGetPair &setget = prop->get();
		Variant::CallError ce;
		Variant result = proxy.call(setget.second, NULL, 0, ce);
		if (ce.error == Variant::CallError::CALL_OK) {
			if (r_valid) {
				*r_valid = true;
			}
			return result;
		}
	}
	return Variant();
}

void MethodOverride::setvar(const Variant &p_key, const Variant &p_value, bool *r_valid) {
	const PropertyMap::Element *prop = m_properties.find(p_key);
	if (prop) {
		const SetGetPair &setget = prop->get();
		const Variant *args[] = { &p_value };
		Variant::CallError ce;
		call(setget.first, args, 1, ce);
		if (ce.error == Variant::CallError::CALL_OK) {
			if (r_valid) {
				*r_valid = true;
			}
		}
	}
}

Variant MethodOverride::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	MethodMap::Element *method = m_methods.find(p_method);
	MethodInfo *info = NULL;
	if (method) {
		info = &method->get();
	} else {
		info = &m_methods.insert(p_method, MethodInfo())->get();
	}
	if (info->m_return) {
		Ref<FuncRef> func = info->m_return;
		if (func.is_valid()) {
			func->call_func(p_args, p_argcount, r_error);
		} else {
			r_error.error = Variant::CallError::CALL_OK;
			return info->m_return;
		}
	}
	return Variant();
}

void MethodOverride::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bind_method", "name"), &MethodOverride::bind_method);
	ClassDB::bind_method(D_METHOD("add_property", "name", "setter", "getter"), &MethodOverride::add_property);
}
