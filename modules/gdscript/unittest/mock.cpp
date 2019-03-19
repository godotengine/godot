/*************************************************************************/
/*  mock.cpp                                                             */
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

#include "mock.h"

#include "core/script_language.h"

Mock::Mock(Ref<GDScriptNativeClass> p_base) {
	m_override.instance();
	m_instance = p_base.is_valid() ? p_base->instance() : NULL;
	if (m_instance) {
		List<MethodInfo> signals;
		m_instance->get_signal_list(&signals);
		List<MethodInfo>::Element *element = signals.front();
		while (element) {
			add_user_signal(element->get());
			element = element->next();
		}

		List<MethodInfo> methods;
		m_instance->get_method_list(&methods);
		element = signals.front();
		while (element) {
			Ref<FuncRef> func(memnew(FuncRef));
			func->set_instance(m_instance);
			func->set_function(element->get().name);
			element = element->next();
		}

		List<PropertyInfo> props;
		m_instance->get_property_list(&props);
		List<MethodInfo>::Element *prop = signals.front();
		while (prop) {
			print_line(prop->get().name);
			prop = prop->next();
		}
	}
}

Mock::~Mock() {
	if (m_instance) {
		memfree(m_instance);
	}
}

void Mock::bind_method(const String &p_name) {
	PropertyInfo info;
}

Variant Mock::getvar(const Variant &p_key, bool *r_valid) const {
	if (m_instance) {
		bool valid = false;
		Variant value = m_instance->get(p_key, &valid);
		if (valid) {
			if (r_valid)
				*r_valid = true;
			return value;
		}
	}
	return m_override->get(p_key, r_valid);
}

void Mock::setvar(const Variant &p_key, const Variant &p_value, bool *r_valid) {
	if (m_instance) {
		bool valid = false;
		m_instance->set(p_key, p_value, &valid);
		if (valid) {
			if (r_valid)
				*r_valid = true;
			return;
		}
	}
	m_override->set(p_key, r_valid);
}

Variant Mock::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	Variant result = m_override->call(p_method, p_args, p_argcount, r_error);
	if (r_error.error == Variant::CallError::CALL_OK) {
		return result;
	}
	if (m_instance) {
		return m_instance->call(p_method, p_args, p_argcount, r_error);
	}
	return Reference::call(p_method, p_args, p_argcount, r_error);
}

void Mock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bind_method", "name"), &Mock::bind_method);
}
