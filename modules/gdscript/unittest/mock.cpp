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

bool has_signal(const Object *p_object, const StringName p_signal) {
	List<MethodInfo> signals;
	p_object->get_signal_list(&signals);
	List<MethodInfo>::Element *signal = signals.front();
	while (signal) {
		if (signal->get().name == p_signal) {
			return true;
		}
		signal = signal->next();
	}
	return false;
}

Mock::Mock(Ref<GDScriptNativeClass> p_base) {
	m_override.instance();
	m_instance = p_base.is_valid() ? p_base->instance() : NULL;
	if (m_instance) {
		List<MethodInfo> signals;
		m_instance->get_signal_list(&signals);
		List<MethodInfo>::Element *signal = signals.front();
		while (signal) {
			const MethodInfo &info = signal->get();
			if (!has_signal(this, info.name)) {
				add_user_signal(info);
			}
			Vector<Variant> binds;
			binds.resize(1);
			if (!is_connected(info.name, this, "_handle_signal")) {
				binds.set(0, info.name);
				m_instance->connect(info.name, this, "_handle_signal", binds);
				signal = signal->next();
			}
		}

		List<MethodInfo> methods;
		m_instance->get_method_list(&methods);
		List<MethodInfo>::Element *method = methods.front();
		while (method) {
			Ref<FuncRef> func(memnew(FuncRef));
			func->set_instance(m_instance);
			func->set_function(method->get().name);
			bind_method(method->get().name, func);
			method = method->next();
		}

		List<PropertyInfo> props;
		m_instance->get_property_list(&props);
		List<PropertyInfo>::Element *prop = props.front();
		while (prop) {
			PropertyInfo &info = prop->get();
			const String &name = info.name;
			StringName getter = ClassDB::get_property_getter(m_instance->get_class_name(), name);
			if (getter) {
			}
			StringName setter = ClassDB::get_property_setter(m_instance->get_class_name(), name);
			if (setter) {
			}
			prop = prop->next();
		}

		Node *node = cast_to<Node>(m_instance);
		if (node) {
			add_child(node);
		}
	}
}

Mock::~Mock() {
	if (m_instance) {
		memfree(m_instance);
	}
}

void Mock::bind_method(const String &p_name, const Variant &value) {
}

void Mock::add_property(const String &p_name, const StringName setter, const StringName getter) {
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
	return Node::call(p_method, p_args, p_argcount, r_error);
}

Variant Mock::_handle_signal(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	Error error = emit_signal(*p_args[p_argcount - 1], p_args, p_argcount - 1);
	if (error == Error::OK) {
		r_error.error = Variant::CallError::CALL_OK;
	}
	return Variant();
}

void Mock::_bind_methods() {
	ClassDB::bind_method(D_METHOD("bind_method", "name"), &Mock::bind_method);
	ClassDB::bind_method(D_METHOD("add_property", "name", "setter", "getter"), &Mock::add_property);

	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "_handle_signal", &Mock::_handle_signal);
}
