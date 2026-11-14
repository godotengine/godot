/**************************************************************************/
/*  native_class.cpp                                                      */
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

#include "native_class.h"

#include "core/object/class_db.h"
#include "core/object/gdtype.h"
#include "core/object/object.h"
#include "core/string/string_name.h"

bool GDScriptNativeClass::_get(const StringName &p_name, Variant &r_ret) const {
	bool ok;
	int64_t v = ClassDB::get_integer_constant(name, p_name, &ok);

	if (ok) {
		r_ret = v;
		return true;
	}

	const MethodBind *method = ClassDB::get_method(name, p_name);
	if (method && method->is_static()) {
		// Native static method.
		r_ret = Callable(this, p_name);
		return true;
	}

	return false;
}

void GDScriptNativeClass::_bind_methods() {
	ClassDB::bind_method(D_METHOD("new"), &GDScriptNativeClass::_new);
}

Variant GDScriptNativeClass::_new() {
	Object *o = instantiate();
	ERR_FAIL_NULL_V_MSG(o, Variant(), "Class type: '" + String(name) + "' is not instantiable.");

	RefCounted *rc = Object::cast_to<RefCounted>(o);
	if (rc) {
		return Ref<RefCounted>(rc);
	} else {
		return o;
	}
}

Object *GDScriptNativeClass::instantiate() {
	return ClassDB::instantiate_no_placeholders(name);
}

Variant GDScriptNativeClass::callp(const StringName &p_method, const Variant **p_args, int p_argcount, Callable::CallError &r_error) {
	if (p_method == SNAME("new")) {
		// Constructor.
		return Object::callp(p_method, p_args, p_argcount, r_error);
	}

	const MethodBind *method = ClassDB::get_method(name, p_method);
	if (method && method->is_static()) {
		// Native static method.
		return method->call(nullptr, p_args, p_argcount, r_error);
	}

	r_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();
}

CompanionObject::CompanionObject(const StringName &p_name) : GDScriptNativeClass(p_name), companion_gdtype(&GDScriptNativeClass::get_gdtype_static(), StringName("GDScriptNativeClass")) {}

void CompanionObject::_notification(int p_what) {
	if (p_what == NOTIFICATION_POSTINITIALIZE) {
		companion_gdtype.initialize();

		List<MethodInfo> methods;
		ClassDB::get_method_list(name, &methods);
		for (auto method : methods) {
			if (method.flags & MethodFlags::METHOD_FLAG_STATIC) {
				// Static method binds don't use the object, we can just rebind them.

				// HACK: `GDType::bind_method` only uses non const things, when `take_ownership=true`.
				MethodBind *bind = const_cast<MethodBind *>(ClassDB::get_method(name, method.name));
				companion_gdtype.bind_method(bind, false);
			}
		}

		// TODO: How to finalize `companion_gdtype`?
	}
}
