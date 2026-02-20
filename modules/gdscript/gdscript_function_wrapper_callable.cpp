/**************************************************************************/
/*  gdscript_function_wrapper_callable.cpp                                */
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

#include "gdscript_function_wrapper_callable.h"

#include "gdscript.h"

#include "core/templates/hashfuncs.h"

bool GDScriptFunctionWrapperCallable::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	// Wrapper callables are only compared by reference.
	return p_a == p_b;
}

bool GDScriptFunctionWrapperCallable::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	// Wrapper callables are only compared by reference.
	return p_a < p_b;
}

bool GDScriptFunctionWrapperCallable::is_valid() const {
	return function != nullptr;
}

uint32_t GDScriptFunctionWrapperCallable::hash() const {
	return h;
}

String GDScriptFunctionWrapperCallable::get_as_text() const {
	if (function != nullptr && function->get_name() != StringName()) {
		return function->get_name().operator String();
	}
	return "<invalid GDScript function wrapper>";
}

CallableCustom::CompareEqualFunc GDScriptFunctionWrapperCallable::get_compare_equal_func() const {
	return compare_equal;
}

CallableCustom::CompareLessFunc GDScriptFunctionWrapperCallable::get_compare_less_func() const {
	return compare_less;
}

ObjectID GDScriptFunctionWrapperCallable::get_object() const {
	return object;
}

StringName GDScriptFunctionWrapperCallable::get_method() const {
	return (function == nullptr) ? "<invalid GDScript function wrapper>" : function->get_name();
}

int GDScriptFunctionWrapperCallable::get_argument_count(bool &r_is_valid) const {
	if (function == nullptr) {
		r_is_valid = false;
		return 0;
	}
	r_is_valid = true;
	return function->get_argument_count();
}

void GDScriptFunctionWrapperCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	Object *obj = ObjectDB::get_instance(object);

#ifdef DEBUG_ENABLED
	if (object.is_valid()) {
		if (obj == nullptr) {
			ERR_PRINT("Trying to call a method on a previously freed instance.");
			r_call_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		} else if (obj->get_script_instance() == nullptr || obj->get_script_instance()->get_language() != GDScriptLanguage::get_singleton()) {
			ERR_PRINT("Trying to call a GDScript function wrapper with an invalid instance.");
			r_call_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
			return;
		}
	}
#endif // DEBUG_ENABLED

	if (function == nullptr) {
		r_return_value = Variant();
		r_call_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return;
	}

	GDScriptInstance *instance = (obj == nullptr) ? nullptr : static_cast<GDScriptInstance *>(obj->get_script_instance());
	r_return_value = function->call(instance, p_arguments, p_argcount, r_call_error);
}

GDScriptFunctionWrapperCallable::GDScriptFunctionWrapperCallable(ObjectID p_object, GDScriptFunction *p_function) :
		object(p_object), function(p_function) {
	ERR_FAIL_NULL(p_function);

	h = (uint32_t)hash_murmur3_one_64((uint64_t)this);
}

Callable GDScriptFunctionWrapperCallable::make_callable(const Variant &p_self, int64_t p_function_ptr) {
	ObjectID object = (p_self.get_type() == Variant::OBJECT) ? p_self.operator ObjectID() : ObjectID();
	GDScriptFunction *function = (GDScriptFunction *)p_function_ptr;

	return Callable(memnew(GDScriptFunctionWrapperCallable(object, function)));
}
