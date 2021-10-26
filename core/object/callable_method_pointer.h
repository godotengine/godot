/*************************************************************************/
/*  callable_method_pointer.h                                            */
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

#ifndef CALLABLE_METHOD_POINTER_H
#define CALLABLE_METHOD_POINTER_H

#include "core/object/object.h"
#include "core/templates/hashfuncs.h"
#include "core/templates/simple_type.h"
#include "core/variant/binder_common.h"
#include "core/variant/callable.h"

class CallableCustomMethodPointerBase : public CallableCustom {
	uint32_t *comp_ptr;
	uint32_t comp_size;
	uint32_t h;
#ifdef DEBUG_METHODS_ENABLED
	const char *text = "";
#endif
	static bool compare_equal(const CallableCustom *p_a, const CallableCustom *p_b);
	static bool compare_less(const CallableCustom *p_a, const CallableCustom *p_b);

protected:
	void _setup(uint32_t *p_base_ptr, uint32_t p_ptr_size);

public:
#ifdef DEBUG_METHODS_ENABLED
	void set_text(const char *p_text) {
		text = p_text;
	}
	virtual String get_as_text() const {
		return text;
	}
#else
	virtual String get_as_text() const {
		return String();
	}
#endif
	virtual CompareEqualFunc get_compare_equal_func() const;
	virtual CompareLessFunc get_compare_less_func() const;

	virtual uint32_t hash() const;
};

template <class T, class... P>
class CallableCustomMethodPointer : public CallableCustomMethodPointerBase {
	struct Data {
		T *instance;
#ifdef DEBUG_ENABLED
		uint64_t object_id;
#endif
		void (T::*method)(P...);
	} data;

public:
	virtual ObjectID get_object() const {
#ifdef DEBUG_ENABLED
		if (ObjectDB::get_instance(ObjectID(data.object_id)) == nullptr) {
			return ObjectID();
		}
#endif
		return data.instance->get_instance_id();
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_MSG(ObjectDB::get_instance(ObjectID(data.object_id)) == nullptr, "Invalid Object id '" + uitos(data.object_id) + "', can't call method.");
#endif
		call_with_variant_args(data.instance, data.method, p_arguments, p_argcount, r_call_error);
	}

	CallableCustomMethodPointer(T *p_instance, void (T::*p_method)(P...)) {
		memset(&data, 0, sizeof(Data)); // Clear beforehand, may have padding bytes.
		data.instance = p_instance;
#ifdef DEBUG_ENABLED
		data.object_id = p_instance->get_instance_id();
#endif
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <class T, class... P>
Callable create_custom_callable_function_pointer(T *p_instance,
#ifdef DEBUG_METHODS_ENABLED
		const char *p_func_text,
#endif
		void (T::*p_method)(P...)) {
	typedef CallableCustomMethodPointer<T, P...> CCMP; // Messes with memnew otherwise.
	CCMP *ccmp = memnew(CCMP(p_instance, p_method));
#ifdef DEBUG_METHODS_ENABLED
	ccmp->set_text(p_func_text + 1); // Try to get rid of the ampersand.
#endif
	return Callable(ccmp);
}

// VERSION WITH RETURN

template <class T, class R, class... P>
class CallableCustomMethodPointerRet : public CallableCustomMethodPointerBase {
	struct Data {
		T *instance;
#ifdef DEBUG_ENABLED
		uint64_t object_id;
#endif
		R(T::*method)
		(P...);
	} data;

public:
	virtual ObjectID get_object() const {
#ifdef DEBUG_ENABLED
		if (ObjectDB::get_instance(ObjectID(data.object_id)) == nullptr) {
			return ObjectID();
		}
#endif
		return data.instance->get_instance_id();
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_MSG(ObjectDB::get_instance(ObjectID(data.object_id)) == nullptr, "Invalid Object id '" + uitos(data.object_id) + "', can't call method.");
#endif
		call_with_variant_args_ret(data.instance, data.method, p_arguments, p_argcount, r_return_value, r_call_error);
	}

	CallableCustomMethodPointerRet(T *p_instance, R (T::*p_method)(P...)) {
		memset(&data, 0, sizeof(Data)); // Clear beforehand, may have padding bytes.
		data.instance = p_instance;
#ifdef DEBUG_ENABLED
		data.object_id = p_instance->get_instance_id();
#endif
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <class T, class R, class... P>
Callable create_custom_callable_function_pointer(T *p_instance,
#ifdef DEBUG_METHODS_ENABLED
		const char *p_func_text,
#endif
		R (T::*p_method)(P...)) {
	typedef CallableCustomMethodPointerRet<T, R, P...> CCMP; // Messes with memnew otherwise.
	CCMP *ccmp = memnew(CCMP(p_instance, p_method));
#ifdef DEBUG_METHODS_ENABLED
	ccmp->set_text(p_func_text + 1); // Try to get rid of the ampersand.
#endif
	return Callable(ccmp);
}

// CONST VERSION WITH RETURN

template <class T, class R, class... P>
class CallableCustomMethodPointerRetC : public CallableCustomMethodPointerBase {
	struct Data {
		T *instance;
#ifdef DEBUG_ENABLED
		uint64_t object_id;
#endif
		R(T::*method)
		(P...) const;
	} data;

public:
	virtual ObjectID get_object() const {
#ifdef DEBUG_ENABLED
		if (ObjectDB::get_instance(ObjectID(data.object_id)) == nullptr) {
			return ObjectID();
		}
#endif
		return data.instance->get_instance_id();
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
#ifdef DEBUG_ENABLED
		ERR_FAIL_COND_MSG(ObjectDB::get_instance(ObjectID(data.object_id)) == nullptr, "Invalid Object id '" + uitos(data.object_id) + "', can't call method.");
#endif
		call_with_variant_args_retc(data.instance, data.method, p_arguments, p_argcount, r_return_value, r_call_error);
	}

	CallableCustomMethodPointerRetC(T *p_instance, R (T::*p_method)(P...) const) {
		memset(&data, 0, sizeof(Data)); // Clear beforehand, may have padding bytes.
		data.instance = p_instance;
#ifdef DEBUG_ENABLED
		data.object_id = p_instance->get_instance_id();
#endif
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <class T, class R, class... P>
Callable create_custom_callable_function_pointer(T *p_instance,
#ifdef DEBUG_METHODS_ENABLED
		const char *p_func_text,
#endif
		R (T::*p_method)(P...) const) {
	typedef CallableCustomMethodPointerRetC<T, R, P...> CCMP; // Messes with memnew otherwise.
	CCMP *ccmp = memnew(CCMP(p_instance, p_method));
#ifdef DEBUG_METHODS_ENABLED
	ccmp->set_text(p_func_text + 1); // Try to get rid of the ampersand.
#endif
	return Callable(ccmp);
}

#ifdef DEBUG_METHODS_ENABLED
#define callable_mp(I, M) create_custom_callable_function_pointer(I, #M, M)
#else
#define callable_mp(I, M) create_custom_callable_function_pointer(I, M)
#endif

#endif // CALLABLE_METHOD_POINTER_H
