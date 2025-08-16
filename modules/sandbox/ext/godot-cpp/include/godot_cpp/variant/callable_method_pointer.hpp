/**************************************************************************/
/*  callable_method_pointer.hpp                                           */
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

#pragma once

#include <godot_cpp/core/binder_common.hpp>
#include <godot_cpp/variant/variant.hpp>

namespace godot {

class CallableCustomMethodPointerBase : public CallableCustomBase {
	uint32_t *comp_ptr = nullptr;
	uint32_t comp_size;
	uint32_t h;

protected:
	void _setup(uint32_t *p_base_ptr, uint32_t p_ptr_size);

public:
	_FORCE_INLINE_ const uint32_t *get_comp_ptr() const { return comp_ptr; }
	_FORCE_INLINE_ uint32_t get_comp_size() const { return comp_size; }
	_FORCE_INLINE_ uint32_t get_hash() const { return h; }
};

namespace internal {

Callable create_callable_from_ccmp(CallableCustomMethodPointerBase *p_callable_method_pointer);

} // namespace internal

//
// No return value.
//

template <typename T, typename... P>
class CallableCustomMethodPointer : public CallableCustomMethodPointerBase {
	struct Data {
		T *instance;
		void (T::*method)(P...);
	} data;
	static_assert(sizeof(Data) % 4 == 0);

public:
	virtual ObjectID get_object() const override {
		return ObjectID(data.instance->get_instance_id());
	}

	virtual int get_argument_count(bool &r_is_valid) const override {
		r_is_valid = true;
		return sizeof...(P);
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const override {
		call_with_variant_args(data.instance, data.method, p_arguments, p_argcount, r_call_error);
	}

	CallableCustomMethodPointer(T *p_instance, void (T::*p_method)(P...)) {
		memset(&data, 0, sizeof(Data));
		data.instance = p_instance;
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <typename T, typename... P>
Callable create_custom_callable_function_pointer(T *p_instance, void (T::*p_method)(P...)) {
	typedef CallableCustomMethodPointer<T, P...> CCMP;
	CCMP *ccmp = memnew(CCMP(p_instance, p_method));
	return ::godot::internal::create_callable_from_ccmp(ccmp);
}

//
// With return value.
//

template <typename T, typename R, typename... P>
class CallableCustomMethodPointerRet : public CallableCustomMethodPointerBase {
	struct Data {
		T *instance;
		R (T::*method)(P...);
	} data;
	static_assert(sizeof(Data) % 4 == 0);

public:
	virtual ObjectID get_object() const override {
		return ObjectID(data.instance->get_instance_id());
	}

	virtual int get_argument_count(bool &r_is_valid) const override {
		r_is_valid = true;
		return sizeof...(P);
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const override {
		call_with_variant_args_ret(data.instance, data.method, p_arguments, p_argcount, r_return_value, r_call_error);
	}

	CallableCustomMethodPointerRet(T *p_instance, R (T::*p_method)(P...)) {
		memset(&data, 0, sizeof(Data));
		data.instance = p_instance;
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <typename T, typename R, typename... P>
Callable create_custom_callable_function_pointer(T *p_instance, R (T::*p_method)(P...)) {
	typedef CallableCustomMethodPointerRet<T, R, P...> CCMP; // Messes with memnew otherwise.
	CCMP *ccmp = memnew(CCMP(p_instance, p_method));
	return ::godot::internal::create_callable_from_ccmp(ccmp);
}

//
// Const with return value.
//

template <typename T, typename R, typename... P>
class CallableCustomMethodPointerRetC : public CallableCustomMethodPointerBase {
	struct Data {
		T *instance;
		R (T::*method)(P...) const;
	} data;
	static_assert(sizeof(Data) % 4 == 0);

public:
	virtual ObjectID get_object() const override {
		return ObjectID(data.instance->get_instance_id());
	}

	virtual int get_argument_count(bool &r_is_valid) const override {
		r_is_valid = true;
		return sizeof...(P);
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const override {
		call_with_variant_args_retc(data.instance, data.method, p_arguments, p_argcount, r_return_value, r_call_error);
	}

	CallableCustomMethodPointerRetC(const T *p_instance, R (T::*p_method)(P...) const) {
		memset(&data, 0, sizeof(Data));
		data.instance = const_cast<T *>(p_instance);
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <typename T, typename R, typename... P>
Callable create_custom_callable_function_pointer(const T *p_instance, R (T::*p_method)(P...) const) {
	typedef CallableCustomMethodPointerRetC<T, R, P...> CCMP; // Messes with memnew otherwise.
	CCMP *ccmp = memnew(CCMP(p_instance, p_method));
	return ::godot::internal::create_callable_from_ccmp(ccmp);
}

//
// Static method with no return value.
//

template <typename... P>
class CallableCustomStaticMethodPointer : public CallableCustomMethodPointerBase {
	struct Data {
		void (*method)(P...);
	} data;
	static_assert(sizeof(Data) % 4 == 0);

public:
	virtual ObjectID get_object() const override {
		return ObjectID();
	}

	virtual int get_argument_count(bool &r_is_valid) const override {
		r_is_valid = true;
		return sizeof...(P);
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const override {
		call_with_variant_args_static_ret(data.method, p_arguments, p_argcount, r_return_value, r_call_error);
		r_return_value = Variant();
	}

	CallableCustomStaticMethodPointer(void (*p_method)(P...)) {
		memset(&data, 0, sizeof(Data));
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <typename... P>
Callable create_custom_callable_static_function_pointer(void (*p_method)(P...)) {
	typedef CallableCustomStaticMethodPointer<P...> CCMP;
	CCMP *ccmp = memnew(CCMP(p_method));
	return ::godot::internal::create_callable_from_ccmp(ccmp);
}

//
// Static method with return value.
//

template <typename R, typename... P>
class CallableCustomStaticMethodPointerRet : public CallableCustomMethodPointerBase {
	struct Data {
		R (*method)(P...);
	} data;
	static_assert(sizeof(Data) % 4 == 0);

public:
	virtual ObjectID get_object() const override {
		return ObjectID();
	}

	virtual int get_argument_count(bool &r_is_valid) const override {
		r_is_valid = true;
		return sizeof...(P);
	}

	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, GDExtensionCallError &r_call_error) const override {
		call_with_variant_args_static_ret(data.method, p_arguments, p_argcount, r_return_value, r_call_error);
	}

	CallableCustomStaticMethodPointerRet(R (*p_method)(P...)) {
		memset(&data, 0, sizeof(Data));
		data.method = p_method;
		_setup((uint32_t *)&data, sizeof(Data));
	}
};

template <typename R, typename... P>
Callable create_custom_callable_static_function_pointer(R (*p_method)(P...)) {
	typedef CallableCustomStaticMethodPointerRet<R, P...> CCMP;
	CCMP *ccmp = memnew(CCMP(p_method));
	return ::godot::internal::create_callable_from_ccmp(ccmp);
}

//
// The API:
//

#define callable_mp(I, M) ::godot::create_custom_callable_function_pointer(I, M)
#define callable_mp_static(M) ::godot::create_custom_callable_static_function_pointer(M)

} // namespace godot
