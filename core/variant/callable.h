/*************************************************************************/
/*  callable.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef CALLABLE_H
#define CALLABLE_H

#include "core/object/object_id.h"
#include "core/string/string_name.h"
#include "core/templates/list.h"

class Object;
class Variant;
class CallableCustom;

// This is an abstraction of things that can be called.
// It is used for signals and other cases where efficient calling of functions
// is required. It is designed for the standard case (object and method)
// but can be optimized or customized.

// Enforce 16 bytes with `alignas` to avoid arch-specific alignment issues on x86 vs armv7.
class Callable {
	alignas(8) StringName method;
	union {
		uint64_t object = 0;
		CallableCustom *custom;
	};

public:
	struct CallError {
		enum Error {
			CALL_OK,
			CALL_ERROR_INVALID_METHOD,
			CALL_ERROR_INVALID_ARGUMENT, // expected is variant type
			CALL_ERROR_TOO_MANY_ARGUMENTS, // expected is number of arguments
			CALL_ERROR_TOO_FEW_ARGUMENTS, // expected is number of arguments
			CALL_ERROR_INSTANCE_IS_NULL,
		};
		Error error = Error::CALL_OK;
		int argument = 0;
		int expected = 0;
	};

	void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, CallError &r_call_error) const;
	void call_deferred(const Variant **p_arguments, int p_argcount) const;

	void rpc(int p_id, const Variant **p_arguments, int p_argcount, CallError &r_call_error) const;

	_FORCE_INLINE_ bool is_null() const {
		return method == StringName() && object == 0;
	}
	_FORCE_INLINE_ bool is_custom() const {
		return method == StringName() && custom != nullptr;
	}
	_FORCE_INLINE_ bool is_standard() const {
		return method != StringName();
	}
	bool is_valid() const;

	Callable bind(const Variant **p_arguments, int p_argcount) const;
	Callable unbind(int p_argcount) const;

	Object *get_object() const;
	ObjectID get_object_id() const;
	StringName get_method() const;
	CallableCustom *get_custom() const;

	uint32_t hash() const;

	const Callable *get_base_comparator() const; //used for bind/unbind to do less precise comparisons (ignoring binds) in signal connect/disconnect

	bool operator==(const Callable &p_callable) const;
	bool operator!=(const Callable &p_callable) const;
	bool operator<(const Callable &p_callable) const;

	void operator=(const Callable &p_callable);

	operator String() const;

	Callable(const Object *p_object, const StringName &p_method);
	Callable(ObjectID p_object, const StringName &p_method);
	Callable(CallableCustom *p_custom);
	Callable(const Callable &p_callable);
	Callable() {}
	~Callable();
};

class CallableCustom {
	friend class Callable;
	SafeRefCount ref_count;
	bool referenced = false;

public:
	typedef bool (*CompareEqualFunc)(const CallableCustom *p_a, const CallableCustom *p_b);
	typedef bool (*CompareLessFunc)(const CallableCustom *p_a, const CallableCustom *p_b);

	//for every type that inherits, these must always be the same for this type
	virtual uint32_t hash() const = 0;
	virtual String get_as_text() const = 0;
	virtual CompareEqualFunc get_compare_equal_func() const = 0;
	virtual CompareLessFunc get_compare_less_func() const = 0;
	virtual ObjectID get_object() const = 0; //must always be able to provide an object
	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const = 0;
	virtual void rpc(int p_peer_id, const Variant **p_arguments, int p_argcount, Callable::CallError &r_call_error) const;
	virtual const Callable *get_base_comparator() const;

	CallableCustom();
	virtual ~CallableCustom() {}
};

// This is just a proxy object to object signals, its only
// allocated on demand by/for scripting languages so it can
// be put inside a Variant, but it is not
// used by the engine itself.

// Enforce 16 bytes with `alignas` to avoid arch-specific alignment issues on x86 vs armv7.
class Signal {
	alignas(8) StringName name;
	ObjectID object;

public:
	_FORCE_INLINE_ bool is_null() const {
		return object.is_null() && name == StringName();
	}
	Object *get_object() const;
	ObjectID get_object_id() const;
	StringName get_name() const;

	bool operator==(const Signal &p_signal) const;
	bool operator!=(const Signal &p_signal) const;
	bool operator<(const Signal &p_signal) const;

	operator String() const;

	Error emit(const Variant **p_arguments, int p_argcount) const;
	Error connect(const Callable &p_callable, uint32_t p_flags = 0);
	void disconnect(const Callable &p_callable);
	bool is_connected(const Callable &p_callable) const;

	Array get_connections() const;
	Signal(const Object *p_object, const StringName &p_name);
	Signal(ObjectID p_object, const StringName &p_name);
	Signal() {}
};

#endif // CALLABLE_H
