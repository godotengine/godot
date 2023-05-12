/**************************************************************************/
/*  signal_awaiter_utils.cpp                                              */
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

#include "signal_awaiter_utils.h"

#include "csharp_script.h"
#include "mono_gd/gd_mono_cache.h"

Error gd_mono_connect_signal_awaiter(Object *p_source, const StringName &p_signal, Object *p_target, GCHandleIntPtr p_awaiter_handle_ptr) {
	ERR_FAIL_NULL_V(p_source, ERR_INVALID_DATA);
	ERR_FAIL_NULL_V(p_target, ERR_INVALID_DATA);

	// TODO: Use pooling for ManagedCallable instances.
	MonoGCHandleData awaiter_handle(p_awaiter_handle_ptr, gdmono::GCHandleType::STRONG_HANDLE);
	SignalAwaiterCallable *awaiter_callable = memnew(SignalAwaiterCallable(p_target, awaiter_handle, p_signal));
	Callable callable = Callable(awaiter_callable);

	return p_source->connect(p_signal, callable, Object::CONNECT_ONE_SHOT);
}

bool SignalAwaiterCallable::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	// Only called if both instances are of type SignalAwaiterCallable. Static cast is safe.
	const SignalAwaiterCallable *a = static_cast<const SignalAwaiterCallable *>(p_a);
	const SignalAwaiterCallable *b = static_cast<const SignalAwaiterCallable *>(p_b);
	return a->awaiter_handle.handle.value == b->awaiter_handle.handle.value;
}

bool SignalAwaiterCallable::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	if (compare_equal(p_a, p_b)) {
		return false;
	}
	return p_a < p_b;
}

uint32_t SignalAwaiterCallable::hash() const {
	uint32_t hash = signal.hash();
	return hash_murmur3_one_64(target_id, hash);
}

String SignalAwaiterCallable::get_as_text() const {
	Object *base = ObjectDB::get_instance(target_id);
	if (base) {
		String class_name = base->get_class();
		Ref<Script> script = base->get_script();
		if (script.is_valid() && script->get_path().is_resource_file()) {
			class_name += "(" + script->get_path().get_file() + ")";
		}
		return class_name + "::SignalAwaiterMiddleman::" + String(signal);
	} else {
		return "null::SignalAwaiterMiddleman::" + String(signal);
	}
}

CallableCustom::CompareEqualFunc SignalAwaiterCallable::get_compare_equal_func() const {
	return compare_equal_func_ptr;
}

CallableCustom::CompareLessFunc SignalAwaiterCallable::get_compare_less_func() const {
	return compare_less_func_ptr;
}

ObjectID SignalAwaiterCallable::get_object() const {
	return target_id;
}

StringName SignalAwaiterCallable::get_signal() const {
	return signal;
}

void SignalAwaiterCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD; // Can't find anything better
	r_return_value = Variant();

#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_MSG(target_id.is_valid() && !ObjectDB::get_instance(target_id),
			"Resumed after await, but class instance is gone.");
#endif

	bool awaiter_is_null = false;
	GDMonoCache::managed_callbacks.SignalAwaiter_SignalCallback(awaiter_handle.get_intptr(), p_arguments, p_argcount, &awaiter_is_null);

	if (awaiter_is_null) {
		r_call_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return;
	}

	r_call_error.error = Callable::CallError::CALL_OK;
}

SignalAwaiterCallable::SignalAwaiterCallable(Object *p_target, MonoGCHandleData p_awaiter_handle, const StringName &p_signal) :
		target_id(p_target->get_instance_id()),
		awaiter_handle(p_awaiter_handle),
		signal(p_signal) {
}

SignalAwaiterCallable::~SignalAwaiterCallable() {
	awaiter_handle.release();
}

bool EventSignalCallable::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	const EventSignalCallable *a = static_cast<const EventSignalCallable *>(p_a);
	const EventSignalCallable *b = static_cast<const EventSignalCallable *>(p_b);

	if (a->owner != b->owner) {
		return false;
	}

	if (a->event_signal_name != b->event_signal_name) {
		return false;
	}

	return true;
}

bool EventSignalCallable::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	if (compare_equal(p_a, p_b)) {
		return false;
	}
	return p_a < p_b;
}

uint32_t EventSignalCallable::hash() const {
	uint32_t hash = event_signal_name.hash();
	return hash_murmur3_one_64(owner->get_instance_id(), hash);
}

String EventSignalCallable::get_as_text() const {
	String class_name = owner->get_class();
	Ref<Script> script = owner->get_script();
	if (script.is_valid() && script->get_path().is_resource_file()) {
		class_name += "(" + script->get_path().get_file() + ")";
	}
	return class_name + "::EventSignalMiddleman::" + String(event_signal_name);
}

CallableCustom::CompareEqualFunc EventSignalCallable::get_compare_equal_func() const {
	return compare_equal_func_ptr;
}

CallableCustom::CompareLessFunc EventSignalCallable::get_compare_less_func() const {
	return compare_less_func_ptr;
}

ObjectID EventSignalCallable::get_object() const {
	return owner->get_instance_id();
}

StringName EventSignalCallable::get_signal() const {
	return event_signal_name;
}

void EventSignalCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD; // Can't find anything better
	r_return_value = Variant();

	CSharpInstance *csharp_instance = CAST_CSHARP_INSTANCE(owner->get_script_instance());
	ERR_FAIL_NULL(csharp_instance);

	GCHandleIntPtr owner_gchandle_intptr = csharp_instance->get_gchandle_intptr();

	bool awaiter_is_null = false;
	GDMonoCache::managed_callbacks.ScriptManagerBridge_RaiseEventSignal(
			owner_gchandle_intptr, &event_signal_name,
			p_arguments, p_argcount, &awaiter_is_null);

	if (awaiter_is_null) {
		r_call_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return;
	}

	r_call_error.error = Callable::CallError::CALL_OK;
}

EventSignalCallable::EventSignalCallable(Object *p_owner, const StringName &p_event_signal_name) :
		owner(p_owner),
		event_signal_name(p_event_signal_name) {
}
