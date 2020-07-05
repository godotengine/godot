/*************************************************************************/
/*  signal_awaiter_utils.cpp                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "signal_awaiter_utils.h"

#include "csharp_script.h"
#include "mono_gd/gd_mono_cache.h"
#include "mono_gd/gd_mono_class.h"
#include "mono_gd/gd_mono_marshal.h"
#include "mono_gd/gd_mono_utils.h"

Error gd_mono_connect_signal_awaiter(Object *p_source, const StringName &p_signal, Object *p_target, MonoObject *p_awaiter) {
	ERR_FAIL_NULL_V(p_source, ERR_INVALID_DATA);
	ERR_FAIL_NULL_V(p_target, ERR_INVALID_DATA);

	// TODO: Use pooling for ManagedCallable instances.
	SignalAwaiterCallable *awaiter_callable = memnew(SignalAwaiterCallable(p_target, p_awaiter, p_signal));
	Callable callable = Callable(awaiter_callable);

	return p_source->connect(p_signal, callable, Vector<Variant>(), Object::CONNECT_ONESHOT);
}

bool SignalAwaiterCallable::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	const SignalAwaiterCallable *a = static_cast<const SignalAwaiterCallable *>(p_a);
	const SignalAwaiterCallable *b = static_cast<const SignalAwaiterCallable *>(p_b);

	if (a->target_id != b->target_id) {
		return false;
	}

	if (a->signal != b->signal) {
		return false;
	}

	return true;
}

bool SignalAwaiterCallable::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	if (compare_equal(p_a, p_b)) {
		return false;
	}
	return p_a < p_b;
}

uint32_t SignalAwaiterCallable::hash() const {
	uint32_t hash = signal.hash();
	return hash_djb2_one_64(target_id, hash);
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

void SignalAwaiterCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD; // Can't find anything better
	r_return_value = Variant();

#ifdef DEBUG_ENABLED
	ERR_FAIL_COND_MSG(target_id.is_valid() && !ObjectDB::get_instance(target_id),
			"Resumed after await, but class instance is gone.");
#endif

	MonoArray *signal_args = mono_array_new(mono_domain_get(), CACHED_CLASS_RAW(MonoObject), p_argcount);

	for (int i = 0; i < p_argcount; i++) {
		MonoObject *boxed = GDMonoMarshal::variant_to_mono_object(*p_arguments[i]);
		mono_array_setref(signal_args, i, boxed);
	}

	MonoObject *awaiter = awaiter_handle.get_target();

	if (!awaiter) {
		r_call_error.error = Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return;
	}

	MonoException *exc = nullptr;
	CACHED_METHOD_THUNK(SignalAwaiter, SignalCallback).invoke(awaiter, signal_args, &exc);

	if (exc) {
		GDMonoUtils::set_pending_exception(exc);
		ERR_FAIL();
	} else {
		r_call_error.error = Callable::CallError::CALL_OK;
	}
}

SignalAwaiterCallable::SignalAwaiterCallable(Object *p_target, MonoObject *p_awaiter, const StringName &p_signal) :
		target_id(p_target->get_instance_id()),
		awaiter_handle(MonoGCHandleData::new_strong_handle(p_awaiter)),
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

	if (a->event_signal != b->event_signal) {
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
	uint32_t hash = event_signal->field->get_name().hash();
	return hash_djb2_one_64(owner->get_instance_id(), hash);
}

String EventSignalCallable::get_as_text() const {
	String class_name = owner->get_class();
	Ref<Script> script = owner->get_script();
	if (script.is_valid() && script->get_path().is_resource_file()) {
		class_name += "(" + script->get_path().get_file() + ")";
	}
	StringName signal = event_signal->field->get_name();
	return class_name + "::EventSignalMiddleman::" + String(signal);
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
	return event_signal->field->get_name();
}

void EventSignalCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD; // Can't find anything better
	r_return_value = Variant();

	ERR_FAIL_COND(p_argcount < event_signal->invoke_method->get_parameters_count());

	CSharpInstance *csharp_instance = CAST_CSHARP_INSTANCE(owner->get_script_instance());
	ERR_FAIL_NULL(csharp_instance);

	MonoObject *owner_managed = csharp_instance->get_mono_object();
	ERR_FAIL_NULL(owner_managed);

	MonoObject *delegate_field_value = event_signal->field->get_value(owner_managed);
	if (!delegate_field_value) {
		r_call_error.error = Callable::CallError::CALL_OK;
		return;
	}

	MonoException *exc = nullptr;
	event_signal->invoke_method->invoke(delegate_field_value, p_arguments, &exc);

	if (exc) {
		GDMonoUtils::set_pending_exception(exc);
		ERR_FAIL();
	} else {
		r_call_error.error = Callable::CallError::CALL_OK;
	}
}

EventSignalCallable::EventSignalCallable(Object *p_owner, const CSharpScript::EventSignal *p_event_signal) :
		owner(p_owner),
		event_signal(p_event_signal) {
}
