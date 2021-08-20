/*************************************************************************/
/*  managed_callable.cpp                                                 */
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

#include "managed_callable.h"

#include "csharp_script.h"
#include "mono_gd/gd_mono_cache.h"
#include "mono_gd/gd_mono_marshal.h"
#include "mono_gd/gd_mono_utils.h"

#ifdef GD_MONO_HOT_RELOAD
SelfList<ManagedCallable>::List ManagedCallable::instances;
Map<ManagedCallable *, Array> ManagedCallable::instances_pending_reload;
Mutex ManagedCallable::instances_mutex;
#endif

bool ManagedCallable::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	const ManagedCallable *a = static_cast<const ManagedCallable *>(p_a);
	const ManagedCallable *b = static_cast<const ManagedCallable *>(p_b);

	if (!a->delegate_handle || !b->delegate_handle) {
		if (!a->delegate_handle && !b->delegate_handle) {
			return true;
		}
		return false;
	}

	// Call Delegate's 'Equals'
	MonoException *exc = nullptr;
	MonoBoolean res = CACHED_METHOD_THUNK(DelegateUtils, DelegateEquals)
							  .invoke(a->delegate_handle,
									  b->delegate_handle, &exc);
	UNHANDLED_EXCEPTION(exc);
	return (bool)res;
}

bool ManagedCallable::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	if (compare_equal(p_a, p_b)) {
		return false;
	}
	return p_a < p_b;
}

uint32_t ManagedCallable::hash() const {
	return hash_djb2_one_64((uint64_t)delegate_handle);
}

String ManagedCallable::get_as_text() const {
	return "Delegate::Invoke";
}

CallableCustom::CompareEqualFunc ManagedCallable::get_compare_equal_func() const {
	return compare_equal_func_ptr;
}

CallableCustom::CompareLessFunc ManagedCallable::get_compare_less_func() const {
	return compare_less_func_ptr;
}

ObjectID ManagedCallable::get_object() const {
	// TODO: If the delegate target extends Godot.Object, use that instead!
	return CSharpLanguage::get_singleton()->get_managed_callable_middleman()->get_instance_id();
}

void ManagedCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const {
	r_call_error.error = Callable::CallError::CALL_ERROR_INVALID_METHOD; // Can't find anything better
	r_return_value = Variant();

	MonoException *exc = nullptr;
	CACHED_METHOD_THUNK(DelegateUtils, InvokeWithVariantArgs)
			.invoke(delegate_handle, p_arguments,
					p_argcount, &r_return_value, &exc);

	if (exc) {
		GDMonoUtils::set_pending_exception(exc);
	} else {
		r_call_error.error = Callable::CallError::CALL_OK;
	}
}

void ManagedCallable::release_delegate_handle() {
	if (delegate_handle) {
		MonoException *exc = nullptr;
		CACHED_METHOD_THUNK(DelegateUtils, FreeGCHandle).invoke(delegate_handle, &exc);

		if (exc) {
			GDMonoUtils::debug_print_unhandled_exception(exc);
		}

		delegate_handle = nullptr;
	}
}

// Why you do this clang-format...
/* clang-format off */
ManagedCallable::ManagedCallable(void *p_delegate_handle) : delegate_handle(p_delegate_handle) {
#ifdef GD_MONO_HOT_RELOAD
	{
		MutexLock lock(instances_mutex);
		instances.add(&self_instance);
	}
#endif
}
/* clang-format on */

ManagedCallable::~ManagedCallable() {
#ifdef GD_MONO_HOT_RELOAD
	{
		MutexLock lock(instances_mutex);
		instances.remove(&self_instance);
		instances_pending_reload.erase(this);
	}
#endif

	release_delegate_handle();
}
