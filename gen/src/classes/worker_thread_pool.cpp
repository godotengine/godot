/**************************************************************************/
/*  worker_thread_pool.cpp                                                */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#include <godot_cpp/classes/worker_thread_pool.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/callable.hpp>

namespace godot {

WorkerThreadPool *WorkerThreadPool::singleton = nullptr;

WorkerThreadPool *WorkerThreadPool::get_singleton() {
	if (unlikely(singleton == nullptr)) {
		GDExtensionObjectPtr singleton_obj = ::godot::gdextension_interface::global_get_singleton(WorkerThreadPool::get_class_static()._native_ptr());
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton_obj, nullptr);
#endif // DEBUG_ENABLED
		singleton = reinterpret_cast<WorkerThreadPool *>(::godot::gdextension_interface::object_get_instance_binding(singleton_obj, ::godot::gdextension_interface::token, &WorkerThreadPool::_gde_binding_callbacks));
#ifdef DEBUG_ENABLED
		ERR_FAIL_NULL_V(singleton, nullptr);
#endif // DEBUG_ENABLED
		if (likely(singleton)) {
			ClassDB::_register_engine_singleton(WorkerThreadPool::get_class_static(), singleton);
		}
	}
	return singleton;
}

WorkerThreadPool::~WorkerThreadPool() {
	if (singleton == this) {
		ClassDB::_unregister_engine_singleton(WorkerThreadPool::get_class_static());
		singleton = nullptr;
	}
}

int64_t WorkerThreadPool::add_task(const Callable &p_action, bool p_high_priority, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("add_task")._native_ptr(), 3745067146);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int8_t p_high_priority_encoded;
	PtrToArg<bool>::encode(p_high_priority, &p_high_priority_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_action, &p_high_priority_encoded, &p_description);
}

bool WorkerThreadPool::is_task_completed(int64_t p_task_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("is_task_completed")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_task_id_encoded;
	PtrToArg<int64_t>::encode(p_task_id, &p_task_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_task_id_encoded);
}

Error WorkerThreadPool::wait_for_task_completion(int64_t p_task_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("wait_for_task_completion")._native_ptr(), 844576869);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Error(0)));
	int64_t p_task_id_encoded;
	PtrToArg<int64_t>::encode(p_task_id, &p_task_id_encoded);
	return (Error)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_task_id_encoded);
}

int64_t WorkerThreadPool::get_caller_task_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("get_caller_task_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int64_t WorkerThreadPool::add_group_task(const Callable &p_action, int32_t p_elements, int32_t p_tasks_needed, bool p_high_priority, const String &p_description) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("add_group_task")._native_ptr(), 1801953219);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_elements_encoded;
	PtrToArg<int64_t>::encode(p_elements, &p_elements_encoded);
	int64_t p_tasks_needed_encoded;
	PtrToArg<int64_t>::encode(p_tasks_needed, &p_tasks_needed_encoded);
	int8_t p_high_priority_encoded;
	PtrToArg<bool>::encode(p_high_priority, &p_high_priority_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_action, &p_elements_encoded, &p_tasks_needed_encoded, &p_high_priority_encoded, &p_description);
}

bool WorkerThreadPool::is_group_task_completed(int64_t p_group_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("is_group_task_completed")._native_ptr(), 1116898809);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_group_id_encoded;
	PtrToArg<int64_t>::encode(p_group_id, &p_group_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_group_id_encoded);
}

uint32_t WorkerThreadPool::get_group_processed_element_count(int64_t p_group_id) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("get_group_processed_element_count")._native_ptr(), 923996154);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_group_id_encoded;
	PtrToArg<int64_t>::encode(p_group_id, &p_group_id_encoded);
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_group_id_encoded);
}

void WorkerThreadPool::wait_for_group_task_completion(int64_t p_group_id) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("wait_for_group_task_completion")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_group_id_encoded;
	PtrToArg<int64_t>::encode(p_group_id, &p_group_id_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_group_id_encoded);
}

int64_t WorkerThreadPool::get_caller_group_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(WorkerThreadPool::get_class_static()._native_ptr(), StringName("get_caller_group_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

} // namespace godot
