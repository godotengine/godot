/**************************************************************************/
/*  worker_thread_pool.hpp                                                */
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

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Callable;

class WorkerThreadPool : public Object {
	GDEXTENSION_CLASS(WorkerThreadPool, Object)

	static WorkerThreadPool *singleton;

public:
	static WorkerThreadPool *get_singleton();

	int64_t add_task(const Callable &p_action, bool p_high_priority = false, const String &p_description = String());
	bool is_task_completed(int64_t p_task_id) const;
	Error wait_for_task_completion(int64_t p_task_id);
	int64_t get_caller_task_id() const;
	int64_t add_group_task(const Callable &p_action, int32_t p_elements, int32_t p_tasks_needed = -1, bool p_high_priority = false, const String &p_description = String());
	bool is_group_task_completed(int64_t p_group_id) const;
	uint32_t get_group_processed_element_count(int64_t p_group_id) const;
	void wait_for_group_task_completion(int64_t p_group_id);
	int64_t get_caller_group_id() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
	}

	~WorkerThreadPool();

public:
	enum {
		INVALID_TASK_ID = -1
	};
	typedef int64_t TaskID;
	typedef int64_t GroupID;
	TaskID add_native_task(void (*p_func)(void *), void *p_userdata, bool p_high_priority = false, const String &p_description = String());
	GroupID add_native_group_task(void (*p_func)(void *, uint32_t), void *p_userdata, int p_elements, int p_tasks = -1, bool p_high_priority = false, const String &p_description = String());
};

} // namespace godot

