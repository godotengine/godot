/**************************************************************************/
/*  worker_thread_pool.h                                                  */
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

#ifndef WORKER_THREAD_POOL_H
#define WORKER_THREAD_POOL_H

#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/local_vector.h"
#include "core/templates/paged_allocator.h"
#include "core/templates/rid.h"
#include "core/templates/safe_refcount.h"

class WorkerThreadPool : public Object {
	GDCLASS(WorkerThreadPool, Object)
public:
	enum {
		INVALID_TASK_ID = -1
	};

	typedef int64_t TaskID;
	typedef int64_t GroupID;

private:
	struct Task;

	struct BaseTemplateUserdata {
		virtual void callback() {}
		virtual void callback_indexed(uint32_t p_index) {}
		virtual ~BaseTemplateUserdata() {}
	};

	struct Group {
		GroupID self;
		SafeNumeric<uint32_t> index;
		SafeNumeric<uint32_t> completed_index;
		uint32_t max = 0;
		Semaphore done_semaphore;
		SafeFlag completed;
		SafeNumeric<uint32_t> finished;
		uint32_t tasks_used = 0;
		TightLocalVector<Task *> low_priority_native_tasks;
	};

	struct Task {
		Callable callable;
		void (*native_func)(void *) = nullptr;
		void (*native_group_func)(void *, uint32_t) = nullptr;
		void *native_func_userdata = nullptr;
		String description;
		Semaphore done_semaphore;
		bool completed = false;
		Group *group = nullptr;
		SelfList<Task> task_elem;
		uint32_t waiting = 0;
		bool low_priority = false;
		BaseTemplateUserdata *template_userdata = nullptr;
		Thread *low_priority_thread = nullptr;
		int pool_thread_index = -1;

		void free_template_userdata();
		Task() :
				task_elem(this) {}
	};

	PagedAllocator<Task> task_allocator;
	PagedAllocator<Group> group_allocator;
	PagedAllocator<Thread> native_thread_allocator;

	SelfList<Task>::List low_priority_task_queue;
	SelfList<Task>::List task_queue;

	Mutex task_mutex;
	Semaphore task_available_semaphore;

	struct ThreadData {
		uint32_t index;
		Thread thread;
		Task *current_low_prio_task = nullptr;
		bool ready_for_scripting = false;
	};

	TightLocalVector<ThreadData> threads;
	bool exit_threads = false;

	HashMap<Thread::ID, int> thread_ids;
	HashMap<TaskID, Task *> tasks;
	HashMap<GroupID, Group *> groups;

	bool use_native_low_priority_threads = false;
	uint32_t max_low_priority_threads = 0;
	uint32_t low_priority_threads_used = 0;
	uint32_t low_priority_tasks_running = 0;
	uint32_t low_priority_tasks_awaiting_others = 0;

	uint64_t last_task = 1;

	static void _thread_function(void *p_user);
	static void _native_low_priority_thread_function(void *p_user);

	void _process_task_queue();
	void _process_task(Task *task);

	void _post_task(Task *p_task, bool p_high_priority);

	bool _try_promote_low_priority_task();
	void _prevent_low_prio_saturation_deadlock();

	static WorkerThreadPool *singleton;

	TaskID _add_task(const Callable &p_callable, void (*p_func)(void *), void *p_userdata, BaseTemplateUserdata *p_template_userdata, bool p_high_priority, const String &p_description);
	GroupID _add_group_task(const Callable &p_callable, void (*p_func)(void *, uint32_t), void *p_userdata, BaseTemplateUserdata *p_template_userdata, int p_elements, int p_tasks, bool p_high_priority, const String &p_description);

	template <class C, class M, class U>
	struct TaskUserData : public BaseTemplateUserdata {
		C *instance;
		M method;
		U userdata;
		virtual void callback() override {
			(instance->*method)(userdata);
		}
	};

	template <class C, class M, class U>
	struct GroupUserData : public BaseTemplateUserdata {
		C *instance;
		M method;
		U userdata;
		virtual void callback_indexed(uint32_t p_index) override {
			(instance->*method)(p_index, userdata);
		}
	};

protected:
	static void _bind_methods();

public:
	template <class C, class M, class U>
	TaskID add_template_task(C *p_instance, M p_method, U p_userdata, bool p_high_priority = false, const String &p_description = String()) {
		typedef TaskUserData<C, M, U> TUD;
		TUD *ud = memnew(TUD);
		ud->instance = p_instance;
		ud->method = p_method;
		ud->userdata = p_userdata;
		return _add_task(Callable(), nullptr, nullptr, ud, p_high_priority, p_description);
	}
	TaskID add_native_task(void (*p_func)(void *), void *p_userdata, bool p_high_priority = false, const String &p_description = String());
	TaskID add_task(const Callable &p_action, bool p_high_priority = false, const String &p_description = String());

	bool is_task_completed(TaskID p_task_id) const;
	Error wait_for_task_completion(TaskID p_task_id);

	template <class C, class M, class U>
	GroupID add_template_group_task(C *p_instance, M p_method, U p_userdata, int p_elements, int p_tasks = -1, bool p_high_priority = false, const String &p_description = String()) {
		typedef GroupUserData<C, M, U> GroupUD;
		GroupUD *ud = memnew(GroupUD);
		ud->instance = p_instance;
		ud->method = p_method;
		ud->userdata = p_userdata;
		return _add_group_task(Callable(), nullptr, nullptr, ud, p_elements, p_tasks, p_high_priority, p_description);
	}
	GroupID add_native_group_task(void (*p_func)(void *, uint32_t), void *p_userdata, int p_elements, int p_tasks = -1, bool p_high_priority = false, const String &p_description = String());
	GroupID add_group_task(const Callable &p_action, int p_elements, int p_tasks = -1, bool p_high_priority = false, const String &p_description = String());
	uint32_t get_group_processed_element_count(GroupID p_group) const;
	bool is_group_task_completed(GroupID p_group) const;
	void wait_for_group_task_completion(GroupID p_group);

	_FORCE_INLINE_ int get_thread_count() const { return threads.size(); }

	static WorkerThreadPool *get_singleton() { return singleton; }
	void init(int p_thread_count = -1, bool p_use_native_threads_low_priority = true, float p_low_priority_task_ratio = 0.3);
	void finish();
	WorkerThreadPool();
	~WorkerThreadPool();
};

#endif // WORKER_THREAD_POOL_H
