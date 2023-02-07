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

#include "core/config/project_settings.h"
#include "core/os/memory.h"
#include "core/os/os.h"
#include "core/os/semaphore.h"
#include "core/os/thread.h"
#include "core/templates/local_vector.h"
#include "core/templates/paged_allocator.h"
#include "core/templates/rid.h"
#include "core/templates/safe_refcount.h"

struct ABState {
	float sum_time[2] = { 0.0f, 0.0f };
	int sum_samples[2] = { 0, 0 };
	bool running_version_a = false;
	int swap_counter = 0;
	char enabled = 0; // 0, 'y', 'n'
};
class ABTester {
private:
	ABState &state;
	String location;
	bool default_to_version_a;
	float time;

public:
	_FORCE_INLINE_ bool version_a() {
		if (state.enabled == 'n') {
			return default_to_version_a;
		}
		return state.running_version_a;
	}

	_FORCE_INLINE_ ABTester(
			ABState &_state,
			String _name,
			String _location,
			bool _default_to_version_a) :
			state(_state),
			location(_location),
			default_to_version_a(_default_to_version_a) {
		if (state.enabled == 0) {
			String glob = GLOBAL_GET("debug/performance/include_ab_tests");
			state.enabled = _name.match(glob) ? 'y' : 'n';

			if (!glob.is_empty()) {
				print_line(glob + " matched " + _name + ": " + state.enabled);
			}
		}
		if (state.enabled == 'n') {
			return;
		}

		if (++state.swap_counter >= 5) {
			state.running_version_a = !state.running_version_a;
			state.swap_counter = 0;
		}
		time = OS::get_singleton()->get_ticks_usec();
	}
	_FORCE_INLINE_ ~ABTester() {
		if (state.enabled == 'n') {
			return;
		}

		if (state.swap_counter < 1) {
			// Discard one warm-up iteration. This is probably a placebo.
			return;
		}
		float now = OS::get_singleton()->get_ticks_usec();

		state.sum_time[state.running_version_a] += (now - time) / 1e3;
		state.sum_samples[state.running_version_a]++;

		if (state.sum_samples[0] + state.sum_samples[1] >= 100) {
			state.sum_time[0] /= state.sum_samples[0];
			state.sum_time[1] /= state.sum_samples[1];

			float a_speedup = state.sum_time[0] / state.sum_time[1];
			print_line("\n" + location);
			printf("    A=%.2f ms    B=%.2f ms    ", state.sum_time[1], state.sum_time[0]);
			printf("Version %c is %.2f%% faster\n",
					a_speedup > 1.0f ? 'A' : 'B',
					((a_speedup > 1.0f ? a_speedup : 1.0f / a_speedup) * 100) - 100);

			if ((a_speedup > 1.0f) != default_to_version_a) {
				WARN_PRINT(String() + "The default version is slower. Please swap the first parameter to AB_TEST(" + (default_to_version_a ? "false" : "true") + ", ...)");
			}

			state.sum_time[0] = state.sum_time[1] = 0.0f;
			state.sum_samples[0] = state.sum_samples[1] = 0;
		}
	}
};

#define AB_TEST(default_to_version_a, CODE_BLOCK)                                  \
	{                                                                              \
		static ABState _benchmark_ab_state;                                        \
		ABTester _benchmark_ab_tester(                                             \
				_benchmark_ab_state,                                               \
				__FILE__,                                                          \
				String(FUNCTION_STR) + "()    " + __FILE__ + ":" + itos(__LINE__), \
				default_to_version_a);                                             \
		if (_benchmark_ab_tester.version_a()) {                                    \
			constexpr bool VERSION_A = true;                                       \
			CODE_BLOCK                                                             \
		} else {                                                                   \
			constexpr bool VERSION_A = false;                                      \
			CODE_BLOCK                                                             \
		}                                                                          \
	}

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
		bool waiting = false; // Waiting for completion
		bool low_priority = false;
		BaseTemplateUserdata *template_userdata = nullptr;
		Thread *low_priority_thread = nullptr;

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
	};

	TightLocalVector<ThreadData> threads;
	SafeFlag exit_threads;

	HashMap<Thread::ID, int> thread_ids;
	HashMap<TaskID, Task *> tasks;
	HashMap<GroupID, Group *> groups;

	bool use_native_low_priority_threads = false;
	uint32_t max_low_priority_threads = 0;
	SafeNumeric<uint32_t> low_priority_threads_used;

	uint64_t last_task = 1;

	static void _thread_function(void *p_user);
	static void _native_low_priority_thread_function(void *p_user);

	void _process_task_queue();
	void _process_task(Task *task);

	void _post_task(Task *p_task, bool p_high_priority);

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
	void wait_for_task_completion(TaskID p_task_id);

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
