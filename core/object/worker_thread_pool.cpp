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

#include "worker_thread_pool.h"

#include "core/object/script_language.h"
#include "core/os/os.h"
#include "core/os/thread_safe.h"

void WorkerThreadPool::Task::free_template_userdata() {
	ERR_FAIL_NULL(template_userdata);
	ERR_FAIL_NULL(native_func_userdata);
	BaseTemplateUserdata *btu = (BaseTemplateUserdata *)native_func_userdata;
	memdelete(btu);
}

WorkerThreadPool *WorkerThreadPool::singleton = nullptr;

void WorkerThreadPool::_process_task_queue() {
	task_mutex.lock();
	Task *task = task_queue.first()->self();
	task_queue.remove(task_queue.first());
	task_mutex.unlock();
	_process_task(task);
}

void WorkerThreadPool::_process_task(Task *p_task) {
	bool low_priority = p_task->low_priority;
	int pool_thread_index = -1;
	Task *prev_low_prio_task = nullptr; // In case this is recursively called.

	if (!use_native_low_priority_threads) {
		// Tasks must start with this unset. They are free to set-and-forget otherwise.
		set_current_thread_safe_for_nodes(false);
		pool_thread_index = thread_ids[Thread::get_caller_id()];
		ThreadData &curr_thread = threads[pool_thread_index];
		// Since the WorkerThreadPool is started before the script server,
		// its pre-created threads can't have ScriptServer::thread_enter() called on them early.
		// Therefore, we do it late at the first opportunity, so in case the task
		// about to be run uses scripting, guarantees are held.
		if (!curr_thread.ready_for_scripting && ScriptServer::are_languages_initialized()) {
			ScriptServer::thread_enter();
			curr_thread.ready_for_scripting = true;
		}
		task_mutex.lock();
		p_task->pool_thread_index = pool_thread_index;
		if (low_priority) {
			low_priority_tasks_running++;
			prev_low_prio_task = curr_thread.current_low_prio_task;
			curr_thread.current_low_prio_task = p_task;
		} else {
			curr_thread.current_low_prio_task = nullptr;
		}
		task_mutex.unlock();
	}

	if (p_task->group) {
		// Handling a group
		bool do_post = false;

		while (true) {
			uint32_t work_index = p_task->group->index.postincrement();

			if (work_index >= p_task->group->max) {
				break;
			}
			if (p_task->native_group_func) {
				p_task->native_group_func(p_task->native_func_userdata, work_index);
			} else if (p_task->template_userdata) {
				p_task->template_userdata->callback_indexed(work_index);
			} else {
				p_task->callable.call(work_index);
			}

			// This is the only way to ensure posting is done when all tasks are really complete.
			uint32_t completed_amount = p_task->group->completed_index.increment();

			if (completed_amount == p_task->group->max) {
				do_post = true;
			}
		}

		if (do_post && p_task->template_userdata) {
			memdelete(p_task->template_userdata); // This is no longer needed at this point, so get rid of it.
		}

		if (low_priority && use_native_low_priority_threads) {
			p_task->completed = true;
			p_task->done_semaphore.post();
			if (do_post) {
				p_task->group->completed.set_to(true);
			}
		} else {
			if (do_post) {
				p_task->group->done_semaphore.post();
				p_task->group->completed.set_to(true);
			}
			uint32_t max_users = p_task->group->tasks_used + 1; // Add 1 because the thread waiting for it is also user. Read before to avoid another thread freeing task after increment.
			uint32_t finished_users = p_task->group->finished.increment();

			if (finished_users == max_users) {
				// Get rid of the group, because nobody else is using it.
				task_mutex.lock();
				group_allocator.free(p_task->group);
				task_mutex.unlock();
			}

			// For groups, tasks get rid of themselves.

			task_mutex.lock();
			task_allocator.free(p_task);
			task_mutex.unlock();
		}
	} else {
		if (p_task->native_func) {
			p_task->native_func(p_task->native_func_userdata);
		} else if (p_task->template_userdata) {
			p_task->template_userdata->callback();
			memdelete(p_task->template_userdata);
		} else {
			p_task->callable.call();
		}

		task_mutex.lock();
		p_task->completed = true;
		for (uint8_t i = 0; i < p_task->waiting; i++) {
			p_task->done_semaphore.post();
		}
		if (!use_native_low_priority_threads) {
			p_task->pool_thread_index = -1;
		}
		task_mutex.unlock(); // Keep mutex down to here since on unlock the task may be freed.
	}

	// Task may have been freed by now (all callers notified).
	p_task = nullptr;

	if (!use_native_low_priority_threads) {
		bool post = false;
		task_mutex.lock();
		ThreadData &curr_thread = threads[pool_thread_index];
		curr_thread.current_low_prio_task = prev_low_prio_task;
		if (low_priority) {
			low_priority_threads_used--;
			low_priority_tasks_running--;
			// A low prioriry task was freed, so see if we can move a pending one to the high priority queue.
			if (_try_promote_low_priority_task()) {
				post = true;
			}

			if (low_priority_tasks_awaiting_others == low_priority_tasks_running) {
				_prevent_low_prio_saturation_deadlock();
			}
		}
		task_mutex.unlock();
		if (post) {
			task_available_semaphore.post();
		}
	}
}

void WorkerThreadPool::_thread_function(void *p_user) {
	while (true) {
		singleton->task_available_semaphore.wait();
		if (singleton->exit_threads) {
			break;
		}
		singleton->_process_task_queue();
	}
}

void WorkerThreadPool::_native_low_priority_thread_function(void *p_user) {
	Task *task = (Task *)p_user;
	singleton->_process_task(task);
}

void WorkerThreadPool::_post_task(Task *p_task, bool p_high_priority) {
	// Fall back to processing on the calling thread if there are no worker threads.
	// Separated into its own variable to make it easier to extend this logic
	// in custom builds.
	bool process_on_calling_thread = threads.size() == 0;
	if (process_on_calling_thread) {
		_process_task(p_task);
		return;
	}

	task_mutex.lock();
	p_task->low_priority = !p_high_priority;
	if (!p_high_priority && use_native_low_priority_threads) {
		p_task->low_priority_thread = native_thread_allocator.alloc();
		task_mutex.unlock();

		if (p_task->group) {
			p_task->group->low_priority_native_tasks.push_back(p_task);
		}
		p_task->low_priority_thread->start(_native_low_priority_thread_function, p_task); // Pask task directly to thread.
	} else if (p_high_priority || low_priority_threads_used < max_low_priority_threads) {
		task_queue.add_last(&p_task->task_elem);
		if (!p_high_priority) {
			low_priority_threads_used++;
		}
		task_mutex.unlock();
		task_available_semaphore.post();
	} else {
		// Too many threads using low priority, must go to queue.
		low_priority_task_queue.add_last(&p_task->task_elem);
		task_mutex.unlock();
	}
}

bool WorkerThreadPool::_try_promote_low_priority_task() {
	if (low_priority_task_queue.first()) {
		Task *low_prio_task = low_priority_task_queue.first()->self();
		low_priority_task_queue.remove(low_priority_task_queue.first());
		task_queue.add_last(&low_prio_task->task_elem);
		low_priority_threads_used++;
		return true;
	} else {
		return false;
	}
}

void WorkerThreadPool::_prevent_low_prio_saturation_deadlock() {
	if (low_priority_tasks_awaiting_others == low_priority_tasks_running) {
#ifdef DEV_ENABLED
		print_verbose("WorkerThreadPool: Low-prio slots saturated with tasks all waiting for other low-prio tasks. Attempting to avoid deadlock by scheduling one extra task.");
#endif
		// In order not to create dependency cycles, we can only schedule the next one.
		// We'll keep doing the same until the deadlock is broken,
		SelfList<Task> *to_promote = low_priority_task_queue.first();
		if (to_promote) {
			low_priority_task_queue.remove(to_promote);
			task_queue.add_last(to_promote);
			low_priority_threads_used++;
			task_available_semaphore.post();
		}
	}
}

WorkerThreadPool::TaskID WorkerThreadPool::add_native_task(void (*p_func)(void *), void *p_userdata, bool p_high_priority, const String &p_description) {
	return _add_task(Callable(), p_func, p_userdata, nullptr, p_high_priority, p_description);
}

WorkerThreadPool::TaskID WorkerThreadPool::_add_task(const Callable &p_callable, void (*p_func)(void *), void *p_userdata, BaseTemplateUserdata *p_template_userdata, bool p_high_priority, const String &p_description) {
	task_mutex.lock();
	// Get a free task
	Task *task = task_allocator.alloc();
	TaskID id = last_task++;
	task->callable = p_callable;
	task->native_func = p_func;
	task->native_func_userdata = p_userdata;
	task->description = p_description;
	task->template_userdata = p_template_userdata;
	tasks.insert(id, task);
	task_mutex.unlock();

	_post_task(task, p_high_priority);

	return id;
}

WorkerThreadPool::TaskID WorkerThreadPool::add_task(const Callable &p_action, bool p_high_priority, const String &p_description) {
	return _add_task(p_action, nullptr, nullptr, nullptr, p_high_priority, p_description);
}

bool WorkerThreadPool::is_task_completed(TaskID p_task_id) const {
	task_mutex.lock();
	const Task *const *taskp = tasks.getptr(p_task_id);
	if (!taskp) {
		task_mutex.unlock();
		ERR_FAIL_V_MSG(false, "Invalid Task ID"); // Invalid task
	}

	bool completed = (*taskp)->completed;
	task_mutex.unlock();

	return completed;
}

Error WorkerThreadPool::wait_for_task_completion(TaskID p_task_id) {
	task_mutex.lock();
	Task **taskp = tasks.getptr(p_task_id);
	if (!taskp) {
		task_mutex.unlock();
		ERR_FAIL_V_MSG(ERR_INVALID_PARAMETER, "Invalid Task ID"); // Invalid task
	}
	Task *task = *taskp;

	if (!task->completed) {
		if (!use_native_low_priority_threads && task->pool_thread_index != -1) { // Otherwise, it's not running yet.
			int caller_pool_th_index = thread_ids.has(Thread::get_caller_id()) ? thread_ids[Thread::get_caller_id()] : -1;
			if (caller_pool_th_index == task->pool_thread_index) {
				// Deadlock prevention.
				// Waiting for a task run on this same thread? That means the task to be awaited started waiting as well
				// and another task was run to make use of the thread in the meantime, with enough bad luck as to
				// the need to wait for the original task arose in turn.
				// In other words, the task we want to wait for is buried in the stack.
				// Let's report the caller about the issue to it handles as it sees fit.
				task_mutex.unlock();
				return ERR_BUSY;
			}
		}

		task->waiting++;

		bool is_low_prio_waiting_for_another = false;
		if (!use_native_low_priority_threads) {
			// Deadlock prevention:
			// If all low-prio tasks are waiting for other low-prio tasks and there are no more free low-prio slots,
			// we have a no progressable situation. We can apply a workaround, consisting in promoting an awaited queued
			// low-prio task to the schedule queue so it can run and break the "impasse".
			// NOTE: A similar reasoning could be made about high priority tasks, but there are usually much more
			// than low-prio. Therefore, a deadlock there would only happen when dealing with a very complex task graph
			// or when there are too few worker threads (limited platforms or exotic settings). If that turns out to be
			// an issue in the real world, a further fix can be applied against that.
			if (task->low_priority) {
				bool awaiter_is_a_low_prio_task = thread_ids.has(Thread::get_caller_id()) && threads[thread_ids[Thread::get_caller_id()]].current_low_prio_task;
				if (awaiter_is_a_low_prio_task) {
					is_low_prio_waiting_for_another = true;
					low_priority_tasks_awaiting_others++;
					if (low_priority_tasks_awaiting_others == low_priority_tasks_running) {
						_prevent_low_prio_saturation_deadlock();
					}
				}
			}
		}

		task_mutex.unlock();

		if (use_native_low_priority_threads && task->low_priority) {
			task->done_semaphore.wait();
		} else {
			bool current_is_pool_thread = thread_ids.has(Thread::get_caller_id());
			if (current_is_pool_thread) {
				// We are an actual process thread, we must not be blocked so continue processing stuff if available.
				bool must_exit = false;
				while (true) {
					if (task->done_semaphore.try_wait()) {
						// If done, exit
						break;
					}
					if (!must_exit) {
						if (task_available_semaphore.try_wait()) {
							if (exit_threads) {
								must_exit = true;
							} else {
								// Solve tasks while they are around.
								bool safe_for_nodes_backup = is_current_thread_safe_for_nodes();
								_process_task_queue();
								set_current_thread_safe_for_nodes(safe_for_nodes_backup);
								continue;
							}
						} else if (!use_native_low_priority_threads && task->low_priority) {
							// A low prioriry task started waiting, so see if we can move a pending one to the high priority queue.
							task_mutex.lock();
							bool post = _try_promote_low_priority_task();
							task_mutex.unlock();
							if (post) {
								task_available_semaphore.post();
							}
						}
					}
					OS::get_singleton()->delay_usec(1); // Microsleep, this could be converted to waiting for multiple objects in supported platforms for a bit more performance.
				}
			} else {
				task->done_semaphore.wait();
			}
		}

		task_mutex.lock();
		if (is_low_prio_waiting_for_another) {
			low_priority_tasks_awaiting_others--;
		}

		task->waiting--;
	}

	if (task->waiting == 0) {
		if (use_native_low_priority_threads && task->low_priority) {
			task->low_priority_thread->wait_to_finish();
			native_thread_allocator.free(task->low_priority_thread);
		}
		tasks.erase(p_task_id);
		task_allocator.free(task);
	}

	task_mutex.unlock();
	return OK;
}

WorkerThreadPool::GroupID WorkerThreadPool::_add_group_task(const Callable &p_callable, void (*p_func)(void *, uint32_t), void *p_userdata, BaseTemplateUserdata *p_template_userdata, int p_elements, int p_tasks, bool p_high_priority, const String &p_description) {
	ERR_FAIL_COND_V(p_elements < 0, INVALID_TASK_ID);
	if (p_tasks < 0) {
		p_tasks = MAX(1u, threads.size());
	}

	task_mutex.lock();
	Group *group = group_allocator.alloc();
	GroupID id = last_task++;
	group->max = p_elements;
	group->self = id;

	Task **tasks_posted = nullptr;
	if (p_elements == 0) {
		// Should really not call it with zero Elements, but at least it should work.
		group->completed.set_to(true);
		group->done_semaphore.post();
		group->tasks_used = 0;
		p_tasks = 0;
		if (p_template_userdata) {
			memdelete(p_template_userdata);
		}

	} else {
		group->tasks_used = p_tasks;
		tasks_posted = (Task **)alloca(sizeof(Task *) * p_tasks);
		for (int i = 0; i < p_tasks; i++) {
			Task *task = task_allocator.alloc();
			task->native_group_func = p_func;
			task->native_func_userdata = p_userdata;
			task->description = p_description;
			task->group = group;
			task->callable = p_callable;
			task->template_userdata = p_template_userdata;
			tasks_posted[i] = task;
			// No task ID is used.
		}
	}

	groups[id] = group;
	task_mutex.unlock();

	for (int i = 0; i < p_tasks; i++) {
		_post_task(tasks_posted[i], p_high_priority);
	}

	return id;
}

WorkerThreadPool::GroupID WorkerThreadPool::add_native_group_task(void (*p_func)(void *, uint32_t), void *p_userdata, int p_elements, int p_tasks, bool p_high_priority, const String &p_description) {
	return _add_group_task(Callable(), p_func, p_userdata, nullptr, p_elements, p_tasks, p_high_priority, p_description);
}

WorkerThreadPool::GroupID WorkerThreadPool::add_group_task(const Callable &p_action, int p_elements, int p_tasks, bool p_high_priority, const String &p_description) {
	return _add_group_task(p_action, nullptr, nullptr, nullptr, p_elements, p_tasks, p_high_priority, p_description);
}

uint32_t WorkerThreadPool::get_group_processed_element_count(GroupID p_group) const {
	task_mutex.lock();
	const Group *const *groupp = groups.getptr(p_group);
	if (!groupp) {
		task_mutex.unlock();
		ERR_FAIL_V_MSG(0, "Invalid Group ID");
	}
	uint32_t elements = (*groupp)->completed_index.get();
	task_mutex.unlock();
	return elements;
}
bool WorkerThreadPool::is_group_task_completed(GroupID p_group) const {
	task_mutex.lock();
	const Group *const *groupp = groups.getptr(p_group);
	if (!groupp) {
		task_mutex.unlock();
		ERR_FAIL_V_MSG(false, "Invalid Group ID");
	}
	bool completed = (*groupp)->completed.is_set();
	task_mutex.unlock();
	return completed;
}

void WorkerThreadPool::wait_for_group_task_completion(GroupID p_group) {
	task_mutex.lock();
	Group **groupp = groups.getptr(p_group);
	task_mutex.unlock();
	if (!groupp) {
		ERR_FAIL_MSG("Invalid Group ID");
	}
	Group *group = *groupp;

	if (group->low_priority_native_tasks.size() > 0) {
		for (Task *task : group->low_priority_native_tasks) {
			task->low_priority_thread->wait_to_finish();
			task_mutex.lock();
			native_thread_allocator.free(task->low_priority_thread);
			task_allocator.free(task);
			task_mutex.unlock();
		}

		task_mutex.lock();
		group_allocator.free(group);
		task_mutex.unlock();
	} else {
		group->done_semaphore.wait();

		uint32_t max_users = group->tasks_used + 1; // Add 1 because the thread waiting for it is also user. Read before to avoid another thread freeing task after increment.
		uint32_t finished_users = group->finished.increment(); // fetch happens before inc, so increment later.

		if (finished_users == max_users) {
			// All tasks using this group are gone (finished before the group), so clear the group too.
			task_mutex.lock();
			group_allocator.free(group);
			task_mutex.unlock();
		}
	}

	task_mutex.lock(); // This mutex is needed when Physics 2D and/or 3D is selected to run on a separate thread.
	groups.erase(p_group);
	task_mutex.unlock();
}

void WorkerThreadPool::init(int p_thread_count, bool p_use_native_threads_low_priority, float p_low_priority_task_ratio) {
	ERR_FAIL_COND(threads.size() > 0);
	if (p_thread_count < 0) {
		p_thread_count = OS::get_singleton()->get_default_thread_pool_size();
	}

	if (p_use_native_threads_low_priority) {
		max_low_priority_threads = 0;
	} else {
		max_low_priority_threads = CLAMP(p_thread_count * p_low_priority_task_ratio, 1, p_thread_count - 1);
	}

	use_native_low_priority_threads = p_use_native_threads_low_priority;

	threads.resize(p_thread_count);

	for (uint32_t i = 0; i < threads.size(); i++) {
		threads[i].index = i;
		threads[i].thread.start(&WorkerThreadPool::_thread_function, &threads[i]);
		thread_ids.insert(threads[i].thread.get_id(), i);
	}
}

void WorkerThreadPool::finish() {
	if (threads.size() == 0) {
		return;
	}

	task_mutex.lock();
	SelfList<Task> *E = low_priority_task_queue.first();
	while (E) {
		print_error("Task waiting was never re-claimed: " + E->self()->description);
		E = E->next();
	}
	task_mutex.unlock();

	exit_threads = true;

	for (uint32_t i = 0; i < threads.size(); i++) {
		task_available_semaphore.post();
	}

	for (ThreadData &data : threads) {
		data.thread.wait_to_finish();
	}

	threads.clear();
}

void WorkerThreadPool::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_task", "action", "high_priority", "description"), &WorkerThreadPool::add_task, DEFVAL(false), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("is_task_completed", "task_id"), &WorkerThreadPool::is_task_completed);
	ClassDB::bind_method(D_METHOD("wait_for_task_completion", "task_id"), &WorkerThreadPool::wait_for_task_completion);

	ClassDB::bind_method(D_METHOD("add_group_task", "action", "elements", "tasks_needed", "high_priority", "description"), &WorkerThreadPool::add_group_task, DEFVAL(-1), DEFVAL(false), DEFVAL(String()));
	ClassDB::bind_method(D_METHOD("is_group_task_completed", "group_id"), &WorkerThreadPool::is_group_task_completed);
	ClassDB::bind_method(D_METHOD("get_group_processed_element_count", "group_id"), &WorkerThreadPool::get_group_processed_element_count);
	ClassDB::bind_method(D_METHOD("wait_for_group_task_completion", "group_id"), &WorkerThreadPool::wait_for_group_task_completion);
}

WorkerThreadPool::WorkerThreadPool() {
	singleton = this;
}

WorkerThreadPool::~WorkerThreadPool() {
	finish();
}
