/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_THREADPOOL_H
#define RTC_IMPL_THREADPOOL_H

#include "common.hpp"
#include "init.hpp"
#include "internals.hpp"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace rtc::impl {

template <class F, class... Args>
using invoke_future_t = std::future<std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>>;

class ThreadPool final {
public:
	using clock = std::chrono::steady_clock;

	static ThreadPool &Instance();

	ThreadPool(const ThreadPool &) = delete;
	ThreadPool &operator=(const ThreadPool &) = delete;
	ThreadPool(ThreadPool &&) = delete;
	ThreadPool &operator=(ThreadPool &&) = delete;

	int count() const;
	void spawn(int count = 1);
	void join();
	void clear();
	void run();
	bool runOne();

	template <class F, class... Args>
	auto enqueue(F &&f, Args &&...args) noexcept -> invoke_future_t<F, Args...>;

	template <class F, class... Args>
	auto schedule(clock::duration delay, F &&f, Args &&...args) noexcept
	    -> invoke_future_t<F, Args...>;

	template <class F, class... Args>
	auto schedule(clock::time_point time, F &&f, Args &&...args) noexcept
	    -> invoke_future_t<F, Args...>;

private:
	ThreadPool();
	~ThreadPool();

	std::function<void()> dequeue(); // returns null function if joining

	std::vector<std::thread> mWorkers;
	std::atomic<int> mBusyWorkers = 0;
	std::atomic<bool> mJoining = false;

	struct Task {
		clock::time_point time;
		std::function<void()> func;
		bool operator>(const Task &other) const { return time > other.time; }
		bool operator<(const Task &other) const { return time < other.time; }
	};
	std::priority_queue<Task, std::deque<Task>, std::greater<Task>> mTasks;

	std::condition_variable mTasksCondition, mWaitingCondition;
	mutable std::mutex mMutex, mWorkersMutex;
};

template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args) noexcept -> invoke_future_t<F, Args...> {
	return schedule(clock::now(), std::forward<F>(f), std::forward<Args>(args)...);
}

template <class F, class... Args>
auto ThreadPool::schedule(clock::duration delay, F &&f, Args &&...args) noexcept
    -> invoke_future_t<F, Args...> {
	return schedule(clock::now() + delay, std::forward<F>(f), std::forward<Args>(args)...);
}

template <class F, class... Args>
auto ThreadPool::schedule(clock::time_point time, F &&f, Args &&...args) noexcept
    -> invoke_future_t<F, Args...> {
	std::unique_lock lock(mMutex);
	using R = std::invoke_result_t<std::decay_t<F>, std::decay_t<Args>...>;
	auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
	auto task = std::make_shared<std::packaged_task<R()>>([bound = std::move(bound)]() mutable {
		try {
			return bound();
		} catch (const std::exception &e) {
			PLOG_WARNING << e.what();
			throw;
		}
	});
	std::future<R> result = task->get_future();

	mTasks.push({time, [task = std::move(task)]() { return (*task)(); }});
	mTasksCondition.notify_one();
	return result;
}

} // namespace rtc::impl

#endif
