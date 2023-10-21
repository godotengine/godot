/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_PROCESSOR_H
#define RTC_IMPL_PROCESSOR_H

#include "common.hpp"
#include "queue.hpp"
#include "threadpool.hpp"

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>

namespace rtc::impl {

// Processed tasks in order by delegating them to the thread pool
class Processor {
public:
	Processor(size_t limit = 0);
	virtual ~Processor();

	Processor(const Processor &) = delete;
	Processor &operator=(const Processor &) = delete;
	Processor(Processor &&) = delete;
	Processor &operator=(Processor &&) = delete;

	void join();

	template <class F, class... Args> void enqueue(F &&f, Args &&...args) noexcept;

private:
	void schedule();

	Queue<std::function<void()>> mTasks;
	bool mPending = false; // true iff a task is pending in the thread pool

	mutable std::mutex mMutex;
	std::condition_variable mCondition;
};

class TearDownProcessor final : public Processor {
public:
	static TearDownProcessor &Instance();

private:
	TearDownProcessor();
	~TearDownProcessor();
};

template <class F, class... Args> void Processor::enqueue(F &&f, Args &&...args) noexcept {
	std::unique_lock lock(mMutex);
	auto bound = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
	auto task = [this, bound = std::move(bound)]() mutable {
		scope_guard guard(std::bind(&Processor::schedule, this)); // chain the next task
		return bound();
	};

	if (!mPending) {
		ThreadPool::Instance().enqueue(std::move(task));
		mPending = true;
	} else {
		mTasks.push(std::move(task));
	}
}

} // namespace rtc::impl

#endif
