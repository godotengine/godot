/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_QUEUE_H
#define RTC_IMPL_QUEUE_H

#include "common.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace rtc::impl {

template <typename T> class Queue {
public:
	using amount_function = std::function<size_t(const T &element)>;

	Queue(size_t limit = 0, amount_function func = nullptr);
	~Queue();

	void stop();
	bool running() const;
	bool empty() const;
	bool full() const;
	size_t size() const;   // elements
	size_t amount() const; // amount
	void push(T element);
	optional<T> pop();
	optional<T> peek();
	optional<T> exchange(T element);

private:
	const size_t mLimit;
	size_t mAmount;
	std::queue<T> mQueue;
	std::condition_variable mPushCondition;
	amount_function mAmountFunction;
	bool mStopping = false;

	mutable std::mutex mMutex;
};

template <typename T>
Queue<T>::Queue(size_t limit, amount_function func) : mLimit(limit), mAmount(0) {
	mAmountFunction = func ? func : [](const T &element) -> size_t {
		static_cast<void>(element);
		return 1;
	};
}

template <typename T> Queue<T>::~Queue() { stop(); }

template <typename T> void Queue<T>::stop() {
	std::lock_guard lock(mMutex);
	mStopping = true;
	mPushCondition.notify_all();
}

template <typename T> bool Queue<T>::running() const {
	std::lock_guard lock(mMutex);
	return !mQueue.empty() || !mStopping;
}

template <typename T> bool Queue<T>::empty() const {
	std::lock_guard lock(mMutex);
	return mQueue.empty();
}

template <typename T> bool Queue<T>::full() const {
	std::lock_guard lock(mMutex);
	return mQueue.size() >= mLimit;
}

template <typename T> size_t Queue<T>::size() const {
	std::lock_guard lock(mMutex);
	return mQueue.size();
}

template <typename T> size_t Queue<T>::amount() const {
	std::lock_guard lock(mMutex);
	return mAmount;
}

template <typename T> void Queue<T>::push(T element) {
	std::unique_lock lock(mMutex);
	mPushCondition.wait(lock, [this]() { return !mLimit || mQueue.size() < mLimit || mStopping; });
	if (mStopping)
		return;

	mAmount += mAmountFunction(element);
	mQueue.emplace(std::move(element));
}

template <typename T> optional<T> Queue<T>::pop() {
	std::unique_lock lock(mMutex);
	if (mQueue.empty())
		return nullopt;

	mAmount -= mAmountFunction(mQueue.front());
	optional<T> element{std::move(mQueue.front())};
	mQueue.pop();
	return element;
}

template <typename T> optional<T> Queue<T>::peek() {
	std::unique_lock lock(mMutex);
	return !mQueue.empty() ? std::make_optional(mQueue.front()) : nullopt;
}

template <typename T> optional<T> Queue<T>::exchange(T element) {
	std::unique_lock lock(mMutex);
	if (mQueue.empty())
		return nullopt;

	std::swap(mQueue.front(), element);
	return std::make_optional(std::move(element));
}

} // namespace rtc::impl

#endif
