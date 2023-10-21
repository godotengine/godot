/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_UTILS_H
#define RTC_UTILS_H

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <utility>

namespace rtc {

// overloaded helper
template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// weak_ptr bind helper
template <typename F, typename T, typename... Args> auto weak_bind(F &&f, T *t, Args &&..._args) {
	return [bound = std::bind(f, t, _args...), weak_this = t->weak_from_this()](auto &&...args) {
		if (auto shared_this = weak_this.lock())
			return bound(args...);
		else
			return static_cast<decltype(bound(args...))>(false);
	};
}

// scope_guard helper
class scope_guard final {
public:
	scope_guard(std::function<void()> func) : function(std::move(func)) {}
	scope_guard(scope_guard &&other) = delete;
	scope_guard(const scope_guard &) = delete;
	void operator=(const scope_guard &) = delete;

	~scope_guard() {
		if (function)
			function();
	}

private:
	std::function<void()> function;
};

// callback with built-in synchronization
template <typename... Args> class synchronized_callback {
public:
	synchronized_callback() = default;
	synchronized_callback(synchronized_callback &&cb) { *this = std::move(cb); }
	synchronized_callback(const synchronized_callback &cb) { *this = cb; }
	synchronized_callback(std::function<void(Args...)> func) { *this = std::move(func); }
	virtual ~synchronized_callback() { *this = nullptr; }

	synchronized_callback &operator=(synchronized_callback &&cb) {
		std::scoped_lock lock(mutex, cb.mutex);
		set(std::exchange(cb.callback, nullptr));
		return *this;
	}

	synchronized_callback &operator=(const synchronized_callback &cb) {
		std::scoped_lock lock(mutex, cb.mutex);
		set(cb.callback);
		return *this;
	}

	synchronized_callback &operator=(std::function<void(Args...)> func) {
		std::lock_guard lock(mutex);
		set(std::move(func));
		return *this;
	}

	bool operator()(Args... args) const {
		std::lock_guard lock(mutex);
		return call(std::move(args)...);
	}

	operator bool() const {
		std::lock_guard lock(mutex);
		return callback ? true : false;
	}

protected:
	virtual void set(std::function<void(Args...)> func) { callback = std::move(func); }
	virtual bool call(Args... args) const {
		if (!callback)
			return false;

		callback(std::move(args)...);
		return true;
	}

	std::function<void(Args...)> callback;
	mutable std::recursive_mutex mutex;
};

// callback with built-in synchronization and replay of the last missed call
template <typename... Args>
class synchronized_stored_callback final : public synchronized_callback<Args...> {
public:
	template <typename... CArgs>
	synchronized_stored_callback(CArgs &&...cargs)
	    : synchronized_callback<Args...>(std::forward<CArgs>(cargs)...) {}
	~synchronized_stored_callback() {}

private:
	void set(std::function<void(Args...)> func) {
		synchronized_callback<Args...>::set(func);
		if (func && stored) {
			std::apply(func, std::move(*stored));
			stored.reset();
		}
	}

	bool call(Args... args) const {
		if (!synchronized_callback<Args...>::call(args...))
			stored.emplace(std::move(args)...);

		return true;
	}

	mutable std::optional<std::tuple<Args...>> stored;
};

// pimpl base class
template <typename T> using impl_ptr = std::shared_ptr<T>;
template <typename T> class CheshireCat {
public:
	CheshireCat(impl_ptr<T> impl) : mImpl(std::move(impl)) {}
	template <typename... Args>
	CheshireCat(Args... args) : mImpl(std::make_shared<T>(std::forward<Args>(args)...)) {}
	CheshireCat(CheshireCat<T> &&cc) { *this = std::move(cc); }
	CheshireCat(const CheshireCat<T> &) = delete;

	virtual ~CheshireCat() = default;

	CheshireCat &operator=(CheshireCat<T> &&cc) {
		mImpl = std::move(cc.mImpl);
		return *this;
	};
	CheshireCat &operator=(const CheshireCat<T> &) = delete;

protected:
	impl_ptr<T> impl() { return mImpl; }
	impl_ptr<const T> impl() const { return mImpl; }

private:
	impl_ptr<T> mImpl;
};

} // namespace rtc

#endif
