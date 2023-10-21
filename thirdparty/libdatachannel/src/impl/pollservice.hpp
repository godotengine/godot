/**
 * Copyright (c) 2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_POLL_SERVICE_H
#define RTC_IMPL_POLL_SERVICE_H

#include "common.hpp"
#include "internals.hpp"
#include "pollinterrupter.hpp"
#include "socket.hpp"

#if RTC_ENABLE_WEBSOCKET

#include <chrono>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rtc::impl {

class PollService {
public:
	using clock = std::chrono::steady_clock;

	static PollService &Instance();

	PollService(const PollService &) = delete;
	PollService &operator=(const PollService &) = delete;
	PollService(PollService &&) = delete;
	PollService &operator=(PollService &&) = delete;

	void start();
	void join();

	enum class Direction { Both, In, Out };
	enum class Event { None, Error, Timeout, In, Out };

	struct Params {
		Direction direction;
		optional<clock::duration> timeout;
		std::function<void(Event)> callback;
	};

	void add(socket_t sock, Params params);
	void remove(socket_t sock);

private:
	PollService();
	~PollService();

	void prepare(std::vector<struct pollfd> &pfds, optional<clock::time_point> &next);
	void process(std::vector<struct pollfd> &pfds);
	void runLoop();

	struct SocketEntry {
		Params params;
		optional<clock::time_point> until;
	};

	using SocketMap = std::unordered_map<socket_t, SocketEntry>;
	unique_ptr<SocketMap> mSocks;
	unique_ptr<PollInterrupter> mInterrupter;

	std::recursive_mutex mMutex;
	std::thread mThread;
	bool mStopped;
};

std::ostream &operator<<(std::ostream &out, PollService::Direction direction);

} // namespace rtc::impl

#endif

#endif
