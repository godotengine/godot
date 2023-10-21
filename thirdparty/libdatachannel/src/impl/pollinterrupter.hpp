/**
 * Copyright (c) 2020-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_POLL_INTERRUPTER_H
#define RTC_IMPL_POLL_INTERRUPTER_H

#include "common.hpp"
#include "socket.hpp"

#if RTC_ENABLE_WEBSOCKET

namespace rtc::impl {

// Utility class to interrupt poll()
class PollInterrupter final {
public:
	PollInterrupter();
	~PollInterrupter();

	PollInterrupter(const PollInterrupter &other) = delete;
	void operator=(const PollInterrupter &other) = delete;

	void prepare(struct pollfd &pfd);
	void process(struct pollfd &pfd);
	void interrupt();

private:
#ifdef _WIN32
	socket_t mSock;
#else // assume POSIX
	int mPipeIn, mPipeOut;
#endif
};

} // namespace rtc::impl

#endif

#endif
