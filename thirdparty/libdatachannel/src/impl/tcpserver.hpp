/**
 * Copyright (c) 2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_TCP_SERVER_H
#define RTC_IMPL_TCP_SERVER_H

#include "common.hpp"
#include "pollinterrupter.hpp"
#include "queue.hpp"
#include "socket.hpp"
#include "tcptransport.hpp"

#if RTC_ENABLE_WEBSOCKET

namespace rtc::impl {

class TcpServer final {
public:
	TcpServer(uint16_t port, const char *bindAddress = nullptr);
	~TcpServer();

	TcpServer(const TcpServer &other) = delete;
	void operator=(const TcpServer &other) = delete;

	shared_ptr<TcpTransport> accept();
	void close();

	uint16_t port() const { return mPort; }

private:
	void listen(uint16_t port, const char *bindAddress);

	uint16_t mPort;
	socket_t mSock = INVALID_SOCKET;
	std::mutex mSockMutex;
	PollInterrupter mInterrupter;
};

} // namespace rtc::impl

#endif

#endif
