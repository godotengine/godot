/**
 * Copyright (c) 2020-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_WS_TRANSPORT_H
#define RTC_IMPL_WS_TRANSPORT_H

#include "common.hpp"
#include "transport.hpp"
#include "wshandshake.hpp"

#if RTC_ENABLE_WEBSOCKET

#include <atomic>

namespace rtc::impl {

class HttpProxyTransport;
class TcpTransport;
class TlsTransport;

class WsTransport final : public Transport, public std::enable_shared_from_this<WsTransport> {
public:
	WsTransport(
	    variant<shared_ptr<TcpTransport>, shared_ptr<HttpProxyTransport>, shared_ptr<TlsTransport>>
	        lower,
	    shared_ptr<WsHandshake> handshake, int maxOutstandingPings, message_callback recvCallback,
	    state_callback stateCallback);
	~WsTransport();

	void start() override;
	void stop() override;
	bool send(message_ptr message) override;
	void close();
	void incoming(message_ptr message) override;

	bool isClient() const { return mIsClient; }

private:
	enum Opcode : uint8_t {
		CONTINUATION = 0,
		TEXT_FRAME = 1,
		BINARY_FRAME = 2,
		CLOSE = 8,
		PING = 9,
		PONG = 10,
	};

	struct Frame {
		Opcode opcode = BINARY_FRAME;
		byte *payload = nullptr;
		size_t length = 0;
		bool fin = true;
		bool mask = true;
	};

	bool sendHttpRequest();
	bool sendHttpError(int code);
	bool sendHttpResponse();

	size_t readFrame(byte *buffer, size_t size, Frame &frame);
	void recvFrame(const Frame &frame);
	bool sendFrame(const Frame &frame);

	void addOutstandingPing();

	const shared_ptr<WsHandshake> mHandshake;
	const bool mIsClient;
	const int mMaxOutstandingPings;

	binary mBuffer;
	binary mPartial;
	Opcode mPartialOpcode;
	std::mutex mSendMutex;
	int mOutstandingPings = 0;
	std::atomic<bool> mCloseSent = false;
};

} // namespace rtc::impl

#endif

#endif
