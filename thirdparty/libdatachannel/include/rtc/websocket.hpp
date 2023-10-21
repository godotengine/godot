/**
 * Copyright (c) 2020-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_WEBSOCKET_H
#define RTC_WEBSOCKET_H

#if RTC_ENABLE_WEBSOCKET

#include "channel.hpp"
#include "common.hpp"
#include "configuration.hpp" // for ProxyServer

namespace rtc {

namespace impl {

struct WebSocket;

}

class RTC_CPP_EXPORT WebSocket final : private CheshireCat<impl::WebSocket>, public Channel {
public:
	enum class State : int {
		Connecting = 0,
		Open = 1,
		Closing = 2,
		Closed = 3,
	};

	struct Configuration {
		bool disableTlsVerification = false; // if true, don't verify the TLS certificate
		optional<ProxyServer> proxyServer;   // only non-authenticated http supported for now
		std::vector<string> protocols;
		optional<std::chrono::milliseconds> connectionTimeout; // zero to disable
		optional<std::chrono::milliseconds> pingInterval;      // zero to disable
		optional<int> maxOutstandingPings;
	};

	WebSocket();
	WebSocket(Configuration config);
	WebSocket(impl_ptr<impl::WebSocket> impl);
	~WebSocket() override;

	State readyState() const;

	bool isOpen() const override;
	bool isClosed() const override;
	size_t maxMessageSize() const override;

	void open(const string &url);
	void close() override;
	void forceClose();
	bool send(const message_variant data) override;
	bool send(const byte *data, size_t size) override;

	optional<string> remoteAddress() const;
	optional<string> path() const;

private:
	using CheshireCat<impl::WebSocket>::impl;
};

} // namespace rtc

#endif

#endif // RTC_WEBSOCKET_H
