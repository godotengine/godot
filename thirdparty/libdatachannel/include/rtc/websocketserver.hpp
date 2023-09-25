/**
 * Copyright (c) 2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_WEBSOCKETSERVER_H
#define RTC_WEBSOCKETSERVER_H

#if RTC_ENABLE_WEBSOCKET

#include "common.hpp"
#include "websocket.hpp"

namespace rtc {

namespace impl {

struct WebSocketServer;

}

class RTC_CPP_EXPORT WebSocketServer final : private CheshireCat<impl::WebSocketServer> {
public:
	struct Configuration {
		uint16_t port = 8080;
		bool enableTls = false;
		optional<string> certificatePemFile;
		optional<string> keyPemFile;
		optional<string> keyPemPass;
		optional<string> bindAddress;
		optional<std::chrono::milliseconds> connectionTimeout;
	};

	WebSocketServer();
	WebSocketServer(Configuration config);
	~WebSocketServer();

	void stop();

	uint16_t port() const;

	void onClient(std::function<void(shared_ptr<WebSocket>)> callback);

private:
	using CheshireCat<impl::WebSocketServer>::impl;
};

} // namespace rtc

#endif

#endif // RTC_WEBSOCKET_H
