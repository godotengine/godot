/**
 * Copyright (c) 2020-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_WEBSOCKET_H
#define RTC_IMPL_WEBSOCKET_H

#if RTC_ENABLE_WEBSOCKET

#include "channel.hpp"
#include "common.hpp"
#include "httpproxytransport.hpp"
#include "init.hpp"
#include "message.hpp"
#include "queue.hpp"
#include "tcptransport.hpp"
#include "tlstransport.hpp"
#include "wstransport.hpp"

#include "rtc/websocket.hpp"

#include <atomic>
#include <thread>

namespace rtc::impl {

struct WebSocket final : public Channel, public std::enable_shared_from_this<WebSocket> {
	using State = rtc::WebSocket::State;
	using Configuration = rtc::WebSocket::Configuration;

	WebSocket(optional<Configuration> optConfig = nullopt, certificate_ptr certificate = nullptr);
	~WebSocket();

	void open(const string &url);
	void close();
	void remoteClose();
	bool outgoing(message_ptr message);
	void incoming(message_ptr message);

	optional<message_variant> receive() override;
	optional<message_variant> peek() override;
	size_t availableAmount() const override;

	bool isOpen() const;
	bool isClosed() const;
	size_t maxMessageSize() const;

	bool changeState(State state);

	shared_ptr<TcpTransport> setTcpTransport(shared_ptr<TcpTransport> transport);
	shared_ptr<HttpProxyTransport> initProxyTransport();
	shared_ptr<TlsTransport> initTlsTransport();
	shared_ptr<WsTransport> initWsTransport();
	shared_ptr<TcpTransport> getTcpTransport() const;
	shared_ptr<TlsTransport> getTlsTransport() const;
	shared_ptr<WsTransport> getWsTransport() const;
	shared_ptr<WsHandshake> getWsHandshake() const;

	void closeTransports();

	const Configuration config;

	std::atomic<State> state = State::Closed;

private:
	void scheduleConnectionTimeout();

	const init_token mInitToken = Init::Instance().token();

	const certificate_ptr mCertificate;
	bool mIsSecure;

	optional<string> mHostname; // for TLS SNI and Proxy
	optional<string> mService;  // for Proxy

	shared_ptr<TcpTransport> mTcpTransport;
	shared_ptr<HttpProxyTransport> mProxyTransport;
	shared_ptr<TlsTransport> mTlsTransport;
	shared_ptr<WsTransport> mWsTransport;
	shared_ptr<WsHandshake> mWsHandshake;

	Queue<message_ptr> mRecvQueue;
};

} // namespace rtc::impl

#endif

#endif // RTC_IMPL_WEBSOCKET_H
