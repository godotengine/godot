/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_TLS_TRANSPORT_H
#define RTC_IMPL_TLS_TRANSPORT_H

#include "certificate.hpp"
#include "common.hpp"
#include "queue.hpp"
#include "tls.hpp"
#include "transport.hpp"

#if RTC_ENABLE_WEBSOCKET

#include <atomic>
#include <thread>

namespace rtc::impl {

class TcpTransport;
class HttpProxyTransport;

class TlsTransport : public Transport, public std::enable_shared_from_this<TlsTransport> {
public:
	static void Init();
	static void Cleanup();

	TlsTransport(variant<shared_ptr<TcpTransport>, shared_ptr<HttpProxyTransport>> lower,
	             optional<string> host, certificate_ptr certificate, state_callback callback);
	virtual ~TlsTransport();

	void start() override;
	void stop() override;
	bool send(message_ptr message) override;

	bool isClient() const { return mIsClient; }

protected:
	virtual void incoming(message_ptr message) override;
	virtual bool outgoing(message_ptr message) override;
	virtual void postHandshake();

	void enqueueRecv();
	void doRecv();

	const optional<string> mHost;
	const bool mIsClient;

	Queue<message_ptr> mIncomingQueue;
	std::atomic<int> mPendingRecvCount = 0;
	std::mutex mRecvMutex;

#if USE_GNUTLS
	gnutls_session_t mSession;

	message_ptr mIncomingMessage;
	size_t mIncomingMessagePosition = 0;
	std::atomic<bool> mOutgoingResult = true;

	static ssize_t WriteCallback(gnutls_transport_ptr_t ptr, const void *data, size_t len);
	static ssize_t ReadCallback(gnutls_transport_ptr_t ptr, void *data, size_t maxlen);
	static int TimeoutCallback(gnutls_transport_ptr_t ptr, unsigned int ms);

#elif USE_MBEDTLS
	mbedtls_entropy_context mEntropy;
	mbedtls_ctr_drbg_context mDrbg;
	mbedtls_ssl_config mConf;
	mbedtls_ssl_context mSsl;

	std::mutex mSslMutex;
	std::atomic<bool> mOutgoingResult = true;

	message_ptr mIncomingMessage;
	size_t mIncomingMessagePosition = 0;

	static int WriteCallback(void *ctx, const unsigned char *buf, size_t len);
	static int ReadCallback(void *ctx, unsigned char *buf, size_t len);

#else
	SSL_CTX *mCtx;
	SSL *mSsl;
	BIO *mInBio, *mOutBio;
	std::mutex mSslMutex;

	bool flushOutput();

	static int TransportExIndex;

	static void InfoCallback(const SSL *ssl, int where, int ret);
#endif
};

} // namespace rtc::impl

#endif

#endif
