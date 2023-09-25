/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_DTLS_TRANSPORT_H
#define RTC_IMPL_DTLS_TRANSPORT_H

#include "certificate.hpp"
#include "common.hpp"
#include "queue.hpp"
#include "tls.hpp"
#include "transport.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

namespace rtc::impl {

class IceTransport;

class DtlsTransport : public Transport, public std::enable_shared_from_this<DtlsTransport> {
public:
	static void Init();
	static void Cleanup();

	using verifier_callback = std::function<bool(const std::string &fingerprint)>;

	DtlsTransport(shared_ptr<IceTransport> lower, certificate_ptr certificate, optional<size_t> mtu,
	              verifier_callback verifierCallback, state_callback stateChangeCallback);
	~DtlsTransport();

	virtual void start() override;
	virtual void stop() override;
	virtual bool send(message_ptr message) override; // false if dropped

	bool isClient() const { return mIsClient; }

protected:
	virtual void incoming(message_ptr message) override;
	virtual bool outgoing(message_ptr message) override;
	virtual bool demuxMessage(message_ptr message);
	virtual void postHandshake();

	void enqueueRecv();
	void doRecv();

	const optional<size_t> mMtu;
	const certificate_ptr mCertificate;
	const verifier_callback mVerifierCallback;
	const bool mIsClient;

	Queue<message_ptr> mIncomingQueue;
	std::atomic<int> mPendingRecvCount = 0;
	std::mutex mRecvMutex;
	std::atomic<unsigned int> mCurrentDscp = 0;
	std::atomic<bool> mOutgoingResult = true;

#if USE_GNUTLS
	gnutls_session_t mSession;
	std::mutex mSendMutex;

	static int CertificateCallback(gnutls_session_t session);
	static ssize_t WriteCallback(gnutls_transport_ptr_t ptr, const void *data, size_t len);
	static ssize_t ReadCallback(gnutls_transport_ptr_t ptr, void *data, size_t maxlen);
	static int TimeoutCallback(gnutls_transport_ptr_t ptr, unsigned int ms);

#elif USE_MBEDTLS
	mbedtls_entropy_context mEntropy;
	mbedtls_ctr_drbg_context mDrbg;
	mbedtls_ssl_config mConf;
	mbedtls_ssl_context mSsl;

	std::mutex mSslMutex;

	uint32_t mFinMs = 0, mIntMs = 0;
	std::chrono::time_point<std::chrono::steady_clock> mTimerSetAt;

	char mMasterSecret[48];
	char mRandBytes[64];
	mbedtls_tls_prf_types mTlsProfile = MBEDTLS_SSL_TLS_PRF_NONE;

	static int CertificateCallback(void *ctx, mbedtls_x509_crt *crt, int depth, uint32_t *flags);
	static int WriteCallback(void *ctx, const unsigned char *buf, size_t len);
	static int ReadCallback(void *ctx, unsigned char *buf, size_t len);
	static void ExportKeysCallback(void *ctx, mbedtls_ssl_key_export_type type,
	                               const unsigned char *secret, size_t secret_len,
	                               const unsigned char client_random[32],
	                               const unsigned char server_random[32],
	                               mbedtls_tls_prf_types tls_prf_type);
	static void SetTimerCallback(void *ctx, uint32_t int_ms, uint32_t fin_ms);
	static int GetTimerCallback(void *ctx);

#else // OPENSSL
	SSL_CTX *mCtx = NULL;
	SSL *mSsl = NULL;
	BIO *mInBio, *mOutBio;
	std::mutex mSslMutex;

	void handleTimeout();

	static BIO_METHOD *BioMethods;
	static int TransportExIndex;
	static std::mutex GlobalMutex;

	static int CertificateCallback(int preverify_ok, X509_STORE_CTX *ctx);
	static void InfoCallback(const SSL *ssl, int where, int ret);

	static int BioMethodNew(BIO *bio);
	static int BioMethodFree(BIO *bio);
	static int BioMethodWrite(BIO *bio, const char *in, int inl);
	static long BioMethodCtrl(BIO *bio, int cmd, long num, void *ptr);
#endif
};

} // namespace rtc::impl

#endif
