/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_CERTIFICATE_H
#define RTC_IMPL_CERTIFICATE_H

#include "common.hpp"
#include "configuration.hpp" // for CertificateType
#include "init.hpp"
#include "tls.hpp"

#include <future>
#include <tuple>

namespace rtc::impl {

class Certificate {
public:
	static Certificate FromString(string crt_pem, string key_pem);
	static Certificate FromFile(const string &crt_pem_file, const string &key_pem_file,
	                            const string &pass = "");
	static Certificate Generate(CertificateType type, const string &commonName);

#if USE_GNUTLS
	Certificate(gnutls_x509_crt_t crt, gnutls_x509_privkey_t privkey);
	gnutls_certificate_credentials_t credentials() const;
#elif USE_MBEDTLS
	Certificate(shared_ptr<mbedtls_x509_crt> crt, shared_ptr<mbedtls_pk_context> pk);
	std::tuple<shared_ptr<mbedtls_x509_crt>, shared_ptr<mbedtls_pk_context>> credentials() const;
#else // OPENSSL
	Certificate(shared_ptr<X509> x509, shared_ptr<EVP_PKEY> pkey);
	std::tuple<X509 *, EVP_PKEY *> credentials() const;
#endif

	string fingerprint() const;

private:
	const init_token mInitToken = Init::Instance().token();

#if USE_GNUTLS
	Certificate(shared_ptr<gnutls_certificate_credentials_t> creds);
	const shared_ptr<gnutls_certificate_credentials_t> mCredentials;
#elif USE_MBEDTLS
	const shared_ptr<mbedtls_x509_crt> mCrt;
	const shared_ptr<mbedtls_pk_context> mPk;
#else
	const shared_ptr<X509> mX509;
	const shared_ptr<EVP_PKEY> mPKey;
#endif

	const string mFingerprint;
};

#if USE_GNUTLS
string make_fingerprint(gnutls_certificate_credentials_t credentials);
string make_fingerprint(gnutls_x509_crt_t crt);
#elif USE_MBEDTLS
string make_fingerprint(mbedtls_x509_crt *crt);
#else
string make_fingerprint(X509 *x509);
#endif

using certificate_ptr = shared_ptr<Certificate>;
using future_certificate_ptr = std::shared_future<certificate_ptr>;

future_certificate_ptr make_certificate(CertificateType type = CertificateType::Default);

} // namespace rtc::impl

#endif
