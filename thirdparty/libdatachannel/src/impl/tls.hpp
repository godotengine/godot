/**
 * Copyright (c) 2019-2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_TLS_H
#define RTC_TLS_H

#include "common.hpp"

#include <chrono>

#if USE_GNUTLS

#include <gnutls/gnutls.h>

#include <gnutls/crypto.h>
#include <gnutls/dtls.h>
#include <gnutls/x509.h>

namespace rtc::gnutls {

bool check(int ret, const string &message = "GnuTLS error");

gnutls_certificate_credentials_t *new_credentials();
void free_credentials(gnutls_certificate_credentials_t *creds);

gnutls_x509_crt_t *new_crt();
void free_crt(gnutls_x509_crt_t *crt);

gnutls_x509_privkey_t *new_privkey();
void free_privkey(gnutls_x509_privkey_t *privkey);

gnutls_datum_t make_datum(char *data, size_t size);

} // namespace rtc::gnutls

#elif USE_MBEDTLS

#include "mbedtls/ctr_drbg.h"
#include "mbedtls/ecdsa.h"
#include "mbedtls/entropy.h"
#include "mbedtls/error.h"
#include "mbedtls/pk.h"
#include "mbedtls/rsa.h"
#include "mbedtls/sha256.h"
#include "mbedtls/ssl.h"
#include "mbedtls/x509_crt.h"

namespace rtc::mbedtls {

bool check(int ret, const string &message = "MbedTLS error");

string format_time(const std::chrono::system_clock::time_point &tp);

std::shared_ptr<mbedtls_pk_context> new_pk_context();
std::shared_ptr<mbedtls_x509_crt> new_x509_crt();

} // namespace rtc::mbedtls

#else // OPENSSL

#ifdef _WIN32
// Include winsock2.h header first since OpenSSL may include winsock.h
#include <winsock2.h>
#endif

#include <openssl/ssl.h>

#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/pem.h>
#include <openssl/x509.h>

#ifndef BIO_EOF
#define BIO_EOF -1
#endif

namespace rtc::openssl {

void init();
string error_string(unsigned long error);

bool check(int success, const string &message = "OpenSSL error");
bool check_error(int err, const string &message = "OpenSSL error");

BIO *BIO_new_from_file(const string &filename);

} // namespace rtc::openssl

#endif

#endif
