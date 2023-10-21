/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "certificate.hpp"
#include "threadpool.hpp"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace rtc::impl {

#if USE_GNUTLS

Certificate Certificate::FromString(string crt_pem, string key_pem) {
	PLOG_DEBUG << "Importing certificate from PEM string (GnuTLS)";

	shared_ptr<gnutls_certificate_credentials_t> creds(gnutls::new_credentials(),
	                                                   gnutls::free_credentials);
	gnutls_datum_t crt_datum = gnutls::make_datum(crt_pem.data(), crt_pem.size());
	gnutls_datum_t key_datum = gnutls::make_datum(key_pem.data(), key_pem.size());
	gnutls::check(
	    gnutls_certificate_set_x509_key_mem(*creds, &crt_datum, &key_datum, GNUTLS_X509_FMT_PEM),
	    "Unable to import PEM certificate and key");

	return Certificate(std::move(creds));
}

Certificate Certificate::FromFile(const string &crt_pem_file, const string &key_pem_file,
                                  const string &pass) {
	PLOG_DEBUG << "Importing certificate from PEM file (GnuTLS): " << crt_pem_file;

	shared_ptr<gnutls_certificate_credentials_t> creds(gnutls::new_credentials(),
	                                                   gnutls::free_credentials);
	gnutls::check(gnutls_certificate_set_x509_key_file2(*creds, crt_pem_file.c_str(),
	                                                    key_pem_file.c_str(), GNUTLS_X509_FMT_PEM,
	                                                    pass.c_str(), 0),
	              "Unable to import PEM certificate and key from file");

	return Certificate(std::move(creds));
}

Certificate Certificate::Generate(CertificateType type, const string &commonName) {
	PLOG_DEBUG << "Generating certificate (GnuTLS)";

	using namespace gnutls;
	unique_ptr<gnutls_x509_crt_t, decltype(&free_crt)> crt(new_crt(), free_crt);
	unique_ptr<gnutls_x509_privkey_t, decltype(&free_privkey)> privkey(new_privkey(), free_privkey);

	switch (type) {
	// RFC 8827 WebRTC Security Architecture 6.5. Communications Security
	// All implementations MUST support DTLS 1.2 with the TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
	// cipher suite and the P-256 curve
	// See https://www.rfc-editor.org/rfc/rfc8827.html#section-6.5
	case CertificateType::Default:
	case CertificateType::Ecdsa: {
		gnutls::check(gnutls_x509_privkey_generate(*privkey, GNUTLS_PK_ECDSA,
		                                           GNUTLS_CURVE_TO_BITS(GNUTLS_ECC_CURVE_SECP256R1),
		                                           0),
		              "Unable to generate ECDSA P-256 key pair");
		break;
	}
	case CertificateType::Rsa: {
		const unsigned int bits = 2048;
		gnutls::check(gnutls_x509_privkey_generate(*privkey, GNUTLS_PK_RSA, bits, 0),
		              "Unable to generate RSA key pair");
		break;
	}
	default:
		throw std::invalid_argument("Unknown certificate type");
	}

	using namespace std::chrono;
	auto now = time_point_cast<seconds>(system_clock::now());
	gnutls_x509_crt_set_activation_time(*crt, (now - hours(1)).time_since_epoch().count());
	gnutls_x509_crt_set_expiration_time(*crt, (now + hours(24 * 365)).time_since_epoch().count());
	gnutls_x509_crt_set_version(*crt, 1);
	gnutls_x509_crt_set_key(*crt, *privkey);
	gnutls_x509_crt_set_dn_by_oid(*crt, GNUTLS_OID_X520_COMMON_NAME, 0, commonName.data(),
	                              commonName.size());

	const size_t serialSize = 16;
	char serial[serialSize];
	gnutls_rnd(GNUTLS_RND_NONCE, serial, serialSize);
	gnutls_x509_crt_set_serial(*crt, serial, serialSize);

	gnutls::check(gnutls_x509_crt_sign2(*crt, *crt, *privkey, GNUTLS_DIG_SHA256, 0),
	              "Unable to auto-sign certificate");

	return Certificate(*crt, *privkey);
}

Certificate::Certificate(gnutls_x509_crt_t crt, gnutls_x509_privkey_t privkey)
    : mCredentials(gnutls::new_credentials(), gnutls::free_credentials),
      mFingerprint(make_fingerprint(crt)) {

	gnutls::check(gnutls_certificate_set_x509_key(*mCredentials, &crt, 1, privkey),
	              "Unable to set certificate and key pair in credentials");
}

Certificate::Certificate(shared_ptr<gnutls_certificate_credentials_t> creds)
    : mCredentials(std::move(creds)), mFingerprint(make_fingerprint(*mCredentials)) {}

gnutls_certificate_credentials_t Certificate::credentials() const { return *mCredentials; }

string make_fingerprint(gnutls_certificate_credentials_t credentials) {
	auto new_crt_list = [credentials]() -> gnutls_x509_crt_t * {
		gnutls_x509_crt_t *crt_list = nullptr;
		unsigned int crt_list_size = 0;
		gnutls::check(gnutls_certificate_get_x509_crt(credentials, 0, &crt_list, &crt_list_size));
		assert(crt_list_size == 1);
		return crt_list;
	};

	auto free_crt_list = [](gnutls_x509_crt_t *crt_list) {
		gnutls_x509_crt_deinit(crt_list[0]);
		gnutls_free(crt_list);
	};

	unique_ptr<gnutls_x509_crt_t, decltype(free_crt_list)> crt_list(new_crt_list(), free_crt_list);

	return make_fingerprint(*crt_list);
}

string make_fingerprint(gnutls_x509_crt_t crt) {
	const size_t size = 32;
	unsigned char buffer[size];
	size_t len = size;
	gnutls::check(gnutls_x509_crt_get_fingerprint(crt, GNUTLS_DIG_SHA256, buffer, &len),
	              "X509 fingerprint error");

	std::ostringstream oss;
	oss << std::hex << std::uppercase << std::setfill('0');
	for (size_t i = 0; i < len; ++i) {
		if (i)
			oss << std::setw(1) << ':';
		oss << std::setw(2) << unsigned(buffer[i]);
	}
	return oss.str();
}

#elif USE_MBEDTLS
string make_fingerprint(mbedtls_x509_crt *crt) {
	const int size = 32;
	uint8_t buffer[size];
	std::stringstream fingerprint;

	mbedtls::check(
	    mbedtls_sha256(crt->raw.p, crt->raw.len, reinterpret_cast<unsigned char *>(buffer), 0),
	    "Failed to generate certificate fingerprint");

	for (auto i = 0; i < size; i++) {
		fingerprint << std::setfill('0') << std::setw(2) << std::hex << static_cast<int>(buffer[i]);
		if (i != (size - 1)) {
			fingerprint << ":";
		}
	}

	return fingerprint.str();
}

Certificate::Certificate(shared_ptr<mbedtls_x509_crt> crt, shared_ptr<mbedtls_pk_context> pk)
    : mCrt(crt), mPk(pk), mFingerprint(make_fingerprint(crt.get())) {}

Certificate Certificate::FromString(string crt_pem, string key_pem) {
	PLOG_DEBUG << "Importing certificate from PEM string (MbedTLS)";

	auto crt = mbedtls::new_x509_crt();
	auto pk = mbedtls::new_pk_context();

	mbedtls::check(mbedtls_x509_crt_parse(crt.get(),
	                                      reinterpret_cast<const unsigned char *>(crt_pem.c_str()),
	                                      crt_pem.length()),
	               "Failed to parse certificate");
	mbedtls::check(mbedtls_pk_parse_key(pk.get(),
	                                    reinterpret_cast<const unsigned char *>(key_pem.c_str()),
	                                    key_pem.size(), NULL, 0, NULL, 0),
	               "Failed to parse key");

	return Certificate(std::move(crt), std::move(pk));
}

Certificate Certificate::FromFile(const string &crt_pem_file, const string &key_pem_file,
                                  const string &pass) {
	PLOG_DEBUG << "Importing certificate from PEM file (MbedTLS): " << crt_pem_file;

	auto crt = mbedtls::new_x509_crt();
	auto pk = mbedtls::new_pk_context();

	mbedtls::check(mbedtls_x509_crt_parse_file(crt.get(), crt_pem_file.c_str()),
	               "Failed to parse certificate");
	mbedtls::check(mbedtls_pk_parse_keyfile(pk.get(), key_pem_file.c_str(), pass.c_str(), 0, NULL),
	               "Failed to parse key");

	return Certificate(std::move(crt), std::move(pk));
}

Certificate Certificate::Generate(CertificateType type, const string &commonName) {
	PLOG_DEBUG << "Generating certificate (MbedTLS)";

	mbedtls_entropy_context entropy;
	mbedtls_ctr_drbg_context drbg;
	mbedtls_x509write_cert wcrt;
	mbedtls_mpi serial;
	auto crt = mbedtls::new_x509_crt();
	auto pk = mbedtls::new_pk_context();

	mbedtls_entropy_init(&entropy);
	mbedtls_ctr_drbg_init(&drbg);
	mbedtls_ctr_drbg_set_prediction_resistance(&drbg, MBEDTLS_CTR_DRBG_PR_ON);
	mbedtls_x509write_crt_init(&wcrt);
	mbedtls_mpi_init(&serial);

	try {
		mbedtls::check(mbedtls_ctr_drbg_seed(
		    &drbg, mbedtls_entropy_func, &entropy,
		    reinterpret_cast<const unsigned char *>(commonName.data()), commonName.size()));

		switch (type) {
		// RFC 8827 WebRTC Security Architecture 6.5. Communications Security
		// All implementations MUST support DTLS 1.2 with the
		// TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256 cipher suite and the P-256 curve
		// See https://www.rfc-editor.org/rfc/rfc8827.html#section-6.5
		case CertificateType::Default:
		case CertificateType::Ecdsa: {
			mbedtls::check(mbedtls_pk_setup(pk.get(), mbedtls_pk_info_from_type(MBEDTLS_PK_ECKEY)));
			mbedtls::check(mbedtls_ecp_gen_key(MBEDTLS_ECP_DP_SECP256R1, mbedtls_pk_ec(*pk.get()),
			                                   mbedtls_ctr_drbg_random, &drbg),
			               "Unable to generate ECDSA P-256 key pair");
			break;
		}
		case CertificateType::Rsa: {
			const unsigned int nbits = 2048;
			const int exponent = 65537;

			mbedtls::check(mbedtls_pk_setup(pk.get(), mbedtls_pk_info_from_type(MBEDTLS_PK_RSA)));
			mbedtls::check(mbedtls_rsa_gen_key(mbedtls_pk_rsa(*pk.get()), mbedtls_ctr_drbg_random,
			                                   &drbg, nbits, exponent),
			               "Unable to generate RSA key pair");
			break;
		}
		default:
			throw std::invalid_argument("Unknown certificate type");
		}

		auto now = std::chrono::system_clock::now();
		string notBefore = mbedtls::format_time(now - std::chrono::hours(1));
		string notAfter = mbedtls::format_time(now + std::chrono::hours(24 * 365));

		const size_t serialBufferSize = 16;
		unsigned char serialBuffer[serialBufferSize];
		mbedtls::check(mbedtls_ctr_drbg_random(&drbg, serialBuffer, serialBufferSize),
		               "Failed to generate certificate");
		mbedtls::check(mbedtls_mpi_read_binary(&serial, serialBuffer, serialBufferSize),
		               "Failed to generate certificate");

		std::string name = std::string("O=" + commonName + ",CN=" + commonName);
		mbedtls::check(mbedtls_x509write_crt_set_serial(&wcrt, &serial),
		               "Failed to generate certificate");
		mbedtls::check(mbedtls_x509write_crt_set_subject_name(&wcrt, name.c_str()),
		               "Failed to generate certificate");
		mbedtls::check(mbedtls_x509write_crt_set_issuer_name(&wcrt, name.c_str()),
		               "Failed to generate certificate");
		mbedtls::check(
		    mbedtls_x509write_crt_set_validity(&wcrt, notBefore.c_str(), notAfter.c_str()),
		    "Failed to generate certificate");

		mbedtls_x509write_crt_set_version(&wcrt, MBEDTLS_X509_CRT_VERSION_3);
		mbedtls_x509write_crt_set_subject_key(&wcrt, pk.get());
		mbedtls_x509write_crt_set_issuer_key(&wcrt, pk.get());
		mbedtls_x509write_crt_set_md_alg(&wcrt, MBEDTLS_MD_SHA256);

		const size_t certificateBufferSize = 4096;
		unsigned char certificateBuffer[certificateBufferSize];
		std::memset(certificateBuffer, 0, certificateBufferSize);

		auto certificateLen = mbedtls_x509write_crt_der(
		    &wcrt, certificateBuffer, certificateBufferSize, mbedtls_ctr_drbg_random, &drbg);
		if (certificateLen <= 0) {
			throw std::runtime_error("Certificate generation failed");
		}

		mbedtls::check(mbedtls_x509_crt_parse_der(
		                   crt.get(), (certificateBuffer + certificateBufferSize - certificateLen),
		                   certificateLen),
		               "Failed to generate certificate");
	} catch (...) {
		mbedtls_entropy_free(&entropy);
		mbedtls_ctr_drbg_free(&drbg);
		mbedtls_x509write_crt_free(&wcrt);
		mbedtls_mpi_free(&serial);
		throw;
	}

	mbedtls_entropy_free(&entropy);
	mbedtls_ctr_drbg_free(&drbg);
	mbedtls_x509write_crt_free(&wcrt);
	mbedtls_mpi_free(&serial);
	return Certificate(std::move(crt), std::move(pk));
}

std::tuple<shared_ptr<mbedtls_x509_crt>, shared_ptr<mbedtls_pk_context>>
Certificate::credentials() const {
	return {mCrt, mPk};
}

#else // OPENSSL

#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/rsa.h>

namespace {

// Dummy password callback that copies the password from user data
int dummy_pass_cb(char *buf, int size, int /*rwflag*/, void *u) {
	const char *pass = static_cast<char *>(u);
	return snprintf(buf, size, "%s", pass);
}

} // namespace

Certificate Certificate::FromString(string crt_pem, string key_pem) {
	PLOG_DEBUG << "Importing certificate from PEM string (OpenSSL)";

	BIO *bio = BIO_new(BIO_s_mem());
	BIO_write(bio, crt_pem.data(), int(crt_pem.size()));
	auto x509 = shared_ptr<X509>(PEM_read_bio_X509(bio, nullptr, nullptr, nullptr), X509_free);
	BIO_free(bio);
	if (!x509)
		throw std::invalid_argument("Unable to import PEM certificate");

	bio = BIO_new(BIO_s_mem());
	BIO_write(bio, key_pem.data(), int(key_pem.size()));
	auto pkey = shared_ptr<EVP_PKEY>(PEM_read_bio_PrivateKey(bio, nullptr, nullptr, nullptr),
	                                 EVP_PKEY_free);
	BIO_free(bio);
	if (!pkey)
		throw std::invalid_argument("Unable to import PEM key");

	return Certificate(x509, pkey);
}

Certificate Certificate::FromFile(const string &crt_pem_file, const string &key_pem_file,
                                  const string &pass) {
	PLOG_DEBUG << "Importing certificate from PEM file (OpenSSL): " << crt_pem_file;

	BIO *bio = openssl::BIO_new_from_file(crt_pem_file);
	if (!bio)
		throw std::invalid_argument("Unable to open PEM certificate file");

	auto x509 = shared_ptr<X509>(PEM_read_bio_X509(bio, nullptr, nullptr, nullptr), X509_free);
	BIO_free(bio);
	if (!x509)
		throw std::invalid_argument("Unable to import PEM certificate from file");

	bio = openssl::BIO_new_from_file(key_pem_file);
	if (!bio)
		throw std::invalid_argument("Unable to open PEM key file");

	auto pkey = shared_ptr<EVP_PKEY>(
	    PEM_read_bio_PrivateKey(bio, nullptr, dummy_pass_cb, const_cast<char *>(pass.c_str())),
	    EVP_PKEY_free);
	BIO_free(bio);
	if (!pkey)
		throw std::invalid_argument("Unable to import PEM key from file");

	return Certificate(x509, pkey);
}

Certificate Certificate::Generate(CertificateType type, const string &commonName) {
	PLOG_DEBUG << "Generating certificate (OpenSSL)";

	shared_ptr<X509> x509(X509_new(), X509_free);
	unique_ptr<BIGNUM, decltype(&BN_free)> serial_number(BN_new(), BN_free);
	unique_ptr<X509_NAME, decltype(&X509_NAME_free)> name(X509_NAME_new(), X509_NAME_free);
	if (!x509 || !serial_number || !name)
		throw std::runtime_error("Unable to allocate structures for certificate generation");

	shared_ptr<EVP_PKEY> pkey;
	switch (type) {
	// RFC 8827 WebRTC Security Architecture 6.5. Communications Security
	// All implementations MUST support DTLS 1.2 with the TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
	// cipher suite and the P-256 curve
	// See https://www.rfc-editor.org/rfc/rfc8827.html#section-6.5
	case CertificateType::Default:
	case CertificateType::Ecdsa: {
		PLOG_VERBOSE << "Generating ECDSA P-256 key pair";
#if OPENSSL_VERSION_NUMBER >= 0x30000000
		pkey = shared_ptr<EVP_PKEY>(EVP_EC_gen("P-256"), EVP_PKEY_free);
#else
		pkey = shared_ptr<EVP_PKEY>(EVP_PKEY_new(), EVP_PKEY_free);
		unique_ptr<EC_KEY, decltype(&EC_KEY_free)> ecc(
		    EC_KEY_new_by_curve_name(NID_X9_62_prime256v1), EC_KEY_free);
		if (!pkey || !ecc)
			throw std::runtime_error("Unable to allocate structure for ECDSA P-256 key pair");

		EC_KEY_set_asn1_flag(ecc.get(), OPENSSL_EC_NAMED_CURVE); // Set ASN1 OID
		if (!EC_KEY_generate_key(ecc.get()) ||
		    !EVP_PKEY_assign_EC_KEY(pkey.get(),
		                            ecc.release())) // the key will be freed when pkey is freed
#endif
		if (!pkey)
			throw std::runtime_error("Unable to generate ECDSA P-256 key pair");

		break;
	}
	case CertificateType::Rsa: {
		PLOG_VERBOSE << "Generating RSA key pair";
		const unsigned int bits = 2048;
#if OPENSSL_VERSION_NUMBER >= 0x30000000
		pkey = shared_ptr<EVP_PKEY>(EVP_RSA_gen(bits), EVP_PKEY_free);
#else
		pkey = shared_ptr<EVP_PKEY>(EVP_PKEY_new(), EVP_PKEY_free);
		unique_ptr<RSA, decltype(&RSA_free)> rsa(RSA_new(), RSA_free);
		unique_ptr<BIGNUM, decltype(&BN_free)> exponent(BN_new(), BN_free);
		if (!pkey || !rsa || !exponent)
			throw std::runtime_error("Unable to allocate structures for RSA key pair");

		const unsigned int e = 65537; // 2^16 + 1
		if (!BN_set_word(exponent.get(), e) ||
		    !RSA_generate_key_ex(rsa.get(), bits, exponent.get(), NULL) ||
		    !EVP_PKEY_assign_RSA(pkey.get(),
		                         rsa.release())) // the key will be freed when pkey is freed
#endif
		if (!pkey)
			throw std::runtime_error("Unable to generate RSA key pair");

		break;
	}
	default:
		throw std::invalid_argument("Unknown certificate type");
	}

	const size_t serialSize = 16;
	auto *commonNameBytes =
	    reinterpret_cast<unsigned char *>(const_cast<char *>(commonName.c_str()));

	if (!X509_set_pubkey(x509.get(), pkey.get()))
		throw std::runtime_error("Unable to set certificate public key");

	if (!X509_gmtime_adj(X509_getm_notBefore(x509.get()), 3600 * -1) ||
	    !X509_gmtime_adj(X509_getm_notAfter(x509.get()), 3600 * 24 * 365) ||
	    !X509_set_version(x509.get(), 1) || !BN_rand(serial_number.get(), serialSize, 0, 0) ||
	    !BN_to_ASN1_INTEGER(serial_number.get(), X509_get_serialNumber(x509.get())) ||
	    !X509_NAME_add_entry_by_NID(name.get(), NID_commonName, MBSTRING_UTF8, commonNameBytes, -1,
	                                -1, 0) ||
	    !X509_set_subject_name(x509.get(), name.get()) ||
	    !X509_set_issuer_name(x509.get(), name.get()))
		throw std::runtime_error("Unable to set certificate properties");

	if (!X509_sign(x509.get(), pkey.get(), EVP_sha256()))
		throw std::runtime_error("Unable to auto-sign certificate");

	return Certificate(x509, pkey);
}

Certificate::Certificate(shared_ptr<X509> x509, shared_ptr<EVP_PKEY> pkey)
    : mX509(std::move(x509)), mPKey(std::move(pkey)), mFingerprint(make_fingerprint(mX509.get())) {}

std::tuple<X509 *, EVP_PKEY *> Certificate::credentials() const {
	return {mX509.get(), mPKey.get()};
}

string make_fingerprint(X509 *x509) {
	const size_t size = 32;
	unsigned char buffer[size];
	unsigned int len = size;
	if (!X509_digest(x509, EVP_sha256(), buffer, &len))
		throw std::runtime_error("X509 fingerprint error");

	std::ostringstream oss;
	oss << std::hex << std::uppercase << std::setfill('0');
	for (size_t i = 0; i < len; ++i) {
		if (i)
			oss << std::setw(1) << ':';
		oss << std::setw(2) << unsigned(buffer[i]);
	}
	return oss.str();
}

#endif

// Common for GnuTLS, Mbed TLS, and OpenSSL

future_certificate_ptr make_certificate(CertificateType type) {
	return ThreadPool::Instance().enqueue([type, token = Init::Instance().token()]() {
		return std::make_shared<Certificate>(Certificate::Generate(type, "libdatachannel"));
	});
}

string Certificate::fingerprint() const { return mFingerprint; }

} // namespace rtc::impl
