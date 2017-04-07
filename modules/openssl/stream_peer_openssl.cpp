/*************************************************************************/
/*  stream_peer_openssl.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "stream_peer_openssl.h"
//hostname matching code from curl

//#include <openssl/applink.c> // To prevent crashing (see the OpenSSL FAQ)

bool StreamPeerOpenSSL::_match_host_name(const char *name, const char *hostname) {

	return Tool_Curl_cert_hostcheck(name, hostname) == CURL_HOST_MATCH;
	//print_line("MATCH: "+String(name)+" vs "+String(hostname));
	//return true;
}

Error StreamPeerOpenSSL::_match_common_name(const char *hostname, const X509 *server_cert) {

	int common_name_loc = -1;
	X509_NAME_ENTRY *common_name_entry = NULL;
	ASN1_STRING *common_name_asn1 = NULL;
	char *common_name_str = NULL;

	// Find the position of the CN field in the Subject field of the certificate
	common_name_loc = X509_NAME_get_index_by_NID(X509_get_subject_name((X509 *)server_cert), NID_commonName, -1);

	ERR_FAIL_COND_V(common_name_loc < 0, ERR_INVALID_PARAMETER);

	// Extract the CN field
	common_name_entry = X509_NAME_get_entry(X509_get_subject_name((X509 *)server_cert), common_name_loc);

	ERR_FAIL_COND_V(common_name_entry == NULL, ERR_INVALID_PARAMETER);

	// Convert the CN field to a C string
	common_name_asn1 = X509_NAME_ENTRY_get_data(common_name_entry);

	ERR_FAIL_COND_V(common_name_asn1 == NULL, ERR_INVALID_PARAMETER);

	common_name_str = (char *)ASN1_STRING_data(common_name_asn1);

	// Make sure there isn't an embedded NUL character in the CN
	bool malformed_certificate = (size_t)ASN1_STRING_length(common_name_asn1) != strlen(common_name_str);

	ERR_FAIL_COND_V(malformed_certificate, ERR_INVALID_PARAMETER);

	// Compare expected hostname with the CN

	return _match_host_name(common_name_str, hostname) ? OK : FAILED;
}

/**
* Tries to find a match for hostname in the certificate's Subject Alternative Name extension.
*
*/

Error StreamPeerOpenSSL::_match_subject_alternative_name(const char *hostname, const X509 *server_cert) {

	Error result = FAILED;
	int i;
	int san_names_nb = -1;
	STACK_OF(GENERAL_NAME) *san_names = NULL;

	// Try to extract the names within the SAN extension from the certificate
	san_names = (STACK_OF(GENERAL_NAME) *)X509_get_ext_d2i((X509 *)server_cert, NID_subject_alt_name, NULL, NULL);
	if (san_names == NULL) {
		return ERR_FILE_NOT_FOUND;
	}
	san_names_nb = sk_GENERAL_NAME_num(san_names);

	// Check each name within the extension
	for (i = 0; i < san_names_nb; i++) {
		const GENERAL_NAME *current_name = sk_GENERAL_NAME_value(san_names, i);

		if (current_name->type == GEN_DNS) {
			// Current name is a DNS name, let's check it
			char *dns_name = (char *)ASN1_STRING_data(current_name->d.dNSName);

			// Make sure there isn't an embedded NUL character in the DNS name
			if ((size_t)ASN1_STRING_length(current_name->d.dNSName) != strlen(dns_name)) {
				result = ERR_INVALID_PARAMETER;
				break;
			} else { // Compare expected hostname with the DNS name
				if (_match_host_name(dns_name, hostname)) {
					result = OK;
					break;
				}
			}
		}
	}
	sk_GENERAL_NAME_pop_free(san_names, GENERAL_NAME_free);

	return result;
}

/* See http://archives.seul.org/libevent/users/Jan-2013/msg00039.html */
int StreamPeerOpenSSL::_cert_verify_callback(X509_STORE_CTX *x509_ctx, void *arg) {

	/* This is the function that OpenSSL would call if we hadn't called
	 * SSL_CTX_set_cert_verify_callback().  Therefore, we are "wrapping"
	 * the default functionality, rather than replacing it. */

	bool base_cert_valid = X509_verify_cert(x509_ctx);
	if (!base_cert_valid) {
		print_line("Cause: " + String(X509_verify_cert_error_string(X509_STORE_CTX_get_error(x509_ctx))));
		ERR_print_errors_fp(stdout);
	}
	X509 *server_cert = X509_STORE_CTX_get_current_cert(x509_ctx);

	ERR_FAIL_COND_V(!server_cert, 0);

	char cert_str[256];
	X509_NAME_oneline(X509_get_subject_name(server_cert),
			cert_str, sizeof(cert_str));

	print_line("CERT STR: " + String(cert_str));
	print_line("VALID: " + itos(base_cert_valid));

	if (!base_cert_valid)
		return 0;

	StreamPeerOpenSSL *ssl = (StreamPeerOpenSSL *)arg;

	if (ssl->validate_hostname) {

		Error err = _match_subject_alternative_name(ssl->hostname.utf8().get_data(), server_cert);

		if (err == ERR_FILE_NOT_FOUND) {

			err = _match_common_name(ssl->hostname.utf8().get_data(), server_cert);
		}

		if (err != OK) {

			ssl->status = STATUS_ERROR_HOSTNAME_MISMATCH;
			return 0;
		}
	}

	return 1;
}

int StreamPeerOpenSSL::_bio_create(BIO *b) {
	b->init = 1;
	b->num = 0;
	b->ptr = NULL;
	b->flags = 0;
	return 1;
}

int StreamPeerOpenSSL::_bio_destroy(BIO *b) {
	if (b == NULL)
		return 0;

	b->ptr = NULL; /* sb_tls_remove() will free it */
	b->init = 0;
	b->flags = 0;
	return 1;
}

int StreamPeerOpenSSL::_bio_read(BIO *b, char *buf, int len) {

	if (buf == NULL || len <= 0) return 0;

	StreamPeerOpenSSL *sp = (StreamPeerOpenSSL *)b->ptr;

	ERR_FAIL_COND_V(sp == NULL, 0);

	BIO_clear_retry_flags(b);
	if (sp->use_blocking) {

		Error err = sp->base->get_data((uint8_t *)buf, len);
		if (err != OK) {
			return -1;
		}

		return len;
	} else {

		int got;
		Error err = sp->base->get_partial_data((uint8_t *)buf, len, got);
		if (err != OK) {
			return -1;
		}
		if (got == 0) {
			BIO_set_retry_read(b);
		}
		return got;
	}

	//unreachable
	return 0;
}

int StreamPeerOpenSSL::_bio_write(BIO *b, const char *buf, int len) {

	if (buf == NULL || len <= 0) return 0;

	StreamPeerOpenSSL *sp = (StreamPeerOpenSSL *)b->ptr;

	ERR_FAIL_COND_V(sp == NULL, 0);

	BIO_clear_retry_flags(b);
	if (sp->use_blocking) {

		Error err = sp->base->put_data((const uint8_t *)buf, len);
		if (err != OK) {
			return -1;
		}

		return len;
	} else {

		int sent;
		Error err = sp->base->put_partial_data((const uint8_t *)buf, len, sent);
		if (err != OK) {
			return -1;
		}
		if (sent == 0) {
			BIO_set_retry_write(b);
		}
		return sent;
	}

	//unreachable
	return 0;
}

long StreamPeerOpenSSL::_bio_ctrl(BIO *b, int cmd, long num, void *ptr) {
	if (cmd == BIO_CTRL_FLUSH) {
		/* The OpenSSL library needs this */
		return 1;
	}
	return 0;
}

int StreamPeerOpenSSL::_bio_gets(BIO *b, char *buf, int len) {
	return -1;
}

int StreamPeerOpenSSL::_bio_puts(BIO *b, const char *str) {
	return _bio_write(b, str, strlen(str));
}

BIO_METHOD StreamPeerOpenSSL::_bio_method = {
	/* it's a source/sink BIO */
	(100 | 0x400),
	"streampeer glue",
	_bio_write,
	_bio_read,
	_bio_puts,
	_bio_gets,
	_bio_ctrl,
	_bio_create,
	_bio_destroy
};

Error StreamPeerOpenSSL::connect_to_stream(Ref<StreamPeer> p_base, bool p_validate_certs, const String &p_for_hostname) {

	if (connected)
		disconnect_from_stream();

	hostname = p_for_hostname;
	status = STATUS_DISCONNECTED;

	// Set up a SSL_CTX object, which will tell our BIO object how to do its work
	ctx = SSL_CTX_new(SSLv23_client_method());
	base = p_base;
	validate_certs = p_validate_certs;
	validate_hostname = p_for_hostname != "";

	if (p_validate_certs) {

		if (certs.size()) {
			//yay for undocumented OpenSSL functions

			X509_STORE *store = SSL_CTX_get_cert_store(ctx);
			for (int i = 0; i < certs.size(); i++) {

				X509_STORE_add_cert(store, certs[i]);
			}
#if 0
			const unsigned char *in=(const unsigned char *)certs.ptr();
			X509 *Cert = d2i_X509(NULL, &in, certs.size()-1);
			if (!Cert) {
				print_line(String(ERR_error_string(ERR_get_error(),NULL)));
			}
			ERR_FAIL_COND_V(!Cert,ERR_PARSE_ERROR);

			X509_STORE *store = SSL_CTX_get_cert_store(ctx);
			X509_STORE_add_cert(store,Cert);

			//char *str = X509_NAME_oneline(X509_get_subject_name(Cert),0,0);
			//printf ("subject: %s\n", str); /* [1] */
#endif
		}

		//used for testing
		//int res = SSL_CTX_load_verify_locations(ctx,"/etc/ssl/certs/ca-certificates.crt",NULL);
		//print_line("verify locations res: "+itos(res));

		/* Ask OpenSSL to verify the server certificate.  Note that this
		 * does NOT include verifying that the hostname is correct.
		 * So, by itself, this means anyone with any legitimate
		 * CA-issued certificate for any website, can impersonate any
		 * other website in the world.  This is not good.  See "The
		 * Most Dangerous Code in the World" article at
		 * https://crypto.stanford.edu/~dabo/pubs/abstracts/ssl-client-bugs.html
		 */
		SSL_CTX_set_verify(ctx, SSL_VERIFY_PEER, NULL);
		/* This is how we solve the problem mentioned in the previous
		 * comment.  We "wrap" OpenSSL's validation routine in our
		 * own routine, which also validates the hostname by calling
		 * the code provided by iSECPartners.  Note that even though
		 * the "Everything You've Always Wanted to Know About
		 * Certificate Validation With OpenSSL (But Were Afraid to
		 * Ask)" paper from iSECPartners says very explicitly not to
		 * call SSL_CTX_set_cert_verify_callback (at the bottom of
		 * page 2), what we're doing here is safe because our
		 * cert_verify_callback() calls X509_verify_cert(), which is
		 * OpenSSL's built-in routine which would have been called if
		 * we hadn't set the callback.  Therefore, we're just
		 * "wrapping" OpenSSL's routine, not replacing it. */
		SSL_CTX_set_cert_verify_callback(ctx, _cert_verify_callback, this);

		//Let the verify_callback catch the verify_depth error so that we get an appropriate error in the logfile. (??)
		SSL_CTX_set_verify_depth(ctx, max_cert_chain_depth + 1);
	}

	ssl = SSL_new(ctx);
	bio = BIO_new(&_bio_method);
	bio->ptr = this;
	SSL_set_bio(ssl, bio, bio);

	if (p_for_hostname != String()) {
		SSL_set_tlsext_host_name(ssl, p_for_hostname.utf8().get_data());
	}

	use_blocking = true; // let handshake use blocking
	// Set the SSL to automatically retry on failure.
	SSL_set_mode(ssl, SSL_MODE_AUTO_RETRY);

	// Same as before, try to connect.
	int result = SSL_connect(ssl);

	print_line("CONNECTION RESULT: " + itos(result));
	if (result < 1) {
		ERR_print_errors_fp(stdout);
		_print_error(result);
	}

	X509 *peer = SSL_get_peer_certificate(ssl);

	if (peer) {
		bool cert_ok = SSL_get_verify_result(ssl) == X509_V_OK;
		print_line("cert_ok: " + itos(cert_ok));

	} else if (validate_certs) {
		status = STATUS_ERROR_NO_CERTIFICATE;
	}

	connected = true;
	status = STATUS_CONNECTED;

	return OK;
}

Error StreamPeerOpenSSL::accept_stream(Ref<StreamPeer> p_base) {

	return ERR_UNAVAILABLE;
}

void StreamPeerOpenSSL::_print_error(int err) {

	err = SSL_get_error(ssl, err);
	switch (err) {
		case SSL_ERROR_NONE: ERR_PRINT("NO ERROR: The TLS/SSL I/O operation completed"); break;
		case SSL_ERROR_ZERO_RETURN: ERR_PRINT("The TLS/SSL connection has been closed.");
		case SSL_ERROR_WANT_READ:
		case SSL_ERROR_WANT_WRITE:
			ERR_PRINT("The operation did not complete.");
			break;
		case SSL_ERROR_WANT_CONNECT:
		case SSL_ERROR_WANT_ACCEPT:
			ERR_PRINT("The connect/accept operation did not complete");
			break;
		case SSL_ERROR_WANT_X509_LOOKUP:
			ERR_PRINT("The operation did not complete because an application callback set by SSL_CTX_set_client_cert_cb() has asked to be called again.");
			break;
		case SSL_ERROR_SYSCALL:
			ERR_PRINT("Some I/O error occurred. The OpenSSL error queue may contain more information on the error.");
			break;
		case SSL_ERROR_SSL:
			ERR_PRINT("A failure in the SSL library occurred, usually a protocol error.");
			break;
	}
}

Error StreamPeerOpenSSL::put_data(const uint8_t *p_data, int p_bytes) {

	ERR_FAIL_COND_V(!connected, ERR_UNCONFIGURED);

	while (p_bytes > 0) {
		int ret = SSL_write(ssl, p_data, p_bytes);
		if (ret <= 0) {
			_print_error(ret);
			disconnect_from_stream();
			return ERR_CONNECTION_ERROR;
		}
		p_data += ret;
		p_bytes -= ret;
	}

	return OK;
}

Error StreamPeerOpenSSL::put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) {

	ERR_FAIL_COND_V(!connected, ERR_UNCONFIGURED);
	if (p_bytes == 0)
		return OK;

	Error err = put_data(p_data, p_bytes);
	if (err != OK)
		return err;

	r_sent = p_bytes;
	return OK;
}

Error StreamPeerOpenSSL::get_data(uint8_t *p_buffer, int p_bytes) {

	ERR_FAIL_COND_V(!connected, ERR_UNCONFIGURED);

	while (p_bytes > 0) {

		int ret = SSL_read(ssl, p_buffer, p_bytes);
		if (ret <= 0) {
			_print_error(ret);
			disconnect_from_stream();
			return ERR_CONNECTION_ERROR;
		}
		p_buffer += ret;
		p_bytes -= ret;
	}

	return OK;
}

Error StreamPeerOpenSSL::get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) {

	ERR_FAIL_COND_V(!connected, ERR_UNCONFIGURED);
	if (p_bytes == 0) {
		r_received = 0;
		return OK;
	}

	Error err = get_data(p_buffer, p_bytes);
	if (err != OK)
		return err;
	r_received = p_bytes;
	return OK;
}

int StreamPeerOpenSSL::get_available_bytes() const {

	ERR_FAIL_COND_V(!connected, 0);

	return SSL_pending(ssl);
}
StreamPeerOpenSSL::StreamPeerOpenSSL() {

	ctx = NULL;
	ssl = NULL;
	bio = NULL;
	connected = false;
	use_blocking = true; //might be improved int the future, but for now it always blocks
	max_cert_chain_depth = 9;
	flags = 0;
}

void StreamPeerOpenSSL::disconnect_from_stream() {

	if (!connected)
		return;
	SSL_shutdown(ssl);
	SSL_free(ssl);
	SSL_CTX_free(ctx);
	base = Ref<StreamPeer>();
	connected = false;
	validate_certs = false;
	validate_hostname = false;
	status = STATUS_DISCONNECTED;
}

StreamPeerOpenSSL::Status StreamPeerOpenSSL::get_status() const {

	return status;
}

StreamPeerOpenSSL::~StreamPeerOpenSSL() {
	disconnect_from_stream();
}

StreamPeerSSL *StreamPeerOpenSSL::_create_func() {

	return memnew(StreamPeerOpenSSL);
}

Vector<X509 *> StreamPeerOpenSSL::certs;

void StreamPeerOpenSSL::_load_certs(const PoolByteArray &p_array) {

	PoolByteArray::Read r = p_array.read();
	BIO *mem = BIO_new(BIO_s_mem());
	BIO_puts(mem, (const char *)r.ptr());
	while (true) {
		X509 *cert = PEM_read_bio_X509(mem, NULL, 0, NULL);
		if (!cert)
			break;
		certs.push_back(cert);
	}
	BIO_free(mem);
}

void StreamPeerOpenSSL::initialize_ssl() {

	available = true;

	load_certs_func = _load_certs;

	_create = _create_func;
	CRYPTO_malloc_init(); // Initialize malloc, free, etc for OpenSSL's use
	SSL_library_init(); // Initialize OpenSSL's SSL libraries
	SSL_load_error_strings(); // Load SSL error strings
	ERR_load_BIO_strings(); // Load BIO error strings
	OpenSSL_add_all_algorithms(); // Load all available encryption algorithms
	String certs_path = GLOBAL_DEF("network/ssl/certificates", "");
	GlobalConfig::get_singleton()->set_custom_property_info("network/ssl/certificates", PropertyInfo(Variant::STRING, "network/ssl/certificates", PROPERTY_HINT_FILE, "*.crt"));
	if (certs_path != "") {

		FileAccess *f = FileAccess::open(certs_path, FileAccess::READ);
		if (f) {
			PoolByteArray arr;
			int flen = f->get_len();
			arr.resize(flen + 1);
			{
				PoolByteArray::Write w = arr.write();
				f->get_buffer(w.ptr(), flen);
				w[flen] = 0; //end f string
			}

			memdelete(f);

			_load_certs(arr);
			print_line("Loaded certs from '" + certs_path + "':  " + itos(certs.size()));
		}
	}
	String config_path = GLOBAL_DEF("network/ssl/config", "");
	GlobalConfig::get_singleton()->set_custom_property_info("network/ssl/config", PropertyInfo(Variant::STRING, "network/ssl/config", PROPERTY_HINT_FILE, "*.cnf"));
	if (config_path != "") {

		Vector<uint8_t> data = FileAccess::get_file_as_array(config_path);
		if (data.size()) {
			data.push_back(0);
			BIO *mem = BIO_new(BIO_s_mem());
			BIO_puts(mem, (const char *)data.ptr());

			while (true) {
				X509 *cert = PEM_read_bio_X509(mem, NULL, 0, NULL);
				if (!cert)
					break;
				certs.push_back(cert);
			}
			BIO_free(mem);
		}
		print_line("Loaded certs from '" + certs_path + "':  " + itos(certs.size()));
	}
}

void StreamPeerOpenSSL::finalize_ssl() {

	for (int i = 0; i < certs.size(); i++) {
		X509_free(certs[i]);
	}
	certs.clear();
}
