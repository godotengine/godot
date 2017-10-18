/*************************************************************************/
/*  stream_peer_openssl.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef STREAM_PEER_OPEN_SSL_H
#define STREAM_PEER_OPEN_SSL_H

#include "io/stream_peer_ssl.h"
#include "os/file_access.h"
#include "project_settings.h"

#include "thirdparty/misc/curl_hostcheck.h"

#include <openssl/bio.h> // BIO objects for I/O
#include <openssl/err.h> // Error reporting
#include <openssl/ssl.h> // SSL and SSL_CTX for SSL connections
#include <openssl/x509v3.h>

#include <stdio.h>

class StreamPeerOpenSSL : public StreamPeerSSL {
private:
	static int _bio_create(BIO *b);
	static int _bio_destroy(BIO *b);
	static int _bio_read(BIO *b, char *buf, int len);
	static int _bio_write(BIO *b, const char *buf, int len);
	static long _bio_ctrl(BIO *b, int cmd, long num, void *ptr);
	static int _bio_gets(BIO *b, char *buf, int len);
	static int _bio_puts(BIO *b, const char *str);

#if OPENSSL_VERSION_NUMBER >= 0x10100000L && !defined(LIBRESSL_VERSION_NUMBER)
	static BIO_METHOD *_bio_method;
#else
	static BIO_METHOD _bio_method;
#endif
	static BIO_METHOD *_get_bio_method();

	static bool _match_host_name(const char *name, const char *hostname);
	static Error _match_common_name(const char *hostname, const X509 *server_cert);
	static Error _match_subject_alternative_name(const char *hostname, const X509 *server_cert);

	static int _cert_verify_callback(X509_STORE_CTX *x509_ctx, void *arg);

	Status status;
	String hostname;
	int max_cert_chain_depth;
	SSL_CTX *ctx;
	SSL *ssl;
	BIO *bio;
	bool connected;
	int flags;
	bool use_blocking;
	bool validate_certs;
	bool validate_hostname;

	Ref<StreamPeer> base;

	static StreamPeerSSL *_create_func();
	void _print_error(int err);

	static Vector<X509 *> certs;

	static void _load_certs(const PoolByteArray &p_array);

protected:
	static void _bind_methods();

public:
	virtual Error accept_stream(Ref<StreamPeer> p_base);
	virtual Error connect_to_stream(Ref<StreamPeer> p_base, bool p_validate_certs = false, const String &p_for_hostname = String());
	virtual Status get_status() const;

	virtual void disconnect_from_stream();

	virtual Error put_data(const uint8_t *p_data, int p_bytes);
	virtual Error put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent);

	virtual Error get_data(uint8_t *p_buffer, int p_bytes);
	virtual Error get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received);

	virtual int get_available_bytes() const;

	static void initialize_ssl();
	static void finalize_ssl();

	StreamPeerOpenSSL();
	~StreamPeerOpenSSL();
};

#endif // STREAM_PEER_SSL_H
