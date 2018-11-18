/*************************************************************************/
/*  stream_peer_openssl.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "stream_peer_mbed_tls.h"

#include "core/io/stream_peer_tcp.h"
#include "core/os/file_access.h"

#include <mbedtls/platform_util.h>

static void my_debug(void *ctx, int level,
		const char *file, int line,
		const char *str) {

	printf("%s:%04d: %s", file, line, str);
	fflush(stdout);
}

void _print_error(int ret) {
	printf("mbedtls error: returned -0x%x\n\n", -ret);
	fflush(stdout);
}

int StreamPeerMbedTLS::bio_send(void *ctx, const unsigned char *buf, size_t len) {

	if (buf == NULL || len <= 0) return 0;

	StreamPeerMbedTLS *sp = (StreamPeerMbedTLS *)ctx;

	ERR_FAIL_COND_V(sp == NULL, 0);

	int sent;
	Error err = sp->base->put_partial_data((const uint8_t *)buf, len, sent);
	if (err != OK) {
		return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
	}
	if (sent == 0) {
		return MBEDTLS_ERR_SSL_WANT_WRITE;
	}
	return sent;
}

int StreamPeerMbedTLS::bio_recv(void *ctx, unsigned char *buf, size_t len) {

	if (buf == NULL || len <= 0) return 0;

	StreamPeerMbedTLS *sp = (StreamPeerMbedTLS *)ctx;

	ERR_FAIL_COND_V(sp == NULL, 0);

	int got;
	Error err = sp->base->get_partial_data((uint8_t *)buf, len, got);
	if (err != OK) {
		return MBEDTLS_ERR_SSL_INTERNAL_ERROR;
	}
	if (got == 0) {
		return MBEDTLS_ERR_SSL_WANT_READ;
	}
	return got;
}

void StreamPeerMbedTLS::_cleanup() {

	mbedtls_ssl_free(&ssl);
	mbedtls_ssl_config_free(&conf);
	mbedtls_ctr_drbg_free(&ctr_drbg);
	mbedtls_entropy_free(&entropy);

	base = Ref<StreamPeer>();
	status = STATUS_DISCONNECTED;
}

Error StreamPeerMbedTLS::_do_handshake() {
	int ret = 0;
	while ((ret = mbedtls_ssl_handshake(&ssl)) != 0) {
		if (ret != MBEDTLS_ERR_SSL_WANT_READ && ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
			// An error occurred.
			ERR_PRINTS("TLS handshake error: " + itos(ret));
			_print_error(ret);
			disconnect_from_stream();
			status = STATUS_ERROR;
			return FAILED;
		}

		// Handshake is still in progress.
		if (!blocking_handshake) {
			// Will retry via poll later
			return OK;
		}
	}

	status = STATUS_CONNECTED;
	return OK;
}

Error StreamPeerMbedTLS::connect_to_stream(Ref<StreamPeer> p_base, bool p_validate_certs, const String &p_for_hostname) {

	base = p_base;
	int ret = 0;
	int authmode = p_validate_certs ? MBEDTLS_SSL_VERIFY_REQUIRED : MBEDTLS_SSL_VERIFY_NONE;

	mbedtls_ssl_init(&ssl);
	mbedtls_ssl_config_init(&conf);
	mbedtls_ctr_drbg_init(&ctr_drbg);
	mbedtls_entropy_init(&entropy);

	ret = mbedtls_ctr_drbg_seed(&ctr_drbg, mbedtls_entropy_func, &entropy, NULL, 0);
	if (ret != 0) {
		ERR_PRINTS(" failed\n  ! mbedtls_ctr_drbg_seed returned an error" + itos(ret));
		_cleanup();
		return FAILED;
	}

	mbedtls_ssl_config_defaults(&conf,
			MBEDTLS_SSL_IS_CLIENT,
			MBEDTLS_SSL_TRANSPORT_STREAM,
			MBEDTLS_SSL_PRESET_DEFAULT);

	mbedtls_ssl_conf_authmode(&conf, authmode);
	mbedtls_ssl_conf_ca_chain(&conf, &cacert, NULL);
	mbedtls_ssl_conf_rng(&conf, mbedtls_ctr_drbg_random, &ctr_drbg);
	mbedtls_ssl_conf_dbg(&conf, my_debug, stdout);
	mbedtls_ssl_setup(&ssl, &conf);
	mbedtls_ssl_set_hostname(&ssl, p_for_hostname.utf8().get_data());

	mbedtls_ssl_set_bio(&ssl, this, bio_send, bio_recv, NULL);

	status = STATUS_HANDSHAKING;

	if ((ret = _do_handshake()) != OK) {
		status = STATUS_ERROR_HOSTNAME_MISMATCH;
		return FAILED;
	}

	return OK;
}

Error StreamPeerMbedTLS::accept_stream(Ref<StreamPeer> p_base) {

	return OK;
}

Error StreamPeerMbedTLS::put_data(const uint8_t *p_data, int p_bytes) {

	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_UNCONFIGURED);

	Error err;
	int sent = 0;

	while (p_bytes > 0) {
		err = put_partial_data(p_data, p_bytes, sent);

		if (err != OK) {
			return err;
		}

		p_data += sent;
		p_bytes -= sent;
	}

	return OK;
}

Error StreamPeerMbedTLS::put_partial_data(const uint8_t *p_data, int p_bytes, int &r_sent) {

	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_UNCONFIGURED);

	r_sent = 0;

	if (p_bytes == 0)
		return OK;

	int ret = mbedtls_ssl_write(&ssl, p_data, p_bytes);
	if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
		// Non blocking IO
		ret = 0;
	} else if (ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) {
		// Clean close
		disconnect_from_stream();
		return ERR_FILE_EOF;
	} else if (ret <= 0) {
		_print_error(ret);
		disconnect_from_stream();
		return ERR_CONNECTION_ERROR;
	}

	r_sent = ret;
	return OK;
}

Error StreamPeerMbedTLS::get_data(uint8_t *p_buffer, int p_bytes) {

	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_UNCONFIGURED);

	Error err;

	int got = 0;
	while (p_bytes > 0) {

		err = get_partial_data(p_buffer, p_bytes, got);

		if (err != OK) {
			return err;
		}

		p_buffer += got;
		p_bytes -= got;
	}

	return OK;
}

Error StreamPeerMbedTLS::get_partial_data(uint8_t *p_buffer, int p_bytes, int &r_received) {

	ERR_FAIL_COND_V(status != STATUS_CONNECTED, ERR_UNCONFIGURED);

	r_received = 0;

	int ret = mbedtls_ssl_read(&ssl, p_buffer, p_bytes);
	if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
		ret = 0; // non blocking io
	} else if (ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) {
		// Clean close
		disconnect_from_stream();
		return ERR_FILE_EOF;
	} else if (ret <= 0) {
		_print_error(ret);
		disconnect_from_stream();
		return ERR_CONNECTION_ERROR;
	}

	r_received = ret;
	return OK;
}

void StreamPeerMbedTLS::poll() {

	ERR_FAIL_COND(status != STATUS_CONNECTED && status != STATUS_HANDSHAKING);
	ERR_FAIL_COND(!base.is_valid());

	if (status == STATUS_HANDSHAKING) {
		_do_handshake();
		return;
	}

	int ret = mbedtls_ssl_read(&ssl, NULL, 0);

	if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
		// Nothing to read/write (non blocking IO)
	} else if (ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) {
		// Clean close (disconnect)
		disconnect_from_stream();
		return;
	} else if (ret < 0) {
		_print_error(ret);
		disconnect_from_stream();
		return;
	}

	Ref<StreamPeerTCP> tcp = base;
	if (tcp.is_valid() && tcp->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
		disconnect_from_stream();
		return;
	}
}

int StreamPeerMbedTLS::get_available_bytes() const {

	ERR_FAIL_COND_V(status != STATUS_CONNECTED, 0);

	return mbedtls_ssl_get_bytes_avail(&ssl);
}
StreamPeerMbedTLS::StreamPeerMbedTLS() {

	status = STATUS_DISCONNECTED;
}

StreamPeerMbedTLS::~StreamPeerMbedTLS() {
	disconnect_from_stream();
}

void StreamPeerMbedTLS::disconnect_from_stream() {

	if (status != STATUS_CONNECTED && status != STATUS_HANDSHAKING)
		return;

	Ref<StreamPeerTCP> tcp = base;
	if (tcp.is_valid() && tcp->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
		// We are still connected on the socket, try to send close notity.
		mbedtls_ssl_close_notify(&ssl);
	}

	_cleanup();
}

StreamPeerMbedTLS::Status StreamPeerMbedTLS::get_status() const {

	return status;
}

StreamPeerSSL *StreamPeerMbedTLS::_create_func() {

	return memnew(StreamPeerMbedTLS);
}

mbedtls_x509_crt StreamPeerMbedTLS::cacert;

void StreamPeerMbedTLS::_load_certs(const PoolByteArray &p_array) {
	int arr_len = p_array.size();
	PoolByteArray::Read r = p_array.read();
	int err = mbedtls_x509_crt_parse(&cacert, &r[0], arr_len);
	if (err != 0) {
		WARN_PRINTS("Error parsing some certificates: " + itos(err));
	}
}

void StreamPeerMbedTLS::initialize_ssl() {

	_create = _create_func;
	load_certs_func = _load_certs;

	mbedtls_x509_crt_init(&cacert);

#ifdef DEBUG_ENABLED
	mbedtls_debug_set_threshold(1);
#endif

	available = true;
}

void StreamPeerMbedTLS::finalize_ssl() {

	available = false;
	_create = NULL;
	load_certs_func = NULL;
	mbedtls_x509_crt_free(&cacert);
}
