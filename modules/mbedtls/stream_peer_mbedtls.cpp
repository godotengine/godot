/*************************************************************************/
/*  stream_peer_mbedtls.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "stream_peer_mbedtls.h"

#include "core/io/stream_peer_tcp.h"
#include "core/os/file_access.h"

int StreamPeerMbedTLS::bio_send(void *ctx, const unsigned char *buf, size_t len) {
	if (buf == nullptr || len <= 0) {
		return 0;
	}

	StreamPeerMbedTLS *sp = (StreamPeerMbedTLS *)ctx;

	ERR_FAIL_COND_V(sp == nullptr, 0);

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
	if (buf == nullptr || len <= 0) {
		return 0;
	}

	StreamPeerMbedTLS *sp = (StreamPeerMbedTLS *)ctx;

	ERR_FAIL_COND_V(sp == nullptr, 0);

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
	ssl_ctx->clear();
	base = Ref<StreamPeer>();
	status = STATUS_DISCONNECTED;
}

Error StreamPeerMbedTLS::_do_handshake() {
	int ret = 0;
	while ((ret = mbedtls_ssl_handshake(ssl_ctx->get_context())) != 0) {
		if (ret != MBEDTLS_ERR_SSL_WANT_READ && ret != MBEDTLS_ERR_SSL_WANT_WRITE) {
			// An error occurred.
			ERR_PRINT("TLS handshake error: " + itos(ret));
			SSLContextMbedTLS::print_mbedtls_error(ret);
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

Error StreamPeerMbedTLS::connect_to_stream(Ref<StreamPeer> p_base, bool p_validate_certs, const String &p_for_hostname, Ref<X509Certificate> p_ca_certs) {
	ERR_FAIL_COND_V(p_base.is_null(), ERR_INVALID_PARAMETER);

	base = p_base;
	int authmode = p_validate_certs ? MBEDTLS_SSL_VERIFY_REQUIRED : MBEDTLS_SSL_VERIFY_NONE;

	Error err = ssl_ctx->init_client(MBEDTLS_SSL_TRANSPORT_STREAM, authmode, p_ca_certs);
	ERR_FAIL_COND_V(err != OK, err);

	mbedtls_ssl_set_hostname(ssl_ctx->get_context(), p_for_hostname.utf8().get_data());
	mbedtls_ssl_set_bio(ssl_ctx->get_context(), this, bio_send, bio_recv, nullptr);

	status = STATUS_HANDSHAKING;

	if (_do_handshake() != OK) {
		status = STATUS_ERROR_HOSTNAME_MISMATCH;
		return FAILED;
	}

	return OK;
}

Error StreamPeerMbedTLS::accept_stream(Ref<StreamPeer> p_base, Ref<CryptoKey> p_key, Ref<X509Certificate> p_cert, Ref<X509Certificate> p_ca_chain) {
	ERR_FAIL_COND_V(p_base.is_null(), ERR_INVALID_PARAMETER);

	Error err = ssl_ctx->init_server(MBEDTLS_SSL_TRANSPORT_STREAM, MBEDTLS_SSL_VERIFY_NONE, p_key, p_cert);
	ERR_FAIL_COND_V(err != OK, err);

	base = p_base;

	mbedtls_ssl_set_bio(ssl_ctx->get_context(), this, bio_send, bio_recv, nullptr);

	status = STATUS_HANDSHAKING;

	if (_do_handshake() != OK) {
		return FAILED;
	}

	status = STATUS_CONNECTED;
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

	if (p_bytes == 0) {
		return OK;
	}

	int ret = mbedtls_ssl_write(ssl_ctx->get_context(), p_data, p_bytes);
	if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
		// Non blocking IO
		ret = 0;
	} else if (ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) {
		// Clean close
		disconnect_from_stream();
		return ERR_FILE_EOF;
	} else if (ret <= 0) {
		SSLContextMbedTLS::print_mbedtls_error(ret);
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

	int ret = mbedtls_ssl_read(ssl_ctx->get_context(), p_buffer, p_bytes);
	if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
		ret = 0; // non blocking io
	} else if (ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) {
		// Clean close
		disconnect_from_stream();
		return ERR_FILE_EOF;
	} else if (ret <= 0) {
		SSLContextMbedTLS::print_mbedtls_error(ret);
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

	// We could pass nullptr as second parameter, but some behaviour sanitizers don't seem to like that.
	// Passing a 1 byte buffer to workaround it.
	uint8_t byte;
	int ret = mbedtls_ssl_read(ssl_ctx->get_context(), &byte, 0);

	if (ret == MBEDTLS_ERR_SSL_WANT_READ || ret == MBEDTLS_ERR_SSL_WANT_WRITE) {
		// Nothing to read/write (non blocking IO)
	} else if (ret == MBEDTLS_ERR_SSL_PEER_CLOSE_NOTIFY) {
		// Clean close (disconnect)
		disconnect_from_stream();
		return;
	} else if (ret < 0) {
		SSLContextMbedTLS::print_mbedtls_error(ret);
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

	return mbedtls_ssl_get_bytes_avail(&(ssl_ctx->ssl));
}

StreamPeerMbedTLS::StreamPeerMbedTLS() {
	ssl_ctx.instance();
}

StreamPeerMbedTLS::~StreamPeerMbedTLS() {
	disconnect_from_stream();
}

void StreamPeerMbedTLS::disconnect_from_stream() {
	if (status != STATUS_CONNECTED && status != STATUS_HANDSHAKING) {
		return;
	}

	Ref<StreamPeerTCP> tcp = base;
	if (tcp.is_valid() && tcp->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
		// We are still connected on the socket, try to send close notify.
		mbedtls_ssl_close_notify(ssl_ctx->get_context());
	}

	_cleanup();
}

StreamPeerMbedTLS::Status StreamPeerMbedTLS::get_status() const {
	return status;
}

StreamPeerSSL *StreamPeerMbedTLS::_create_func() {
	return memnew(StreamPeerMbedTLS);
}

void StreamPeerMbedTLS::initialize_ssl() {
	_create = _create_func;
	available = true;
}

void StreamPeerMbedTLS::finalize_ssl() {
	available = false;
	_create = nullptr;
}
