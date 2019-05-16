/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 */

#include "core/private.h"

#if defined(LWS_WITH_MBEDTLS) || (defined(OPENSSL_VERSION_NUMBER) && \
				  OPENSSL_VERSION_NUMBER >= 0x10002000L)
static int
alpn_cb(SSL *s, const unsigned char **out, unsigned char *outlen,
	const unsigned char *in, unsigned int inlen, void *arg)
{
#if !defined(LWS_WITH_MBEDTLS)
	struct alpn_ctx *alpn_ctx = (struct alpn_ctx *)arg;

	if (SSL_select_next_proto((unsigned char **)out, outlen, alpn_ctx->data,
				  alpn_ctx->len, in, inlen) !=
	    OPENSSL_NPN_NEGOTIATED)
		return SSL_TLSEXT_ERR_NOACK;
#endif

	return SSL_TLSEXT_ERR_OK;
}
#endif

void
lws_context_init_alpn(struct lws_vhost *vhost)
{
#if defined(LWS_WITH_MBEDTLS) || (defined(OPENSSL_VERSION_NUMBER) && \
				  OPENSSL_VERSION_NUMBER >= 0x10002000L)
	const char *alpn_comma = vhost->context->tls.alpn_default;

	if (vhost->tls.alpn)
		alpn_comma = vhost->tls.alpn;

	lwsl_info(" Server '%s' advertising ALPN: %s\n",
		    vhost->name, alpn_comma);
	vhost->tls.alpn_ctx.len = lws_alpn_comma_to_openssl(alpn_comma,
					vhost->tls.alpn_ctx.data,
					sizeof(vhost->tls.alpn_ctx.data) - 1);

	SSL_CTX_set_alpn_select_cb(vhost->tls.ssl_ctx, alpn_cb, &vhost->tls.alpn_ctx);
#else
	lwsl_err(
		" HTTP2 / ALPN configured but not supported by OpenSSL 0x%lx\n",
		    OPENSSL_VERSION_NUMBER);
#endif // OPENSSL_VERSION_NUMBER >= 0x10002000L
}

int
lws_tls_server_conn_alpn(struct lws *wsi)
{
#if defined(LWS_WITH_MBEDTLS) || (defined(OPENSSL_VERSION_NUMBER) && \
				  OPENSSL_VERSION_NUMBER >= 0x10002000L)
	const unsigned char *name = NULL;
	char cstr[10];
	unsigned len;

	SSL_get0_alpn_selected(wsi->tls.ssl, &name, &len);
	if (!len) {
		lwsl_info("no ALPN upgrade\n");
		return 0;
	}

	if (len > sizeof(cstr) - 1)
		len = sizeof(cstr) - 1;

	memcpy(cstr, name, len);
	cstr[len] = '\0';

	lwsl_info("negotiated '%s' using ALPN\n", cstr);
	wsi->tls.use_ssl |= LCCSCF_USE_SSL;

	return lws_role_call_alpn_negotiated(wsi, (const char *)cstr);
#endif // OPENSSL_VERSION_NUMBER >= 0x10002000L

	return 0;
}

LWS_VISIBLE int
lws_context_init_server_ssl(const struct lws_context_creation_info *info,
			    struct lws_vhost *vhost)
{
	struct lws_context *context = vhost->context;
	struct lws wsi;

	if (!lws_check_opt(info->options,
			   LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT)) {
		vhost->tls.use_ssl = 0;

		return 0;
	}

	/*
	 * If he is giving a cert filepath, take it as a sign he wants to use
	 * it on this vhost.  User code can leave the cert filepath NULL and
	 * set the LWS_SERVER_OPTION_CREATE_VHOST_SSL_CTX option itself, in
	 * which case he's expected to set up the cert himself at
	 * LWS_CALLBACK_OPENSSL_LOAD_EXTRA_SERVER_VERIFY_CERTS, which
	 * provides the vhost SSL_CTX * in the user parameter.
	 */
	if (info->ssl_cert_filepath)
		vhost->options |= LWS_SERVER_OPTION_CREATE_VHOST_SSL_CTX;

	if (info->port != CONTEXT_PORT_NO_LISTEN) {

		vhost->tls.use_ssl = lws_check_opt(vhost->options,
					LWS_SERVER_OPTION_CREATE_VHOST_SSL_CTX);

		if (vhost->tls.use_ssl && info->ssl_cipher_list)
			lwsl_notice(" SSL ciphers: '%s'\n",
						info->ssl_cipher_list);

		if (vhost->tls.use_ssl)
			lwsl_notice(" Using SSL mode\n");
		else
			lwsl_notice(" Using non-SSL mode\n");
	}

	/*
	 * give him a fake wsi with context + vhost set, so he can use
	 * lws_get_context() in the callback
	 */
	memset(&wsi, 0, sizeof(wsi));
	wsi.vhost = vhost;
	wsi.context = context;

	/*
	 * as a server, if we are requiring clients to identify themselves
	 * then set the backend up for it
	 */
	if (lws_check_opt(info->options,
			  LWS_SERVER_OPTION_ALLOW_NON_SSL_ON_SSL_PORT))
		/* Normally SSL listener rejects non-ssl, optionally allow */
		vhost->tls.allow_non_ssl_on_ssl_port = 1;

	/*
	 * give user code a chance to load certs into the server
	 * allowing it to verify incoming client certs
	 */
	if (vhost->tls.use_ssl) {
		if (lws_tls_server_vhost_backend_init(info, vhost, &wsi))
			return -1;

		lws_tls_server_client_cert_verify_config(vhost);

		if (vhost->protocols[0].callback(&wsi,
			    LWS_CALLBACK_OPENSSL_LOAD_EXTRA_SERVER_VERIFY_CERTS,
			    vhost->tls.ssl_ctx, vhost, 0))
			return -1;
	}

	if (vhost->tls.use_ssl)
		lws_context_init_alpn(vhost);

	return 0;
}

LWS_VISIBLE int
lws_server_socket_service_ssl(struct lws *wsi, lws_sockfd_type accept_fd)
{
	struct lws_context *context = wsi->context;
	struct lws_vhost *vh;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	int n;
        char buf[256];

        (void)buf;

	if (!LWS_SSL_ENABLED(wsi->vhost))
		return 0;

	switch (lwsi_state(wsi)) {
	case LRS_SSL_INIT:

		if (wsi->tls.ssl)
			lwsl_err("%s: leaking ssl\n", __func__);
		if (accept_fd == LWS_SOCK_INVALID)
			assert(0);
		if (context->simultaneous_ssl_restriction &&
		    context->simultaneous_ssl >=
		    	    context->simultaneous_ssl_restriction) {
			lwsl_notice("unable to deal with SSL connection\n");
			return 1;
		}

		if (lws_tls_server_new_nonblocking(wsi, accept_fd)) {
			if (accept_fd != LWS_SOCK_INVALID)
				compatible_close(accept_fd);
			goto fail;
		}

		if (context->simultaneous_ssl_restriction &&
		    ++context->simultaneous_ssl ==
				    context->simultaneous_ssl_restriction)
			/* that was the last allowed SSL connection */
			lws_gate_accepts(context, 0);

#if defined(LWS_WITH_STATS)
		context->updated = 1;
#endif
		/*
		 * we are not accepted yet, but we need to enter ourselves
		 * as a live connection.  That way we can retry when more
		 * pieces come if we're not sorted yet
		 */
		lwsi_set_state(wsi, LRS_SSL_ACK_PENDING);

		lws_pt_lock(pt, __func__);
		if (__insert_wsi_socket_into_fds(context, wsi)) {
			lwsl_err("%s: failed to insert into fds\n", __func__);
			goto fail;
		}
		lws_pt_unlock(pt);

		lws_set_timeout(wsi, PENDING_TIMEOUT_SSL_ACCEPT,
				context->timeout_secs);

		lwsl_debug("inserted SSL accept into fds, trying SSL_accept\n");

		/* fallthru */

	case LRS_SSL_ACK_PENDING:

		if (lws_change_pollfd(wsi, LWS_POLLOUT, 0)) {
			lwsl_err("%s: lws_change_pollfd failed\n", __func__);
			goto fail;
		}

		lws_latency_pre(context, wsi);

		if (wsi->vhost->tls.allow_non_ssl_on_ssl_port) {

			n = recv(wsi->desc.sockfd, (char *)pt->serv_buf,
				 context->pt_serv_buf_size, MSG_PEEK);

		/*
		 * optionally allow non-SSL connect on SSL listening socket
		 * This is disabled by default, if enabled it goes around any
		 * SSL-level access control (eg, client-side certs) so leave
		 * it disabled unless you know it's not a problem for you
		 */
			if (n >= 1 && pt->serv_buf[0] >= ' ') {
				/*
				* TLS content-type for Handshake is 0x16, and
				* for ChangeCipherSpec Record, it's 0x14
				*
				* A non-ssl session will start with the HTTP
				* method in ASCII.  If we see it's not a legit
				* SSL handshake kill the SSL for this
				* connection and try to handle as a HTTP
				* connection upgrade directly.
				*/
				wsi->tls.use_ssl = 0;

				lws_tls_server_abort_connection(wsi);
				/*
				 * care... this creates wsi with no ssl
				 * when ssl is enabled and normally
				 * mandatory
				 */
				wsi->tls.ssl = NULL;
				if (lws_check_opt(context->options,
				    LWS_SERVER_OPTION_REDIRECT_HTTP_TO_HTTPS))
					wsi->tls.redirect_to_https = 1;
				lwsl_debug("accepted as non-ssl\n");
				goto accepted;
			}
			if (!n) {
				/*
				 * connection is gone, fail out
				 */
				lwsl_debug("PEEKed 0\n");
				goto fail;
			}
			if (n < 0 && (LWS_ERRNO == LWS_EAGAIN ||
				      LWS_ERRNO == LWS_EWOULDBLOCK)) {
				/*
				 * well, we get no way to know ssl or not
				 * so go around again waiting for something
				 * to come and give us a hint, or timeout the
				 * connection.
				 */
				if (lws_change_pollfd(wsi, 0, LWS_POLLIN)) {
					lwsl_info("%s: change_pollfd failed\n",
						  __func__);
					return -1;
				}

				lwsl_info("SSL_ERROR_WANT_READ\n");
				return 0;
			}
		}

		/* normal SSL connection processing path */

#if defined(LWS_WITH_STATS)
		if (!wsi->accept_start_us)
			wsi->accept_start_us = time_in_microseconds();
#endif
		errno = 0;
		lws_stats_atomic_bump(wsi->context, pt,
				      LWSSTATS_C_SSL_CONNECTIONS_ACCEPT_SPIN, 1);
		n = lws_tls_server_accept(wsi);
		lws_latency(context, wsi,
			"SSL_accept LRS_SSL_ACK_PENDING\n", n, n == 1);
		lwsl_info("SSL_accept says %d\n", n);
		switch (n) {
		case LWS_SSL_CAPABLE_DONE:
			break;
		case LWS_SSL_CAPABLE_ERROR:
			lws_stats_atomic_bump(wsi->context, pt,
					      LWSSTATS_C_SSL_CONNECTIONS_FAILED, 1);
	                lwsl_info("SSL_accept failed socket %u: %d\n",
	                		wsi->desc.sockfd, n);
			wsi->socket_is_permanently_unusable = 1;
			goto fail;

		default: /* MORE_SERVICE */
			return 0;
		}

		lws_stats_atomic_bump(wsi->context, pt,
				      LWSSTATS_C_SSL_CONNECTIONS_ACCEPTED, 1);
#if defined(LWS_WITH_STATS)
		lws_stats_atomic_bump(wsi->context, pt,
				      LWSSTATS_MS_SSL_CONNECTIONS_ACCEPTED_DELAY,
				      time_in_microseconds() - wsi->accept_start_us);
		wsi->accept_start_us = time_in_microseconds();
#endif

accepted:

		/* adapt our vhost to match the SNI SSL_CTX that was chosen */
		vh = context->vhost_list;
		while (vh) {
			if (!vh->being_destroyed && wsi->tls.ssl &&
			    vh->tls.ssl_ctx == lws_tls_ctx_from_wsi(wsi)) {
				lwsl_info("setting wsi to vh %s\n", vh->name);
				wsi->vhost = vh;
				break;
			}
			vh = vh->vhost_next;
		}

		/* OK, we are accepted... give him some time to negotiate */
		lws_set_timeout(wsi, PENDING_TIMEOUT_ESTABLISH_WITH_SERVER,
				context->timeout_secs);

		lwsi_set_state(wsi, LRS_ESTABLISHED);
		if (lws_tls_server_conn_alpn(wsi))
			goto fail;
		lwsl_debug("accepted new SSL conn\n");
		break;

	default:
		break;
	}

	return 0;

fail:
	return 1;
}

