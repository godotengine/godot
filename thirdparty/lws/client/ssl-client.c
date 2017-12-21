/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2017 Andy Green <andy@warmcat.com>
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

#include "private-libwebsockets.h"

extern int openssl_websocket_private_data_index,
    openssl_SSL_CTX_private_data_index;

extern void
lws_ssl_bind_passphrase(SSL_CTX *ssl_ctx, struct lws_context_creation_info *info);

extern int lws_ssl_get_error(struct lws *wsi, int n);

#if defined(USE_WOLFSSL)
#else

static int
OpenSSL_client_verify_callback(int preverify_ok, X509_STORE_CTX *x509_ctx)
{
#if defined(LWS_WITH_MBEDTLS)
	lwsl_notice("%s\n", __func__);

	return 0;
#else
	SSL *ssl;
	int n;
	struct lws *wsi;

	/* keep old behaviour accepting self-signed server certs */
	if (!preverify_ok) {
		int err = X509_STORE_CTX_get_error(x509_ctx);

		if (err != X509_V_OK) {
			ssl = X509_STORE_CTX_get_ex_data(x509_ctx, SSL_get_ex_data_X509_STORE_CTX_idx());
			wsi = SSL_get_ex_data(ssl, openssl_websocket_private_data_index);

			if ((err == X509_V_ERR_DEPTH_ZERO_SELF_SIGNED_CERT ||
					err == X509_V_ERR_SELF_SIGNED_CERT_IN_CHAIN) &&
					wsi->use_ssl & LCCSCF_ALLOW_SELFSIGNED) {
				lwsl_notice("accepting self-signed certificate (verify_callback)\n");
				X509_STORE_CTX_set_error(x509_ctx, X509_V_OK);
				return 1;	// ok
			} else if ((err == X509_V_ERR_CERT_NOT_YET_VALID ||
					err == X509_V_ERR_CERT_HAS_EXPIRED) &&
					wsi->use_ssl & LCCSCF_ALLOW_EXPIRED) {
				if (err == X509_V_ERR_CERT_NOT_YET_VALID)
					lwsl_notice("accepting not yet valid certificate (verify_callback)\n");
				else if (err == X509_V_ERR_CERT_HAS_EXPIRED)
					lwsl_notice("accepting expired certificate (verify_callback)\n");
				X509_STORE_CTX_set_error(x509_ctx, X509_V_OK);
				return 1;	// ok
			}
		}
	}

	ssl = X509_STORE_CTX_get_ex_data(x509_ctx, SSL_get_ex_data_X509_STORE_CTX_idx());
	wsi = SSL_get_ex_data(ssl, openssl_websocket_private_data_index);

	n = lws_get_context_protocol(wsi->context, 0).callback(wsi,
			LWS_CALLBACK_OPENSSL_PERFORM_SERVER_CERT_VERIFICATION,
			x509_ctx, ssl, preverify_ok);

	/* keep old behaviour if something wrong with server certs */
	/* if ssl error is overruled in callback and cert is ok,
	 * X509_STORE_CTX_set_error(x509_ctx, X509_V_OK); must be set and
	 * return value is 0 from callback */
	if (!preverify_ok) {
		int err = X509_STORE_CTX_get_error(x509_ctx);

		if (err != X509_V_OK) {	/* cert validation error was not handled in callback */
			int depth = X509_STORE_CTX_get_error_depth(x509_ctx);
			const char* msg = X509_verify_cert_error_string(err);
			lwsl_err("SSL error: %s (preverify_ok=%d;err=%d;depth=%d)\n", msg, preverify_ok, err, depth);
			return preverify_ok;	// not ok
		}
	}
	/* convert callback return code from 0 = OK to verify callback return value 1 = OK */
	return !n;
#endif
}
#endif

int
lws_ssl_client_bio_create(struct lws *wsi)
{
	char hostname[128], *p;

	if (lws_hdr_copy(wsi, hostname, sizeof(hostname),
			 _WSI_TOKEN_CLIENT_HOST) <= 0) {
		lwsl_err("%s: Unable to get hostname\n", __func__);

		return -1;
	}

	/*
	 * remove any :port part on the hostname... necessary for network
	 * connection but typical certificates do not contain it
	 */
	p = hostname;
	while (*p) {
		if (*p == ':') {
			*p = '\0';
			break;
		}
		p++;
	}

	wsi->ssl = SSL_new(wsi->vhost->ssl_client_ctx);
	if (!wsi->ssl) {
		lwsl_err("SSL_new failed: %s\n",
		         ERR_error_string(lws_ssl_get_error(wsi, 0), NULL));
		lws_ssl_elaborate_error();
		return -1;
	}

#if defined (LWS_HAVE_SSL_SET_INFO_CALLBACK)
	if (wsi->vhost->ssl_info_event_mask)
		SSL_set_info_callback(wsi->ssl, lws_ssl_info_callback);
#endif

#if defined LWS_HAVE_X509_VERIFY_PARAM_set1_host
	X509_VERIFY_PARAM *param;
	(void)param;

	if (!(wsi->use_ssl & LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK)) {
		param = SSL_get0_param(wsi->ssl);
		/* Enable automatic hostname checks */
		X509_VERIFY_PARAM_set_hostflags(param,
						X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
		X509_VERIFY_PARAM_set1_host(param, hostname, 0);
	}

#endif

#if !defined(USE_WOLFSSL) && !defined(LWS_WITH_MBEDTLS)
#ifndef USE_OLD_CYASSL
	/* OpenSSL_client_verify_callback will be called @ SSL_connect() */
	SSL_set_verify(wsi->ssl, SSL_VERIFY_PEER, OpenSSL_client_verify_callback);
#endif
#endif

#if !defined(USE_WOLFSSL) && !defined(LWS_WITH_MBEDTLS)
	SSL_set_mode(wsi->ssl,  SSL_MODE_ACCEPT_MOVING_WRITE_BUFFER);
#endif
	/*
	 * use server name indication (SNI), if supported,
	 * when establishing connection
	 */
#ifdef USE_WOLFSSL
#ifdef USE_OLD_CYASSL
#ifdef CYASSL_SNI_HOST_NAME
	CyaSSL_UseSNI(wsi->ssl, CYASSL_SNI_HOST_NAME, hostname, strlen(hostname));
#endif
#else
#ifdef WOLFSSL_SNI_HOST_NAME
	wolfSSL_UseSNI(wsi->ssl, WOLFSSL_SNI_HOST_NAME, hostname, strlen(hostname));
#endif
#endif
#else
#if defined(LWS_WITH_MBEDTLS)
	if (wsi->vhost->x509_client_CA)
		SSL_set_verify(wsi->ssl, SSL_VERIFY_PEER, OpenSSL_client_verify_callback);
	else
		SSL_set_verify(wsi->ssl, SSL_VERIFY_NONE, OpenSSL_client_verify_callback);

#else
#ifdef SSL_CTRL_SET_TLSEXT_HOSTNAME
	SSL_set_tlsext_host_name(wsi->ssl, hostname);
#endif
#endif
#endif

#ifdef USE_WOLFSSL
	/*
	 * wolfSSL/CyaSSL does certificate verification differently
	 * from OpenSSL.
	 * If we should ignore the certificate, we need to set
	 * this before SSL_new and SSL_connect is called.
	 * Otherwise the connect will simply fail with error code -155
	 */
#ifdef USE_OLD_CYASSL
	if (wsi->use_ssl == 2)
		CyaSSL_set_verify(wsi->ssl, SSL_VERIFY_NONE, NULL);
#else
	if (wsi->use_ssl == 2)
		wolfSSL_set_verify(wsi->ssl, SSL_VERIFY_NONE, NULL);
#endif
#endif /* USE_WOLFSSL */

#if !defined(LWS_WITH_MBEDTLS)
	wsi->client_bio = BIO_new_socket(wsi->desc.sockfd, BIO_NOCLOSE);
	SSL_set_bio(wsi->ssl, wsi->client_bio, wsi->client_bio);
#else
	SSL_set_fd(wsi->ssl, wsi->desc.sockfd);
#endif

#ifdef USE_WOLFSSL
#ifdef USE_OLD_CYASSL
	CyaSSL_set_using_nonblock(wsi->ssl, 1);
#else
	wolfSSL_set_using_nonblock(wsi->ssl, 1);
#endif
#else
#if !defined(LWS_WITH_MBEDTLS)
	BIO_set_nbio(wsi->client_bio, 1); /* nonblocking */
#endif
#endif

#if !defined(LWS_WITH_MBEDTLS)
	SSL_set_ex_data(wsi->ssl, openssl_websocket_private_data_index,
			wsi);
#endif

	return 0;
}

#if defined(LWS_WITH_MBEDTLS)
int ERR_get_error(void)
{
	return 0;
}
#endif

int
lws_ssl_client_connect1(struct lws *wsi)
{
	struct lws_context *context = wsi->context;
	int n = 0;

	lws_latency_pre(context, wsi);

	n = SSL_connect(wsi->ssl);

	lws_latency(context, wsi,
	  "SSL_connect LWSCM_WSCL_ISSUE_HANDSHAKE", n, n > 0);

	if (n < 0) {
		n = lws_ssl_get_error(wsi, n);

		if (n == SSL_ERROR_WANT_READ)
			goto some_wait;

		if (n == SSL_ERROR_WANT_WRITE) {
			/*
			 * wants us to retry connect due to
			 * state of the underlying ssl layer...
			 * but since it may be stalled on
			 * blocked write, no incoming data may
			 * arrive to trigger the retry.
			 * Force (possibly many times if the SSL
			 * state persists in returning the
			 * condition code, but other sockets
			 * are getting serviced inbetweentimes)
			 * us to get called back when writable.
			 */
			lwsl_info("%s: WANT_WRITE... retrying\n", __func__);
			lws_callback_on_writable(wsi);
some_wait:
			wsi->mode = LWSCM_WSCL_WAITING_SSL;

			return 0; /* no error */
		}

		{
			struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
			char *p = (char *)&pt->serv_buf[0];
			char *sb = p;

			lwsl_err("ssl hs1 error, X509_V_ERR = %d: %s\n",
				 n, ERR_error_string(n, sb));
			lws_ssl_elaborate_error();
		}

		n = -1;
	}

	if (n <= 0) {
		/*
		 * retry if new data comes until we
		 * run into the connection timeout or win
		 */

		unsigned long error = ERR_get_error();

		if (error != SSL_ERROR_NONE) {
			struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
			char *p = (char *)&pt->serv_buf[0];
			char *sb = p;
			lwsl_err("SSL connect error %lu: %s\n",
				error, ERR_error_string(error, sb));
			return -1;
		}

		return 0;
	}

	return 1;
}

int
lws_ssl_client_connect2(struct lws *wsi)
{
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	char *p = (char *)&pt->serv_buf[0];
	char *sb = p;
	int n = 0;

	if (wsi->mode == LWSCM_WSCL_WAITING_SSL) {
		lws_latency_pre(context, wsi);
		n = SSL_connect(wsi->ssl);
		lwsl_debug("%s: SSL_connect says %d\n", __func__, n);

		lws_latency(context, wsi,
			    "SSL_connect LWSCM_WSCL_WAITING_SSL", n, n > 0);

		if (n < 0) {
			n = lws_ssl_get_error(wsi, n);

			if (n == SSL_ERROR_WANT_READ) {
				lwsl_info("SSL_connect WANT_READ... retrying\n");

				wsi->mode = LWSCM_WSCL_WAITING_SSL;

				return 0; /* no error */
			}

			if (n == SSL_ERROR_WANT_WRITE) {
				/*
				 * wants us to retry connect due to
				 * state of the underlying ssl layer...
				 * but since it may be stalled on
				 * blocked write, no incoming data may
				 * arrive to trigger the retry.
				 * Force (possibly many times if the SSL
				 * state persists in returning the
				 * condition code, but other sockets
				 * are getting serviced inbetweentimes)
				 * us to get called back when writable.
				 */
				lwsl_info("SSL_connect WANT_WRITE... retrying\n");
				lws_callback_on_writable(wsi);

				wsi->mode = LWSCM_WSCL_WAITING_SSL;

				return 0; /* no error */
			}

			n = -1;
		}

		if (n <= 0) {
			/*
			 * retry if new data comes until we
			 * run into the connection timeout or win
			 */
			unsigned long error = ERR_get_error();
			if (error != SSL_ERROR_NONE) {
				lwsl_err("SSL connect error %lu: %s\n",
					 error, ERR_error_string(error, sb));
				return -1;
			}
		}
	}

#if defined(LWS_WITH_MBEDTLS)
	{
		X509 *peer = SSL_get_peer_certificate(wsi->ssl);

		if (!peer) {
			lwsl_notice("peer did not provide cert\n");

			return -1;
		}
		lwsl_notice("peer provided cert\n");
	}
#endif

#ifndef USE_WOLFSSL
	/*
	 * See comment above about wolfSSL certificate
	 * verification
	 */
	lws_latency_pre(context, wsi);
	n = SSL_get_verify_result(wsi->ssl);
	lws_latency(context, wsi,
		"SSL_get_verify_result LWS_CONNMODE..HANDSHAKE", n, n > 0);

	lwsl_debug("get_verify says %d\n", n);

	if (n != X509_V_OK) {
		if ((n == X509_V_ERR_DEPTH_ZERO_SELF_SIGNED_CERT ||
		     n == X509_V_ERR_SELF_SIGNED_CERT_IN_CHAIN) &&
		     (wsi->use_ssl & LCCSCF_ALLOW_SELFSIGNED)) {
			lwsl_notice("accepting self-signed certificate\n");
		} else if ((n == X509_V_ERR_CERT_NOT_YET_VALID ||
		            n == X509_V_ERR_CERT_HAS_EXPIRED) &&
		     (wsi->use_ssl & LCCSCF_ALLOW_EXPIRED)) {
			lwsl_notice("accepting expired certificate\n");
		} else if (n == X509_V_ERR_CERT_NOT_YET_VALID) {
			lwsl_notice("Cert is from the future... "
				    "probably our clock... accepting...\n");
		} else {
			lwsl_err("server's cert didn't look good, X509_V_ERR = %d: %s\n",
				 n, ERR_error_string(n, sb));
			lws_ssl_elaborate_error();
			return -1;
		}
	}

#endif /* USE_WOLFSSL */

	return 1;
}


int lws_context_init_client_ssl(struct lws_context_creation_info *info,
				struct lws_vhost *vhost)
{
	SSL_METHOD *method = NULL;
	struct lws wsi;
	unsigned long error;
	const char *ca_filepath = info->ssl_ca_filepath;
#if !defined(LWS_WITH_MBEDTLS)
	const char *cipher_list = info->ssl_cipher_list;
	const char *private_key_filepath = info->ssl_private_key_filepath;
	const char *cert_filepath = info->ssl_cert_filepath;
	int n;

	if (vhost->options & LWS_SERVER_OPTION_ONLY_RAW)
		return 0;

	/*
	 *  for backwards-compatibility default to using ssl_... members, but
	 * if the newer client-specific ones are given, use those
	 */
	if (info->client_ssl_cipher_list)
		cipher_list = info->client_ssl_cipher_list;
	if (info->client_ssl_cert_filepath)
		cert_filepath = info->client_ssl_cert_filepath;
	if (info->client_ssl_private_key_filepath)
		private_key_filepath = info->client_ssl_private_key_filepath;
#endif
	if (info->client_ssl_ca_filepath)
		ca_filepath = info->client_ssl_ca_filepath;

	if (!lws_check_opt(info->options, LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT))
		return 0;

	if (vhost->ssl_client_ctx)
		return 0;

	if (info->provided_client_ssl_ctx) {
		/* use the provided OpenSSL context if given one */
		vhost->ssl_client_ctx = info->provided_client_ssl_ctx;
		/* nothing for lib to delete */
		vhost->user_supplied_ssl_ctx = 1;

		return 0;
	}

	/* basic openssl init already happened in context init */

	/* choose the most recent spin of the api */
#if defined(LWS_HAVE_TLS_CLIENT_METHOD)
	method = (SSL_METHOD *)TLS_client_method();
#elif defined(LWS_HAVE_TLSV1_2_CLIENT_METHOD)
	method = (SSL_METHOD *)TLSv1_2_client_method();
#else
	method = (SSL_METHOD *)SSLv23_client_method();
#endif
	if (!method) {
		error = ERR_get_error();
		lwsl_err("problem creating ssl method %lu: %s\n",
			error, ERR_error_string(error,
				      (char *)vhost->context->pt[0].serv_buf));
		return 1;
	}
	/* create context */
	vhost->ssl_client_ctx = SSL_CTX_new(method);
	if (!vhost->ssl_client_ctx) {
		error = ERR_get_error();
		lwsl_err("problem creating ssl context %lu: %s\n",
			error, ERR_error_string(error,
				      (char *)vhost->context->pt[0].serv_buf));
		return 1;
	}

	lwsl_notice("created client ssl context for %s\n", vhost->name);

#ifdef SSL_OP_NO_COMPRESSION
	SSL_CTX_set_options(vhost->ssl_client_ctx, SSL_OP_NO_COMPRESSION);
#endif

#if defined(LWS_WITH_MBEDTLS)
	if (ca_filepath) {
		lws_filepos_t len;
		uint8_t *buf;
		/*
		 * prototype this here, the shim does not export it in the
		 * header, and we need to use the shim unchanged for ESP32 case
		 */
		X509 *d2i_X509(X509 **cert, const unsigned char *buffer, long len);

		if (alloc_file(vhost->context, ca_filepath, &buf, &len)) {
			lwsl_err("Load CA cert file %s failed\n", ca_filepath);
			return 1;
		}

		vhost->x509_client_CA = d2i_X509(NULL, buf, len);
		free(buf);
		if (!vhost->x509_client_CA) {
			lwsl_err("client CA: x509 parse failed\n");
			return 1;
		}

		SSL_CTX_add_client_CA(vhost->ssl_client_ctx,
				      vhost->x509_client_CA);

		lwsl_notice("client loaded CA for verification %s\n", ca_filepath);
	}
#else
	SSL_CTX_set_options(vhost->ssl_client_ctx,
			    SSL_OP_CIPHER_SERVER_PREFERENCE);

	if (cipher_list)
		SSL_CTX_set_cipher_list(vhost->ssl_client_ctx, cipher_list);

#ifdef LWS_SSL_CLIENT_USE_OS_CA_CERTS
	if (!lws_check_opt(info->options, LWS_SERVER_OPTION_DISABLE_OS_CA_CERTS))
		/* loads OS default CA certs */
		SSL_CTX_set_default_verify_paths(vhost->ssl_client_ctx);
#endif

	/* openssl init for cert verification (for client sockets) */
	if (!ca_filepath) {
		if (!SSL_CTX_load_verify_locations(
			vhost->ssl_client_ctx, NULL, LWS_OPENSSL_CLIENT_CERTS))
			lwsl_err("Unable to load SSL Client certs from %s "
			    "(set by LWS_OPENSSL_CLIENT_CERTS) -- "
			    "client ssl isn't going to work\n",
			    LWS_OPENSSL_CLIENT_CERTS);
	} else
		if (!SSL_CTX_load_verify_locations(
			vhost->ssl_client_ctx, ca_filepath, NULL)) {
			lwsl_err(
				"Unable to load SSL Client certs "
				"file from %s -- client ssl isn't "
				"going to work\n", info->client_ssl_ca_filepath);
			lws_ssl_elaborate_error();
		}
		else
			lwsl_info("loaded ssl_ca_filepath\n");

	/*
	 * callback allowing user code to load extra verification certs
	 * helping the client to verify server identity
	 */

	/* support for client-side certificate authentication */
	if (cert_filepath) {
		lwsl_notice("%s: doing cert filepath\n", __func__);
		n = SSL_CTX_use_certificate_chain_file(vhost->ssl_client_ctx,
						       cert_filepath);
		if (n < 1) {
			lwsl_err("problem %d getting cert '%s'\n", n,
				 cert_filepath);
			lws_ssl_elaborate_error();
			return 1;
		}
		lwsl_notice("Loaded client cert %s\n", cert_filepath);
	}
	if (private_key_filepath) {
		lwsl_notice("%s: doing private key filepath\n", __func__);
		lws_ssl_bind_passphrase(vhost->ssl_client_ctx, info);
		/* set the private key from KeyFile */
		if (SSL_CTX_use_PrivateKey_file(vhost->ssl_client_ctx,
		    private_key_filepath, SSL_FILETYPE_PEM) != 1) {
			lwsl_err("use_PrivateKey_file '%s'\n",
				 private_key_filepath);
			lws_ssl_elaborate_error();
			return 1;
		}
		lwsl_notice("Loaded client cert private key %s\n",
			    private_key_filepath);

		/* verify private key */
		if (!SSL_CTX_check_private_key(vhost->ssl_client_ctx)) {
			lwsl_err("Private SSL key doesn't match cert\n");
			return 1;
		}
	}
#endif
	/*
	 * give him a fake wsi with context set, so he can use
	 * lws_get_context() in the callback
	 */
	memset(&wsi, 0, sizeof(wsi));
	wsi.vhost = vhost;
	wsi.context = vhost->context;

	vhost->protocols[0].callback(&wsi,
			LWS_CALLBACK_OPENSSL_LOAD_EXTRA_CLIENT_VERIFY_CERTS,
				       vhost->ssl_client_ctx, NULL, 0);

	return 0;
}
