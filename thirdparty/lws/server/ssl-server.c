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

#if !defined(LWS_WITH_MBEDTLS)
static int
OpenSSL_verify_callback(int preverify_ok, X509_STORE_CTX *x509_ctx)
{
	SSL *ssl;
	int n;
	struct lws *wsi;

	ssl = X509_STORE_CTX_get_ex_data(x509_ctx,
		SSL_get_ex_data_X509_STORE_CTX_idx());

	/*
	 * !!! nasty openssl requires the index to come as a library-scope
	 * static
	 */
	wsi = SSL_get_ex_data(ssl, openssl_websocket_private_data_index);

	n = wsi->vhost->protocols[0].callback(wsi,
			LWS_CALLBACK_OPENSSL_PERFORM_CLIENT_CERT_VERIFICATION,
					   x509_ctx, ssl, preverify_ok);

	/* convert return code from 0 = OK to 1 = OK */
	return !n;
}
#endif

static int
lws_context_ssl_init_ecdh(struct lws_vhost *vhost)
{
#ifdef LWS_SSL_SERVER_WITH_ECDH_CERT
	EC_KEY *EC_key = NULL;
	EVP_PKEY *pkey;
	int KeyType;
	X509 *x;

	if (!lws_check_opt(vhost->context->options, LWS_SERVER_OPTION_SSL_ECDH))
		return 0;

	lwsl_notice(" Using ECDH certificate support\n");

	/* Get X509 certificate from ssl context */
	x = sk_X509_value(vhost->ssl_ctx->extra_certs, 0);
	if (!x) {
		lwsl_err("%s: x is NULL\n", __func__);
		return 1;
	}
	/* Get the public key from certificate */
	pkey = X509_get_pubkey(x);
	if (!pkey) {
		lwsl_err("%s: pkey is NULL\n", __func__);

		return 1;
	}
	/* Get the key type */
	KeyType = EVP_PKEY_type(pkey->type);

	if (EVP_PKEY_EC != KeyType) {
		lwsl_notice("Key type is not EC\n");
		return 0;
	}
	/* Get the key */
	EC_key = EVP_PKEY_get1_EC_KEY(pkey);
	/* Set ECDH parameter */
	if (!EC_key) {
		lwsl_err("%s: ECDH key is NULL \n", __func__);
		return 1;
	}
	SSL_CTX_set_tmp_ecdh(vhost->ssl_ctx, EC_key);
	EC_KEY_free(EC_key);
#endif
	return 0;
}

static int
lws_context_ssl_init_ecdh_curve(struct lws_context_creation_info *info,
				struct lws_vhost *vhost)
{
#if defined(LWS_HAVE_OPENSSL_ECDH_H) && !defined(LWS_WITH_MBEDTLS)
	EC_KEY *ecdh;
	int ecdh_nid;
	const char *ecdh_curve = "prime256v1";

	if (info->ecdh_curve)
		ecdh_curve = info->ecdh_curve;

	ecdh_nid = OBJ_sn2nid(ecdh_curve);
	if (NID_undef == ecdh_nid) {
		lwsl_err("SSL: Unknown curve name '%s'", ecdh_curve);
		return 1;
	}

	ecdh = EC_KEY_new_by_curve_name(ecdh_nid);
	if (NULL == ecdh) {
		lwsl_err("SSL: Unable to create curve '%s'", ecdh_curve);
		return 1;
	}
	SSL_CTX_set_tmp_ecdh(vhost->ssl_ctx, ecdh);
	EC_KEY_free(ecdh);

	SSL_CTX_set_options(vhost->ssl_ctx, SSL_OP_SINGLE_ECDH_USE);

	lwsl_notice(" SSL ECDH curve '%s'\n", ecdh_curve);
#else
#if !defined(LWS_WITH_MBEDTLS)
	lwsl_notice(" OpenSSL doesn't support ECDH\n");
#endif
#endif
	return 0;
}

#if !defined(LWS_WITH_MBEDTLS) && defined(SSL_TLSEXT_ERR_NOACK) && !defined(OPENSSL_NO_TLSEXT)
static int
lws_ssl_server_name_cb(SSL *ssl, int *ad, void *arg)
{
	struct lws_context *context = (struct lws_context *)arg;
	struct lws_vhost *vhost, *vh;
	const char *servername;

	if (!ssl)
		return SSL_TLSEXT_ERR_NOACK;

	/*
	 * We can only get ssl accepted connections by using a vhost's ssl_ctx
	 * find out which listening one took us and only match vhosts on the
	 * same port.
	 */
	vh = context->vhost_list;
	while (vh) {
		if (!vh->being_destroyed && ssl && vh->ssl_ctx == SSL_get_SSL_CTX(ssl))
			break;
		vh = vh->vhost_next;
	}

	if (!vh) {
		assert(vh); /* can't match the incoming vh? */
		return SSL_TLSEXT_ERR_OK;
	}

	servername = SSL_get_servername(ssl, TLSEXT_NAMETYPE_host_name);
	if (!servername) {
		/* the client doesn't know what hostname it wants */
		lwsl_info("SNI: Unknown ServerName: %s\n", servername);

		return SSL_TLSEXT_ERR_OK;
	}

	vhost = lws_select_vhost(context, vh->listen_port, servername);
	if (!vhost) {
		lwsl_info("SNI: none: %s:%d\n", servername, vh->listen_port);

		return SSL_TLSEXT_ERR_OK;
	}

	lwsl_info("SNI: Found: %s:%d\n", servername, vh->listen_port);

	/* select the ssl ctx from the selected vhost for this conn */
	SSL_set_SSL_CTX(ssl, vhost->ssl_ctx);

	return SSL_TLSEXT_ERR_OK;
}
#endif

LWS_VISIBLE int
lws_context_init_server_ssl(struct lws_context_creation_info *info,
			    struct lws_vhost *vhost)
{
	struct lws_context *context = vhost->context;
	struct lws wsi;
	unsigned long error;
	int n;

	if (!lws_check_opt(info->options, LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT)) {
		vhost->use_ssl = 0;
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
		info->options |= LWS_SERVER_OPTION_CREATE_VHOST_SSL_CTX;

	if (info->port != CONTEXT_PORT_NO_LISTEN) {

		vhost->use_ssl = lws_check_opt(info->options,
					LWS_SERVER_OPTION_CREATE_VHOST_SSL_CTX);

		if (vhost->use_ssl && info->ssl_cipher_list)
			lwsl_notice(" SSL ciphers: '%s'\n", info->ssl_cipher_list);

		if (vhost->use_ssl)
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

	(void)n;
	(void)error;

	/*
	 * Firefox insists on SSLv23 not SSLv3
	 * Konq disables SSLv2 by default now, SSLv23 works
	 *
	 * SSLv23_server_method() is the openssl method for "allow all TLS
	 * versions", compared to e.g. TLSv1_2_server_method() which only allows
	 * tlsv1.2. Unwanted versions must be disabled using SSL_CTX_set_options()
	 */
#if !defined(LWS_WITH_MBEDTLS)
	{
		SSL_METHOD *method;

		method = (SSL_METHOD *)SSLv23_server_method();
		if (!method) {
			error = ERR_get_error();
			lwsl_err("problem creating ssl method %lu: %s\n",
					error, ERR_error_string(error,
					      (char *)context->pt[0].serv_buf));
			return 1;
		}
		vhost->ssl_ctx = SSL_CTX_new(method);	/* create context */
		if (!vhost->ssl_ctx) {
			error = ERR_get_error();
			lwsl_err("problem creating ssl context %lu: %s\n",
					error, ERR_error_string(error,
					      (char *)context->pt[0].serv_buf));
			return 1;
		}
	}
#else
	{
		const SSL_METHOD *method = TLSv1_2_server_method();

		vhost->ssl_ctx = SSL_CTX_new(method);	/* create context */
		if (!vhost->ssl_ctx) {
			lwsl_err("problem creating ssl context\n");
			return 1;
		}

	}
#endif
#if !defined(LWS_WITH_MBEDTLS)

	/* associate the lws context with the SSL_CTX */

	SSL_CTX_set_ex_data(vhost->ssl_ctx,
			openssl_SSL_CTX_private_data_index, (char *)vhost->context);
	/* Disable SSLv2 and SSLv3 */
	SSL_CTX_set_options(vhost->ssl_ctx, SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3);
#ifdef SSL_OP_NO_COMPRESSION
	SSL_CTX_set_options(vhost->ssl_ctx, SSL_OP_NO_COMPRESSION);
#endif
	SSL_CTX_set_options(vhost->ssl_ctx, SSL_OP_SINGLE_DH_USE);
	SSL_CTX_set_options(vhost->ssl_ctx, SSL_OP_CIPHER_SERVER_PREFERENCE);

	if (info->ssl_cipher_list)
		SSL_CTX_set_cipher_list(vhost->ssl_ctx,
						info->ssl_cipher_list);
#endif

	/* as a server, are we requiring clients to identify themselves? */

	if (lws_check_opt(info->options,
			  LWS_SERVER_OPTION_REQUIRE_VALID_OPENSSL_CLIENT_CERT)) {
		int verify_options = SSL_VERIFY_PEER;

		if (!lws_check_opt(info->options,
				   LWS_SERVER_OPTION_PEER_CERT_NOT_REQUIRED))
			verify_options |= SSL_VERIFY_FAIL_IF_NO_PEER_CERT;

#if !defined(LWS_WITH_MBEDTLS)
		SSL_CTX_set_session_id_context(vhost->ssl_ctx,
				(unsigned char *)context, sizeof(void *));

		/* absolutely require the client cert */

		SSL_CTX_set_verify(vhost->ssl_ctx,
		       verify_options, OpenSSL_verify_callback);
#endif
	}

#if !defined(LWS_WITH_MBEDTLS) && !defined(OPENSSL_NO_TLSEXT)
	SSL_CTX_set_tlsext_servername_callback(vhost->ssl_ctx,
					       lws_ssl_server_name_cb);
	SSL_CTX_set_tlsext_servername_arg(vhost->ssl_ctx, context);
#endif

	/*
	 * give user code a chance to load certs into the server
	 * allowing it to verify incoming client certs
	 */
#if !defined(LWS_WITH_MBEDTLS)
	if (info->ssl_ca_filepath &&
	    !SSL_CTX_load_verify_locations(vhost->ssl_ctx,
					   info->ssl_ca_filepath, NULL)) {
		lwsl_err("%s: SSL_CTX_load_verify_locations unhappy\n", __func__);
	}
#endif
	if (vhost->use_ssl) {
		if (lws_context_ssl_init_ecdh_curve(info, vhost))
			return -1;

		vhost->protocols[0].callback(&wsi,
			LWS_CALLBACK_OPENSSL_LOAD_EXTRA_SERVER_VERIFY_CERTS,
			vhost->ssl_ctx, NULL, 0);
	}

	if (lws_check_opt(info->options, LWS_SERVER_OPTION_ALLOW_NON_SSL_ON_SSL_PORT))
		/* Normally SSL listener rejects non-ssl, optionally allow */
		vhost->allow_non_ssl_on_ssl_port = 1;

	if (info->ssl_options_set)
		SSL_CTX_set_options(vhost->ssl_ctx, info->ssl_options_set);

/* SSL_clear_options introduced in 0.9.8m */
#if !defined(LWS_WITH_MBEDTLS)
#if (OPENSSL_VERSION_NUMBER >= 0x009080df) && !defined(USE_WOLFSSL)
	if (info->ssl_options_clear)
		SSL_CTX_clear_options(vhost->ssl_ctx, info->ssl_options_clear);
#endif
#endif

	lwsl_info(" SSL options 0x%lX\n", SSL_CTX_get_options(vhost->ssl_ctx));

	if (vhost->use_ssl && info->ssl_cert_filepath) {
		/*
		 * The user code can choose to either pass the cert and
		 * key filepaths using the info members like this, or it can
		 * leave them NULL; force the vhost SSL_CTX init using the info
		 * options flag LWS_SERVER_OPTION_CREATE_VHOST_SSL_CTX; and
		 * set up the cert himself using the user callback
		 * LWS_CALLBACK_OPENSSL_LOAD_EXTRA_SERVER_VERIFY_CERTS, which
		 * happened just above and has the vhost SSL_CTX * in the user
		 * parameter.
		 */
#if !defined(LWS_WITH_MBEDTLS)
		/* set the local certificate from CertFile */
		n = SSL_CTX_use_certificate_chain_file(vhost->ssl_ctx,
					info->ssl_cert_filepath);
		if (n != 1) {
			error = ERR_get_error();
			lwsl_err("problem getting cert '%s' %lu: %s\n",
				info->ssl_cert_filepath,
				error,
				ERR_error_string(error,
					      (char *)context->pt[0].serv_buf));
			return 1;
		}
		lws_ssl_bind_passphrase(vhost->ssl_ctx, info);
#else
		uint8_t *p;
		lws_filepos_t flen;
		int err;

		if (alloc_pem_to_der_file(vhost->context, info->ssl_cert_filepath, &p,
				                &flen)) {
			lwsl_err("couldn't find cert file %s\n",
				 info->ssl_cert_filepath);

			return 1;
		}
		err = SSL_CTX_use_certificate_ASN1(vhost->ssl_ctx, flen, p);
		if (!err) {
			lwsl_err("Problem loading cert\n");
			return 1;
		}
#if !defined(LWS_WITH_ESP32)
		free(p);
		p = NULL;
#endif

		if (info->ssl_private_key_filepath) {
			if (alloc_pem_to_der_file(vhost->context,
				       info->ssl_private_key_filepath, &p, &flen)) {
				lwsl_err("couldn't find cert file %s\n",
					 info->ssl_cert_filepath);

				return 1;
			}
			err = SSL_CTX_use_PrivateKey_ASN1(0, vhost->ssl_ctx, p, flen);
			if (!err) {
				lwsl_err("Problem loading key\n");

				return 1;
			}
		}

#if !defined(LWS_WITH_ESP32)
		free(p);
		p = NULL;
#endif
#endif
		if (info->ssl_private_key_filepath != NULL) {
#if !defined(LWS_WITH_MBEDTLS)
			/* set the private key from KeyFile */
			if (SSL_CTX_use_PrivateKey_file(vhost->ssl_ctx,
				     info->ssl_private_key_filepath,
						       SSL_FILETYPE_PEM) != 1) {
				error = ERR_get_error();
				lwsl_err("ssl problem getting key '%s' %lu: %s\n",
					 info->ssl_private_key_filepath, error,
					 ERR_error_string(error,
					      (char *)context->pt[0].serv_buf));
				return 1;
			}
#endif
		} else
			if (vhost->protocols[0].callback(&wsi,
				LWS_CALLBACK_OPENSSL_CONTEXT_REQUIRES_PRIVATE_KEY,
				vhost->ssl_ctx, NULL, 0)) {
				lwsl_err("ssl private key not set\n");

				return 1;
			}
#if !defined(LWS_WITH_MBEDTLS)
		/* verify private key */
		if (!SSL_CTX_check_private_key(vhost->ssl_ctx)) {
			lwsl_err("Private SSL key doesn't match cert\n");
			return 1;
		}
#endif
	}
	if (vhost->use_ssl) {
		if (lws_context_ssl_init_ecdh(vhost))
			return 1;

		/*
		 * SSL is happy and has a cert it's content with
		 * If we're supporting HTTP2, initialize that
		 */
		lws_context_init_http2_ssl(vhost);
	}

	return 0;
}

