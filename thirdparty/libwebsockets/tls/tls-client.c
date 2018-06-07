/*
 * libwebsockets - client-related ssl code independent of backend
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

#include "core/private.h"

int
lws_ssl_client_connect1(struct lws *wsi)
{
	struct lws_context *context = wsi->context;
	int n = 0;

	lws_latency_pre(context, wsi);
	n = lws_tls_client_connect(wsi);
	lws_latency(context, wsi, "SSL_connect hs", n, n > 0);

	switch (n) {
	case LWS_SSL_CAPABLE_ERROR:
		return -1;
	case LWS_SSL_CAPABLE_DONE:
		return 1; /* connected */
	case LWS_SSL_CAPABLE_MORE_SERVICE_WRITE:
		lws_callback_on_writable(wsi);
		/* fallthru */
	case LWS_SSL_CAPABLE_MORE_SERVICE_READ:
		lwsi_set_state(wsi, LRS_WAITING_SSL);
		break;
	case LWS_SSL_CAPABLE_MORE_SERVICE:
		break;
	}

	return 0; /* retry */
}

int
lws_ssl_client_connect2(struct lws *wsi, char *errbuf, int len)
{
	int n = 0;

	if (lwsi_state(wsi) == LRS_WAITING_SSL) {
		lws_latency_pre(wsi->context, wsi);

		n = lws_tls_client_connect(wsi);
		lwsl_debug("%s: SSL_connect says %d\n", __func__, n);
		lws_latency(wsi->context, wsi,
			    "SSL_connect LRS_WAITING_SSL", n, n > 0);

		switch (n) {
		case LWS_SSL_CAPABLE_ERROR:
			lws_snprintf(errbuf, len, "client connect failed");
			return -1;
		case LWS_SSL_CAPABLE_DONE:
			break; /* connected */
		case LWS_SSL_CAPABLE_MORE_SERVICE_WRITE:
			lws_callback_on_writable(wsi);
			/* fallthru */
		case LWS_SSL_CAPABLE_MORE_SERVICE_READ:
			lwsi_set_state(wsi, LRS_WAITING_SSL);
			/* fallthru */
		case LWS_SSL_CAPABLE_MORE_SERVICE:
			return 0;
		}
	}

	if (lws_tls_client_confirm_peer_cert(wsi, errbuf, len))
		return -1;

	return 1;
}


int lws_context_init_client_ssl(const struct lws_context_creation_info *info,
				struct lws_vhost *vhost)
{
	const char *ca_filepath = info->ssl_ca_filepath;
	const char *cipher_list = info->ssl_cipher_list;
	const char *private_key_filepath = info->ssl_private_key_filepath;
	const char *cert_filepath = info->ssl_cert_filepath;
	struct lws wsi;

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

	if (info->client_ssl_ca_filepath)
		ca_filepath = info->client_ssl_ca_filepath;

	if (!lws_check_opt(info->options, LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT))
		return 0;

	if (vhost->tls.ssl_client_ctx)
		return 0;

	if (info->provided_client_ssl_ctx) {
		/* use the provided OpenSSL context if given one */
		vhost->tls.ssl_client_ctx = info->provided_client_ssl_ctx;
		/* nothing for lib to delete */
		vhost->tls.user_supplied_ssl_ctx = 1;

		return 0;
	}

	if (lws_tls_client_create_vhost_context(vhost, info, cipher_list,
						ca_filepath, cert_filepath,
						private_key_filepath))
		return 1;

	lwsl_notice("created client ssl context for %s\n", vhost->name);

	/*
	 * give him a fake wsi with context set, so he can use
	 * lws_get_context() in the callback
	 */
	memset(&wsi, 0, sizeof(wsi));
	wsi.vhost = vhost;
	wsi.context = vhost->context;

	vhost->protocols[0].callback(&wsi,
			LWS_CALLBACK_OPENSSL_LOAD_EXTRA_CLIENT_VERIFY_CERTS,
				       vhost->tls.ssl_client_ctx, NULL, 0);

	return 0;
}
