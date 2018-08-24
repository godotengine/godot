/*
 * libwebsockets - mbedtls-specific client TLS code
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

static int
OpenSSL_client_verify_callback(int preverify_ok, X509_STORE_CTX *x509_ctx)
{
	return 0;
}

int
lws_ssl_client_bio_create(struct lws *wsi)
{
	X509_VERIFY_PARAM *param;
	char hostname[128], *p;
	const char *alpn_comma = wsi->context->tls.alpn_default;
	struct alpn_ctx protos;

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

	wsi->tls.ssl = SSL_new(wsi->vhost->tls.ssl_client_ctx);
	if (!wsi->tls.ssl)
		return -1;

	if (wsi->vhost->tls.ssl_info_event_mask)
		SSL_set_info_callback(wsi->tls.ssl, lws_ssl_info_callback);

	if (!(wsi->tls.use_ssl & LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK)) {
		param = SSL_get0_param(wsi->tls.ssl);
		/* Enable automatic hostname checks */
	//	X509_VERIFY_PARAM_set_hostflags(param,
	//				X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
		X509_VERIFY_PARAM_set1_host(param, hostname, 0);
	}

	if (wsi->vhost->tls.alpn)
		alpn_comma = wsi->vhost->tls.alpn;

	if (lws_hdr_copy(wsi, hostname, sizeof(hostname),
			 _WSI_TOKEN_CLIENT_ALPN) > 0)
		alpn_comma = hostname;

	lwsl_info("%s: %p: client conn sending ALPN list '%s'\n",
		  __func__, wsi, alpn_comma);

	protos.len = lws_alpn_comma_to_openssl(alpn_comma, protos.data,
					       sizeof(protos.data) - 1);

	/* with mbedtls, protos is not pointed to after exit from this call */
	SSL_set_alpn_select_cb(wsi->tls.ssl, &protos);

	/*
	 * use server name indication (SNI), if supported,
	 * when establishing connection
	 */
	SSL_set_verify(wsi->tls.ssl, SSL_VERIFY_PEER,
		       OpenSSL_client_verify_callback);

	SSL_set_fd(wsi->tls.ssl, wsi->desc.sockfd);

	return 0;
}

int ERR_get_error(void)
{
	return 0;
}

enum lws_ssl_capable_status
lws_tls_client_connect(struct lws *wsi)
{
	int m, n = SSL_connect(wsi->tls.ssl);
	const unsigned char *prot;
	unsigned int len;

	if (n == 1) {
		SSL_get0_alpn_selected(wsi->tls.ssl, &prot, &len);
		lws_role_call_alpn_negotiated(wsi, (const char *)prot);
		lwsl_info("client connect OK\n");
		return LWS_SSL_CAPABLE_DONE;
	}

	m = SSL_get_error(wsi->tls.ssl, n);

	if (m == SSL_ERROR_WANT_READ || SSL_want_read(wsi->tls.ssl))
		return LWS_SSL_CAPABLE_MORE_SERVICE_READ;

	if (m == SSL_ERROR_WANT_WRITE || SSL_want_write(wsi->tls.ssl))
		return LWS_SSL_CAPABLE_MORE_SERVICE_WRITE;

	if (!n) /* we don't know what he wants, but he says to retry */
		return LWS_SSL_CAPABLE_MORE_SERVICE;

	return LWS_SSL_CAPABLE_ERROR;
}

int
lws_tls_client_confirm_peer_cert(struct lws *wsi, char *ebuf, int ebuf_len)
{
	int n;
	X509 *peer = SSL_get_peer_certificate(wsi->tls.ssl);
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];
	char *sb = (char *)&pt->serv_buf[0];

	if (!peer) {
		lwsl_info("peer did not provide cert\n");

		return -1;
	}
	lwsl_info("peer provided cert\n");

	n = SSL_get_verify_result(wsi->tls.ssl);
	lws_latency(wsi->context, wsi,
			"SSL_get_verify_result LWS_CONNMODE..HANDSHAKE", n, n > 0);

        lwsl_debug("get_verify says %d\n", n);

	if (n == X509_V_OK)
		return 0;

	if (n == X509_V_ERR_HOSTNAME_MISMATCH &&
	    (wsi->tls.use_ssl & LCCSCF_SKIP_SERVER_CERT_HOSTNAME_CHECK)) {
		lwsl_info("accepting certificate for invalid hostname\n");
		return 0;
	}

	if (n == X509_V_ERR_INVALID_CA &&
	    (wsi->tls.use_ssl & LCCSCF_ALLOW_SELFSIGNED)) {
		lwsl_info("accepting certificate from untrusted CA\n");
		return 0;
	}

	if ((n == X509_V_ERR_CERT_NOT_YET_VALID ||
	     n == X509_V_ERR_CERT_HAS_EXPIRED) &&
	     (wsi->tls.use_ssl & LCCSCF_ALLOW_EXPIRED)) {
		lwsl_info("accepting expired or not yet valid certificate\n");

		return 0;
	}
	lws_snprintf(ebuf, ebuf_len,
		"server's cert didn't look good, X509_V_ERR = %d: %s\n",
		 n, ERR_error_string(n, sb));
	lwsl_info("%s\n", ebuf);
	lws_ssl_elaborate_error();

	return -1;
}

int
lws_tls_client_create_vhost_context(struct lws_vhost *vh,
				    const struct lws_context_creation_info *info,
				    const char *cipher_list,
				    const char *ca_filepath,
				    const char *cert_filepath,
				    const char *private_key_filepath)
{
	X509 *d2i_X509(X509 **cert, const unsigned char *buffer, long len);
	SSL_METHOD *method = (SSL_METHOD *)TLS_client_method();
	unsigned long error;
	lws_filepos_t len;
	uint8_t *buf;

	if (!method) {
		error = ERR_get_error();
		lwsl_err("problem creating ssl method %lu: %s\n",
			error, ERR_error_string(error,
				      (char *)vh->context->pt[0].serv_buf));
		return 1;
	}
	/* create context */
	vh->tls.ssl_client_ctx = SSL_CTX_new(method);
	if (!vh->tls.ssl_client_ctx) {
		error = ERR_get_error();
		lwsl_err("problem creating ssl context %lu: %s\n",
			error, ERR_error_string(error,
				      (char *)vh->context->pt[0].serv_buf));
		return 1;
	}

	if (!ca_filepath)
		return 0;

	if (alloc_file(vh->context, ca_filepath, &buf, &len)) {
		lwsl_err("Load CA cert file %s failed\n", ca_filepath);
		return 1;
	}

	vh->tls.x509_client_CA = d2i_X509(NULL, buf, len);
	free(buf);
	if (!vh->tls.x509_client_CA) {
		lwsl_err("client CA: x509 parse failed\n");
		return 1;
	}

	if (!vh->tls.ssl_ctx)
		SSL_CTX_add_client_CA(vh->tls.ssl_client_ctx, vh->tls.x509_client_CA);
	else
		SSL_CTX_add_client_CA(vh->tls.ssl_ctx, vh->tls.x509_client_CA);

	lwsl_notice("client loaded CA for verification %s\n", ca_filepath);

	return 0;
}
