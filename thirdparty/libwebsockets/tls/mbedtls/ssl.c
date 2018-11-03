/*
 * libwebsockets - mbedTLS-specific lws apis
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
#include <mbedtls/oid.h>

void
lws_ssl_elaborate_error(void)
{
}

int
lws_context_init_ssl_library(const struct lws_context_creation_info *info)
{
	lwsl_info(" Compiled with MbedTLS support\n");

	if (!lws_check_opt(info->options, LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT))
		lwsl_info(" SSL disabled: no LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT\n");

	return 0;
}

LWS_VISIBLE void
lws_ssl_destroy(struct lws_vhost *vhost)
{
	if (!lws_check_opt(vhost->context->options,
			   LWS_SERVER_OPTION_DO_SSL_GLOBAL_INIT))
		return;

	if (vhost->tls.ssl_ctx)
		SSL_CTX_free(vhost->tls.ssl_ctx);
	if (!vhost->tls.user_supplied_ssl_ctx && vhost->tls.ssl_client_ctx)
		SSL_CTX_free(vhost->tls.ssl_client_ctx);

	if (vhost->tls.x509_client_CA)
		X509_free(vhost->tls.x509_client_CA);
}

LWS_VISIBLE int
lws_ssl_capable_read(struct lws *wsi, unsigned char *buf, int len)
{
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];
	int n = 0, m;

	if (!wsi->tls.ssl)
		return lws_ssl_capable_read_no_ssl(wsi, buf, len);

	lws_stats_atomic_bump(context, pt, LWSSTATS_C_API_READ, 1);

	errno = 0;
	n = SSL_read(wsi->tls.ssl, buf, len);
#if defined(LWS_WITH_ESP32)
	if (!n && errno == ENOTCONN) {
		lwsl_debug("%p: SSL_read ENOTCONN\n", wsi);
		return LWS_SSL_CAPABLE_ERROR;
	}
#endif
#if defined(LWS_WITH_STATS)
	if (!wsi->seen_rx) {
                lws_stats_atomic_bump(wsi->context, pt,
                		      LWSSTATS_MS_SSL_RX_DELAY,
				time_in_microseconds() - wsi->accept_start_us);
                lws_stats_atomic_bump(wsi->context, pt,
                		      LWSSTATS_C_SSL_CONNS_HAD_RX, 1);
		wsi->seen_rx = 1;
	}
#endif


	lwsl_debug("%p: SSL_read says %d\n", wsi, n);
	/* manpage: returning 0 means connection shut down */
	if (!n) {
		wsi->socket_is_permanently_unusable = 1;

		return LWS_SSL_CAPABLE_ERROR;
	}

	if (n < 0) {
		m = SSL_get_error(wsi->tls.ssl, n);
		lwsl_debug("%p: ssl err %d errno %d\n", wsi, m, errno);
		if (m == SSL_ERROR_ZERO_RETURN ||
		    m == SSL_ERROR_SYSCALL)
			return LWS_SSL_CAPABLE_ERROR;

		if (m == SSL_ERROR_WANT_READ || SSL_want_read(wsi->tls.ssl)) {
			lwsl_debug("%s: WANT_READ\n", __func__);
			lwsl_debug("%p: LWS_SSL_CAPABLE_MORE_SERVICE\n", wsi);
			return LWS_SSL_CAPABLE_MORE_SERVICE;
		}
		if (m == SSL_ERROR_WANT_WRITE || SSL_want_write(wsi->tls.ssl)) {
			lwsl_debug("%s: WANT_WRITE\n", __func__);
			lwsl_debug("%p: LWS_SSL_CAPABLE_MORE_SERVICE\n", wsi);
			return LWS_SSL_CAPABLE_MORE_SERVICE;
		}
		wsi->socket_is_permanently_unusable = 1;

		return LWS_SSL_CAPABLE_ERROR;
	}

	lws_stats_atomic_bump(context, pt, LWSSTATS_B_READ, n);

	if (wsi->vhost)
		wsi->vhost->conn_stats.rx += n;

	/*
	 * if it was our buffer that limited what we read,
	 * check if SSL has additional data pending inside SSL buffers.
	 *
	 * Because these won't signal at the network layer with POLLIN
	 * and if we don't realize, this data will sit there forever
	 */
	if (n != len)
		goto bail;
	if (!wsi->tls.ssl)
		goto bail;

	if (!SSL_pending(wsi->tls.ssl))
		goto bail;

	if (wsi->tls.pending_read_list_next)
		return n;
	if (wsi->tls.pending_read_list_prev)
		return n;
	if (pt->tls.pending_read_list == wsi)
		return n;

	/* add us to the linked list of guys with pending ssl */
	if (pt->tls.pending_read_list)
		pt->tls.pending_read_list->tls.pending_read_list_prev = wsi;

	wsi->tls.pending_read_list_next = pt->tls.pending_read_list;
	wsi->tls.pending_read_list_prev = NULL;
	pt->tls.pending_read_list = wsi;

	return n;
bail:
	lws_ssl_remove_wsi_from_buffered_list(wsi);

	return n;
}

LWS_VISIBLE int
lws_ssl_pending(struct lws *wsi)
{
	if (!wsi->tls.ssl)
		return 0;

	return SSL_pending(wsi->tls.ssl);
}

LWS_VISIBLE int
lws_ssl_capable_write(struct lws *wsi, unsigned char *buf, int len)
{
	int n, m;

	if (!wsi->tls.ssl)
		return lws_ssl_capable_write_no_ssl(wsi, buf, len);

	n = SSL_write(wsi->tls.ssl, buf, len);
	if (n > 0)
		return n;

	m = SSL_get_error(wsi->tls.ssl, n);
	if (m != SSL_ERROR_SYSCALL) {
		if (m == SSL_ERROR_WANT_READ || SSL_want_read(wsi->tls.ssl)) {
			lwsl_notice("%s: want read\n", __func__);

			return LWS_SSL_CAPABLE_MORE_SERVICE;
		}

		if (m == SSL_ERROR_WANT_WRITE || SSL_want_write(wsi->tls.ssl)) {
			lws_set_blocking_send(wsi);
			lwsl_notice("%s: want write\n", __func__);

			return LWS_SSL_CAPABLE_MORE_SERVICE;
		}
	}

	lwsl_debug("%s failed: %d\n",__func__, m);
	wsi->socket_is_permanently_unusable = 1;

	return LWS_SSL_CAPABLE_ERROR;
}

int openssl_SSL_CTX_private_data_index;

void
lws_ssl_info_callback(const SSL *ssl, int where, int ret)
{
	struct lws *wsi;
	struct lws_context *context;
	struct lws_ssl_info si;

	context = (struct lws_context *)SSL_CTX_get_ex_data(
					SSL_get_SSL_CTX(ssl),
					openssl_SSL_CTX_private_data_index);
	if (!context)
		return;
	wsi = wsi_from_fd(context, SSL_get_fd(ssl));
	if (!wsi)
		return;

	if (!(where & wsi->vhost->tls.ssl_info_event_mask))
		return;

	si.where = where;
	si.ret = ret;

	if (user_callback_handle_rxflow(wsi->protocol->callback,
					wsi, LWS_CALLBACK_SSL_INFO,
					wsi->user_space, &si, 0))
		lws_set_timeout(wsi, PENDING_TIMEOUT_KILLED_BY_SSL_INFO, -1);
}


LWS_VISIBLE int
lws_ssl_close(struct lws *wsi)
{
	lws_sockfd_type n;

	if (!wsi->tls.ssl)
		return 0; /* not handled */

#if defined (LWS_HAVE_SSL_SET_INFO_CALLBACK)
	/* kill ssl callbacks, becausse we will remove the fd from the
	 * table linking it to the wsi
	 */
	if (wsi->vhost->tls.ssl_info_event_mask)
		SSL_set_info_callback(wsi->tls.ssl, NULL);
#endif

	n = SSL_get_fd(wsi->tls.ssl);
	if (!wsi->socket_is_permanently_unusable)
		SSL_shutdown(wsi->tls.ssl);
	compatible_close(n);
	SSL_free(wsi->tls.ssl);
	wsi->tls.ssl = NULL;

	if (!lwsi_role_client(wsi) &&
	    wsi->context->simultaneous_ssl_restriction &&
	    wsi->context->simultaneous_ssl-- ==
			    wsi->context->simultaneous_ssl_restriction)
		/* we made space and can do an accept */
		lws_gate_accepts(wsi->context, 1);

#if defined(LWS_WITH_STATS)
	wsi->context->updated = 1;
#endif

	return 1; /* handled */
}

void
lws_ssl_SSL_CTX_destroy(struct lws_vhost *vhost)
{
	if (vhost->tls.ssl_ctx)
		SSL_CTX_free(vhost->tls.ssl_ctx);

	if (!vhost->tls.user_supplied_ssl_ctx && vhost->tls.ssl_client_ctx)
		SSL_CTX_free(vhost->tls.ssl_client_ctx);
#if defined(LWS_WITH_ACME)
	lws_tls_acme_sni_cert_destroy(vhost);
#endif
}

void
lws_ssl_context_destroy(struct lws_context *context)
{
}

lws_tls_ctx *
lws_tls_ctx_from_wsi(struct lws *wsi)
{
	if (!wsi->tls.ssl)
		return NULL;

	return SSL_get_SSL_CTX(wsi->tls.ssl);
}

enum lws_ssl_capable_status
__lws_tls_shutdown(struct lws *wsi)
{
	int n = SSL_shutdown(wsi->tls.ssl);

	lwsl_debug("SSL_shutdown=%d for fd %d\n", n, wsi->desc.sockfd);

	switch (n) {
	case 1: /* successful completion */
		n = shutdown(wsi->desc.sockfd, SHUT_WR);
		return LWS_SSL_CAPABLE_DONE;

	case 0: /* needs a retry */
		__lws_change_pollfd(wsi, 0, LWS_POLLIN);
		return LWS_SSL_CAPABLE_MORE_SERVICE;

	default: /* fatal error, or WANT */
		n = SSL_get_error(wsi->tls.ssl, n);
		if (n != SSL_ERROR_SYSCALL && n != SSL_ERROR_SSL) {
			if (SSL_want_read(wsi->tls.ssl)) {
				lwsl_debug("(wants read)\n");
				__lws_change_pollfd(wsi, 0, LWS_POLLIN);
				return LWS_SSL_CAPABLE_MORE_SERVICE_READ;
			}
			if (SSL_want_write(wsi->tls.ssl)) {
				lwsl_debug("(wants write)\n");
				__lws_change_pollfd(wsi, 0, LWS_POLLOUT);
				return LWS_SSL_CAPABLE_MORE_SERVICE_WRITE;
			}
		}
		return LWS_SSL_CAPABLE_ERROR;
	}
}

static time_t
lws_tls_mbedtls_time_to_unix(mbedtls_x509_time *xtime)
{
	struct tm t;

	if (!xtime || !xtime->year || xtime->year < 0)
		return (time_t)(long long)-1;

	memset(&t, 0, sizeof(t));

	t.tm_year = xtime->year - 1900;
	t.tm_mon = xtime->mon - 1; /* mbedtls months are 1+, tm are 0+ */
	t.tm_mday = xtime->day - 1; /* mbedtls days are 1+, tm are 0+ */
	t.tm_hour = xtime->hour;
	t.tm_min = xtime->min;
	t.tm_sec = xtime->sec;
	t.tm_isdst = -1;

	return mktime(&t);
}

static int
lws_tls_mbedtls_get_x509_name(mbedtls_x509_name *name,
			      union lws_tls_cert_info_results *buf, size_t len)
{
	while (name) {
		if (MBEDTLS_OID_CMP(MBEDTLS_OID_AT_CN, &name->oid)) {
			name = name->next;
			continue;
		}

		if (len - 1 < name->val.len)
			return -1;

		memcpy(&buf->ns.name[0], name->val.p, name->val.len);
		buf->ns.name[name->val.len] = '\0';
		buf->ns.len = name->val.len;

		return 0;
	}

	return -1;
}

static int
lws_tls_mbedtls_cert_info(mbedtls_x509_crt *x509, enum lws_tls_cert_info type,
			  union lws_tls_cert_info_results *buf, size_t len)
{
	if (!x509)
		return -1;

	switch (type) {
	case LWS_TLS_CERT_INFO_VALIDITY_FROM:
		buf->time = lws_tls_mbedtls_time_to_unix(&x509->valid_from);
		if (buf->time == (time_t)(long long)-1)
			return -1;
		break;

	case LWS_TLS_CERT_INFO_VALIDITY_TO:
		buf->time = lws_tls_mbedtls_time_to_unix(&x509->valid_to);
		if (buf->time == (time_t)(long long)-1)
			return -1;
		break;

	case LWS_TLS_CERT_INFO_COMMON_NAME:
		return lws_tls_mbedtls_get_x509_name(&x509->subject, buf, len);

	case LWS_TLS_CERT_INFO_ISSUER_NAME:
		return lws_tls_mbedtls_get_x509_name(&x509->issuer, buf, len);

	case LWS_TLS_CERT_INFO_USAGE:
		buf->usage = x509->key_usage;
		break;

	case LWS_TLS_CERT_INFO_OPAQUE_PUBLIC_KEY:
	{
		char *p = buf->ns.name;
		size_t r = len, u;

		switch (mbedtls_pk_get_type(&x509->pk)) {
		case MBEDTLS_PK_RSA:
		{
			mbedtls_rsa_context *rsa = mbedtls_pk_rsa(x509->pk);

			if (mbedtls_mpi_write_string(&rsa->N, 16, p, r, &u))
				return -1;
			r -= u;
			p += u;
			if (mbedtls_mpi_write_string(&rsa->E, 16, p, r, &u))
				return -1;

			p += u;
			buf->ns.len = lws_ptr_diff(p, buf->ns.name);
			break;
		}
		case MBEDTLS_PK_ECKEY:
		{
			mbedtls_ecp_keypair *ecp = mbedtls_pk_ec(x509->pk);

			if (mbedtls_mpi_write_string(&ecp->Q.X, 16, p, r, &u))
				 return -1;
			r -= u;
			p += u;
			if (mbedtls_mpi_write_string(&ecp->Q.Y, 16, p, r, &u))
				 return -1;
			r -= u;
			p += u;
			if (mbedtls_mpi_write_string(&ecp->Q.Z, 16, p, r, &u))
				 return -1;
			p += u;
			buf->ns.len = lws_ptr_diff(p, buf->ns.name);
			break;
		}
		default:
			lwsl_notice("%s: x509 has unsupported pubkey type %d\n",
				    __func__,
				    mbedtls_pk_get_type(&x509->pk));

			return -1;
		}
		break;
	}

	default:
		return -1;
	}

	return 0;
}

LWS_VISIBLE LWS_EXTERN int
lws_tls_vhost_cert_info(struct lws_vhost *vhost, enum lws_tls_cert_info type,
		        union lws_tls_cert_info_results *buf, size_t len)
{
	mbedtls_x509_crt *x509 = ssl_ctx_get_mbedtls_x509_crt(vhost->tls.ssl_ctx);

	return lws_tls_mbedtls_cert_info(x509, type, buf, len);
}

LWS_VISIBLE int
lws_tls_peer_cert_info(struct lws *wsi, enum lws_tls_cert_info type,
		       union lws_tls_cert_info_results *buf, size_t len)
{
	mbedtls_x509_crt *x509;

	wsi = lws_get_network_wsi(wsi);

	x509 = ssl_get_peer_mbedtls_x509_crt(wsi->tls.ssl);

	if (!x509)
		return -1;

	switch (type) {
	case LWS_TLS_CERT_INFO_VERIFIED:
		buf->verified = SSL_get_verify_result(wsi->tls.ssl) == X509_V_OK;
		return 0;
	default:
		return lws_tls_mbedtls_cert_info(x509, type, buf, len);
	}

	return -1;
}

static int
tops_fake_POLLIN_for_buffered_mbedtls(struct lws_context_per_thread *pt)
{
	return lws_tls_fake_POLLIN_for_buffered(pt);
}

static int
tops_periodic_housekeeping_mbedtls(struct lws_context *context, time_t now)
{
	int n;

	n = lws_compare_time_t(context, now, context->tls.last_cert_check_s);
	if ((!context->tls.last_cert_check_s || n > (24 * 60 * 60)) &&
	    !lws_tls_check_all_cert_lifetimes(context))
		context->tls.last_cert_check_s = now;

	return 0;
}

const struct lws_tls_ops tls_ops_mbedtls = {
	/* fake_POLLIN_for_buffered */	tops_fake_POLLIN_for_buffered_mbedtls,
	/* periodic_housekeeping */	tops_periodic_housekeeping_mbedtls,
};
