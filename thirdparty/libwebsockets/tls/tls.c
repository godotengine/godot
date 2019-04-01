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

#include "core/private.h"

/*
 * fakes POLLIN on all tls guys with buffered rx
 *
 * returns nonzero if any tls guys had POLLIN faked
 */

int
lws_tls_fake_POLLIN_for_buffered(struct lws_context_per_thread *pt)
{
	struct lws *wsi, *wsi_next;
	int ret = 0;

	wsi = pt->tls.pending_read_list;
	while (wsi && wsi->position_in_fds_table != LWS_NO_FDS_POS) {
		wsi_next = wsi->tls.pending_read_list_next;
		pt->fds[wsi->position_in_fds_table].revents |=
			pt->fds[wsi->position_in_fds_table].events & LWS_POLLIN;
		ret |= pt->fds[wsi->position_in_fds_table].revents & LWS_POLLIN;

		wsi = wsi_next;
	}

	return !!ret;
}

void
__lws_ssl_remove_wsi_from_buffered_list(struct lws *wsi)
{
	struct lws_context *context = wsi->context;
	struct lws_context_per_thread *pt = &context->pt[(int)wsi->tsi];

	if (!wsi->tls.pending_read_list_prev &&
	    !wsi->tls.pending_read_list_next &&
	    pt->tls.pending_read_list != wsi)
		/* we are not on the list */
		return;

	/* point previous guy's next to our next */
	if (!wsi->tls.pending_read_list_prev)
		pt->tls.pending_read_list = wsi->tls.pending_read_list_next;
	else
		wsi->tls.pending_read_list_prev->tls.pending_read_list_next =
			wsi->tls.pending_read_list_next;

	/* point next guy's previous to our previous */
	if (wsi->tls.pending_read_list_next)
		wsi->tls.pending_read_list_next->tls.pending_read_list_prev =
			wsi->tls.pending_read_list_prev;

	wsi->tls.pending_read_list_prev = NULL;
	wsi->tls.pending_read_list_next = NULL;
}

void
lws_ssl_remove_wsi_from_buffered_list(struct lws *wsi)
{
	struct lws_context_per_thread *pt = &wsi->context->pt[(int)wsi->tsi];

	lws_pt_lock(pt, __func__);
	__lws_ssl_remove_wsi_from_buffered_list(wsi);
	lws_pt_unlock(pt);
}

#if defined(LWS_WITH_ESP32)
int alloc_file(struct lws_context *context, const char *filename, uint8_t **buf,
	       lws_filepos_t *amount)
{
	nvs_handle nvh;
	size_t s;
	int n = 0;

	ESP_ERROR_CHECK(nvs_open("lws-station", NVS_READWRITE, &nvh));
	if (nvs_get_blob(nvh, filename, NULL, &s) != ESP_OK) {
		n = 1;
		goto bail;
	}
	*buf = lws_malloc(s + 1, "alloc_file");
	if (!*buf) {
		n = 2;
		goto bail;
	}
	if (nvs_get_blob(nvh, filename, (char *)*buf, &s) != ESP_OK) {
		lws_free(*buf);
		n = 1;
		goto bail;
	}

	*amount = s;
	(*buf)[s] = '\0';

	lwsl_notice("%s: nvs: read %s, %d bytes\n", __func__, filename, (int)s);

bail:
	nvs_close(nvh);

	return n;
}
#else
int alloc_file(struct lws_context *context, const char *filename, uint8_t **buf,
		lws_filepos_t *amount)
{
	FILE *f;
	size_t s;
	int n = 0;

	f = fopen(filename, "rb");
	if (f == NULL) {
		n = 1;
		goto bail;
	}

	if (fseek(f, 0, SEEK_END) != 0) {
		n = 1;
		goto bail;
	}

	s = ftell(f);
	if (s == (size_t)-1) {
		n = 1;
		goto bail;
	}

	if (fseek(f, 0, SEEK_SET) != 0) {
		n = 1;
		goto bail;
	}

	*buf = lws_malloc(s, "alloc_file");
	if (!*buf) {
		n = 2;
		goto bail;
	}

	if (fread(*buf, s, 1, f) != 1) {
		lws_free(*buf);
		n = 1;
		goto bail;
	}

	*amount = s;

bail:
	if (f)
		fclose(f);

	return n;

}
#endif

int
lws_tls_alloc_pem_to_der_file(struct lws_context *context, const char *filename,
			const char *inbuf, lws_filepos_t inlen,
		      uint8_t **buf, lws_filepos_t *amount)
{
	const uint8_t *pem, *p, *end;
	uint8_t *q;
	lws_filepos_t len;
	int n;

	if (filename) {
		n = alloc_file(context, filename, (uint8_t **)&pem, &len);
		if (n)
			return n;
	} else {
		pem = (const uint8_t *)inbuf;
		len = inlen;
	}

	/* trim the first line */

	p = pem;
	end = p + len;
	if (strncmp((char *)p, "-----", 5))
		goto bail;
	p += 5;
	while (p < end && *p != '\n' && *p != '-')
		p++;

	if (*p != '-')
		goto bail;

	while (p < end && *p != '\n')
		p++;

	if (p >= end)
		goto bail;

	p++;

	/* trim the last line */

	q = (uint8_t *)end - 2;

	while (q > pem && *q != '\n')
		q--;

	if (*q != '\n')
		goto bail;

	*q = '\0';

	*amount = lws_b64_decode_string((char *)p, (char *)pem,
					(int)(long long)len);
	*buf = (uint8_t *)pem;

	return 0;

bail:
	lws_free((uint8_t *)pem);

	return 4;
}

int
lws_tls_check_cert_lifetime(struct lws_vhost *v)
{
	union lws_tls_cert_info_results ir;
	time_t now = (time_t)lws_now_secs(), life = 0;
	struct lws_acme_cert_aging_args caa;
	int n;

	if (v->tls.ssl_ctx && !v->tls.skipped_certs) {

		if (now < 1464083026) /* May 2016 */
			/* our clock is wrong and we can't judge the certs */
			return -1;

		n = lws_tls_vhost_cert_info(v, LWS_TLS_CERT_INFO_VALIDITY_TO, &ir, 0);
		if (n)
			return 1;

		life = (ir.time - now) / (24 * 3600);
		lwsl_notice("   vhost %s: cert expiry: %dd\n", v->name, (int)life);
	} else
		lwsl_notice("   vhost %s: no cert\n", v->name);

	memset(&caa, 0, sizeof(caa));
	caa.vh = v;
	lws_broadcast(v->context, LWS_CALLBACK_VHOST_CERT_AGING, (void *)&caa,
		      (size_t)(ssize_t)life);

	return 0;
}

int
lws_tls_check_all_cert_lifetimes(struct lws_context *context)
{
	struct lws_vhost *v = context->vhost_list;

	while (v) {
		if (lws_tls_check_cert_lifetime(v) < 0)
			return -1;
		v = v->vhost_next;
	}

	return 0;
}
#if !defined(LWS_WITH_ESP32) && !defined(LWS_PLAT_OPTEE)
static int
lws_tls_extant(const char *name)
{
	/* it exists if we can open it... */
	int fd = open(name, O_RDONLY), n;
	char buf[1];

	if (fd < 0)
		return 1;

	/* and we can read at least one byte out of it */
	n = read(fd, buf, 1);
	close(fd);

	return n != 1;
}
#endif
/*
 * Returns 0 if the filepath "name" exists and can be read from.
 *
 * In addition, if "name".upd exists, backup "name" to "name.old.1"
 * and rename "name".upd to "name" before reporting its existence.
 *
 * There are four situations and three results possible:
 *
 * 1) LWS_TLS_EXTANT_NO: There are no certs at all (we are waiting for them to
 *    be provisioned).  We also feel like this if we need privs we don't have
 *    any more to look in the directory.
 *
 * 2) There are provisioned certs written (xxx.upd) and we still have root
 *    privs... in this case we rename any existing cert to have a backup name
 *    and move the upd cert into place with the correct name.  This then becomes
 *    situation 4 for the caller.
 *
 * 3) LWS_TLS_EXTANT_ALTERNATIVE: There are provisioned certs written (xxx.upd)
 *    but we no longer have the privs needed to read or rename them.  In this
 *    case, indicate that the caller should use temp copies if any we do have
 *    rights to access.  This is normal after we have updated the cert.
 *
 *    But if we dropped privs, we can't detect the provisioned xxx.upd cert +
 *    key, because we can't see in the dir.  So we have to upgrade NO to
 *    ALTERNATIVE when we actually have the in-memory alternative.
 *
 * 4) LWS_TLS_EXTANT_YES: The certs are present with the correct name and we
 *    have the rights to read them.
 */
enum lws_tls_extant
lws_tls_use_any_upgrade_check_extant(const char *name)
{
#if !defined(LWS_PLAT_OPTEE)

	int n;

#if !defined(LWS_WITH_ESP32)
	char buf[256];

	lws_snprintf(buf, sizeof(buf) - 1, "%s.upd", name);
	if (!lws_tls_extant(buf)) {
		/* ah there is an updated file... how about the desired file? */
		if (!lws_tls_extant(name)) {
			/* rename the desired file */
			for (n = 0; n < 50; n++) {
				lws_snprintf(buf, sizeof(buf) - 1,
					     "%s.old.%d", name, n);
				if (!rename(name, buf))
					break;
			}
			if (n == 50) {
				lwsl_notice("unable to rename %s\n", name);

				return LWS_TLS_EXTANT_ALTERNATIVE;
			}
			lws_snprintf(buf, sizeof(buf) - 1, "%s.upd", name);
		}
		/* desired file is out of the way, rename the updated file */
		if (rename(buf, name)) {
			lwsl_notice("unable to rename %s to %s\n", buf, name);

			return LWS_TLS_EXTANT_ALTERNATIVE;
		}
	}

	if (lws_tls_extant(name))
		return LWS_TLS_EXTANT_NO;
#else
	nvs_handle nvh;
	size_t s = 8192;

	if (nvs_open("lws-station", NVS_READWRITE, &nvh)) {
		lwsl_notice("%s: can't open nvs\n", __func__);
		return LWS_TLS_EXTANT_NO;
	}

	n = nvs_get_blob(nvh, name, NULL, &s);
	nvs_close(nvh);

	if (n)
		return LWS_TLS_EXTANT_NO;
#endif
#endif
	return LWS_TLS_EXTANT_YES;
}

/*
 * LWS_TLS_EXTANT_NO         : skip adding the cert
 * LWS_TLS_EXTANT_YES        : use the cert and private key paths normally
 * LWS_TLS_EXTANT_ALTERNATIVE: normal paths not usable, try alternate if poss
 */
enum lws_tls_extant
lws_tls_generic_cert_checks(struct lws_vhost *vhost, const char *cert,
			    const char *private_key)
{
	int n, m;

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

	if (!cert || !private_key)
		return LWS_TLS_EXTANT_NO;

	n = lws_tls_use_any_upgrade_check_extant(cert);
	if (n == LWS_TLS_EXTANT_ALTERNATIVE)
		return LWS_TLS_EXTANT_ALTERNATIVE;
	m = lws_tls_use_any_upgrade_check_extant(private_key);
	if (m == LWS_TLS_EXTANT_ALTERNATIVE)
		return LWS_TLS_EXTANT_ALTERNATIVE;

	if ((n == LWS_TLS_EXTANT_NO || m == LWS_TLS_EXTANT_NO) &&
	    (vhost->options & LWS_SERVER_OPTION_IGNORE_MISSING_CERT)) {
		lwsl_notice("Ignoring missing %s or %s\n", cert, private_key);
		vhost->tls.skipped_certs = 1;

		return LWS_TLS_EXTANT_NO;
	}

	/*
	 * the cert + key exist
	 */

	return LWS_TLS_EXTANT_YES;
}

#if !defined(LWS_NO_SERVER)
/*
 * update the cert for every vhost using the given path
 */

LWS_VISIBLE int
lws_tls_cert_updated(struct lws_context *context, const char *certpath,
		     const char *keypath,
		     const char *mem_cert, size_t len_mem_cert,
		     const char *mem_privkey, size_t len_mem_privkey)
{
	struct lws wsi;

	wsi.context = context;

	lws_start_foreach_ll(struct lws_vhost *, v, context->vhost_list) {
		wsi.vhost = v;
		if (v->tls.alloc_cert_path && v->tls.key_path &&
		    !strcmp(v->tls.alloc_cert_path, certpath) &&
		    !strcmp(v->tls.key_path, keypath)) {
			lws_tls_server_certs_load(v, &wsi, certpath, keypath,
						  mem_cert, len_mem_cert,
						  mem_privkey, len_mem_privkey);

			if (v->tls.skipped_certs)
				lwsl_notice("%s: vhost %s: cert unset\n",
					    __func__, v->name);
		}
	} lws_end_foreach_ll(v, vhost_next);

	return 0;
}
#endif

int
lws_gate_accepts(struct lws_context *context, int on)
{
	struct lws_vhost *v = context->vhost_list;

	lwsl_notice("%s: on = %d\n", __func__, on);

#if defined(LWS_WITH_STATS)
	context->updated = 1;
#endif

	while (v) {
		if (v->tls.use_ssl && v->lserv_wsi &&
		    lws_change_pollfd(v->lserv_wsi, (LWS_POLLIN) * !on,
				      (LWS_POLLIN) * on))
			lwsl_notice("Unable to set accept POLLIN %d\n", on);

		v = v->vhost_next;
	}

	return 0;
}

/* comma-separated alpn list, like "h2,http/1.1" to openssl alpn format */

int
lws_alpn_comma_to_openssl(const char *comma, uint8_t *os, int len)
{
	uint8_t *oos = os, *plen = NULL;

	while (*comma && len > 1) {
		if (!plen && *comma == ' ') {
			comma++;
			continue;
		}
		if (!plen) {
			plen = os++;
			len--;
		}

		if (*comma == ',') {
			*plen = lws_ptr_diff(os, plen + 1);
			plen = NULL;
			comma++;
		} else {
			*os++ = *comma++;
			len--;
		}
	}

	if (plen)
		*plen = lws_ptr_diff(os, plen + 1);

	return lws_ptr_diff(os, oos);
}

