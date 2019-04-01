/*
 * libwebsockets - generic RSA api hiding the backend
 *
 * Copyright (C) 2017 Andy Green <andy@warmcat.com>
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
 *
 *  lws_genhash provides a hash / hmac abstraction api in lws that works the
 *  same whether you are using openssl or mbedtls hash functions underneath.
 */
#include "core/private.h"

LWS_VISIBLE void
lws_jwk_destroy_genrsa_elements(struct lws_genrsa_elements *el)
{
	int n;

	for (n = 0; n < LWS_COUNT_RSA_ELEMENTS; n++)
		if (el->e[n].buf)
			lws_free_set_NULL(el->e[n].buf);
}

LWS_VISIBLE int
lws_genrsa_create(struct lws_genrsa_ctx *ctx, struct lws_genrsa_elements *el)
{
	int n;

	memset(ctx, 0, sizeof(*ctx));
	ctx->ctx = lws_zalloc(sizeof(*ctx->ctx), "genrsa");
	if (!ctx->ctx)
		return 1;

	mbedtls_rsa_init(ctx->ctx, MBEDTLS_RSA_PKCS_V15, 0);

	{
		mbedtls_mpi *mpi[LWS_COUNT_RSA_ELEMENTS] = {
			&ctx->ctx->E, &ctx->ctx->N, &ctx->ctx->D, &ctx->ctx->P,
			&ctx->ctx->Q, &ctx->ctx->DP, &ctx->ctx->DQ,
			&ctx->ctx->QP,
		};

		for (n = 0; n < LWS_COUNT_RSA_ELEMENTS; n++)
			if (el->e[n].buf &&
			    mbedtls_mpi_read_binary(mpi[n], el->e[n].buf,
					    	    el->e[n].len)) {
				lwsl_notice("mpi load failed\n");
				lws_free_set_NULL(ctx->ctx);

				return -1;
			}
	}

	ctx->ctx->len = el->e[JWK_KEY_N].len;

	return 0;
}

static int
_rngf(void *context, unsigned char *buf, size_t len)
{
	if ((size_t)lws_get_random(context, buf, len) == len)
		return 0;

	return -1;
}

LWS_VISIBLE int
lws_genrsa_new_keypair(struct lws_context *context, struct lws_genrsa_ctx *ctx,
		       struct lws_genrsa_elements *el, int bits)
{
	int n;

	memset(ctx, 0, sizeof(*ctx));
	ctx->ctx = lws_zalloc(sizeof(*ctx->ctx), "genrsa");
	if (!ctx->ctx)
		return -1;

	mbedtls_rsa_init(ctx->ctx, MBEDTLS_RSA_PKCS_V15, 0);

	n = mbedtls_rsa_gen_key(ctx->ctx, _rngf, context, bits, 65537);
	if (n) {
		lwsl_err("mbedtls_rsa_gen_key failed 0x%x\n", -n);
		goto cleanup_1;
	}

	{
		mbedtls_mpi *mpi[LWS_COUNT_RSA_ELEMENTS] = {
			&ctx->ctx->E, &ctx->ctx->N, &ctx->ctx->D, &ctx->ctx->P,
			&ctx->ctx->Q, &ctx->ctx->DP, &ctx->ctx->DQ,
			&ctx->ctx->QP,
		};

		for (n = 0; n < LWS_COUNT_RSA_ELEMENTS; n++)
			if (mbedtls_mpi_size(mpi[n])) {
				el->e[n].buf = lws_malloc(
					mbedtls_mpi_size(mpi[n]), "genrsakey");
				if (!el->e[n].buf)
					goto cleanup;
				el->e[n].len = mbedtls_mpi_size(mpi[n]);
				mbedtls_mpi_write_binary(mpi[n], el->e[n].buf,
							 el->e[n].len);
			}
	}

	return 0;

cleanup:
	for (n = 0; n < LWS_COUNT_RSA_ELEMENTS; n++)
		if (el->e[n].buf)
			lws_free_set_NULL(el->e[n].buf);
cleanup_1:
	lws_free(ctx->ctx);

	return -1;
}

LWS_VISIBLE int
lws_genrsa_public_decrypt(struct lws_genrsa_ctx *ctx, const uint8_t *in,
			   size_t in_len, uint8_t *out, size_t out_max)
{
	size_t olen = 0;
	int n;

	ctx->ctx->len = in_len;
	n = mbedtls_rsa_rsaes_pkcs1_v15_decrypt(ctx->ctx, NULL, NULL,
						MBEDTLS_RSA_PUBLIC,
						&olen, in, out, out_max);
	if (n) {
		lwsl_notice("%s: -0x%x\n", __func__, -n);

		return -1;
	}

	return olen;
}

LWS_VISIBLE int
lws_genrsa_public_encrypt(struct lws_genrsa_ctx *ctx, const uint8_t *in,
			   size_t in_len, uint8_t *out)
{
	int n;

	ctx->ctx->len = in_len;
	n = mbedtls_rsa_rsaes_pkcs1_v15_encrypt(ctx->ctx, NULL, NULL,
						MBEDTLS_RSA_PRIVATE,
						in_len, in, out);
	if (n) {
		lwsl_notice("%s: -0x%x\n", __func__, -n);

		return -1;
	}

	return 0;
}

static int
lws_genrsa_genrsa_hash_to_mbed_hash(enum lws_genhash_types hash_type)
{
	int h = -1;

	switch (hash_type) {
	case LWS_GENHASH_TYPE_SHA1:
		h = MBEDTLS_MD_SHA1;
		break;
	case LWS_GENHASH_TYPE_SHA256:
		h = MBEDTLS_MD_SHA256;
		break;
	case LWS_GENHASH_TYPE_SHA384:
		h = MBEDTLS_MD_SHA384;
		break;
	case LWS_GENHASH_TYPE_SHA512:
		h = MBEDTLS_MD_SHA512;
		break;
	}

	return h;
}

LWS_VISIBLE int
lws_genrsa_public_verify(struct lws_genrsa_ctx *ctx, const uint8_t *in,
			 enum lws_genhash_types hash_type, const uint8_t *sig,
			 size_t sig_len)
{
	int n, h = lws_genrsa_genrsa_hash_to_mbed_hash(hash_type);

	if (h < 0)
		return -1;

	n = mbedtls_rsa_rsassa_pkcs1_v15_verify(ctx->ctx, NULL, NULL,
						MBEDTLS_RSA_PUBLIC,
						h, 0, in, sig);
	if (n < 0) {
		lwsl_notice("%s: -0x%x\n", __func__, -n);

		return -1;
	}

	return n;
}

LWS_VISIBLE int
lws_genrsa_public_sign(struct lws_genrsa_ctx *ctx, const uint8_t *in,
			 enum lws_genhash_types hash_type, uint8_t *sig,
			 size_t sig_len)
{
	int n, h = lws_genrsa_genrsa_hash_to_mbed_hash(hash_type);

	if (h < 0)
		return -1;

	/*
	 * The "sig" buffer must be as large as the size of ctx->N
	 * (eg. 128 bytes if RSA-1024 is used).
	 */
	if (sig_len < ctx->ctx->len)
		return -1;

	n = mbedtls_rsa_rsassa_pkcs1_v15_sign(ctx->ctx, NULL, NULL,
					      MBEDTLS_RSA_PRIVATE, h, 0, in,
					      sig);
	if (n < 0) {
		lwsl_notice("%s: -0x%x\n", __func__, -n);

		return -1;
	}

	return ctx->ctx->len;
}

LWS_VISIBLE int
lws_genrsa_render_pkey_asn1(struct lws_genrsa_ctx *ctx, int _private,
			    uint8_t *pkey_asn1, size_t pkey_asn1_len)
{
	uint8_t *p = pkey_asn1, *totlen, *end = pkey_asn1 + pkey_asn1_len - 1;
	mbedtls_mpi *mpi[LWS_COUNT_RSA_ELEMENTS] = {
		&ctx->ctx->N, &ctx->ctx->E, &ctx->ctx->D, &ctx->ctx->P,
		&ctx->ctx->Q, &ctx->ctx->DP, &ctx->ctx->DQ,
		&ctx->ctx->QP,
	};
	int n;

	/* 30 82  - sequence
	 *   09 29  <-- length(0x0929) less 4 bytes
	 * 02 01 <- length (1)
	 *  00
	 * 02 82
	 *  02 01 <- length (513)  N
	 *  ...
	 *
	 *  02 03 <- length (3) E
	 *    01 00 01
	 *
	 * 02 82
	 *   02 00 <- length (512) D P Q EXP1 EXP2 COEFF
	 *
	 *  */

	*p++ = 0x30;
	*p++ = 0x82;
	totlen = p;
	p += 2;

	*p++ = 0x02;
	*p++ = 0x01;
	*p++ = 0x00;

	for (n = 0; n < LWS_COUNT_RSA_ELEMENTS; n++) {
		int m = mbedtls_mpi_size(mpi[n]);
		uint8_t *elen;

		*p++ = 0x02;
		elen = p;
		if (m < 0x7f)
			*p++ = m;
		else {
			*p++ = 0x82;
			*p++ = m >> 8;
			*p++ = m & 0xff;
		}

		if (p + m > end)
			return -1;

		mbedtls_mpi_write_binary(mpi[n], p, m);
		if (p[0] & 0x80) {
			p[0] = 0x00;
			mbedtls_mpi_write_binary(mpi[n], &p[1], m);
			m++;
		}
		if (m < 0x7f)
			*elen = m;
		else {
			*elen++ = 0x82;
			*elen++ = m >> 8;
			*elen = m & 0xff;
		}
		p += m;
	}

	n = lws_ptr_diff(p, pkey_asn1);

	*totlen++ = (n - 4) >> 8;
	*totlen = (n - 4) & 0xff;

	return n;
}

LWS_VISIBLE void
lws_genrsa_destroy(struct lws_genrsa_ctx *ctx)
{
	if (!ctx->ctx)
		return;
	mbedtls_rsa_free(ctx->ctx);
	lws_free(ctx->ctx);
	ctx->ctx = NULL;
}
