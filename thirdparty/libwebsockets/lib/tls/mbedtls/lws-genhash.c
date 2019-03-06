/*
 * libwebsockets - generic hash and HMAC api hiding the backend
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
#include "libwebsockets.h"
#include <mbedtls/version.h>

#if (MBEDTLS_VERSION_NUMBER >= 0x02070000)
#define MBA(fn) fn##_ret
#else
#define MBA(fn) fn
#endif

size_t
lws_genhash_size(enum lws_genhash_types type)
{
	switch(type) {
	case LWS_GENHASH_TYPE_SHA1:
		return 20;
	case LWS_GENHASH_TYPE_SHA256:
		return 32;
	case LWS_GENHASH_TYPE_SHA384:
		return 48;
	case LWS_GENHASH_TYPE_SHA512:
		return 64;
	}

	return 0;
}

int
lws_genhash_init(struct lws_genhash_ctx *ctx, enum lws_genhash_types type)
{
	ctx->type = type;

	switch (ctx->type) {
	case LWS_GENHASH_TYPE_SHA1:
		mbedtls_sha1_init(&ctx->u.sha1);
		MBA(mbedtls_sha1_starts)(&ctx->u.sha1);
		break;
	case LWS_GENHASH_TYPE_SHA256:
		mbedtls_sha256_init(&ctx->u.sha256);
		MBA(mbedtls_sha256_starts)(&ctx->u.sha256, 0);
		break;
	case LWS_GENHASH_TYPE_SHA384:
		mbedtls_sha512_init(&ctx->u.sha512);
		MBA(mbedtls_sha512_starts)(&ctx->u.sha512, 1 /* is384 */);
		break;
	case LWS_GENHASH_TYPE_SHA512:
		mbedtls_sha512_init(&ctx->u.sha512);
		MBA(mbedtls_sha512_starts)(&ctx->u.sha512, 0);
		break;
	default:
		return 1;
	}

	return 0;
}

int
lws_genhash_update(struct lws_genhash_ctx *ctx, const void *in, size_t len)
{
	switch (ctx->type) {
	case LWS_GENHASH_TYPE_SHA1:
		MBA(mbedtls_sha1_update)(&ctx->u.sha1, in, len);
		break;
	case LWS_GENHASH_TYPE_SHA256:
		MBA(mbedtls_sha256_update)(&ctx->u.sha256, in, len);
		break;
	case LWS_GENHASH_TYPE_SHA384:
		MBA(mbedtls_sha512_update)(&ctx->u.sha512, in, len);
		break;
	case LWS_GENHASH_TYPE_SHA512:
		MBA(mbedtls_sha512_update)(&ctx->u.sha512, in, len);
		break;
	}

	return 0;
}

int
lws_genhash_destroy(struct lws_genhash_ctx *ctx, void *result)
{
	switch (ctx->type) {
	case LWS_GENHASH_TYPE_SHA1:
		MBA(mbedtls_sha1_finish)(&ctx->u.sha1, result);
		mbedtls_sha1_free(&ctx->u.sha1);
		break;
	case LWS_GENHASH_TYPE_SHA256:
		MBA(mbedtls_sha256_finish)(&ctx->u.sha256, result);
		mbedtls_sha256_free(&ctx->u.sha256);
		break;
	case LWS_GENHASH_TYPE_SHA384:
		MBA(mbedtls_sha512_finish)(&ctx->u.sha512, result);
		mbedtls_sha512_free(&ctx->u.sha512);
		break;
	case LWS_GENHASH_TYPE_SHA512:
		MBA(mbedtls_sha512_finish)(&ctx->u.sha512, result);
		mbedtls_sha512_free(&ctx->u.sha512);
		break;
	}

	return 0;
}

size_t
lws_genhmac_size(enum lws_genhmac_types type)
{
	switch(type) {
	case LWS_GENHMAC_TYPE_SHA256:
		return 32;
	case LWS_GENHMAC_TYPE_SHA384:
		return 48;
	case LWS_GENHMAC_TYPE_SHA512:
		return 64;
	}

	return 0;
}

int
lws_genhmac_init(struct lws_genhmac_ctx *ctx, enum lws_genhmac_types type,
		 const uint8_t *key, size_t key_len)
{
	int t;

	ctx->type = type;

	switch (type) {
	case LWS_GENHMAC_TYPE_SHA256:
		t = MBEDTLS_MD_SHA256;
		break;
	case LWS_GENHMAC_TYPE_SHA384:
		t = MBEDTLS_MD_SHA384;
		break;
	case LWS_GENHMAC_TYPE_SHA512:
		t = MBEDTLS_MD_SHA512;
		break;
	default:
		return -1;
	}

	ctx->hmac = mbedtls_md_info_from_type(t);
	if (!ctx->hmac)
		return -1;

	if (mbedtls_md_init_ctx(&ctx->ctx, ctx->hmac))
		return -1;

	if (mbedtls_md_hmac_starts(&ctx->ctx, key, key_len)) {
		mbedtls_md_free(&ctx->ctx);
		ctx->hmac = NULL;

		return -1;
	}

	return 0;
}

int
lws_genhmac_update(struct lws_genhmac_ctx *ctx, const void *in, size_t len)
{
	if (mbedtls_md_hmac_update(&ctx->ctx, in, len))
		return -1;

	return 0;
}

int
lws_genhmac_destroy(struct lws_genhmac_ctx *ctx, void *result)
{
	int n = 0;

	if (result)
		n = mbedtls_md_hmac_finish(&ctx->ctx, result);

	mbedtls_md_free(&ctx->ctx);
	ctx->hmac = NULL;
	if (n)
		return -1;

	return 0;
}
