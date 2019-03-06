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
 *
 * included from libwebsockets.h
 */

/*! \defgroup generic hash
 * ## Generic Hash related functions
 *
 * Lws provides generic hash / digest accessors that abstract the ones
 * provided by whatever OpenSSL library you are linking against.
 *
 * It lets you use the same code if you build against mbedtls or OpenSSL
 * for example.
 */
///@{

enum lws_genhash_types {
	LWS_GENHASH_TYPE_SHA1,
	LWS_GENHASH_TYPE_SHA256,
	LWS_GENHASH_TYPE_SHA384,
	LWS_GENHASH_TYPE_SHA512,
};

enum lws_genhmac_types {
	LWS_GENHMAC_TYPE_SHA256,
	LWS_GENHMAC_TYPE_SHA384,
	LWS_GENHMAC_TYPE_SHA512,
};

#define LWS_GENHASH_LARGEST 64

struct lws_genhash_ctx {
        uint8_t type;
#if defined(LWS_WITH_MBEDTLS)
        union {
		mbedtls_sha1_context sha1;
		mbedtls_sha256_context sha256;
		mbedtls_sha512_context sha512; /* 384 also uses this */
		const mbedtls_md_info_t *hmac;
        } u;
#else
        const EVP_MD *evp_type;
        EVP_MD_CTX *mdctx;
#endif
};

struct lws_genhmac_ctx {
        uint8_t type;
#if defined(LWS_WITH_MBEDTLS)
	const mbedtls_md_info_t *hmac;
	mbedtls_md_context_t ctx;
#else
        const EVP_MD *evp_type;
        EVP_MD_CTX *ctx;
#endif
};

/** lws_genhash_size() - get hash size in bytes
 *
 * \param type:	one of LWS_GENHASH_TYPE_...
 *
 * Returns number of bytes in this type of hash
 */
LWS_VISIBLE LWS_EXTERN size_t LWS_WARN_UNUSED_RESULT
lws_genhash_size(enum lws_genhash_types type);

/** lws_genhmac_size() - get hash size in bytes
 *
 * \param type:	one of LWS_GENHASH_TYPE_...
 *
 * Returns number of bytes in this type of hmac
 */
LWS_VISIBLE LWS_EXTERN size_t LWS_WARN_UNUSED_RESULT
lws_genhmac_size(enum lws_genhmac_types type);

/** lws_genhash_init() - prepare your struct lws_genhash_ctx for use
 *
 * \param ctx: your struct lws_genhash_ctx
 * \param type:	one of LWS_GENHASH_TYPE_...
 *
 * Initializes the hash context for the type you requested
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_genhash_init(struct lws_genhash_ctx *ctx, enum lws_genhash_types type);

/** lws_genhash_update() - digest len bytes of the buffer starting at in
 *
 * \param ctx: your struct lws_genhash_ctx
 * \param in: start of the bytes to digest
 * \param len: count of bytes to digest
 *
 * Updates the state of your hash context to reflect digesting len bytes from in
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_genhash_update(struct lws_genhash_ctx *ctx, const void *in, size_t len);

/** lws_genhash_destroy() - copy out the result digest and destroy the ctx
 *
 * \param ctx: your struct lws_genhash_ctx
 * \param result: NULL, or where to copy the result hash
 *
 * Finalizes the hash and copies out the digest.  Destroys any allocations such
 * that ctx can safely go out of scope after calling this.
 *
 * NULL result is supported so that you can destroy the ctx cleanly on error
 * conditions, where there is no valid result.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genhash_destroy(struct lws_genhash_ctx *ctx, void *result);

/** lws_genhmac_init() - prepare your struct lws_genhmac_ctx for use
 *
 * \param ctx: your struct lws_genhmac_ctx
 * \param type:	one of LWS_GENHMAC_TYPE_...
 * \param key: pointer to the start of the HMAC key
 * \param key_len: length of the HMAC key
 *
 * Initializes the hash context for the type you requested
 *
 * If the return is nonzero, it failed and there is nothing needing to be
 * destroyed.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_genhmac_init(struct lws_genhmac_ctx *ctx, enum lws_genhmac_types type,
		 const uint8_t *key, size_t key_len);

/** lws_genhmac_update() - digest len bytes of the buffer starting at in
 *
 * \param ctx: your struct lws_genhmac_ctx
 * \param in: start of the bytes to digest
 * \param len: count of bytes to digest
 *
 * Updates the state of your hash context to reflect digesting len bytes from in
 *
 * If the return is nonzero, it failed and needs destroying.
 */
LWS_VISIBLE LWS_EXTERN int LWS_WARN_UNUSED_RESULT
lws_genhmac_update(struct lws_genhmac_ctx *ctx, const void *in, size_t len);

/** lws_genhmac_destroy() - copy out the result digest and destroy the ctx
 *
 * \param ctx: your struct lws_genhmac_ctx
 * \param result: NULL, or where to copy the result hash
 *
 * Finalizes the hash and copies out the digest.  Destroys any allocations such
 * that ctx can safely go out of scope after calling this.
 *
 * NULL result is supported so that you can destroy the ctx cleanly on error
 * conditions, where there is no valid result.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genhmac_destroy(struct lws_genhmac_ctx *ctx, void *result);
///@}
