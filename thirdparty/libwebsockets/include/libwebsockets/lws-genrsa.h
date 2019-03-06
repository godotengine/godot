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

/*! \defgroup generic RSA
 * ## Generic RSA related functions
 *
 * Lws provides generic RSA functions that abstract the ones
 * provided by whatever OpenSSL library you are linking against.
 *
 * It lets you use the same code if you build against mbedtls or OpenSSL
 * for example.
 */
///@{

enum enum_jwk_tok {
	JWK_KEY_E,
	JWK_KEY_N,
	JWK_KEY_D,
	JWK_KEY_P,
	JWK_KEY_Q,
	JWK_KEY_DP,
	JWK_KEY_DQ,
	JWK_KEY_QI,
	JWK_KTY, /* also serves as count of real elements */
	JWK_KEY,
};

#define LWS_COUNT_RSA_ELEMENTS JWK_KTY

struct lws_genrsa_ctx {
#if defined(LWS_WITH_MBEDTLS)
	mbedtls_rsa_context *ctx;
#else
	BIGNUM *bn[LWS_COUNT_RSA_ELEMENTS];
	RSA *rsa;
#endif
};

struct lws_genrsa_element {
	uint8_t *buf;
	uint16_t len;
};

struct lws_genrsa_elements {
	struct lws_genrsa_element e[LWS_COUNT_RSA_ELEMENTS];
};

/** lws_jwk_destroy_genrsa_elements() - Free allocations in genrsa_elements
 *
 * \param el: your struct lws_genrsa_elements
 *
 * This is a helper for user code making use of struct lws_genrsa_elements
 * where the elements are allocated on the heap, it frees any non-NULL
 * buf element and sets the buf to NULL.
 *
 * NB: lws_genrsa_public_... apis do not need this as they take care of the key
 * creation and destruction themselves.
 */
LWS_VISIBLE LWS_EXTERN void
lws_jwk_destroy_genrsa_elements(struct lws_genrsa_elements *el);

/** lws_genrsa_public_decrypt_create() - Create RSA public decrypt context
 *
 * \param ctx: your struct lws_genrsa_ctx
 * \param el: struct prepared with key element data
 *
 * Creates an RSA context with a public key associated with it, formed from
 * the key elements in \p el.
 *
 * Returns 0 for OK or nonzero for error.
 *
 * This and related APIs operate identically with OpenSSL or mbedTLS backends.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genrsa_create(struct lws_genrsa_ctx *ctx, struct lws_genrsa_elements *el);

/** lws_genrsa_new_keypair() - Create new RSA keypair
 *
 * \param context: your struct lws_context (may be used for RNG)
 * \param ctx: your struct lws_genrsa_ctx
 * \param el: struct to get the new key element data allocated into it
 * \param bits: key size, eg, 4096
 *
 * Creates a new RSA context and generates a new keypair into it, with \p bits
 * bits.
 *
 * Returns 0 for OK or nonzero for error.
 *
 * This and related APIs operate identically with OpenSSL or mbedTLS backends.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genrsa_new_keypair(struct lws_context *context, struct lws_genrsa_ctx *ctx,
		       struct lws_genrsa_elements *el, int bits);

/** lws_genrsa_public_decrypt() - Perform RSA public decryption
 *
 * \param ctx: your struct lws_genrsa_ctx
 * \param in: encrypted input
 * \param in_len: length of encrypted input
 * \param out: decrypted output
 * \param out_max: size of output buffer
 *
 * Performs the decryption.
 *
 * Returns <0 for error, or length of decrypted data.
 *
 * This and related APIs operate identically with OpenSSL or mbedTLS backends.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genrsa_public_decrypt(struct lws_genrsa_ctx *ctx, const uint8_t *in,
			  size_t in_len, uint8_t *out, size_t out_max);

/** lws_genrsa_public_verify() - Perform RSA public verification
 *
 * \param ctx: your struct lws_genrsa_ctx
 * \param in: unencrypted payload (usually a recomputed hash)
 * \param hash_type: one of LWS_GENHASH_TYPE_
 * \param sig: pointer to the signature we received with the payload
 * \param sig_len: length of the signature we are checking in bytes
 *
 * Returns <0 for error, or 0 if signature matches the payload + key.
 *
 * This and related APIs operate identically with OpenSSL or mbedTLS backends.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genrsa_public_verify(struct lws_genrsa_ctx *ctx, const uint8_t *in,
			 enum lws_genhash_types hash_type,
			 const uint8_t *sig, size_t sig_len);

/** lws_genrsa_public_sign() - Create RSA signature
 *
 * \param ctx: your struct lws_genrsa_ctx
 * \param in: precomputed hash
 * \param hash_type: one of LWS_GENHASH_TYPE_
 * \param sig: pointer to buffer to take signature
 * \param sig_len: length of the buffer (must be >= length of key N)
 *
 * Returns <0 for error, or 0 for success.
 *
 * This and related APIs operate identically with OpenSSL or mbedTLS backends.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genrsa_public_sign(struct lws_genrsa_ctx *ctx, const uint8_t *in,
		       enum lws_genhash_types hash_type, uint8_t *sig,
		       size_t sig_len);

/** lws_genrsa_public_decrypt_destroy() - Destroy RSA public decrypt context
 *
 * \param ctx: your struct lws_genrsa_ctx
 *
 * Destroys any allocations related to \p ctx.
 *
 * This and related APIs operate identically with OpenSSL or mbedTLS backends.
 */
LWS_VISIBLE LWS_EXTERN void
lws_genrsa_destroy(struct lws_genrsa_ctx *ctx);

/** lws_genrsa_render_pkey_asn1() - Exports public or private key to ASN1/DER
 *
 * \param ctx: your struct lws_genrsa_ctx
 * \param _private: 0 = public part only, 1 = all parts of the key
 * \param pkey_asn1: pointer to buffer to take the ASN1
 * \param pkey_asn1_len: max size of the pkey_asn1_len
 *
 * Returns length of pkey_asn1 written, or -1 for error.
 */
LWS_VISIBLE LWS_EXTERN int
lws_genrsa_render_pkey_asn1(struct lws_genrsa_ctx *ctx, int _private,
			    uint8_t *pkey_asn1, size_t pkey_asn1_len);
///@}
