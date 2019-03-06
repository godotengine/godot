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

/*! \defgroup jws JSON Web Signature
 * ## JSON Web Signature API
 *
 * Lws provides an API to check and create RFC7515 JSON Web Signatures
 *
 * SHA256/384/512 HMAC, and RSA 256/384/512 are supported.
 *
 * The API uses your TLS library crypto, but works exactly the same no matter
 * what you TLS backend is.
 */
///@{

LWS_VISIBLE LWS_EXTERN int
lws_jws_confirm_sig(const char *in, size_t len, struct lws_jwk *jwk);

/**
 * lws_jws_sign_from_b64() - add b64 sig to b64 hdr + payload
 *
 * \param b64_hdr: protected header encoded in b64, may be NULL
 * \param hdr_len: bytes in b64 coding of protected header
 * \param b64_pay: payload encoded in b64
 * \param pay_len: bytes in b64 coding of payload
 * \param b64_sig: buffer to write the b64 encoded signature into
 * \param sig_len: max bytes we can write at b64_sig
 * \param hash_type: one of LWS_GENHASH_TYPE_SHA[256|384|512]
 * \param jwk: the struct lws_jwk containing the signing key
 *
 * This adds a b64-coded JWS signature of the b64-encoded protected header
 * and b64-encoded payload, at \p b64_sig.  The signature will be as large
 * as the N element of the RSA key when the RSA key is used, eg, 512 bytes for
 * a 4096-bit key, and then b64-encoding on top.
 *
 * In some special cases, there is only payload to sign and no header, in that
 * case \p b64_hdr may be NULL, and only the payload will be hashed before
 * signing.
 *
 * Returns the length of the encoded signature written to \p b64_sig, or -1.
 */
LWS_VISIBLE LWS_EXTERN int
lws_jws_sign_from_b64(const char *b64_hdr, size_t hdr_len, const char *b64_pay,
		      size_t pay_len, char *b64_sig, size_t sig_len,
		      enum lws_genhash_types hash_type, struct lws_jwk *jwk);

/**
 * lws_jws_create_packet() - add b64 sig to b64 hdr + payload
 *
 * \param jwk: the struct lws_jwk containing the signing key
 * \param payload: unencoded payload JSON
 * \param len: length of unencoded payload JSON
 * \param nonce: Nonse string to include in protected header
 * \param out: buffer to take signed packet
 * \param out_len: size of \p out buffer
 *
 * This creates a "flattened" JWS packet from the jwk and the plaintext
 * payload, and signs it.  The packet is written into \p out.
 *
 * This does the whole packet assembly and signing, calling through to
 * lws_jws_sign_from_b64() as part of the process.
 *
 * Returns the length written to \p out, or -1.
 */
LWS_VISIBLE LWS_EXTERN int
lws_jws_create_packet(struct lws_jwk *jwk, const char *payload, size_t len,
		      const char *nonce, char *out, size_t out_len);

/**
 * lws_jws_base64_enc() - encode input data into b64url data
 *
 * \param in: the incoming plaintext
 * \param in_len: the length of the incoming plaintext in bytes
 * \param out: the buffer to store the b64url encoded data to
 * \param out_max: the length of \p out in bytes
 *
 * Returns either -1 if problems, or the number of bytes written to \p out.
 */
LWS_VISIBLE LWS_EXTERN int
lws_jws_base64_enc(const char *in, size_t in_len, char *out, size_t out_max);
///@}
