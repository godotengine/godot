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

/** \defgroup sha SHA and B64 helpers
 * ##SHA and B64 helpers
 *
 * These provide SHA-1 and B64 helper apis
 */
///@{
#ifdef LWS_SHA1_USE_OPENSSL_NAME
#define lws_SHA1 SHA1
#else
/**
 * lws_SHA1(): make a SHA-1 digest of a buffer
 *
 * \param d: incoming buffer
 * \param n: length of incoming buffer
 * \param md: buffer for message digest (must be >= 20 bytes)
 *
 * Reduces any size buffer into a 20-byte SHA-1 hash.
 */
LWS_VISIBLE LWS_EXTERN unsigned char *
lws_SHA1(const unsigned char *d, size_t n, unsigned char *md);
#endif
/**
 * lws_b64_encode_string(): encode a string into base 64
 *
 * \param in: incoming buffer
 * \param in_len: length of incoming buffer
 * \param out: result buffer
 * \param out_size: length of result buffer
 *
 * Encodes a string using b64
 */
LWS_VISIBLE LWS_EXTERN int
lws_b64_encode_string(const char *in, int in_len, char *out, int out_size);
/**
 * lws_b64_encode_string_url(): encode a string into base 64
 *
 * \param in: incoming buffer
 * \param in_len: length of incoming buffer
 * \param out: result buffer
 * \param out_size: length of result buffer
 *
 * Encodes a string using b64 with the "URL" variant (+ -> -, and / -> _)
 */
LWS_VISIBLE LWS_EXTERN int
lws_b64_encode_string_url(const char *in, int in_len, char *out, int out_size);
/**
 * lws_b64_decode_string(): decode a string from base 64
 *
 * \param in: incoming buffer
 * \param out: result buffer
 * \param out_size: length of result buffer
 *
 * Decodes a NUL-terminated string using b64
 */
LWS_VISIBLE LWS_EXTERN int
lws_b64_decode_string(const char *in, char *out, int out_size);
/**
 * lws_b64_decode_string_len(): decode a string from base 64
 *
 * \param in: incoming buffer
 * \param in_len: length of incoming buffer
 * \param out: result buffer
 * \param out_size: length of result buffer
 *
 * Decodes a range of chars using b64
 */
LWS_VISIBLE LWS_EXTERN int
lws_b64_decode_string_len(const char *in, int in_len, char *out, int out_size);
///@}

