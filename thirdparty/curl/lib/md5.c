/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 1998 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

#include "curl_setup.h"

#ifndef CURL_DISABLE_CRYPTO_AUTH

#include <curl/curl.h>

#include "curl_md5.h"
#include "curl_hmac.h"
#include "warnless.h"

#ifdef USE_MBEDTLS
#include <mbedtls/version.h>

#if(MBEDTLS_VERSION_NUMBER >= 0x02070000) && \
   (MBEDTLS_VERSION_NUMBER < 0x03000000)
  #define HAS_MBEDTLS_RESULT_CODE_BASED_FUNCTIONS
#endif
#endif /* USE_MBEDTLS */

#if defined(USE_OPENSSL) && !defined(USE_AMISSL)
  #include <openssl/opensslconf.h>
  #if !defined(OPENSSL_NO_MD5) && !defined(OPENSSL_NO_DEPRECATED_3_0)
    #define USE_OPENSSL_MD5
  #endif
#endif

#ifdef USE_WOLFSSL
  #include <wolfssl/options.h>
  #ifndef NO_MD5
    #define USE_WOLFSSL_MD5
  #endif
#endif

#if defined(USE_GNUTLS)

#include <nettle/md5.h>
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

typedef struct md5_ctx MD5_CTX;

static void MD5_Init(MD5_CTX *ctx)
{
  md5_init(ctx);
}

static void MD5_Update(MD5_CTX *ctx,
                       const unsigned char *input,
                       unsigned int inputLen)
{
  md5_update(ctx, inputLen, input);
}

static void MD5_Final(unsigned char *digest, MD5_CTX *ctx)
{
  md5_digest(ctx, 16, digest);
}

#elif defined(USE_OPENSSL_MD5) || defined(USE_WOLFSSL_MD5)

/* When OpenSSL or wolfSSL is available, we use their MD5 functions. */
#include <openssl/md5.h>
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

#elif defined(USE_MBEDTLS)

#include <mbedtls/md5.h>

#include "curl_memory.h"

/* The last #include file should be: */
#include "memdebug.h"

typedef mbedtls_md5_context MD5_CTX;

static void MD5_Init(MD5_CTX *ctx)
{
#if !defined(HAS_MBEDTLS_RESULT_CODE_BASED_FUNCTIONS)
  (void) mbedtls_md5_starts(ctx);
#else
  (void) mbedtls_md5_starts_ret(ctx);
#endif
}

static void MD5_Update(MD5_CTX *ctx,
                       const unsigned char *data,
                       unsigned int length)
{
#if !defined(HAS_MBEDTLS_RESULT_CODE_BASED_FUNCTIONS)
  (void) mbedtls_md5_update(ctx, data, length);
#else
  (void) mbedtls_md5_update_ret(ctx, data, length);
#endif
}

static void MD5_Final(unsigned char *digest, MD5_CTX *ctx)
{
#if !defined(HAS_MBEDTLS_RESULT_CODE_BASED_FUNCTIONS)
  (void) mbedtls_md5_finish(ctx, digest);
#else
  (void) mbedtls_md5_finish_ret(ctx, digest);
#endif
}

#elif (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && \
              (__MAC_OS_X_VERSION_MAX_ALLOWED >= 1040) && \
       defined(__MAC_OS_X_VERSION_MIN_ALLOWED) && \
              (__MAC_OS_X_VERSION_MIN_ALLOWED < 101500)) || \
      (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && \
              (__IPHONE_OS_VERSION_MAX_ALLOWED >= 20000))

/* For Apple operating systems: CommonCrypto has the functions we need.
   These functions are available on Tiger and later, as well as iOS 2.0
   and later. If you're building for an older cat, well, sorry.

   Declaring the functions as static like this seems to be a bit more
   reliable than defining COMMON_DIGEST_FOR_OPENSSL on older cats. */
#  include <CommonCrypto/CommonDigest.h>
#  define MD5_CTX CC_MD5_CTX
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

static void MD5_Init(MD5_CTX *ctx)
{
  CC_MD5_Init(ctx);
}

static void MD5_Update(MD5_CTX *ctx,
                       const unsigned char *input,
                       unsigned int inputLen)
{
  CC_MD5_Update(ctx, input, inputLen);
}

static void MD5_Final(unsigned char *digest, MD5_CTX *ctx)
{
  CC_MD5_Final(digest, ctx);
}

#elif defined(USE_WIN32_CRYPTO)

#include <wincrypt.h>
#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

struct md5_ctx {
  HCRYPTPROV hCryptProv;
  HCRYPTHASH hHash;
};
typedef struct md5_ctx MD5_CTX;

static void MD5_Init(MD5_CTX *ctx)
{
  if(CryptAcquireContext(&ctx->hCryptProv, NULL, NULL, PROV_RSA_FULL,
                         CRYPT_VERIFYCONTEXT | CRYPT_SILENT)) {
    CryptCreateHash(ctx->hCryptProv, CALG_MD5, 0, 0, &ctx->hHash);
  }
}

static void MD5_Update(MD5_CTX *ctx,
                       const unsigned char *input,
                       unsigned int inputLen)
{
  CryptHashData(ctx->hHash, (unsigned char *)input, inputLen, 0);
}

static void MD5_Final(unsigned char *digest, MD5_CTX *ctx)
{
  unsigned long length = 0;
  CryptGetHashParam(ctx->hHash, HP_HASHVAL, NULL, &length, 0);
  if(length == 16)
    CryptGetHashParam(ctx->hHash, HP_HASHVAL, digest, &length, 0);
  if(ctx->hHash)
    CryptDestroyHash(ctx->hHash);
  if(ctx->hCryptProv)
    CryptReleaseContext(ctx->hCryptProv, 0);
}

#else

/* When no other crypto library is available we use this code segment */

/*
 * This is an OpenSSL-compatible implementation of the RSA Data Security, Inc.
 * MD5 Message-Digest Algorithm (RFC 1321).
 *
 * Homepage:
 https://openwall.info/wiki/people/solar/software/public-domain-source-code/md5
 *
 * Author:
 * Alexander Peslyak, better known as Solar Designer <solar at openwall.com>
 *
 * This software was written by Alexander Peslyak in 2001.  No copyright is
 * claimed, and the software is hereby placed in the public domain.
 * In case this attempt to disclaim copyright and place the software in the
 * public domain is deemed null and void, then the software is
 * Copyright (c) 2001 Alexander Peslyak and it is hereby released to the
 * general public under the following terms:
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted.
 *
 * There's ABSOLUTELY NO WARRANTY, express or implied.
 *
 * (This is a heavily cut-down "BSD license".)
 *
 * This differs from Colin Plumb's older public domain implementation in that
 * no exactly 32-bit integer data type is required (any 32-bit or wider
 * unsigned integer data type will do), there's no compile-time endianness
 * configuration, and the function prototypes match OpenSSL's.  No code from
 * Colin Plumb's implementation has been reused; this comment merely compares
 * the properties of the two independent implementations.
 *
 * The primary goals of this implementation are portability and ease of use.
 * It is meant to be fast, but not as fast as possible.  Some known
 * optimizations are not included to reduce source code size and avoid
 * compile-time configuration.
 */

#include <string.h>

/* The last #include files should be: */
#include "curl_memory.h"
#include "memdebug.h"

/* Any 32-bit or wider unsigned integer data type will do */
typedef unsigned int MD5_u32plus;

struct md5_ctx {
  MD5_u32plus lo, hi;
  MD5_u32plus a, b, c, d;
  unsigned char buffer[64];
  MD5_u32plus block[16];
};
typedef struct md5_ctx MD5_CTX;

static void MD5_Init(MD5_CTX *ctx);
static void MD5_Update(MD5_CTX *ctx, const void *data, unsigned long size);
static void MD5_Final(unsigned char *result, MD5_CTX *ctx);

/*
 * The basic MD5 functions.
 *
 * F and G are optimized compared to their RFC 1321 definitions for
 * architectures that lack an AND-NOT instruction, just like in Colin Plumb's
 * implementation.
 */
#define F(x, y, z)                      ((z) ^ ((x) & ((y) ^ (z))))
#define G(x, y, z)                      ((y) ^ ((z) & ((x) ^ (y))))
#define H(x, y, z)                      (((x) ^ (y)) ^ (z))
#define H2(x, y, z)                     ((x) ^ ((y) ^ (z)))
#define I(x, y, z)                      ((y) ^ ((x) | ~(z)))

/*
 * The MD5 transformation for all four rounds.
 */
#define STEP(f, a, b, c, d, x, t, s) \
        (a) += f((b), (c), (d)) + (x) + (t); \
        (a) = (((a) << (s)) | (((a) & 0xffffffff) >> (32 - (s)))); \
        (a) += (b);

/*
 * SET reads 4 input bytes in little-endian byte order and stores them
 * in a properly aligned word in host byte order.
 *
 * The check for little-endian architectures that tolerate unaligned
 * memory accesses is just an optimization.  Nothing will break if it
 * doesn't work.
 */
#if defined(__i386__) || defined(__x86_64__) || defined(__vax__)
#define SET(n) \
        (*(MD5_u32plus *)(void *)&ptr[(n) * 4])
#define GET(n) \
        SET(n)
#else
#define SET(n) \
        (ctx->block[(n)] = \
        (MD5_u32plus)ptr[(n) * 4] | \
        ((MD5_u32plus)ptr[(n) * 4 + 1] << 8) | \
        ((MD5_u32plus)ptr[(n) * 4 + 2] << 16) | \
        ((MD5_u32plus)ptr[(n) * 4 + 3] << 24))
#define GET(n) \
        (ctx->block[(n)])
#endif

/*
 * This processes one or more 64-byte data blocks, but does NOT update
 * the bit counters.  There are no alignment requirements.
 */
static const void *body(MD5_CTX *ctx, const void *data, unsigned long size)
{
  const unsigned char *ptr;
  MD5_u32plus a, b, c, d;

  ptr = (const unsigned char *)data;

  a = ctx->a;
  b = ctx->b;
  c = ctx->c;
  d = ctx->d;

  do {
    MD5_u32plus saved_a, saved_b, saved_c, saved_d;

    saved_a = a;
    saved_b = b;
    saved_c = c;
    saved_d = d;

/* Round 1 */
    STEP(F, a, b, c, d, SET(0), 0xd76aa478, 7)
    STEP(F, d, a, b, c, SET(1), 0xe8c7b756, 12)
    STEP(F, c, d, a, b, SET(2), 0x242070db, 17)
    STEP(F, b, c, d, a, SET(3), 0xc1bdceee, 22)
    STEP(F, a, b, c, d, SET(4), 0xf57c0faf, 7)
    STEP(F, d, a, b, c, SET(5), 0x4787c62a, 12)
    STEP(F, c, d, a, b, SET(6), 0xa8304613, 17)
    STEP(F, b, c, d, a, SET(7), 0xfd469501, 22)
    STEP(F, a, b, c, d, SET(8), 0x698098d8, 7)
    STEP(F, d, a, b, c, SET(9), 0x8b44f7af, 12)
    STEP(F, c, d, a, b, SET(10), 0xffff5bb1, 17)
    STEP(F, b, c, d, a, SET(11), 0x895cd7be, 22)
    STEP(F, a, b, c, d, SET(12), 0x6b901122, 7)
    STEP(F, d, a, b, c, SET(13), 0xfd987193, 12)
    STEP(F, c, d, a, b, SET(14), 0xa679438e, 17)
    STEP(F, b, c, d, a, SET(15), 0x49b40821, 22)

/* Round 2 */
    STEP(G, a, b, c, d, GET(1), 0xf61e2562, 5)
    STEP(G, d, a, b, c, GET(6), 0xc040b340, 9)
    STEP(G, c, d, a, b, GET(11), 0x265e5a51, 14)
    STEP(G, b, c, d, a, GET(0), 0xe9b6c7aa, 20)
    STEP(G, a, b, c, d, GET(5), 0xd62f105d, 5)
    STEP(G, d, a, b, c, GET(10), 0x02441453, 9)
    STEP(G, c, d, a, b, GET(15), 0xd8a1e681, 14)
    STEP(G, b, c, d, a, GET(4), 0xe7d3fbc8, 20)
    STEP(G, a, b, c, d, GET(9), 0x21e1cde6, 5)
    STEP(G, d, a, b, c, GET(14), 0xc33707d6, 9)
    STEP(G, c, d, a, b, GET(3), 0xf4d50d87, 14)
    STEP(G, b, c, d, a, GET(8), 0x455a14ed, 20)
    STEP(G, a, b, c, d, GET(13), 0xa9e3e905, 5)
    STEP(G, d, a, b, c, GET(2), 0xfcefa3f8, 9)
    STEP(G, c, d, a, b, GET(7), 0x676f02d9, 14)
    STEP(G, b, c, d, a, GET(12), 0x8d2a4c8a, 20)

/* Round 3 */
    STEP(H, a, b, c, d, GET(5), 0xfffa3942, 4)
    STEP(H2, d, a, b, c, GET(8), 0x8771f681, 11)
    STEP(H, c, d, a, b, GET(11), 0x6d9d6122, 16)
    STEP(H2, b, c, d, a, GET(14), 0xfde5380c, 23)
    STEP(H, a, b, c, d, GET(1), 0xa4beea44, 4)
    STEP(H2, d, a, b, c, GET(4), 0x4bdecfa9, 11)
    STEP(H, c, d, a, b, GET(7), 0xf6bb4b60, 16)
    STEP(H2, b, c, d, a, GET(10), 0xbebfbc70, 23)
    STEP(H, a, b, c, d, GET(13), 0x289b7ec6, 4)
    STEP(H2, d, a, b, c, GET(0), 0xeaa127fa, 11)
    STEP(H, c, d, a, b, GET(3), 0xd4ef3085, 16)
    STEP(H2, b, c, d, a, GET(6), 0x04881d05, 23)
    STEP(H, a, b, c, d, GET(9), 0xd9d4d039, 4)
    STEP(H2, d, a, b, c, GET(12), 0xe6db99e5, 11)
    STEP(H, c, d, a, b, GET(15), 0x1fa27cf8, 16)
    STEP(H2, b, c, d, a, GET(2), 0xc4ac5665, 23)

/* Round 4 */
    STEP(I, a, b, c, d, GET(0), 0xf4292244, 6)
    STEP(I, d, a, b, c, GET(7), 0x432aff97, 10)
    STEP(I, c, d, a, b, GET(14), 0xab9423a7, 15)
    STEP(I, b, c, d, a, GET(5), 0xfc93a039, 21)
    STEP(I, a, b, c, d, GET(12), 0x655b59c3, 6)
    STEP(I, d, a, b, c, GET(3), 0x8f0ccc92, 10)
    STEP(I, c, d, a, b, GET(10), 0xffeff47d, 15)
    STEP(I, b, c, d, a, GET(1), 0x85845dd1, 21)
    STEP(I, a, b, c, d, GET(8), 0x6fa87e4f, 6)
    STEP(I, d, a, b, c, GET(15), 0xfe2ce6e0, 10)
    STEP(I, c, d, a, b, GET(6), 0xa3014314, 15)
    STEP(I, b, c, d, a, GET(13), 0x4e0811a1, 21)
    STEP(I, a, b, c, d, GET(4), 0xf7537e82, 6)
    STEP(I, d, a, b, c, GET(11), 0xbd3af235, 10)
    STEP(I, c, d, a, b, GET(2), 0x2ad7d2bb, 15)
    STEP(I, b, c, d, a, GET(9), 0xeb86d391, 21)

    a += saved_a;
    b += saved_b;
    c += saved_c;
    d += saved_d;

    ptr += 64;
  } while(size -= 64);

  ctx->a = a;
  ctx->b = b;
  ctx->c = c;
  ctx->d = d;

  return ptr;
}

static void MD5_Init(MD5_CTX *ctx)
{
  ctx->a = 0x67452301;
  ctx->b = 0xefcdab89;
  ctx->c = 0x98badcfe;
  ctx->d = 0x10325476;

  ctx->lo = 0;
  ctx->hi = 0;
}

static void MD5_Update(MD5_CTX *ctx, const void *data, unsigned long size)
{
  MD5_u32plus saved_lo;
  unsigned long used;

  saved_lo = ctx->lo;
  ctx->lo = (saved_lo + size) & 0x1fffffff;
  if(ctx->lo < saved_lo)
    ctx->hi++;
  ctx->hi += (MD5_u32plus)size >> 29;

  used = saved_lo & 0x3f;

  if(used) {
    unsigned long available = 64 - used;

    if(size < available) {
      memcpy(&ctx->buffer[used], data, size);
      return;
    }

    memcpy(&ctx->buffer[used], data, available);
    data = (const unsigned char *)data + available;
    size -= available;
    body(ctx, ctx->buffer, 64);
  }

  if(size >= 64) {
    data = body(ctx, data, size & ~(unsigned long)0x3f);
    size &= 0x3f;
  }

  memcpy(ctx->buffer, data, size);
}

static void MD5_Final(unsigned char *result, MD5_CTX *ctx)
{
  unsigned long used, available;

  used = ctx->lo & 0x3f;

  ctx->buffer[used++] = 0x80;

  available = 64 - used;

  if(available < 8) {
    memset(&ctx->buffer[used], 0, available);
    body(ctx, ctx->buffer, 64);
    used = 0;
    available = 64;
  }

  memset(&ctx->buffer[used], 0, available - 8);

  ctx->lo <<= 3;
  ctx->buffer[56] = curlx_ultouc((ctx->lo)&0xff);
  ctx->buffer[57] = curlx_ultouc((ctx->lo >> 8)&0xff);
  ctx->buffer[58] = curlx_ultouc((ctx->lo >> 16)&0xff);
  ctx->buffer[59] = curlx_ultouc(ctx->lo >> 24);
  ctx->buffer[60] = curlx_ultouc((ctx->hi)&0xff);
  ctx->buffer[61] = curlx_ultouc((ctx->hi >> 8)&0xff);
  ctx->buffer[62] = curlx_ultouc((ctx->hi >> 16)&0xff);
  ctx->buffer[63] = curlx_ultouc(ctx->hi >> 24);

  body(ctx, ctx->buffer, 64);

  result[0] = curlx_ultouc((ctx->a)&0xff);
  result[1] = curlx_ultouc((ctx->a >> 8)&0xff);
  result[2] = curlx_ultouc((ctx->a >> 16)&0xff);
  result[3] = curlx_ultouc(ctx->a >> 24);
  result[4] = curlx_ultouc((ctx->b)&0xff);
  result[5] = curlx_ultouc((ctx->b >> 8)&0xff);
  result[6] = curlx_ultouc((ctx->b >> 16)&0xff);
  result[7] = curlx_ultouc(ctx->b >> 24);
  result[8] = curlx_ultouc((ctx->c)&0xff);
  result[9] = curlx_ultouc((ctx->c >> 8)&0xff);
  result[10] = curlx_ultouc((ctx->c >> 16)&0xff);
  result[11] = curlx_ultouc(ctx->c >> 24);
  result[12] = curlx_ultouc((ctx->d)&0xff);
  result[13] = curlx_ultouc((ctx->d >> 8)&0xff);
  result[14] = curlx_ultouc((ctx->d >> 16)&0xff);
  result[15] = curlx_ultouc(ctx->d >> 24);

  memset(ctx, 0, sizeof(*ctx));
}

#endif /* CRYPTO LIBS */

const struct HMAC_params Curl_HMAC_MD5[] = {
  {
    /* Hash initialization function. */
    CURLX_FUNCTION_CAST(HMAC_hinit_func, MD5_Init),
    /* Hash update function. */
    CURLX_FUNCTION_CAST(HMAC_hupdate_func, MD5_Update),
    /* Hash computation end function. */
    CURLX_FUNCTION_CAST(HMAC_hfinal_func, MD5_Final),
    /* Size of hash context structure. */
    sizeof(MD5_CTX),
    /* Maximum key length. */
    64,
    /* Result size. */
    16
  }
};

const struct MD5_params Curl_DIGEST_MD5[] = {
  {
    /* Digest initialization function */
    CURLX_FUNCTION_CAST(Curl_MD5_init_func, MD5_Init),
    /* Digest update function */
    CURLX_FUNCTION_CAST(Curl_MD5_update_func, MD5_Update),
    /* Digest computation end function */
    CURLX_FUNCTION_CAST(Curl_MD5_final_func, MD5_Final),
    /* Size of digest context struct */
    sizeof(MD5_CTX),
    /* Result size */
    16
  }
};

/*
 * @unittest: 1601
 */
void Curl_md5it(unsigned char *outbuffer, const unsigned char *input,
                const size_t len)
{
  MD5_CTX ctx;

  MD5_Init(&ctx);
  MD5_Update(&ctx, input, curlx_uztoui(len));
  MD5_Final(outbuffer, &ctx);
}

struct MD5_context *Curl_MD5_init(const struct MD5_params *md5params)
{
  struct MD5_context *ctxt;

  /* Create MD5 context */
  ctxt = malloc(sizeof(*ctxt));

  if(!ctxt)
    return ctxt;

  ctxt->md5_hashctx = malloc(md5params->md5_ctxtsize);

  if(!ctxt->md5_hashctx) {
    free(ctxt);
    return NULL;
  }

  ctxt->md5_hash = md5params;

  (*md5params->md5_init_func)(ctxt->md5_hashctx);

  return ctxt;
}

CURLcode Curl_MD5_update(struct MD5_context *context,
                         const unsigned char *data,
                         unsigned int len)
{
  (*context->md5_hash->md5_update_func)(context->md5_hashctx, data, len);

  return CURLE_OK;
}

CURLcode Curl_MD5_final(struct MD5_context *context, unsigned char *result)
{
  (*context->md5_hash->md5_final_func)(result, context->md5_hashctx);

  free(context->md5_hashctx);
  free(context);

  return CURLE_OK;
}

#endif /* CURL_DISABLE_CRYPTO_AUTH */
