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

#if defined(USE_CURL_NTLM_CORE)

/*
 * NTLM details:
 *
 * https://davenport.sourceforge.io/ntlm.html
 * https://www.innovation.ch/java/ntlm.html
 */

/* Please keep the SSL backend-specific #if branches in this order:

   1. USE_OPENSSL
   2. USE_GNUTLS
   3. USE_NSS
   4. USE_MBEDTLS
   5. USE_SECTRANSP
   6. USE_OS400CRYPTO
   7. USE_WIN32_CRYPTO

   This ensures that:
   - the same SSL branch gets activated throughout this source
     file even if multiple backends are enabled at the same time.
   - OpenSSL and NSS have higher priority than Windows Crypt, due
     to issues with the latter supporting NTLM2Session responses
     in NTLM type-3 messages.
 */

#if defined(USE_OPENSSL)
  #include <openssl/opensslconf.h>
  #if !defined(OPENSSL_NO_DES) && !defined(OPENSSL_NO_DEPRECATED_3_0)
    #define USE_OPENSSL_DES
  #endif
#endif

#if defined(USE_OPENSSL_DES) || defined(USE_WOLFSSL)

#ifdef USE_WOLFSSL
#include <wolfssl/options.h>
#endif

#  include <openssl/des.h>
#  include <openssl/md5.h>
#  include <openssl/ssl.h>
#  include <openssl/rand.h>
#  if (defined(OPENSSL_VERSION_NUMBER) && \
       (OPENSSL_VERSION_NUMBER < 0x00907001L)) && !defined(USE_WOLFSSL)
#    define DES_key_schedule des_key_schedule
#    define DES_cblock des_cblock
#    define DES_set_odd_parity des_set_odd_parity
#    define DES_set_key des_set_key
#    define DES_ecb_encrypt des_ecb_encrypt
#    define DESKEY(x) x
#    define DESKEYARG(x) x
#  else
#    define DESKEYARG(x) *x
#    define DESKEY(x) &x
#  endif

#elif defined(USE_GNUTLS)

#  include <nettle/des.h>

#elif defined(USE_NSS)

#  include <nss.h>
#  include <pk11pub.h>
#  include <hasht.h>

#elif defined(USE_MBEDTLS)

#  include <mbedtls/des.h>

#elif defined(USE_SECTRANSP)

#  include <CommonCrypto/CommonCryptor.h>
#  include <CommonCrypto/CommonDigest.h>

#elif defined(USE_OS400CRYPTO)
#  include "cipher.mih"  /* mih/cipher */
#elif defined(USE_WIN32_CRYPTO)
#  include <wincrypt.h>
#else
#  error "Can't compile NTLM support without a crypto library with DES."
#endif

#include "urldata.h"
#include "non-ascii.h"
#include "strcase.h"
#include "curl_ntlm_core.h"
#include "curl_md5.h"
#include "curl_hmac.h"
#include "warnless.h"
#include "curl_endian.h"
#include "curl_des.h"
#include "curl_md4.h"
/* The last 3 #include files should be in this order */
#include "curl_printf.h"
#include "curl_memory.h"
#include "memdebug.h"

#define NTLMv2_BLOB_SIGNATURE "\x01\x01\x00\x00"
#define NTLMv2_BLOB_LEN       (44 -16 + ntlm->target_info_len + 4)

/*
* Turns a 56-bit key into being 64-bit wide.
*/
static void extend_key_56_to_64(const unsigned char *key_56, char *key)
{
  key[0] = key_56[0];
  key[1] = (unsigned char)(((key_56[0] << 7) & 0xFF) | (key_56[1] >> 1));
  key[2] = (unsigned char)(((key_56[1] << 6) & 0xFF) | (key_56[2] >> 2));
  key[3] = (unsigned char)(((key_56[2] << 5) & 0xFF) | (key_56[3] >> 3));
  key[4] = (unsigned char)(((key_56[3] << 4) & 0xFF) | (key_56[4] >> 4));
  key[5] = (unsigned char)(((key_56[4] << 3) & 0xFF) | (key_56[5] >> 5));
  key[6] = (unsigned char)(((key_56[5] << 2) & 0xFF) | (key_56[6] >> 6));
  key[7] = (unsigned char) ((key_56[6] << 1) & 0xFF);
}

#if defined(USE_OPENSSL_DES) || defined(USE_WOLFSSL)
/*
 * Turns a 56 bit key into the 64 bit, odd parity key and sets the key.  The
 * key schedule ks is also set.
 */
static void setup_des_key(const unsigned char *key_56,
                          DES_key_schedule DESKEYARG(ks))
{
  DES_cblock key;

  /* Expand the 56-bit key to 64-bits */
  extend_key_56_to_64(key_56, (char *) &key);

  /* Set the key parity to odd */
  DES_set_odd_parity(&key);

  /* Set the key */
  DES_set_key_unchecked(&key, ks);
}

#elif defined(USE_GNUTLS)

static void setup_des_key(const unsigned char *key_56,
                          struct des_ctx *des)
{
  char key[8];

  /* Expand the 56-bit key to 64-bits */
  extend_key_56_to_64(key_56, key);

  /* Set the key parity to odd */
  Curl_des_set_odd_parity((unsigned char *) key, sizeof(key));

  /* Set the key */
  des_set_key(des, (const uint8_t *) key);
}

#elif defined(USE_NSS)

/*
 * Expands a 56 bit key KEY_56 to 64 bit and encrypts 64 bit of data, using
 * the expanded key.  The caller is responsible for giving 64 bit of valid
 * data is IN and (at least) 64 bit large buffer as OUT.
 */
static bool encrypt_des(const unsigned char *in, unsigned char *out,
                        const unsigned char *key_56)
{
  const CK_MECHANISM_TYPE mech = CKM_DES_ECB; /* DES cipher in ECB mode */
  char key[8];                                /* expanded 64 bit key */
  SECItem key_item;
  PK11SymKey *symkey = NULL;
  SECItem *param = NULL;
  PK11Context *ctx = NULL;
  int out_len;                                /* not used, required by NSS */
  bool rv = FALSE;

  /* use internal slot for DES encryption (requires NSS to be initialized) */
  PK11SlotInfo *slot = PK11_GetInternalKeySlot();
  if(!slot)
    return FALSE;

  /* Expand the 56-bit key to 64-bits */
  extend_key_56_to_64(key_56, key);

  /* Set the key parity to odd */
  Curl_des_set_odd_parity((unsigned char *) key, sizeof(key));

  /* Import the key */
  key_item.data = (unsigned char *)key;
  key_item.len = sizeof(key);
  symkey = PK11_ImportSymKey(slot, mech, PK11_OriginUnwrap, CKA_ENCRYPT,
                             &key_item, NULL);
  if(!symkey)
    goto fail;

  /* Create the DES encryption context */
  param = PK11_ParamFromIV(mech, /* no IV in ECB mode */ NULL);
  if(!param)
    goto fail;
  ctx = PK11_CreateContextBySymKey(mech, CKA_ENCRYPT, symkey, param);
  if(!ctx)
    goto fail;

  /* Perform the encryption */
  if(SECSuccess == PK11_CipherOp(ctx, out, &out_len, /* outbuflen */ 8,
                                 (unsigned char *)in, /* inbuflen */ 8)
      && SECSuccess == PK11_Finalize(ctx))
    rv = /* all OK */ TRUE;

fail:
  /* cleanup */
  if(ctx)
    PK11_DestroyContext(ctx, PR_TRUE);
  if(symkey)
    PK11_FreeSymKey(symkey);
  if(param)
    SECITEM_FreeItem(param, PR_TRUE);
  PK11_FreeSlot(slot);
  return rv;
}

#elif defined(USE_MBEDTLS)

static bool encrypt_des(const unsigned char *in, unsigned char *out,
                        const unsigned char *key_56)
{
  mbedtls_des_context ctx;
  char key[8];

  /* Expand the 56-bit key to 64-bits */
  extend_key_56_to_64(key_56, key);

  /* Set the key parity to odd */
  mbedtls_des_key_set_parity((unsigned char *) key);

  /* Perform the encryption */
  mbedtls_des_init(&ctx);
  mbedtls_des_setkey_enc(&ctx, (unsigned char *) key);
  return mbedtls_des_crypt_ecb(&ctx, in, out) == 0;
}

#elif defined(USE_SECTRANSP)

static bool encrypt_des(const unsigned char *in, unsigned char *out,
                        const unsigned char *key_56)
{
  char key[8];
  size_t out_len;
  CCCryptorStatus err;

  /* Expand the 56-bit key to 64-bits */
  extend_key_56_to_64(key_56, key);

  /* Set the key parity to odd */
  Curl_des_set_odd_parity((unsigned char *) key, sizeof(key));

  /* Perform the encryption */
  err = CCCrypt(kCCEncrypt, kCCAlgorithmDES, kCCOptionECBMode, key,
                kCCKeySizeDES, NULL, in, 8 /* inbuflen */, out,
                8 /* outbuflen */, &out_len);

  return err == kCCSuccess;
}

#elif defined(USE_OS400CRYPTO)

static bool encrypt_des(const unsigned char *in, unsigned char *out,
                        const unsigned char *key_56)
{
  char key[8];
  _CIPHER_Control_T ctl;

  /* Setup the cipher control structure */
  ctl.Func_ID = ENCRYPT_ONLY;
  ctl.Data_Len = sizeof(key);

  /* Expand the 56-bit key to 64-bits */
  extend_key_56_to_64(key_56, ctl.Crypto_Key);

  /* Set the key parity to odd */
  Curl_des_set_odd_parity((unsigned char *) ctl.Crypto_Key, ctl.Data_Len);

  /* Perform the encryption */
  _CIPHER((_SPCPTR *) &out, &ctl, (_SPCPTR *) &in);

  return TRUE;
}

#elif defined(USE_WIN32_CRYPTO)

static bool encrypt_des(const unsigned char *in, unsigned char *out,
                        const unsigned char *key_56)
{
  HCRYPTPROV hprov;
  HCRYPTKEY hkey;
  struct {
    BLOBHEADER hdr;
    unsigned int len;
    char key[8];
  } blob;
  DWORD len = 8;

  /* Acquire the crypto provider */
  if(!CryptAcquireContext(&hprov, NULL, NULL, PROV_RSA_FULL,
                          CRYPT_VERIFYCONTEXT | CRYPT_SILENT))
    return FALSE;

  /* Setup the key blob structure */
  memset(&blob, 0, sizeof(blob));
  blob.hdr.bType = PLAINTEXTKEYBLOB;
  blob.hdr.bVersion = 2;
  blob.hdr.aiKeyAlg = CALG_DES;
  blob.len = sizeof(blob.key);

  /* Expand the 56-bit key to 64-bits */
  extend_key_56_to_64(key_56, blob.key);

  /* Set the key parity to odd */
  Curl_des_set_odd_parity((unsigned char *) blob.key, sizeof(blob.key));

  /* Import the key */
  if(!CryptImportKey(hprov, (BYTE *) &blob, sizeof(blob), 0, 0, &hkey)) {
    CryptReleaseContext(hprov, 0);

    return FALSE;
  }

  memcpy(out, in, 8);

  /* Perform the encryption */
  CryptEncrypt(hkey, 0, FALSE, 0, out, &len, len);

  CryptDestroyKey(hkey);
  CryptReleaseContext(hprov, 0);

  return TRUE;
}

#endif /* defined(USE_WIN32_CRYPTO) */

 /*
  * takes a 21 byte array and treats it as 3 56-bit DES keys. The
  * 8 byte plaintext is encrypted with each key and the resulting 24
  * bytes are stored in the results array.
  */
void Curl_ntlm_core_lm_resp(const unsigned char *keys,
                            const unsigned char *plaintext,
                            unsigned char *results)
{
#if defined(USE_OPENSSL_DES) || defined(USE_WOLFSSL)
  DES_key_schedule ks;

  setup_des_key(keys, DESKEY(ks));
  DES_ecb_encrypt((DES_cblock*) plaintext, (DES_cblock*) results,
                  DESKEY(ks), DES_ENCRYPT);

  setup_des_key(keys + 7, DESKEY(ks));
  DES_ecb_encrypt((DES_cblock*) plaintext, (DES_cblock*) (results + 8),
                  DESKEY(ks), DES_ENCRYPT);

  setup_des_key(keys + 14, DESKEY(ks));
  DES_ecb_encrypt((DES_cblock*) plaintext, (DES_cblock*) (results + 16),
                  DESKEY(ks), DES_ENCRYPT);
#elif defined(USE_GNUTLS)
  struct des_ctx des;
  setup_des_key(keys, &des);
  des_encrypt(&des, 8, results, plaintext);
  setup_des_key(keys + 7, &des);
  des_encrypt(&des, 8, results + 8, plaintext);
  setup_des_key(keys + 14, &des);
  des_encrypt(&des, 8, results + 16, plaintext);
#elif defined(USE_NSS) || defined(USE_MBEDTLS) || defined(USE_SECTRANSP) \
  || defined(USE_OS400CRYPTO) || defined(USE_WIN32_CRYPTO)
  encrypt_des(plaintext, results, keys);
  encrypt_des(plaintext, results + 8, keys + 7);
  encrypt_des(plaintext, results + 16, keys + 14);
#endif
}

/*
 * Set up lanmanager hashed password
 */
CURLcode Curl_ntlm_core_mk_lm_hash(struct Curl_easy *data,
                                   const char *password,
                                   unsigned char *lmbuffer /* 21 bytes */)
{
  CURLcode result;
  unsigned char pw[14];
  static const unsigned char magic[] = {
    0x4B, 0x47, 0x53, 0x21, 0x40, 0x23, 0x24, 0x25 /* i.e. KGS!@#$% */
  };
  size_t len = CURLMIN(strlen(password), 14);

  Curl_strntoupper((char *)pw, password, len);
  memset(&pw[len], 0, 14 - len);

  /*
   * The LanManager hashed password needs to be created using the
   * password in the network encoding not the host encoding.
   */
  result = Curl_convert_to_network(data, (char *)pw, 14);
  if(result)
    return result;

  {
    /* Create LanManager hashed password. */

#if defined(USE_OPENSSL_DES) || defined(USE_WOLFSSL)
    DES_key_schedule ks;

    setup_des_key(pw, DESKEY(ks));
    DES_ecb_encrypt((DES_cblock *)magic, (DES_cblock *)lmbuffer,
                    DESKEY(ks), DES_ENCRYPT);

    setup_des_key(pw + 7, DESKEY(ks));
    DES_ecb_encrypt((DES_cblock *)magic, (DES_cblock *)(lmbuffer + 8),
                    DESKEY(ks), DES_ENCRYPT);
#elif defined(USE_GNUTLS)
    struct des_ctx des;
    setup_des_key(pw, &des);
    des_encrypt(&des, 8, lmbuffer, magic);
    setup_des_key(pw + 7, &des);
    des_encrypt(&des, 8, lmbuffer + 8, magic);
#elif defined(USE_NSS) || defined(USE_MBEDTLS) || defined(USE_SECTRANSP) \
  || defined(USE_OS400CRYPTO) || defined(USE_WIN32_CRYPTO)
    encrypt_des(magic, lmbuffer, pw);
    encrypt_des(magic, lmbuffer + 8, pw + 7);
#endif

    memset(lmbuffer + 16, 0, 21 - 16);
  }

  return CURLE_OK;
}

#ifdef USE_NTRESPONSES
static void ascii_to_unicode_le(unsigned char *dest, const char *src,
                                size_t srclen)
{
  size_t i;
  for(i = 0; i < srclen; i++) {
    dest[2 * i] = (unsigned char)src[i];
    dest[2 * i + 1] = '\0';
  }
}

#if defined(USE_NTLM_V2) && !defined(USE_WINDOWS_SSPI)

static void ascii_uppercase_to_unicode_le(unsigned char *dest,
                                          const char *src, size_t srclen)
{
  size_t i;
  for(i = 0; i < srclen; i++) {
    dest[2 * i] = (unsigned char)(Curl_raw_toupper(src[i]));
    dest[2 * i + 1] = '\0';
  }
}

#endif /* USE_NTLM_V2 && !USE_WINDOWS_SSPI */

/*
 * Set up nt hashed passwords
 * @unittest: 1600
 */
CURLcode Curl_ntlm_core_mk_nt_hash(struct Curl_easy *data,
                                   const char *password,
                                   unsigned char *ntbuffer /* 21 bytes */)
{
  size_t len = strlen(password);
  unsigned char *pw;
  CURLcode result;
  if(len > SIZE_T_MAX/2) /* avoid integer overflow */
    return CURLE_OUT_OF_MEMORY;
  pw = len ? malloc(len * 2) : (unsigned char *)strdup("");
  if(!pw)
    return CURLE_OUT_OF_MEMORY;

  ascii_to_unicode_le(pw, password, len);

  /*
   * The NT hashed password needs to be created using the password in the
   * network encoding not the host encoding.
   */
  result = Curl_convert_to_network(data, (char *)pw, len * 2);
  if(!result) {
    /* Create NT hashed password. */
    Curl_md4it(ntbuffer, pw, 2 * len);
    memset(ntbuffer + 16, 0, 21 - 16);
  }
  free(pw);

  return result;
}

#if defined(USE_NTLM_V2) && !defined(USE_WINDOWS_SSPI)

/* Timestamp in tenths of a microsecond since January 1, 1601 00:00:00 UTC. */
struct ms_filetime {
  unsigned int dwLowDateTime;
  unsigned int dwHighDateTime;
};

/* Convert a time_t to an MS FILETIME (MS-DTYP section 2.3.3). */
static void time2filetime(struct ms_filetime *ft, time_t t)
{
#if SIZEOF_TIME_T > 4
  t = (t + CURL_OFF_T_C(11644473600)) * 10000000;
  ft->dwLowDateTime = (unsigned int) (t & 0xFFFFFFFF);
  ft->dwHighDateTime = (unsigned int) (t >> 32);
#else
  unsigned int r, s;
  unsigned int i;

  ft->dwLowDateTime = t & 0xFFFFFFFF;
  ft->dwHighDateTime = 0;

# ifndef HAVE_TIME_T_UNSIGNED
  /* Extend sign if needed. */
  if(ft->dwLowDateTime & 0x80000000)
    ft->dwHighDateTime = ~0;
# endif

  /* Bias seconds to Jan 1, 1601.
     134774 days = 11644473600 seconds = 0x2B6109100 */
  r = ft->dwLowDateTime;
  ft->dwLowDateTime = (ft->dwLowDateTime + 0xB6109100U) & 0xFFFFFFFF;
  ft->dwHighDateTime += ft->dwLowDateTime < r? 0x03: 0x02;

  /* Convert to tenths of microseconds. */
  ft->dwHighDateTime *= 10000000;
  i = 32;
  do {
    i -= 8;
    s = ((ft->dwLowDateTime >> i) & 0xFF) * (10000000 - 1);
    r = (s << i) & 0xFFFFFFFF;
    s >>= 1;   /* Split shift to avoid width overflow. */
    s >>= 31 - i;
    ft->dwLowDateTime = (ft->dwLowDateTime + r) & 0xFFFFFFFF;
    if(ft->dwLowDateTime < r)
      s++;
    ft->dwHighDateTime += s;
  } while(i);
  ft->dwHighDateTime &= 0xFFFFFFFF;
#endif
}

/* This creates the NTLMv2 hash by using NTLM hash as the key and Unicode
 * (uppercase UserName + Domain) as the data
 */
CURLcode Curl_ntlm_core_mk_ntlmv2_hash(const char *user, size_t userlen,
                                       const char *domain, size_t domlen,
                                       unsigned char *ntlmhash,
                                       unsigned char *ntlmv2hash)
{
  /* Unicode representation */
  size_t identity_len;
  unsigned char *identity;
  CURLcode result = CURLE_OK;

  if((userlen > CURL_MAX_INPUT_LENGTH) || (domlen > CURL_MAX_INPUT_LENGTH))
    return CURLE_OUT_OF_MEMORY;

  identity_len = (userlen + domlen) * 2;
  identity = malloc(identity_len + 1);

  if(!identity)
    return CURLE_OUT_OF_MEMORY;

  ascii_uppercase_to_unicode_le(identity, user, userlen);
  ascii_to_unicode_le(identity + (userlen << 1), domain, domlen);

  result = Curl_hmacit(Curl_HMAC_MD5, ntlmhash, 16, identity, identity_len,
                       ntlmv2hash);
  free(identity);

  return result;
}

/*
 * Curl_ntlm_core_mk_ntlmv2_resp()
 *
 * This creates the NTLMv2 response as set in the ntlm type-3 message.
 *
 * Parameters:
 *
 * ntlmv2hash       [in] - The ntlmv2 hash (16 bytes)
 * challenge_client [in] - The client nonce (8 bytes)
 * ntlm             [in] - The ntlm data struct being used to read TargetInfo
                           and Server challenge received in the type-2 message
 * ntresp          [out] - The address where a pointer to newly allocated
 *                         memory holding the NTLMv2 response.
 * ntresp_len      [out] - The length of the output message.
 *
 * Returns CURLE_OK on success.
 */
CURLcode Curl_ntlm_core_mk_ntlmv2_resp(unsigned char *ntlmv2hash,
                                       unsigned char *challenge_client,
                                       struct ntlmdata *ntlm,
                                       unsigned char **ntresp,
                                       unsigned int *ntresp_len)
{
/* NTLMv2 response structure :
------------------------------------------------------------------------------
0     HMAC MD5         16 bytes
------BLOB--------------------------------------------------------------------
16    Signature        0x01010000
20    Reserved         long (0x00000000)
24    Timestamp        LE, 64-bit signed value representing the number of
                       tenths of a microsecond since January 1, 1601.
32    Client Nonce     8 bytes
40    Unknown          4 bytes
44    Target Info      N bytes (from the type-2 message)
44+N  Unknown          4 bytes
------------------------------------------------------------------------------
*/

  unsigned int len = 0;
  unsigned char *ptr = NULL;
  unsigned char hmac_output[HMAC_MD5_LENGTH];
  struct ms_filetime tw;

  CURLcode result = CURLE_OK;

  /* Calculate the timestamp */
#ifdef DEBUGBUILD
  char *force_timestamp = getenv("CURL_FORCETIME");
  if(force_timestamp)
    time2filetime(&tw, (time_t) 0);
  else
#endif
    time2filetime(&tw, time(NULL));

  /* Calculate the response len */
  len = HMAC_MD5_LENGTH + NTLMv2_BLOB_LEN;

  /* Allocate the response */
  ptr = calloc(1, len);
  if(!ptr)
    return CURLE_OUT_OF_MEMORY;

  /* Create the BLOB structure */
  msnprintf((char *)ptr + HMAC_MD5_LENGTH, NTLMv2_BLOB_LEN,
            "%c%c%c%c"           /* NTLMv2_BLOB_SIGNATURE */
            "%c%c%c%c"           /* Reserved = 0 */
            "%c%c%c%c%c%c%c%c",  /* Timestamp */
            NTLMv2_BLOB_SIGNATURE[0], NTLMv2_BLOB_SIGNATURE[1],
            NTLMv2_BLOB_SIGNATURE[2], NTLMv2_BLOB_SIGNATURE[3],
            0, 0, 0, 0,
            LONGQUARTET(tw.dwLowDateTime), LONGQUARTET(tw.dwHighDateTime));

  memcpy(ptr + 32, challenge_client, 8);
  memcpy(ptr + 44, ntlm->target_info, ntlm->target_info_len);

  /* Concatenate the Type 2 challenge with the BLOB and do HMAC MD5 */
  memcpy(ptr + 8, &ntlm->nonce[0], 8);
  result = Curl_hmacit(Curl_HMAC_MD5, ntlmv2hash, HMAC_MD5_LENGTH, ptr + 8,
                    NTLMv2_BLOB_LEN + 8, hmac_output);
  if(result) {
    free(ptr);
    return result;
  }

  /* Concatenate the HMAC MD5 output  with the BLOB */
  memcpy(ptr, hmac_output, HMAC_MD5_LENGTH);

  /* Return the response */
  *ntresp = ptr;
  *ntresp_len = len;

  return result;
}

/*
 * Curl_ntlm_core_mk_lmv2_resp()
 *
 * This creates the LMv2 response as used in the ntlm type-3 message.
 *
 * Parameters:
 *
 * ntlmv2hash        [in] - The ntlmv2 hash (16 bytes)
 * challenge_client  [in] - The client nonce (8 bytes)
 * challenge_client  [in] - The server challenge (8 bytes)
 * lmresp           [out] - The LMv2 response (24 bytes)
 *
 * Returns CURLE_OK on success.
 */
CURLcode  Curl_ntlm_core_mk_lmv2_resp(unsigned char *ntlmv2hash,
                                      unsigned char *challenge_client,
                                      unsigned char *challenge_server,
                                      unsigned char *lmresp)
{
  unsigned char data[16];
  unsigned char hmac_output[16];
  CURLcode result = CURLE_OK;

  memcpy(&data[0], challenge_server, 8);
  memcpy(&data[8], challenge_client, 8);

  result = Curl_hmacit(Curl_HMAC_MD5, ntlmv2hash, 16, &data[0], 16,
                       hmac_output);
  if(result)
    return result;

  /* Concatenate the HMAC MD5 output  with the client nonce */
  memcpy(lmresp, hmac_output, 16);
  memcpy(lmresp + 16, challenge_client, 8);

  return result;
}

#endif /* USE_NTLM_V2 && !USE_WINDOWS_SSPI */

#endif /* USE_NTRESPONSES */

#endif /* USE_CURL_NTLM_CORE */
