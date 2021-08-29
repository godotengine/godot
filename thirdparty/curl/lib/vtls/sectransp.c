/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2012 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 * Copyright (C) 2012 - 2017, Nick Zitzmann, <nickzman@gmail.com>.
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

/*
 * Source file for all iOS and macOS SecureTransport-specific code for the
 * TLS/SSL layer. No code but vtls.c should ever call or use these functions.
 */

#include "curl_setup.h"

#include "urldata.h" /* for the Curl_easy definition */
#include "curl_base64.h"
#include "strtok.h"
#include "multiif.h"
#include "strcase.h"
#include "x509asn1.h"
#include "strerror.h"

#ifdef USE_SECTRANSP

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-pointer-compare"
#endif /* __clang__ */

#include <limits.h>

#include <Security/Security.h>
/* For some reason, when building for iOS, the omnibus header above does
 * not include SecureTransport.h as of iOS SDK 5.1. */
#include <Security/SecureTransport.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CommonCrypto/CommonDigest.h>

/* The Security framework has changed greatly between iOS and different macOS
   versions, and we will try to support as many of them as we can (back to
   Leopard and iOS 5) by using macros and weak-linking.

   In general, you want to build this using the most recent OS SDK, since some
   features require curl to be built against the latest SDK. TLS 1.1 and 1.2
   support, for instance, require the macOS 10.8 SDK or later. TLS 1.3
   requires the macOS 10.13 or iOS 11 SDK or later. */
#if (TARGET_OS_MAC && !(TARGET_OS_EMBEDDED || TARGET_OS_IPHONE))

#if MAC_OS_X_VERSION_MAX_ALLOWED < 1050
#error "The Secure Transport back-end requires Leopard or later."
#endif /* MAC_OS_X_VERSION_MAX_ALLOWED < 1050 */

#define CURL_BUILD_IOS 0
#define CURL_BUILD_IOS_7 0
#define CURL_BUILD_IOS_9 0
#define CURL_BUILD_IOS_11 0
#define CURL_BUILD_IOS_13 0
#define CURL_BUILD_MAC 1
/* This is the maximum API level we are allowed to use when building: */
#define CURL_BUILD_MAC_10_5 MAC_OS_X_VERSION_MAX_ALLOWED >= 1050
#define CURL_BUILD_MAC_10_6 MAC_OS_X_VERSION_MAX_ALLOWED >= 1060
#define CURL_BUILD_MAC_10_7 MAC_OS_X_VERSION_MAX_ALLOWED >= 1070
#define CURL_BUILD_MAC_10_8 MAC_OS_X_VERSION_MAX_ALLOWED >= 1080
#define CURL_BUILD_MAC_10_9 MAC_OS_X_VERSION_MAX_ALLOWED >= 1090
#define CURL_BUILD_MAC_10_11 MAC_OS_X_VERSION_MAX_ALLOWED >= 101100
#define CURL_BUILD_MAC_10_13 MAC_OS_X_VERSION_MAX_ALLOWED >= 101300
#define CURL_BUILD_MAC_10_15 MAC_OS_X_VERSION_MAX_ALLOWED >= 101500
/* These macros mean "the following code is present to allow runtime backward
   compatibility with at least this cat or earlier":
   (You set this at build-time using the compiler command line option
   "-mmacosx-version-min.") */
#define CURL_SUPPORT_MAC_10_5 MAC_OS_X_VERSION_MIN_REQUIRED <= 1050
#define CURL_SUPPORT_MAC_10_6 MAC_OS_X_VERSION_MIN_REQUIRED <= 1060
#define CURL_SUPPORT_MAC_10_7 MAC_OS_X_VERSION_MIN_REQUIRED <= 1070
#define CURL_SUPPORT_MAC_10_8 MAC_OS_X_VERSION_MIN_REQUIRED <= 1080
#define CURL_SUPPORT_MAC_10_9 MAC_OS_X_VERSION_MIN_REQUIRED <= 1090

#elif TARGET_OS_EMBEDDED || TARGET_OS_IPHONE
#define CURL_BUILD_IOS 1
#define CURL_BUILD_IOS_7 __IPHONE_OS_VERSION_MAX_ALLOWED >= 70000
#define CURL_BUILD_IOS_9 __IPHONE_OS_VERSION_MAX_ALLOWED >= 90000
#define CURL_BUILD_IOS_11 __IPHONE_OS_VERSION_MAX_ALLOWED >= 110000
#define CURL_BUILD_IOS_13 __IPHONE_OS_VERSION_MAX_ALLOWED >= 130000
#define CURL_BUILD_MAC 0
#define CURL_BUILD_MAC_10_5 0
#define CURL_BUILD_MAC_10_6 0
#define CURL_BUILD_MAC_10_7 0
#define CURL_BUILD_MAC_10_8 0
#define CURL_BUILD_MAC_10_9 0
#define CURL_BUILD_MAC_10_11 0
#define CURL_BUILD_MAC_10_13 0
#define CURL_BUILD_MAC_10_15 0
#define CURL_SUPPORT_MAC_10_5 0
#define CURL_SUPPORT_MAC_10_6 0
#define CURL_SUPPORT_MAC_10_7 0
#define CURL_SUPPORT_MAC_10_8 0
#define CURL_SUPPORT_MAC_10_9 0

#else
#error "The Secure Transport back-end requires iOS or macOS."
#endif /* (TARGET_OS_MAC && !(TARGET_OS_EMBEDDED || TARGET_OS_IPHONE)) */

#if CURL_BUILD_MAC
#include <sys/sysctl.h>
#endif /* CURL_BUILD_MAC */

#include "urldata.h"
#include "sendf.h"
#include "inet_pton.h"
#include "connect.h"
#include "select.h"
#include "vtls.h"
#include "sectransp.h"
#include "curl_printf.h"
#include "strdup.h"

#include "curl_memory.h"
/* The last #include file should be: */
#include "memdebug.h"

/* From MacTypes.h (which we can't include because it isn't present in iOS: */
#define ioErr -36
#define paramErr -50

struct ssl_backend_data {
  SSLContextRef ssl_ctx;
  curl_socket_t ssl_sockfd;
  bool ssl_direction; /* true if writing, false if reading */
  size_t ssl_write_buffered_length;
};

struct st_cipher {
  const char *name; /* Cipher suite IANA name. It starts with "TLS_" prefix */
  const char *alias_name; /* Alias name is the same as OpenSSL cipher name */
  SSLCipherSuite num; /* Cipher suite code/number defined in IANA registry */
  bool weak; /* Flag to mark cipher as weak based on previous implementation
                of Secure Transport back-end by CURL */
};

/* Macro to initialize st_cipher data structure: stringify id to name, cipher
   number/id, 'weak' suite flag
 */
#define CIPHER_DEF(num, alias, weak) \
  { #num, alias, num, weak }

/*
 Macro to initialize st_cipher data structure with name, code (IANA cipher
 number/id value), and 'weak' suite flag. The first 28 cipher suite numbers
 have the same IANA code for both SSL and TLS standards: numbers 0x0000 to
 0x001B. They have different names though. The first 4 letters of the cipher
 suite name are the protocol name: "SSL_" or "TLS_", rest of the IANA name is
 the same for both SSL and TLS cipher suite name.
 The second part of the problem is that macOS/iOS SDKs don't define all TLS
 codes but only 12 of them. The SDK defines all SSL codes though, i.e. SSL_NUM
 constant is always defined for those 28 ciphers while TLS_NUM is defined only
 for 12 of the first 28 ciphers. Those 12 TLS cipher codes match to
 corresponding SSL enum value and represent the same cipher suite. Therefore
 we'll use the SSL enum value for those cipher suites because it is defined
 for all 28 of them.
 We make internal data consistent and based on TLS names, i.e. all st_cipher
 item names start with the "TLS_" prefix.
 Summarizing all the above, those 28 first ciphers are presented in our table
 with both TLS and SSL names. Their cipher numbers are assigned based on the
 SDK enum value for the SSL cipher, which matches to IANA TLS number.
 */
#define CIPHER_DEF_SSLTLS(num_wo_prefix, alias, weak) \
  { "TLS_" #num_wo_prefix, alias, SSL_##num_wo_prefix, weak }

/*
 Cipher suites were marked as weak based on the following:
 RC4 encryption - rfc7465, the document contains a list of deprecated ciphers.
     Marked in the code below as weak.
 RC2 encryption - many mentions, was found vulnerable to a relatively easy
     attack https://link.springer.com/chapter/10.1007%2F3-540-69710-1_14
     Marked in the code below as weak.
 DES and IDEA encryption - rfc5469, has a list of deprecated ciphers.
     Marked in the code below as weak.
 Anonymous Diffie-Hellman authentication and anonymous elliptic curve
     Diffie-Hellman - vulnerable to a man-in-the-middle attack. Deprecated by
     RFC 4346 aka TLS 1.1 (section A.5, page 60)
 Null bulk encryption suites - not encrypted communication
 Export ciphers, i.e. ciphers with restrictions to be used outside the US for
     software exported to some countries, they were excluded from TLS 1.1
     version. More precisely, they were noted as ciphers which MUST NOT be
     negotiated in RFC 4346 aka TLS 1.1 (section A.5, pages 60 and 61).
     All of those filters were considered weak because they contain a weak
     algorithm like DES, RC2 or RC4, and already considered weak by other
     criteria.
 3DES - NIST deprecated it and is going to retire it by 2023
 https://csrc.nist.gov/News/2017/Update-to-Current-Use-and-Deprecation-of-TDEA
     OpenSSL https://www.openssl.org/blog/blog/2016/08/24/sweet32/ also
     deprecated those ciphers. Some other libraries also consider it
     vulnerable or at least not strong enough.

 CBC ciphers are vulnerable with SSL3.0 and TLS1.0:
 https://www.cisco.com/c/en/us/support/docs/security/email-security-appliance
 /118518-technote-esa-00.html
     We don't take care of this issue because it is resolved by later TLS
     versions and for us, it requires more complicated checks, we need to
     check a protocol version also. Vulnerability doesn't look very critical
     and we do not filter out those cipher suites.
 */

#define CIPHER_WEAK_NOT_ENCRYPTED   TRUE
#define CIPHER_WEAK_RC_ENCRYPTION   TRUE
#define CIPHER_WEAK_DES_ENCRYPTION  TRUE
#define CIPHER_WEAK_IDEA_ENCRYPTION TRUE
#define CIPHER_WEAK_ANON_AUTH       TRUE
#define CIPHER_WEAK_3DES_ENCRYPTION TRUE
#define CIPHER_STRONG_ENOUGH        FALSE

/* Please do not change the order of the first ciphers available for SSL.
   Do not insert and do not delete any of them. Code below
   depends on their order and continuity.
   If you add a new cipher, please maintain order by number, i.e.
   insert in between existing items to appropriate place based on
   cipher suite IANA number
*/
const static struct st_cipher ciphertable[] = {
  /* SSL version 3.0 and initial TLS 1.0 cipher suites.
     Defined since SDK 10.2.8 */
  CIPHER_DEF_SSLTLS(NULL_WITH_NULL_NULL,                           /* 0x0000 */
                    NULL,
                    CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF_SSLTLS(RSA_WITH_NULL_MD5,                             /* 0x0001 */
                    "NULL-MD5",
                    CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF_SSLTLS(RSA_WITH_NULL_SHA,                             /* 0x0002 */
                    "NULL-SHA",
                    CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF_SSLTLS(RSA_EXPORT_WITH_RC4_40_MD5,                    /* 0x0003 */
                    "EXP-RC4-MD5",
                    CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF_SSLTLS(RSA_WITH_RC4_128_MD5,                          /* 0x0004 */
                    "RC4-MD5",
                    CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF_SSLTLS(RSA_WITH_RC4_128_SHA,                          /* 0x0005 */
                    "RC4-SHA",
                    CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF_SSLTLS(RSA_EXPORT_WITH_RC2_CBC_40_MD5,                /* 0x0006 */
                    "EXP-RC2-CBC-MD5",
                    CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF_SSLTLS(RSA_WITH_IDEA_CBC_SHA,                         /* 0x0007 */
                    "IDEA-CBC-SHA",
                    CIPHER_WEAK_IDEA_ENCRYPTION),
  CIPHER_DEF_SSLTLS(RSA_EXPORT_WITH_DES40_CBC_SHA,                 /* 0x0008 */
                    "EXP-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(RSA_WITH_DES_CBC_SHA,                          /* 0x0009 */
                    "DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(RSA_WITH_3DES_EDE_CBC_SHA,                     /* 0x000A */
                    "DES-CBC3-SHA",
                    CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DH_DSS_EXPORT_WITH_DES40_CBC_SHA,              /* 0x000B */
                    "EXP-DH-DSS-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DH_DSS_WITH_DES_CBC_SHA,                       /* 0x000C */
                    "DH-DSS-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DH_DSS_WITH_3DES_EDE_CBC_SHA,                  /* 0x000D */
                    "DH-DSS-DES-CBC3-SHA",
                    CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DH_RSA_EXPORT_WITH_DES40_CBC_SHA,              /* 0x000E */
                    "EXP-DH-RSA-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DH_RSA_WITH_DES_CBC_SHA,                       /* 0x000F */
                    "DH-RSA-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DH_RSA_WITH_3DES_EDE_CBC_SHA,                  /* 0x0010 */
                    "DH-RSA-DES-CBC3-SHA",
                    CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DHE_DSS_EXPORT_WITH_DES40_CBC_SHA,             /* 0x0011 */
                    "EXP-EDH-DSS-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DHE_DSS_WITH_DES_CBC_SHA,                      /* 0x0012 */
                    "EDH-DSS-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DHE_DSS_WITH_3DES_EDE_CBC_SHA,                 /* 0x0013 */
                    "DHE-DSS-DES-CBC3-SHA",
                    CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DHE_RSA_EXPORT_WITH_DES40_CBC_SHA,             /* 0x0014 */
                    "EXP-EDH-RSA-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DHE_RSA_WITH_DES_CBC_SHA,                      /* 0x0015 */
                    "EDH-RSA-DES-CBC-SHA",
                    CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DHE_RSA_WITH_3DES_EDE_CBC_SHA,                 /* 0x0016 */
                    "DHE-RSA-DES-CBC3-SHA",
                    CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF_SSLTLS(DH_anon_EXPORT_WITH_RC4_40_MD5,                /* 0x0017 */
                    "EXP-ADH-RC4-MD5",
                    CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF_SSLTLS(DH_anon_WITH_RC4_128_MD5,                      /* 0x0018 */
                    "ADH-RC4-MD5",
                    CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF_SSLTLS(DH_anon_EXPORT_WITH_DES40_CBC_SHA,             /* 0x0019 */
                    "EXP-ADH-DES-CBC-SHA",
                    CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF_SSLTLS(DH_anon_WITH_DES_CBC_SHA,                      /* 0x001A */
                    "ADH-DES-CBC-SHA",
                    CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF_SSLTLS(DH_anon_WITH_3DES_EDE_CBC_SHA,                 /* 0x001B */
                    "ADH-DES-CBC3-SHA",
                    CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(SSL_FORTEZZA_DMS_WITH_NULL_SHA,                       /* 0x001C */
             NULL,
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(SSL_FORTEZZA_DMS_WITH_FORTEZZA_CBC_SHA,               /* 0x001D */
             NULL,
             CIPHER_STRONG_ENOUGH),

#if CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7
  /* RFC 4785 - Pre-Shared Key (PSK) Ciphersuites with NULL Encryption */
  CIPHER_DEF(TLS_PSK_WITH_NULL_SHA,                                /* 0x002C */
             "PSK-NULL-SHA",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_DHE_PSK_WITH_NULL_SHA,                            /* 0x002D */
             "DHE-PSK-NULL-SHA",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_RSA_PSK_WITH_NULL_SHA,                            /* 0x002E */
             "RSA-PSK-NULL-SHA",
             CIPHER_WEAK_NOT_ENCRYPTED),
#endif /* CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7 */

  /* TLS addenda using AES, per RFC 3268. Defined since SDK 10.4u */
  CIPHER_DEF(TLS_RSA_WITH_AES_128_CBC_SHA,                         /* 0x002F */
             "AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_DSS_WITH_AES_128_CBC_SHA,                      /* 0x0030 */
             "DH-DSS-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_RSA_WITH_AES_128_CBC_SHA,                      /* 0x0031 */
             "DH-RSA-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_DSS_WITH_AES_128_CBC_SHA,                     /* 0x0032 */
             "DHE-DSS-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_RSA_WITH_AES_128_CBC_SHA,                     /* 0x0033 */
             "DHE-RSA-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_anon_WITH_AES_128_CBC_SHA,                     /* 0x0034 */
             "ADH-AES128-SHA",
             CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF(TLS_RSA_WITH_AES_256_CBC_SHA,                         /* 0x0035 */
             "AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_DSS_WITH_AES_256_CBC_SHA,                      /* 0x0036 */
             "DH-DSS-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_RSA_WITH_AES_256_CBC_SHA,                      /* 0x0037 */
             "DH-RSA-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_DSS_WITH_AES_256_CBC_SHA,                     /* 0x0038 */
             "DHE-DSS-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_RSA_WITH_AES_256_CBC_SHA,                     /* 0x0039 */
             "DHE-RSA-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_anon_WITH_AES_256_CBC_SHA,                     /* 0x003A */
             "ADH-AES256-SHA",
             CIPHER_WEAK_ANON_AUTH),

#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
  /* TLS 1.2 addenda, RFC 5246 */
  /* Server provided RSA certificate for key exchange. */
  CIPHER_DEF(TLS_RSA_WITH_NULL_SHA256,                             /* 0x003B */
             "NULL-SHA256",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_RSA_WITH_AES_128_CBC_SHA256,                      /* 0x003C */
             "AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_WITH_AES_256_CBC_SHA256,                      /* 0x003D */
             "AES256-SHA256",
             CIPHER_STRONG_ENOUGH),
  /* Server-authenticated (and optionally client-authenticated)
     Diffie-Hellman. */
  CIPHER_DEF(TLS_DH_DSS_WITH_AES_128_CBC_SHA256,                   /* 0x003E */
             "DH-DSS-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_RSA_WITH_AES_128_CBC_SHA256,                   /* 0x003F */
             "DH-RSA-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_DSS_WITH_AES_128_CBC_SHA256,                  /* 0x0040 */
             "DHE-DSS-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),

  /* TLS 1.2 addenda, RFC 5246 */
  CIPHER_DEF(TLS_DHE_RSA_WITH_AES_128_CBC_SHA256,                  /* 0x0067 */
             "DHE-RSA-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_DSS_WITH_AES_256_CBC_SHA256,                   /* 0x0068 */
             "DH-DSS-AES256-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_RSA_WITH_AES_256_CBC_SHA256,                   /* 0x0069 */
             "DH-RSA-AES256-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_DSS_WITH_AES_256_CBC_SHA256,                  /* 0x006A */
             "DHE-DSS-AES256-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_RSA_WITH_AES_256_CBC_SHA256,                  /* 0x006B */
             "DHE-RSA-AES256-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_anon_WITH_AES_128_CBC_SHA256,                  /* 0x006C */
             "ADH-AES128-SHA256",
             CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF(TLS_DH_anon_WITH_AES_256_CBC_SHA256,                  /* 0x006D */
             "ADH-AES256-SHA256",
             CIPHER_WEAK_ANON_AUTH),
#endif /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */

#if CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7
  /* Addendum from RFC 4279, TLS PSK */
  CIPHER_DEF(TLS_PSK_WITH_RC4_128_SHA,                             /* 0x008A */
             "PSK-RC4-SHA",
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(TLS_PSK_WITH_3DES_EDE_CBC_SHA,                        /* 0x008B */
             "PSK-3DES-EDE-CBC-SHA",
             CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(TLS_PSK_WITH_AES_128_CBC_SHA,                         /* 0x008C */
             "PSK-AES128-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_PSK_WITH_AES_256_CBC_SHA,                         /* 0x008D */
             "PSK-AES256-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_PSK_WITH_RC4_128_SHA,                         /* 0x008E */
             "DHE-PSK-RC4-SHA",
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(TLS_DHE_PSK_WITH_3DES_EDE_CBC_SHA,                    /* 0x008F */
             "DHE-PSK-3DES-EDE-CBC-SHA",
             CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(TLS_DHE_PSK_WITH_AES_128_CBC_SHA,                     /* 0x0090 */
             "DHE-PSK-AES128-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_PSK_WITH_AES_256_CBC_SHA,                     /* 0x0091 */
             "DHE-PSK-AES256-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_PSK_WITH_RC4_128_SHA,                         /* 0x0092 */
             "RSA-PSK-RC4-SHA",
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(TLS_RSA_PSK_WITH_3DES_EDE_CBC_SHA,                    /* 0x0093 */
             "RSA-PSK-3DES-EDE-CBC-SHA",
             CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(TLS_RSA_PSK_WITH_AES_128_CBC_SHA,                     /* 0x0094 */
             "RSA-PSK-AES128-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_PSK_WITH_AES_256_CBC_SHA,                     /* 0x0095 */
             "RSA-PSK-AES256-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
#endif /* CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7 */

#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
  /* Addenda from rfc 5288 AES Galois Counter Mode (GCM) Cipher Suites
     for TLS. */
  CIPHER_DEF(TLS_RSA_WITH_AES_128_GCM_SHA256,                      /* 0x009C */
             "AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_WITH_AES_256_GCM_SHA384,                      /* 0x009D */
             "AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_RSA_WITH_AES_128_GCM_SHA256,                  /* 0x009E */
             "DHE-RSA-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_RSA_WITH_AES_256_GCM_SHA384,                  /* 0x009F */
             "DHE-RSA-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_RSA_WITH_AES_128_GCM_SHA256,                   /* 0x00A0 */
             "DH-RSA-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_RSA_WITH_AES_256_GCM_SHA384,                   /* 0x00A1 */
             "DH-RSA-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_DSS_WITH_AES_128_GCM_SHA256,                  /* 0x00A2 */
             "DHE-DSS-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_DSS_WITH_AES_256_GCM_SHA384,                  /* 0x00A3 */
             "DHE-DSS-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_DSS_WITH_AES_128_GCM_SHA256,                   /* 0x00A4 */
             "DH-DSS-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_DSS_WITH_AES_256_GCM_SHA384,                   /* 0x00A5 */
             "DH-DSS-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DH_anon_WITH_AES_128_GCM_SHA256,                  /* 0x00A6 */
             "ADH-AES128-GCM-SHA256",
             CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF(TLS_DH_anon_WITH_AES_256_GCM_SHA384,                  /* 0x00A7 */
             "ADH-AES256-GCM-SHA384",
             CIPHER_WEAK_ANON_AUTH),
#endif /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */

#if CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7
  /* RFC 5487 - PSK with SHA-256/384 and AES GCM */
  CIPHER_DEF(TLS_PSK_WITH_AES_128_GCM_SHA256,                      /* 0x00A8 */
             "PSK-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_PSK_WITH_AES_256_GCM_SHA384,                      /* 0x00A9 */
             "PSK-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_PSK_WITH_AES_128_GCM_SHA256,                  /* 0x00AA */
             "DHE-PSK-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_PSK_WITH_AES_256_GCM_SHA384,                  /* 0x00AB */
             "DHE-PSK-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_PSK_WITH_AES_128_GCM_SHA256,                  /* 0x00AC */
             "RSA-PSK-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_PSK_WITH_AES_256_GCM_SHA384,                  /* 0x00AD */
             "RSA-PSK-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_PSK_WITH_AES_128_CBC_SHA256,                      /* 0x00AE */
             "PSK-AES128-CBC-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_PSK_WITH_AES_256_CBC_SHA384,                      /* 0x00AF */
             "PSK-AES256-CBC-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_PSK_WITH_NULL_SHA256,                             /* 0x00B0 */
             "PSK-NULL-SHA256",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_PSK_WITH_NULL_SHA384,                             /* 0x00B1 */
             "PSK-NULL-SHA384",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_DHE_PSK_WITH_AES_128_CBC_SHA256,                  /* 0x00B2 */
             "DHE-PSK-AES128-CBC-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_PSK_WITH_AES_256_CBC_SHA384,                  /* 0x00B3 */
             "DHE-PSK-AES256-CBC-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_DHE_PSK_WITH_NULL_SHA256,                         /* 0x00B4 */
             "DHE-PSK-NULL-SHA256",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_DHE_PSK_WITH_NULL_SHA384,                         /* 0x00B5 */
             "DHE-PSK-NULL-SHA384",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_RSA_PSK_WITH_AES_128_CBC_SHA256,                  /* 0x00B6 */
             "RSA-PSK-AES128-CBC-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_PSK_WITH_AES_256_CBC_SHA384,                  /* 0x00B7 */
             "RSA-PSK-AES256-CBC-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_RSA_PSK_WITH_NULL_SHA256,                         /* 0x00B8 */
             "RSA-PSK-NULL-SHA256",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_RSA_PSK_WITH_NULL_SHA384,                         /* 0x00B9 */
             "RSA-PSK-NULL-SHA384",
             CIPHER_WEAK_NOT_ENCRYPTED),
#endif /* CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7 */

  /* RFC 5746 - Secure Renegotiation. This is not a real suite,
     it is a response to initiate negotiation again */
  CIPHER_DEF(TLS_EMPTY_RENEGOTIATION_INFO_SCSV,                    /* 0x00FF */
             NULL,
             CIPHER_STRONG_ENOUGH),

#if CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11
  /* TLS 1.3 standard cipher suites for ChaCha20+Poly1305.
     Note: TLS 1.3 ciphersuites do not specify the key exchange
     algorithm -- they only specify the symmetric ciphers.
     Cipher alias name matches to OpenSSL cipher name, and for
     TLS 1.3 ciphers */
  CIPHER_DEF(TLS_AES_128_GCM_SHA256,                               /* 0x1301 */
             NULL,  /* The OpenSSL cipher name matches to the IANA name */
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_AES_256_GCM_SHA384,                               /* 0x1302 */
             NULL,  /* The OpenSSL cipher name matches to the IANA name */
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_CHACHA20_POLY1305_SHA256,                         /* 0x1303 */
             NULL,  /* The OpenSSL cipher name matches to the IANA name */
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_AES_128_CCM_SHA256,                               /* 0x1304 */
             NULL,  /* The OpenSSL cipher name matches to the IANA name */
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_AES_128_CCM_8_SHA256,                             /* 0x1305 */
             NULL,  /* The OpenSSL cipher name matches to the IANA name */
             CIPHER_STRONG_ENOUGH),
#endif /* CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11 */

#if CURL_BUILD_MAC_10_6 || CURL_BUILD_IOS
  /* ECDSA addenda, RFC 4492 */
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_NULL_SHA,                         /* 0xC001 */
             "ECDH-ECDSA-NULL-SHA",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_RC4_128_SHA,                      /* 0xC002 */
             "ECDH-ECDSA-RC4-SHA",
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_3DES_EDE_CBC_SHA,                 /* 0xC003 */
             "ECDH-ECDSA-DES-CBC3-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA,                  /* 0xC004 */
             "ECDH-ECDSA-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA,                  /* 0xC005 */
             "ECDH-ECDSA-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_NULL_SHA,                        /* 0xC006 */
             "ECDHE-ECDSA-NULL-SHA",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_RC4_128_SHA,                     /* 0xC007 */
             "ECDHE-ECDSA-RC4-SHA",
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA,                /* 0xC008 */
             "ECDHE-ECDSA-DES-CBC3-SHA",
             CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA,                 /* 0xC009 */
             "ECDHE-ECDSA-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA,                 /* 0xC00A */
             "ECDHE-ECDSA-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_NULL_SHA,                           /* 0xC00B */
             "ECDH-RSA-NULL-SHA",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_RC4_128_SHA,                        /* 0xC00C */
             "ECDH-RSA-RC4-SHA",
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_3DES_EDE_CBC_SHA,                   /* 0xC00D */
             "ECDH-RSA-DES-CBC3-SHA",
             CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_AES_128_CBC_SHA,                    /* 0xC00E */
             "ECDH-RSA-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_AES_256_CBC_SHA,                    /* 0xC00F */
             "ECDH-RSA-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_NULL_SHA,                          /* 0xC010 */
             "ECDHE-RSA-NULL-SHA",
             CIPHER_WEAK_NOT_ENCRYPTED),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_RC4_128_SHA,                       /* 0xC011 */
             "ECDHE-RSA-RC4-SHA",
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA,                  /* 0xC012 */
             "ECDHE-RSA-DES-CBC3-SHA",
             CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA,                   /* 0xC013 */
             "ECDHE-RSA-AES128-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA,                   /* 0xC014 */
             "ECDHE-RSA-AES256-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_anon_WITH_NULL_SHA,                          /* 0xC015 */
             "AECDH-NULL-SHA",
             CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF(TLS_ECDH_anon_WITH_RC4_128_SHA,                       /* 0xC016 */
             "AECDH-RC4-SHA",
             CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF(TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA,                  /* 0xC017 */
             "AECDH-DES-CBC3-SHA",
             CIPHER_WEAK_3DES_ENCRYPTION),
  CIPHER_DEF(TLS_ECDH_anon_WITH_AES_128_CBC_SHA,                   /* 0xC018 */
             "AECDH-AES128-SHA",
             CIPHER_WEAK_ANON_AUTH),
  CIPHER_DEF(TLS_ECDH_anon_WITH_AES_256_CBC_SHA,                   /* 0xC019 */
             "AECDH-AES256-SHA",
             CIPHER_WEAK_ANON_AUTH),
#endif /* CURL_BUILD_MAC_10_6 || CURL_BUILD_IOS */

#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
  /* Addenda from rfc 5289  Elliptic Curve Cipher Suites with
     HMAC SHA-256/384. */
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256,              /* 0xC023 */
             "ECDHE-ECDSA-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384,              /* 0xC024 */
             "ECDHE-ECDSA-AES256-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_AES_128_CBC_SHA256,               /* 0xC025 */
             "ECDH-ECDSA-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_AES_256_CBC_SHA384,               /* 0xC026 */
             "ECDH-ECDSA-AES256-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256,                /* 0xC027 */
             "ECDHE-RSA-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384,                /* 0xC028 */
             "ECDHE-RSA-AES256-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_AES_128_CBC_SHA256,                 /* 0xC029 */
             "ECDH-RSA-AES128-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_AES_256_CBC_SHA384,                 /* 0xC02A */
             "ECDH-RSA-AES256-SHA384",
             CIPHER_STRONG_ENOUGH),
  /* Addenda from rfc 5289  Elliptic Curve Cipher Suites with
     SHA-256/384 and AES Galois Counter Mode (GCM) */
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,              /* 0xC02B */
             "ECDHE-ECDSA-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,              /* 0xC02C */
             "ECDHE-ECDSA-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_AES_128_GCM_SHA256,               /* 0xC02D */
             "ECDH-ECDSA-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_ECDSA_WITH_AES_256_GCM_SHA384,               /* 0xC02E */
             "ECDH-ECDSA-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,                /* 0xC02F */
             "ECDHE-RSA-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,                /* 0xC030 */
             "ECDHE-RSA-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_AES_128_GCM_SHA256,                 /* 0xC031 */
             "ECDH-RSA-AES128-GCM-SHA256",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDH_RSA_WITH_AES_256_GCM_SHA384,                 /* 0xC032 */
             "ECDH-RSA-AES256-GCM-SHA384",
             CIPHER_STRONG_ENOUGH),
#endif /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */

#if CURL_BUILD_MAC_10_15 || CURL_BUILD_IOS_13
  /* ECDHE_PSK Cipher Suites for Transport Layer Security (TLS), RFC 5489 */
  CIPHER_DEF(TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA,                   /* 0xC035 */
             "ECDHE-PSK-AES128-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA,                   /* 0xC036 */
             "ECDHE-PSK-AES256-CBC-SHA",
             CIPHER_STRONG_ENOUGH),
#endif /* CURL_BUILD_MAC_10_15 || CURL_BUILD_IOS_13 */

#if CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11
  /* Addenda from rfc 7905  ChaCha20-Poly1305 Cipher Suites for
     Transport Layer Security (TLS). */
  CIPHER_DEF(TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,          /* 0xCCA8 */
             "ECDHE-RSA-CHACHA20-POLY1305",
             CIPHER_STRONG_ENOUGH),
  CIPHER_DEF(TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,        /* 0xCCA9 */
             "ECDHE-ECDSA-CHACHA20-POLY1305",
             CIPHER_STRONG_ENOUGH),
#endif /* CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11 */

#if CURL_BUILD_MAC_10_15 || CURL_BUILD_IOS_13
  /* ChaCha20-Poly1305 Cipher Suites for Transport Layer Security (TLS),
     RFC 7905 */
  CIPHER_DEF(TLS_PSK_WITH_CHACHA20_POLY1305_SHA256,                /* 0xCCAB */
             "PSK-CHACHA20-POLY1305",
             CIPHER_STRONG_ENOUGH),
#endif /* CURL_BUILD_MAC_10_15 || CURL_BUILD_IOS_13 */

  /* Tags for SSL 2 cipher kinds which are not specified for SSL 3.
     Defined since SDK 10.2.8 */
  CIPHER_DEF(SSL_RSA_WITH_RC2_CBC_MD5,                             /* 0xFF80 */
             NULL,
             CIPHER_WEAK_RC_ENCRYPTION),
  CIPHER_DEF(SSL_RSA_WITH_IDEA_CBC_MD5,                            /* 0xFF81 */
             NULL,
             CIPHER_WEAK_IDEA_ENCRYPTION),
  CIPHER_DEF(SSL_RSA_WITH_DES_CBC_MD5,                             /* 0xFF82 */
             NULL,
             CIPHER_WEAK_DES_ENCRYPTION),
  CIPHER_DEF(SSL_RSA_WITH_3DES_EDE_CBC_MD5,                        /* 0xFF83 */
             NULL,
             CIPHER_WEAK_3DES_ENCRYPTION),
};

#define NUM_OF_CIPHERS sizeof(ciphertable)/sizeof(ciphertable[0])


/* pinned public key support tests */

/* version 1 supports macOS 10.12+ and iOS 10+ */
#if ((TARGET_OS_IPHONE && __IPHONE_OS_VERSION_MIN_REQUIRED >= 100000) || \
    (!TARGET_OS_IPHONE && __MAC_OS_X_VERSION_MIN_REQUIRED  >= 101200))
#define SECTRANSP_PINNEDPUBKEY_V1 1
#endif

/* version 2 supports MacOSX 10.7+ */
#if (!TARGET_OS_IPHONE && __MAC_OS_X_VERSION_MIN_REQUIRED >= 1070)
#define SECTRANSP_PINNEDPUBKEY_V2 1
#endif

#if defined(SECTRANSP_PINNEDPUBKEY_V1) || defined(SECTRANSP_PINNEDPUBKEY_V2)
/* this backend supports CURLOPT_PINNEDPUBLICKEY */
#define SECTRANSP_PINNEDPUBKEY 1
#endif /* SECTRANSP_PINNEDPUBKEY */

#ifdef SECTRANSP_PINNEDPUBKEY
/* both new and old APIs return rsa keys missing the spki header (not DER) */
static const unsigned char rsa4096SpkiHeader[] = {
                                       0x30, 0x82, 0x02, 0x22, 0x30, 0x0d,
                                       0x06, 0x09, 0x2a, 0x86, 0x48, 0x86,
                                       0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05,
                                       0x00, 0x03, 0x82, 0x02, 0x0f, 0x00};

static const unsigned char rsa2048SpkiHeader[] = {
                                       0x30, 0x82, 0x01, 0x22, 0x30, 0x0d,
                                       0x06, 0x09, 0x2a, 0x86, 0x48, 0x86,
                                       0xf7, 0x0d, 0x01, 0x01, 0x01, 0x05,
                                       0x00, 0x03, 0x82, 0x01, 0x0f, 0x00};
#ifdef SECTRANSP_PINNEDPUBKEY_V1
/* the *new* version doesn't return DER encoded ecdsa certs like the old... */
static const unsigned char ecDsaSecp256r1SpkiHeader[] = {
                                       0x30, 0x59, 0x30, 0x13, 0x06, 0x07,
                                       0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02,
                                       0x01, 0x06, 0x08, 0x2a, 0x86, 0x48,
                                       0xce, 0x3d, 0x03, 0x01, 0x07, 0x03,
                                       0x42, 0x00};

static const unsigned char ecDsaSecp384r1SpkiHeader[] = {
                                       0x30, 0x76, 0x30, 0x10, 0x06, 0x07,
                                       0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02,
                                       0x01, 0x06, 0x05, 0x2b, 0x81, 0x04,
                                       0x00, 0x22, 0x03, 0x62, 0x00};
#endif /* SECTRANSP_PINNEDPUBKEY_V1 */
#endif /* SECTRANSP_PINNEDPUBKEY */

/* The following two functions were ripped from Apple sample code,
 * with some modifications: */
static OSStatus SocketRead(SSLConnectionRef connection,
                           void *data,          /* owned by
                                                 * caller, data
                                                 * RETURNED */
                           size_t *dataLength)  /* IN/OUT */
{
  size_t bytesToGo = *dataLength;
  size_t initLen = bytesToGo;
  UInt8 *currData = (UInt8 *)data;
  /*int sock = *(int *)connection;*/
  struct ssl_connect_data *connssl = (struct ssl_connect_data *)connection;
  struct ssl_backend_data *backend = connssl->backend;
  int sock = backend->ssl_sockfd;
  OSStatus rtn = noErr;
  size_t bytesRead;
  ssize_t rrtn;
  int theErr;

  *dataLength = 0;

  for(;;) {
    bytesRead = 0;
    rrtn = read(sock, currData, bytesToGo);
    if(rrtn <= 0) {
      /* this is guesswork... */
      theErr = errno;
      if(rrtn == 0) { /* EOF = server hung up */
        /* the framework will turn this into errSSLClosedNoNotify */
        rtn = errSSLClosedGraceful;
      }
      else /* do the switch */
        switch(theErr) {
          case ENOENT:
            /* connection closed */
            rtn = errSSLClosedGraceful;
            break;
          case ECONNRESET:
            rtn = errSSLClosedAbort;
            break;
          case EAGAIN:
            rtn = errSSLWouldBlock;
            backend->ssl_direction = false;
            break;
          default:
            rtn = ioErr;
            break;
        }
      break;
    }
    else {
      bytesRead = rrtn;
    }
    bytesToGo -= bytesRead;
    currData  += bytesRead;

    if(bytesToGo == 0) {
      /* filled buffer with incoming data, done */
      break;
    }
  }
  *dataLength = initLen - bytesToGo;

  return rtn;
}

static OSStatus SocketWrite(SSLConnectionRef connection,
                            const void *data,
                            size_t *dataLength)  /* IN/OUT */
{
  size_t bytesSent = 0;
  /*int sock = *(int *)connection;*/
  struct ssl_connect_data *connssl = (struct ssl_connect_data *)connection;
  struct ssl_backend_data *backend = connssl->backend;
  int sock = backend->ssl_sockfd;
  ssize_t length;
  size_t dataLen = *dataLength;
  const UInt8 *dataPtr = (UInt8 *)data;
  OSStatus ortn;
  int theErr;

  *dataLength = 0;

  do {
    length = write(sock,
                   (char *)dataPtr + bytesSent,
                   dataLen - bytesSent);
  } while((length > 0) &&
           ( (bytesSent += length) < dataLen) );

  if(length <= 0) {
    theErr = errno;
    if(theErr == EAGAIN) {
      ortn = errSSLWouldBlock;
      backend->ssl_direction = true;
    }
    else {
      ortn = ioErr;
    }
  }
  else {
    ortn = noErr;
  }
  *dataLength = bytesSent;
  return ortn;
}

#ifndef CURL_DISABLE_VERBOSE_STRINGS
CF_INLINE const char *TLSCipherNameForNumber(SSLCipherSuite cipher)
{
  /* The first ciphers in the ciphertable are continuos. Here we do small
     optimization and instead of loop directly get SSL name by cipher number.
   */
  if(cipher <= SSL_FORTEZZA_DMS_WITH_FORTEZZA_CBC_SHA) {
    return ciphertable[cipher].name;
  }
  /* Iterate through the rest of the ciphers */
  for(size_t i = SSL_FORTEZZA_DMS_WITH_FORTEZZA_CBC_SHA + 1;
      i < NUM_OF_CIPHERS;
      ++i) {
    if(ciphertable[i].num == cipher) {
      return ciphertable[i].name;
    }
  }
  return ciphertable[SSL_NULL_WITH_NULL_NULL].name;
}
#endif /* !CURL_DISABLE_VERBOSE_STRINGS */

#if CURL_BUILD_MAC
CF_INLINE void GetDarwinVersionNumber(int *major, int *minor)
{
  int mib[2];
  char *os_version;
  size_t os_version_len;
  char *os_version_major, *os_version_minor;
  char *tok_buf;

  /* Get the Darwin kernel version from the kernel using sysctl(): */
  mib[0] = CTL_KERN;
  mib[1] = KERN_OSRELEASE;
  if(sysctl(mib, 2, NULL, &os_version_len, NULL, 0) == -1)
    return;
  os_version = malloc(os_version_len*sizeof(char));
  if(!os_version)
    return;
  if(sysctl(mib, 2, os_version, &os_version_len, NULL, 0) == -1) {
    free(os_version);
    return;
  }

  /* Parse the version: */
  os_version_major = strtok_r(os_version, ".", &tok_buf);
  os_version_minor = strtok_r(NULL, ".", &tok_buf);
  *major = atoi(os_version_major);
  *minor = atoi(os_version_minor);
  free(os_version);
}
#endif /* CURL_BUILD_MAC */

/* Apple provides a myriad of ways of getting information about a certificate
   into a string. Some aren't available under iOS or newer cats. So here's
   a unified function for getting a string describing the certificate that
   ought to work in all cats starting with Leopard. */
CF_INLINE CFStringRef getsubject(SecCertificateRef cert)
{
  CFStringRef server_cert_summary = CFSTR("(null)");

#if CURL_BUILD_IOS
  /* iOS: There's only one way to do this. */
  server_cert_summary = SecCertificateCopySubjectSummary(cert);
#else
#if CURL_BUILD_MAC_10_7
  /* Lion & later: Get the long description if we can. */
  if(SecCertificateCopyLongDescription != NULL)
    server_cert_summary =
      SecCertificateCopyLongDescription(NULL, cert, NULL);
  else
#endif /* CURL_BUILD_MAC_10_7 */
#if CURL_BUILD_MAC_10_6
  /* Snow Leopard: Get the certificate summary. */
  if(SecCertificateCopySubjectSummary != NULL)
    server_cert_summary = SecCertificateCopySubjectSummary(cert);
  else
#endif /* CURL_BUILD_MAC_10_6 */
  /* Leopard is as far back as we go... */
  (void)SecCertificateCopyCommonName(cert, &server_cert_summary);
#endif /* CURL_BUILD_IOS */
  return server_cert_summary;
}

static CURLcode CopyCertSubject(struct Curl_easy *data,
                                SecCertificateRef cert, char **certp)
{
  CFStringRef c = getsubject(cert);
  CURLcode result = CURLE_OK;
  const char *direct;
  char *cbuf = NULL;
  *certp = NULL;

  if(!c) {
    failf(data, "SSL: invalid CA certificate subject");
    return CURLE_PEER_FAILED_VERIFICATION;
  }

  /* If the subject is already available as UTF-8 encoded (ie 'direct') then
     use that, else convert it. */
  direct = CFStringGetCStringPtr(c, kCFStringEncodingUTF8);
  if(direct) {
    *certp = strdup(direct);
    if(!*certp) {
      failf(data, "SSL: out of memory");
      result = CURLE_OUT_OF_MEMORY;
    }
  }
  else {
    size_t cbuf_size = ((size_t)CFStringGetLength(c) * 4) + 1;
    cbuf = calloc(cbuf_size, 1);
    if(cbuf) {
      if(!CFStringGetCString(c, cbuf, cbuf_size,
                             kCFStringEncodingUTF8)) {
        failf(data, "SSL: invalid CA certificate subject");
        result = CURLE_PEER_FAILED_VERIFICATION;
      }
      else
        /* pass back the buffer */
        *certp = cbuf;
    }
    else {
      failf(data, "SSL: couldn't allocate %zu bytes of memory", cbuf_size);
      result = CURLE_OUT_OF_MEMORY;
    }
  }
  if(result)
    free(cbuf);
  CFRelease(c);
  return result;
}

#if CURL_SUPPORT_MAC_10_6
/* The SecKeychainSearch API was deprecated in Lion, and using it will raise
   deprecation warnings, so let's not compile this unless it's necessary: */
static OSStatus CopyIdentityWithLabelOldSchool(char *label,
                                               SecIdentityRef *out_c_a_k)
{
  OSStatus status = errSecItemNotFound;
  SecKeychainAttributeList attr_list;
  SecKeychainAttribute attr;
  SecKeychainSearchRef search = NULL;
  SecCertificateRef cert = NULL;

  /* Set up the attribute list: */
  attr_list.count = 1L;
  attr_list.attr = &attr;

  /* Set up our lone search criterion: */
  attr.tag = kSecLabelItemAttr;
  attr.data = label;
  attr.length = (UInt32)strlen(label);

  /* Start searching: */
  status = SecKeychainSearchCreateFromAttributes(NULL,
                                                 kSecCertificateItemClass,
                                                 &attr_list,
                                                 &search);
  if(status == noErr) {
    status = SecKeychainSearchCopyNext(search,
                                       (SecKeychainItemRef *)&cert);
    if(status == noErr && cert) {
      /* If we found a certificate, does it have a private key? */
      status = SecIdentityCreateWithCertificate(NULL, cert, out_c_a_k);
      CFRelease(cert);
    }
  }

  if(search)
    CFRelease(search);
  return status;
}
#endif /* CURL_SUPPORT_MAC_10_6 */

static OSStatus CopyIdentityWithLabel(char *label,
                                      SecIdentityRef *out_cert_and_key)
{
  OSStatus status = errSecItemNotFound;

#if CURL_BUILD_MAC_10_7 || CURL_BUILD_IOS
  CFArrayRef keys_list;
  CFIndex keys_list_count;
  CFIndex i;
  CFStringRef common_name;

  /* SecItemCopyMatching() was introduced in iOS and Snow Leopard.
     kSecClassIdentity was introduced in Lion. If both exist, let's use them
     to find the certificate. */
  if(SecItemCopyMatching != NULL && kSecClassIdentity != NULL) {
    CFTypeRef keys[5];
    CFTypeRef values[5];
    CFDictionaryRef query_dict;
    CFStringRef label_cf = CFStringCreateWithCString(NULL, label,
      kCFStringEncodingUTF8);

    /* Set up our search criteria and expected results: */
    values[0] = kSecClassIdentity; /* we want a certificate and a key */
    keys[0] = kSecClass;
    values[1] = kCFBooleanTrue;    /* we want a reference */
    keys[1] = kSecReturnRef;
    values[2] = kSecMatchLimitAll; /* kSecMatchLimitOne would be better if the
                                    * label matching below worked correctly */
    keys[2] = kSecMatchLimit;
    /* identity searches need a SecPolicyRef in order to work */
    values[3] = SecPolicyCreateSSL(false, NULL);
    keys[3] = kSecMatchPolicy;
    /* match the name of the certificate (doesn't work in macOS 10.12.1) */
    values[4] = label_cf;
    keys[4] = kSecAttrLabel;
    query_dict = CFDictionaryCreate(NULL, (const void **)keys,
                                    (const void **)values, 5L,
                                    &kCFCopyStringDictionaryKeyCallBacks,
                                    &kCFTypeDictionaryValueCallBacks);
    CFRelease(values[3]);

    /* Do we have a match? */
    status = SecItemCopyMatching(query_dict, (CFTypeRef *) &keys_list);

    /* Because kSecAttrLabel matching doesn't work with kSecClassIdentity,
     * we need to find the correct identity ourselves */
    if(status == noErr) {
      keys_list_count = CFArrayGetCount(keys_list);
      *out_cert_and_key = NULL;
      status = 1;
      for(i = 0; i<keys_list_count; i++) {
        OSStatus err = noErr;
        SecCertificateRef cert = NULL;
        SecIdentityRef identity =
          (SecIdentityRef) CFArrayGetValueAtIndex(keys_list, i);
        err = SecIdentityCopyCertificate(identity, &cert);
        if(err == noErr) {
          OSStatus copy_status = noErr;
#if CURL_BUILD_IOS
          common_name = SecCertificateCopySubjectSummary(cert);
#elif CURL_BUILD_MAC_10_7
          copy_status = SecCertificateCopyCommonName(cert, &common_name);
#endif
          if(copy_status == noErr &&
            CFStringCompare(common_name, label_cf, 0) == kCFCompareEqualTo) {
            CFRelease(cert);
            CFRelease(common_name);
            CFRetain(identity);
            *out_cert_and_key = identity;
            status = noErr;
            break;
          }
          CFRelease(common_name);
        }
        CFRelease(cert);
      }
    }

    if(keys_list)
      CFRelease(keys_list);
    CFRelease(query_dict);
    CFRelease(label_cf);
  }
  else {
#if CURL_SUPPORT_MAC_10_6
    /* On Leopard and Snow Leopard, fall back to SecKeychainSearch. */
    status = CopyIdentityWithLabelOldSchool(label, out_cert_and_key);
#endif /* CURL_SUPPORT_MAC_10_6 */
  }
#elif CURL_SUPPORT_MAC_10_6
  /* For developers building on older cats, we have no choice but to fall back
     to SecKeychainSearch. */
  status = CopyIdentityWithLabelOldSchool(label, out_cert_and_key);
#endif /* CURL_BUILD_MAC_10_7 || CURL_BUILD_IOS */
  return status;
}

static OSStatus CopyIdentityFromPKCS12File(const char *cPath,
                                           const struct curl_blob *blob,
                                           const char *cPassword,
                                           SecIdentityRef *out_cert_and_key)
{
  OSStatus status = errSecItemNotFound;
  CFURLRef pkcs_url = NULL;
  CFStringRef password = cPassword ? CFStringCreateWithCString(NULL,
    cPassword, kCFStringEncodingUTF8) : NULL;
  CFDataRef pkcs_data = NULL;

  /* We can import P12 files on iOS or OS X 10.7 or later: */
  /* These constants are documented as having first appeared in 10.6 but they
     raise linker errors when used on that cat for some reason. */
#if CURL_BUILD_MAC_10_7 || CURL_BUILD_IOS
  bool resource_imported;

  if(blob) {
    pkcs_data = CFDataCreate(kCFAllocatorDefault,
                             (const unsigned char *)blob->data, blob->len);
    status = (pkcs_data != NULL) ? errSecSuccess : errSecAllocate;
    resource_imported = (pkcs_data != NULL);
  }
  else {
    pkcs_url =
      CFURLCreateFromFileSystemRepresentation(NULL,
                                              (const UInt8 *)cPath,
                                              strlen(cPath), false);
    resource_imported =
      CFURLCreateDataAndPropertiesFromResource(NULL,
                                               pkcs_url, &pkcs_data,
                                               NULL, NULL, &status);
  }

  if(resource_imported) {
    CFArrayRef items = NULL;

  /* On iOS SecPKCS12Import will never add the client certificate to the
   * Keychain.
   *
   * It gives us back a SecIdentityRef that we can use directly. */
#if CURL_BUILD_IOS
    const void *cKeys[] = {kSecImportExportPassphrase};
    const void *cValues[] = {password};
    CFDictionaryRef options = CFDictionaryCreate(NULL, cKeys, cValues,
      password ? 1L : 0L, NULL, NULL);

    if(options != NULL) {
      status = SecPKCS12Import(pkcs_data, options, &items);
      CFRelease(options);
    }


  /* On macOS SecPKCS12Import will always add the client certificate to
   * the Keychain.
   *
   * As this doesn't match iOS, and apps may not want to see their client
   * certificate saved in the user's keychain, we use SecItemImport
   * with a NULL keychain to avoid importing it.
   *
   * This returns a SecCertificateRef from which we can construct a
   * SecIdentityRef.
   */
#elif CURL_BUILD_MAC_10_7
    SecItemImportExportKeyParameters keyParams;
    SecExternalFormat inputFormat = kSecFormatPKCS12;
    SecExternalItemType inputType = kSecItemTypeCertificate;

    memset(&keyParams, 0x00, sizeof(keyParams));
    keyParams.version    = SEC_KEY_IMPORT_EXPORT_PARAMS_VERSION;
    keyParams.passphrase = password;

    status = SecItemImport(pkcs_data, NULL, &inputFormat, &inputType,
                           0, &keyParams, NULL, &items);
#endif


    /* Extract the SecIdentityRef */
    if(status == errSecSuccess && items && CFArrayGetCount(items)) {
      CFIndex i, count;
      count = CFArrayGetCount(items);

      for(i = 0; i < count; i++) {
        CFTypeRef item = (CFTypeRef) CFArrayGetValueAtIndex(items, i);
        CFTypeID  itemID = CFGetTypeID(item);

        if(itemID == CFDictionaryGetTypeID()) {
          CFTypeRef identity = (CFTypeRef) CFDictionaryGetValue(
                                                 (CFDictionaryRef) item,
                                                 kSecImportItemIdentity);
          CFRetain(identity);
          *out_cert_and_key = (SecIdentityRef) identity;
          break;
        }
#if CURL_BUILD_MAC_10_7
        else if(itemID == SecCertificateGetTypeID()) {
          status = SecIdentityCreateWithCertificate(NULL,
                                                 (SecCertificateRef) item,
                                                 out_cert_and_key);
          break;
        }
#endif
      }
    }

    if(items)
      CFRelease(items);
    CFRelease(pkcs_data);
  }
#endif /* CURL_BUILD_MAC_10_7 || CURL_BUILD_IOS */
  if(password)
    CFRelease(password);
  if(pkcs_url)
    CFRelease(pkcs_url);
  return status;
}

/* This code was borrowed from nss.c, with some modifications:
 * Determine whether the nickname passed in is a filename that needs to
 * be loaded as a PEM or a regular NSS nickname.
 *
 * returns 1 for a file
 * returns 0 for not a file
 */
CF_INLINE bool is_file(const char *filename)
{
  struct_stat st;

  if(!filename)
    return false;

  if(stat(filename, &st) == 0)
    return S_ISREG(st.st_mode);
  return false;
}

#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
static CURLcode sectransp_version_from_curl(SSLProtocol *darwinver,
                                            long ssl_version)
{
  switch(ssl_version) {
    case CURL_SSLVERSION_TLSv1_0:
      *darwinver = kTLSProtocol1;
      return CURLE_OK;
    case CURL_SSLVERSION_TLSv1_1:
      *darwinver = kTLSProtocol11;
      return CURLE_OK;
    case CURL_SSLVERSION_TLSv1_2:
      *darwinver = kTLSProtocol12;
      return CURLE_OK;
    case CURL_SSLVERSION_TLSv1_3:
      /* TLS 1.3 support first appeared in iOS 11 and macOS 10.13 */
#if (CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) && HAVE_BUILTIN_AVAILABLE == 1
      if(__builtin_available(macOS 10.13, iOS 11.0, *)) {
        *darwinver = kTLSProtocol13;
        return CURLE_OK;
      }
#endif /* (CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) &&
          HAVE_BUILTIN_AVAILABLE == 1 */
      break;
  }
  return CURLE_SSL_CONNECT_ERROR;
}
#endif

static CURLcode
set_ssl_version_min_max(struct Curl_easy *data, struct connectdata *conn,
                        int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  long ssl_version = SSL_CONN_CONFIG(version);
  long ssl_version_max = SSL_CONN_CONFIG(version_max);
  long max_supported_version_by_os;

  /* macOS 10.5-10.7 supported TLS 1.0 only.
     macOS 10.8 and later, and iOS 5 and later, added TLS 1.1 and 1.2.
     macOS 10.13 and later, and iOS 11 and later, added TLS 1.3. */
#if (CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) && HAVE_BUILTIN_AVAILABLE == 1
  if(__builtin_available(macOS 10.13, iOS 11.0, *)) {
    max_supported_version_by_os = CURL_SSLVERSION_MAX_TLSv1_3;
  }
  else {
    max_supported_version_by_os = CURL_SSLVERSION_MAX_TLSv1_2;
  }
#else
  max_supported_version_by_os = CURL_SSLVERSION_MAX_TLSv1_2;
#endif /* (CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) &&
          HAVE_BUILTIN_AVAILABLE == 1 */

  switch(ssl_version) {
    case CURL_SSLVERSION_DEFAULT:
    case CURL_SSLVERSION_TLSv1:
      ssl_version = CURL_SSLVERSION_TLSv1_0;
      break;
  }

  switch(ssl_version_max) {
    case CURL_SSLVERSION_MAX_NONE:
    case CURL_SSLVERSION_MAX_DEFAULT:
      ssl_version_max = max_supported_version_by_os;
      break;
  }

#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
  if(SSLSetProtocolVersionMax != NULL) {
    SSLProtocol darwin_ver_min = kTLSProtocol1;
    SSLProtocol darwin_ver_max = kTLSProtocol1;
    CURLcode result = sectransp_version_from_curl(&darwin_ver_min,
                                                  ssl_version);
    if(result) {
      failf(data, "unsupported min version passed via CURLOPT_SSLVERSION");
      return result;
    }
    result = sectransp_version_from_curl(&darwin_ver_max,
                                         ssl_version_max >> 16);
    if(result) {
      failf(data, "unsupported max version passed via CURLOPT_SSLVERSION");
      return result;
    }

    (void)SSLSetProtocolVersionMin(backend->ssl_ctx, darwin_ver_min);
    (void)SSLSetProtocolVersionMax(backend->ssl_ctx, darwin_ver_max);
    return result;
  }
  else {
#if CURL_SUPPORT_MAC_10_8
    long i = ssl_version;
    (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                       kSSLProtocolAll,
                                       false);
    for(; i <= (ssl_version_max >> 16); i++) {
      switch(i) {
        case CURL_SSLVERSION_TLSv1_0:
          (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                            kTLSProtocol1,
                                            true);
          break;
        case CURL_SSLVERSION_TLSv1_1:
          (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                            kTLSProtocol11,
                                            true);
          break;
        case CURL_SSLVERSION_TLSv1_2:
          (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                            kTLSProtocol12,
                                            true);
          break;
        case CURL_SSLVERSION_TLSv1_3:
          failf(data, "Your version of the OS does not support TLSv1.3");
          return CURLE_SSL_CONNECT_ERROR;
      }
    }
    return CURLE_OK;
#endif  /* CURL_SUPPORT_MAC_10_8 */
  }
#endif  /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */
  failf(data, "Secure Transport: cannot set SSL protocol");
  return CURLE_SSL_CONNECT_ERROR;
}

static bool is_cipher_suite_strong(SSLCipherSuite suite_num)
{
  for(size_t i = 0; i < NUM_OF_CIPHERS; ++i) {
    if(ciphertable[i].num == suite_num) {
      return !ciphertable[i].weak;
    }
  }
  /* If the cipher is not in our list, assume it is a new one
     and therefore strong. Previous implementation was the same,
     if cipher suite is not in the list, it was considered strong enough */
  return true;
}

static bool is_separator(char c)
{
  /* Return whether character is a cipher list separator. */
  switch(c) {
  case ' ':
  case '\t':
  case ':':
  case ',':
  case ';':
    return true;
  }
  return false;
}

static CURLcode sectransp_set_default_ciphers(struct Curl_easy *data,
                                              SSLContextRef ssl_ctx)
{
  size_t all_ciphers_count = 0UL, allowed_ciphers_count = 0UL, i;
  SSLCipherSuite *all_ciphers = NULL, *allowed_ciphers = NULL;
  OSStatus err = noErr;

#if CURL_BUILD_MAC
  int darwinver_maj = 0, darwinver_min = 0;

  GetDarwinVersionNumber(&darwinver_maj, &darwinver_min);
#endif /* CURL_BUILD_MAC */

  /* Disable cipher suites that ST supports but are not safe. These ciphers
     are unlikely to be used in any case since ST gives other ciphers a much
     higher priority, but it's probably better that we not connect at all than
     to give the user a false sense of security if the server only supports
     insecure ciphers. (Note: We don't care about SSLv2-only ciphers.) */
  err = SSLGetNumberSupportedCiphers(ssl_ctx, &all_ciphers_count);
  if(err != noErr) {
    failf(data, "SSL: SSLGetNumberSupportedCiphers() failed: OSStatus %d",
          err);
    return CURLE_SSL_CIPHER;
  }
  all_ciphers = malloc(all_ciphers_count*sizeof(SSLCipherSuite));
  if(!all_ciphers) {
    failf(data, "SSL: Failed to allocate memory for all ciphers");
    return CURLE_OUT_OF_MEMORY;
  }
  allowed_ciphers = malloc(all_ciphers_count*sizeof(SSLCipherSuite));
  if(!allowed_ciphers) {
    Curl_safefree(all_ciphers);
    failf(data, "SSL: Failed to allocate memory for allowed ciphers");
    return CURLE_OUT_OF_MEMORY;
  }
  err = SSLGetSupportedCiphers(ssl_ctx, all_ciphers,
                               &all_ciphers_count);
  if(err != noErr) {
    Curl_safefree(all_ciphers);
    Curl_safefree(allowed_ciphers);
    return CURLE_SSL_CIPHER;
  }
  for(i = 0UL ; i < all_ciphers_count ; i++) {
#if CURL_BUILD_MAC
   /* There's a known bug in early versions of Mountain Lion where ST's ECC
      ciphers (cipher suite 0xC001 through 0xC032) simply do not work.
      Work around the problem here by disabling those ciphers if we are
      running in an affected version of OS X. */
    if(darwinver_maj == 12 && darwinver_min <= 3 &&
       all_ciphers[i] >= 0xC001 && all_ciphers[i] <= 0xC032) {
      continue;
    }
#endif /* CURL_BUILD_MAC */
    if(is_cipher_suite_strong(all_ciphers[i])) {
      allowed_ciphers[allowed_ciphers_count++] = all_ciphers[i];
    }
  }
  err = SSLSetEnabledCiphers(ssl_ctx, allowed_ciphers,
                             allowed_ciphers_count);
  Curl_safefree(all_ciphers);
  Curl_safefree(allowed_ciphers);
  if(err != noErr) {
    failf(data, "SSL: SSLSetEnabledCiphers() failed: OSStatus %d", err);
    return CURLE_SSL_CIPHER;
  }
  return CURLE_OK;
}

static CURLcode sectransp_set_selected_ciphers(struct Curl_easy *data,
                                               SSLContextRef ssl_ctx,
                                               const char *ciphers)
{
  size_t ciphers_count = 0;
  const char *cipher_start = ciphers;
  OSStatus err = noErr;
  SSLCipherSuite selected_ciphers[NUM_OF_CIPHERS];

  if(!ciphers)
    return CURLE_OK;

  while(is_separator(*ciphers))     /* Skip initial separators. */
    ciphers++;
  if(!*ciphers)
    return CURLE_OK;

  cipher_start = ciphers;
  while(*cipher_start && ciphers_count < NUM_OF_CIPHERS) {
    bool cipher_found = FALSE;
    size_t cipher_len = 0;
    const char *cipher_end = NULL;
    bool tls_name = FALSE;

    /* Skip separators */
    while(is_separator(*cipher_start))
       cipher_start++;
    if(*cipher_start == '\0') {
      break;
    }
    /* Find last position of a cipher in the ciphers string */
    cipher_end = cipher_start;
    while (*cipher_end != '\0' && !is_separator(*cipher_end)) {
      ++cipher_end;
    }

    /* IANA cipher names start with the TLS_ or SSL_ prefix.
       If the 4th symbol of the cipher is '_' we look for a cipher in the
       table by its (TLS) name.
       Otherwise, we try to match cipher by an alias. */
    if(cipher_start[3] == '_') {
      tls_name = TRUE;
    }
    /* Iterate through the cipher table and look for the cipher, starting
       the cipher number 0x01 because the 0x00 is not the real cipher */
    cipher_len = cipher_end - cipher_start;
    for(size_t i = 1; i < NUM_OF_CIPHERS; ++i) {
      const char *table_cipher_name = NULL;
      if(tls_name) {
        table_cipher_name = ciphertable[i].name;
      }
      else if(ciphertable[i].alias_name != NULL) {
        table_cipher_name = ciphertable[i].alias_name;
      }
      else {
        continue;
      }
      /* Compare a part of the string between separators with a cipher name
         in the table and make sure we matched the whole cipher name */
      if(strncmp(cipher_start, table_cipher_name, cipher_len) == 0
          && table_cipher_name[cipher_len] == '\0') {
        selected_ciphers[ciphers_count] = ciphertable[i].num;
        ++ciphers_count;
        cipher_found = TRUE;
        break;
      }
    }
    if(!cipher_found) {
      /* It would be more human-readable if we print the wrong cipher name
         but we don't want to allocate any additional memory and copy the name
         into it, then add it into logs.
         Also, we do not modify an original cipher list string. We just point
         to positions where cipher starts and ends in the cipher list string.
         The message is a bit cryptic and longer than necessary but can be
         understood by humans. */
      failf(data, "SSL: cipher string \"%s\" contains unsupported cipher name"
            " starting position %d and ending position %d",
            ciphers,
            cipher_start - ciphers,
            cipher_end - ciphers);
      return CURLE_SSL_CIPHER;
    }
    if(*cipher_end) {
      cipher_start = cipher_end + 1;
    }
    else {
      break;
    }
  }
  /* All cipher suites in the list are found. Report to logs as-is */
  infof(data, "SSL: Setting cipher suites list \"%s\"", ciphers);

  err = SSLSetEnabledCiphers(ssl_ctx, selected_ciphers, ciphers_count);
  if(err != noErr) {
    failf(data, "SSL: SSLSetEnabledCiphers() failed: OSStatus %d", err);
    return CURLE_SSL_CIPHER;
  }
  return CURLE_OK;
}

static CURLcode sectransp_connect_step1(struct Curl_easy *data,
                                        struct connectdata *conn,
                                        int sockindex)
{
  curl_socket_t sockfd = conn->sock[sockindex];
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  const struct curl_blob *ssl_cablob = SSL_CONN_CONFIG(ca_info_blob);
  const char * const ssl_cafile =
    /* CURLOPT_CAINFO_BLOB overrides CURLOPT_CAINFO */
    (ssl_cablob ? NULL : SSL_CONN_CONFIG(CAfile));
  const bool verifypeer = SSL_CONN_CONFIG(verifypeer);
  char * const ssl_cert = SSL_SET_OPTION(primary.clientcert);
  const struct curl_blob *ssl_cert_blob = SSL_SET_OPTION(primary.cert_blob);
  bool isproxy = SSL_IS_PROXY();
  const char * const hostname = SSL_HOST_NAME();
  const long int port = SSL_HOST_PORT();
#ifdef ENABLE_IPV6
  struct in6_addr addr;
#else
  struct in_addr addr;
#endif /* ENABLE_IPV6 */
  char *ciphers;
  OSStatus err = noErr;
#if CURL_BUILD_MAC
  int darwinver_maj = 0, darwinver_min = 0;

  GetDarwinVersionNumber(&darwinver_maj, &darwinver_min);
#endif /* CURL_BUILD_MAC */

#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
  if(SSLCreateContext != NULL) {  /* use the newer API if available */
    if(backend->ssl_ctx)
      CFRelease(backend->ssl_ctx);
    backend->ssl_ctx = SSLCreateContext(NULL, kSSLClientSide, kSSLStreamType);
    if(!backend->ssl_ctx) {
      failf(data, "SSL: couldn't create a context!");
      return CURLE_OUT_OF_MEMORY;
    }
  }
  else {
  /* The old ST API does not exist under iOS, so don't compile it: */
#if CURL_SUPPORT_MAC_10_8
    if(backend->ssl_ctx)
      (void)SSLDisposeContext(backend->ssl_ctx);
    err = SSLNewContext(false, &(backend->ssl_ctx));
    if(err != noErr) {
      failf(data, "SSL: couldn't create a context: OSStatus %d", err);
      return CURLE_OUT_OF_MEMORY;
    }
#endif /* CURL_SUPPORT_MAC_10_8 */
  }
#else
  if(backend->ssl_ctx)
    (void)SSLDisposeContext(backend->ssl_ctx);
  err = SSLNewContext(false, &(backend->ssl_ctx));
  if(err != noErr) {
    failf(data, "SSL: couldn't create a context: OSStatus %d", err);
    return CURLE_OUT_OF_MEMORY;
  }
#endif /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */
  backend->ssl_write_buffered_length = 0UL; /* reset buffered write length */

  /* check to see if we've been told to use an explicit SSL/TLS version */
#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
  if(SSLSetProtocolVersionMax != NULL) {
    switch(conn->ssl_config.version) {
    case CURL_SSLVERSION_TLSv1:
      (void)SSLSetProtocolVersionMin(backend->ssl_ctx, kTLSProtocol1);
#if (CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) && HAVE_BUILTIN_AVAILABLE == 1
      if(__builtin_available(macOS 10.13, iOS 11.0, *)) {
        (void)SSLSetProtocolVersionMax(backend->ssl_ctx, kTLSProtocol13);
      }
      else {
        (void)SSLSetProtocolVersionMax(backend->ssl_ctx, kTLSProtocol12);
      }
#else
      (void)SSLSetProtocolVersionMax(backend->ssl_ctx, kTLSProtocol12);
#endif /* (CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) &&
          HAVE_BUILTIN_AVAILABLE == 1 */
      break;
    case CURL_SSLVERSION_DEFAULT:
    case CURL_SSLVERSION_TLSv1_0:
    case CURL_SSLVERSION_TLSv1_1:
    case CURL_SSLVERSION_TLSv1_2:
    case CURL_SSLVERSION_TLSv1_3:
      {
        CURLcode result = set_ssl_version_min_max(data, conn, sockindex);
        if(result != CURLE_OK)
          return result;
        break;
      }
    case CURL_SSLVERSION_SSLv3:
    case CURL_SSLVERSION_SSLv2:
      failf(data, "SSL versions not supported");
      return CURLE_NOT_BUILT_IN;
    default:
      failf(data, "Unrecognized parameter passed via CURLOPT_SSLVERSION");
      return CURLE_SSL_CONNECT_ERROR;
    }
  }
  else {
#if CURL_SUPPORT_MAC_10_8
    (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                       kSSLProtocolAll,
                                       false);
    switch(conn->ssl_config.version) {
    case CURL_SSLVERSION_DEFAULT:
    case CURL_SSLVERSION_TLSv1:
      (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                         kTLSProtocol1,
                                         true);
      (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                         kTLSProtocol11,
                                         true);
      (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                         kTLSProtocol12,
                                         true);
      break;
    case CURL_SSLVERSION_TLSv1_0:
    case CURL_SSLVERSION_TLSv1_1:
    case CURL_SSLVERSION_TLSv1_2:
    case CURL_SSLVERSION_TLSv1_3:
      {
        CURLcode result = set_ssl_version_min_max(data, conn, sockindex);
        if(result != CURLE_OK)
          return result;
        break;
      }
    case CURL_SSLVERSION_SSLv3:
    case CURL_SSLVERSION_SSLv2:
      failf(data, "SSL versions not supported");
      return CURLE_NOT_BUILT_IN;
    default:
      failf(data, "Unrecognized parameter passed via CURLOPT_SSLVERSION");
      return CURLE_SSL_CONNECT_ERROR;
    }
#endif  /* CURL_SUPPORT_MAC_10_8 */
  }
#else
  if(conn->ssl_config.version_max != CURL_SSLVERSION_MAX_NONE) {
    failf(data, "Your version of the OS does not support to set maximum"
                " SSL/TLS version");
    return CURLE_SSL_CONNECT_ERROR;
  }
  (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx, kSSLProtocolAll, false);
  switch(conn->ssl_config.version) {
  case CURL_SSLVERSION_DEFAULT:
  case CURL_SSLVERSION_TLSv1:
  case CURL_SSLVERSION_TLSv1_0:
    (void)SSLSetProtocolVersionEnabled(backend->ssl_ctx,
                                       kTLSProtocol1,
                                       true);
    break;
  case CURL_SSLVERSION_TLSv1_1:
    failf(data, "Your version of the OS does not support TLSv1.1");
    return CURLE_SSL_CONNECT_ERROR;
  case CURL_SSLVERSION_TLSv1_2:
    failf(data, "Your version of the OS does not support TLSv1.2");
    return CURLE_SSL_CONNECT_ERROR;
  case CURL_SSLVERSION_TLSv1_3:
    failf(data, "Your version of the OS does not support TLSv1.3");
    return CURLE_SSL_CONNECT_ERROR;
  case CURL_SSLVERSION_SSLv2:
  case CURL_SSLVERSION_SSLv3:
    failf(data, "SSL versions not supported");
    return CURLE_NOT_BUILT_IN;
  default:
    failf(data, "Unrecognized parameter passed via CURLOPT_SSLVERSION");
    return CURLE_SSL_CONNECT_ERROR;
  }
#endif /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */

#if (CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) && HAVE_BUILTIN_AVAILABLE == 1
  if(conn->bits.tls_enable_alpn) {
    if(__builtin_available(macOS 10.13.4, iOS 11, tvOS 11, *)) {
      CFMutableArrayRef alpnArr = CFArrayCreateMutable(NULL, 0,
                                                       &kCFTypeArrayCallBacks);

#ifdef USE_HTTP2
      if(data->state.httpwant >= CURL_HTTP_VERSION_2
#ifndef CURL_DISABLE_PROXY
         && (!isproxy || !conn->bits.tunnel_proxy)
#endif
        ) {
        CFArrayAppendValue(alpnArr, CFSTR(ALPN_H2));
        infof(data, "ALPN, offering %s", ALPN_H2);
      }
#endif

      CFArrayAppendValue(alpnArr, CFSTR(ALPN_HTTP_1_1));
      infof(data, "ALPN, offering %s", ALPN_HTTP_1_1);

      /* expects length prefixed preference ordered list of protocols in wire
       * format
       */
      err = SSLSetALPNProtocols(backend->ssl_ctx, alpnArr);
      if(err != noErr)
        infof(data, "WARNING: failed to set ALPN protocols; OSStatus %d",
              err);
      CFRelease(alpnArr);
    }
  }
#endif

  if(SSL_SET_OPTION(key)) {
    infof(data, "WARNING: SSL: CURLOPT_SSLKEY is ignored by Secure "
          "Transport. The private key must be in the Keychain.");
  }

  if(ssl_cert || ssl_cert_blob) {
    bool is_cert_data = ssl_cert_blob != NULL;
    bool is_cert_file = (!is_cert_data) && is_file(ssl_cert);
    SecIdentityRef cert_and_key = NULL;

    /* User wants to authenticate with a client cert. Look for it. Assume that
       the user wants to use an identity loaded from the Keychain. If not, try
       it as a file on disk */

    if(!is_cert_data)
      err = CopyIdentityWithLabel(ssl_cert, &cert_and_key);
    else
      err = !noErr;
    if((err != noErr) && (is_cert_file || is_cert_data)) {
      if(!SSL_SET_OPTION(cert_type))
        infof(data, "SSL: Certificate type not set, assuming "
              "PKCS#12 format.");
      else if(!strcasecompare(SSL_SET_OPTION(cert_type), "P12")) {
        failf(data, "SSL: The Security framework only supports "
              "loading identities that are in PKCS#12 format.");
        return CURLE_SSL_CERTPROBLEM;
      }

      err = CopyIdentityFromPKCS12File(ssl_cert, ssl_cert_blob,
                                       SSL_SET_OPTION(key_passwd),
                                       &cert_and_key);
    }

    if(err == noErr && cert_and_key) {
      SecCertificateRef cert = NULL;
      CFTypeRef certs_c[1];
      CFArrayRef certs;

      /* If we found one, print it out: */
      err = SecIdentityCopyCertificate(cert_and_key, &cert);
      if(err == noErr) {
        char *certp;
        CURLcode result = CopyCertSubject(data, cert, &certp);
        if(!result) {
          infof(data, "Client certificate: %s", certp);
          free(certp);
        }

        CFRelease(cert);
        if(result == CURLE_PEER_FAILED_VERIFICATION)
          return CURLE_SSL_CERTPROBLEM;
        if(result)
          return result;
      }
      certs_c[0] = cert_and_key;
      certs = CFArrayCreate(NULL, (const void **)certs_c, 1L,
                            &kCFTypeArrayCallBacks);
      err = SSLSetCertificate(backend->ssl_ctx, certs);
      if(certs)
        CFRelease(certs);
      if(err != noErr) {
        failf(data, "SSL: SSLSetCertificate() failed: OSStatus %d", err);
        return CURLE_SSL_CERTPROBLEM;
      }
      CFRelease(cert_and_key);
    }
    else {
      const char *cert_showfilename_error =
        is_cert_data ? "(memory blob)" : ssl_cert;

      switch(err) {
      case errSecAuthFailed: case -25264: /* errSecPkcs12VerifyFailure */
        failf(data, "SSL: Incorrect password for the certificate \"%s\" "
                    "and its private key.", cert_showfilename_error);
        break;
      case -26275: /* errSecDecode */ case -25257: /* errSecUnknownFormat */
        failf(data, "SSL: Couldn't make sense of the data in the "
                    "certificate \"%s\" and its private key.",
                    cert_showfilename_error);
        break;
      case -25260: /* errSecPassphraseRequired */
        failf(data, "SSL The certificate \"%s\" requires a password.",
                    cert_showfilename_error);
        break;
      case errSecItemNotFound:
        failf(data, "SSL: Can't find the certificate \"%s\" and its private "
                    "key in the Keychain.", cert_showfilename_error);
        break;
      default:
        failf(data, "SSL: Can't load the certificate \"%s\" and its private "
                    "key: OSStatus %d", cert_showfilename_error, err);
        break;
      }
      return CURLE_SSL_CERTPROBLEM;
    }
  }

  /* SSL always tries to verify the peer, this only says whether it should
   * fail to connect if the verification fails, or if it should continue
   * anyway. In the latter case the result of the verification is checked with
   * SSL_get_verify_result() below. */
#if CURL_BUILD_MAC_10_6 || CURL_BUILD_IOS
  /* Snow Leopard introduced the SSLSetSessionOption() function, but due to
     a library bug with the way the kSSLSessionOptionBreakOnServerAuth flag
     works, it doesn't work as expected under Snow Leopard, Lion or
     Mountain Lion.
     So we need to call SSLSetEnableCertVerify() on those older cats in order
     to disable certificate validation if the user turned that off.
     (SecureTransport will always validate the certificate chain by
     default.)
  Note:
  Darwin 11.x.x is Lion (10.7)
  Darwin 12.x.x is Mountain Lion (10.8)
  Darwin 13.x.x is Mavericks (10.9)
  Darwin 14.x.x is Yosemite (10.10)
  Darwin 15.x.x is El Capitan (10.11)
  */
#if CURL_BUILD_MAC
  if(SSLSetSessionOption != NULL && darwinver_maj >= 13) {
#else
  if(SSLSetSessionOption != NULL) {
#endif /* CURL_BUILD_MAC */
    bool break_on_auth = !conn->ssl_config.verifypeer ||
      ssl_cafile || ssl_cablob;
    err = SSLSetSessionOption(backend->ssl_ctx,
                              kSSLSessionOptionBreakOnServerAuth,
                              break_on_auth);
    if(err != noErr) {
      failf(data, "SSL: SSLSetSessionOption() failed: OSStatus %d", err);
      return CURLE_SSL_CONNECT_ERROR;
    }
  }
  else {
#if CURL_SUPPORT_MAC_10_8
    err = SSLSetEnableCertVerify(backend->ssl_ctx,
                                 conn->ssl_config.verifypeer?true:false);
    if(err != noErr) {
      failf(data, "SSL: SSLSetEnableCertVerify() failed: OSStatus %d", err);
      return CURLE_SSL_CONNECT_ERROR;
    }
#endif /* CURL_SUPPORT_MAC_10_8 */
  }
#else
  err = SSLSetEnableCertVerify(backend->ssl_ctx,
                               conn->ssl_config.verifypeer?true:false);
  if(err != noErr) {
    failf(data, "SSL: SSLSetEnableCertVerify() failed: OSStatus %d", err);
    return CURLE_SSL_CONNECT_ERROR;
  }
#endif /* CURL_BUILD_MAC_10_6 || CURL_BUILD_IOS */

  if((ssl_cafile || ssl_cablob) && verifypeer) {
    bool is_cert_data = ssl_cablob != NULL;
    bool is_cert_file = (!is_cert_data) && is_file(ssl_cafile);

    if(!(is_cert_file || is_cert_data)) {
      failf(data, "SSL: can't load CA certificate file %s",
            ssl_cafile ? ssl_cafile : "(blob memory)");
      return CURLE_SSL_CACERT_BADFILE;
    }
  }

  /* Configure hostname check. SNI is used if available.
   * Both hostname check and SNI require SSLSetPeerDomainName().
   * Also: the verifyhost setting influences SNI usage */
  if(conn->ssl_config.verifyhost) {
    err = SSLSetPeerDomainName(backend->ssl_ctx, hostname,
    strlen(hostname));

    if(err != noErr) {
      infof(data, "WARNING: SSL: SSLSetPeerDomainName() failed: OSStatus %d",
            err);
    }

    if((Curl_inet_pton(AF_INET, hostname, &addr))
  #ifdef ENABLE_IPV6
    || (Curl_inet_pton(AF_INET6, hostname, &addr))
  #endif
       ) {
      infof(data, "WARNING: using IP address, SNI is being disabled by "
            "the OS.");
    }
  }
  else {
    infof(data, "WARNING: disabling hostname validation also disables SNI.");
  }

  ciphers = SSL_CONN_CONFIG(cipher_list);
  if(ciphers) {
    err = sectransp_set_selected_ciphers(data, backend->ssl_ctx, ciphers);
  }
  else {
    err = sectransp_set_default_ciphers(data, backend->ssl_ctx);
  }
  if(err != noErr) {
    failf(data, "SSL: Unable to set ciphers for SSL/TLS handshake. "
          "Error code: %d", err);
    return CURLE_SSL_CIPHER;
  }

#if CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7
  /* We want to enable 1/n-1 when using a CBC cipher unless the user
     specifically doesn't want us doing that: */
  if(SSLSetSessionOption != NULL) {
    SSLSetSessionOption(backend->ssl_ctx, kSSLSessionOptionSendOneByteRecord,
                        !SSL_SET_OPTION(enable_beast));
    SSLSetSessionOption(backend->ssl_ctx, kSSLSessionOptionFalseStart,
                      data->set.ssl.falsestart); /* false start support */
  }
#endif /* CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7 */

  /* Check if there's a cached ID we can/should use here! */
  if(SSL_SET_OPTION(primary.sessionid)) {
    char *ssl_sessionid;
    size_t ssl_sessionid_len;

    Curl_ssl_sessionid_lock(data);
    if(!Curl_ssl_getsessionid(data, conn, isproxy, (void **)&ssl_sessionid,
                              &ssl_sessionid_len, sockindex)) {
      /* we got a session id, use it! */
      err = SSLSetPeerID(backend->ssl_ctx, ssl_sessionid, ssl_sessionid_len);
      Curl_ssl_sessionid_unlock(data);
      if(err != noErr) {
        failf(data, "SSL: SSLSetPeerID() failed: OSStatus %d", err);
        return CURLE_SSL_CONNECT_ERROR;
      }
      /* Informational message */
      infof(data, "SSL re-using session ID");
    }
    /* If there isn't one, then let's make one up! This has to be done prior
       to starting the handshake. */
    else {
      CURLcode result;
      ssl_sessionid =
        aprintf("%s:%d:%d:%s:%ld",
                ssl_cafile ? ssl_cafile : "(blob memory)",
                verifypeer, SSL_CONN_CONFIG(verifyhost), hostname, port);
      ssl_sessionid_len = strlen(ssl_sessionid);

      err = SSLSetPeerID(backend->ssl_ctx, ssl_sessionid, ssl_sessionid_len);
      if(err != noErr) {
        Curl_ssl_sessionid_unlock(data);
        failf(data, "SSL: SSLSetPeerID() failed: OSStatus %d", err);
        return CURLE_SSL_CONNECT_ERROR;
      }

      result = Curl_ssl_addsessionid(data, conn, isproxy, ssl_sessionid,
                                     ssl_sessionid_len, sockindex, NULL);
      Curl_ssl_sessionid_unlock(data);
      if(result) {
        failf(data, "failed to store ssl session");
        return result;
      }
    }
  }

  err = SSLSetIOFuncs(backend->ssl_ctx, SocketRead, SocketWrite);
  if(err != noErr) {
    failf(data, "SSL: SSLSetIOFuncs() failed: OSStatus %d", err);
    return CURLE_SSL_CONNECT_ERROR;
  }

  /* pass the raw socket into the SSL layers */
  /* We need to store the FD in a constant memory address, because
   * SSLSetConnection() will not copy that address. I've found that
   * conn->sock[sockindex] may change on its own. */
  backend->ssl_sockfd = sockfd;
  err = SSLSetConnection(backend->ssl_ctx, connssl);
  if(err != noErr) {
    failf(data, "SSL: SSLSetConnection() failed: %d", err);
    return CURLE_SSL_CONNECT_ERROR;
  }

  connssl->connecting_state = ssl_connect_2;
  return CURLE_OK;
}

static long pem_to_der(const char *in, unsigned char **out, size_t *outlen)
{
  char *sep_start, *sep_end, *cert_start, *cert_end;
  size_t i, j, err;
  size_t len;
  unsigned char *b64;

  /* Jump through the separators at the beginning of the certificate. */
  sep_start = strstr(in, "-----");
  if(!sep_start)
    return 0;
  cert_start = strstr(sep_start + 1, "-----");
  if(!cert_start)
    return -1;

  cert_start += 5;

  /* Find separator after the end of the certificate. */
  cert_end = strstr(cert_start, "-----");
  if(!cert_end)
    return -1;

  sep_end = strstr(cert_end + 1, "-----");
  if(!sep_end)
    return -1;
  sep_end += 5;

  len = cert_end - cert_start;
  b64 = malloc(len + 1);
  if(!b64)
    return -1;

  /* Create base64 string without linefeeds. */
  for(i = 0, j = 0; i < len; i++) {
    if(cert_start[i] != '\r' && cert_start[i] != '\n')
      b64[j++] = cert_start[i];
  }
  b64[j] = '\0';

  err = Curl_base64_decode((const char *)b64, out, outlen);
  free(b64);
  if(err) {
    free(*out);
    return -1;
  }

  return sep_end - in;
}

static int read_cert(const char *file, unsigned char **out, size_t *outlen)
{
  int fd;
  ssize_t n, len = 0, cap = 512;
  unsigned char buf[512], *data;

  fd = open(file, 0);
  if(fd < 0)
    return -1;

  data = malloc(cap);
  if(!data) {
    close(fd);
    return -1;
  }

  for(;;) {
    n = read(fd, buf, sizeof(buf));
    if(n < 0) {
      close(fd);
      free(data);
      return -1;
    }
    else if(n == 0) {
      close(fd);
      break;
    }

    if(len + n >= cap) {
      cap *= 2;
      data = Curl_saferealloc(data, cap);
      if(!data) {
        close(fd);
        return -1;
      }
    }

    memcpy(data + len, buf, n);
    len += n;
  }
  data[len] = '\0';

  *out = data;
  *outlen = len;

  return 0;
}

static int append_cert_to_array(struct Curl_easy *data,
                                const unsigned char *buf, size_t buflen,
                                CFMutableArrayRef array)
{
    CFDataRef certdata = CFDataCreate(kCFAllocatorDefault, buf, buflen);
    char *certp;
    CURLcode result;
    if(!certdata) {
      failf(data, "SSL: failed to allocate array for CA certificate");
      return CURLE_OUT_OF_MEMORY;
    }

    SecCertificateRef cacert =
      SecCertificateCreateWithData(kCFAllocatorDefault, certdata);
    CFRelease(certdata);
    if(!cacert) {
      failf(data, "SSL: failed to create SecCertificate from CA certificate");
      return CURLE_SSL_CACERT_BADFILE;
    }

    /* Check if cacert is valid. */
    result = CopyCertSubject(data, cacert, &certp);
    switch(result) {
      case CURLE_OK:
        break;
      case CURLE_PEER_FAILED_VERIFICATION:
        return CURLE_SSL_CACERT_BADFILE;
      case CURLE_OUT_OF_MEMORY:
      default:
        return result;
    }
    free(certp);

    CFArrayAppendValue(array, cacert);
    CFRelease(cacert);

    return CURLE_OK;
}

static CURLcode verify_cert_buf(struct Curl_easy *data,
                                const unsigned char *certbuf, size_t buflen,
                                SSLContextRef ctx)
{
  int n = 0, rc;
  long res;
  unsigned char *der;
  size_t derlen, offset = 0;

  /*
   * Certbuf now contains the contents of the certificate file, which can be
   * - a single DER certificate,
   * - a single PEM certificate or
   * - a bunch of PEM certificates (certificate bundle).
   *
   * Go through certbuf, and convert any PEM certificate in it into DER
   * format.
   */
  CFMutableArrayRef array = CFArrayCreateMutable(kCFAllocatorDefault, 0,
                                                 &kCFTypeArrayCallBacks);
  if(!array) {
    failf(data, "SSL: out of memory creating CA certificate array");
    return CURLE_OUT_OF_MEMORY;
  }

  while(offset < buflen) {
    n++;

    /*
     * Check if the certificate is in PEM format, and convert it to DER. If
     * this fails, we assume the certificate is in DER format.
     */
    res = pem_to_der((const char *)certbuf + offset, &der, &derlen);
    if(res < 0) {
      CFRelease(array);
      failf(data, "SSL: invalid CA certificate #%d (offset %zu) in bundle",
            n, offset);
      return CURLE_SSL_CACERT_BADFILE;
    }
    offset += res;

    if(res == 0 && offset == 0) {
      /* This is not a PEM file, probably a certificate in DER format. */
      rc = append_cert_to_array(data, certbuf, buflen, array);
      if(rc != CURLE_OK) {
        CFRelease(array);
        return rc;
      }
      break;
    }
    else if(res == 0) {
      /* No more certificates in the bundle. */
      break;
    }

    rc = append_cert_to_array(data, der, derlen, array);
    free(der);
    if(rc != CURLE_OK) {
      CFRelease(array);
      return rc;
    }
  }

  SecTrustRef trust;
  OSStatus ret = SSLCopyPeerTrust(ctx, &trust);
  if(!trust) {
    failf(data, "SSL: error getting certificate chain");
    CFRelease(array);
    return CURLE_PEER_FAILED_VERIFICATION;
  }
  else if(ret != noErr) {
    CFRelease(array);
    failf(data, "SSLCopyPeerTrust() returned error %d", ret);
    return CURLE_PEER_FAILED_VERIFICATION;
  }

  ret = SecTrustSetAnchorCertificates(trust, array);
  if(ret != noErr) {
    CFRelease(array);
    CFRelease(trust);
    failf(data, "SecTrustSetAnchorCertificates() returned error %d", ret);
    return CURLE_PEER_FAILED_VERIFICATION;
  }
  ret = SecTrustSetAnchorCertificatesOnly(trust, true);
  if(ret != noErr) {
    CFRelease(array);
    CFRelease(trust);
    failf(data, "SecTrustSetAnchorCertificatesOnly() returned error %d", ret);
    return CURLE_PEER_FAILED_VERIFICATION;
  }

  SecTrustResultType trust_eval = 0;
  ret = SecTrustEvaluate(trust, &trust_eval);
  CFRelease(array);
  CFRelease(trust);
  if(ret != noErr) {
    failf(data, "SecTrustEvaluate() returned error %d", ret);
    return CURLE_PEER_FAILED_VERIFICATION;
  }

  switch(trust_eval) {
    case kSecTrustResultUnspecified:
    case kSecTrustResultProceed:
      return CURLE_OK;

    case kSecTrustResultRecoverableTrustFailure:
    case kSecTrustResultDeny:
    default:
      failf(data, "SSL: certificate verification failed (result: %d)",
            trust_eval);
      return CURLE_PEER_FAILED_VERIFICATION;
  }
}

static CURLcode verify_cert(struct Curl_easy *data, const char *cafile,
                            const struct curl_blob *ca_info_blob,
                            SSLContextRef ctx)
{
  int result;
  unsigned char *certbuf;
  size_t buflen;

  if(ca_info_blob) {
    certbuf = (unsigned char *)malloc(ca_info_blob->len + 1);
    if(!certbuf) {
      return CURLE_OUT_OF_MEMORY;
    }
    buflen = ca_info_blob->len;
    memcpy(certbuf, ca_info_blob->data, ca_info_blob->len);
    certbuf[ca_info_blob->len]='\0';
  }
  else if(cafile) {
    if(read_cert(cafile, &certbuf, &buflen) < 0) {
      failf(data, "SSL: failed to read or invalid CA certificate");
      return CURLE_SSL_CACERT_BADFILE;
    }
  }
  else
    return CURLE_SSL_CACERT_BADFILE;

  result = verify_cert_buf(data, certbuf, buflen, ctx);
  free(certbuf);
  return result;
}


#ifdef SECTRANSP_PINNEDPUBKEY
static CURLcode pkp_pin_peer_pubkey(struct Curl_easy *data,
                                    SSLContextRef ctx,
                                    const char *pinnedpubkey)
{  /* Scratch */
  size_t pubkeylen, realpubkeylen, spkiHeaderLength = 24;
  unsigned char *pubkey = NULL, *realpubkey = NULL;
  const unsigned char *spkiHeader = NULL;
  CFDataRef publicKeyBits = NULL;

  /* Result is returned to caller */
  CURLcode result = CURLE_SSL_PINNEDPUBKEYNOTMATCH;

  /* if a path wasn't specified, don't pin */
  if(!pinnedpubkey)
    return CURLE_OK;


  if(!ctx)
    return result;

  do {
    SecTrustRef trust;
    OSStatus ret = SSLCopyPeerTrust(ctx, &trust);
    if(ret != noErr || !trust)
      break;

    SecKeyRef keyRef = SecTrustCopyPublicKey(trust);
    CFRelease(trust);
    if(!keyRef)
      break;

#ifdef SECTRANSP_PINNEDPUBKEY_V1

    publicKeyBits = SecKeyCopyExternalRepresentation(keyRef, NULL);
    CFRelease(keyRef);
    if(!publicKeyBits)
      break;

#elif SECTRANSP_PINNEDPUBKEY_V2

    OSStatus success = SecItemExport(keyRef, kSecFormatOpenSSL, 0, NULL,
                                     &publicKeyBits);
    CFRelease(keyRef);
    if(success != errSecSuccess || !publicKeyBits)
      break;

#endif /* SECTRANSP_PINNEDPUBKEY_V2 */

    pubkeylen = CFDataGetLength(publicKeyBits);
    pubkey = (unsigned char *)CFDataGetBytePtr(publicKeyBits);

    switch(pubkeylen) {
      case 526:
        /* 4096 bit RSA pubkeylen == 526 */
        spkiHeader = rsa4096SpkiHeader;
        break;
      case 270:
        /* 2048 bit RSA pubkeylen == 270 */
        spkiHeader = rsa2048SpkiHeader;
        break;
#ifdef SECTRANSP_PINNEDPUBKEY_V1
      case 65:
        /* ecDSA secp256r1 pubkeylen == 65 */
        spkiHeader = ecDsaSecp256r1SpkiHeader;
        spkiHeaderLength = 26;
        break;
      case 97:
        /* ecDSA secp384r1 pubkeylen == 97 */
        spkiHeader = ecDsaSecp384r1SpkiHeader;
        spkiHeaderLength = 23;
        break;
      default:
        infof(data, "SSL: unhandled public key length: %d", pubkeylen);
#elif SECTRANSP_PINNEDPUBKEY_V2
      default:
        /* ecDSA secp256r1 pubkeylen == 91 header already included?
         * ecDSA secp384r1 header already included too
         * we assume rest of algorithms do same, so do nothing
         */
        result = Curl_pin_peer_pubkey(data, pinnedpubkey, pubkey,
                                    pubkeylen);
#endif /* SECTRANSP_PINNEDPUBKEY_V2 */
        continue; /* break from loop */
    }

    realpubkeylen = pubkeylen + spkiHeaderLength;
    realpubkey = malloc(realpubkeylen);
    if(!realpubkey)
      break;

    memcpy(realpubkey, spkiHeader, spkiHeaderLength);
    memcpy(realpubkey + spkiHeaderLength, pubkey, pubkeylen);

    result = Curl_pin_peer_pubkey(data, pinnedpubkey, realpubkey,
                                  realpubkeylen);

  } while(0);

  Curl_safefree(realpubkey);
  if(publicKeyBits != NULL)
    CFRelease(publicKeyBits);

  return result;
}
#endif /* SECTRANSP_PINNEDPUBKEY */

static CURLcode
sectransp_connect_step2(struct Curl_easy *data, struct connectdata *conn,
                        int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  OSStatus err;
  SSLCipherSuite cipher;
  SSLProtocol protocol = 0;
  const char * const hostname = SSL_HOST_NAME();

  DEBUGASSERT(ssl_connect_2 == connssl->connecting_state
              || ssl_connect_2_reading == connssl->connecting_state
              || ssl_connect_2_writing == connssl->connecting_state);

  /* Here goes nothing: */
  err = SSLHandshake(backend->ssl_ctx);

  if(err != noErr) {
    switch(err) {
      case errSSLWouldBlock:  /* they're not done with us yet */
        connssl->connecting_state = backend->ssl_direction ?
            ssl_connect_2_writing : ssl_connect_2_reading;
        return CURLE_OK;

      /* The below is errSSLServerAuthCompleted; it's not defined in
        Leopard's headers */
      case -9841:
        if((SSL_CONN_CONFIG(CAfile) || SSL_CONN_CONFIG(ca_info_blob)) &&
           SSL_CONN_CONFIG(verifypeer)) {
          CURLcode result = verify_cert(data, SSL_CONN_CONFIG(CAfile),
                                        SSL_CONN_CONFIG(ca_info_blob),
                                        backend->ssl_ctx);
          if(result)
            return result;
        }
        /* the documentation says we need to call SSLHandshake() again */
        return sectransp_connect_step2(data, conn, sockindex);

      /* Problem with encrypt / decrypt */
      case errSSLPeerDecodeError:
        failf(data, "Decode failed");
        break;
      case errSSLDecryptionFail:
      case errSSLPeerDecryptionFail:
        failf(data, "Decryption failed");
        break;
      case errSSLPeerDecryptError:
        failf(data, "A decryption error occurred");
        break;
      case errSSLBadCipherSuite:
        failf(data, "A bad SSL cipher suite was encountered");
        break;
      case errSSLCrypto:
        failf(data, "An underlying cryptographic error was encountered");
        break;
#if CURL_BUILD_MAC_10_11 || CURL_BUILD_IOS_9
      case errSSLWeakPeerEphemeralDHKey:
        failf(data, "Indicates a weak ephemeral Diffie-Hellman key");
        break;
#endif

      /* Problem with the message record validation */
      case errSSLBadRecordMac:
      case errSSLPeerBadRecordMac:
        failf(data, "A record with a bad message authentication code (MAC) "
                    "was encountered");
        break;
      case errSSLRecordOverflow:
      case errSSLPeerRecordOverflow:
        failf(data, "A record overflow occurred");
        break;

      /* Problem with zlib decompression */
      case errSSLPeerDecompressFail:
        failf(data, "Decompression failed");
        break;

      /* Problem with access */
      case errSSLPeerAccessDenied:
        failf(data, "Access was denied");
        break;
      case errSSLPeerInsufficientSecurity:
        failf(data, "There is insufficient security for this operation");
        break;

      /* These are all certificate problems with the server: */
      case errSSLXCertChainInvalid:
        failf(data, "SSL certificate problem: Invalid certificate chain");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLUnknownRootCert:
        failf(data, "SSL certificate problem: Untrusted root certificate");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLNoRootCert:
        failf(data, "SSL certificate problem: No root certificate");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLCertNotYetValid:
        failf(data, "SSL certificate problem: The certificate chain had a "
                    "certificate that is not yet valid");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLCertExpired:
      case errSSLPeerCertExpired:
        failf(data, "SSL certificate problem: Certificate chain had an "
              "expired certificate");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLBadCert:
      case errSSLPeerBadCert:
        failf(data, "SSL certificate problem: Couldn't understand the server "
              "certificate format");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLPeerUnsupportedCert:
        failf(data, "SSL certificate problem: An unsupported certificate "
                    "format was encountered");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLPeerCertRevoked:
        failf(data, "SSL certificate problem: The certificate was revoked");
        return CURLE_PEER_FAILED_VERIFICATION;
      case errSSLPeerCertUnknown:
        failf(data, "SSL certificate problem: The certificate is unknown");
        return CURLE_PEER_FAILED_VERIFICATION;

      /* These are all certificate problems with the client: */
      case errSecAuthFailed:
        failf(data, "SSL authentication failed");
        break;
      case errSSLPeerHandshakeFail:
        failf(data, "SSL peer handshake failed, the server most likely "
              "requires a client certificate to connect");
        break;
      case errSSLPeerUnknownCA:
        failf(data, "SSL server rejected the client certificate due to "
              "the certificate being signed by an unknown certificate "
              "authority");
        break;

      /* This error is raised if the server's cert didn't match the server's
         host name: */
      case errSSLHostNameMismatch:
        failf(data, "SSL certificate peer verification failed, the "
              "certificate did not match \"%s\"\n", conn->host.dispname);
        return CURLE_PEER_FAILED_VERIFICATION;

      /* Problem with SSL / TLS negotiation */
      case errSSLNegotiation:
        failf(data, "Could not negotiate an SSL cipher suite with the server");
        break;
      case errSSLBadConfiguration:
        failf(data, "A configuration error occurred");
        break;
      case errSSLProtocol:
        failf(data, "SSL protocol error");
        break;
      case errSSLPeerProtocolVersion:
        failf(data, "A bad protocol version was encountered");
        break;
      case errSSLPeerNoRenegotiation:
        failf(data, "No renegotiation is allowed");
        break;

      /* Generic handshake errors: */
      case errSSLConnectionRefused:
        failf(data, "Server dropped the connection during the SSL handshake");
        break;
      case errSSLClosedAbort:
        failf(data, "Server aborted the SSL handshake");
        break;
      case errSSLClosedGraceful:
        failf(data, "The connection closed gracefully");
        break;
      case errSSLClosedNoNotify:
        failf(data, "The server closed the session with no notification");
        break;
      /* Sometimes paramErr happens with buggy ciphers: */
      case paramErr:
      case errSSLInternal:
      case errSSLPeerInternalError:
        failf(data, "Internal SSL engine error encountered during the "
              "SSL handshake");
        break;
      case errSSLFatalAlert:
        failf(data, "Fatal SSL engine error encountered during the SSL "
              "handshake");
        break;
      /* Unclassified error */
      case errSSLBufferOverflow:
        failf(data, "An insufficient buffer was provided");
        break;
      case errSSLIllegalParam:
        failf(data, "An illegal parameter was encountered");
        break;
      case errSSLModuleAttach:
        failf(data, "Module attach failure");
        break;
      case errSSLSessionNotFound:
        failf(data, "An attempt to restore an unknown session failed");
        break;
      case errSSLPeerExportRestriction:
        failf(data, "An export restriction occurred");
        break;
      case errSSLPeerUserCancelled:
        failf(data, "The user canceled the operation");
        break;
      case errSSLPeerUnexpectedMsg:
        failf(data, "Peer rejected unexpected message");
        break;
#if CURL_BUILD_MAC_10_11 || CURL_BUILD_IOS_9
      /* Treaing non-fatal error as fatal like before */
      case errSSLClientHelloReceived:
        failf(data, "A non-fatal result for providing a server name "
                    "indication");
        break;
#endif

      /* Error codes defined in the enum but should never be returned.
         We list them here just in case. */
#if CURL_BUILD_MAC_10_6
      /* Only returned when kSSLSessionOptionBreakOnCertRequested is set */
      case errSSLClientCertRequested:
        failf(data, "Server requested a client certificate during the "
              "handshake");
        return CURLE_SSL_CLIENTCERT;
#endif
#if CURL_BUILD_MAC_10_9
      /* Alias for errSSLLast, end of error range */
      case errSSLUnexpectedRecord:
        failf(data, "Unexpected (skipped) record in DTLS");
        break;
#endif
      default:
        /* May also return codes listed in Security Framework Result Codes */
        failf(data, "Unknown SSL protocol error in connection to %s:%d",
              hostname, err);
        break;
    }
    return CURLE_SSL_CONNECT_ERROR;
  }
  else {
    /* we have been connected fine, we're not waiting for anything else. */
    connssl->connecting_state = ssl_connect_3;

#ifdef SECTRANSP_PINNEDPUBKEY
    if(data->set.str[STRING_SSL_PINNEDPUBLICKEY]) {
      CURLcode result =
        pkp_pin_peer_pubkey(data, backend->ssl_ctx,
                            data->set.str[STRING_SSL_PINNEDPUBLICKEY]);
      if(result) {
        failf(data, "SSL: public key does not match pinned public key!");
        return result;
      }
    }
#endif /* SECTRANSP_PINNEDPUBKEY */

    /* Informational message */
    (void)SSLGetNegotiatedCipher(backend->ssl_ctx, &cipher);
    (void)SSLGetNegotiatedProtocolVersion(backend->ssl_ctx, &protocol);
    switch(protocol) {
      case kSSLProtocol2:
        infof(data, "SSL 2.0 connection using %s",
              TLSCipherNameForNumber(cipher));
        break;
      case kSSLProtocol3:
        infof(data, "SSL 3.0 connection using %s",
              TLSCipherNameForNumber(cipher));
        break;
      case kTLSProtocol1:
        infof(data, "TLS 1.0 connection using %s",
              TLSCipherNameForNumber(cipher));
        break;
#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
      case kTLSProtocol11:
        infof(data, "TLS 1.1 connection using %s",
              TLSCipherNameForNumber(cipher));
        break;
      case kTLSProtocol12:
        infof(data, "TLS 1.2 connection using %s",
              TLSCipherNameForNumber(cipher));
        break;
#endif /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */
#if CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11
      case kTLSProtocol13:
        infof(data, "TLS 1.3 connection using %s",
              TLSCipherNameForNumber(cipher));
        break;
#endif /* CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11 */
      default:
        infof(data, "Unknown protocol connection");
        break;
    }

#if(CURL_BUILD_MAC_10_13 || CURL_BUILD_IOS_11) && HAVE_BUILTIN_AVAILABLE == 1
    if(conn->bits.tls_enable_alpn) {
      if(__builtin_available(macOS 10.13.4, iOS 11, tvOS 11, *)) {
        CFArrayRef alpnArr = NULL;
        CFStringRef chosenProtocol = NULL;
        err = SSLCopyALPNProtocols(backend->ssl_ctx, &alpnArr);

        if(err == noErr && alpnArr && CFArrayGetCount(alpnArr) >= 1)
          chosenProtocol = CFArrayGetValueAtIndex(alpnArr, 0);

#ifdef USE_HTTP2
        if(chosenProtocol &&
           !CFStringCompare(chosenProtocol, CFSTR(ALPN_H2), 0)) {
          conn->negnpn = CURL_HTTP_VERSION_2;
        }
        else
#endif
        if(chosenProtocol &&
           !CFStringCompare(chosenProtocol, CFSTR(ALPN_HTTP_1_1), 0)) {
          conn->negnpn = CURL_HTTP_VERSION_1_1;
        }
        else
          infof(data, "ALPN, server did not agree to a protocol");

        Curl_multiuse_state(data, conn->negnpn == CURL_HTTP_VERSION_2 ?
                            BUNDLE_MULTIPLEX : BUNDLE_NO_MULTIUSE);

        /* chosenProtocol is a reference to the string within alpnArr
           and doesn't need to be freed separately */
        if(alpnArr)
          CFRelease(alpnArr);
      }
    }
#endif

    return CURLE_OK;
  }
}

static CURLcode
add_cert_to_certinfo(struct Curl_easy *data,
                     SecCertificateRef server_cert,
                     int idx)
{
  CURLcode result = CURLE_OK;
  const char *beg;
  const char *end;
  CFDataRef cert_data = SecCertificateCopyData(server_cert);

  if(!cert_data)
    return CURLE_PEER_FAILED_VERIFICATION;

  beg = (const char *)CFDataGetBytePtr(cert_data);
  end = beg + CFDataGetLength(cert_data);
  result = Curl_extract_certinfo(data, idx, beg, end);
  CFRelease(cert_data);
  return result;
}

static CURLcode
collect_server_cert_single(struct Curl_easy *data,
                           SecCertificateRef server_cert,
                           CFIndex idx)
{
  CURLcode result = CURLE_OK;
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  if(data->set.verbose) {
    char *certp;
    result = CopyCertSubject(data, server_cert, &certp);
    if(!result) {
      infof(data, "Server certificate: %s", certp);
      free(certp);
    }
  }
#endif
  if(data->set.ssl.certinfo)
    result = add_cert_to_certinfo(data, server_cert, (int)idx);
  return result;
}

/* This should be called during step3 of the connection at the earliest */
static CURLcode
collect_server_cert(struct Curl_easy *data,
                    struct connectdata *conn,
                    int sockindex)
{
#ifndef CURL_DISABLE_VERBOSE_STRINGS
  const bool show_verbose_server_cert = data->set.verbose;
#else
  const bool show_verbose_server_cert = false;
#endif
  CURLcode result = data->set.ssl.certinfo ?
    CURLE_PEER_FAILED_VERIFICATION : CURLE_OK;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  CFArrayRef server_certs = NULL;
  SecCertificateRef server_cert;
  OSStatus err;
  CFIndex i, count;
  SecTrustRef trust = NULL;

  if(!show_verbose_server_cert && !data->set.ssl.certinfo)
    return CURLE_OK;

  if(!backend->ssl_ctx)
    return result;

#if CURL_BUILD_MAC_10_7 || CURL_BUILD_IOS
#if CURL_BUILD_IOS
#pragma unused(server_certs)
  err = SSLCopyPeerTrust(backend->ssl_ctx, &trust);
  /* For some reason, SSLCopyPeerTrust() can return noErr and yet return
     a null trust, so be on guard for that: */
  if(err == noErr && trust) {
    count = SecTrustGetCertificateCount(trust);
    if(data->set.ssl.certinfo)
      result = Curl_ssl_init_certinfo(data, (int)count);
    for(i = 0L ; !result && (i < count) ; i++) {
      server_cert = SecTrustGetCertificateAtIndex(trust, i);
      result = collect_server_cert_single(data, server_cert, i);
    }
    CFRelease(trust);
  }
#else
  /* SSLCopyPeerCertificates() is deprecated as of Mountain Lion.
     The function SecTrustGetCertificateAtIndex() is officially present
     in Lion, but it is unfortunately also present in Snow Leopard as
     private API and doesn't work as expected. So we have to look for
     a different symbol to make sure this code is only executed under
     Lion or later. */
  if(SecTrustEvaluateAsync != NULL) {
#pragma unused(server_certs)
    err = SSLCopyPeerTrust(backend->ssl_ctx, &trust);
    /* For some reason, SSLCopyPeerTrust() can return noErr and yet return
       a null trust, so be on guard for that: */
    if(err == noErr && trust) {
      count = SecTrustGetCertificateCount(trust);
      if(data->set.ssl.certinfo)
        result = Curl_ssl_init_certinfo(data, (int)count);
      for(i = 0L ; !result && (i < count) ; i++) {
        server_cert = SecTrustGetCertificateAtIndex(trust, i);
        result = collect_server_cert_single(data, server_cert, i);
      }
      CFRelease(trust);
    }
  }
  else {
#if CURL_SUPPORT_MAC_10_8
    err = SSLCopyPeerCertificates(backend->ssl_ctx, &server_certs);
    /* Just in case SSLCopyPeerCertificates() returns null too... */
    if(err == noErr && server_certs) {
      count = CFArrayGetCount(server_certs);
      if(data->set.ssl.certinfo)
        result = Curl_ssl_init_certinfo(data, (int)count);
      for(i = 0L ; !result && (i < count) ; i++) {
        server_cert = (SecCertificateRef)CFArrayGetValueAtIndex(server_certs,
                                                                i);
        result = collect_server_cert_single(data, server_cert, i);
      }
      CFRelease(server_certs);
    }
#endif /* CURL_SUPPORT_MAC_10_8 */
  }
#endif /* CURL_BUILD_IOS */
#else
#pragma unused(trust)
  err = SSLCopyPeerCertificates(backend->ssl_ctx, &server_certs);
  if(err == noErr) {
    count = CFArrayGetCount(server_certs);
    if(data->set.ssl.certinfo)
      result = Curl_ssl_init_certinfo(data, (int)count);
    for(i = 0L ; !result && (i < count) ; i++) {
      server_cert = (SecCertificateRef)CFArrayGetValueAtIndex(server_certs, i);
      result = collect_server_cert_single(data, server_cert, i);
    }
    CFRelease(server_certs);
  }
#endif /* CURL_BUILD_MAC_10_7 || CURL_BUILD_IOS */
  return result;
}

static CURLcode
sectransp_connect_step3(struct Curl_easy *data, struct connectdata *conn,
                        int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];

  /* There is no step 3!
   * Well, okay, let's collect server certificates, and if verbose mode is on,
   * let's print the details of the server certificates. */
  const CURLcode result = collect_server_cert(data, conn, sockindex);
  if(result)
    return result;

  connssl->connecting_state = ssl_connect_done;
  return CURLE_OK;
}

static Curl_recv sectransp_recv;
static Curl_send sectransp_send;

static CURLcode
sectransp_connect_common(struct Curl_easy *data,
                         struct connectdata *conn,
                         int sockindex,
                         bool nonblocking,
                         bool *done)
{
  CURLcode result;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  curl_socket_t sockfd = conn->sock[sockindex];
  int what;

  /* check if the connection has already been established */
  if(ssl_connection_complete == connssl->state) {
    *done = TRUE;
    return CURLE_OK;
  }

  if(ssl_connect_1 == connssl->connecting_state) {
    /* Find out how much more time we're allowed */
    const timediff_t timeout_ms = Curl_timeleft(data, NULL, TRUE);

    if(timeout_ms < 0) {
      /* no need to continue if time already is up */
      failf(data, "SSL connection timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    result = sectransp_connect_step1(data, conn, sockindex);
    if(result)
      return result;
  }

  while(ssl_connect_2 == connssl->connecting_state ||
        ssl_connect_2_reading == connssl->connecting_state ||
        ssl_connect_2_writing == connssl->connecting_state) {

    /* check allowed time left */
    const timediff_t timeout_ms = Curl_timeleft(data, NULL, TRUE);

    if(timeout_ms < 0) {
      /* no need to continue if time already is up */
      failf(data, "SSL connection timeout");
      return CURLE_OPERATION_TIMEDOUT;
    }

    /* if ssl is expecting something, check if it's available. */
    if(connssl->connecting_state == ssl_connect_2_reading ||
       connssl->connecting_state == ssl_connect_2_writing) {

      curl_socket_t writefd = ssl_connect_2_writing ==
      connssl->connecting_state?sockfd:CURL_SOCKET_BAD;
      curl_socket_t readfd = ssl_connect_2_reading ==
      connssl->connecting_state?sockfd:CURL_SOCKET_BAD;

      what = Curl_socket_check(readfd, CURL_SOCKET_BAD, writefd,
                               nonblocking ? 0 : timeout_ms);
      if(what < 0) {
        /* fatal error */
        failf(data, "select/poll on SSL socket, errno: %d", SOCKERRNO);
        return CURLE_SSL_CONNECT_ERROR;
      }
      else if(0 == what) {
        if(nonblocking) {
          *done = FALSE;
          return CURLE_OK;
        }
        else {
          /* timeout */
          failf(data, "SSL connection timeout");
          return CURLE_OPERATION_TIMEDOUT;
        }
      }
      /* socket is readable or writable */
    }

    /* Run transaction, and return to the caller if it failed or if this
     * connection is done nonblocking and this loop would execute again. This
     * permits the owner of a multi handle to abort a connection attempt
     * before step2 has completed while ensuring that a client using select()
     * or epoll() will always have a valid fdset to wait on.
     */
    result = sectransp_connect_step2(data, conn, sockindex);
    if(result || (nonblocking &&
                  (ssl_connect_2 == connssl->connecting_state ||
                   ssl_connect_2_reading == connssl->connecting_state ||
                   ssl_connect_2_writing == connssl->connecting_state)))
      return result;

  } /* repeat step2 until all transactions are done. */


  if(ssl_connect_3 == connssl->connecting_state) {
    result = sectransp_connect_step3(data, conn, sockindex);
    if(result)
      return result;
  }

  if(ssl_connect_done == connssl->connecting_state) {
    connssl->state = ssl_connection_complete;
    conn->recv[sockindex] = sectransp_recv;
    conn->send[sockindex] = sectransp_send;
    *done = TRUE;
  }
  else
    *done = FALSE;

  /* Reset our connect state machine */
  connssl->connecting_state = ssl_connect_1;

  return CURLE_OK;
}

static CURLcode sectransp_connect_nonblocking(struct Curl_easy *data,
                                              struct connectdata *conn,
                                              int sockindex, bool *done)
{
  return sectransp_connect_common(data, conn, sockindex, TRUE, done);
}

static CURLcode sectransp_connect(struct Curl_easy *data,
                                  struct connectdata *conn, int sockindex)
{
  CURLcode result;
  bool done = FALSE;

  result = sectransp_connect_common(data, conn, sockindex, FALSE, &done);

  if(result)
    return result;

  DEBUGASSERT(done);

  return CURLE_OK;
}

static void sectransp_close(struct Curl_easy *data, struct connectdata *conn,
                            int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;

  (void) data;

  if(backend->ssl_ctx) {
    (void)SSLClose(backend->ssl_ctx);
#if CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS
    if(SSLCreateContext != NULL)
      CFRelease(backend->ssl_ctx);
#if CURL_SUPPORT_MAC_10_8
    else
      (void)SSLDisposeContext(backend->ssl_ctx);
#endif  /* CURL_SUPPORT_MAC_10_8 */
#else
    (void)SSLDisposeContext(backend->ssl_ctx);
#endif /* CURL_BUILD_MAC_10_8 || CURL_BUILD_IOS */
    backend->ssl_ctx = NULL;
  }
  backend->ssl_sockfd = 0;
}

static int sectransp_shutdown(struct Curl_easy *data,
                              struct connectdata *conn, int sockindex)
{
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  ssize_t nread;
  int what;
  int rc;
  char buf[120];
  int loop = 10; /* avoid getting stuck */

  if(!backend->ssl_ctx)
    return 0;

#ifndef CURL_DISABLE_FTP
  if(data->set.ftp_ccc != CURLFTPSSL_CCC_ACTIVE)
    return 0;
#endif

  sectransp_close(data, conn, sockindex);

  rc = 0;

  what = SOCKET_READABLE(conn->sock[sockindex], SSL_SHUTDOWN_TIMEOUT);

  while(loop--) {
    if(what < 0) {
      /* anything that gets here is fatally bad */
      failf(data, "select/poll on SSL socket, errno: %d", SOCKERRNO);
      rc = -1;
      break;
    }

    if(!what) {                                /* timeout */
      failf(data, "SSL shutdown timeout");
      break;
    }

    /* Something to read, let's do it and hope that it is the close
     notify alert from the server. No way to SSL_Read now, so use read(). */

    nread = read(conn->sock[sockindex], buf, sizeof(buf));

    if(nread < 0) {
      char buffer[STRERROR_LEN];
      failf(data, "read: %s",
            Curl_strerror(errno, buffer, sizeof(buffer)));
      rc = -1;
    }

    if(nread <= 0)
      break;

    what = SOCKET_READABLE(conn->sock[sockindex], 0);
  }

  return rc;
}

static void sectransp_session_free(void *ptr)
{
  /* ST, as of iOS 5 and Mountain Lion, has no public method of deleting a
     cached session ID inside the Security framework. There is a private
     function that does this, but I don't want to have to explain to you why I
     got your application rejected from the App Store due to the use of a
     private API, so the best we can do is free up our own char array that we
     created way back in sectransp_connect_step1... */
  Curl_safefree(ptr);
}

static size_t sectransp_version(char *buffer, size_t size)
{
  return msnprintf(buffer, size, "SecureTransport");
}

/*
 * This function uses SSLGetSessionState to determine connection status.
 *
 * Return codes:
 *     1 means the connection is still in place
 *     0 means the connection has been closed
 *    -1 means the connection status is unknown
 */
static int sectransp_check_cxn(struct connectdata *conn)
{
  struct ssl_connect_data *connssl = &conn->ssl[FIRSTSOCKET];
  struct ssl_backend_data *backend = connssl->backend;
  OSStatus err;
  SSLSessionState state;

  if(backend->ssl_ctx) {
    err = SSLGetSessionState(backend->ssl_ctx, &state);
    if(err == noErr)
      return state == kSSLConnected || state == kSSLHandshake;
    return -1;
  }
  return 0;
}

static bool sectransp_data_pending(const struct connectdata *conn,
                                   int connindex)
{
  const struct ssl_connect_data *connssl = &conn->ssl[connindex];
  struct ssl_backend_data *backend = connssl->backend;
  OSStatus err;
  size_t buffer;

  if(backend->ssl_ctx) {  /* SSL is in use */
    err = SSLGetBufferedReadSize(backend->ssl_ctx, &buffer);
    if(err == noErr)
      return buffer > 0UL;
    return false;
  }
  else
    return false;
}

static CURLcode sectransp_random(struct Curl_easy *data UNUSED_PARAM,
                                 unsigned char *entropy, size_t length)
{
  /* arc4random_buf() isn't available on cats older than Lion, so let's
     do this manually for the benefit of the older cats. */
  size_t i;
  u_int32_t random_number = 0;

  (void)data;

  for(i = 0 ; i < length ; i++) {
    if(i % sizeof(u_int32_t) == 0)
      random_number = arc4random();
    entropy[i] = random_number & 0xFF;
    random_number >>= 8;
  }
  i = random_number = 0;
  return CURLE_OK;
}

static CURLcode sectransp_sha256sum(const unsigned char *tmp, /* input */
                                    size_t tmplen,
                                    unsigned char *sha256sum, /* output */
                                    size_t sha256len)
{
  assert(sha256len >= CURL_SHA256_DIGEST_LENGTH);
  (void)CC_SHA256(tmp, (CC_LONG)tmplen, sha256sum);
  return CURLE_OK;
}

static bool sectransp_false_start(void)
{
#if CURL_BUILD_MAC_10_9 || CURL_BUILD_IOS_7
  if(SSLSetSessionOption != NULL)
    return TRUE;
#endif
  return FALSE;
}

static ssize_t sectransp_send(struct Curl_easy *data,
                              int sockindex,
                              const void *mem,
                              size_t len,
                              CURLcode *curlcode)
{
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[sockindex];
  struct ssl_backend_data *backend = connssl->backend;
  size_t processed = 0UL;
  OSStatus err;

  /* The SSLWrite() function works a little differently than expected. The
     fourth argument (processed) is currently documented in Apple's
     documentation as: "On return, the length, in bytes, of the data actually
     written."

     Now, one could interpret that as "written to the socket," but actually,
     it returns the amount of data that was written to a buffer internal to
     the SSLContextRef instead. So it's possible for SSLWrite() to return
     errSSLWouldBlock and a number of bytes "written" because those bytes were
     encrypted and written to a buffer, not to the socket.

     So if this happens, then we need to keep calling SSLWrite() over and
     over again with no new data until it quits returning errSSLWouldBlock. */

  /* Do we have buffered data to write from the last time we were called? */
  if(backend->ssl_write_buffered_length) {
    /* Write the buffered data: */
    err = SSLWrite(backend->ssl_ctx, NULL, 0UL, &processed);
    switch(err) {
      case noErr:
        /* processed is always going to be 0 because we didn't write to
           the buffer, so return how much was written to the socket */
        processed = backend->ssl_write_buffered_length;
        backend->ssl_write_buffered_length = 0UL;
        break;
      case errSSLWouldBlock: /* argh, try again */
        *curlcode = CURLE_AGAIN;
        return -1L;
      default:
        failf(data, "SSLWrite() returned error %d", err);
        *curlcode = CURLE_SEND_ERROR;
        return -1L;
    }
  }
  else {
    /* We've got new data to write: */
    err = SSLWrite(backend->ssl_ctx, mem, len, &processed);
    if(err != noErr) {
      switch(err) {
        case errSSLWouldBlock:
          /* Data was buffered but not sent, we have to tell the caller
             to try sending again, and remember how much was buffered */
          backend->ssl_write_buffered_length = len;
          *curlcode = CURLE_AGAIN;
          return -1L;
        default:
          failf(data, "SSLWrite() returned error %d", err);
          *curlcode = CURLE_SEND_ERROR;
          return -1L;
      }
    }
  }
  return (ssize_t)processed;
}

static ssize_t sectransp_recv(struct Curl_easy *data,
                              int num,
                              char *buf,
                              size_t buffersize,
                              CURLcode *curlcode)
{
  struct connectdata *conn = data->conn;
  struct ssl_connect_data *connssl = &conn->ssl[num];
  struct ssl_backend_data *backend = connssl->backend;
  size_t processed = 0UL;
  OSStatus err;

  again:
  err = SSLRead(backend->ssl_ctx, buf, buffersize, &processed);

  if(err != noErr) {
    switch(err) {
      case errSSLWouldBlock:  /* return how much we read (if anything) */
        if(processed)
          return (ssize_t)processed;
        *curlcode = CURLE_AGAIN;
        return -1L;
        break;

      /* errSSLClosedGraceful - server gracefully shut down the SSL session
         errSSLClosedNoNotify - server hung up on us instead of sending a
           closure alert notice, read() is returning 0
         Either way, inform the caller that the server disconnected. */
      case errSSLClosedGraceful:
      case errSSLClosedNoNotify:
        *curlcode = CURLE_OK;
        return -1L;
        break;

        /* The below is errSSLPeerAuthCompleted; it's not defined in
           Leopard's headers */
      case -9841:
        if((SSL_CONN_CONFIG(CAfile) || SSL_CONN_CONFIG(ca_info_blob)) &&
           SSL_CONN_CONFIG(verifypeer)) {
          CURLcode result = verify_cert(data, SSL_CONN_CONFIG(CAfile),
                                        SSL_CONN_CONFIG(ca_info_blob),
                                        backend->ssl_ctx);
          if(result)
            return result;
        }
        goto again;
      default:
        failf(data, "SSLRead() return error %d", err);
        *curlcode = CURLE_RECV_ERROR;
        return -1L;
        break;
    }
  }
  return (ssize_t)processed;
}

static void *sectransp_get_internals(struct ssl_connect_data *connssl,
                                     CURLINFO info UNUSED_PARAM)
{
  struct ssl_backend_data *backend = connssl->backend;
  (void)info;
  return backend->ssl_ctx;
}

const struct Curl_ssl Curl_ssl_sectransp = {
  { CURLSSLBACKEND_SECURETRANSPORT, "secure-transport" }, /* info */

  SSLSUPP_CAINFO_BLOB |
  SSLSUPP_CERTINFO |
#ifdef SECTRANSP_PINNEDPUBKEY
  SSLSUPP_PINNEDPUBKEY,
#else
  0,
#endif /* SECTRANSP_PINNEDPUBKEY */

  sizeof(struct ssl_backend_data),

  Curl_none_init,                     /* init */
  Curl_none_cleanup,                  /* cleanup */
  sectransp_version,                  /* version */
  sectransp_check_cxn,                /* check_cxn */
  sectransp_shutdown,                 /* shutdown */
  sectransp_data_pending,             /* data_pending */
  sectransp_random,                   /* random */
  Curl_none_cert_status_request,      /* cert_status_request */
  sectransp_connect,                  /* connect */
  sectransp_connect_nonblocking,      /* connect_nonblocking */
  Curl_ssl_getsock,                   /* getsock */
  sectransp_get_internals,            /* get_internals */
  sectransp_close,                    /* close_one */
  Curl_none_close_all,                /* close_all */
  sectransp_session_free,             /* session_free */
  Curl_none_set_engine,               /* set_engine */
  Curl_none_set_engine_default,       /* set_engine_default */
  Curl_none_engines_list,             /* engines_list */
  sectransp_false_start,              /* false_start */
  sectransp_sha256sum,                /* sha256sum */
  NULL,                               /* associate_connection */
  NULL                                /* disassociate_connection */
};

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#endif /* USE_SECTRANSP */
