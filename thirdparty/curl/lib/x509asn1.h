#ifndef HEADER_CURL_X509ASN1_H
#define HEADER_CURL_X509ASN1_H

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

#if defined(USE_GSKIT) || defined(USE_NSS) || defined(USE_GNUTLS) || \
    defined(USE_WOLFSSL) || defined(USE_SCHANNEL) || defined(USE_SECTRANSP)

#include "urldata.h"

/*
 * Constants.
 */

/* Largest supported ASN.1 structure. */
#define CURL_ASN1_MAX                   ((size_t) 0x40000)      /* 256K */

/* ASN.1 classes. */
#define CURL_ASN1_UNIVERSAL             0
#define CURL_ASN1_APPLICATION           1
#define CURL_ASN1_CONTEXT_SPECIFIC      2
#define CURL_ASN1_PRIVATE               3

/* ASN.1 types. */
#define CURL_ASN1_BOOLEAN               1
#define CURL_ASN1_INTEGER               2
#define CURL_ASN1_BIT_STRING            3
#define CURL_ASN1_OCTET_STRING          4
#define CURL_ASN1_NULL                  5
#define CURL_ASN1_OBJECT_IDENTIFIER     6
#define CURL_ASN1_OBJECT_DESCRIPTOR     7
#define CURL_ASN1_INSTANCE_OF           8
#define CURL_ASN1_REAL                  9
#define CURL_ASN1_ENUMERATED            10
#define CURL_ASN1_EMBEDDED              11
#define CURL_ASN1_UTF8_STRING           12
#define CURL_ASN1_RELATIVE_OID          13
#define CURL_ASN1_SEQUENCE              16
#define CURL_ASN1_SET                   17
#define CURL_ASN1_NUMERIC_STRING        18
#define CURL_ASN1_PRINTABLE_STRING      19
#define CURL_ASN1_TELETEX_STRING        20
#define CURL_ASN1_VIDEOTEX_STRING       21
#define CURL_ASN1_IA5_STRING            22
#define CURL_ASN1_UTC_TIME              23
#define CURL_ASN1_GENERALIZED_TIME      24
#define CURL_ASN1_GRAPHIC_STRING        25
#define CURL_ASN1_VISIBLE_STRING        26
#define CURL_ASN1_GENERAL_STRING        27
#define CURL_ASN1_UNIVERSAL_STRING      28
#define CURL_ASN1_CHARACTER_STRING      29
#define CURL_ASN1_BMP_STRING            30


/*
 * Types.
 */

/* ASN.1 parsed element. */
struct Curl_asn1Element {
  const char *header;         /* Pointer to header byte. */
  const char *beg;            /* Pointer to element data. */
  const char *end;            /* Pointer to 1st byte after element. */
  unsigned char class;        /* ASN.1 element class. */
  unsigned char tag;          /* ASN.1 element tag. */
  bool          constructed;  /* Element is constructed. */
};


/* ASN.1 OID table entry. */
struct Curl_OID {
  const char *numoid;  /* Dotted-numeric OID. */
  const char *textoid; /* OID name. */
};


/* X509 certificate: RFC 5280. */
struct Curl_X509certificate {
  struct Curl_asn1Element certificate;
  struct Curl_asn1Element version;
  struct Curl_asn1Element serialNumber;
  struct Curl_asn1Element signatureAlgorithm;
  struct Curl_asn1Element signature;
  struct Curl_asn1Element issuer;
  struct Curl_asn1Element notBefore;
  struct Curl_asn1Element notAfter;
  struct Curl_asn1Element subject;
  struct Curl_asn1Element subjectPublicKeyInfo;
  struct Curl_asn1Element subjectPublicKeyAlgorithm;
  struct Curl_asn1Element subjectPublicKey;
  struct Curl_asn1Element issuerUniqueID;
  struct Curl_asn1Element subjectUniqueID;
  struct Curl_asn1Element extensions;
};

/*
 * Prototypes.
 */

const char *Curl_getASN1Element(struct Curl_asn1Element *elem,
                                const char *beg, const char *end);
const char *Curl_ASN1tostr(struct Curl_asn1Element *elem, int type);
const char *Curl_DNtostr(struct Curl_asn1Element *dn);
int Curl_parseX509(struct Curl_X509certificate *cert,
                   const char *beg, const char *end);
CURLcode Curl_extract_certinfo(struct Curl_easy *data, int certnum,
                               const char *beg, const char *end);
CURLcode Curl_verifyhost(struct Curl_easy *data, struct connectdata *conn,
                         const char *beg, const char *end);
#endif /* USE_GSKIT or USE_NSS or USE_GNUTLS or USE_WOLFSSL or USE_SCHANNEL
        * or USE_SECTRANSP */
#endif /* HEADER_CURL_X509ASN1_H */
