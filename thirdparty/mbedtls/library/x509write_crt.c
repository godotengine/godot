/*
 *  X.509 certificate writing
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
/*
 * References:
 * - certificates: RFC 5280, updated by RFC 6818
 * - CSRs: PKCS#10 v1.7 aka RFC 2986
 * - attributes: PKCS#9 v2.0 aka RFC 2985
 */

#include "common.h"

#if defined(MBEDTLS_X509_CRT_WRITE_C)

#include "mbedtls/x509_crt.h"
#include "mbedtls/asn1write.h"
#include "mbedtls/error.h"
#include "mbedtls/oid.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/sha1.h"

#include <string.h>

#if defined(MBEDTLS_PEM_WRITE_C)
#include "mbedtls/pem.h"
#endif /* MBEDTLS_PEM_WRITE_C */

void mbedtls_x509write_crt_init(mbedtls_x509write_cert *ctx)
{
    memset(ctx, 0, sizeof(mbedtls_x509write_cert));

    mbedtls_mpi_init(&ctx->serial);
    ctx->version = MBEDTLS_X509_CRT_VERSION_3;
}

void mbedtls_x509write_crt_free(mbedtls_x509write_cert *ctx)
{
    mbedtls_mpi_free(&ctx->serial);

    mbedtls_asn1_free_named_data_list(&ctx->subject);
    mbedtls_asn1_free_named_data_list(&ctx->issuer);
    mbedtls_asn1_free_named_data_list(&ctx->extensions);

    mbedtls_platform_zeroize(ctx, sizeof(mbedtls_x509write_cert));
}

void mbedtls_x509write_crt_set_version(mbedtls_x509write_cert *ctx,
                                       int version)
{
    ctx->version = version;
}

void mbedtls_x509write_crt_set_md_alg(mbedtls_x509write_cert *ctx,
                                      mbedtls_md_type_t md_alg)
{
    ctx->md_alg = md_alg;
}

void mbedtls_x509write_crt_set_subject_key(mbedtls_x509write_cert *ctx,
                                           mbedtls_pk_context *key)
{
    ctx->subject_key = key;
}

void mbedtls_x509write_crt_set_issuer_key(mbedtls_x509write_cert *ctx,
                                          mbedtls_pk_context *key)
{
    ctx->issuer_key = key;
}

int mbedtls_x509write_crt_set_subject_name(mbedtls_x509write_cert *ctx,
                                           const char *subject_name)
{
    return mbedtls_x509_string_to_names(&ctx->subject, subject_name);
}

int mbedtls_x509write_crt_set_issuer_name(mbedtls_x509write_cert *ctx,
                                          const char *issuer_name)
{
    return mbedtls_x509_string_to_names(&ctx->issuer, issuer_name);
}

int mbedtls_x509write_crt_set_serial(mbedtls_x509write_cert *ctx,
                                     const mbedtls_mpi *serial)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if (mbedtls_mpi_size(serial) > MBEDTLS_X509_RFC5280_MAX_SERIAL_LEN) {
        return MBEDTLS_ERR_X509_BAD_INPUT_DATA;
    }

    if ((ret = mbedtls_mpi_copy(&ctx->serial, serial)) != 0) {
        return ret;
    }

    return 0;
}

int mbedtls_x509write_crt_set_validity(mbedtls_x509write_cert *ctx,
                                       const char *not_before,
                                       const char *not_after)
{
    if (strlen(not_before) != MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1 ||
        strlen(not_after)  != MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1) {
        return MBEDTLS_ERR_X509_BAD_INPUT_DATA;
    }
    strncpy(ctx->not_before, not_before, MBEDTLS_X509_RFC5280_UTC_TIME_LEN);
    strncpy(ctx->not_after, not_after, MBEDTLS_X509_RFC5280_UTC_TIME_LEN);
    ctx->not_before[MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1] = 'Z';
    ctx->not_after[MBEDTLS_X509_RFC5280_UTC_TIME_LEN - 1] = 'Z';

    return 0;
}

int mbedtls_x509write_crt_set_extension(mbedtls_x509write_cert *ctx,
                                        const char *oid, size_t oid_len,
                                        int critical,
                                        const unsigned char *val, size_t val_len)
{
    return mbedtls_x509_set_extension(&ctx->extensions, oid, oid_len,
                                      critical, val, val_len);
}

int mbedtls_x509write_crt_set_basic_constraints(mbedtls_x509write_cert *ctx,
                                                int is_ca, int max_pathlen)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char buf[9];
    unsigned char *c = buf + sizeof(buf);
    size_t len = 0;

    memset(buf, 0, sizeof(buf));

    if (is_ca && max_pathlen > 127) {
        return MBEDTLS_ERR_X509_BAD_INPUT_DATA;
    }

    if (is_ca) {
        if (max_pathlen >= 0) {
            MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_int(&c, buf,
                                                             max_pathlen));
        }
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_bool(&c, buf, 1));
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(&c, buf,
                                                     MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    return
        mbedtls_x509write_crt_set_extension(ctx, MBEDTLS_OID_BASIC_CONSTRAINTS,
                                            MBEDTLS_OID_SIZE(MBEDTLS_OID_BASIC_CONSTRAINTS),
                                            is_ca, buf + sizeof(buf) - len, len);
}

#if defined(MBEDTLS_SHA1_C)
int mbedtls_x509write_crt_set_subject_key_identifier(mbedtls_x509write_cert *ctx)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char buf[MBEDTLS_MPI_MAX_SIZE * 2 + 20]; /* tag, length + 2xMPI */
    unsigned char *c = buf + sizeof(buf);
    size_t len = 0;

    memset(buf, 0, sizeof(buf));
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_pk_write_pubkey(&c, buf, ctx->subject_key));

    ret = mbedtls_sha1_ret(buf + sizeof(buf) - len, len,
                           buf + sizeof(buf) - 20);
    if (ret != 0) {
        return ret;
    }
    c = buf + sizeof(buf) - 20;
    len = 20;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_asn1_write_tag(&c, buf, MBEDTLS_ASN1_OCTET_STRING));

    return mbedtls_x509write_crt_set_extension(ctx,
                                               MBEDTLS_OID_SUBJECT_KEY_IDENTIFIER,
                                               MBEDTLS_OID_SIZE(MBEDTLS_OID_SUBJECT_KEY_IDENTIFIER),
                                               0, buf + sizeof(buf) - len, len);
}

int mbedtls_x509write_crt_set_authority_key_identifier(mbedtls_x509write_cert *ctx)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char buf[MBEDTLS_MPI_MAX_SIZE * 2 + 20]; /* tag, length + 2xMPI */
    unsigned char *c = buf + sizeof(buf);
    size_t len = 0;

    memset(buf, 0, sizeof(buf));
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_pk_write_pubkey(&c, buf, ctx->issuer_key));

    ret = mbedtls_sha1_ret(buf + sizeof(buf) - len, len,
                           buf + sizeof(buf) - 20);
    if (ret != 0) {
        return ret;
    }
    c = buf + sizeof(buf) - 20;
    len = 20;

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_asn1_write_tag(&c, buf, MBEDTLS_ASN1_CONTEXT_SPECIFIC | 0));

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_asn1_write_tag(&c, buf,
                                                MBEDTLS_ASN1_CONSTRUCTED |
                                                MBEDTLS_ASN1_SEQUENCE));

    return mbedtls_x509write_crt_set_extension(
        ctx, MBEDTLS_OID_AUTHORITY_KEY_IDENTIFIER,
        MBEDTLS_OID_SIZE(MBEDTLS_OID_AUTHORITY_KEY_IDENTIFIER),
        0, buf + sizeof(buf) - len, len);
}
#endif /* MBEDTLS_SHA1_C */

int mbedtls_x509write_crt_set_key_usage(mbedtls_x509write_cert *ctx,
                                        unsigned int key_usage)
{
    unsigned char buf[5] = { 0 }, ku[2] = { 0 };
    unsigned char *c;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    const unsigned int allowed_bits = MBEDTLS_X509_KU_DIGITAL_SIGNATURE |
                                      MBEDTLS_X509_KU_NON_REPUDIATION   |
                                      MBEDTLS_X509_KU_KEY_ENCIPHERMENT  |
                                      MBEDTLS_X509_KU_DATA_ENCIPHERMENT |
                                      MBEDTLS_X509_KU_KEY_AGREEMENT     |
                                      MBEDTLS_X509_KU_KEY_CERT_SIGN     |
                                      MBEDTLS_X509_KU_CRL_SIGN          |
                                      MBEDTLS_X509_KU_ENCIPHER_ONLY     |
                                      MBEDTLS_X509_KU_DECIPHER_ONLY;

    /* Check that nothing other than the allowed flags is set */
    if ((key_usage & ~allowed_bits) != 0) {
        return MBEDTLS_ERR_X509_FEATURE_UNAVAILABLE;
    }

    c = buf + 5;
    MBEDTLS_PUT_UINT16_LE(key_usage, ku, 0);
    ret = mbedtls_asn1_write_named_bitstring(&c, buf, ku, 9);

    if (ret < 0) {
        return ret;
    } else if (ret < 3 || ret > 5) {
        return MBEDTLS_ERR_X509_INVALID_FORMAT;
    }

    ret = mbedtls_x509write_crt_set_extension(ctx, MBEDTLS_OID_KEY_USAGE,
                                              MBEDTLS_OID_SIZE(MBEDTLS_OID_KEY_USAGE),
                                              1, c, (size_t) ret);
    if (ret != 0) {
        return ret;
    }

    return 0;
}

int mbedtls_x509write_crt_set_ns_cert_type(mbedtls_x509write_cert *ctx,
                                           unsigned char ns_cert_type)
{
    unsigned char buf[4] = { 0 };
    unsigned char *c;
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    c = buf + 4;

    ret = mbedtls_asn1_write_named_bitstring(&c, buf, &ns_cert_type, 8);
    if (ret < 3 || ret > 4) {
        return ret;
    }

    ret = mbedtls_x509write_crt_set_extension(ctx, MBEDTLS_OID_NS_CERT_TYPE,
                                              MBEDTLS_OID_SIZE(MBEDTLS_OID_NS_CERT_TYPE),
                                              0, c, (size_t) ret);
    if (ret != 0) {
        return ret;
    }

    return 0;
}

static int x509_write_time(unsigned char **p, unsigned char *start,
                           const char *t, size_t size)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t len = 0;

    /*
     * write MBEDTLS_ASN1_UTC_TIME if year < 2050 (2 bytes shorter)
     */
    if (t[0] < '2' || (t[0] == '2' && t[1] == '0' && t[2] < '5')) {
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_raw_buffer(p, start,
                                                                (const unsigned char *) t + 2,
                                                                size - 2));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start,
                                                         MBEDTLS_ASN1_UTC_TIME));
    } else {
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_raw_buffer(p, start,
                                                                (const unsigned char *) t,
                                                                size));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(p, start, len));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(p, start,
                                                         MBEDTLS_ASN1_GENERALIZED_TIME));
    }

    return (int) len;
}

int mbedtls_x509write_crt_der(mbedtls_x509write_cert *ctx,
                              unsigned char *buf, size_t size,
                              int (*f_rng)(void *, unsigned char *, size_t),
                              void *p_rng)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    const char *sig_oid;
    size_t sig_oid_len = 0;
    unsigned char *c, *c2;
    unsigned char hash[64];
    unsigned char sig[MBEDTLS_PK_SIGNATURE_MAX_SIZE];
    size_t sub_len = 0, pub_len = 0, sig_and_oid_len = 0, sig_len;
    size_t len = 0;
    mbedtls_pk_type_t pk_alg;
    int write_sig_null_par;

    /*
     * Prepare data to be signed at the end of the target buffer
     */
    c = buf + size;

    /* Signature algorithm needed in TBS, and later for actual signature */

    /* There's no direct way of extracting a signature algorithm
     * (represented as an element of mbedtls_pk_type_t) from a PK instance. */
    if (mbedtls_pk_can_do(ctx->issuer_key, MBEDTLS_PK_RSA)) {
        pk_alg = MBEDTLS_PK_RSA;
    } else if (mbedtls_pk_can_do(ctx->issuer_key, MBEDTLS_PK_ECDSA)) {
        pk_alg = MBEDTLS_PK_ECDSA;
    } else {
        return MBEDTLS_ERR_X509_INVALID_ALG;
    }

    if ((ret = mbedtls_oid_get_oid_by_sig_alg(pk_alg, ctx->md_alg,
                                              &sig_oid, &sig_oid_len)) != 0) {
        return ret;
    }

    /*
     *  Extensions  ::=  SEQUENCE SIZE (1..MAX) OF Extension
     */

    /* Only for v3 */
    if (ctx->version == MBEDTLS_X509_CRT_VERSION_3) {
        MBEDTLS_ASN1_CHK_ADD(len,
                             mbedtls_x509_write_extensions(&c,
                                                           buf, ctx->extensions));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
        MBEDTLS_ASN1_CHK_ADD(len,
                             mbedtls_asn1_write_tag(&c, buf,
                                                    MBEDTLS_ASN1_CONSTRUCTED |
                                                    MBEDTLS_ASN1_SEQUENCE));
        MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
        MBEDTLS_ASN1_CHK_ADD(len,
                             mbedtls_asn1_write_tag(&c, buf,
                                                    MBEDTLS_ASN1_CONTEXT_SPECIFIC |
                                                    MBEDTLS_ASN1_CONSTRUCTED | 3));
    }

    /*
     *  SubjectPublicKeyInfo
     */
    MBEDTLS_ASN1_CHK_ADD(pub_len,
                         mbedtls_pk_write_pubkey_der(ctx->subject_key,
                                                     buf, c - buf));
    c -= pub_len;
    len += pub_len;

    /*
     *  Subject  ::=  Name
     */
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_x509_write_names(&c, buf,
                                                  ctx->subject));

    /*
     *  Validity ::= SEQUENCE {
     *       notBefore      Time,
     *       notAfter       Time }
     */
    sub_len = 0;

    MBEDTLS_ASN1_CHK_ADD(sub_len,
                         x509_write_time(&c, buf, ctx->not_after,
                                         MBEDTLS_X509_RFC5280_UTC_TIME_LEN));

    MBEDTLS_ASN1_CHK_ADD(sub_len,
                         x509_write_time(&c, buf, ctx->not_before,
                                         MBEDTLS_X509_RFC5280_UTC_TIME_LEN));

    len += sub_len;
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, sub_len));
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_asn1_write_tag(&c, buf,
                                                MBEDTLS_ASN1_CONSTRUCTED |
                                                MBEDTLS_ASN1_SEQUENCE));

    /*
     *  Issuer  ::=  Name
     */
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_x509_write_names(&c, buf,
                                                       ctx->issuer));

    /*
     *  Signature   ::=  AlgorithmIdentifier
     */
    if (pk_alg == MBEDTLS_PK_ECDSA) {
        /*
         * The AlgorithmIdentifier's parameters field must be absent for DSA/ECDSA signature
         * algorithms, see https://www.rfc-editor.org/rfc/rfc5480#page-17 and
         * https://www.rfc-editor.org/rfc/rfc5758#section-3.
         */
        write_sig_null_par = 0;
    } else {
        write_sig_null_par = 1;
    }
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_asn1_write_algorithm_identifier_ext(&c, buf,
                                                                     sig_oid, strlen(sig_oid),
                                                                     0, write_sig_null_par));

    /*
     *  Serial   ::=  INTEGER
     */
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_mpi(&c, buf,
                                                     &ctx->serial));

    /*
     *  Version  ::=  INTEGER  {  v1(0), v2(1), v3(2)  }
     */

    /* Can be omitted for v1 */
    if (ctx->version != MBEDTLS_X509_CRT_VERSION_1) {
        sub_len = 0;
        MBEDTLS_ASN1_CHK_ADD(sub_len,
                             mbedtls_asn1_write_int(&c, buf, ctx->version));
        len += sub_len;
        MBEDTLS_ASN1_CHK_ADD(len,
                             mbedtls_asn1_write_len(&c, buf, sub_len));
        MBEDTLS_ASN1_CHK_ADD(len,
                             mbedtls_asn1_write_tag(&c, buf,
                                                    MBEDTLS_ASN1_CONTEXT_SPECIFIC |
                                                    MBEDTLS_ASN1_CONSTRUCTED | 0));
    }

    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len,
                         mbedtls_asn1_write_tag(&c, buf, MBEDTLS_ASN1_CONSTRUCTED |
                                                MBEDTLS_ASN1_SEQUENCE));

    /*
     * Make signature
     */

    /* Compute hash of CRT. */
    if ((ret = mbedtls_md(mbedtls_md_info_from_type(ctx->md_alg), c,
                          len, hash)) != 0) {
        return ret;
    }

    if ((ret = mbedtls_pk_sign(ctx->issuer_key, ctx->md_alg,
                               hash, 0, sig, &sig_len,
                               f_rng, p_rng)) != 0) {
        return ret;
    }

    /* Move CRT to the front of the buffer to have space
     * for the signature. */
    memmove(buf, c, len);
    c = buf + len;

    /* Add signature at the end of the buffer,
     * making sure that it doesn't underflow
     * into the CRT buffer. */
    c2 = buf + size;
    MBEDTLS_ASN1_CHK_ADD(sig_and_oid_len, mbedtls_x509_write_sig(&c2, c,
                                                                 sig_oid, sig_oid_len,
                                                                 sig, sig_len, pk_alg));

    /*
     * Memory layout after this step:
     *
     * buf       c=buf+len                c2            buf+size
     * [CRT0,...,CRTn, UNUSED, ..., UNUSED, SIG0, ..., SIGm]
     */

    /* Move raw CRT to just before the signature. */
    c = c2 - len;
    memmove(c, buf, len);

    len += sig_and_oid_len;
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_len(&c, buf, len));
    MBEDTLS_ASN1_CHK_ADD(len, mbedtls_asn1_write_tag(&c, buf,
                                                     MBEDTLS_ASN1_CONSTRUCTED |
                                                     MBEDTLS_ASN1_SEQUENCE));

    return (int) len;
}

#define PEM_BEGIN_CRT           "-----BEGIN CERTIFICATE-----\n"
#define PEM_END_CRT             "-----END CERTIFICATE-----\n"

#if defined(MBEDTLS_PEM_WRITE_C)
int mbedtls_x509write_crt_pem(mbedtls_x509write_cert *crt,
                              unsigned char *buf, size_t size,
                              int (*f_rng)(void *, unsigned char *, size_t),
                              void *p_rng)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    size_t olen;

    if ((ret = mbedtls_x509write_crt_der(crt, buf, size,
                                         f_rng, p_rng)) < 0) {
        return ret;
    }

    if ((ret = mbedtls_pem_write_buffer(PEM_BEGIN_CRT, PEM_END_CRT,
                                        buf + size - ret, ret,
                                        buf, size, &olen)) != 0) {
        return ret;
    }

    return 0;
}
#endif /* MBEDTLS_PEM_WRITE_C */

#endif /* MBEDTLS_X509_CRT_WRITE_C */
