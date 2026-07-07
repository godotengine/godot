/**
 * \file rsa_internal.h
 *
 * \brief Internal-only RSA public-key cryptosystem API.
 *
 * This file declares RSA-related functions that are to be used
 * only from within the Mbed TLS library itself.
 *
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef MBEDTLS_RSA_INTERNAL_H
#define MBEDTLS_RSA_INTERNAL_H

#include "mbedtls/rsa.h"
#include "mbedtls/asn1.h"

/**
 * \brief           Parse a PKCS#1 (ASN.1) encoded private RSA key.
 *
 * \param rsa       The RSA context where parsed data will be stored.
 * \param key       The buffer that contains the key.
 * \param keylen    The length of the key buffer in bytes.
 *
 * \return          0 on success.
 * \return          MBEDTLS_ERR_ASN1_xxx in case of ASN.1 parsing errors.
 * \return          MBEDTLS_ERR_RSA_xxx in case of RSA internal failures while
 *                  parsing data.
 * \return          MBEDTLS_ERR_RSA_KEY_CHECK_FAILED if validity checks on the
 *                  provided key fail.
 */
int mbedtls_rsa_parse_key(mbedtls_rsa_context *rsa, const unsigned char *key, size_t keylen);

/**
 * \brief           Parse a PKCS#1 (ASN.1) encoded public RSA key.
 *
 * \param rsa       The RSA context where parsed data will be stored.
 * \param key       The buffer that contains the key.
 * \param keylen    The length of the key buffer in bytes.
 *
 * \return          0 on success.
 * \return          MBEDTLS_ERR_ASN1_xxx in case of ASN.1 parsing errors.
 * \return          MBEDTLS_ERR_RSA_xxx in case of RSA internal failures while
 *                  parsing data.
 * \return          MBEDTLS_ERR_RSA_KEY_CHECK_FAILED if validity checks on the
 *                  provided key fail.
 */
int mbedtls_rsa_parse_pubkey(mbedtls_rsa_context *rsa, const unsigned char *key, size_t keylen);

/**
 * \brief           Write a PKCS#1 (ASN.1) encoded private RSA key.
 *
 * \param rsa       The RSA context which contains the data to be written.
 * \param start     Beginning of the buffer that will be filled with the
 *                  private key.
 * \param p         End of the buffer that will be filled with the private key.
 *                  On successful return, the referenced pointer will be
 *                  updated in order to point to the beginning of written data.
 *
 * \return          On success, the number of bytes written to the output buffer
 *                  (i.e. a value > 0).
 * \return          MBEDTLS_ERR_RSA_BAD_INPUT_DATA if the RSA context does not
 *                  contain a valid key pair.
 * \return          MBEDTLS_ERR_ASN1_xxx in case of failure while writing to the
 *                  output buffer.
 *
 * \note            The output buffer is filled backward, i.e. starting from its
 *                  end and moving toward its start.
 */
int mbedtls_rsa_write_key(const mbedtls_rsa_context *rsa, unsigned char *start,
                          unsigned char **p);

/**
 * \brief           Parse a PKCS#1 (ASN.1) encoded public RSA key.
 *
 * \param rsa       The RSA context which contains the data to be written.
 * \param start     Beginning of the buffer that will be filled with the
 *                  private key.
 * \param p         End of the buffer that will be filled with the private key.
 *                  On successful return, the referenced pointer will be
 *                  updated in order to point to the beginning of written data.
 *
 * \return          On success, the number of bytes written to the output buffer
 *                  (i.e. a value > 0).
 * \return          MBEDTLS_ERR_RSA_BAD_INPUT_DATA if the RSA context does not
 *                  contain a valid public key.
 * \return          MBEDTLS_ERR_ASN1_xxx in case of failure while writing to the
 *                  output buffer.
 *
 * \note            The output buffer is filled backward, i.e. starting from its
 *                  end and moving toward its start.
 */
int mbedtls_rsa_write_pubkey(const mbedtls_rsa_context *rsa, unsigned char *start,
                             unsigned char **p);

#if defined(MBEDTLS_PKCS1_V21)
/**
 * \brief This function is analogue to \c mbedtls_rsa_rsassa_pss_sign().
 *        The only difference between them is that this function is more flexible
 *        on the parameters of \p ctx that are set with \c mbedtls_rsa_set_padding().
 *
 * \note  Compared to its counterpart, this function:
 *        - does not check the padding setting of \p ctx.
 *        - allows the hash_id of \p ctx to be MBEDTLS_MD_NONE,
 *          in which case it uses \p md_alg as the hash_id.
 *
 * \note  Refer to \c mbedtls_rsa_rsassa_pss_sign() for a description
 *        of the functioning and parameters of this function.
 */
int mbedtls_rsa_rsassa_pss_sign_no_mode_check(mbedtls_rsa_context *ctx,
                                              int (*f_rng)(void *, unsigned char *, size_t),
                                              void *p_rng,
                                              mbedtls_md_type_t md_alg,
                                              unsigned int hashlen,
                                              const unsigned char *hash,
                                              unsigned char *sig);
#endif /* MBEDTLS_PKCS1_V21 */

/* This would normally be in rsa_invasive.h but it didn't exist before 3.6
 * became an LTS, and I'd rather not add files in LTS if it can be avoided. */
#if defined(MBEDTLS_TEST_HOOKS)
#if defined(MBEDTLS_PKCS1_V15) && defined(MBEDTLS_RSA_C) && !defined(MBEDTLS_RSA_ALT)

/** This function performs the unpadding part of a PKCS#1 v1.5 decryption
 *  operation (EME-PKCS1-v1_5 decoding).
 *
 * \note The return value from this function is a sensitive value
 *       (this is unusual). #MBEDTLS_ERR_RSA_OUTPUT_TOO_LARGE shouldn't happen
 *       in a well-written application, but 0 vs #MBEDTLS_ERR_RSA_INVALID_PADDING
 *       is often a situation that an attacker can provoke and leaking which
 *       one is the result is precisely the information the attacker wants.
 *
 * \param input          The input buffer which is the payload inside PKCS#1v1.5
 *                       encryption padding, called the "encoded message EM"
 *                       by the terminology.
 * \param ilen           The length of the payload in the \p input buffer.
 * \param output         The buffer for the payload, called "message M" by the
 *                       PKCS#1 terminology. This must be a writable buffer of
 *                       length \p output_max_len bytes.
 * \param olen           The address at which to store the length of
 *                       the payload. This must not be \c NULL.
 * \param output_max_len The length in bytes of the output buffer \p output.
 *
 * \return      \c 0 on success.
 * \return      #MBEDTLS_ERR_RSA_OUTPUT_TOO_LARGE
 *              The output buffer is too small for the unpadded payload.
 * \return      #MBEDTLS_ERR_RSA_INVALID_PADDING
 *              The input doesn't contain properly formatted padding.
 */
MBEDTLS_STATIC_TESTABLE int mbedtls_ct_rsaes_pkcs1_v15_unpadding(
    unsigned char *input, size_t ilen,
    unsigned char *output, size_t output_max_len, size_t *olen);
#endif /* MBEDTLS_PKCS1_V15 && MBEDTLS_RSA_C && ! MBEDTLS_RSA_ALT */
#endif /* MBEDTLS_TEST_HOOKS */

#if defined(MBEDTLS_PKCS1_V15) && defined(MBEDTLS_RSA_C)
/** Decompose sensitive return values out of a return code, in constant time.
 *
 * \param invalid_padding_in    The value of \p combined_ret that indicates
 *                              invalid padding.
 * \param invalid_padding_out   The value to set \p problem to in case of
 *                              invalid padding.
 * \param output_too_large_in   The value of \p combined_ret that indicates
 *                              an insufficient output buffer size.
 * \param output_too_large_out  The value to set \p problem to in case of
 *                              an insufficient output buffer size.
 * \param combined_ret          The value to decompose.
 * \param[out] problem          On output:
 *                              - \p invalid_padding_out,
 *                                if \p combined_ret = \p invalid_padding_in;
 *                              - \p output_too_large_out,
 *                                if \p combined_ret = \p output_too_large_in;
 *                              - otherwise \c 0.
 *
 * \return                      - \c 0 if \p combined_ret = \p invalid_padding_in
 *                                or \p combined_ret = \p output_too_large_in;
 *                              - otherwise \c combined_ret.
 */
int mbedtls_rsa_decrypt_decompose_ret(
    int invalid_padding_in, int invalid_padding_out,
    int output_too_large_in, int output_too_large_out,
    int combined_ret,
    int *problem);
#endif

#endif /* rsa_internal.h */
