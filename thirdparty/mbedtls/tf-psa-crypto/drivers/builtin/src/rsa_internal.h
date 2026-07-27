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
#ifndef TF_PSA_CRYPTO_RSA_INTERNAL_H
#define TF_PSA_CRYPTO_RSA_INTERNAL_H

#include "mbedtls/private/rsa.h"
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

#if defined(MBEDTLS_PKCS1_V15)
/**
 * \brief          This function performs a PKCS#1 v1.5 decryption
 *                 operation (RSAES-PKCS1-v1_5-DECRYPT).
 *
 * \warning        This is an inherently dangerous function (CWE-242). Unless
 *                 it is used in a side channel free and safe way, the calling
 *                 code is vulnerable.
 *
 * \note           The output buffer length \c output_max_len should be
 *                 as large as the size \p ctx->len of \p ctx->N, for example,
 *                 128 Bytes if RSA-1024 is used, to be able to hold an
 *                 arbitrary decrypted message. If it is not large enough to
 *                 hold the decryption of the particular ciphertext provided,
 *                 the function returns #PSA_ERROR_BUFFER_TOO_SMALL.
 *
 * \param ctx      The initialized RSA context to use.
 * \param f_rng    The RNG function. This is used for blinding and is
 *                 mandatory; see mbedtls_rsa_private() for more.
 * \param p_rng    The RNG context to be passed to \p f_rng. This may be
 *                 \c NULL if \p f_rng doesn't need a context.
 * \param olen     The address at which to store the length of
 *                 the plaintext. This must not be \c NULL.
 * \param input    The ciphertext buffer. This must be a readable buffer
 *                 of length \c ctx->len Bytes. For example, \c 256 Bytes
 *                 for an 2048-bit RSA modulus.
 * \param output   The buffer used to hold the plaintext. This must
 *                 be a writable buffer of length \p output_max_len Bytes.
 * \param output_max_len The length in Bytes of the output buffer \p output.
 * \param sensitive_ret
 *                 If this function returns \c 0, this is set to either 0 for
 *                 success, or #PSA_ERROR_INVALID_PADDING if the padding
 *                 was invalid, or #PSA_ERROR_BUFFER_TOO_SMALL if the
 *                 padding was valid but resulted in a plaintext larger than the
 *                 output buffer.
 *                 If this function returns non-zero this is set to \c 0.
 *
 * \return         \c 0 on success, or when the only error is invalid padding,
 *                 or valid padding that results in a plaintext larger than the
 *                 output buffer.
 * \return         An \c MBEDTLS_ERR_RSA_XXX error code on other failure.
 *
 */
int mbedtls_rsa_rsaes_pkcs1_v15_decrypt_ext(mbedtls_rsa_context *ctx,
                                            int (*f_rng)(void *, unsigned char *, size_t),
                                            void *p_rng,
                                            size_t *olen,
                                            const unsigned char *input,
                                            unsigned char *output,
                                            size_t output_max_len,
                                            int *sensitive_ret);
#endif

#if defined(MBEDTLS_PKCS1_V21)
/**
 * \brief This function is analogue to \c mbedtls_rsa_rsassa_pss_sign_ext().
 *        The only difference between them is that this function is more flexible
 *        on the parameters of \p ctx that are set with \c mbedtls_rsa_set_padding().
 *
 * \note  Compared to its counterpart, this function:
 *        - does not check the padding setting of \p ctx.
 *        - allows the hash_id of \p ctx to be MBEDTLS_MD_NONE,
 *          in which case it uses \p md_alg as the hash_id.
 *
 * \note  Refer to \c mbedtls_rsa_rsassa_pss_sign_ext() for a description
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

#endif /* TF_PSA_CRYPTO_RSA_INTERNAL_H */
