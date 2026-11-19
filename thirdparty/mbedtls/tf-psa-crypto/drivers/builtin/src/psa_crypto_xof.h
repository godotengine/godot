/*
 *  PSA XOF (extendable-output function) layer on top of software crypto
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef TF_PSA_CRYPTO_PSA_CRYPTO_XOF_H
#define TF_PSA_CRYPTO_PSA_CRYPTO_XOF_H

#include <psa/crypto.h>

/** Set up a multipart XOF operation using built-in code.
 *
 * If an error occurs at any step after a call to mbedtls_psa_xof_setup(), the
 * operation will need to be reset by a call to mbedtls_psa_xof_abort(). The
 * core may call mbedtls_psa_xof_abort() at any time after the operation
 * has been initialized.
 *
 * After a successful call to mbedtls_psa_xof_setup(), the core must
 * eventually terminate the operation by calling mbedtls_psa_xof_abort().
 *
 * \warning The core must call the functions as directed. Otherwise, the
 *          behavior is undefined, although the driver will try to limit
 *          the damage to potentially data leakage and memory leaks but
 *          avoid memory corruption as long as the operation structure has
 *          been initialited.
 *
 * \param[in,out] operation The operation object to set up. It must have
 *                          been initialized to all-zero and not yet be in use.
 * \param alg               The XOF algorithm to compute (\c PSA_ALG_XXX value
 *                          such that #PSA_ALG_IS_XOF(\p alg) is true).
 *
 * \retval #PSA_SUCCESS
 *         Success.
 * \retval #PSA_ERROR_NOT_SUPPORTED
 *         \p alg is not supported
 * \retval #PSA_ERROR_BAD_STATE
 *         The operation state is not valid (it must be inactive).
 * \retval #PSA_ERROR_INSUFFICIENT_MEMORY \emptydescription
 * \retval #PSA_ERROR_CORRUPTION_DETECTED \emptydescription
 */
psa_status_t mbedtls_psa_xof_setup(
    mbedtls_psa_xof_operation_t *operation,
    psa_algorithm_t alg);

/** Set the context in a multipart XOF operation.
 *
 * The core must call mbedtls_psa_xof_setup() before calling this function.
 * The core must call this function as directed in the description of
 * the XOF algorithm, generally before calling mbedtls_psa_xof_update().
 * The core must not call this function if the XOF algorithm does not use
 * a context.
 *
 * If this function returns an error status, the operation enters an error
 * state and must be aborted by calling mbedtls_psa_xof_abort().
 *
 * \param[in,out] operation Active XOF operation.
 * \param[in] context       Buffer containing the message fragment to add.
 * \param context_length    Size of the \p input buffer in bytes.
 *
 * \retval #PSA_SUCCESS
 *         Success.
 * \retval #PSA_ERROR_BAD_STATE
 *         The operation state is not valid (it must be active).
 * \retval #PSA_ERROR_INSUFFICIENT_MEMORY \emptydescription
 * \retval #PSA_ERROR_CORRUPTION_DETECTED \emptydescription
 */
psa_status_t mbedtls_psa_xof_set_context(
    mbedtls_psa_xof_operation_t *operation,
    const uint8_t *context, size_t context_length);

/** Add an input fragment to a multipart XOF operation.
 *
 * The core must call mbedtls_psa_xof_setup() before calling this function.
 * The core must not call this function after calling
 * mbedtls_psa_xof_output() on the operation.
 *
 * This function can be called multiple times successively, to pass
 * input incrementally.
 *
 * If the XOF algorithm requires a context, the core must call
 * mbedtls_psa_xof_set_context() before this function. If the XOF
 * algorithm can use an optional context, the core must call
 * mbedtls_psa_xof_set_context() before this function, if at all.
 *
 * If this function returns an error status, the operation enters an error
 * state and must be aborted by calling mbedtls_psa_xof_abort().
 *
 * \param[in,out] operation Active XOF operation.
 * \param[in] input         Buffer containing the message fragment to add.
 * \param input_length      Size of the \p input buffer in bytes.
 *
 * \retval #PSA_SUCCESS
 *         Success.
 * \retval #PSA_ERROR_BAD_STATE
 *         The operation state is not valid (it must be active).
 * \retval #PSA_ERROR_INSUFFICIENT_MEMORY \emptydescription
 * \retval #PSA_ERROR_CORRUPTION_DETECTED \emptydescription
 */
psa_status_t mbedtls_psa_xof_update(
    mbedtls_psa_xof_operation_t *operation,
    const uint8_t *input, size_t input_length);

/** Obtain some output from a XOF operation.
 *
 * The core must call mbedtls_psa_xof_setup() before calling this function.
 *
 * This function calculates the incremental XOF output of the message formed
 * by concatenating the inputs passed to preceding calls to
 * mbedtls_psa_xof_update().
 *
 * This function can be called multiple times successively, to obtain
 * output incrementally.
 *
 * \param[in,out] operation     Active xof operation.
 * \param[out] output           Buffer where the XOF output is to be written.
 * \param output_size           Size of the \p output buffer in bytes.
 *
 * \retval #PSA_SUCCESS
 *         Success.
 * \retval #PSA_ERROR_BAD_STATE
 *         The operation state is not valid (it must be active).
 * \retval #PSA_ERROR_INSUFFICIENT_MEMORY \emptydescription
 * \retval #PSA_ERROR_CORRUPTION_DETECTED \emptydescription
 */
psa_status_t mbedtls_psa_xof_output(
    mbedtls_psa_xof_operation_t *operation,
    uint8_t *output, size_t output_size);

/** Abort an Mbed TLS xof operation.
 *
 * \note The signature of this function is that of a PSA driver xof_abort
 *       entry point. This function behaves as a xof_abort entry point as
 *       defined in the PSA driver interface specification for transparent
 *       drivers.
 *
 * Aborting an operation frees all associated resources except for the
 * \p operation structure itself. Once aborted, the operation object
 * can be reused for another operation by calling
 * mbedtls_psa_xof_setup() again.
 *
 * You may call this function any time after the operation object has
 * been initialized by one of the methods described in #psa_xof_operation_t.
 *
 * \param[in,out] operation     Initialized XOF operation.
 *
 * \retval #PSA_SUCCESS \emptydescription
 * \retval #PSA_ERROR_CORRUPTION_DETECTED \emptydescription
 */
psa_status_t mbedtls_psa_xof_abort(
    mbedtls_psa_xof_operation_t *operation);

#endif /* TF_PSA_CRYPTO_PSA_CRYPTO_XOF_H */
