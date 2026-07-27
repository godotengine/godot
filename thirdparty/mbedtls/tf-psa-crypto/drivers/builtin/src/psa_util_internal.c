/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "tf_psa_crypto_common.h"

/* This is needed for MBEDTLS_ERR_XXX macros */
#include <mbedtls/private/error_common.h>

#include "psa_util_internal.h"

#if defined(MBEDTLS_PSA_CRYPTO_CLIENT)

#include <psa/crypto.h>

#if defined(MBEDTLS_MD_LIGHT)
#include <mbedtls/md.h>
#endif
#if defined(MBEDTLS_LMS_C)
#include <mbedtls/lms.h>
#endif
#if defined(PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY) ||    \
    defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_BASIC)
#include <mbedtls/private/rsa.h>
#endif
#if defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY)
#include <mbedtls/private/ecp.h>
#endif
#if defined(MBEDTLS_PK_C)
#include <mbedtls/pk.h>
#if defined(MBEDTLS_PK_HAVE_PRIVATE_HEADER)
#include <mbedtls/private/pk_private.h>
#endif /* MBEDTLS_PK_HAVE_PRIVATE_HEADER */
#endif
#if defined(MBEDTLS_BLOCK_CIPHER_SOME_PSA)
#include <mbedtls/private/cipher.h>
#endif

/* PSA_SUCCESS is kept at the top of each error table since
 * it's the most common status when everything functions properly. */
#if defined(MBEDTLS_MD_LIGHT)
const mbedtls_error_pair_t psa_to_md_errors[] =
{
    { PSA_SUCCESS,                     0 },
    { PSA_ERROR_NOT_SUPPORTED,         MBEDTLS_ERR_MD_FEATURE_UNAVAILABLE },
    { PSA_ERROR_INVALID_ARGUMENT,      MBEDTLS_ERR_MD_BAD_INPUT_DATA },
    { PSA_ERROR_INSUFFICIENT_MEMORY,   MBEDTLS_ERR_MD_ALLOC_FAILED }
};
#endif

#if defined(MBEDTLS_BLOCK_CIPHER_SOME_PSA)
const mbedtls_error_pair_t psa_to_cipher_errors[] =
{
    { PSA_SUCCESS,                     0 },
    { PSA_ERROR_NOT_SUPPORTED,         MBEDTLS_ERR_CIPHER_FEATURE_UNAVAILABLE },
    { PSA_ERROR_INVALID_ARGUMENT,      MBEDTLS_ERR_CIPHER_BAD_INPUT_DATA },
    { PSA_ERROR_INSUFFICIENT_MEMORY,   MBEDTLS_ERR_CIPHER_ALLOC_FAILED }
};
#endif

#if defined(MBEDTLS_LMS_C)
const mbedtls_error_pair_t psa_to_lms_errors[] =
{
    { PSA_SUCCESS,                     0 },
    { PSA_ERROR_BUFFER_TOO_SMALL,      MBEDTLS_ERR_LMS_BUFFER_TOO_SMALL },
    { PSA_ERROR_INVALID_ARGUMENT,      MBEDTLS_ERR_LMS_BAD_INPUT_DATA }
};
#endif

#if defined(PSA_WANT_KEY_TYPE_RSA_PUBLIC_KEY) ||    \
    defined(PSA_WANT_KEY_TYPE_RSA_KEY_PAIR_BASIC)
const mbedtls_error_pair_t psa_to_pk_rsa_errors[] =
{
    { PSA_SUCCESS,                     0 },
    { PSA_ERROR_NOT_PERMITTED,         MBEDTLS_ERR_RSA_BAD_INPUT_DATA },
    { PSA_ERROR_INVALID_ARGUMENT,      MBEDTLS_ERR_RSA_BAD_INPUT_DATA },
    { PSA_ERROR_INVALID_HANDLE,        MBEDTLS_ERR_RSA_BAD_INPUT_DATA },
    { PSA_ERROR_BUFFER_TOO_SMALL,      MBEDTLS_ERR_RSA_OUTPUT_TOO_LARGE },
    { PSA_ERROR_INSUFFICIENT_ENTROPY,  MBEDTLS_ERR_RSA_RNG_FAILED },
    { PSA_ERROR_INVALID_SIGNATURE,     MBEDTLS_ERR_RSA_VERIFY_FAILED },
    { PSA_ERROR_INVALID_PADDING,       MBEDTLS_ERR_RSA_INVALID_PADDING }
};
#endif

#if defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY)
const mbedtls_error_pair_t psa_to_pk_ecdsa_errors[] =
{
    { PSA_SUCCESS,                     0 },
    { PSA_ERROR_NOT_PERMITTED,         MBEDTLS_ERR_ECP_BAD_INPUT_DATA },
    { PSA_ERROR_INVALID_ARGUMENT,      MBEDTLS_ERR_ECP_BAD_INPUT_DATA },
    { PSA_ERROR_INVALID_HANDLE,        MBEDTLS_ERR_ECP_FEATURE_UNAVAILABLE },
    { PSA_ERROR_BUFFER_TOO_SMALL,      MBEDTLS_ERR_ECP_BUFFER_TOO_SMALL },
    { PSA_ERROR_INSUFFICIENT_ENTROPY,  MBEDTLS_ERR_ECP_RANDOM_FAILED },
    { PSA_ERROR_INVALID_SIGNATURE,     MBEDTLS_ERR_ECP_VERIFY_FAILED }
};
#endif

int psa_generic_status_to_mbedtls(psa_status_t status)
{
    switch (status) {
        case PSA_SUCCESS:
            return 0;
        case PSA_ERROR_NOT_SUPPORTED:
            return MBEDTLS_ERR_PLATFORM_FEATURE_UNSUPPORTED;
        case PSA_ERROR_CORRUPTION_DETECTED:
            return MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
        case PSA_ERROR_COMMUNICATION_FAILURE:
        case PSA_ERROR_HARDWARE_FAILURE:
            return MBEDTLS_ERR_PLATFORM_HW_ACCEL_FAILED;
        case PSA_ERROR_NOT_PERMITTED:
        default:
            return MBEDTLS_ERR_ERROR_GENERIC_ERROR;
    }
}

int psa_status_to_mbedtls(psa_status_t status,
                          const mbedtls_error_pair_t *local_translations,
                          size_t local_errors_num,
                          int (*fallback_f)(psa_status_t))
{
    for (size_t i = 0; i < local_errors_num; i++) {
        if (status == local_translations[i].psa_status) {
            return local_translations[i].mbedtls_error;
        }
    }
    return fallback_f(status);
}

#if defined(MBEDTLS_PK_C)
int psa_pk_status_to_mbedtls(psa_status_t status)
{
    switch (status) {
        case PSA_ERROR_INVALID_HANDLE:
            return MBEDTLS_ERR_PK_KEY_INVALID_FORMAT;
        case PSA_ERROR_BUFFER_TOO_SMALL:
            return MBEDTLS_ERR_PK_BUFFER_TOO_SMALL;
        case PSA_ERROR_NOT_SUPPORTED:
            return MBEDTLS_ERR_PK_FEATURE_UNAVAILABLE;
        case PSA_ERROR_INVALID_ARGUMENT:
            return MBEDTLS_ERR_PK_INVALID_ALG;
        case PSA_ERROR_NOT_PERMITTED:
            return MBEDTLS_ERR_PK_TYPE_MISMATCH;
        case PSA_ERROR_INSUFFICIENT_MEMORY:
            return MBEDTLS_ERR_PK_ALLOC_FAILED;
        case PSA_ERROR_BAD_STATE:
            return MBEDTLS_ERR_PK_BAD_INPUT_DATA;
        case PSA_ERROR_DATA_CORRUPT:
        case PSA_ERROR_DATA_INVALID:
        case PSA_ERROR_STORAGE_FAILURE:
            return MBEDTLS_ERR_PK_FILE_IO_ERROR;
        default:
            return psa_generic_status_to_mbedtls(status);
    }
}
#endif /* MBEDTLS_PK_C */

/****************************************************************/
/* Key management */
/****************************************************************/

#if defined(PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY)
psa_ecc_family_t mbedtls_ecc_group_to_psa(mbedtls_ecp_group_id grpid,
                                          size_t *bits)
{
    switch (grpid) {
#if defined(PSA_WANT_ECC_SECP_R1_256)
        case MBEDTLS_ECP_DP_SECP256R1:
            *bits = 256;
            return PSA_ECC_FAMILY_SECP_R1;
#endif
#if defined(PSA_WANT_ECC_SECP_R1_384)
        case MBEDTLS_ECP_DP_SECP384R1:
            *bits = 384;
            return PSA_ECC_FAMILY_SECP_R1;
#endif
#if defined(PSA_WANT_ECC_SECP_R1_521)
        case MBEDTLS_ECP_DP_SECP521R1:
            *bits = 521;
            return PSA_ECC_FAMILY_SECP_R1;
#endif
#if defined(PSA_WANT_ECC_BRAINPOOL_P_R1_256)
        case MBEDTLS_ECP_DP_BP256R1:
            *bits = 256;
            return PSA_ECC_FAMILY_BRAINPOOL_P_R1;
#endif
#if defined(PSA_WANT_ECC_BRAINPOOL_P_R1_384)
        case MBEDTLS_ECP_DP_BP384R1:
            *bits = 384;
            return PSA_ECC_FAMILY_BRAINPOOL_P_R1;
#endif
#if defined(PSA_WANT_ECC_BRAINPOOL_P_R1_512)
        case MBEDTLS_ECP_DP_BP512R1:
            *bits = 512;
            return PSA_ECC_FAMILY_BRAINPOOL_P_R1;
#endif
#if defined(PSA_WANT_ECC_MONTGOMERY_255)
        case MBEDTLS_ECP_DP_CURVE25519:
            *bits = 255;
            return PSA_ECC_FAMILY_MONTGOMERY;
#endif
#if defined(PSA_WANT_ECC_SECP_K1_256)
        case MBEDTLS_ECP_DP_SECP256K1:
            *bits = 256;
            return PSA_ECC_FAMILY_SECP_K1;
#endif
#if defined(PSA_WANT_ECC_MONTGOMERY_448)
        case MBEDTLS_ECP_DP_CURVE448:
            *bits = 448;
            return PSA_ECC_FAMILY_MONTGOMERY;
#endif
        default:
            *bits = 0;
            return 0;
    }
}

mbedtls_ecp_group_id mbedtls_ecc_group_from_psa(psa_ecc_family_t family,
                                                size_t bits)
{
    switch (family) {
        case PSA_ECC_FAMILY_SECP_R1:
            switch (bits) {
#if defined(PSA_WANT_ECC_SECP_R1_256)
                case 256:
                    return MBEDTLS_ECP_DP_SECP256R1;
#endif
#if defined(PSA_WANT_ECC_SECP_R1_384)
                case 384:
                    return MBEDTLS_ECP_DP_SECP384R1;
#endif
#if defined(PSA_WANT_ECC_SECP_R1_521)
                case 521:
                    return MBEDTLS_ECP_DP_SECP521R1;
#endif
            }
            break;

        case PSA_ECC_FAMILY_BRAINPOOL_P_R1:
            switch (bits) {
#if defined(PSA_WANT_ECC_BRAINPOOL_P_R1_256)
                case 256:
                    return MBEDTLS_ECP_DP_BP256R1;
#endif
#if defined(PSA_WANT_ECC_BRAINPOOL_P_R1_384)
                case 384:
                    return MBEDTLS_ECP_DP_BP384R1;
#endif
#if defined(PSA_WANT_ECC_BRAINPOOL_P_R1_512)
                case 512:
                    return MBEDTLS_ECP_DP_BP512R1;
#endif
            }
            break;

        case PSA_ECC_FAMILY_MONTGOMERY:
            switch (bits) {
#if defined(PSA_WANT_ECC_MONTGOMERY_255)
                case 255:
                    return MBEDTLS_ECP_DP_CURVE25519;
#endif
#if defined(PSA_WANT_ECC_MONTGOMERY_448)
                case 448:
                    return MBEDTLS_ECP_DP_CURVE448;
#endif
            }
            break;

        case PSA_ECC_FAMILY_SECP_K1:
            switch (bits) {
#if defined(PSA_WANT_ECC_SECP_K1_256)
                case 256:
                    return MBEDTLS_ECP_DP_SECP256K1;
#endif
            }
            break;
    }

    return MBEDTLS_ECP_DP_NONE;
}
#endif /* PSA_WANT_KEY_TYPE_ECC_PUBLIC_KEY */
#endif /* MBEDTLS_PSA_CRYPTO_CLIENT */
