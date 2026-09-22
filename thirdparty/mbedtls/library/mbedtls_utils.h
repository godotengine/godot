#include "mbedtls/pk.h"
#include "psa/crypto.h"

#ifndef MBEDTLS_UTILS_H
#define MBEDTLS_UTILS_H

/* Return the PSA algorithm associated to the given combination of "sigalg" and "hash_alg". */
static inline psa_algorithm_t mbedtls_psa_alg_from_pk_sigalg(mbedtls_pk_sigalg_t sigalg,
                                                             psa_algorithm_t hash_alg)
{
    switch (sigalg) {
        case MBEDTLS_PK_SIGALG_RSA_PKCS1V15:
            return PSA_ALG_RSA_PKCS1V15_SIGN(hash_alg);
        case MBEDTLS_PK_SIGALG_RSA_PSS:
            return PSA_ALG_RSA_PSS(hash_alg);
        case MBEDTLS_PK_SIGALG_ECDSA:
            return MBEDTLS_PK_ALG_ECDSA(hash_alg);
        default:
            return PSA_ALG_NONE;
    }
}

#endif /* MBEDTLS_UTILS_H */
