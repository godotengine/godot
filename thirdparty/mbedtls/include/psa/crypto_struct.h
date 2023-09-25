/**
 * \file psa/crypto_struct.h
 *
 * \brief PSA cryptography module: Mbed TLS structured type implementations
 *
 * \note This file may not be included directly. Applications must
 * include psa/crypto.h.
 *
 * This file contains the definitions of some data structures with
 * implementation-specific definitions.
 *
 * In implementations with isolation between the application and the
 * cryptography module, it is expected that the front-end and the back-end
 * would have different versions of this file.
 *
 * <h3>Design notes about multipart operation structures</h3>
 *
 * For multipart operations without driver delegation support, each multipart
 * operation structure contains a `psa_algorithm_t alg` field which indicates
 * which specific algorithm the structure is for. When the structure is not in
 * use, `alg` is 0. Most of the structure consists of a union which is
 * discriminated by `alg`.
 *
 * For multipart operations with driver delegation support, each multipart
 * operation structure contains an `unsigned int id` field indicating which
 * driver got assigned to do the operation. When the structure is not in use,
 * 'id' is 0. The structure contains also a driver context which is the union
 * of the contexts of all drivers able to handle the type of multipart
 * operation.
 *
 * Note that when `alg` or `id` is 0, the content of other fields is undefined.
 * In particular, it is not guaranteed that a freshly-initialized structure
 * is all-zero: we initialize structures to something like `{0, 0}`, which
 * is only guaranteed to initializes the first member of the union;
 * GCC and Clang initialize the whole structure to 0 (at the time of writing),
 * but MSVC and CompCert don't.
 *
 * In Mbed TLS, multipart operation structures live independently from
 * the key. This allows Mbed TLS to free the key objects when destroying
 * a key slot. If a multipart operation needs to remember the key after
 * the setup function returns, the operation structure needs to contain a
 * copy of the key.
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef PSA_CRYPTO_STRUCT_H
#define PSA_CRYPTO_STRUCT_H
#include "mbedtls/private_access.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Include the Mbed TLS configuration file, the way Mbed TLS does it
 * in each of its header files. */
#include "mbedtls/build_info.h"

/* Include the context definition for the compiled-in drivers for the primitive
 * algorithms. */
#include "psa/crypto_driver_contexts_primitives.h"

struct psa_hash_operation_s {
    /** Unique ID indicating which driver got assigned to do the
     * operation. Since driver contexts are driver-specific, swapping
     * drivers halfway through the operation is not supported.
     * ID values are auto-generated in psa_driver_wrappers.h.
     * ID value zero means the context is not valid or not assigned to
     * any driver (i.e. the driver context is not active, in use). */
    unsigned int MBEDTLS_PRIVATE(id);
    psa_driver_hash_context_t MBEDTLS_PRIVATE(ctx);
};

#define PSA_HASH_OPERATION_INIT { 0, { 0 } }
static inline struct psa_hash_operation_s psa_hash_operation_init(void)
{
    const struct psa_hash_operation_s v = PSA_HASH_OPERATION_INIT;
    return v;
}

struct psa_cipher_operation_s {
    /** Unique ID indicating which driver got assigned to do the
     * operation. Since driver contexts are driver-specific, swapping
     * drivers halfway through the operation is not supported.
     * ID values are auto-generated in psa_crypto_driver_wrappers.h
     * ID value zero means the context is not valid or not assigned to
     * any driver (i.e. none of the driver contexts are active). */
    unsigned int MBEDTLS_PRIVATE(id);

    unsigned int MBEDTLS_PRIVATE(iv_required) : 1;
    unsigned int MBEDTLS_PRIVATE(iv_set) : 1;

    uint8_t MBEDTLS_PRIVATE(default_iv_length);

    psa_driver_cipher_context_t MBEDTLS_PRIVATE(ctx);
};

#define PSA_CIPHER_OPERATION_INIT { 0, 0, 0, 0, { 0 } }
static inline struct psa_cipher_operation_s psa_cipher_operation_init(void)
{
    const struct psa_cipher_operation_s v = PSA_CIPHER_OPERATION_INIT;
    return v;
}

/* Include the context definition for the compiled-in drivers for the composite
 * algorithms. */
#include "psa/crypto_driver_contexts_composites.h"

struct psa_mac_operation_s {
    /** Unique ID indicating which driver got assigned to do the
     * operation. Since driver contexts are driver-specific, swapping
     * drivers halfway through the operation is not supported.
     * ID values are auto-generated in psa_driver_wrappers.h
     * ID value zero means the context is not valid or not assigned to
     * any driver (i.e. none of the driver contexts are active). */
    unsigned int MBEDTLS_PRIVATE(id);
    uint8_t MBEDTLS_PRIVATE(mac_size);
    unsigned int MBEDTLS_PRIVATE(is_sign) : 1;
    psa_driver_mac_context_t MBEDTLS_PRIVATE(ctx);
};

#define PSA_MAC_OPERATION_INIT { 0, 0, 0, { 0 } }
static inline struct psa_mac_operation_s psa_mac_operation_init(void)
{
    const struct psa_mac_operation_s v = PSA_MAC_OPERATION_INIT;
    return v;
}

struct psa_aead_operation_s {

    /** Unique ID indicating which driver got assigned to do the
     * operation. Since driver contexts are driver-specific, swapping
     * drivers halfway through the operation is not supported.
     * ID values are auto-generated in psa_crypto_driver_wrappers.h
     * ID value zero means the context is not valid or not assigned to
     * any driver (i.e. none of the driver contexts are active). */
    unsigned int MBEDTLS_PRIVATE(id);

    psa_algorithm_t MBEDTLS_PRIVATE(alg);
    psa_key_type_t MBEDTLS_PRIVATE(key_type);

    size_t MBEDTLS_PRIVATE(ad_remaining);
    size_t MBEDTLS_PRIVATE(body_remaining);

    unsigned int MBEDTLS_PRIVATE(nonce_set) : 1;
    unsigned int MBEDTLS_PRIVATE(lengths_set) : 1;
    unsigned int MBEDTLS_PRIVATE(ad_started) : 1;
    unsigned int MBEDTLS_PRIVATE(body_started) : 1;
    unsigned int MBEDTLS_PRIVATE(is_encrypt) : 1;

    psa_driver_aead_context_t MBEDTLS_PRIVATE(ctx);
};

#define PSA_AEAD_OPERATION_INIT { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, { 0 } }
static inline struct psa_aead_operation_s psa_aead_operation_init(void)
{
    const struct psa_aead_operation_s v = PSA_AEAD_OPERATION_INIT;
    return v;
}

/* Include the context definition for the compiled-in drivers for the key
 * derivation algorithms. */
#include "psa/crypto_driver_contexts_key_derivation.h"

struct psa_key_derivation_s {
    psa_algorithm_t MBEDTLS_PRIVATE(alg);
    unsigned int MBEDTLS_PRIVATE(can_output_key) : 1;
    size_t MBEDTLS_PRIVATE(capacity);
    psa_driver_key_derivation_context_t MBEDTLS_PRIVATE(ctx);
};

/* This only zeroes out the first byte in the union, the rest is unspecified. */
#define PSA_KEY_DERIVATION_OPERATION_INIT { 0, 0, 0, { 0 } }
static inline struct psa_key_derivation_s psa_key_derivation_operation_init(
    void)
{
    const struct psa_key_derivation_s v = PSA_KEY_DERIVATION_OPERATION_INIT;
    return v;
}

struct psa_key_policy_s {
    psa_key_usage_t MBEDTLS_PRIVATE(usage);
    psa_algorithm_t MBEDTLS_PRIVATE(alg);
    psa_algorithm_t MBEDTLS_PRIVATE(alg2);
};
typedef struct psa_key_policy_s psa_key_policy_t;

#define PSA_KEY_POLICY_INIT { 0, 0, 0 }
static inline struct psa_key_policy_s psa_key_policy_init(void)
{
    const struct psa_key_policy_s v = PSA_KEY_POLICY_INIT;
    return v;
}

/* The type used internally for key sizes.
 * Public interfaces use size_t, but internally we use a smaller type. */
typedef uint16_t psa_key_bits_t;
/* The maximum value of the type used to represent bit-sizes.
 * This is used to mark an invalid key size. */
#define PSA_KEY_BITS_TOO_LARGE          ((psa_key_bits_t) -1)
/* The maximum size of a key in bits.
 * Currently defined as the maximum that can be represented, rounded down
 * to a whole number of bytes.
 * This is an uncast value so that it can be used in preprocessor
 * conditionals. */
#define PSA_MAX_KEY_BITS 0xfff8

/** A mask of flags that can be stored in key attributes.
 *
 * This type is also used internally to store flags in slots. Internal
 * flags are defined in library/psa_crypto_core.h. Internal flags may have
 * the same value as external flags if they are properly handled during
 * key creation and in psa_get_key_attributes.
 */
typedef uint16_t psa_key_attributes_flag_t;

#define MBEDTLS_PSA_KA_FLAG_HAS_SLOT_NUMBER     \
    ((psa_key_attributes_flag_t) 0x0001)

/* A mask of key attribute flags used externally only.
 * Only meant for internal checks inside the library. */
#define MBEDTLS_PSA_KA_MASK_EXTERNAL_ONLY (      \
        MBEDTLS_PSA_KA_FLAG_HAS_SLOT_NUMBER |    \
        0)

/* A mask of key attribute flags used both internally and externally.
 * Currently there aren't any. */
#define MBEDTLS_PSA_KA_MASK_DUAL_USE (          \
        0)

typedef struct {
    psa_key_type_t MBEDTLS_PRIVATE(type);
    psa_key_bits_t MBEDTLS_PRIVATE(bits);
    psa_key_lifetime_t MBEDTLS_PRIVATE(lifetime);
    mbedtls_svc_key_id_t MBEDTLS_PRIVATE(id);
    psa_key_policy_t MBEDTLS_PRIVATE(policy);
    psa_key_attributes_flag_t MBEDTLS_PRIVATE(flags);
} psa_core_key_attributes_t;

#define PSA_CORE_KEY_ATTRIBUTES_INIT { PSA_KEY_TYPE_NONE, 0,            \
                                       PSA_KEY_LIFETIME_VOLATILE,       \
                                       MBEDTLS_SVC_KEY_ID_INIT,         \
                                       PSA_KEY_POLICY_INIT, 0 }

struct psa_key_attributes_s {
    psa_core_key_attributes_t MBEDTLS_PRIVATE(core);
#if defined(MBEDTLS_PSA_CRYPTO_SE_C)
    psa_key_slot_number_t MBEDTLS_PRIVATE(slot_number);
#endif /* MBEDTLS_PSA_CRYPTO_SE_C */
    void *MBEDTLS_PRIVATE(domain_parameters);
    size_t MBEDTLS_PRIVATE(domain_parameters_size);
};

#if defined(MBEDTLS_PSA_CRYPTO_SE_C)
#define PSA_KEY_ATTRIBUTES_INIT { PSA_CORE_KEY_ATTRIBUTES_INIT, 0, NULL, 0 }
#else
#define PSA_KEY_ATTRIBUTES_INIT { PSA_CORE_KEY_ATTRIBUTES_INIT, NULL, 0 }
#endif

static inline struct psa_key_attributes_s psa_key_attributes_init(void)
{
    const struct psa_key_attributes_s v = PSA_KEY_ATTRIBUTES_INIT;
    return v;
}

static inline void psa_set_key_id(psa_key_attributes_t *attributes,
                                  mbedtls_svc_key_id_t key)
{
    psa_key_lifetime_t lifetime = attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(lifetime);

    attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(id) = key;

    if (PSA_KEY_LIFETIME_IS_VOLATILE(lifetime)) {
        attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(lifetime) =
            PSA_KEY_LIFETIME_FROM_PERSISTENCE_AND_LOCATION(
                PSA_KEY_LIFETIME_PERSISTENT,
                PSA_KEY_LIFETIME_GET_LOCATION(lifetime));
    }
}

static inline mbedtls_svc_key_id_t psa_get_key_id(
    const psa_key_attributes_t *attributes)
{
    return attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(id);
}

#ifdef MBEDTLS_PSA_CRYPTO_KEY_ID_ENCODES_OWNER
static inline void mbedtls_set_key_owner_id(psa_key_attributes_t *attributes,
                                            mbedtls_key_owner_id_t owner)
{
    attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(id).MBEDTLS_PRIVATE(owner) = owner;
}
#endif

static inline void psa_set_key_lifetime(psa_key_attributes_t *attributes,
                                        psa_key_lifetime_t lifetime)
{
    attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(lifetime) = lifetime;
    if (PSA_KEY_LIFETIME_IS_VOLATILE(lifetime)) {
#ifdef MBEDTLS_PSA_CRYPTO_KEY_ID_ENCODES_OWNER
        attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(id).MBEDTLS_PRIVATE(key_id) = 0;
#else
        attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(id) = 0;
#endif
    }
}

static inline psa_key_lifetime_t psa_get_key_lifetime(
    const psa_key_attributes_t *attributes)
{
    return attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(lifetime);
}

static inline void psa_extend_key_usage_flags(psa_key_usage_t *usage_flags)
{
    if (*usage_flags & PSA_KEY_USAGE_SIGN_HASH) {
        *usage_flags |= PSA_KEY_USAGE_SIGN_MESSAGE;
    }

    if (*usage_flags & PSA_KEY_USAGE_VERIFY_HASH) {
        *usage_flags |= PSA_KEY_USAGE_VERIFY_MESSAGE;
    }
}

static inline void psa_set_key_usage_flags(psa_key_attributes_t *attributes,
                                           psa_key_usage_t usage_flags)
{
    psa_extend_key_usage_flags(&usage_flags);
    attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(policy).MBEDTLS_PRIVATE(usage) = usage_flags;
}

static inline psa_key_usage_t psa_get_key_usage_flags(
    const psa_key_attributes_t *attributes)
{
    return attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(policy).MBEDTLS_PRIVATE(usage);
}

static inline void psa_set_key_algorithm(psa_key_attributes_t *attributes,
                                         psa_algorithm_t alg)
{
    attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(policy).MBEDTLS_PRIVATE(alg) = alg;
}

static inline psa_algorithm_t psa_get_key_algorithm(
    const psa_key_attributes_t *attributes)
{
    return attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(policy).MBEDTLS_PRIVATE(alg);
}

/* This function is declared in crypto_extra.h, which comes after this
 * header file, but we need the function here, so repeat the declaration. */
psa_status_t psa_set_key_domain_parameters(psa_key_attributes_t *attributes,
                                           psa_key_type_t type,
                                           const uint8_t *data,
                                           size_t data_length);

static inline void psa_set_key_type(psa_key_attributes_t *attributes,
                                    psa_key_type_t type)
{
    if (attributes->MBEDTLS_PRIVATE(domain_parameters) == NULL) {
        /* Common case: quick path */
        attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(type) = type;
    } else {
        /* Call the bigger function to free the old domain parameters.
         * Ignore any errors which may arise due to type requiring
         * non-default domain parameters, since this function can't
         * report errors. */
        (void) psa_set_key_domain_parameters(attributes, type, NULL, 0);
    }
}

static inline psa_key_type_t psa_get_key_type(
    const psa_key_attributes_t *attributes)
{
    return attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(type);
}

static inline void psa_set_key_bits(psa_key_attributes_t *attributes,
                                    size_t bits)
{
    if (bits > PSA_MAX_KEY_BITS) {
        attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(bits) = PSA_KEY_BITS_TOO_LARGE;
    } else {
        attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(bits) = (psa_key_bits_t) bits;
    }
}

static inline size_t psa_get_key_bits(
    const psa_key_attributes_t *attributes)
{
    return attributes->MBEDTLS_PRIVATE(core).MBEDTLS_PRIVATE(bits);
}

/**
 * \brief The context for PSA interruptible hash signing.
 */
struct psa_sign_hash_interruptible_operation_s {
    /** Unique ID indicating which driver got assigned to do the
     * operation. Since driver contexts are driver-specific, swapping
     * drivers halfway through the operation is not supported.
     * ID values are auto-generated in psa_crypto_driver_wrappers.h
     * ID value zero means the context is not valid or not assigned to
     * any driver (i.e. none of the driver contexts are active). */
    unsigned int MBEDTLS_PRIVATE(id);

    psa_driver_sign_hash_interruptible_context_t MBEDTLS_PRIVATE(ctx);

    unsigned int MBEDTLS_PRIVATE(error_occurred) : 1;

    uint32_t MBEDTLS_PRIVATE(num_ops);
};

#define PSA_SIGN_HASH_INTERRUPTIBLE_OPERATION_INIT { 0, { 0 }, 0, 0 }

static inline struct psa_sign_hash_interruptible_operation_s
psa_sign_hash_interruptible_operation_init(void)
{
    const struct psa_sign_hash_interruptible_operation_s v =
        PSA_SIGN_HASH_INTERRUPTIBLE_OPERATION_INIT;

    return v;
}

/**
 * \brief The context for PSA interruptible hash verification.
 */
struct psa_verify_hash_interruptible_operation_s {
    /** Unique ID indicating which driver got assigned to do the
     * operation. Since driver contexts are driver-specific, swapping
     * drivers halfway through the operation is not supported.
     * ID values are auto-generated in psa_crypto_driver_wrappers.h
     * ID value zero means the context is not valid or not assigned to
     * any driver (i.e. none of the driver contexts are active). */
    unsigned int MBEDTLS_PRIVATE(id);

    psa_driver_verify_hash_interruptible_context_t MBEDTLS_PRIVATE(ctx);

    unsigned int MBEDTLS_PRIVATE(error_occurred) : 1;

    uint32_t MBEDTLS_PRIVATE(num_ops);
};

#define PSA_VERIFY_HASH_INTERRUPTIBLE_OPERATION_INIT { 0, { 0 }, 0, 0 }

static inline struct psa_verify_hash_interruptible_operation_s
psa_verify_hash_interruptible_operation_init(void)
{
    const struct psa_verify_hash_interruptible_operation_s v =
        PSA_VERIFY_HASH_INTERRUPTIBLE_OPERATION_INIT;

    return v;
}

#ifdef __cplusplus
}
#endif

#endif /* PSA_CRYPTO_STRUCT_H */
