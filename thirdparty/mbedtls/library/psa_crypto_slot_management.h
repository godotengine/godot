/*
 *  PSA crypto layer on top of Mbed TLS crypto
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#ifndef PSA_CRYPTO_SLOT_MANAGEMENT_H
#define PSA_CRYPTO_SLOT_MANAGEMENT_H

#include "psa/crypto.h"
#include "psa_crypto_core.h"
#include "psa_crypto_se.h"

/** Range of volatile key identifiers.
 *
 *  The last #MBEDTLS_PSA_KEY_SLOT_COUNT identifiers of the implementation
 *  range of key identifiers are reserved for volatile key identifiers.
 *  A volatile key identifier is equal to #PSA_KEY_ID_VOLATILE_MIN plus the
 *  index of the key slot containing the volatile key definition.
 */

/** The minimum value for a volatile key identifier.
 */
#define PSA_KEY_ID_VOLATILE_MIN  (PSA_KEY_ID_VENDOR_MAX - \
                                  MBEDTLS_PSA_KEY_SLOT_COUNT + 1)

/** The maximum value for a volatile key identifier.
 */
#define PSA_KEY_ID_VOLATILE_MAX  PSA_KEY_ID_VENDOR_MAX

/** Test whether a key identifier is a volatile key identifier.
 *
 * \param key_id  Key identifier to test.
 *
 * \retval 1
 *         The key identifier is a volatile key identifier.
 * \retval 0
 *         The key identifier is not a volatile key identifier.
 */
static inline int psa_key_id_is_volatile(psa_key_id_t key_id)
{
    return (key_id >= PSA_KEY_ID_VOLATILE_MIN) &&
           (key_id <= PSA_KEY_ID_VOLATILE_MAX);
}

/** Get the description of a key given its identifier and lock it.
 *
 * The descriptions of volatile keys and loaded persistent keys are stored in
 * key slots. This function returns a pointer to the key slot containing the
 * description of a key given its identifier.
 *
 * In case of a persistent key, the function loads the description of the key
 * into a key slot if not already done.
 *
 * On success, the returned key slot is locked. It is the responsibility of
 * the caller to unlock the key slot when it does not access it anymore.
 *
 * \param key           Key identifier to query.
 * \param[out] p_slot   On success, `*p_slot` contains a pointer to the
 *                      key slot containing the description of the key
 *                      identified by \p key.
 *
 * \retval #PSA_SUCCESS
 *         \p *p_slot contains a pointer to the key slot containing the
 *         description of the key identified by \p key.
 *         The key slot counter has been incremented.
 * \retval #PSA_ERROR_BAD_STATE
 *         The library has not been initialized.
 * \retval #PSA_ERROR_INVALID_HANDLE
 *         \p key is not a valid key identifier.
 * \retval #PSA_ERROR_INSUFFICIENT_MEMORY
 *         \p key is a persistent key identifier. The implementation does not
 *         have sufficient resources to load the persistent key. This can be
 *         due to a lack of empty key slot, or available memory.
 * \retval #PSA_ERROR_DOES_NOT_EXIST
 *         There is no key with key identifier \p key.
 * \retval #PSA_ERROR_CORRUPTION_DETECTED \emptydescription
 * \retval #PSA_ERROR_STORAGE_FAILURE \emptydescription
 * \retval #PSA_ERROR_DATA_CORRUPT \emptydescription
 */
psa_status_t psa_get_and_lock_key_slot(mbedtls_svc_key_id_t key,
                                       psa_key_slot_t **p_slot);

/** Initialize the key slot structures.
 *
 * \retval #PSA_SUCCESS
 *         Currently this function always succeeds.
 */
psa_status_t psa_initialize_key_slots(void);

/** Delete all data from key slots in memory.
 *
 * This does not affect persistent storage. */
void psa_wipe_all_key_slots(void);

/** Find a free key slot.
 *
 * This function returns a key slot that is available for use and is in its
 * ground state (all-bits-zero). On success, the key slot is locked. It is
 * the responsibility of the caller to unlock the key slot when it does not
 * access it anymore.
 *
 * \param[out] volatile_key_id   On success, volatile key identifier
 *                               associated to the returned slot.
 * \param[out] p_slot            On success, a pointer to the slot.
 *
 * \retval #PSA_SUCCESS \emptydescription
 * \retval #PSA_ERROR_INSUFFICIENT_MEMORY \emptydescription
 * \retval #PSA_ERROR_BAD_STATE \emptydescription
 */
psa_status_t psa_get_empty_key_slot(psa_key_id_t *volatile_key_id,
                                    psa_key_slot_t **p_slot);

/** Lock a key slot.
 *
 * This function increments the key slot lock counter by one.
 *
 * \param[in] slot  The key slot.
 *
 * \retval #PSA_SUCCESS
               The key slot lock counter was incremented.
 * \retval #PSA_ERROR_CORRUPTION_DETECTED
 *             The lock counter already reached its maximum value and was not
 *             increased.
 */
static inline psa_status_t psa_lock_key_slot(psa_key_slot_t *slot)
{
    if (slot->lock_count >= SIZE_MAX) {
        return PSA_ERROR_CORRUPTION_DETECTED;
    }

    slot->lock_count++;

    return PSA_SUCCESS;
}

/** Unlock a key slot.
 *
 * This function decrements the key slot lock counter by one.
 *
 * \note To ease the handling of errors in retrieving a key slot
 *       a NULL input pointer is valid, and the function returns
 *       successfully without doing anything in that case.
 *
 * \param[in] slot  The key slot.
 * \retval #PSA_SUCCESS
 *             \p slot is NULL or the key slot lock counter has been
 *             decremented successfully.
 * \retval #PSA_ERROR_CORRUPTION_DETECTED
 *             The lock counter was equal to 0.
 *
 */
psa_status_t psa_unlock_key_slot(psa_key_slot_t *slot);

/** Test whether a lifetime designates a key in an external cryptoprocessor.
 *
 * \param lifetime      The lifetime to test.
 *
 * \retval 1
 *         The lifetime designates an external key. There should be a
 *         registered driver for this lifetime, otherwise the key cannot
 *         be created or manipulated.
 * \retval 0
 *         The lifetime designates a key that is volatile or in internal
 *         storage.
 */
static inline int psa_key_lifetime_is_external(psa_key_lifetime_t lifetime)
{
    return PSA_KEY_LIFETIME_GET_LOCATION(lifetime)
           != PSA_KEY_LOCATION_LOCAL_STORAGE;
}

/** Validate a key's location.
 *
 * This function checks whether the key's attributes point to a location that
 * is known to the PSA Core, and returns the driver function table if the key
 * is to be found in an external location.
 *
 * \param[in] lifetime      The key lifetime attribute.
 * \param[out] p_drv        On success, when a key is located in external
 *                          storage, returns a pointer to the driver table
 *                          associated with the key's storage location.
 *
 * \retval #PSA_SUCCESS \emptydescription
 * \retval #PSA_ERROR_INVALID_ARGUMENT \emptydescription
 */
psa_status_t psa_validate_key_location(psa_key_lifetime_t lifetime,
                                       psa_se_drv_table_entry_t **p_drv);

/** Validate the persistence of a key.
 *
 * \param[in] lifetime  The key lifetime attribute.
 *
 * \retval #PSA_SUCCESS \emptydescription
 * \retval #PSA_ERROR_NOT_SUPPORTED The key is persistent but persistent keys
 *             are not supported.
 */
psa_status_t psa_validate_key_persistence(psa_key_lifetime_t lifetime);

/** Validate a key identifier.
 *
 * \param[in] key           The key identifier.
 * \param[in] vendor_ok     Non-zero to indicate that key identifiers in the
 *                          vendor range are allowed, volatile key identifiers
 *                          excepted \c 0 otherwise.
 *
 * \retval <> 0 if the key identifier is valid, 0 otherwise.
 */
int psa_is_valid_key_id(mbedtls_svc_key_id_t key, int vendor_ok);

#endif /* PSA_CRYPTO_SLOT_MANAGEMENT_H */
