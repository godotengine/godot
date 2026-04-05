/**
 * \file threading.h
 *
 * \brief Threading abstraction layer
 */
/*
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
#ifndef MBEDTLS_THREADING_H
#define MBEDTLS_THREADING_H
#include "mbedtls/private_access.h"

#include "mbedtls/build_info.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Bad input parameters to function. */
#define MBEDTLS_ERR_THREADING_BAD_INPUT_DATA              -0x001C
/** Locking / unlocking / free failed with error code. */
#define MBEDTLS_ERR_THREADING_MUTEX_ERROR                 -0x001E

#if defined(MBEDTLS_THREADING_PTHREAD)
#include <pthread.h>
typedef struct mbedtls_threading_mutex_t {
    pthread_mutex_t MBEDTLS_PRIVATE(mutex);

    /* WARNING - state should only be accessed when holding the mutex lock in
     * framework/tests/src/threading_helpers.c, otherwise corruption can occur.
     * state will be 0 after a failed init or a free, and nonzero after a
     * successful init. This field is for testing only and thus not considered
     * part of the public API of Mbed TLS and may change without notice.*/
    char MBEDTLS_PRIVATE(state);

} mbedtls_threading_mutex_t;
#endif

#if defined(MBEDTLS_THREADING_ALT)
/* You should define the mbedtls_threading_mutex_t type in your header */
#include "threading_alt.h"

/**
 * \brief           Set your alternate threading implementation function
 *                  pointers and initialize global mutexes. If used, this
 *                  function must be called once in the main thread before any
 *                  other Mbed TLS function is called, and
 *                  mbedtls_threading_free_alt() must be called once in the main
 *                  thread after all other Mbed TLS functions.
 *
 * \warning         \p mutex_init and \p mutex_free don't return a status code.
 *                  If \p mutex_init fails, it should leave the mutex in
 *                  a state such that \p mutex_lock will reliably return
 *                  #MBEDTLS_ERR_THREADING_MUTEX_ERROR called on this mutex,
 *                  and \p mutex_free will do nothing.
 *
 * \param mutex_init    The init function implementation. <br>
 *                      The behavior is undefined if the mutex is already
 *                      initialized and has not been destroyed.
 *                      On platforms where mutex initialization can fail,
 *                      since this function does not return a status code,
 *                      it must leave the mutex object in a safe state where
 *                      subsequent function calls will not cause undefined
 *                      behavior: after a call to \p mutex_init, the
 *                      function \p mutex_lock must either succeed or
 *                      fail with a nonzero status code, and the function
 *                      \p mutex_free must free any resources associated
 *                      with the mutex..
 * \param mutex_free    The destroy function implementation. <br>
 *                      This function must free any resources associated
 *                      with the mutex object. <br>
 *                      This function must work reliably if \p mutex_init
 *                      has been called on the mutex and \p mutex_free
 *                      has not yet been called. <br>
 *                      The behavior is undefined if the mutex was not
 *                      initialized, if it has already been destroyed,
 *                      if it is currently locked, or if this function
 *                      is called concurrently from multiple threads.
 * \param mutex_lock    The lock function implementation. <br>
 *                      This function must work reliably on any mutex
 *                      which is not currently locked and on which
 *                      \p mutex_init has already been called but
 *                      \p mutex_free has not been called yet. <br>
 *                      The behavior is undefined if the mutex was not
 *                      initialized, if it has already been destroyed, or if
 *                      it is currently locked by the calling thread.
 * \param mutex_unlock  The unlock function implementation. <br>
 *                      The behavior is undefined if the mutex is not
 *                      currently locked by the calling thread.
 */
void mbedtls_threading_set_alt(void (*mutex_init)(mbedtls_threading_mutex_t *),
                               void (*mutex_free)(mbedtls_threading_mutex_t *),
                               int (*mutex_lock)(mbedtls_threading_mutex_t *),
                               int (*mutex_unlock)(mbedtls_threading_mutex_t *));

/**
 * \brief               Free global mutexes.
 */
void mbedtls_threading_free_alt(void);
#endif /* MBEDTLS_THREADING_ALT */

#if defined(MBEDTLS_THREADING_C)
/*
 * The function pointers for mutex_init, mutex_free, mutex_ and mutex_unlock
 *
 * All these functions are expected to work or the result will be undefined.
 */
extern void (*mbedtls_mutex_init)(mbedtls_threading_mutex_t *mutex);
extern void (*mbedtls_mutex_free)(mbedtls_threading_mutex_t *mutex);
extern int (*mbedtls_mutex_lock)(mbedtls_threading_mutex_t *mutex);
extern int (*mbedtls_mutex_unlock)(mbedtls_threading_mutex_t *mutex);

/*
 * Global mutexes
 */
#if defined(MBEDTLS_FS_IO)
extern mbedtls_threading_mutex_t mbedtls_threading_readdir_mutex;
#endif

#if defined(MBEDTLS_HAVE_TIME_DATE) && !defined(MBEDTLS_PLATFORM_GMTIME_R_ALT)
/* This mutex may or may not be used in the default definition of
 * mbedtls_platform_gmtime_r(), but in order to determine that,
 * we need to check POSIX features, hence modify _POSIX_C_SOURCE.
 * With the current approach, this declaration is orphaned, lacking
 * an accompanying definition, in case mbedtls_platform_gmtime_r()
 * doesn't need it, but that's not a problem. */
extern mbedtls_threading_mutex_t mbedtls_threading_gmtime_mutex;
#endif /* MBEDTLS_HAVE_TIME_DATE && !MBEDTLS_PLATFORM_GMTIME_R_ALT */

#if defined(MBEDTLS_PSA_CRYPTO_C)
/*
 * A mutex used to make the PSA subsystem thread safe.
 *
 * key_slot_mutex protects the registered_readers and
 * state variable for all key slots in &global_data.key_slots.
 *
 * This mutex must be held when any read from or write to a state or
 * registered_readers field is performed, i.e. when calling functions:
 * psa_key_slot_state_transition(), psa_register_read(), psa_unregister_read(),
 * psa_key_slot_has_readers() and psa_wipe_key_slot(). */
extern mbedtls_threading_mutex_t mbedtls_threading_key_slot_mutex;

/*
 * A mutex used to make the non-rng PSA global_data struct members thread safe.
 *
 * This mutex must be held when reading or writing to any of the PSA global_data
 * structure members, other than the rng_state or rng struct. */
extern mbedtls_threading_mutex_t mbedtls_threading_psa_globaldata_mutex;

/*
 * A mutex used to make the PSA global_data rng data thread safe.
 *
 * This mutex must be held when reading or writing to the PSA
 * global_data rng_state or rng struct members. */
extern mbedtls_threading_mutex_t mbedtls_threading_psa_rngdata_mutex;
#endif

#endif /* MBEDTLS_THREADING_C */

#ifdef __cplusplus
}
#endif

#endif /* threading.h */
