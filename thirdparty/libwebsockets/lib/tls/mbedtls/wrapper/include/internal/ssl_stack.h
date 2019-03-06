#ifndef _SSL_STACK_H_
#define _SSL_STACK_H_

#ifdef __cplusplus
 extern "C" {
#endif

#include "ssl_types.h"

#define STACK_OF(type)  struct stack_st_##type

#define SKM_DEFINE_STACK_OF(t1, t2, t3) \
    STACK_OF(t1); \
    static ossl_inline STACK_OF(t1) *sk_##t1##_new_null(void) \
    { \
        return (STACK_OF(t1) *)OPENSSL_sk_new_null(); \
    } \

#define DEFINE_STACK_OF(t) SKM_DEFINE_STACK_OF(t, t, t)

/**
 * @brief create a openssl stack object
 *
 * @param c - stack function
 *
 * @return openssl stack object point
 */
OPENSSL_STACK* OPENSSL_sk_new(OPENSSL_sk_compfunc c);

/**
 * @brief create a NULL function openssl stack object
 *
 * @param none
 *
 * @return openssl stack object point
 */
OPENSSL_STACK *OPENSSL_sk_new_null(void);

/**
 * @brief free openssl stack object
 *
 * @param openssl stack object point
 *
 * @return none
 */
void OPENSSL_sk_free(OPENSSL_STACK *stack);

#ifdef __cplusplus
}
#endif

#endif
