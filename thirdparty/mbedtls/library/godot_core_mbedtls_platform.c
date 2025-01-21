#if !defined(MBEDTLS_CONFIG_FILE)
#include "mbedtls/config.h"
#else
#include MBEDTLS_CONFIG_FILE
#endif

#include <stdio.h>
#include <string.h>
#include <stddef.h>

#ifdef MBEDTLS_PLATFORM_ZEROIZE_ALT
static void *(*const volatile memset_func)(void *, int, size_t) = memset;

void mbedtls_platform_zeroize(void *buf, size_t len) {
    memset_func( buf, 0, len );
}
#endif
