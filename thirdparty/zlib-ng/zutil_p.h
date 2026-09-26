/* zutil_p.h -- Private inline functions used internally in zlib-ng
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef ZUTIL_P_H
#define ZUTIL_P_H

#include <stdlib.h>

// Zlib-ng's default alloc/free implementation, used unless
// application supplies its own alloc/free functions.
static inline void *zng_alloc(size_t size) {
    return (void *)malloc(size);
}

static inline void zng_free(void *ptr) {
    free(ptr);
}

#endif
