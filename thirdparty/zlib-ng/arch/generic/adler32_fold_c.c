/* adler32_fold.c -- adler32 folding interface
 * Copyright (C) 2022 Adam Stylinski
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "functable.h"

#include <limits.h>

Z_INTERNAL uint32_t adler32_fold_copy_c(uint32_t adler, uint8_t *dst, const uint8_t *src, size_t len) {
    adler = FUNCTABLE_CALL(adler32)(adler, src, len);
    memcpy(dst, src, len);
    return adler;
}
