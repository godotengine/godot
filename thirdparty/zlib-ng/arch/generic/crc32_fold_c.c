/* crc32_fold.c -- crc32 folding interface
 * Copyright (C) 2021 Nathan Moinvaziri
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#include "zbuild.h"
#include "zutil.h"
#include "functable.h"
#include "crc32.h"

Z_INTERNAL uint32_t crc32_fold_reset_c(crc32_fold *crc) {
    crc->value = CRC32_INITIAL_VALUE;
    return crc->value;
}

Z_INTERNAL void crc32_fold_copy_c(crc32_fold *crc, uint8_t *dst, const uint8_t *src, size_t len) {
    crc->value = FUNCTABLE_CALL(crc32)(crc->value, src, len);
    memcpy(dst, src, len);
}

Z_INTERNAL void crc32_fold_c(crc32_fold *crc, const uint8_t *src, size_t len, uint32_t init_crc) {
    /* Note: while this is basically the same thing as the vanilla CRC function, we still need
     * a functable entry for it so that we can generically dispatch to this function with the
     * same arguments for the versions that _do_ do a folding CRC but we don't want a copy. The
     * init_crc is an unused argument in this context */
    Z_UNUSED(init_crc);
    crc->value = FUNCTABLE_CALL(crc32)(crc->value, src, len);
}

Z_INTERNAL uint32_t crc32_fold_final_c(crc32_fold *crc) {
    return crc->value;
}
