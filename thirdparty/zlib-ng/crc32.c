/* crc32.c -- compute the CRC-32 of a data stream
 * Copyright (C) 1995-2022 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * This interleaved implementation of a CRC makes use of pipelined multiple
 * arithmetic-logic units, commonly found in modern CPU cores. It is due to
 * Kadatch and Jenkins (2010). See doc/crc-doc.1.0.pdf in this distribution.
 */

#include "zbuild.h"
#include "functable.h"
#include "crc32_braid_tbl.h"

/* ========================================================================= */

const uint32_t * Z_EXPORT PREFIX(get_crc_table)(void) {
    return (const uint32_t *)crc_table;
}

#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(crc32_z)(unsigned long crc, const unsigned char *buf, size_t len) {
    if (buf == NULL) return 0;

    return (unsigned long)FUNCTABLE_CALL(crc32)((uint32_t)crc, buf, len);
}
#else
uint32_t Z_EXPORT PREFIX(crc32_z)(uint32_t crc, const unsigned char *buf, size_t len) {
    if (buf == NULL) return 0;

    return FUNCTABLE_CALL(crc32)(crc, buf, len);
}
#endif

#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(crc32)(unsigned long crc, const unsigned char *buf, unsigned int len) {
    return (unsigned long)PREFIX(crc32_z)((uint32_t)crc, buf, len);
}
#else
uint32_t Z_EXPORT PREFIX(crc32)(uint32_t crc, const unsigned char *buf, uint32_t len) {
    return PREFIX(crc32_z)(crc, buf, len);
}
#endif
