/* crc32_braid_comb.c -- compute the CRC-32 of a data stream
 * Copyright (C) 1995-2022 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * This interleaved implementation of a CRC makes use of pipelined multiple
 * arithmetic-logic units, commonly found in modern CPU cores. It is due to
 * Kadatch and Jenkins (2010). See doc/crc-doc.1.0.pdf in this distribution.
 */

#include "zutil.h"
#include "crc32_braid_p.h"
#include "crc32_braid_tbl.h"
#include "crc32_braid_comb_p.h"

/* ========================================================================= */
static uint32_t crc32_combine_(uint32_t crc1, uint32_t crc2, z_off64_t len2) {
    return multmodp(x2nmodp(len2, 3), crc1) ^ crc2;
}
static uint32_t crc32_combine_gen_(z_off64_t len2) {
     return x2nmodp(len2, 3);
}
static uint32_t crc32_combine_op_(uint32_t crc1, uint32_t crc2, const uint32_t op) {
    return multmodp(op, crc1) ^ crc2;
}

/* ========================================================================= */

#ifdef ZLIB_COMPAT
unsigned long Z_EXPORT PREFIX(crc32_combine)(unsigned long crc1, unsigned long crc2, z_off_t len2) {
    return (unsigned long)crc32_combine_((uint32_t)crc1, (uint32_t)crc2, len2);
}
unsigned long Z_EXPORT PREFIX4(crc32_combine)(unsigned long crc1, unsigned long crc2, z_off64_t len2) {
    return (unsigned long)crc32_combine_((uint32_t)crc1, (uint32_t)crc2, len2);
}
unsigned long Z_EXPORT PREFIX(crc32_combine_gen)(z_off_t len2) {
    return crc32_combine_gen_(len2);
}
unsigned long Z_EXPORT PREFIX4(crc32_combine_gen)(z_off64_t len2) {
    return crc32_combine_gen_(len2);
}
unsigned long Z_EXPORT PREFIX(crc32_combine_op)(unsigned long crc1, unsigned long crc2, const unsigned long op) {
    return (unsigned long)crc32_combine_op_((uint32_t)crc1, (uint32_t)crc2, (uint32_t)op);
}
#else
uint32_t Z_EXPORT PREFIX4(crc32_combine)(uint32_t crc1, uint32_t crc2, z_off64_t len2) {
    return crc32_combine_(crc1, crc2, len2);
}
uint32_t Z_EXPORT PREFIX(crc32_combine_gen)(z_off64_t len2) {
    return crc32_combine_gen_(len2);
}
uint32_t Z_EXPORT PREFIX(crc32_combine_op)(uint32_t crc1, uint32_t crc2, const uint32_t op) {
    return crc32_combine_op_(crc1, crc2, op);
}
#endif

/* ========================================================================= */
