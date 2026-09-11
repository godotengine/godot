/* crc32_braid.c -- compute the CRC-32 of a data stream
 * Copyright (C) 1995-2022 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * This interleaved implementation of a CRC makes use of pipelined multiple
 * arithmetic-logic units, commonly found in modern CPU cores. It is due to
 * Kadatch and Jenkins (2010). See doc/crc-doc.1.0.pdf in this distribution.
 */

#include "zbuild.h"
#include "crc32_braid_p.h"
#include "crc32_braid_tbl.h"

/*
  A CRC of a message is computed on BRAID_N braids of words in the message, where
  each word consists of BRAID_W bytes (4 or 8). If BRAID_N is 3, for example, then
  three running sparse CRCs are calculated respectively on each braid, at these
  indices in the array of words: 0, 3, 6, ..., 1, 4, 7, ..., and 2, 5, 8, ...
  This is done starting at a word boundary, and continues until as many blocks of
  BRAID_N * BRAID_W bytes as are available have been processed. The results are
  combined into a single CRC at the end. For this code, BRAID_N must be in the
  range 1..6 and BRAID_W must be 4 or 8. The upper limit on BRAID_N can be increased
  if desired by adding more #if blocks, extending the patterns apparent in the code.
  In addition, crc32 tables would need to be regenerated, if the maximum BRAID_N
  value is increased.

  BRAID_N and BRAID_W are chosen empirically by benchmarking the execution time
  on a given processor. The choices for BRAID_N and BRAID_W below were based on
  testing on Intel Kaby Lake i7, AMD Ryzen 7, ARM Cortex-A57, Sparc64-VII, PowerPC
  POWER9, and MIPS64 Octeon II processors.
  The Intel, AMD, and ARM processors were all fastest with BRAID_N=5, BRAID_W=8.
  The Sparc, PowerPC, and MIPS64 were all fastest at BRAID_N=5, BRAID_W=4.
  They were all tested with either gcc or clang, all using the -O3 optimization
  level. Your mileage may vary.
*/

/* ========================================================================= */
#ifdef BRAID_W
/*
  Return the CRC of the BRAID_W bytes in the word_t data, taking the
  least-significant byte of the word as the first byte of data, without any pre
  or post conditioning. This is used to combine the CRCs of each braid.
 */
#  if BYTE_ORDER == LITTLE_ENDIAN
static uint32_t crc_word(z_word_t data) {
    int k;
    for (k = 0; k < BRAID_W; k++)
        data = (data >> 8) ^ crc_table[data & 0xff];
    return (uint32_t)data;
}
#  elif BYTE_ORDER == BIG_ENDIAN
static z_word_t crc_word(z_word_t data) {
    int k;
    for (k = 0; k < BRAID_W; k++)
        data = (data << 8) ^
            crc_big_table[(data >> ((BRAID_W - 1) << 3)) & 0xff];
    return data;
}
#  endif /* BYTE_ORDER */
#endif /* BRAID_W */

/* ========================================================================= */
Z_INTERNAL uint32_t crc32_braid_internal(uint32_t c, const uint8_t *buf, size_t len) {

#ifdef BRAID_W
    /* If provided enough bytes, do a braided CRC calculation. */
    if (len >= BRAID_N * BRAID_W + BRAID_W - 1) {
        size_t blks;
        z_word_t const *words;
        int k;

        /* Compute the CRC up to a z_word_t boundary. */
        while (len && ((uintptr_t)buf & (BRAID_W - 1)) != 0) {
            len--;
            CRC_DO1;
        }

        /* Compute the CRC on as many BRAID_N z_word_t blocks as are available. */
        blks = len / (BRAID_N * BRAID_W);
        len -= blks * BRAID_N * BRAID_W;
        words = (z_word_t const *)buf;

        z_word_t crc0, word0, comb;
#if BRAID_N > 1
        z_word_t crc1, word1;
#if BRAID_N > 2
        z_word_t crc2, word2;
#if BRAID_N > 3
        z_word_t crc3, word3;
#if BRAID_N > 4
        z_word_t crc4, word4;
#if BRAID_N > 5
        z_word_t crc5, word5;
#endif
#endif
#endif
#endif
#endif
        /* Initialize the CRC for each braid. */
        crc0 = ZSWAPWORD(c);
#if BRAID_N > 1
        crc1 = 0;
#if BRAID_N > 2
        crc2 = 0;
#if BRAID_N > 3
        crc3 = 0;
#if BRAID_N > 4
        crc4 = 0;
#if BRAID_N > 5
        crc5 = 0;
#endif
#endif
#endif
#endif
#endif
        /* Process the first blks-1 blocks, computing the CRCs on each braid independently. */
        while (--blks) {
            /* Load the word for each braid into registers. */
            word0 = crc0 ^ words[0];
#if BRAID_N > 1
            word1 = crc1 ^ words[1];
#if BRAID_N > 2
            word2 = crc2 ^ words[2];
#if BRAID_N > 3
            word3 = crc3 ^ words[3];
#if BRAID_N > 4
            word4 = crc4 ^ words[4];
#if BRAID_N > 5
            word5 = crc5 ^ words[5];
#endif
#endif
#endif
#endif
#endif
            words += BRAID_N;

            /* Compute and update the CRC for each word. The loop should get unrolled. */
            crc0 = BRAID_TABLE[0][word0 & 0xff];
#if BRAID_N > 1
            crc1 = BRAID_TABLE[0][word1 & 0xff];
#if BRAID_N > 2
            crc2 = BRAID_TABLE[0][word2 & 0xff];
#if BRAID_N > 3
            crc3 = BRAID_TABLE[0][word3 & 0xff];
#if BRAID_N > 4
            crc4 = BRAID_TABLE[0][word4 & 0xff];
#if BRAID_N > 5
            crc5 = BRAID_TABLE[0][word5 & 0xff];
#endif
#endif
#endif
#endif
#endif
            for (k = 1; k < BRAID_W; k++) {
                crc0 ^= BRAID_TABLE[k][(word0 >> (k << 3)) & 0xff];
#if BRAID_N > 1
                crc1 ^= BRAID_TABLE[k][(word1 >> (k << 3)) & 0xff];
#if BRAID_N > 2
                crc2 ^= BRAID_TABLE[k][(word2 >> (k << 3)) & 0xff];
#if BRAID_N > 3
                crc3 ^= BRAID_TABLE[k][(word3 >> (k << 3)) & 0xff];
#if BRAID_N > 4
                crc4 ^= BRAID_TABLE[k][(word4 >> (k << 3)) & 0xff];
#if BRAID_N > 5
                crc5 ^= BRAID_TABLE[k][(word5 >> (k << 3)) & 0xff];
#endif
#endif
#endif
#endif
#endif
            }
        }

        /* Process the last block, combining the CRCs of the BRAID_N braids at the same time. */
        comb = crc_word(crc0 ^ words[0]);
#if BRAID_N > 1
        comb = crc_word(crc1 ^ words[1] ^ comb);
#if BRAID_N > 2
        comb = crc_word(crc2 ^ words[2] ^ comb);
#if BRAID_N > 3
        comb = crc_word(crc3 ^ words[3] ^ comb);
#if BRAID_N > 4
        comb = crc_word(crc4 ^ words[4] ^ comb);
#if BRAID_N > 5
        comb = crc_word(crc5 ^ words[5] ^ comb);
#endif
#endif
#endif
#endif
#endif
        words += BRAID_N;
        Assert(comb <= UINT32_MAX, "comb should fit in uint32_t");
        c = (uint32_t)ZSWAPWORD(comb);

        /* Update the pointer to the remaining bytes to process. */
        buf = (const unsigned char *)words;
    }

#endif /* BRAID_W */

    /* Complete the computation of the CRC on any remaining bytes. */
    while (len >= 8) {
        len -= 8;
        CRC_DO8;
    }
    while (len) {
        len--;
        CRC_DO1;
    }

    /* Return the CRC, post-conditioned. */
    return c;
}

Z_INTERNAL uint32_t crc32_braid(uint32_t c, const uint8_t *buf, size_t len) {
    c = (~c) & 0xffffffff;

    c = crc32_braid_internal(c, buf, len);

    /* Return the CRC, post-conditioned. */
    return c ^ 0xffffffff;
}
