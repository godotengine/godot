/**
 *  \brief HAVEGE: HArdware Volatile Entropy Gathering and Expansion
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */
/*
 *  The HAVEGE RNG was designed by Andre Seznec in 2002.
 *
 *  http://www.irisa.fr/caps/projects/hipsor/publi.php
 *
 *  Contact: seznec(at)irisa_dot_fr - orocheco(at)irisa_dot_fr
 */

#include "common.h"

#if defined(MBEDTLS_HAVEGE_C)

#include "mbedtls/havege.h"
#include "mbedtls/timing.h"
#include "mbedtls/platform_util.h"

#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * On average, one iteration accesses two 8-word blocks in the havege WALK
 * table, and generates 16 words in the RES array.
 *
 * The data read in the WALK table is updated and permuted after each use.
 * The result of the hardware clock counter read is used  for this update.
 *
 * 25 conditional tests are present.  The conditional tests are grouped in
 * two nested  groups of 12 conditional tests and 1 test that controls the
 * permutation; on average, there should be 6 tests executed and 3 of them
 * should be mispredicted.
 * ------------------------------------------------------------------------
 */

#define SWAP(X, Y) { uint32_t *T = (X); (X) = (Y); (Y) = T; }

#define TST1_ENTER if (PTEST & 1) { PTEST ^= 3; PTEST >>= 1;
#define TST2_ENTER if (PTEST & 1) { PTEST ^= 3; PTEST >>= 1;

#define TST1_LEAVE U1++; }
#define TST2_LEAVE U2++; }

#define ONE_ITERATION                                   \
                                                        \
    PTEST = PT1 >> 20;                                  \
                                                        \
    TST1_ENTER  TST1_ENTER  TST1_ENTER  TST1_ENTER      \
    TST1_ENTER  TST1_ENTER  TST1_ENTER  TST1_ENTER      \
    TST1_ENTER  TST1_ENTER  TST1_ENTER  TST1_ENTER      \
                                                        \
    TST1_LEAVE  TST1_LEAVE  TST1_LEAVE  TST1_LEAVE      \
    TST1_LEAVE  TST1_LEAVE  TST1_LEAVE  TST1_LEAVE      \
    TST1_LEAVE  TST1_LEAVE  TST1_LEAVE  TST1_LEAVE      \
                                                        \
        PTX = (PT1 >> 18) & 7;                              \
    PT1 &= 0x1FFF;                                      \
    PT2 &= 0x1FFF;                                      \
    CLK = (uint32_t) mbedtls_timing_hardclock();        \
                                                        \
    i = 0;                                              \
    A = &WALK[PT1]; RES[i++] ^= *A;                 \
    B = &WALK[PT2]; RES[i++] ^= *B;                 \
    C = &WALK[PT1 ^ 1]; RES[i++] ^= *C;                 \
    D = &WALK[PT2 ^ 4]; RES[i++] ^= *D;                 \
                                                        \
    IN = (*A >> (1)) ^ (*A << (31)) ^ CLK;              \
    *A = (*B >> (2)) ^ (*B << (30)) ^ CLK;              \
    *B = IN ^ U1;                                       \
    *C = (*C >> (3)) ^ (*C << (29)) ^ CLK;              \
    *D = (*D >> (4)) ^ (*D << (28)) ^ CLK;              \
                                                        \
    A = &WALK[PT1 ^ 2]; RES[i++] ^= *A;                 \
    B = &WALK[PT2 ^ 2]; RES[i++] ^= *B;                 \
    C = &WALK[PT1 ^ 3]; RES[i++] ^= *C;                 \
    D = &WALK[PT2 ^ 6]; RES[i++] ^= *D;                 \
                                                        \
    if (PTEST & 1) SWAP(A, C);                       \
                                                        \
    IN = (*A >> (5)) ^ (*A << (27)) ^ CLK;              \
    *A = (*B >> (6)) ^ (*B << (26)) ^ CLK;              \
    *B = IN; CLK = (uint32_t) mbedtls_timing_hardclock();       \
    *C = (*C >> (7)) ^ (*C << (25)) ^ CLK;              \
    *D = (*D >> (8)) ^ (*D << (24)) ^ CLK;              \
                                                        \
    A = &WALK[PT1 ^ 4];                                 \
    B = &WALK[PT2 ^ 1];                                 \
                                                        \
    PTEST = PT2 >> 1;                                   \
                                                        \
    PT2 = (RES[(i - 8) ^ PTY] ^ WALK[PT2 ^ PTY ^ 7]);   \
    PT2 = ((PT2 & 0x1FFF) & (~8)) ^ ((PT1 ^ 8) & 0x8);  \
    PTY = (PT2 >> 10) & 7;                              \
                                                        \
    TST2_ENTER  TST2_ENTER  TST2_ENTER  TST2_ENTER      \
    TST2_ENTER  TST2_ENTER  TST2_ENTER  TST2_ENTER      \
    TST2_ENTER  TST2_ENTER  TST2_ENTER  TST2_ENTER      \
                                                        \
    TST2_LEAVE  TST2_LEAVE  TST2_LEAVE  TST2_LEAVE      \
    TST2_LEAVE  TST2_LEAVE  TST2_LEAVE  TST2_LEAVE      \
    TST2_LEAVE  TST2_LEAVE  TST2_LEAVE  TST2_LEAVE      \
                                                        \
        C = &WALK[PT1 ^ 5];                                 \
    D = &WALK[PT2 ^ 5];                                 \
                                                        \
    RES[i++] ^= *A;                                     \
    RES[i++] ^= *B;                                     \
    RES[i++] ^= *C;                                     \
    RES[i++] ^= *D;                                     \
                                                        \
    IN = (*A >> (9)) ^ (*A << (23)) ^ CLK;             \
    *A = (*B >> (10)) ^ (*B << (22)) ^ CLK;             \
    *B = IN ^ U2;                                       \
    *C = (*C >> (11)) ^ (*C << (21)) ^ CLK;             \
    *D = (*D >> (12)) ^ (*D << (20)) ^ CLK;             \
                                                        \
    A = &WALK[PT1 ^ 6]; RES[i++] ^= *A;                 \
    B = &WALK[PT2 ^ 3]; RES[i++] ^= *B;                 \
    C = &WALK[PT1 ^ 7]; RES[i++] ^= *C;                 \
    D = &WALK[PT2 ^ 7]; RES[i++] ^= *D;                 \
                                                        \
    IN = (*A >> (13)) ^ (*A << (19)) ^ CLK;             \
    *A = (*B >> (14)) ^ (*B << (18)) ^ CLK;             \
    *B = IN;                                            \
    *C = (*C >> (15)) ^ (*C << (17)) ^ CLK;             \
    *D = (*D >> (16)) ^ (*D << (16)) ^ CLK;             \
                                                        \
    PT1 = (RES[(i - 8) ^ PTX] ^                      \
           WALK[PT1 ^ PTX ^ 7]) & (~1);               \
    PT1 ^= (PT2 ^ 0x10) & 0x10;                         \
                                                        \
    for (n++, i = 0; i < 16; i++)                      \
    hs->pool[n % MBEDTLS_HAVEGE_COLLECT_SIZE] ^= RES[i];

/*
 * Entropy gathering function
 */
static void havege_fill(mbedtls_havege_state *hs)
{
    size_t n = 0;
    size_t i;
    uint32_t  U1,  U2, *A, *B, *C, *D;
    uint32_t PT1, PT2, *WALK, RES[16];
    uint32_t PTX, PTY, CLK, PTEST, IN;

    WALK = hs->WALK;
    PT1  = hs->PT1;
    PT2  = hs->PT2;

    PTX  = U1 = 0;
    PTY  = U2 = 0;

    (void) PTX;

    memset(RES, 0, sizeof(RES));

    while (n < MBEDTLS_HAVEGE_COLLECT_SIZE * 4) {
        ONE_ITERATION
        ONE_ITERATION
        ONE_ITERATION
            ONE_ITERATION
    }

    hs->PT1 = PT1;
    hs->PT2 = PT2;

    hs->offset[0] = 0;
    hs->offset[1] = MBEDTLS_HAVEGE_COLLECT_SIZE / 2;
}

/*
 * HAVEGE initialization
 */
void mbedtls_havege_init(mbedtls_havege_state *hs)
{
    memset(hs, 0, sizeof(mbedtls_havege_state));

    havege_fill(hs);
}

void mbedtls_havege_free(mbedtls_havege_state *hs)
{
    if (hs == NULL) {
        return;
    }

    mbedtls_platform_zeroize(hs, sizeof(mbedtls_havege_state));
}

/*
 * HAVEGE rand function
 */
int mbedtls_havege_random(void *p_rng, unsigned char *buf, size_t len)
{
    uint32_t val;
    size_t use_len;
    mbedtls_havege_state *hs = (mbedtls_havege_state *) p_rng;
    unsigned char *p = buf;

    while (len > 0) {
        use_len = len;
        if (use_len > sizeof(val)) {
            use_len = sizeof(val);
        }

        if (hs->offset[1] >= MBEDTLS_HAVEGE_COLLECT_SIZE) {
            havege_fill(hs);
        }

        val  = hs->pool[hs->offset[0]++];
        val ^= hs->pool[hs->offset[1]++];

        memcpy(p, &val, use_len);

        len -= use_len;
        p += use_len;
    }

    return 0;
}

#endif /* MBEDTLS_HAVEGE_C */
