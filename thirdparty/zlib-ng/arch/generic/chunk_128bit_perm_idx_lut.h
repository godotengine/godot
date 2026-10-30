/* chunk_128bit_perm_idx_lut.h - shared SSSE3/NEON/LSX permutation idx lut for use with chunkmemset family of functions.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#ifndef CHUNK_128BIT_PERM_IDX_LUT_H_
#define CHUNK_128BIT_PERM_IDX_LUT_H_

#include "chunk_permute_table.h"

static const lut_rem_pair perm_idx_lut[13] = {
    {0, 1},      /* 3 */
    {0, 0},      /* don't care */
    {1 * 32, 1}, /* 5 */
    {2 * 32, 4}, /* 6 */
    {3 * 32, 2}, /* 7 */
    {0 * 32, 0}, /* don't care */
    {4 * 32, 7}, /* 9 */
    {5 * 32, 6}, /* 10 */
    {6 * 32, 5}, /* 11 */
    {7 * 32, 4}, /* 12 */
    {8 * 32, 3}, /* 13 */
    {9 * 32, 2}, /* 14 */
    {10 * 32, 1},/* 15 */
};

#endif
