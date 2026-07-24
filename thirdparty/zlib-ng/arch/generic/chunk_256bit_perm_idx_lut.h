/* chunk_256bit_perm_idx_lut.h - shared AVX512/AVX2/LASX permutation idx lut for use with chunkmemset family of functions.
 * For conditions of distribution and use, see copyright notice in zlib.h
 */
#ifndef CHUNK_256BIT_PERM_IDX_LUT_H_
#define CHUNK_256BIT_PERM_IDX_LUT_H_

#include "chunk_permute_table.h"

/* Populate don't cares so that this is a direct lookup (with some indirection into the permute table), because dist can
 * never be 0 - 2, we'll start with an offset, subtracting 3 from the input */
static const lut_rem_pair perm_idx_lut[29] = {
    { 0, 2},                /* 3 */
    { 0, 0},                /* don't care */
    { 1 * 32, 2},           /* 5 */
    { 2 * 32, 2},           /* 6 */
    { 3 * 32, 4},           /* 7 */
    { 0 * 32, 0},           /* don't care */
    { 4 * 32, 5},           /* 9 */
    { 5 * 32, 22},          /* 10 */
    { 6 * 32, 21},          /* 11 */
    { 7 * 32, 20},          /* 12 */
    { 8 * 32, 6},           /* 13 */
    { 9 * 32, 4},           /* 14 */
    {10 * 32, 2},           /* 15 */
    { 0 * 32, 0},           /* don't care */
    {11 * 32, 15},          /* 17 */
    {11 * 32 + 16, 14},     /* 18 */
    {11 * 32 + 16 * 2, 13}, /* 19 */
    {11 * 32 + 16 * 3, 12}, /* 20 */
    {11 * 32 + 16 * 4, 11}, /* 21 */
    {11 * 32 + 16 * 5, 10}, /* 22 */
    {11 * 32 + 16 * 6,  9}, /* 23 */
    {11 * 32 + 16 * 7,  8}, /* 24 */
    {11 * 32 + 16 * 8,  7}, /* 25 */
    {11 * 32 + 16 * 9,  6}, /* 26 */
    {11 * 32 + 16 * 10, 5}, /* 27 */
    {11 * 32 + 16 * 11, 4}, /* 28 */
    {11 * 32 + 16 * 12, 3}, /* 29 */
    {11 * 32 + 16 * 13, 2}, /* 30 */
    {11 * 32 + 16 * 14, 1}  /* 31 */
};

static const uint16_t half_rem_vals[13] = {
    1, 0, 1, 4, 2, 0, 7, 6, 5, 4, 3, 2, 1
};

#endif
