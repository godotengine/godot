/* LzHash.h -- HASH constants for LZ algorithms
2023-03-05 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_LZ_HASH_H
#define ZIP7_INC_LZ_HASH_H

/*
  (kHash2Size >= (1 <<  8)) : Required
  (kHash3Size >= (1 << 16)) : Required
*/

#define kHash2Size (1 << 10)
#define kHash3Size (1 << 16)
// #define kHash4Size (1 << 20)

#define kFix3HashSize (kHash2Size)
#define kFix4HashSize (kHash2Size + kHash3Size)
// #define kFix5HashSize (kHash2Size + kHash3Size + kHash4Size)

/*
  We use up to 3 crc values for hash:
    crc0
    crc1 << Shift_1
    crc2 << Shift_2
  (Shift_1 = 5) and (Shift_2 = 10) is good tradeoff.
  Small values for Shift are not good for collision rate.
  Big value for Shift_2 increases the minimum size
  of hash table, that will be slow for small files.
*/

#define kLzHash_CrcShift_1 5
#define kLzHash_CrcShift_2 10

#endif
