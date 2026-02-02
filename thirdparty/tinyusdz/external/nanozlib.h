/* SPDX-License-Identifier: Apache 2.0 */
/* Copyright 2023 - Present, Light Transport Entertainment Inc. */

/* TODO:
 *
 * - [ ] Stream decoding API
 * - [ ] Stream encoding API
 *
 */

#ifndef NANOZLIB_H_
#define NANOZLIB_H_

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nanoz_status {
  NANOZ_SUCCESS = 0,
  NANOZ_ERROR = -1,  // general error code.
  NANOZ_ERROR_INVALID_ARGUMENT = -2,
  NANOZ_ERROR_CORRUPTED = -3,
  NANOZ_ERROR_INTERNAL = -4,
} nanoz_status_t;

#if 0  // TODO
/* Up to 2GB chunk. */
typedef (*nanoz_stream_read)(const uint8_t *addr, uint8_t *dst_addr, const uint32_t read_bytes, const void *user_ptr);
typedef (*nanoz_stream_write)(const uint8_t *addr, const uint32_t write_bytes, const void *user_ptr);
#endif

/* TODO: Get uncompressed size function */

/*
 * zlib decompression. Up to 2GB compressed data.
 *
 * @param[in] src_addr Source buffer address containing compressed data.
 * @param[in] src_size Source buffer bytes.
 * @param[in] dst_size Destination buffer size. Must be larger than or equal to uncompressed size.
 * @param[out] dst_addr Destination buffer address.
 * @param[out] uncompressed_size Uncompressed bytes.
 * contain `uncompressed_size` bytes.
 * @return NANOZ_SUCCESS upon success.
 *
 * TODO: return error message string.
 */
nanoz_status_t nanoz_uncompress(const unsigned char *src_addr,
                                int32_t src_size,
                                const uint64_t dst_size,
                                unsigned char *dst_addr,
                                uint64_t *uncompressed_size);

/*
 * Compute compress bound.
 */
uint64_t nanoz_compressBound(uint64_t sourceLen);

/*
 * zlib compression. Currently we use stb's zlib_compress
 *
 * @param[in] data Input data
 * @param[in] data_len Input data bytes(up to 2GB)
 * @param[out] out_len Input data
 * @param[in] quality Compression quality(5 or more. Usually 8)
 *
 * @return Compressed bytes upon success. NULL when failed to compress or any input parameter is wrong.
 */
unsigned char *nanoz_compress(unsigned char *data, int data_len, int *out_len,
                              int quality);

#if 0  // TODO
nanoz_status_t nanoz_stream_uncompress(nanoz_stream_read *reader, nanoz_stream_writer *writer);
#endif

#ifdef __cplusplus
}
#endif

#if defined(NANOZLIB_IMPLEMENTATION)

#define WUFFS_IMPLEMENTATION

#define WUFFS_CONFIG__STATIC_FUNCTIONS

#define WUFFS_CONFIG__MODULES
#define WUFFS_CONFIG__MODULE__BASE
#define WUFFS_CONFIG__MODULE__CRC32
#define WUFFS_CONFIG__MODULE__ADLER32
#define WUFFS_CONFIG__MODULE__DEFLATE
#define WUFFS_CONFIG__MODULE__ZLIB

// Use wuffs-unsupported-snapshot.c
#include "wuffs-unsupported-snapshot.c"
//#include "wuffs-v0.3.c"

#define WORK_BUFFER_ARRAY_SIZE \
  WUFFS_ZLIB__DECODER_WORKBUF_LEN_MAX_INCL_WORST_CASE

nanoz_status_t nanoz_uncompress(const unsigned char *src_addr,
                                const int32_t src_size,
                                const uint64_t dst_size,
                                unsigned char *dst_addr,
                                uint64_t *uncompressed_size_out) {
// WUFFS_ZLIB__DECODER_WORKBUF_LEN_MAX_INCL_WORST_CASE = 1, its tiny bytes and
// safe to alloc worbuf at heap location.
#if WORK_BUFFER_ARRAY_SIZE > 0
  uint8_t work_buffer_array[WORK_BUFFER_ARRAY_SIZE];
#else
  // Not all C/C++ compilers support 0-length arrays.
  uint8_t work_buffer_array[1];
#endif

  if (!src_addr) {
    return NANOZ_ERROR_INVALID_ARGUMENT;
  }

  if (src_size < 4) {
    return NANOZ_ERROR_INVALID_ARGUMENT;
  }

  if (!dst_addr) {
    return NANOZ_ERROR_INVALID_ARGUMENT;
  }

  if (dst_size < 1) {
    return NANOZ_ERROR_INVALID_ARGUMENT;
  }

  if (!uncompressed_size_out) {
    return NANOZ_ERROR_INVALID_ARGUMENT;
  }

  wuffs_zlib__decoder dec;
  wuffs_base__status status =
      wuffs_zlib__decoder__initialize(&dec, sizeof dec, WUFFS_VERSION, 0);
  if (!wuffs_base__status__is_ok(&status)) {
    // wuffs_base__status__message(&status);
    return NANOZ_ERROR_INTERNAL;
  }

  // TODO: Streamed decoding?

  wuffs_base__io_buffer dst;
  dst.data.ptr = dst_addr;
  dst.data.len = dst_size;
  dst.meta.wi = 0;
  dst.meta.ri = 0;
  dst.meta.pos = 0;
  dst.meta.closed = false;

  wuffs_base__io_buffer src;
  src.data.ptr = const_cast<uint8_t *>(src_addr);  // remove const
  src.data.len = src_size;
  src.meta.wi = src_size;
  src.meta.ri = 0;
  src.meta.pos = 0;
  src.meta.closed = false;

  status = wuffs_zlib__decoder__transform_io(
      &dec, &dst, &src,
      wuffs_base__make_slice_u8(work_buffer_array, WORK_BUFFER_ARRAY_SIZE));

  uint64_t uncompressed_size{0};

  if (dst.meta.wi) {
    dst.meta.ri = dst.meta.wi;
    uncompressed_size = dst.meta.wi;
    wuffs_base__io_buffer__compact(&dst);
  }

  if (status.repr == wuffs_base__suspension__short_read) {
    // ok
  } else if (status.repr == wuffs_base__suspension__short_write) {
    // read&write should succeed at once.
    return NANOZ_ERROR_CORRUPTED;
  }

  const char *stat_msg = wuffs_base__status__message(&status);
  if (stat_msg) {
    return NANOZ_ERROR_INTERNAL;
  }

  (*uncompressed_size_out) = uncompressed_size;

  return NANOZ_SUCCESS;
}

#ifndef NANOZ_MALLOC
#define NANOZ_MALLOC(sz) malloc(sz)
#define NANOZ_REALLOC(p, newsz) realloc(p, newsz)
#define NANOZ_FREE(p) free(p)
#endif

#ifndef NANOZ_REALLOC_SIZED
#define NANOZ_REALLOC_SIZED(p, oldsz, newsz) NANOZ_REALLOC(p, newsz)
#endif

#ifndef NANOZ_MEMMOVE
#define NANOZ_MEMMOVE(a, b, sz) memmove(a, b, sz)
#endif

#define NANOZ_UCHAR(x) (unsigned char)((x)&0xff)

// #ifndef NANOZ_ZLIB_COMPRESS
//  stretchy buffer; nanoz__sbpush() == vector<>::push_back() --
//  nanoz__sbcount() == vector<>::size()
#define nanoz__sbraw(a) ((int *)(void *)(a)-2)
#define nanoz__sbm(a) nanoz__sbraw(a)[0]
#define nanoz__sbn(a) nanoz__sbraw(a)[1]

#define nanoz__sbneedgrow(a, n) ((a) == 0 || nanoz__sbn(a) + n >= nanoz__sbm(a))
#define nanoz__sbmaybegrow(a, n) \
  (nanoz__sbneedgrow(a, (n)) ? nanoz__sbgrow(a, n) : 0)
#define nanoz__sbgrow(a, n) nanoz__sbgrowf((void **)&(a), (n), sizeof(*(a)))

#define nanoz__sbpush(a, v) \
  (nanoz__sbmaybegrow(a, 1), (a)[nanoz__sbn(a)++] = (v))
#define nanoz__sbcount(a) ((a) ? nanoz__sbn(a) : 0)
#define nanoz__sbfree(a) ((a) ? NANOZ_FREE(nanoz__sbraw(a)), 0 : 0)

static void *nanoz__sbgrowf(void **arr, int increment, int itemsize) {
  int m = *arr ? 2 * nanoz__sbm(*arr) + increment : increment + 1;
  void *p = NANOZ_REALLOC_SIZED(
      *arr ? nanoz__sbraw(*arr) : 0,
      *arr ? (nanoz__sbm(*arr) * itemsize + sizeof(int) * 2) : 0,
      itemsize * m + sizeof(int) * 2);
  if (!p) {
    return nullptr;
  }

  if (p) {
    if (!*arr) ((int *)p)[1] = 0;
    *arr = (void *)((int *)p + 2);
    nanoz__sbm(*arr) = m;
  }
  return *arr;
}

static unsigned char *nanoz__zlib_flushf(unsigned char *data,
                                         unsigned int *bitbuffer,
                                         int *bitcount) {
  while (*bitcount >= 8) {
    nanoz__sbpush(data, NANOZ_UCHAR(*bitbuffer));
    *bitbuffer >>= 8;
    *bitcount -= 8;
  }
  return data;
}

static int nanoz__zlib_bitrev(int code, int codebits) {
  int res = 0;
  while (codebits--) {
    res = (res << 1) | (code & 1);
    code >>= 1;
  }
  return res;
}

static unsigned int nanoz__zlib_countm(unsigned char *a, unsigned char *b,
                                       int limit) {
  int i;
  for (i = 0; i < limit && i < 258; ++i)
    if (a[i] != b[i]) break;
  return i;
}

static unsigned int nanoz__zhash(unsigned char *data) {
  uint32_t hash = data[0] + (data[1] << 8) + (data[2] << 16);
  hash ^= hash << 3;
  hash += hash >> 5;
  hash ^= hash << 4;
  hash += hash >> 17;
  hash ^= hash << 25;
  hash += hash >> 6;
  return hash;
}

#define nanoz__zlib_flush() (out = nanoz__zlib_flushf(out, &bitbuf, &bitcount))
#define nanoz__zlib_add(code, codebits) \
  (bitbuf |= (code) << bitcount, bitcount += (codebits), nanoz__zlib_flush())
#define nanoz__zlib_huffa(b, c) nanoz__zlib_add(nanoz__zlib_bitrev(b, c), c)
// default huffman tables
#define nanoz__zlib_huff1(n) nanoz__zlib_huffa(0x30 + (n), 8)
#define nanoz__zlib_huff2(n) nanoz__zlib_huffa(0x190 + (n)-144, 9)
#define nanoz__zlib_huff3(n) nanoz__zlib_huffa(0 + (n)-256, 7)
#define nanoz__zlib_huff4(n) nanoz__zlib_huffa(0xc0 + (n)-280, 8)
#define nanoz__zlib_huff(n)            \
  ((n) <= 143   ? nanoz__zlib_huff1(n) \
   : (n) <= 255 ? nanoz__zlib_huff2(n) \
   : (n) <= 279 ? nanoz__zlib_huff3(n) \
                : nanoz__zlib_huff4(n))
#define nanoz__zlib_huffb(n) \
  ((n) <= 143 ? nanoz__zlib_huff1(n) : nanoz__zlib_huff2(n))

#define nanoz__ZHASH 16384

// #endif // NANOZ_ZLIB_COMPRESS

unsigned char *nanoz_compress(unsigned char *data, int data_len, int *out_len,
                              int quality) {
  static unsigned short lengthc[] = {
      3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23,  27,
      31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 259};
  static unsigned char lengtheb[] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                                     1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                     4, 4, 4, 4, 5, 5, 5, 5, 0};
  static unsigned short distc[] = {
      1,    2,    3,    4,    5,    7,     9,     13,    17,   25,   33,
      49,   65,   97,   129,  193,  257,   385,   513,   769,  1025, 1537,
      2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 32768};
  static unsigned char disteb[] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                   4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                   9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
  unsigned int bitbuf = 0;
  int i, j, bitcount = 0;
  unsigned char *out = NULL;

  if (!data) {
    return NULL;
  }

  if (data_len < 1) {
    return NULL;
  }

  if (!out_len) {
    return NULL;
  }

  unsigned char ***hash_table =
      (unsigned char ***)NANOZ_MALLOC(nanoz__ZHASH * sizeof(unsigned char **));
  if (hash_table == NULL) return NULL;
  if (quality < 5) quality = 5;

  nanoz__sbpush(out, 0x78);  // DEFLATE 32K window
  nanoz__sbpush(out, 0x5e);  // FLEVEL = 1
  nanoz__zlib_add(1, 1);     // BFINAL = 1
  nanoz__zlib_add(1, 2);     // BTYPE = 1 -- fixed huffman

  for (i = 0; i < nanoz__ZHASH; ++i) hash_table[i] = NULL;

  i = 0;
  while (i < data_len - 3) {
    // hash next 3 bytes of data to be compressed
    int h = nanoz__zhash(data + i) & (nanoz__ZHASH - 1), best = 3;
    unsigned char *bestloc = 0;
    unsigned char **hlist = hash_table[h];
    int n = nanoz__sbcount(hlist);
    for (j = 0; j < n; ++j) {
      if (hlist[j] - data > i - 32768) {  // if entry lies within window
        int d = nanoz__zlib_countm(hlist[j], data + i, data_len - i);
        if (d >= best) {
          best = d;
          bestloc = hlist[j];
        }
      }
    }
    // when hash table entry is too long, delete half the entries
    if (hash_table[h] && nanoz__sbn(hash_table[h]) == 2 * quality) {
      NANOZ_MEMMOVE(hash_table[h], hash_table[h] + quality,
                    sizeof(hash_table[h][0]) * quality);
      nanoz__sbn(hash_table[h]) = quality;
    }
    nanoz__sbpush(hash_table[h], data + i);

    if (bestloc) {
      // "lazy matching" - check match at *next* byte, and if it's better, do
      // cur byte as literal
      h = nanoz__zhash(data + i + 1) & (nanoz__ZHASH - 1);
      hlist = hash_table[h];
      n = nanoz__sbcount(hlist);
      for (j = 0; j < n; ++j) {
        if (hlist[j] - data > i - 32767) {
          int e = nanoz__zlib_countm(hlist[j], data + i + 1, data_len - i - 1);
          if (e > best) {  // if next match is better, bail on current match
            bestloc = NULL;
            break;
          }
        }
      }
    }

    if (bestloc) {
      int d = (int)(data + i - bestloc);  // distance back
      // NANOZ_ASSERT(d <= 32767 && best <= 258);
      if (d <= 32767 && best <= 258) {
        // OK
      } else {
        return NULL;  // FIXME: may leak
      }
      for (j = 0; best > lengthc[j + 1] - 1; ++j)
        ;
      nanoz__zlib_huff(j + 257);
      if (lengtheb[j]) nanoz__zlib_add(best - lengthc[j], lengtheb[j]);
      for (j = 0; d > distc[j + 1] - 1; ++j)
        ;
      nanoz__zlib_add(nanoz__zlib_bitrev(j, 5), 5);
      if (disteb[j]) nanoz__zlib_add(d - distc[j], disteb[j]);
      i += best;
    } else {
      nanoz__zlib_huffb(data[i]);
      ++i;
    }
  }
  // write out final bytes
  for (; i < data_len; ++i) nanoz__zlib_huffb(data[i]);
  nanoz__zlib_huff(256);  // end of block
  // pad with 0 bits to byte boundary
  while (bitcount) nanoz__zlib_add(0, 1);

  for (i = 0; i < nanoz__ZHASH; ++i) (void)nanoz__sbfree(hash_table[i]);
  NANOZ_FREE(hash_table);

  // store uncompressed instead if compression was worse
  if (nanoz__sbn(out) > data_len + 2 + ((data_len + 32766) / 32767) * 5) {
    nanoz__sbn(out) = 2;  // truncate to DEFLATE 32K window and FLEVEL = 1
    for (j = 0; j < data_len;) {
      int blocklen = data_len - j;
      if (blocklen > 32767) blocklen = 32767;
      nanoz__sbpush(
          out,
          data_len - j == blocklen);  // BFINAL = ?, BTYPE = 0 -- no compression
      nanoz__sbpush(out, NANOZ_UCHAR(blocklen));  // LEN
      nanoz__sbpush(out, NANOZ_UCHAR(blocklen >> 8));
      nanoz__sbpush(out, NANOZ_UCHAR(~blocklen));  // NLEN
      nanoz__sbpush(out, NANOZ_UCHAR(~blocklen >> 8));
      memcpy(out + nanoz__sbn(out), data + j, blocklen);
      nanoz__sbn(out) += blocklen;
      j += blocklen;
    }
  }

  {
    // compute adler32 on input
    unsigned int s1 = 1, s2 = 0;
    int blocklen = (int)(data_len % 5552);
    j = 0;
    while (j < data_len) {
      for (i = 0; i < blocklen; ++i) {
        s1 += data[j + i];
        s2 += s1;
      }
      s1 %= 65521;
      s2 %= 65521;
      j += blocklen;
      blocklen = 5552;
    }
    nanoz__sbpush(out, NANOZ_UCHAR(s2 >> 8));
    nanoz__sbpush(out, NANOZ_UCHAR(s2));
    nanoz__sbpush(out, NANOZ_UCHAR(s1 >> 8));
    nanoz__sbpush(out, NANOZ_UCHAR(s1));
  }
  *out_len = nanoz__sbn(out);
  // make returned pointer freeable
  NANOZ_MEMMOVE(nanoz__sbraw(out), out, *out_len);
  return (unsigned char *)nanoz__sbraw(out);
}

// from zlib
uint64_t nanoz_compressBound(uint64_t sourceLen)
{
  // TODO: Overflow check?
  return sourceLen + (sourceLen >> 12ull) + (sourceLen >> 14ull) +
         (sourceLen >> 25ull) + 13ull;
}


#endif  // NANOZDEC_IMPLEMENTATION

#endif /* NANOZDEC_H_ */
