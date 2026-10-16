#ifndef DEFLATE_H_
#define DEFLATE_H_
/* deflate.h -- internal compression state
 * Copyright (C) 1995-2016 Jean-loup Gailly
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the compression library and is
   subject to change. Applications should only use zlib.h.
 */

#include "zutil.h"
#include "zendian.h"
#include "zmemory.h"
#include "crc32.h"

#ifdef S390_DFLTCC_DEFLATE
#  include "arch/s390/dfltcc_common.h"
#  define HAVE_ARCH_DEFLATE_STATE
#endif

/* define NO_GZIP when compiling if you want to disable gzip header and
   trailer creation by deflate().  NO_GZIP would be used to avoid linking in
   the crc code when it is not needed.  For shared libraries, gzip encoding
   should be left enabled. */
#ifndef NO_GZIP
#  define GZIP
#endif

/* define LIT_MEM to slightly increase the speed of deflate (order 1% to 2%) at
   the cost of a larger memory footprint */
#ifndef NO_LIT_MEM
#  define LIT_MEM
#endif

/* ===========================================================================
 * Internal compression state.
 */

#define LENGTH_CODES 29
/* number of length codes, not counting the special END_BLOCK code */

#define LITERALS  256
/* number of literal bytes 0..255 */

#define L_CODES (LITERALS+1+LENGTH_CODES)
/* number of Literal or Length codes, including the END_BLOCK code */

#define D_CODES   30
/* number of distance codes */

#define BL_CODES  19
/* number of codes used to transfer the bit lengths */

#define HEAP_SIZE (2*L_CODES+1)
/* maximum heap size */

#define BIT_BUF_SIZE 64
/* size of bit buffer in bi_buf */

#define END_BLOCK 256
/* end of block literal code */

#define INIT_STATE      1    /* zlib header -> BUSY_STATE */
#ifdef GZIP
#  define GZIP_STATE    4    /* gzip header -> BUSY_STATE | EXTRA_STATE */
#  define EXTRA_STATE   5    /* gzip extra block -> NAME_STATE */
#  define NAME_STATE    6    /* gzip file name -> COMMENT_STATE */
#  define COMMENT_STATE 7    /* gzip comment -> HCRC_STATE */
#  define HCRC_STATE    8    /* gzip header CRC -> BUSY_STATE */
#endif
#define BUSY_STATE      2    /* deflate -> FINISH_STATE */
#define FINISH_STATE    3    /* stream complete */
#ifdef GZIP
#  define MAX_STATE     HCRC_STATE
#else
#  define MAX_STATE     FINISH_STATE
#endif
/* Stream status */

#define HASH_BITS    16u           /* log2(HASH_SIZE) */
#ifndef HASH_SIZE
#  define HASH_SIZE 65536u         /* number of elements in hash table */
#endif
#define HASH_MASK (HASH_SIZE - 1u) /* HASH_SIZE-1 */


/* Data structure describing a single value and its code string. */
typedef struct ct_data_s {
    union {
        uint16_t  freq;       /* frequency count */
        uint16_t  code;       /* bit string */
    } fc;
    union {
        uint16_t  dad;        /* father node in Huffman tree */
        uint16_t  len;        /* length of bit string */
    } dl;
} ct_data;

#define Freq fc.freq
#define Code fc.code
#define Dad  dl.dad
#define Len  dl.len

typedef struct static_tree_desc_s  static_tree_desc;

typedef struct tree_desc_s {
    ct_data                *dyn_tree;  /* the dynamic tree */
    int                    max_code;   /* largest code with non zero frequency */
    const static_tree_desc *stat_desc; /* the corresponding static tree */
} tree_desc;

typedef uint16_t Pos;

/* A Pos is an index in the character window. We use short instead of int to
 * save space in the various tables.
 */
/* Type definitions for hash callbacks */
typedef struct internal_state deflate_state;

typedef uint32_t (* update_hash_cb)        (uint32_t h, uint32_t val);
typedef void     (* insert_string_cb)      (deflate_state *const s, uint32_t str, uint32_t count);
typedef Pos      (* quick_insert_string_cb)(deflate_state *const s, uint32_t str);

uint32_t update_hash             (uint32_t h, uint32_t val);
void     insert_string           (deflate_state *const s, uint32_t str, uint32_t count);
Pos      quick_insert_string     (deflate_state *const s, uint32_t str);

uint32_t update_hash_roll        (uint32_t h, uint32_t val);
void     insert_string_roll      (deflate_state *const s, uint32_t str, uint32_t count);
Pos      quick_insert_string_roll(deflate_state *const s, uint32_t str);

/* Struct for memory allocation handling */
typedef struct deflate_allocs_s {
    char            *buf_start;
    free_func        zfree;
    deflate_state   *state;
    unsigned char   *window;
    unsigned char   *pending_buf;
    Pos             *prev;
    Pos             *head;
} deflate_allocs;

struct ALIGNED_(64) internal_state {
    PREFIX3(stream)      *strm;            /* pointer back to this zlib stream */
    unsigned char        *pending_buf;     /* output still pending */
    unsigned char        *pending_out;     /* next pending byte to output to the stream */
    uint32_t             pending_buf_size; /* size of pending_buf */
    uint32_t             pending;          /* nb of bytes in the pending buffer */
    int                  wrap;             /* bit 0 true for zlib, bit 1 true for gzip */
    uint32_t             gzindex;          /* where in extra, name, or comment */
    PREFIX(gz_headerp)   gzhead;           /* gzip header information to write */
    int                  status;           /* as the name implies */
    int                  last_flush;       /* value of flush param for previous deflate call */
    int                  reproducible;     /* Whether reproducible compression results are required. */

    int block_open;
    /* Whether or not a block is currently open for the QUICK deflation scheme.
     * This is set to 1 if there is an active block, or 0 if the block was just closed.
     */

                /* used by deflate.c: */

    unsigned int  w_size;            /* LZ77 window size (32K by default) */
    unsigned int  w_bits;            /* log2(w_size)  (8..16) */
    unsigned int  w_mask;            /* w_size - 1 */
    unsigned int  lookahead;         /* number of valid bytes ahead in window */

    unsigned int high_water;
    /* High water mark offset in window for initialized bytes -- bytes above
     * this are set to zero in order to avoid memory check warnings when
     * longest match routines access bytes past the input.  This is then
     * updated to the new high water mark.
     */

    unsigned int window_size;
    /* Actual size of window: 2*wSize, except when the user input buffer
     * is directly used as sliding window.
     */

    unsigned char *window;
    /* Sliding window. Input bytes are read into the second half of the window,
     * and move to the first half later to keep a dictionary of at least wSize
     * bytes. With this organization, matches are limited to a distance of
     * wSize-STD_MAX_MATCH bytes, but this ensures that IO is always
     * performed with a length multiple of the block size. Also, it limits
     * the window size to 64K, which is quite useful on MSDOS.
     * To do: use the user input buffer as sliding window.
     */

    Pos *prev;
    /* Link to older string with same hash index. To limit the size of this
     * array to 64K, this link is maintained only for the last 32K strings.
     * An index in this array is thus a window index modulo 32K.
     */

    Pos *head; /* Heads of the hash chains or 0. */

    uint32_t ins_h; /* hash index of string to be inserted */

    int block_start;
    /* Window position at the beginning of the current output block. Gets
     * negative when the window is moved backwards.
     */

    unsigned int match_length;       /* length of best match */
    Pos          prev_match;         /* previous match */
    int          match_available;    /* set if previous match exists */
    unsigned int strstart;           /* start of string to insert */
    unsigned int match_start;        /* start of matching string */

    unsigned int prev_length;
    /* Length of the best match at previous step. Matches not greater than this
     * are discarded. This is used in the lazy match evaluation.
     */

    unsigned int max_chain_length;
    /* To speed up deflation, hash chains are never searched beyond this length.
     * A higher limit improves compression ratio but degrades the speed.
     */

    unsigned int max_lazy_match;
    /* Attempt to find a better match only when the current match is strictly smaller
     * than this value. This mechanism is used only for compression levels >= 4.
     */
#   define max_insert_length  max_lazy_match
    /* Insert new strings in the hash table only if the match length is not
     * greater than this length. This saves time but degrades compression.
     * max_insert_length is used only for compression levels <= 6.
     */

    update_hash_cb          update_hash;
    insert_string_cb        insert_string;
    quick_insert_string_cb  quick_insert_string;
    /* Hash function callbacks that can be configured depending on the deflate
     * algorithm being used */

    int level;    /* compression level (1..9) */
    int strategy; /* favor or force Huffman coding*/

    unsigned int good_match;
    /* Use a faster search when the previous match is longer than this */

    int nice_match; /* Stop searching when current match exceeds this */

#if defined(_M_IX86) || defined(_M_ARM)
    int padding[2];
#endif

    struct crc32_fold_s ALIGNED_(16) crc_fold;

                /* used by trees.c: */
    /* Didn't use ct_data typedef below to suppress compiler warning */
    struct ct_data_s dyn_ltree[HEAP_SIZE];   /* literal and length tree */
    struct ct_data_s dyn_dtree[2*D_CODES+1]; /* distance tree */
    struct ct_data_s bl_tree[2*BL_CODES+1];  /* Huffman tree for bit lengths */

    struct tree_desc_s l_desc;               /* desc. for literal tree */
    struct tree_desc_s d_desc;               /* desc. for distance tree */
    struct tree_desc_s bl_desc;              /* desc. for bit length tree */

    uint16_t bl_count[MAX_BITS+1];
    /* number of codes at each bit length for an optimal tree */

    int heap[2*L_CODES+1];      /* heap used to build the Huffman trees */
    int heap_len;               /* number of elements in the heap */
    int heap_max;               /* element of largest frequency */
    /* The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
     * The same heap array is used to build all trees.
     */

    unsigned char depth[2*L_CODES+1];
    /* Depth of each subtree used as tie breaker for trees of equal frequency
     */

    unsigned int  lit_bufsize;
    /* Size of match buffer for literals/lengths.  There are 4 reasons for
     * limiting lit_bufsize to 64K:
     *   - frequencies can be kept in 16 bit counters
     *   - if compression is not successful for the first block, all input
     *     data is still in the window so we can still emit a stored block even
     *     when input comes from standard input.  (This can also be done for
     *     all blocks if lit_bufsize is not greater than 32K.)
     *   - if compression is not successful for a file smaller than 64K, we can
     *     even emit a stored file instead of a stored block (saving 5 bytes).
     *     This is applicable only for zip (not gzip or zlib).
     *   - creating new Huffman trees less frequently may not provide fast
     *     adaptation to changes in the input data statistics. (Take for
     *     example a binary file with poorly compressible code followed by
     *     a highly compressible string table.) Smaller buffer sizes give
     *     fast adaptation but have of course the overhead of transmitting
     *     trees more frequently.
     *   - I can't count above 4
     */

#ifdef LIT_MEM
#   define LIT_BUFS 5
    uint16_t *d_buf;              /* buffer for distances */
    unsigned char *l_buf;         /* buffer for literals/lengths */
#else
#   define LIT_BUFS 4
    unsigned char *sym_buf;       /* buffer for distances and literals/lengths */
#endif

    unsigned int sym_next;        /* running index in symbol buffer */
    unsigned int sym_end;         /* symbol table full when sym_next reaches this */

    unsigned long opt_len;        /* bit length of current block with optimal trees */
    unsigned long static_len;     /* bit length of current block with static trees */
    unsigned int matches;         /* number of string matches in current block */
    unsigned int insert;          /* bytes at end of window left to insert */

    /* compressed_len and bits_sent are only used if ZLIB_DEBUG is defined */
    unsigned long compressed_len; /* total bit length of compressed file mod 2^32 */
    unsigned long bits_sent;      /* bit length of compressed data sent mod 2^32 */

    deflate_allocs *alloc_bufs;

#ifdef HAVE_ARCH_DEFLATE_STATE
    arch_deflate_state arch;      /* architecture-specific extensions */
#endif

    uint64_t bi_buf;
    /* Output buffer. bits are inserted starting at the bottom (least significant bits). */

    int32_t bi_valid;
    /* Number of valid bits in bi_buf.  All bits above the last valid bit are always zero. */

    /* Reserved for future use and alignment purposes */
    int32_t reserved[19];
#if defined(_M_IX86) || defined(_M_ARM)
    int32_t padding2[4];
#endif
};

typedef enum {
    need_more,      /* block not completed, need more input or more output */
    block_done,     /* block flush performed */
    finish_started, /* finish started, need only more output at next deflate */
    finish_done     /* finish done, accept no more input or output */
} block_state;

/* Output a byte on the stream.
 * IN assertion: there is enough room in pending_buf.
 */
#define put_byte(s, c) { \
    s->pending_buf[s->pending++] = (unsigned char)(c); \
}

/* ===========================================================================
 * Output a short LSB first on the stream.
 * IN assertion: there is enough room in pending_buf.
 */
static inline void put_short(deflate_state *s, uint16_t w) {
#if BYTE_ORDER == BIG_ENDIAN
    w = ZSWAP16(w);
#endif
    zng_memwrite_2(&s->pending_buf[s->pending], w);
    s->pending += 2;
}

/* ===========================================================================
 * Output a short MSB first on the stream.
 * IN assertion: there is enough room in pending_buf.
 */
static inline void put_short_msb(deflate_state *s, uint16_t w) {
#if BYTE_ORDER == LITTLE_ENDIAN
    w = ZSWAP16(w);
#endif
    zng_memwrite_2(&s->pending_buf[s->pending], w);
    s->pending += 2;
}

/* ===========================================================================
 * Output a 32-bit unsigned int LSB first on the stream.
 * IN assertion: there is enough room in pending_buf.
 */
static inline void put_uint32(deflate_state *s, uint32_t dw) {
#if BYTE_ORDER == BIG_ENDIAN
    dw = ZSWAP32(dw);
#endif
    zng_memwrite_4(&s->pending_buf[s->pending], dw);
    s->pending += 4;
}

/* ===========================================================================
 * Output a 32-bit unsigned int MSB first on the stream.
 * IN assertion: there is enough room in pending_buf.
 */
static inline void put_uint32_msb(deflate_state *s, uint32_t dw) {
#if BYTE_ORDER == LITTLE_ENDIAN
    dw = ZSWAP32(dw);
#endif
    zng_memwrite_4(&s->pending_buf[s->pending], dw);
    s->pending += 4;
}

/* ===========================================================================
 * Output a 64-bit unsigned int LSB first on the stream.
 * IN assertion: there is enough room in pending_buf.
 */
static inline void put_uint64(deflate_state *s, uint64_t lld) {
#if BYTE_ORDER == BIG_ENDIAN
    lld = ZSWAP64(lld);
#endif
    zng_memwrite_8(&s->pending_buf[s->pending], lld);
    s->pending += 8;
}

#define MIN_LOOKAHEAD (STD_MAX_MATCH + STD_MIN_MATCH + 1)
/* Minimum amount of lookahead, except at the end of the input file.
 * See deflate.c for comments about the STD_MIN_MATCH+1.
 */

#define MAX_DIST(s)  ((s)->w_size - MIN_LOOKAHEAD)
/* In order to simplify the code, particularly on 16 bit machines, match
 * distances are limited to MAX_DIST instead of WSIZE.
 */

#define WIN_INIT STD_MAX_MATCH
/* Number of bytes after end of data in window to initialize in order to avoid
   memory checker errors from longest match routines */


void Z_INTERNAL PREFIX(fill_window)(deflate_state *s);
void Z_INTERNAL slide_hash_c(deflate_state *s);

        /* in trees.c */
void Z_INTERNAL zng_tr_init(deflate_state *s);
void Z_INTERNAL zng_tr_flush_block(deflate_state *s, char *buf, uint32_t stored_len, int last);
void Z_INTERNAL zng_tr_flush_bits(deflate_state *s);
void Z_INTERNAL zng_tr_align(deflate_state *s);
void Z_INTERNAL zng_tr_stored_block(deflate_state *s, char *buf, uint32_t stored_len, int last);
void Z_INTERNAL PREFIX(flush_pending)(PREFIX3(streamp) strm);
#define d_code(dist) ((dist) < 256 ? zng_dist_code[dist] : zng_dist_code[256+((dist)>>7)])
/* Mapping from a distance to a distance code. dist is the distance - 1 and
 * must not have side effects. zng_dist_code[256] and zng_dist_code[257] are never
 * used.
 */

/* Bit buffer and compress bits calculation debugging */
#ifdef ZLIB_DEBUG
#  define cmpr_bits_add(s, len)     s->compressed_len += (len)
#  define cmpr_bits_align(s)        s->compressed_len = (s->compressed_len + 7) & ~7L
#  define sent_bits_add(s, bits)    s->bits_sent += (bits)
#  define sent_bits_align(s)        s->bits_sent = (s->bits_sent + 7) & ~7L
#else
#  define cmpr_bits_add(s, len)     Z_UNUSED(len)
#  define cmpr_bits_align(s)
#  define sent_bits_add(s, bits)    Z_UNUSED(bits)
#  define sent_bits_align(s)
#endif

/* ===========================================================================
 *  Architecture-specific hooks.
 */
#ifdef S390_DFLTCC_DEFLATE
#  include "arch/s390/dfltcc_deflate.h"
/* DFLTCC instructions require window to be page-aligned */
#  define PAD_WINDOW            PAD_4096
#  define WINDOW_PAD_SIZE       4096
#  define HINT_ALIGNED_WINDOW   HINT_ALIGNED_4096
#else
#  define PAD_WINDOW            PAD_64
#  define WINDOW_PAD_SIZE       64
#  define HINT_ALIGNED_WINDOW   HINT_ALIGNED_64
/* Adjust the window size for the arch-specific deflate code. */
#  define DEFLATE_ADJUST_WINDOW_SIZE(n) (n)
/* Invoked at the beginning of deflateSetDictionary(). Useful for checking arch-specific window data. */
#  define DEFLATE_SET_DICTIONARY_HOOK(strm, dict, dict_len) do {} while (0)
/* Invoked at the beginning of deflateGetDictionary(). Useful for adjusting arch-specific window data. */
#  define DEFLATE_GET_DICTIONARY_HOOK(strm, dict, dict_len) do {} while (0)
/* Invoked at the end of deflateResetKeep(). Useful for initializing arch-specific extension blocks. */
#  define DEFLATE_RESET_KEEP_HOOK(strm) do {} while (0)
/* Invoked at the beginning of deflateParams(). Useful for updating arch-specific compression parameters. */
#  define DEFLATE_PARAMS_HOOK(strm, level, strategy, hook_flush) do {} while (0)
/* Returns whether the last deflate(flush) operation did everything it's supposed to do. */
#  define DEFLATE_DONE(strm, flush) 1
/* Adjusts the upper bound on compressed data length based on compression parameters and uncompressed data length.
 * Useful when arch-specific deflation code behaves differently than regular zlib-ng algorithms. */
#  define DEFLATE_BOUND_ADJUST_COMPLEN(strm, complen, sourceLen) do {} while (0)
/* Returns whether an optimistic upper bound on compressed data length should *not* be used.
 * Useful when arch-specific deflation code behaves differently than regular zlib-ng algorithms. */
#  define DEFLATE_NEED_CONSERVATIVE_BOUND(strm) 0
/* Invoked for each deflate() call. Useful for plugging arch-specific deflation code. */
#  define DEFLATE_HOOK(strm, flush, bstate) 0
/* Returns whether zlib-ng should compute a checksum. Set to 0 if arch-specific deflation code already does that. */
#  define DEFLATE_NEED_CHECKSUM(strm) 1
/* Returns whether reproducibility parameter can be set to a given value. */
#  define DEFLATE_CAN_SET_REPRODUCIBLE(strm, reproducible) 1
#endif

#endif /* DEFLATE_H_ */
