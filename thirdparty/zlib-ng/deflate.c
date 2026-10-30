/* deflate.c -- compress data using the deflation algorithm
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/*
 *  ALGORITHM
 *
 *      The "deflation" process depends on being able to identify portions
 *      of the input text which are identical to earlier input (within a
 *      sliding window trailing behind the input currently being processed).
 *
 *      The most straightforward technique turns out to be the fastest for
 *      most input files: try all possible matches and select the longest.
 *      The key feature of this algorithm is that insertions into the string
 *      dictionary are very simple and thus fast, and deletions are avoided
 *      completely. Insertions are performed at each input character, whereas
 *      string matches are performed only when the previous match ends. So it
 *      is preferable to spend more time in matches to allow very fast string
 *      insertions and avoid deletions. The matching algorithm for small
 *      strings is inspired from that of Rabin & Karp. A brute force approach
 *      is used to find longer strings when a small match has been found.
 *      A similar algorithm is used in comic (by Jan-Mark Wams) and freeze
 *      (by Leonid Broukhis).
 *         A previous version of this file used a more sophisticated algorithm
 *      (by Fiala and Greene) which is guaranteed to run in linear amortized
 *      time, but has a larger average cost, uses more memory and is patented.
 *      However the F&G algorithm may be faster for some highly redundant
 *      files if the parameter max_chain_length (described below) is too large.
 *
 *  ACKNOWLEDGEMENTS
 *
 *      The idea of lazy evaluation of matches is due to Jan-Mark Wams, and
 *      I found it in 'freeze' written by Leonid Broukhis.
 *      Thanks to many people for bug reports and testing.
 *
 *  REFERENCES
 *
 *      Deutsch, L.P.,"DEFLATE Compressed Data Format Specification".
 *      Available in https://tools.ietf.org/html/rfc1951
 *
 *      A description of the Rabin and Karp algorithm is given in the book
 *         "Algorithms" by R. Sedgewick, Addison-Wesley, p252.
 *
 *      Fiala,E.R., and Greene,D.H.
 *         Data Compression with Finite Windows, Comm.ACM, 32,4 (1989) 490-595
 *
 */

#include "zbuild.h"
#include "functable.h"
#include "deflate.h"
#include "deflate_p.h"

/* Avoid conflicts with zlib.h macros */
#ifdef ZLIB_COMPAT
# undef deflateInit
# undef deflateInit2
#endif

const char PREFIX(deflate_copyright)[] = " deflate 1.3.1 Copyright 1995-2024 Jean-loup Gailly and Mark Adler ";
/*
  If you use the zlib library in a product, an acknowledgment is welcome
  in the documentation of your product. If for some reason you cannot
  include such an acknowledgment, I would appreciate that you keep this
  copyright string in the executable of your product.
 */

/* ===========================================================================
 *  Function prototypes.
 */
static int deflateStateCheck      (PREFIX3(stream) *strm);
Z_INTERNAL block_state deflate_stored(deflate_state *s, int flush);
Z_INTERNAL block_state deflate_fast  (deflate_state *s, int flush);
Z_INTERNAL block_state deflate_quick (deflate_state *s, int flush);
#ifndef NO_MEDIUM_STRATEGY
Z_INTERNAL block_state deflate_medium(deflate_state *s, int flush);
#endif
Z_INTERNAL block_state deflate_slow  (deflate_state *s, int flush);
Z_INTERNAL block_state deflate_rle   (deflate_state *s, int flush);
Z_INTERNAL block_state deflate_huff  (deflate_state *s, int flush);
static void lm_set_level         (deflate_state *s, int level);
static void lm_init              (deflate_state *s);

/* ===========================================================================
 * Local data
 */

/* Values for max_lazy_match, good_match and max_chain_length, depending on
 * the desired pack level (0..9). The values given below have been tuned to
 * exclude worst case performance for pathological files. Better values may be
 * found for specific files.
 */
typedef struct config_s {
    uint16_t good_length; /* reduce lazy search above this match length */
    uint16_t max_lazy;    /* do not perform lazy search above this match length */
    uint16_t nice_length; /* quit search above this match length */
    uint16_t max_chain;
    compress_func func;
} config;

static const config configuration_table[10] = {
/*      good lazy nice chain */
/* 0 */ {0,    0,  0,    0, deflate_stored},  /* store only */

#ifdef NO_QUICK_STRATEGY
/* 1 */ {4,    4,  8,    4, deflate_fast}, /* max speed, no lazy matches */
/* 2 */ {4,    5, 16,    8, deflate_fast},
#else
/* 1 */ {0,    0,  0,    0, deflate_quick},
/* 2 */ {4,    4,  8,    4, deflate_fast}, /* max speed, no lazy matches */
#endif

#ifdef NO_MEDIUM_STRATEGY
/* 3 */ {4,    6, 32,   32, deflate_fast},
/* 4 */ {4,    4, 16,   16, deflate_slow},  /* lazy matches */
/* 5 */ {8,   16, 32,   32, deflate_slow},
/* 6 */ {8,   16, 128, 128, deflate_slow},
#else
/* 3 */ {4,    6, 16,    6, deflate_medium},
/* 4 */ {4,   12, 32,   24, deflate_medium},  /* lazy matches */
/* 5 */ {8,   16, 32,   32, deflate_medium},
/* 6 */ {8,   16, 128, 128, deflate_medium},
#endif

/* 7 */ {8,   32, 128,  256, deflate_slow},
/* 8 */ {32, 128, 258, 1024, deflate_slow},
/* 9 */ {32, 258, 258, 4096, deflate_slow}}; /* max compression */

/* Note: the deflate() code requires max_lazy >= STD_MIN_MATCH and max_chain >= 4
 * For deflate_fast() (levels <= 3) good is ignored and lazy has a different
 * meaning.
 */

/* rank Z_BLOCK between Z_NO_FLUSH and Z_PARTIAL_FLUSH */
#define RANK(f) (((f) * 2) - ((f) > 4 ? 9 : 0))


/* ===========================================================================
 * Initialize the hash table. prev[] will be initialized on the fly.
 */
#define CLEAR_HASH(s) do { \
    memset((unsigned char *)s->head, 0, HASH_SIZE * sizeof(*s->head)); \
  } while (0)


#ifdef DEF_ALLOC_DEBUG
#  include <stdio.h>
#  define LOGSZ(name,size)           fprintf(stderr, "%s is %d bytes\n", name, size)
#  define LOGSZP(name,size,loc,pad)  fprintf(stderr, "%s is %d bytes, offset %d, padded %d\n", name, size, loc, pad)
#  define LOGSZPL(name,size,loc,pad) fprintf(stderr, "%s is %d bytes, offset %ld, padded %d\n", name, size, loc, pad)
#else
#  define LOGSZ(name,size)
#  define LOGSZP(name,size,loc,pad)
#  define LOGSZPL(name,size,loc,pad)
#endif

/* ===========================================================================
 * Allocate a big buffer and divide it up into the various buffers deflate needs.
 * Handles alignment of allocated buffer and alignment of individual buffers.
 */
Z_INTERNAL deflate_allocs* alloc_deflate(PREFIX3(stream) *strm, int windowBits, int lit_bufsize) {
    int curr_size = 0;

    /* Define sizes */
    int window_size = DEFLATE_ADJUST_WINDOW_SIZE((1 << windowBits) * 2);
    int prev_size = (1 << windowBits) * (int)sizeof(Pos);
    int head_size = HASH_SIZE * sizeof(Pos);
    int pending_size = lit_bufsize * LIT_BUFS;
    int state_size = sizeof(deflate_state);
    int alloc_size = sizeof(deflate_allocs);

    /* Calculate relative buffer positions and paddings */
    LOGSZP("window", window_size, PAD_WINDOW(curr_size), PADSZ(curr_size,WINDOW_PAD_SIZE));
    int window_pos = PAD_WINDOW(curr_size);
    curr_size = window_pos + window_size;

    LOGSZP("prev", prev_size, PAD_64(curr_size), PADSZ(curr_size,64));
    int prev_pos = PAD_64(curr_size);
    curr_size = prev_pos + prev_size;

    LOGSZP("head", head_size, PAD_64(curr_size), PADSZ(curr_size,64));
    int head_pos = PAD_64(curr_size);
    curr_size = head_pos + head_size;

    LOGSZP("pending", pending_size, PAD_64(curr_size), PADSZ(curr_size,64));
    int pending_pos = PAD_64(curr_size);
    curr_size = pending_pos + pending_size;

    LOGSZP("state", state_size, PAD_64(curr_size), PADSZ(curr_size,64));
    int state_pos = PAD_64(curr_size);
    curr_size = state_pos + state_size;

    LOGSZP("alloc", alloc_size, PAD_16(curr_size), PADSZ(curr_size,16));
    int alloc_pos = PAD_16(curr_size);
    curr_size = alloc_pos + alloc_size;

    /* Add 64-1 or 4096-1 to allow window alignment, and round size of buffer up to multiple of 64 */
    int total_size = PAD_64(curr_size + (WINDOW_PAD_SIZE - 1));

    /* Allocate buffer, align to 64-byte cacheline, and zerofill the resulting buffer */
    char *original_buf = (char *)strm->zalloc(strm->opaque, 1, total_size);
    if (original_buf == NULL)
        return NULL;

    char *buff = (char *)HINT_ALIGNED_WINDOW((char *)PAD_WINDOW(original_buf));
    LOGSZPL("Buffer alloc", total_size, PADSZ((uintptr_t)original_buf,WINDOW_PAD_SIZE), PADSZ(curr_size,WINDOW_PAD_SIZE));

    /* Initialize alloc_bufs */
    deflate_allocs *alloc_bufs  = (struct deflate_allocs_s *)(buff + alloc_pos);
    alloc_bufs->buf_start = original_buf;
    alloc_bufs->zfree = strm->zfree;

    /* Assign buffers */
    alloc_bufs->window = (unsigned char *)HINT_ALIGNED_WINDOW(buff + window_pos);
    alloc_bufs->prev = (Pos *)HINT_ALIGNED_64(buff + prev_pos);
    alloc_bufs->head = (Pos *)HINT_ALIGNED_64(buff + head_pos);
    alloc_bufs->pending_buf = (unsigned char *)HINT_ALIGNED_64(buff + pending_pos);
    alloc_bufs->state = (deflate_state *)HINT_ALIGNED_16(buff + state_pos);

    memset((char *)alloc_bufs->prev, 0, prev_size);

    return alloc_bufs;
}

/* ===========================================================================
 * Free all allocated deflate buffers
 */
static inline void free_deflate(PREFIX3(stream) *strm) {
    deflate_state *state = (deflate_state *)strm->state;

    if (state->alloc_bufs != NULL) {
        deflate_allocs *alloc_bufs = state->alloc_bufs;
        alloc_bufs->zfree(strm->opaque, alloc_bufs->buf_start);
        strm->state = NULL;
    }
}

/* ===========================================================================
 * Initialize deflate state and buffers.
 * This function is hidden in ZLIB_COMPAT builds.
 */
int32_t ZNG_CONDEXPORT PREFIX(deflateInit2)(PREFIX3(stream) *strm, int32_t level, int32_t method, int32_t windowBits,
                                            int32_t memLevel, int32_t strategy) {
    /* Todo: ignore strm->next_in if we use it as window */
    deflate_state *s;
    int wrap = 1;

    /* Initialize functable */
    FUNCTABLE_INIT;

    if (strm == NULL)
        return Z_STREAM_ERROR;

    strm->msg = NULL;
    if (strm->zalloc == NULL) {
        strm->zalloc = PREFIX(zcalloc);
        strm->opaque = NULL;
    }
    if (strm->zfree == NULL)
        strm->zfree = PREFIX(zcfree);

    if (level == Z_DEFAULT_COMPRESSION)
        level = 6;

    if (windowBits < 0) { /* suppress zlib wrapper */
        wrap = 0;
        if (windowBits < -MAX_WBITS)
            return Z_STREAM_ERROR;
        windowBits = -windowBits;
#ifdef GZIP
    } else if (windowBits > MAX_WBITS) {
        wrap = 2;       /* write gzip wrapper instead */
        windowBits -= 16;
#endif
    }
    if (memLevel < 1 || memLevel > MAX_MEM_LEVEL || method != Z_DEFLATED || windowBits < MIN_WBITS ||
        windowBits > MAX_WBITS || level < 0 || level > 9 || strategy < 0 || strategy > Z_FIXED ||
        (windowBits == 8 && wrap != 1)) {
        return Z_STREAM_ERROR;
    }
    if (windowBits == 8)
        windowBits = 9;  /* until 256-byte window bug fixed */

    /* Allocate buffers */
    int lit_bufsize = 1 << (memLevel + 6);
    deflate_allocs *alloc_bufs = alloc_deflate(strm, windowBits, lit_bufsize);
    if (alloc_bufs == NULL)
        return Z_MEM_ERROR;

    s = alloc_bufs->state;
    s->alloc_bufs = alloc_bufs;
    s->window = alloc_bufs->window;
    s->prev = alloc_bufs->prev;
    s->head = alloc_bufs->head;
    s->pending_buf = alloc_bufs->pending_buf;

    strm->state = (struct internal_state *)s;
    s->strm = strm;
    s->status = INIT_STATE;     /* to pass state test in deflateReset() */

    s->wrap = wrap;
    s->gzhead = NULL;
    s->w_bits = (unsigned int)windowBits;
    s->w_size = 1 << s->w_bits;
    s->w_mask = s->w_size - 1;

    s->high_water = 0;      /* nothing written to s->window yet */

    s->lit_bufsize = lit_bufsize; /* 16K elements by default */

    /* We overlay pending_buf and sym_buf. This works since the average size
     * for length/distance pairs over any compressed block is assured to be 31
     * bits or less.
     *
     * Analysis: The longest fixed codes are a length code of 8 bits plus 5
     * extra bits, for lengths 131 to 257. The longest fixed distance codes are
     * 5 bits plus 13 extra bits, for distances 16385 to 32768. The longest
     * possible fixed-codes length/distance pair is then 31 bits total.
     *
     * sym_buf starts one-fourth of the way into pending_buf. So there are
     * three bytes in sym_buf for every four bytes in pending_buf. Each symbol
     * in sym_buf is three bytes -- two for the distance and one for the
     * literal/length. As each symbol is consumed, the pointer to the next
     * sym_buf value to read moves forward three bytes. From that symbol, up to
     * 31 bits are written to pending_buf. The closest the written pending_buf
     * bits gets to the next sym_buf symbol to read is just before the last
     * code is written. At that time, 31*(n-2) bits have been written, just
     * after 24*(n-2) bits have been consumed from sym_buf. sym_buf starts at
     * 8*n bits into pending_buf. (Note that the symbol buffer fills when n-1
     * symbols are written.) The closest the writing gets to what is unread is
     * then n+14 bits. Here n is lit_bufsize, which is 16384 by default, and
     * can range from 128 to 32768.
     *
     * Therefore, at a minimum, there are 142 bits of space between what is
     * written and what is read in the overlain buffers, so the symbols cannot
     * be overwritten by the compressed data. That space is actually 139 bits,
     * due to the three-bit fixed-code block header.
     *
     * That covers the case where either Z_FIXED is specified, forcing fixed
     * codes, or when the use of fixed codes is chosen, because that choice
     * results in a smaller compressed block than dynamic codes. That latter
     * condition then assures that the above analysis also covers all dynamic
     * blocks. A dynamic-code block will only be chosen to be emitted if it has
     * fewer bits than a fixed-code block would for the same set of symbols.
     * Therefore its average symbol length is assured to be less than 31. So
     * the compressed data for a dynamic block also cannot overwrite the
     * symbols from which it is being constructed.
     */

    s->pending_buf_size = s->lit_bufsize * 4;

#ifdef LIT_MEM
    s->d_buf = (uint16_t *)(s->pending_buf + (s->lit_bufsize << 1));
    s->l_buf = s->pending_buf + (s->lit_bufsize << 2);
    s->sym_end = s->lit_bufsize - 1;
#else
    s->sym_buf = s->pending_buf + s->lit_bufsize;
    s->sym_end = (s->lit_bufsize - 1) * 3;
#endif
    /* We avoid equality with lit_bufsize*3 because of wraparound at 64K
     * on 16 bit machines and because stored blocks are restricted to
     * 64K-1 bytes.
     */

    s->level = level;
    s->strategy = strategy;
    s->block_open = 0;
    s->reproducible = 0;

    return PREFIX(deflateReset)(strm);
}

#ifndef ZLIB_COMPAT
int32_t Z_EXPORT PREFIX(deflateInit)(PREFIX3(stream) *strm, int32_t level) {
    return PREFIX(deflateInit2)(strm, level, Z_DEFLATED, MAX_WBITS, DEF_MEM_LEVEL, Z_DEFAULT_STRATEGY);
}
#endif

/* Function used by zlib.h and zlib-ng version 2.0 macros */
int32_t Z_EXPORT PREFIX(deflateInit_)(PREFIX3(stream) *strm, int32_t level, const char *version, int32_t stream_size) {
    if (CHECK_VER_STSIZE(version, stream_size))
        return Z_VERSION_ERROR;
    return PREFIX(deflateInit2)(strm, level, Z_DEFLATED, MAX_WBITS, DEF_MEM_LEVEL, Z_DEFAULT_STRATEGY);
}

/* Function used by zlib.h and zlib-ng version 2.0 macros */
int32_t Z_EXPORT PREFIX(deflateInit2_)(PREFIX3(stream) *strm, int32_t level, int32_t method, int32_t windowBits,
                           int32_t memLevel, int32_t strategy, const char *version, int32_t stream_size) {
    if (CHECK_VER_STSIZE(version, stream_size))
        return Z_VERSION_ERROR;
    return PREFIX(deflateInit2)(strm, level, method, windowBits, memLevel, strategy);
}

/* =========================================================================
 * Check for a valid deflate stream state. Return 0 if ok, 1 if not.
 */
static int deflateStateCheck(PREFIX3(stream) *strm) {
    deflate_state *s;
    if (strm == NULL || strm->zalloc == (alloc_func)0 || strm->zfree == (free_func)0)
        return 1;
    s = strm->state;
    if (s == NULL || s->alloc_bufs == NULL || s->strm != strm || (s->status < INIT_STATE || s->status > MAX_STATE))
        return 1;
    return 0;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateSetDictionary)(PREFIX3(stream) *strm, const uint8_t *dictionary, uint32_t dictLength) {
    deflate_state *s;
    unsigned int str, n;
    int wrap;
    uint32_t avail;
    const unsigned char *next;

    if (deflateStateCheck(strm) || dictionary == NULL)
        return Z_STREAM_ERROR;
    s = strm->state;
    wrap = s->wrap;
    if (wrap == 2 || (wrap == 1 && s->status != INIT_STATE) || s->lookahead)
        return Z_STREAM_ERROR;

    /* when using zlib wrappers, compute Adler-32 for provided dictionary */
    if (wrap == 1)
        strm->adler = FUNCTABLE_CALL(adler32)(strm->adler, dictionary, dictLength);
    DEFLATE_SET_DICTIONARY_HOOK(strm, dictionary, dictLength);  /* hook for IBM Z DFLTCC */
    s->wrap = 0;                    /* avoid computing Adler-32 in read_buf */

    /* if dictionary would fill window, just replace the history */
    if (dictLength >= s->w_size) {
        if (wrap == 0) {            /* already empty otherwise */
            CLEAR_HASH(s);
            s->strstart = 0;
            s->block_start = 0;
            s->insert = 0;
        }
        dictionary += dictLength - s->w_size;  /* use the tail */
        dictLength = s->w_size;
    }

    /* insert dictionary into window and hash */
    avail = strm->avail_in;
    next = strm->next_in;
    strm->avail_in = dictLength;
    strm->next_in = (z_const unsigned char *)dictionary;
    PREFIX(fill_window)(s);
    while (s->lookahead >= STD_MIN_MATCH) {
        str = s->strstart;
        n = s->lookahead - (STD_MIN_MATCH - 1);
        s->insert_string(s, str, n);
        s->strstart = str + n;
        s->lookahead = STD_MIN_MATCH - 1;
        PREFIX(fill_window)(s);
    }
    s->strstart += s->lookahead;
    s->block_start = (int)s->strstart;
    s->insert = s->lookahead;
    s->lookahead = 0;
    s->prev_length = 0;
    s->match_available = 0;
    strm->next_in = (z_const unsigned char *)next;
    strm->avail_in = avail;
    s->wrap = wrap;
    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateGetDictionary)(PREFIX3(stream) *strm, uint8_t *dictionary, uint32_t *dictLength) {
    deflate_state *s;
    unsigned int len;

    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    DEFLATE_GET_DICTIONARY_HOOK(strm, dictionary, dictLength);  /* hook for IBM Z DFLTCC */
    s = strm->state;
    len = s->strstart + s->lookahead;
    if (len > s->w_size)
        len = s->w_size;
    if (dictionary != NULL && len)
        memcpy(dictionary, s->window + s->strstart + s->lookahead - len, len);
    if (dictLength != NULL)
        *dictLength = len;
    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateResetKeep)(PREFIX3(stream) *strm) {
    deflate_state *s;

    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;

    strm->total_in = strm->total_out = 0;
    strm->msg = NULL; /* use zfree if we ever allocate msg dynamically */
    strm->data_type = Z_UNKNOWN;

    s = (deflate_state *)strm->state;
    s->pending = 0;
    s->pending_out = s->pending_buf;

    if (s->wrap < 0)
        s->wrap = -s->wrap; /* was made negative by deflate(..., Z_FINISH); */

    s->status =
#ifdef GZIP
        s->wrap == 2 ? GZIP_STATE :
#endif
        INIT_STATE;

#ifdef GZIP
    if (s->wrap == 2) {
        strm->adler = FUNCTABLE_CALL(crc32_fold_reset)(&s->crc_fold);
    } else
#endif
        strm->adler = ADLER32_INITIAL_VALUE;
    s->last_flush = -2;

    zng_tr_init(s);

    DEFLATE_RESET_KEEP_HOOK(strm);  /* hook for IBM Z DFLTCC */

    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateReset)(PREFIX3(stream) *strm) {
    int ret = PREFIX(deflateResetKeep)(strm);
    if (ret == Z_OK)
        lm_init(strm->state);
    return ret;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateSetHeader)(PREFIX3(stream) *strm, PREFIX(gz_headerp) head) {
    if (deflateStateCheck(strm) || strm->state->wrap != 2)
        return Z_STREAM_ERROR;
    strm->state->gzhead = head;
    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflatePending)(PREFIX3(stream) *strm, uint32_t *pending, int32_t *bits) {
    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    if (pending != NULL)
        *pending = strm->state->pending;
    if (bits != NULL)
        *bits = strm->state->bi_valid;
    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflatePrime)(PREFIX3(stream) *strm, int32_t bits, int32_t value) {
    deflate_state *s;
    uint64_t value64 = (uint64_t)value;
    int32_t put;

    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    s = strm->state;

#ifdef LIT_MEM
    if (bits < 0 || bits > BIT_BUF_SIZE ||
        (unsigned char *)s->d_buf < s->pending_out + ((BIT_BUF_SIZE + 7) >> 3))
        return Z_BUF_ERROR;
#else
    if (bits < 0 || bits > BIT_BUF_SIZE || bits > (int32_t)(sizeof(value) << 3) ||
        s->sym_buf < s->pending_out + ((BIT_BUF_SIZE + 7) >> 3))
        return Z_BUF_ERROR;
#endif

    do {
        put = BIT_BUF_SIZE - s->bi_valid;
        put = MIN(put, bits);

        if (s->bi_valid == 0)
            s->bi_buf = value64;
        else
            s->bi_buf |= (value64 & ((UINT64_C(1) << put) - 1)) << s->bi_valid;
        s->bi_valid += put;
        zng_tr_flush_bits(s);
        value64 >>= put;
        bits -= put;
    } while (bits);
    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateParams)(PREFIX3(stream) *strm, int32_t level, int32_t strategy) {
    deflate_state *s;
    compress_func func;
    int hook_flush = Z_NO_FLUSH;

    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    s = strm->state;

    if (level == Z_DEFAULT_COMPRESSION)
        level = 6;
    if (level < 0 || level > 9 || strategy < 0 || strategy > Z_FIXED)
        return Z_STREAM_ERROR;
    DEFLATE_PARAMS_HOOK(strm, level, strategy, &hook_flush);  /* hook for IBM Z DFLTCC */
    func = configuration_table[s->level].func;

    if (((strategy != s->strategy || func != configuration_table[level].func) && s->last_flush != -2)
        || hook_flush != Z_NO_FLUSH) {
        /* Flush the last buffer. Use Z_BLOCK mode, unless the hook requests a "stronger" one. */
        int flush = RANK(hook_flush) > RANK(Z_BLOCK) ? hook_flush : Z_BLOCK;
        int err = PREFIX(deflate)(strm, flush);
        if (err == Z_STREAM_ERROR)
            return err;
        if (strm->avail_in || ((int)s->strstart - s->block_start) + s->lookahead || !DEFLATE_DONE(strm, flush))
            return Z_BUF_ERROR;
    }
    if (s->level != level) {
        if (s->level == 0 && s->matches != 0) {
            if (s->matches == 1) {
                FUNCTABLE_CALL(slide_hash)(s);
            } else {
                CLEAR_HASH(s);
            }
            s->matches = 0;
        }

        lm_set_level(s, level);
    }
    s->strategy = strategy;
    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateTune)(PREFIX3(stream) *strm, int32_t good_length, int32_t max_lazy, int32_t nice_length, int32_t max_chain) {
    deflate_state *s;

    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    s = strm->state;
    s->good_match = (unsigned int)good_length;
    s->max_lazy_match = (unsigned int)max_lazy;
    s->nice_match = nice_length;
    s->max_chain_length = (unsigned int)max_chain;
    return Z_OK;
}

/* =========================================================================
 * For the default windowBits of 15 and memLevel of 8, this function returns
 * a close to exact, as well as small, upper bound on the compressed size.
 * They are coded as constants here for a reason--if the #define's are
 * changed, then this function needs to be changed as well.  The return
 * value for 15 and 8 only works for those exact settings.
 *
 * For any setting other than those defaults for windowBits and memLevel,
 * the value returned is a conservative worst case for the maximum expansion
 * resulting from using fixed blocks instead of stored blocks, which deflate
 * can emit on compressed data for some combinations of the parameters.
 *
 * This function could be more sophisticated to provide closer upper bounds for
 * every combination of windowBits and memLevel.  But even the conservative
 * upper bound of about 14% expansion does not seem onerous for output buffer
 * allocation.
 */
unsigned long Z_EXPORT PREFIX(deflateBound)(PREFIX3(stream) *strm, unsigned long sourceLen) {
    deflate_state *s;
    unsigned long complen, wraplen;

    /* conservative upper bound for compressed data */
    complen = sourceLen + ((sourceLen + 7) >> 3) + ((sourceLen + 63) >> 6) + 5;
    DEFLATE_BOUND_ADJUST_COMPLEN(strm, complen, sourceLen);  /* hook for IBM Z DFLTCC */

    /* if can't get parameters, return conservative bound plus zlib wrapper */
    if (deflateStateCheck(strm))
        return complen + 6;

    /* compute wrapper length */
    s = strm->state;
    switch (s->wrap) {
    case 0:                                 /* raw deflate */
        wraplen = 0;
        break;
    case 1:                                 /* zlib wrapper */
        wraplen = ZLIB_WRAPLEN + (s->strstart ? 4 : 0);
        break;
#ifdef GZIP
    case 2:                                 /* gzip wrapper */
        wraplen = GZIP_WRAPLEN;
        if (s->gzhead != NULL) {            /* user-supplied gzip header */
            unsigned char *str;
            if (s->gzhead->extra != NULL) {
                wraplen += 2 + s->gzhead->extra_len;
            }
            str = s->gzhead->name;
            if (str != NULL) {
                do {
                    wraplen++;
                } while (*str++);
            }
            str = s->gzhead->comment;
            if (str != NULL) {
                do {
                    wraplen++;
                } while (*str++);
            }
            if (s->gzhead->hcrc)
                wraplen += 2;
        }
        break;
#endif
    default:                                /* for compiler happiness */
        wraplen = ZLIB_WRAPLEN;
    }

    /* if not default parameters, return conservative bound */
    if (DEFLATE_NEED_CONSERVATIVE_BOUND(strm) ||  /* hook for IBM Z DFLTCC */
            s->w_bits != MAX_WBITS || HASH_BITS < 15) {
        if (s->level == 0) {
            /* upper bound for stored blocks with length 127 (memLevel == 1) --
               ~4% overhead plus a small constant */
            complen = sourceLen + (sourceLen >> 5) + (sourceLen >> 7) + (sourceLen >> 11) + 7;
        }

        return complen + wraplen;
    }

#ifndef NO_QUICK_STRATEGY
    return sourceLen                       /* The source size itself */
      + (sourceLen == 0 ? 1 : 0)           /* Always at least one byte for any input */
      + (sourceLen < 9 ? 1 : 0)            /* One extra byte for lengths less than 9 */
      + DEFLATE_QUICK_OVERHEAD(sourceLen)  /* Source encoding overhead, padded to next full byte */
      + DEFLATE_BLOCK_OVERHEAD             /* Deflate block overhead bytes */
      + wraplen;                           /* none, zlib or gzip wrapper */
#else
    return sourceLen + (sourceLen >> 4) + 7 + wraplen;
#endif
}

/* =========================================================================
 * Flush as much pending output as possible. See flush_pending_inline()
 */
Z_INTERNAL void PREFIX(flush_pending)(PREFIX3(stream) *strm) {
    flush_pending_inline(strm);
}

/* ===========================================================================
 * Update the header CRC with the bytes s->pending_buf[beg..s->pending - 1].
 */
#define HCRC_UPDATE(beg) \
    do { \
        if (s->gzhead->hcrc && s->pending > (beg)) \
            strm->adler = PREFIX(crc32)(strm->adler, s->pending_buf + (beg), s->pending - (beg)); \
    } while (0)

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflate)(PREFIX3(stream) *strm, int32_t flush) {
    int32_t old_flush; /* value of flush param for previous deflate call */
    deflate_state *s;

    if (deflateStateCheck(strm) || flush > Z_BLOCK || flush < 0)
        return Z_STREAM_ERROR;
    s = strm->state;

    if (strm->next_out == NULL || (strm->avail_in != 0 && strm->next_in == NULL)
        || (s->status == FINISH_STATE && flush != Z_FINISH)) {
        ERR_RETURN(strm, Z_STREAM_ERROR);
    }
    if (strm->avail_out == 0) {
        ERR_RETURN(strm, Z_BUF_ERROR);
    }

    old_flush = s->last_flush;
    s->last_flush = flush;

    /* Flush as much pending output as possible */
    if (s->pending != 0) {
        flush_pending_inline(strm);
        if (strm->avail_out == 0) {
            /* Since avail_out is 0, deflate will be called again with
             * more output space, but possibly with both pending and
             * avail_in equal to zero. There won't be anything to do,
             * but this is not an error situation so make sure we
             * return OK instead of BUF_ERROR at next call of deflate:
             */
            s->last_flush = -1;
            return Z_OK;
        }

        /* Make sure there is something to do and avoid duplicate consecutive
         * flushes. For repeated and useless calls with Z_FINISH, we keep
         * returning Z_STREAM_END instead of Z_BUF_ERROR.
         */
    } else if (strm->avail_in == 0 && RANK(flush) <= RANK(old_flush) && flush != Z_FINISH) {
        ERR_RETURN(strm, Z_BUF_ERROR);
    }

    /* User must not provide more input after the first FINISH: */
    if (s->status == FINISH_STATE && strm->avail_in != 0)   {
        ERR_RETURN(strm, Z_BUF_ERROR);
    }

    /* Write the header */
    if (s->status == INIT_STATE && s->wrap == 0)
        s->status = BUSY_STATE;
    if (s->status == INIT_STATE) {
        /* zlib header */
        unsigned int header = (Z_DEFLATED + ((s->w_bits-8)<<4)) << 8;
        unsigned int level_flags;

        if (s->strategy >= Z_HUFFMAN_ONLY || s->level < 2)
            level_flags = 0;
        else if (s->level < 6)
            level_flags = 1;
        else if (s->level == 6)
            level_flags = 2;
        else
            level_flags = 3;
        header |= (level_flags << 6);
        if (s->strstart != 0)
            header |= PRESET_DICT;
        header += 31 - (header % 31);

        put_short_msb(s, (uint16_t)header);

        /* Save the adler32 of the preset dictionary: */
        if (s->strstart != 0)
            put_uint32_msb(s, strm->adler);
        strm->adler = ADLER32_INITIAL_VALUE;
        s->status = BUSY_STATE;

        /* Compression must start with an empty pending buffer */
        PREFIX(flush_pending)(strm);
        if (s->pending != 0) {
            s->last_flush = -1;
            return Z_OK;
        }
    }
#ifdef GZIP
    if (s->status == GZIP_STATE) {
        /* gzip header */
        FUNCTABLE_CALL(crc32_fold_reset)(&s->crc_fold);
        put_byte(s, 31);
        put_byte(s, 139);
        put_byte(s, 8);
        if (s->gzhead == NULL) {
            put_uint32(s, 0);
            put_byte(s, 0);
            put_byte(s, s->level == 9 ? 2 :
                     (s->strategy >= Z_HUFFMAN_ONLY || s->level < 2 ? 4 : 0));
            put_byte(s, OS_CODE);
            s->status = BUSY_STATE;

            /* Compression must start with an empty pending buffer */
            PREFIX(flush_pending)(strm);
            if (s->pending != 0) {
                s->last_flush = -1;
                return Z_OK;
            }
        } else {
            put_byte(s, (s->gzhead->text ? 1 : 0) +
                     (s->gzhead->hcrc ? 2 : 0) +
                     (s->gzhead->extra == NULL ? 0 : 4) +
                     (s->gzhead->name == NULL ? 0 : 8) +
                     (s->gzhead->comment == NULL ? 0 : 16)
                     );
            put_uint32(s, s->gzhead->time);
            put_byte(s, s->level == 9 ? 2 : (s->strategy >= Z_HUFFMAN_ONLY || s->level < 2 ? 4 : 0));
            put_byte(s, s->gzhead->os & 0xff);
            if (s->gzhead->extra != NULL)
                put_short(s, (uint16_t)s->gzhead->extra_len);
            if (s->gzhead->hcrc)
                strm->adler = PREFIX(crc32)(strm->adler, s->pending_buf, s->pending);
            s->gzindex = 0;
            s->status = EXTRA_STATE;
        }
    }
    if (s->status == EXTRA_STATE) {
        if (s->gzhead->extra != NULL) {
            uint32_t beg = s->pending;   /* start of bytes to update crc */
            uint32_t left = (s->gzhead->extra_len & 0xffff) - s->gzindex;

            while (s->pending + left > s->pending_buf_size) {
                uint32_t copy = s->pending_buf_size - s->pending;
                memcpy(s->pending_buf + s->pending, s->gzhead->extra + s->gzindex, copy);
                s->pending = s->pending_buf_size;
                HCRC_UPDATE(beg);
                s->gzindex += copy;
                PREFIX(flush_pending)(strm);
                if (s->pending != 0) {
                    s->last_flush = -1;
                    return Z_OK;
                }
                beg = 0;
                left -= copy;
            }
            memcpy(s->pending_buf + s->pending, s->gzhead->extra + s->gzindex, left);
            s->pending += left;
            HCRC_UPDATE(beg);
            s->gzindex = 0;
        }
        s->status = NAME_STATE;
    }
    if (s->status == NAME_STATE) {
        if (s->gzhead->name != NULL) {
            uint32_t beg = s->pending;   /* start of bytes to update crc */
            unsigned char val;

            do {
                if (s->pending == s->pending_buf_size) {
                    HCRC_UPDATE(beg);
                    PREFIX(flush_pending)(strm);
                    if (s->pending != 0) {
                        s->last_flush = -1;
                        return Z_OK;
                    }
                    beg = 0;
                }
                val = s->gzhead->name[s->gzindex++];
                put_byte(s, val);
            } while (val != 0);
            HCRC_UPDATE(beg);
            s->gzindex = 0;
        }
        s->status = COMMENT_STATE;
    }
    if (s->status == COMMENT_STATE) {
        if (s->gzhead->comment != NULL) {
            uint32_t beg = s->pending;  /* start of bytes to update crc */
            unsigned char val;

            do {
                if (s->pending == s->pending_buf_size) {
                    HCRC_UPDATE(beg);
                    PREFIX(flush_pending)(strm);
                    if (s->pending != 0) {
                        s->last_flush = -1;
                        return Z_OK;
                    }
                    beg = 0;
                }
                val = s->gzhead->comment[s->gzindex++];
                put_byte(s, val);
            } while (val != 0);
            HCRC_UPDATE(beg);
        }
        s->status = HCRC_STATE;
    }
    if (s->status == HCRC_STATE) {
        if (s->gzhead->hcrc) {
            if (s->pending + 2 > s->pending_buf_size) {
                PREFIX(flush_pending)(strm);
                if (s->pending != 0) {
                    s->last_flush = -1;
                    return Z_OK;
                }
            }
            put_short(s, (uint16_t)strm->adler);
            FUNCTABLE_CALL(crc32_fold_reset)(&s->crc_fold);
        }
        s->status = BUSY_STATE;

        /* Compression must start with an empty pending buffer */
        flush_pending_inline(strm);
        if (s->pending != 0) {
            s->last_flush = -1;
            return Z_OK;
        }
    }
#endif

    /* Start a new block or continue the current one.
     */
    if (strm->avail_in != 0 || s->lookahead != 0 || (flush != Z_NO_FLUSH && s->status != FINISH_STATE)) {
        block_state bstate;

        bstate = DEFLATE_HOOK(strm, flush, &bstate) ? bstate :  /* hook for IBM Z DFLTCC */
                 s->level == 0 ? deflate_stored(s, flush) :
                 s->strategy == Z_HUFFMAN_ONLY ? deflate_huff(s, flush) :
                 s->strategy == Z_RLE ? deflate_rle(s, flush) :
                 (*(configuration_table[s->level].func))(s, flush);

        if (bstate == finish_started || bstate == finish_done) {
            s->status = FINISH_STATE;
        }
        if (bstate == need_more || bstate == finish_started) {
            if (strm->avail_out == 0) {
                s->last_flush = -1; /* avoid BUF_ERROR next call, see above */
            }
            return Z_OK;
            /* If flush != Z_NO_FLUSH && avail_out == 0, the next call
             * of deflate should use the same flush parameter to make sure
             * that the flush is complete. So we don't have to output an
             * empty block here, this will be done at next call. This also
             * ensures that for a very small output buffer, we emit at most
             * one empty block.
             */
        }
        if (bstate == block_done) {
            if (flush == Z_PARTIAL_FLUSH) {
                zng_tr_align(s);
            } else if (flush != Z_BLOCK) { /* FULL_FLUSH or SYNC_FLUSH */
                zng_tr_stored_block(s, (char*)0, 0L, 0);
                /* For a full flush, this empty block will be recognized
                 * as a special marker by inflate_sync().
                 */
                if (flush == Z_FULL_FLUSH) {
                    CLEAR_HASH(s);             /* forget history */
                    if (s->lookahead == 0) {
                        s->strstart = 0;
                        s->block_start = 0;
                        s->insert = 0;
                    }
                }
            }
            PREFIX(flush_pending)(strm);
            if (strm->avail_out == 0) {
                s->last_flush = -1; /* avoid BUF_ERROR at next call, see above */
                return Z_OK;
            }
        }
    }

    if (flush != Z_FINISH)
        return Z_OK;

    /* Write the trailer */
#ifdef GZIP
    if (s->wrap == 2) {
        strm->adler = FUNCTABLE_CALL(crc32_fold_final)(&s->crc_fold);

        put_uint32(s, strm->adler);
        put_uint32(s, (uint32_t)strm->total_in);
    } else
#endif
    {
        if (s->wrap == 1)
            put_uint32_msb(s, strm->adler);
    }
    flush_pending_inline(strm);
    /* If avail_out is zero, the application will call deflate again
     * to flush the rest.
     */
    if (s->wrap > 0)
        s->wrap = -s->wrap; /* write the trailer only once! */
    if (s->pending == 0) {
        Assert(s->bi_valid == 0, "bi_buf not flushed");
        return Z_STREAM_END;
    }
    return Z_OK;
}

/* ========================================================================= */
int32_t Z_EXPORT PREFIX(deflateEnd)(PREFIX3(stream) *strm) {
    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;

    int32_t status = strm->state->status;

    /* Free allocated buffers */
    free_deflate(strm);

    return status == BUSY_STATE ? Z_DATA_ERROR : Z_OK;
}

/* =========================================================================
 * Copy the source state to the destination state.
 */
int32_t Z_EXPORT PREFIX(deflateCopy)(PREFIX3(stream) *dest, PREFIX3(stream) *source) {
    deflate_state *ds;
    deflate_state *ss;

    if (deflateStateCheck(source) || dest == NULL)
        return Z_STREAM_ERROR;

    ss = source->state;

    memcpy((void *)dest, (void *)source, sizeof(PREFIX3(stream)));

    deflate_allocs *alloc_bufs = alloc_deflate(dest, ss->w_bits, ss->lit_bufsize);
    if (alloc_bufs == NULL)
        return Z_MEM_ERROR;

    ds = alloc_bufs->state;

    dest->state = (struct internal_state *) ds;
    memcpy(ds, ss, sizeof(deflate_state));
    ds->strm = dest;

    ds->alloc_bufs = alloc_bufs;
    ds->window = alloc_bufs->window;
    ds->prev = alloc_bufs->prev;
    ds->head = alloc_bufs->head;
    ds->pending_buf = alloc_bufs->pending_buf;

    if (ds->window == NULL || ds->prev == NULL || ds->head == NULL || ds->pending_buf == NULL) {
        PREFIX(deflateEnd)(dest);
        return Z_MEM_ERROR;
    }

    memcpy(ds->window, ss->window, DEFLATE_ADJUST_WINDOW_SIZE(ds->w_size * 2 * sizeof(unsigned char)));
    memcpy((void *)ds->prev, (void *)ss->prev, ds->w_size * sizeof(Pos));
    memcpy((void *)ds->head, (void *)ss->head, HASH_SIZE * sizeof(Pos));
    memcpy(ds->pending_buf, ss->pending_buf, ds->lit_bufsize * LIT_BUFS);

    ds->pending_out = ds->pending_buf + (ss->pending_out - ss->pending_buf);
#ifdef LIT_MEM
    ds->d_buf = (uint16_t *)(ds->pending_buf + (ds->lit_bufsize << 1));
    ds->l_buf = ds->pending_buf + (ds->lit_bufsize << 2);
#else
    ds->sym_buf = ds->pending_buf + ds->lit_bufsize;
#endif

    ds->l_desc.dyn_tree = ds->dyn_ltree;
    ds->d_desc.dyn_tree = ds->dyn_dtree;
    ds->bl_desc.dyn_tree = ds->bl_tree;

    return Z_OK;
}

/* ===========================================================================
 * Set longest match variables based on level configuration
 */
static void lm_set_level(deflate_state *s, int level) {
    s->max_lazy_match   = configuration_table[level].max_lazy;
    s->good_match       = configuration_table[level].good_length;
    s->nice_match       = configuration_table[level].nice_length;
    s->max_chain_length = configuration_table[level].max_chain;

    /* Use rolling hash for deflate_slow algorithm with level 9. It allows us to
     * properly lookup different hash chains to speed up longest_match search. Since hashing
     * method changes depending on the level we cannot put this into functable. */
    if (s->max_chain_length > 1024) {
        s->update_hash = &update_hash_roll;
        s->insert_string = &insert_string_roll;
        s->quick_insert_string = &quick_insert_string_roll;
    } else {
        s->update_hash = update_hash;
        s->insert_string = insert_string;
        s->quick_insert_string = quick_insert_string;
    }

    s->level = level;
}

/* ===========================================================================
 * Initialize the "longest match" routines for a new zlib stream
 */
static void lm_init(deflate_state *s) {
    s->window_size = 2 * s->w_size;

    CLEAR_HASH(s);

    /* Set the default configuration parameters:
     */
    lm_set_level(s, s->level);

    s->strstart = 0;
    s->block_start = 0;
    s->lookahead = 0;
    s->insert = 0;
    s->prev_length = 0;
    s->match_available = 0;
    s->match_start = 0;
    s->ins_h = 0;
}

/* ===========================================================================
 * Fill the window when the lookahead becomes insufficient.
 * Updates strstart and lookahead.
 *
 * IN assertion: lookahead < MIN_LOOKAHEAD
 * OUT assertions: strstart <= window_size-MIN_LOOKAHEAD
 *    At least one byte has been read, or avail_in == 0; reads are
 *    performed for at least two bytes (required for the zip translate_eol
 *    option -- not supported here).
 */

void Z_INTERNAL PREFIX(fill_window)(deflate_state *s) {
    unsigned n;
    unsigned int more;    /* Amount of free space at the end of the window. */
    unsigned int wsize = s->w_size;

    Assert(s->lookahead < MIN_LOOKAHEAD, "already enough lookahead");

    do {
        more = s->window_size - s->lookahead - s->strstart;

        /* If the window is almost full and there is insufficient lookahead,
         * move the upper half to the lower one to make room in the upper half.
         */
        if (s->strstart >= wsize+MAX_DIST(s)) {
            memcpy(s->window, s->window+wsize, (unsigned)wsize);
            if (s->match_start >= wsize) {
                s->match_start -= wsize;
            } else {
                s->match_start = 0;
                s->prev_length = 0;
            }
            s->strstart    -= wsize; /* we now have strstart >= MAX_DIST */
            s->block_start -= (int)wsize;
            if (s->insert > s->strstart)
                s->insert = s->strstart;
            FUNCTABLE_CALL(slide_hash)(s);
            more += wsize;
        }
        if (s->strm->avail_in == 0)
            break;

        /* If there was no sliding:
         *    strstart <= WSIZE+MAX_DIST-1 && lookahead <= MIN_LOOKAHEAD - 1 &&
         *    more == window_size - lookahead - strstart
         * => more >= window_size - (MIN_LOOKAHEAD-1 + WSIZE + MAX_DIST-1)
         * => more >= window_size - 2*WSIZE + 2
         * In the BIG_MEM or MMAP case (not yet supported),
         *   window_size == input_size + MIN_LOOKAHEAD  &&
         *   strstart + s->lookahead <= input_size => more >= MIN_LOOKAHEAD.
         * Otherwise, window_size == 2*WSIZE so more >= 2.
         * If there was sliding, more >= WSIZE. So in all cases, more >= 2.
         */
        Assert(more >= 2, "more < 2");

        n = read_buf(s->strm, s->window + s->strstart + s->lookahead, more);
        s->lookahead += n;

        /* Initialize the hash value now that we have some input: */
        if (s->lookahead + s->insert >= STD_MIN_MATCH) {
            unsigned int str = s->strstart - s->insert;
            if (UNLIKELY(s->max_chain_length > 1024)) {
                s->ins_h = s->update_hash(s->window[str], s->window[str+1]);
            } else if (str >= 1) {
                s->quick_insert_string(s, str + 2 - STD_MIN_MATCH);
            }
            unsigned int count = s->insert;
            if (UNLIKELY(s->lookahead == 1)) {
                count -= 1;
            }
            if (count > 0) {
                s->insert_string(s, str, count);
                s->insert -= count;
            }
        }
        /* If the whole input has less than STD_MIN_MATCH bytes, ins_h is garbage,
         * but this is not important since only literal bytes will be emitted.
         */
    } while (s->lookahead < MIN_LOOKAHEAD && s->strm->avail_in != 0);

    /* If the WIN_INIT bytes after the end of the current data have never been
     * written, then zero those bytes in order to avoid memory check reports of
     * the use of uninitialized (or uninitialised as Julian writes) bytes by
     * the longest match routines.  Update the high water mark for the next
     * time through here.  WIN_INIT is set to STD_MAX_MATCH since the longest match
     * routines allow scanning to strstart + STD_MAX_MATCH, ignoring lookahead.
     */
    if (s->high_water < s->window_size) {
        unsigned int curr = s->strstart + s->lookahead;
        unsigned int init;

        if (s->high_water < curr) {
            /* Previous high water mark below current data -- zero WIN_INIT
             * bytes or up to end of window, whichever is less.
             */
            init = s->window_size - curr;
            if (init > WIN_INIT)
                init = WIN_INIT;
            memset(s->window + curr, 0, init);
            s->high_water = curr + init;
        } else if (s->high_water < curr + WIN_INIT) {
            /* High water mark at or above current data, but below current data
             * plus WIN_INIT -- zero out to current data plus WIN_INIT, or up
             * to end of window, whichever is less.
             */
            init = curr + WIN_INIT - s->high_water;
            if (init > s->window_size - s->high_water)
                init = s->window_size - s->high_water;
            memset(s->window + s->high_water, 0, init);
            s->high_water += init;
        }
    }

    Assert((unsigned long)s->strstart <= s->window_size - MIN_LOOKAHEAD,
           "not enough room for search");
}

#ifndef ZLIB_COMPAT
/* =========================================================================
 * Checks whether buffer size is sufficient and whether this parameter is a duplicate.
 */
static int32_t deflateSetParamPre(zng_deflate_param_value **out, size_t min_size, zng_deflate_param_value *param) {
    int32_t buf_error = param->size < min_size;

    if (*out != NULL) {
        (*out)->status = Z_BUF_ERROR;
        buf_error = 1;
    }
    *out = param;
    return buf_error;
}

/* ========================================================================= */
int32_t Z_EXPORT zng_deflateSetParams(zng_stream *strm, zng_deflate_param_value *params, size_t count) {
    size_t i;
    deflate_state *s;
    zng_deflate_param_value *new_level = NULL;
    zng_deflate_param_value *new_strategy = NULL;
    zng_deflate_param_value *new_reproducible = NULL;
    int param_buf_error;
    int version_error = 0;
    int buf_error = 0;
    int stream_error = 0;

    /* Initialize the statuses. */
    for (i = 0; i < count; i++)
        params[i].status = Z_OK;

    /* Check whether the stream state is consistent. */
    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    s = strm->state;

    /* Check buffer sizes and detect duplicates. */
    for (i = 0; i < count; i++) {
        switch (params[i].param) {
            case Z_DEFLATE_LEVEL:
                param_buf_error = deflateSetParamPre(&new_level, sizeof(int), &params[i]);
                break;
            case Z_DEFLATE_STRATEGY:
                param_buf_error = deflateSetParamPre(&new_strategy, sizeof(int), &params[i]);
                break;
            case Z_DEFLATE_REPRODUCIBLE:
                param_buf_error = deflateSetParamPre(&new_reproducible, sizeof(int), &params[i]);
                break;
            default:
                params[i].status = Z_VERSION_ERROR;
                version_error = 1;
                param_buf_error = 0;
                break;
        }
        if (param_buf_error) {
            params[i].status = Z_BUF_ERROR;
            buf_error = 1;
        }
    }
    /* Exit early if small buffers or duplicates are detected. */
    if (buf_error)
        return Z_BUF_ERROR;

    /* Apply changes, remember if there were errors. */
    if (new_level != NULL || new_strategy != NULL) {
        int ret = PREFIX(deflateParams)(strm, new_level == NULL ? s->level : *(int *)new_level->buf,
                                        new_strategy == NULL ? s->strategy : *(int *)new_strategy->buf);
        if (ret != Z_OK) {
            if (new_level != NULL)
                new_level->status = Z_STREAM_ERROR;
            if (new_strategy != NULL)
                new_strategy->status = Z_STREAM_ERROR;
            stream_error = 1;
        }
    }
    if (new_reproducible != NULL) {
        int val = *(int *)new_reproducible->buf;
        if (DEFLATE_CAN_SET_REPRODUCIBLE(strm, val)) {
            s->reproducible = val;
        } else {
            new_reproducible->status = Z_STREAM_ERROR;
            stream_error = 1;
        }
    }

    /* Report version errors only if there are no real errors. */
    return stream_error ? Z_STREAM_ERROR : (version_error ? Z_VERSION_ERROR : Z_OK);
}

/* ========================================================================= */
int32_t Z_EXPORT zng_deflateGetParams(zng_stream *strm, zng_deflate_param_value *params, size_t count) {
    deflate_state *s;
    size_t i;
    int32_t buf_error = 0;
    int32_t version_error = 0;

    /* Initialize the statuses. */
    for (i = 0; i < count; i++)
        params[i].status = Z_OK;

    /* Check whether the stream state is consistent. */
    if (deflateStateCheck(strm))
        return Z_STREAM_ERROR;
    s = strm->state;

    for (i = 0; i < count; i++) {
        switch (params[i].param) {
            case Z_DEFLATE_LEVEL:
                if (params[i].size < sizeof(int))
                    params[i].status = Z_BUF_ERROR;
                else
                    *(int *)params[i].buf = s->level;
                break;
            case Z_DEFLATE_STRATEGY:
                if (params[i].size < sizeof(int))
                    params[i].status = Z_BUF_ERROR;
                else
                    *(int *)params[i].buf = s->strategy;
                break;
            case Z_DEFLATE_REPRODUCIBLE:
                if (params[i].size < sizeof(int))
                    params[i].status = Z_BUF_ERROR;
                else
                    *(int *)params[i].buf = s->reproducible;
                break;
            default:
                params[i].status = Z_VERSION_ERROR;
                version_error = 1;
                break;
        }
        if (params[i].status == Z_BUF_ERROR)
            buf_error = 1;
    }
    return buf_error ? Z_BUF_ERROR : (version_error ? Z_VERSION_ERROR : Z_OK);
}
#endif
