/* inflate.c -- zlib decompression
 * Copyright (C) 1995-2022 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zutil.h"
#include "inftrees.h"
#include "inflate.h"
#include "inflate_p.h"
#include "inffixed_tbl.h"
#include "functable.h"

/* Avoid conflicts with zlib.h macros */
#ifdef ZLIB_COMPAT
# undef inflateInit
# undef inflateInit2
#endif

/* function prototypes */
static int inflateStateCheck(PREFIX3(stream) *strm);
static void updatewindow(PREFIX3(stream) *strm, const uint8_t *end, uint32_t len, int32_t cksum);
static uint32_t syncsearch(uint32_t *have, const unsigned char *buf, uint32_t len);

static inline void inf_chksum_cpy(PREFIX3(stream) *strm, uint8_t *dst,
                           const uint8_t *src, uint32_t copy) {
    if (!copy) return;
    struct inflate_state *state = (struct inflate_state*)strm->state;
#ifdef GUNZIP
    if (state->flags) {
        FUNCTABLE_CALL(crc32_fold_copy)(&state->crc_fold, dst, src, copy);
    } else
#endif
    {
        strm->adler = state->check = FUNCTABLE_CALL(adler32_fold_copy)(state->check, dst, src, copy);
    }
}

static inline void inf_chksum(PREFIX3(stream) *strm, const uint8_t *src, uint32_t len) {
    struct inflate_state *state = (struct inflate_state*)strm->state;
#ifdef GUNZIP
    if (state->flags) {
        FUNCTABLE_CALL(crc32_fold)(&state->crc_fold, src, len, 0);
    } else
#endif
    {
        strm->adler = state->check = FUNCTABLE_CALL(adler32)(state->check, src, len);
    }
}

static int inflateStateCheck(PREFIX3(stream) *strm) {
    struct inflate_state *state;
    if (strm == NULL || strm->zalloc == NULL || strm->zfree == NULL)
        return 1;
    state = (struct inflate_state *)strm->state;
    if (state == NULL || state->alloc_bufs == NULL || state->strm != strm || state->mode < HEAD || state->mode > SYNC)
        return 1;
    return 0;
}

int32_t Z_EXPORT PREFIX(inflateResetKeep)(PREFIX3(stream) *strm) {
    struct inflate_state *state;

    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;
    strm->total_in = strm->total_out = state->total = 0;
    strm->msg = NULL;
    if (state->wrap)        /* to support ill-conceived Java test suite */
        strm->adler = state->wrap & 1;
    state->mode = HEAD;
    state->check = ADLER32_INITIAL_VALUE;
    state->last = 0;
    state->havedict = 0;
    state->flags = -1;
    state->head = NULL;
    state->hold = 0;
    state->bits = 0;
    state->lencode = state->distcode = state->next = state->codes;
    state->back = -1;
#ifdef INFLATE_STRICT
    state->dmax = 32768U;
#endif
#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
    state->sane = 1;
#endif
    INFLATE_RESET_KEEP_HOOK(strm);  /* hook for IBM Z DFLTCC */
    Tracev((stderr, "inflate: reset\n"));
    return Z_OK;
}

int32_t Z_EXPORT PREFIX(inflateReset)(PREFIX3(stream) *strm) {
    struct inflate_state *state;

    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;
    state->wsize = 0;
    state->whave = 0;
    state->wnext = 0;
    return PREFIX(inflateResetKeep)(strm);
}

int32_t Z_EXPORT PREFIX(inflateReset2)(PREFIX3(stream) *strm, int32_t windowBits) {
    int wrap;
    struct inflate_state *state;

    /* get the state */
    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;

    /* extract wrap request from windowBits parameter */
    if (windowBits < 0) {
        wrap = 0;
        if (windowBits < -MAX_WBITS)
            return Z_STREAM_ERROR;
        windowBits = -windowBits;
    } else {
        wrap = (windowBits >> 4) + 5;
#ifdef GUNZIP
        if (windowBits < 48)
            windowBits &= MAX_WBITS;
#endif
    }

    /* set number of window bits */
    if (windowBits && (windowBits < MIN_WBITS || windowBits > MAX_WBITS))
        return Z_STREAM_ERROR;

    /* update state and reset the rest of it */
    state->wrap = wrap;
    state->wbits = (unsigned)windowBits;
    return PREFIX(inflateReset)(strm);
}

#ifdef INF_ALLOC_DEBUG
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
 * Allocate a big buffer and divide it up into the various buffers inflate needs.
 * Handles alignment of allocated buffer and alignment of individual buffers.
 */
Z_INTERNAL inflate_allocs* alloc_inflate(PREFIX3(stream) *strm) {
    int curr_size = 0;

    /* Define sizes */
    int window_size = INFLATE_ADJUST_WINDOW_SIZE((1 << MAX_WBITS) + 64); /* 64B padding for chunksize */
    int state_size = sizeof(inflate_state);
    int alloc_size = sizeof(inflate_allocs);

    /* Calculate relative buffer positions and paddings */
    LOGSZP("window", window_size, PAD_WINDOW(curr_size), PADSZ(curr_size,WINDOW_PAD_SIZE));
    int window_pos = PAD_WINDOW(curr_size);
    curr_size = window_pos + window_size;

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
    inflate_allocs *alloc_bufs  = (struct inflate_allocs_s *)(buff + alloc_pos);
    alloc_bufs->buf_start = original_buf;
    alloc_bufs->zfree = strm->zfree;

    alloc_bufs->window =  (unsigned char *)HINT_ALIGNED_WINDOW((buff + window_pos));
    alloc_bufs->state = (inflate_state *)HINT_ALIGNED_64((buff + state_pos));

#ifdef Z_MEMORY_SANITIZER
    /* This is _not_ to subvert the memory sanitizer but to instead unposion some
       data we willingly and purposefully load uninitialized into vector registers
       in order to safely read the last < chunksize bytes of the window. */
    __msan_unpoison(alloc_bufs->window + window_size, 64);
#endif

    return alloc_bufs;
}

/* ===========================================================================
 * Free all allocated inflate buffers
 */
Z_INTERNAL void free_inflate(PREFIX3(stream) *strm) {
    struct inflate_state *state = (struct inflate_state *)strm->state;

    if (state->alloc_bufs != NULL) {
        inflate_allocs *alloc_bufs = state->alloc_bufs;
        alloc_bufs->zfree(strm->opaque, alloc_bufs->buf_start);
        strm->state = NULL;
    }
}

/* ===========================================================================
 * Initialize inflate state and buffers.
 * This function is hidden in ZLIB_COMPAT builds.
 */
int32_t ZNG_CONDEXPORT PREFIX(inflateInit2)(PREFIX3(stream) *strm, int32_t windowBits) {
    struct inflate_state *state;
    int32_t ret;

    /* Initialize functable */
    FUNCTABLE_INIT;

    if (strm == NULL)
        return Z_STREAM_ERROR;
    strm->msg = NULL;                   /* in case we return an error */
    if (strm->zalloc == NULL) {
        strm->zalloc = PREFIX(zcalloc);
        strm->opaque = NULL;
    }
    if (strm->zfree == NULL)
        strm->zfree = PREFIX(zcfree);

    inflate_allocs *alloc_bufs = alloc_inflate(strm);
    if (alloc_bufs == NULL)
        return Z_MEM_ERROR;

    state = alloc_bufs->state;
    state->window = alloc_bufs->window;
    state->alloc_bufs = alloc_bufs;
    state->wbufsize = INFLATE_ADJUST_WINDOW_SIZE((1 << MAX_WBITS) + 64);
    Tracev((stderr, "inflate: allocated\n"));

    strm->state = (struct internal_state *)state;
    state->strm = strm;
    state->mode = HEAD;     /* to pass state test in inflateReset2() */
    ret = PREFIX(inflateReset2)(strm, windowBits);
    if (ret != Z_OK) {
        free_inflate(strm);
    }
    return ret;
}

#ifndef ZLIB_COMPAT
int32_t Z_EXPORT PREFIX(inflateInit)(PREFIX3(stream) *strm) {
    return PREFIX(inflateInit2)(strm, DEF_WBITS);
}
#endif

/* Function used by zlib.h and zlib-ng version 2.0 macros */
int32_t Z_EXPORT PREFIX(inflateInit_)(PREFIX3(stream) *strm, const char *version, int32_t stream_size) {
    if (CHECK_VER_STSIZE(version, stream_size))
        return Z_VERSION_ERROR;
    return PREFIX(inflateInit2)(strm, DEF_WBITS);
}

/* Function used by zlib.h and zlib-ng version 2.0 macros */
int32_t Z_EXPORT PREFIX(inflateInit2_)(PREFIX3(stream) *strm, int32_t windowBits, const char *version, int32_t stream_size) {
    if (CHECK_VER_STSIZE(version, stream_size))
        return Z_VERSION_ERROR;
    return PREFIX(inflateInit2)(strm, windowBits);
}

int32_t Z_EXPORT PREFIX(inflatePrime)(PREFIX3(stream) *strm, int32_t bits, int32_t value) {
    struct inflate_state *state;

    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    if (bits == 0)
        return Z_OK;
    INFLATE_PRIME_HOOK(strm, bits, value);  /* hook for IBM Z DFLTCC */
    state = (struct inflate_state *)strm->state;
    if (bits < 0) {
        state->hold = 0;
        state->bits = 0;
        return Z_OK;
    }
    if (bits > 16 || state->bits + (unsigned int)bits > 32)
        return Z_STREAM_ERROR;
    value &= (1L << bits) - 1;
    state->hold += (uint64_t)value << state->bits;
    state->bits += (unsigned int)bits;
    return Z_OK;
}

/*
   Return state with length and distance decoding tables and index sizes set to
   fixed code decoding.  This returns fixed tables from inffixed_tbl.h.
 */

void Z_INTERNAL PREFIX(fixedtables)(struct inflate_state *state) {
    state->lencode = lenfix;
    state->lenbits = 9;
    state->distcode = distfix;
    state->distbits = 5;
}

/*
   Update the window with the last wsize (normally 32K) bytes written before
   returning.  If window does not exist yet, create it.  This is only called
   when a window is already in use, or when output has been written during this
   inflate call, but the end of the deflate stream has not been reached yet.
   It is also called to create a window for dictionary data when a dictionary
   is loaded.

   Providing output buffers larger than 32K to inflate() should provide a speed
   advantage, since only the last 32K of output is copied to the sliding window
   upon return from inflate(), and since all distances after the first 32K of
   output will fall in the output data, making match copies simpler and faster.
   The advantage may be dependent on the size of the processor's data caches.
 */
static void updatewindow(PREFIX3(stream) *strm, const uint8_t *end, uint32_t len, int32_t cksum) {
    struct inflate_state *state;
    uint32_t dist;

    state = (struct inflate_state *)strm->state;

    /* if window not in use yet, initialize */
    if (state->wsize == 0)
        state->wsize = 1U << state->wbits;

    /* len state->wsize or less output bytes into the circular window */
    if (len >= state->wsize) {
        /* Only do this if the caller specifies to checksum bytes AND the platform requires
         * it (s/390 being the primary exception to this) */
        if (INFLATE_NEED_CHECKSUM(strm) && cksum) {
            /* We have to split the checksum over non-copied and copied bytes */
            if (len > state->wsize)
                inf_chksum(strm, end - len, len - state->wsize);
            inf_chksum_cpy(strm, state->window, end - state->wsize, state->wsize);
        } else {
            memcpy(state->window, end - state->wsize, state->wsize);
        }

        state->wnext = 0;
        state->whave = state->wsize;
    } else {
        dist = state->wsize - state->wnext;
        /* Only do this if the caller specifies to checksum bytes AND the platform requires
         * We need to maintain the correct order here for the checksum */
        dist = MIN(dist, len);
        if (INFLATE_NEED_CHECKSUM(strm) && cksum) {
            inf_chksum_cpy(strm, state->window + state->wnext, end - len, dist);
        } else {
            memcpy(state->window + state->wnext, end - len, dist);
        }
        len -= dist;
        if (len) {
            if (INFLATE_NEED_CHECKSUM(strm) && cksum) {
                inf_chksum_cpy(strm, state->window, end - len, len);
            } else {
                memcpy(state->window, end - len, len);
            }

            state->wnext = len;
            state->whave = state->wsize;
        } else {
            state->wnext += dist;
            if (state->wnext == state->wsize)
                state->wnext = 0;
            if (state->whave < state->wsize)
                state->whave += dist;
        }
    }
}

/*
   Private macros for inflate()
   Look in inflate_p.h for macros shared with inflateBack()
*/

/* Get a byte of input into the bit accumulator, or return from inflate() if there is no input available. */
#define PULLBYTE() \
    do { \
        if (have == 0) goto inf_leave; \
        have--; \
        hold += ((uint64_t)(*next++) << bits); \
        bits += 8; \
    } while (0)

/*
   inflate() uses a state machine to process as much input data and generate as
   much output data as possible before returning.  The state machine is
   structured roughly as follows:

    for (;;) switch (state) {
    ...
    case STATEn:
        if (not enough input data or output space to make progress)
            return;
        ... make progress ...
        state = STATEm;
        break;
    ...
    }

   so when inflate() is called again, the same case is attempted again, and
   if the appropriate resources are provided, the machine proceeds to the
   next state.  The NEEDBITS() macro is usually the way the state evaluates
   whether it can proceed or should return.  NEEDBITS() does the return if
   the requested bits are not available.  The typical use of the BITS macros
   is:

        NEEDBITS(n);
        ... do something with BITS(n) ...
        DROPBITS(n);

   where NEEDBITS(n) either returns from inflate() if there isn't enough
   input left to load n bits into the accumulator, or it continues.  BITS(n)
   gives the low n bits in the accumulator.  When done, DROPBITS(n) drops
   the low n bits off the accumulator.  INITBITS() clears the accumulator
   and sets the number of available bits to zero.  BYTEBITS() discards just
   enough bits to put the accumulator on a byte boundary.  After BYTEBITS()
   and a NEEDBITS(8), then BITS(8) would return the next byte in the stream.

   NEEDBITS(n) uses PULLBYTE() to get an available byte of input, or to return
   if there is no input available.  The decoding of variable length codes uses
   PULLBYTE() directly in order to pull just enough bytes to decode the next
   code, and no more.

   Some states loop until they get enough input, making sure that enough
   state information is maintained to continue the loop where it left off
   if NEEDBITS() returns in the loop.  For example, want, need, and keep
   would all have to actually be part of the saved state in case NEEDBITS()
   returns:

    case STATEw:
        while (want < need) {
            NEEDBITS(n);
            keep[want++] = BITS(n);
            DROPBITS(n);
        }
        state = STATEx;
    case STATEx:

   As shown above, if the next state is also the next case, then the break
   is omitted.

   A state may also return if there is not enough output space available to
   complete that state.  Those states are copying stored data, writing a
   literal byte, and copying a matching string.

   When returning, a "goto inf_leave" is used to update the total counters,
   update the check value, and determine whether any progress has been made
   during that inflate() call in order to return the proper return code.
   Progress is defined as a change in either strm->avail_in or strm->avail_out.
   When there is a window, goto inf_leave will update the window with the last
   output written.  If a goto inf_leave occurs in the middle of decompression
   and there is no window currently, goto inf_leave will create one and copy
   output to the window for the next call of inflate().

   In this implementation, the flush parameter of inflate() only affects the
   return code (per zlib.h).  inflate() always writes as much as possible to
   strm->next_out, given the space available and the provided input--the effect
   documented in zlib.h of Z_SYNC_FLUSH.  Furthermore, inflate() always defers
   the allocation of and copying into a sliding window until necessary, which
   provides the effect documented in zlib.h for Z_FINISH when the entire input
   stream available.  So the only thing the flush parameter actually does is:
   when flush is set to Z_FINISH, inflate() cannot return Z_OK.  Instead it
   will return Z_BUF_ERROR if it has not reached the end of the stream.
 */

int32_t Z_EXPORT PREFIX(inflate)(PREFIX3(stream) *strm, int32_t flush) {
    struct inflate_state *state;
    const unsigned char *next;  /* next input */
    unsigned char *put;         /* next output */
    unsigned char *from;        /* where to copy match bytes from */
    unsigned have, left;        /* available input and output */
    uint64_t hold;              /* bit buffer */
    unsigned bits;              /* bits in bit buffer */
    uint32_t in, out;           /* save starting available input and output */
    unsigned copy;              /* number of stored or match bytes to copy */
    code here;                  /* current decoding table entry */
    code last;                  /* parent table entry */
    unsigned len;               /* length to copy for repeats, bits to drop */
    int32_t ret;                /* return code */
#ifdef GUNZIP
    unsigned char hbuf[4];      /* buffer for gzip header crc calculation */
#endif
    static const uint16_t order[19] = /* permutation of code lengths */
        {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

    if (inflateStateCheck(strm) || strm->next_out == NULL ||
        (strm->next_in == NULL && strm->avail_in != 0))
        return Z_STREAM_ERROR;

    state = (struct inflate_state *)strm->state;
    if (state->mode == TYPE)      /* skip check */
        state->mode = TYPEDO;
    LOAD();
    in = have;
    out = left;
    ret = Z_OK;
    for (;;)
        switch (state->mode) {
        case HEAD:
            if (state->wrap == 0) {
                state->mode = TYPEDO;
                break;
            }
            NEEDBITS(16);
#ifdef GUNZIP
            if ((state->wrap & 2) && hold == 0x8b1f) {  /* gzip header */
                if (state->wbits == 0)
                    state->wbits = MAX_WBITS;
                state->check = CRC32_INITIAL_VALUE;
                CRC2(state->check, hold);
                INITBITS();
                state->mode = FLAGS;
                break;
            }
            if (state->head != NULL)
                state->head->done = -1;
            if (!(state->wrap & 1) ||   /* check if zlib header allowed */
#else
            if (
#endif
                ((BITS(8) << 8) + (hold >> 8)) % 31) {
                SET_BAD("incorrect header check");
                break;
            }
            if (BITS(4) != Z_DEFLATED) {
                SET_BAD("unknown compression method");
                break;
            }
            DROPBITS(4);
            len = BITS(4) + 8;
            if (state->wbits == 0)
                state->wbits = len;
            if (len > MAX_WBITS || len > state->wbits) {
                SET_BAD("invalid window size");
                break;
            }
#ifdef INFLATE_STRICT
            state->dmax = 1U << len;
#endif
            state->flags = 0;               /* indicate zlib header */
            Tracev((stderr, "inflate:   zlib header ok\n"));
            strm->adler = state->check = ADLER32_INITIAL_VALUE;
            state->mode = hold & 0x200 ? DICTID : TYPE;
            INITBITS();
            break;
#ifdef GUNZIP

        case FLAGS:
            NEEDBITS(16);
            state->flags = (int)(hold);
            if ((state->flags & 0xff) != Z_DEFLATED) {
                SET_BAD("unknown compression method");
                break;
            }
            if (state->flags & 0xe000) {
                SET_BAD("unknown header flags set");
                break;
            }
            if (state->head != NULL)
                state->head->text = (int)((hold >> 8) & 1);
            if ((state->flags & 0x0200) && (state->wrap & 4))
                CRC2(state->check, hold);
            INITBITS();
            state->mode = TIME;
            Z_FALLTHROUGH;

        case TIME:
            NEEDBITS(32);
            if (state->head != NULL)
                state->head->time = (unsigned)(hold);
            if ((state->flags & 0x0200) && (state->wrap & 4))
                CRC4(state->check, hold);
            INITBITS();
            state->mode = OS;
            Z_FALLTHROUGH;

        case OS:
            NEEDBITS(16);
            if (state->head != NULL) {
                state->head->xflags = (int)(hold & 0xff);
                state->head->os = (int)(hold >> 8);
            }
            if ((state->flags & 0x0200) && (state->wrap & 4))
                CRC2(state->check, hold);
            INITBITS();
            state->mode = EXLEN;
            Z_FALLTHROUGH;

        case EXLEN:
            if (state->flags & 0x0400) {
                NEEDBITS(16);
                state->length = (uint16_t)hold;
                if (state->head != NULL)
                    state->head->extra_len = (uint16_t)hold;
                if ((state->flags & 0x0200) && (state->wrap & 4))
                    CRC2(state->check, hold);
                INITBITS();
            } else if (state->head != NULL) {
                state->head->extra = NULL;
            }
            state->mode = EXTRA;
            Z_FALLTHROUGH;

        case EXTRA:
            if (state->flags & 0x0400) {
                copy = state->length;
                if (copy > have)
                    copy = have;
                if (copy) {
                    if (state->head != NULL && state->head->extra != NULL) {
                        len = state->head->extra_len - state->length;
                        if (len < state->head->extra_max) {
                            memcpy(state->head->extra + len, next,
                                    len + copy > state->head->extra_max ?
                                    state->head->extra_max - len : copy);
                        }
                    }
                    if ((state->flags & 0x0200) && (state->wrap & 4)) {
                        state->check = PREFIX(crc32)(state->check, next, copy);
                    }
                    have -= copy;
                    next += copy;
                    state->length -= copy;
                }
                if (state->length)
                    goto inf_leave;
            }
            state->length = 0;
            state->mode = NAME;
            Z_FALLTHROUGH;

        case NAME:
            if (state->flags & 0x0800) {
                if (have == 0) goto inf_leave;
                copy = 0;
                do {
                    len = (unsigned)(next[copy++]);
                    if (state->head != NULL && state->head->name != NULL && state->length < state->head->name_max)
                        state->head->name[state->length++] = (unsigned char)len;
                } while (len && copy < have);
                if ((state->flags & 0x0200) && (state->wrap & 4))
                    state->check = PREFIX(crc32)(state->check, next, copy);
                have -= copy;
                next += copy;
                if (len)
                    goto inf_leave;
            } else if (state->head != NULL) {
                state->head->name = NULL;
            }
            state->length = 0;
            state->mode = COMMENT;
            Z_FALLTHROUGH;

        case COMMENT:
            if (state->flags & 0x1000) {
                if (have == 0) goto inf_leave;
                copy = 0;
                do {
                    len = (unsigned)(next[copy++]);
                    if (state->head != NULL && state->head->comment != NULL
                        && state->length < state->head->comm_max)
                        state->head->comment[state->length++] = (unsigned char)len;
                } while (len && copy < have);
                if ((state->flags & 0x0200) && (state->wrap & 4))
                    state->check = PREFIX(crc32)(state->check, next, copy);
                have -= copy;
                next += copy;
                if (len)
                    goto inf_leave;
            } else if (state->head != NULL) {
                state->head->comment = NULL;
            }
            state->mode = HCRC;
            Z_FALLTHROUGH;

        case HCRC:
            if (state->flags & 0x0200) {
                NEEDBITS(16);
                if ((state->wrap & 4) && hold != (state->check & 0xffff)) {
                    SET_BAD("header crc mismatch");
                    break;
                }
                INITBITS();
            }
            if (state->head != NULL) {
                state->head->hcrc = (int)((state->flags >> 9) & 1);
                state->head->done = 1;
            }
            /* compute crc32 checksum if not in raw mode */
            if ((state->wrap & 4) && state->flags)
                strm->adler = state->check = FUNCTABLE_CALL(crc32_fold_reset)(&state->crc_fold);
            state->mode = TYPE;
            break;
#endif
        case DICTID:
            NEEDBITS(32);
            strm->adler = state->check = ZSWAP32((unsigned)hold);
            INITBITS();
            state->mode = DICT;
            Z_FALLTHROUGH;

        case DICT:
            if (state->havedict == 0) {
                RESTORE();
                return Z_NEED_DICT;
            }
            strm->adler = state->check = ADLER32_INITIAL_VALUE;
            state->mode = TYPE;
            Z_FALLTHROUGH;

        case TYPE:
            if (flush == Z_BLOCK || flush == Z_TREES)
                goto inf_leave;
            Z_FALLTHROUGH;

        case TYPEDO:
            /* determine and dispatch block type */
            INFLATE_TYPEDO_HOOK(strm, flush);  /* hook for IBM Z DFLTCC */
            if (state->last) {
                BYTEBITS();
                state->mode = CHECK;
                break;
            }
            NEEDBITS(3);
            state->last = BITS(1);
            DROPBITS(1);
            switch (BITS(2)) {
            case 0:                             /* stored block */
                Tracev((stderr, "inflate:     stored block%s\n", state->last ? " (last)" : ""));
                state->mode = STORED;
                break;
            case 1:                             /* fixed block */
                PREFIX(fixedtables)(state);
                Tracev((stderr, "inflate:     fixed codes block%s\n", state->last ? " (last)" : ""));
                state->mode = LEN_;             /* decode codes */
                if (flush == Z_TREES) {
                    DROPBITS(2);
                    goto inf_leave;
                }
                break;
            case 2:                             /* dynamic block */
                Tracev((stderr, "inflate:     dynamic codes block%s\n", state->last ? " (last)" : ""));
                state->mode = TABLE;
                break;
            case 3:
                SET_BAD("invalid block type");
            }
            DROPBITS(2);
            break;

        case STORED:
            /* get and verify stored block length */
            BYTEBITS();                         /* go to byte boundary */
            NEEDBITS(32);
            if ((hold & 0xffff) != ((hold >> 16) ^ 0xffff)) {
                SET_BAD("invalid stored block lengths");
                break;
            }
            state->length = (uint16_t)hold;
            Tracev((stderr, "inflate:       stored length %u\n", state->length));
            INITBITS();
            state->mode = COPY_;
            if (flush == Z_TREES)
                goto inf_leave;
            Z_FALLTHROUGH;

        case COPY_:
            state->mode = COPY;
            Z_FALLTHROUGH;

        case COPY:
            /* copy stored block from input to output */
            copy = state->length;
            if (copy) {
                copy = MIN(copy, have);
                copy = MIN(copy, left);
                if (copy == 0)
                    goto inf_leave;
                memcpy(put, next, copy);
                have -= copy;
                next += copy;
                left -= copy;
                put += copy;
                state->length -= copy;
                break;
            }
            Tracev((stderr, "inflate:       stored end\n"));
            state->mode = TYPE;
            break;

        case TABLE:
            /* get dynamic table entries descriptor */
            NEEDBITS(14);
            state->nlen = BITS(5) + 257;
            DROPBITS(5);
            state->ndist = BITS(5) + 1;
            DROPBITS(5);
            state->ncode = BITS(4) + 4;
            DROPBITS(4);
#ifndef PKZIP_BUG_WORKAROUND
            if (state->nlen > 286 || state->ndist > 30) {
                SET_BAD("too many length or distance symbols");
                break;
            }
#endif
            Tracev((stderr, "inflate:       table sizes ok\n"));
            state->have = 0;
            state->mode = LENLENS;
            Z_FALLTHROUGH;

        case LENLENS:
            /* get code length code lengths (not a typo) */
            while (state->have < state->ncode) {
                NEEDBITS(3);
                state->lens[order[state->have++]] = (uint16_t)BITS(3);
                DROPBITS(3);
            }
            while (state->have < 19)
                state->lens[order[state->have++]] = 0;
            state->next = state->codes;
            state->lencode = (const code *)(state->next);
            state->lenbits = 7;
            ret = zng_inflate_table(CODES, state->lens, 19, &(state->next), &(state->lenbits), state->work);
            if (ret) {
                SET_BAD("invalid code lengths set");
                break;
            }
            Tracev((stderr, "inflate:       code lengths ok\n"));
            state->have = 0;
            state->mode = CODELENS;
            Z_FALLTHROUGH;

        case CODELENS:
            /* get length and distance code code lengths */
            while (state->have < state->nlen + state->ndist) {
                for (;;) {
                    here = state->lencode[BITS(state->lenbits)];
                    if (here.bits <= bits) break;
                    PULLBYTE();
                }
                if (here.val < 16) {
                    DROPBITS(here.bits);
                    state->lens[state->have++] = here.val;
                } else {
                    if (here.val == 16) {
                        NEEDBITS(here.bits + 2);
                        DROPBITS(here.bits);
                        if (state->have == 0) {
                            SET_BAD("invalid bit length repeat");
                            break;
                        }
                        len = state->lens[state->have - 1];
                        copy = 3 + BITS(2);
                        DROPBITS(2);
                    } else if (here.val == 17) {
                        NEEDBITS(here.bits + 3);
                        DROPBITS(here.bits);
                        len = 0;
                        copy = 3 + BITS(3);
                        DROPBITS(3);
                    } else {
                        NEEDBITS(here.bits + 7);
                        DROPBITS(here.bits);
                        len = 0;
                        copy = 11 + BITS(7);
                        DROPBITS(7);
                    }
                    if (state->have + copy > state->nlen + state->ndist) {
                        SET_BAD("invalid bit length repeat");
                        break;
                    }
                    while (copy) {
                        --copy;
                        state->lens[state->have++] = (uint16_t)len;
                    }
                }
            }

            /* handle error breaks in while */
            if (state->mode == BAD)
                break;

            /* check for end-of-block code (better have one) */
            if (state->lens[256] == 0) {
                SET_BAD("invalid code -- missing end-of-block");
                break;
            }

            /* build code tables -- note: do not change the lenbits or distbits
               values here (10 and 9) without reading the comments in inftrees.h
               concerning the ENOUGH constants, which depend on those values */
            state->next = state->codes;
            state->lencode = (const code *)(state->next);
            state->lenbits = 10;
            ret = zng_inflate_table(LENS, state->lens, state->nlen, &(state->next), &(state->lenbits), state->work);
            if (ret) {
                SET_BAD("invalid literal/lengths set");
                break;
            }
            state->distcode = (const code *)(state->next);
            state->distbits = 9;
            ret = zng_inflate_table(DISTS, state->lens + state->nlen, state->ndist,
                            &(state->next), &(state->distbits), state->work);
            if (ret) {
                SET_BAD("invalid distances set");
                break;
            }
            Tracev((stderr, "inflate:       codes ok\n"));
            state->mode = LEN_;
            if (flush == Z_TREES)
                goto inf_leave;
            Z_FALLTHROUGH;

        case LEN_:
            state->mode = LEN;
            Z_FALLTHROUGH;

        case LEN:
            /* use inflate_fast() if we have enough input and output */
            if (have >= INFLATE_FAST_MIN_HAVE && left >= INFLATE_FAST_MIN_LEFT) {
                RESTORE();
                FUNCTABLE_CALL(inflate_fast)(strm, out);
                LOAD();
                if (state->mode == TYPE)
                    state->back = -1;
                break;
            }
            state->back = 0;

            /* get a literal, length, or end-of-block code */
            for (;;) {
                here = state->lencode[BITS(state->lenbits)];
                if (here.bits <= bits)
                    break;
                PULLBYTE();
            }
            if (here.op && (here.op & 0xf0) == 0) {
                last = here;
                for (;;) {
                    here = state->lencode[last.val + (BITS(last.bits + last.op) >> last.bits)];
                    if ((unsigned)last.bits + (unsigned)here.bits <= bits)
                        break;
                    PULLBYTE();
                }
                DROPBITS(last.bits);
                state->back += last.bits;
            }
            DROPBITS(here.bits);
            state->back += here.bits;
            state->length = here.val;

            /* process literal */
            if ((int)(here.op) == 0) {
                Tracevv((stderr, here.val >= 0x20 && here.val < 0x7f ?
                        "inflate:         literal '%c'\n" :
                        "inflate:         literal 0x%02x\n", here.val));
                state->mode = LIT;
                break;
            }

            /* process end of block */
            if (here.op & 32) {
                Tracevv((stderr, "inflate:         end of block\n"));
                state->back = -1;
                state->mode = TYPE;
                break;
            }

            /* invalid code */
            if (here.op & 64) {
                SET_BAD("invalid literal/length code");
                break;
            }

            /* length code */
            state->extra = (here.op & MAX_BITS);
            state->mode = LENEXT;
            Z_FALLTHROUGH;

        case LENEXT:
            /* get extra bits, if any */
            if (state->extra) {
                NEEDBITS(state->extra);
                state->length += BITS(state->extra);
                DROPBITS(state->extra);
                state->back += state->extra;
            }
            Tracevv((stderr, "inflate:         length %u\n", state->length));
            state->was = state->length;
            state->mode = DIST;
            Z_FALLTHROUGH;

        case DIST:
            /* get distance code */
            for (;;) {
                here = state->distcode[BITS(state->distbits)];
                if (here.bits <= bits)
                    break;
                PULLBYTE();
            }
            if ((here.op & 0xf0) == 0) {
                last = here;
                for (;;) {
                    here = state->distcode[last.val + (BITS(last.bits + last.op) >> last.bits)];
                    if ((unsigned)last.bits + (unsigned)here.bits <= bits)
                        break;
                    PULLBYTE();
                }
                DROPBITS(last.bits);
                state->back += last.bits;
            }
            DROPBITS(here.bits);
            state->back += here.bits;
            if (here.op & 64) {
                SET_BAD("invalid distance code");
                break;
            }
            state->offset = here.val;
            state->extra = (here.op & MAX_BITS);
            state->mode = DISTEXT;
            Z_FALLTHROUGH;

        case DISTEXT:
            /* get distance extra bits, if any */
            if (state->extra) {
                NEEDBITS(state->extra);
                state->offset += BITS(state->extra);
                DROPBITS(state->extra);
                state->back += state->extra;
            }
#ifdef INFLATE_STRICT
            if (state->offset > state->dmax) {
                SET_BAD("invalid distance too far back");
                break;
            }
#endif
            Tracevv((stderr, "inflate:         distance %u\n", state->offset));
            state->mode = MATCH;
            Z_FALLTHROUGH;

        case MATCH:
            /* copy match from window to output */
            if (left == 0)
                goto inf_leave;
            copy = out - left;
            if (state->offset > copy) {         /* copy from window */
                copy = state->offset - copy;
                if (copy > state->whave) {
#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
                    if (state->sane) {
                        SET_BAD("invalid distance too far back");
                        break;
                    }
                    Trace((stderr, "inflate.c too far\n"));
                    copy -= state->whave;
                    copy = MIN(copy, state->length);
                    copy = MIN(copy, left);
                    left -= copy;
                    state->length -= copy;
                    do {
                        *put++ = 0;
                    } while (--copy);
                    if (state->length == 0)
                        state->mode = LEN;
#else
                    SET_BAD("invalid distance too far back");
#endif
                    break;
                }
                if (copy > state->wnext) {
                    copy -= state->wnext;
                    from = state->window + (state->wsize - copy);
                } else {
                    from = state->window + (state->wnext - copy);
                }
                copy = MIN(copy, state->length);
                copy = MIN(copy, left);

                put = chunkcopy_safe(put, from, copy, put + left);
            } else {
                copy = MIN(state->length, left);

                put = FUNCTABLE_CALL(chunkmemset_safe)(put, put - state->offset, copy, left);
            }
            left -= copy;
            state->length -= copy;
            if (state->length == 0)
                state->mode = LEN;
            break;

        case LIT:
            if (left == 0)
                goto inf_leave;
            *put++ = (unsigned char)(state->length);
            left--;
            state->mode = LEN;
            break;

        case CHECK:
            if (state->wrap) {
                NEEDBITS(32);
                out -= left;
                strm->total_out += out;
                state->total += out;

                /* compute crc32 checksum if not in raw mode */
                if (INFLATE_NEED_CHECKSUM(strm) && state->wrap & 4) {
                    if (out) {
                        inf_chksum(strm, put - out, out);
                    }
#ifdef GUNZIP
                    if (state->flags)
                        strm->adler = state->check = FUNCTABLE_CALL(crc32_fold_final)(&state->crc_fold);
#endif
                }
                out = left;
                if ((state->wrap & 4) && (
#ifdef GUNZIP
                     state->flags ? hold :
#endif
                     ZSWAP32((unsigned)hold)) != state->check) {
                    SET_BAD("incorrect data check");
                    break;
                }
                INITBITS();
                Tracev((stderr, "inflate:   check matches trailer\n"));
            }
#ifdef GUNZIP
            state->mode = LENGTH;
            Z_FALLTHROUGH;

        case LENGTH:
            if (state->wrap && state->flags) {
                NEEDBITS(32);
                if ((state->wrap & 4) && hold != (state->total & 0xffffffff)) {
                    SET_BAD("incorrect length check");
                    break;
                }
                INITBITS();
                Tracev((stderr, "inflate:   length matches trailer\n"));
            }
#endif
            state->mode = DONE;
            Z_FALLTHROUGH;

        case DONE:
            /* inflate stream terminated properly */
            ret = Z_STREAM_END;
            goto inf_leave;

        case BAD:
            ret = Z_DATA_ERROR;
            goto inf_leave;

        case SYNC:

        default:                 /* can't happen, but makes compilers happy */
            return Z_STREAM_ERROR;
        }

    /*
       Return from inflate(), updating the total counts and the check value.
       If there was no progress during the inflate() call, return a buffer
       error.  Call updatewindow() to create and/or update the window state.
     */
  inf_leave:
    RESTORE();
    uint32_t check_bytes = out - strm->avail_out;
    if (INFLATE_NEED_UPDATEWINDOW(strm) &&
            (state->wsize || (out != strm->avail_out && state->mode < BAD &&
                 (state->mode < CHECK || flush != Z_FINISH)))) {
        /* update sliding window with respective checksum if not in "raw" mode */
        updatewindow(strm, strm->next_out, check_bytes, state->wrap & 4);
    }
    in -= strm->avail_in;
    out -= strm->avail_out;
    strm->total_in += in;
    strm->total_out += out;
    state->total += out;

    strm->data_type = (int)state->bits + (state->last ? 64 : 0) +
                      (state->mode == TYPE ? 128 : 0) + (state->mode == LEN_ || state->mode == COPY_ ? 256 : 0);
    if (((in == 0 && out == 0) || flush == Z_FINISH) && ret == Z_OK) {
        /* when no sliding window is used, hash the output bytes if no CHECK state */
        if (INFLATE_NEED_CHECKSUM(strm) && !state->wsize && flush == Z_FINISH) {
            inf_chksum(strm, put - check_bytes, check_bytes);
        }
        ret = Z_BUF_ERROR;
    }
    return ret;
}

int32_t Z_EXPORT PREFIX(inflateEnd)(PREFIX3(stream) *strm) {
    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;

    /* Free allocated buffers */
    free_inflate(strm);

    Tracev((stderr, "inflate: end\n"));
    return Z_OK;
}

int32_t Z_EXPORT PREFIX(inflateGetDictionary)(PREFIX3(stream) *strm, uint8_t *dictionary, uint32_t *dictLength) {
    struct inflate_state *state;

    /* check state */
    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;

    INFLATE_GET_DICTIONARY_HOOK(strm, dictionary, dictLength);  /* hook for IBM Z DFLTCC */

    /* copy dictionary */
    if (state->whave && dictionary != NULL) {
        memcpy(dictionary, state->window + state->wnext, state->whave - state->wnext);
        memcpy(dictionary + state->whave - state->wnext, state->window, state->wnext);
    }
    if (dictLength != NULL)
        *dictLength = state->whave;
    return Z_OK;
}

int32_t Z_EXPORT PREFIX(inflateSetDictionary)(PREFIX3(stream) *strm, const uint8_t *dictionary, uint32_t dictLength) {
    struct inflate_state *state;
    unsigned long dictid;

    /* check state */
    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;
    if (state->wrap != 0 && state->mode != DICT)
        return Z_STREAM_ERROR;

    /* check for correct dictionary identifier */
    if (state->mode == DICT) {
        dictid = FUNCTABLE_CALL(adler32)(ADLER32_INITIAL_VALUE, dictionary, dictLength);
        if (dictid != state->check)
            return Z_DATA_ERROR;
    }

    INFLATE_SET_DICTIONARY_HOOK(strm, dictionary, dictLength);  /* hook for IBM Z DFLTCC */

    /* copy dictionary to window using updatewindow(), which will amend the
       existing dictionary if appropriate */
    updatewindow(strm, dictionary + dictLength, dictLength, 0);

    state->havedict = 1;
    Tracev((stderr, "inflate:   dictionary set\n"));
    return Z_OK;
}

int32_t Z_EXPORT PREFIX(inflateGetHeader)(PREFIX3(stream) *strm, PREFIX(gz_headerp) head) {
    struct inflate_state *state;

    /* check state */
    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;
    if ((state->wrap & 2) == 0)
        return Z_STREAM_ERROR;

    /* save header structure */
    state->head = head;
    head->done = 0;
    return Z_OK;
}

/*
   Search buf[0..len-1] for the pattern: 0, 0, 0xff, 0xff.  Return when found
   or when out of input.  When called, *have is the number of pattern bytes
   found in order so far, in 0..3.  On return *have is updated to the new
   state.  If on return *have equals four, then the pattern was found and the
   return value is how many bytes were read including the last byte of the
   pattern.  If *have is less than four, then the pattern has not been found
   yet and the return value is len.  In the latter case, syncsearch() can be
   called again with more data and the *have state.  *have is initialized to
   zero for the first call.
 */
static uint32_t syncsearch(uint32_t *have, const uint8_t *buf, uint32_t len) {
    uint32_t got, next;

    got = *have;
    next = 0;
    while (next < len && got < 4) {
        if ((int)(buf[next]) == (got < 2 ? 0 : 0xff))
            got++;
        else if (buf[next])
            got = 0;
        else
            got = 4 - got;
        next++;
    }
    *have = got;
    return next;
}

int32_t Z_EXPORT PREFIX(inflateSync)(PREFIX3(stream) *strm) {
    struct inflate_state *state;
    size_t in, out;             /* temporary to save total_in and total_out */
    unsigned len;               /* number of bytes to look at or looked at */
    int flags;                  /* temporary to save header status */
    unsigned char buf[4];       /* to restore bit buffer to byte string */

    /* check parameters */
    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;
    if (strm->avail_in == 0 && state->bits < 8)
        return Z_BUF_ERROR;

    /* if first time, start search in bit buffer */
    if (state->mode != SYNC) {
        state->mode = SYNC;
        state->hold >>= state->bits & 7;
        state->bits -= state->bits & 7;
        len = 0;
        while (state->bits >= 8) {
            buf[len++] = (unsigned char)(state->hold);
            state->hold >>= 8;
            state->bits -= 8;
        }
        state->have = 0;
        syncsearch(&(state->have), buf, len);
    }

    /* search available input */
    len = syncsearch(&(state->have), strm->next_in, strm->avail_in);
    strm->avail_in -= len;
    strm->next_in += len;
    strm->total_in += len;

    /* return no joy or set up to restart inflate() on a new block */
    if (state->have != 4)
        return Z_DATA_ERROR;
    if (state->flags == -1)
        state->wrap = 0;    /* if no header yet, treat as raw */
    else
        state->wrap &= ~4;  /* no point in computing a check value now */
    flags = state->flags;
    in = strm->total_in;
    out = strm->total_out;
    PREFIX(inflateReset)(strm);
    strm->total_in = (z_uintmax_t)in; /* Can't use z_size_t here as it will overflow on 64-bit Windows */
    strm->total_out = (z_uintmax_t)out;
    state->flags = flags;
    state->mode = TYPE;
    return Z_OK;
}

/*
   Returns true if inflate is currently at the end of a block generated by
   Z_SYNC_FLUSH or Z_FULL_FLUSH. This function is used by one PPP
   implementation to provide an additional safety check. PPP uses
   Z_SYNC_FLUSH but removes the length bytes of the resulting empty stored
   block. When decompressing, PPP checks that at the end of input packet,
   inflate is waiting for these length bytes.
 */
int32_t Z_EXPORT PREFIX(inflateSyncPoint)(PREFIX3(stream) *strm) {
    struct inflate_state *state;

    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    INFLATE_SYNC_POINT_HOOK(strm);
    state = (struct inflate_state *)strm->state;
    return state->mode == STORED && state->bits == 0;
}

int32_t Z_EXPORT PREFIX(inflateCopy)(PREFIX3(stream) *dest, PREFIX3(stream) *source) {
    struct inflate_state *state;
    struct inflate_state *copy;

    /* check input */
    if (inflateStateCheck(source) || dest == NULL)
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)source->state;

    /* copy stream */
    memcpy((void *)dest, (void *)source, sizeof(PREFIX3(stream)));

    /* allocate space */
    inflate_allocs *alloc_bufs = alloc_inflate(dest);
    if (alloc_bufs == NULL)
        return Z_MEM_ERROR;
    copy = alloc_bufs->state;

    /* copy state */
    memcpy(copy, state, sizeof(struct inflate_state));
    copy->strm = dest;
    if (state->lencode >= state->codes && state->lencode <= state->codes + ENOUGH - 1) {
        copy->lencode = copy->codes + (state->lencode - state->codes);
        copy->distcode = copy->codes + (state->distcode - state->codes);
    }
    copy->next = copy->codes + (state->next - state->codes);
    copy->window = alloc_bufs->window;
    copy->alloc_bufs = alloc_bufs;

    /* window */
    memcpy(copy->window, state->window, INFLATE_ADJUST_WINDOW_SIZE((size_t)state->wsize));

    dest->state = (struct internal_state *)copy;
    return Z_OK;
}

int32_t Z_EXPORT PREFIX(inflateUndermine)(PREFIX3(stream) *strm, int32_t subvert) {
#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
    struct inflate_state *state;

    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;
    state->sane = !subvert;
    return Z_OK;
#else
    Z_UNUSED(strm);
    Z_UNUSED(subvert);
    return Z_DATA_ERROR;
#endif
}

int32_t Z_EXPORT PREFIX(inflateValidate)(PREFIX3(stream) *strm, int32_t check) {
    struct inflate_state *state;

    if (inflateStateCheck(strm))
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;
    if (check && state->wrap)
        state->wrap |= 4;
    else
        state->wrap &= ~4;
    return Z_OK;
}

long Z_EXPORT PREFIX(inflateMark)(PREFIX3(stream) *strm) {
    struct inflate_state *state;

    if (inflateStateCheck(strm))
        return -65536;
    INFLATE_MARK_HOOK(strm);  /* hook for IBM Z DFLTCC */
    state = (struct inflate_state *)strm->state;
    return (long)(((unsigned long)((long)state->back)) << 16) +
        (state->mode == COPY ? state->length :
            (state->mode == MATCH ? state->was - state->length : 0));
}

unsigned long Z_EXPORT PREFIX(inflateCodesUsed)(PREFIX3(stream) *strm) {
    struct inflate_state *state;
    if (strm == NULL || strm->state == NULL)
        return (unsigned long)-1;
    state = (struct inflate_state *)strm->state;
    return (unsigned long)(state->next - state->codes);
}
