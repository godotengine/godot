/* inffast.c -- fast decoding
 * Copyright (C) 1995-2017 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zbuild.h"
#include "zendian.h"
#include "zutil.h"
#include "inftrees.h"
#include "inflate.h"
#include "inflate_p.h"
#include "functable.h"

/*
   Decode literal, length, and distance codes and write out the resulting
   literal and match bytes until either not enough input or output is
   available, an end-of-block is encountered, or a data error is encountered.
   When large enough input and output buffers are supplied to inflate(), for
   example, a 16K input buffer and a 64K output buffer, more than 95% of the
   inflate execution time is spent in this routine.

   Entry assumptions:

        state->mode == LEN
        strm->avail_in >= INFLATE_FAST_MIN_HAVE
        strm->avail_out >= INFLATE_FAST_MIN_LEFT
        start >= strm->avail_out
        state->bits < 8

   On return, state->mode is one of:

        LEN -- ran out of enough output space or enough available input
        TYPE -- reached end of block code, inflate() to interpret next block
        BAD -- error in block data

   Notes:

    - The maximum input bits used by a length/distance pair is 15 bits for the
      length code, 5 bits for the length extra, 15 bits for the distance code,
      and 13 bits for the distance extra.  This totals 48 bits, or six bytes.
      Therefore if strm->avail_in >= 6, then there is enough input to avoid
      checking for available input while decoding.

    - On some architectures, it can be significantly faster (e.g. up to 1.2x
      faster on x86_64) to load from strm->next_in 64 bits, or 8 bytes, at a
      time, so INFLATE_FAST_MIN_HAVE == 8.

    - The maximum bytes that a single length/distance pair can output is 258
      bytes, which is the maximum length that can be coded.  inflate_fast()
      requires strm->avail_out >= 258 for each loop to avoid checking for
      output space.
 */
void Z_INTERNAL INFLATE_FAST(PREFIX3(stream) *strm, uint32_t start) {
    /* start: inflate()'s starting value for strm->avail_out */
    struct inflate_state *state;
    z_const unsigned char *in;  /* local strm->next_in */
    const unsigned char *last;  /* have enough input while in < last */
    unsigned char *out;         /* local strm->next_out */
    unsigned char *beg;         /* inflate()'s initial strm->next_out */
    unsigned char *end;         /* while out < end, enough space available */
    unsigned char *safe;        /* can use chunkcopy provided out < safe */
    unsigned char *window;      /* allocated sliding window, if wsize != 0 */
    unsigned wsize;             /* window size or zero if not using window */
    unsigned whave;             /* valid bytes in the window */
    unsigned wnext;             /* window write index */

    /* hold is a local copy of strm->hold. By default, hold satisfies the same
       invariants that strm->hold does, namely that (hold >> bits) == 0. This
       invariant is kept by loading bits into hold one byte at a time, like:

       hold |= next_byte_of_input << bits; in++; bits += 8;

       If we need to ensure that bits >= 15 then this code snippet is simply
       repeated. Over one iteration of the outermost do/while loop, this
       happens up to six times (48 bits of input), as described in the NOTES
       above.

       However, on some little endian architectures, it can be significantly
       faster to load 64 bits once instead of 8 bits six times:

       if (bits <= 16) {
         hold |= next_8_bytes_of_input << bits; in += 6; bits += 48;
       }

       Unlike the simpler one byte load, shifting the next_8_bytes_of_input
       by bits will overflow and lose those high bits, up to 2 bytes' worth.
       The conservative estimate is therefore that we have read only 6 bytes
       (48 bits). Again, as per the NOTES above, 48 bits is sufficient for the
       rest of the iteration, and we will not need to load another 8 bytes.

       Inside this function, we no longer satisfy (hold >> bits) == 0, but
       this is not problematic, even if that overflow does not land on an 8 bit
       byte boundary. Those excess bits will eventually shift down lower as the
       Huffman decoder consumes input, and when new input bits need to be loaded
       into the bits variable, the same input bits will be or'ed over those
       existing bits. A bitwise or is idempotent: (a | b | b) equals (a | b).
       Note that we therefore write that load operation as "hold |= etc" and not
       "hold += etc".

       Outside that loop, at the end of the function, hold is bitwise and'ed
       with (1<<bits)-1 to drop those excess bits so that, on function exit, we
       keep the invariant that (state->hold >> state->bits) == 0.
    */
    unsigned bits;              /* local strm->bits */
    uint64_t hold;              /* local strm->hold */
    unsigned lmask;             /* mask for first level of length codes */
    unsigned dmask;             /* mask for first level of distance codes */
    code const *lcode;          /* local strm->lencode */
    code const *dcode;          /* local strm->distcode */
    const code *here;           /* retrieved table entry */
    unsigned op;                /* code bits, operation, extra bits, or */
                                /*  window position, window bytes to copy */
    unsigned len;               /* match length, unused bytes */
    unsigned char *from;        /* where to copy match from */
    unsigned dist;              /* match distance */
    unsigned extra_safe;        /* copy chunks safely in all cases */

    /* copy state to local variables */
    state = (struct inflate_state *)strm->state;
    in = strm->next_in;
    last = in + (strm->avail_in - (INFLATE_FAST_MIN_HAVE - 1));
    out = strm->next_out;
    beg = out - (start - strm->avail_out);
    end = out + (strm->avail_out - (INFLATE_FAST_MIN_LEFT - 1));
    safe = out + strm->avail_out;
    wsize = state->wsize;
    whave = state->whave;
    wnext = state->wnext;
    window = state->window;
    hold = state->hold;
    bits = state->bits;
    lcode = state->lencode;
    dcode = state->distcode;
    lmask = (1U << state->lenbits) - 1;
    dmask = (1U << state->distbits) - 1;

    /* Detect if out and window point to the same memory allocation. In this instance it is
       necessary to use safe chunk copy functions to prevent overwriting the window. If the
       window is overwritten then future matches with far distances will fail to copy correctly. */
    extra_safe = (wsize != 0 && out >= window && out + INFLATE_FAST_MIN_LEFT <= window + state->wbufsize);

#define REFILL() do { \
        hold |= load_64_bits(in, bits); \
        in += 7; \
        in -= ((bits >> 3) & 7); \
        bits |= 56; \
    } while (0)

    /* decode literals and length/distances until end-of-block or not enough
       input data or output space */
    do {
        REFILL();
        here = lcode + (hold & lmask);
        if (here->op == 0) {
            *out++ = (unsigned char)(here->val);
            DROPBITS(here->bits);
            here = lcode + (hold & lmask);
            if (here->op == 0) {
                *out++ = (unsigned char)(here->val);
                DROPBITS(here->bits);
                here = lcode + (hold & lmask);
            }
        }
      dolen:
        DROPBITS(here->bits);
        op = here->op;
        if (op == 0) {                          /* literal */
            Tracevv((stderr, here->val >= 0x20 && here->val < 0x7f ?
                    "inflate:         literal '%c'\n" :
                    "inflate:         literal 0x%02x\n", here->val));
            *out++ = (unsigned char)(here->val);
        } else if (op & 16) {                     /* length base */
            len = here->val;
            op &= MAX_BITS;                       /* number of extra bits */
            len += BITS(op);
            DROPBITS(op);
            Tracevv((stderr, "inflate:         length %u\n", len));
            here = dcode + (hold & dmask);
            if (bits < MAX_BITS + MAX_DIST_EXTRA_BITS) {
                REFILL();
            }
          dodist:
            DROPBITS(here->bits);
            op = here->op;
            if (op & 16) {                      /* distance base */
                dist = here->val;
                op &= MAX_BITS;                 /* number of extra bits */
                dist += BITS(op);
#ifdef INFLATE_STRICT
                if (dist > state->dmax) {
                    SET_BAD("invalid distance too far back");
                    break;
                }
#endif
                DROPBITS(op);
                Tracevv((stderr, "inflate:         distance %u\n", dist));
                op = (unsigned)(out - beg);     /* max distance in output */
                if (dist > op) {                /* see if copy from window */
                    op = dist - op;             /* distance back in window */
                    if (op > whave) {
#ifdef INFLATE_ALLOW_INVALID_DISTANCE_TOOFAR_ARRR
                        if (state->sane) {
                            SET_BAD("invalid distance too far back");
                            break;
                        }
                        if (len <= op - whave) {
                            do {
                                *out++ = 0;
                            } while (--len);
                            continue;
                        }
                        len -= op - whave;
                        do {
                            *out++ = 0;
                        } while (--op > whave);
                        if (op == 0) {
                            from = out - dist;
                            do {
                                *out++ = *from++;
                            } while (--len);
                            continue;
                        }
#else
                        SET_BAD("invalid distance too far back");
                        break;
#endif
                    }
                    from = window;
                    if (wnext == 0) {           /* very common case */
                        from += wsize - op;
                    } else if (wnext >= op) {   /* contiguous in window */
                        from += wnext - op;
                    } else {                    /* wrap around window */
                        op -= wnext;
                        from += wsize - op;
                        if (op < len) {         /* some from end of window */
                            len -= op;
                            out = CHUNKCOPY_SAFE(out, from, op, safe);
                            from = window;      /* more from start of window */
                            op = wnext;
                            /* This (rare) case can create a situation where
                               the first chunkcopy below must be checked.
                             */
                        }
                    }
                    if (op < len) {             /* still need some from output */
                        len -= op;
                        if (!extra_safe) {
                            out = CHUNKCOPY_SAFE(out, from, op, safe);
                            out = CHUNKUNROLL(out, &dist, &len);
                            out = CHUNKCOPY_SAFE(out, out - dist, len, safe);
                        } else {
                            out = chunkcopy_safe(out, from, op, safe);
                            out = chunkcopy_safe(out, out - dist, len, safe);
                        }
                    } else {
#ifndef HAVE_MASKED_READWRITE
                        if (extra_safe)
                            out = chunkcopy_safe(out, from, len, safe);
                        else
#endif
                            out = CHUNKCOPY_SAFE(out, from, len, safe);
                    }
#ifndef HAVE_MASKED_READWRITE
                } else if (extra_safe) {
                    /* Whole reference is in range of current output. */
                        out = chunkcopy_safe(out, out - dist, len, safe);
#endif
                } else {
                    /* Whole reference is in range of current output.  No range checks are
                       necessary because we start with room for at least 258 bytes of output,
                       so unroll and roundoff operations can write beyond `out+len` so long
                       as they stay within 258 bytes of `out`.
                    */
                    if (dist >= len || dist >= CHUNKSIZE())
                        out = CHUNKCOPY(out, out - dist, len);
                    else
                        out = CHUNKMEMSET(out, out - dist, len);
                }
            } else if ((op & 64) == 0) {          /* 2nd level distance code */
                here = dcode + here->val + BITS(op);
                goto dodist;
            } else {
                SET_BAD("invalid distance code");
                break;
            }
        } else if ((op & 64) == 0) {              /* 2nd level length code */
            here = lcode + here->val + BITS(op);
            goto dolen;
        } else if (op & 32) {                     /* end-of-block */
            Tracevv((stderr, "inflate:         end of block\n"));
            state->mode = TYPE;
            break;
        } else {
            SET_BAD("invalid literal/length code");
            break;
        }
    } while (in < last && out < end);

    /* return unused bytes (on entry, bits < 8, so in won't go too far back) */
    len = bits >> 3;
    in -= len;
    bits -= len << 3;
    hold &= (UINT64_C(1) << bits) - 1;

    /* update state and return */
    strm->next_in = in;
    strm->next_out = out;
    strm->avail_in = (unsigned)(in < last ? (INFLATE_FAST_MIN_HAVE - 1) + (last - in)
                                          : (INFLATE_FAST_MIN_HAVE - 1) - (in - last));
    strm->avail_out = (unsigned)(out < end ? (INFLATE_FAST_MIN_LEFT - 1) + (end - out)
                                           : (INFLATE_FAST_MIN_LEFT - 1) - (out - end));

    Assert(bits <= 32, "Remaining bits greater than 32");
    state->hold = (uint32_t)hold;
    state->bits = bits;
    return;
}

/*
   inflate_fast() speedups that turned out slower (on a PowerPC G3 750CXe):
   - Using bit fields for code structure
   - Different op definition to avoid & for extra bits (do & for table bits)
   - Three separate decoding do-loops for direct, window, and wnext == 0
   - Special case for distance > 1 copies to do overlapped load and store copy
   - Explicit branch predictions (based on measured branch probabilities)
   - Deferring match copy and interspersed it with decoding subsequent codes
   - Swapping literal/length else
   - Swapping window/direct else
   - Larger unrolled copy loops (three is about right)
   - Moving len -= 3 statement into middle of loop
 */
