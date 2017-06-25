/* infcodes.c -- process literals and length/distance pairs
 * Copyright (C) 1995-2002 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "zutil.h"
#include "inftrees.h"
#include "infblock.h"
#include "infcodes.h"
#include "infutil.h"

/* simplify the use of the inflate_huft type with some defines */
#define exop word.what.Exop
#define bits word.what.Bits

typedef enum {        /* waiting for "i:"=input, "o:"=output, "x:"=nothing */
      START,    /* x: set up for LEN */
      LEN,      /* i: get length/literal/eob next */
      LENEXT,   /* i: getting length extra (have base) */
      DIST,     /* i: get distance next */
      DISTEXT,  /* i: getting distance extra */
      COPY,     /* o: copying bytes in window, waiting for space */
      LIT,      /* o: got literal, waiting for output space */
      WASH,     /* o: got eob, possibly still output waiting */
      END,      /* x: got eob and all data flushed */
      BADCODE}  /* x: got error */
inflate_codes_mode;

/* inflate codes private state */
struct inflate_codes_state {

  /* mode */
  inflate_codes_mode mode;      /* current inflate_codes mode */

  /* mode dependent information */
  uInt len;
  union {
    struct {
      inflate_huft *tree;       /* pointer into tree */
      uInt need;                /* bits needed */
    } code;             /* if LEN or DIST, where in tree */
    uInt lit;           /* if LIT, literal */
    struct {
      uInt get;                 /* bits to get for extra */
      uInt dist;                /* distance back to copy from */
    } copy;             /* if EXT or COPY, where and how much */
  } sub;                /* submode */

  /* mode independent information */
  Byte lbits;           /* ltree bits decoded per branch */
  Byte dbits;           /* dtree bits decoder per branch */
  inflate_huft *ltree;          /* literal/length/eob tree */
  inflate_huft *dtree;          /* distance tree */

};


local inflate_codes_statef *inflate_codes_new( /* bl, bd, tl, td, z) */
uInt bl, uInt bd,
inflate_huft *tl,
inflate_huft *td, /* need separate declaration for Borland C++ */
z_streamp z )
{
  inflate_codes_statef *c;

  if ((c = (inflate_codes_statef *)
       ZALLOC(z,1,sizeof(struct inflate_codes_state))) != Z_NULL)
  {
    c->mode = START;
    c->lbits = (Byte)bl;
    c->dbits = (Byte)bd;
    c->ltree = tl;
    c->dtree = td;
    Tracev((stderr, "inflate:       codes new\n"));
  }
  return c;
}


local int inflate_codes( /* s, z, r) */
inflate_blocks_statef *s,
z_streamp z,
int r )
{
  uInt j;               /* temporary storage */
  inflate_huft *t;      /* temporary pointer */
  uInt e;               /* extra bits or operation */
  uLong b;              /* bit buffer */
  uInt k;               /* bits in bit buffer */
  Bytef *p;             /* input data pointer */
  uInt n;               /* bytes available there */
  Bytef *q;             /* output window write pointer */
  uInt m;               /* bytes to end of window or read pointer */
  Bytef *f;             /* pointer to copy strings from */
  inflate_codes_statef *c = s->sub.decode.codes;  /* codes state */

  /* copy input/output information to locals (UPDATE macro restores) */
  LOAD

  /* process input and output based on current state */
  while (1) switch (c->mode)
  {             /* waiting for "i:"=input, "o:"=output, "x:"=nothing */
    case START:         /* x: set up for LEN */
#ifndef SLOW
      if (m >= 258 && n >= 10)
      {
        UPDATE
        r = inflate_fast(c->lbits, c->dbits, c->ltree, c->dtree, s, z);
        LOAD
        if (r != Z_OK)
        {
          c->mode = r == Z_STREAM_END ? WASH : BADCODE;
          break;
        }
      }
#endif /* !SLOW */
      c->sub.code.need = c->lbits;
      c->sub.code.tree = c->ltree;
      c->mode = LEN;
    case LEN:           /* i: get length/literal/eob next */
      j = c->sub.code.need;
      NEEDBITS(j)
      t = c->sub.code.tree + ((uInt)b & inflate_mask[j]);
      DUMPBITS(t->bits)
      e = (uInt)(t->exop);
      if (e == 0)               /* literal */
      {
        c->sub.lit = t->base;
        Tracevv((stderr, t->base >= 0x20 && t->base < 0x7f ?
                 "inflate:         literal '%c'\n" :
                 "inflate:         literal 0x%02x\n", t->base));
        c->mode = LIT;
        break;
      }
      if (e & 16)               /* length */
      {
        c->sub.copy.get = e & 15;
        c->len = t->base;
        c->mode = LENEXT;
        break;
      }
      if ((e & 64) == 0)        /* next table */
      {
        c->sub.code.need = e;
        c->sub.code.tree = t + t->base;
        break;
      }
      if (e & 32)               /* end of block */
      {
        Tracevv((stderr, "inflate:         end of block\n"));
        c->mode = WASH;
        break;
      }
      c->mode = BADCODE;        /* invalid code */
      z->msg = (char*)"invalid literal/length code";
      r = Z_DATA_ERROR;
      LEAVE
    case LENEXT:        /* i: getting length extra (have base) */
      j = c->sub.copy.get;
      NEEDBITS(j)
      c->len += (uInt)b & inflate_mask[j];
      DUMPBITS(j)
      c->sub.code.need = c->dbits;
      c->sub.code.tree = c->dtree;
      Tracevv((stderr, "inflate:         length %u\n", c->len));
      c->mode = DIST;
    case DIST:          /* i: get distance next */
      j = c->sub.code.need;
      NEEDBITS(j)
      t = c->sub.code.tree + ((uInt)b & inflate_mask[j]);
      DUMPBITS(t->bits)
      e = (uInt)(t->exop);
      if (e & 16)               /* distance */
      {
        c->sub.copy.get = e & 15;
        c->sub.copy.dist = t->base;
        c->mode = DISTEXT;
        break;
      }
      if ((e & 64) == 0)        /* next table */
      {
        c->sub.code.need = e;
        c->sub.code.tree = t + t->base;
        break;
      }
      c->mode = BADCODE;        /* invalid code */
      z->msg = (char*)"invalid distance code";
      r = Z_DATA_ERROR;
      LEAVE
    case DISTEXT:       /* i: getting distance extra */
      j = c->sub.copy.get;
      NEEDBITS(j)
      c->sub.copy.dist += (uInt)b & inflate_mask[j];
      DUMPBITS(j)
      Tracevv((stderr, "inflate:         distance %u\n", c->sub.copy.dist));
      c->mode = COPY;
    case COPY:          /* o: copying bytes in window, waiting for space */
      f = q - c->sub.copy.dist;
      while (f < s->window)             /* modulo window size-"while" instead */
        f += s->end - s->window;        /* of "if" handles invalid distances */
      while (c->len)
      {
        NEEDOUT
        OUTBYTE(*f++)
        if (f == s->end)
          f = s->window;
        c->len--;
      }
      c->mode = START;
      break;
    case LIT:           /* o: got literal, waiting for output space */
      NEEDOUT
      OUTBYTE(c->sub.lit)
      c->mode = START;
      break;
    case WASH:          /* o: got eob, possibly more output */
      if (k > 7)        /* return unused byte, if any */
      {
        Assert(k < 16, "inflate_codes grabbed too many bytes")
        k -= 8;
        n++;
        p--;            /* can always return one */
      }
      FLUSH
      if (s->read != s->write)
        LEAVE
      c->mode = END;
    case END:
      r = Z_STREAM_END;
      LEAVE
    case BADCODE:       /* x: got error */
      r = Z_DATA_ERROR;
      LEAVE
    default:
      r = Z_STREAM_ERROR;
      LEAVE
  }
#ifdef NEED_DUMMY_RETURN
  return Z_STREAM_ERROR;  /* Some dumb compilers complain without this */
#endif
}


local void inflate_codes_free( /* c, z) */
inflate_codes_statef *c,
z_streamp z )
{
  ZFREE(z, c);
  Tracev((stderr, "inflate:       codes free\n"));
}
