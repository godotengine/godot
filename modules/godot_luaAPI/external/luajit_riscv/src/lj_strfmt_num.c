/*
** String formatting for floating-point numbers.
** Copyright (C) 2005-2023 Mike Pall. See Copyright Notice in luajit.h
** Contributed by Peter Cawley.
*/

#include <stdio.h>

#define lj_strfmt_num_c
#define LUA_CORE

#include "lj_obj.h"
#include "lj_buf.h"
#include "lj_str.h"
#include "lj_strfmt.h"

/* -- Precomputed tables -------------------------------------------------- */

/* Rescale factors to push the exponent of a number towards zero. */
#define RESCALE_EXPONENTS(P, N) \
  P(308), P(289), P(270), P(250), P(231), P(212), P(193), P(173), P(154), \
  P(135), P(115), P(96), P(77), P(58), P(38), P(0), P(0), P(0), N(39), N(58), \
  N(77), N(96), N(116), N(135), N(154), N(174), N(193), N(212), N(231), \
  N(251), N(270), N(289)

#define ONE_E_P(X) 1e+0 ## X
#define ONE_E_N(X) 1e-0 ## X
static const int16_t rescale_e[] = { RESCALE_EXPONENTS(-, +) };
static const double rescale_n[] = { RESCALE_EXPONENTS(ONE_E_P, ONE_E_N) };
#undef ONE_E_N
#undef ONE_E_P

/*
** For p in range -70 through 57, this table encodes pairs (m, e) such that
** 4*2^p <= (uint8_t)m*10^e, and is the smallest value for which this holds.
*/
static const int8_t four_ulp_m_e[] = {
  34, -21, 68, -21, 14, -20, 28, -20, 55, -20, 2, -19, 3, -19, 5, -19, 9, -19,
  -82, -18, 35, -18, 7, -17, -117, -17, 28, -17, 56, -17, 112, -16, -33, -16,
  45, -16, 89, -16, -78, -15, 36, -15, 72, -15, -113, -14, 29, -14, 57, -14,
  114, -13, -28, -13, 46, -13, 91, -12, -74, -12, 37, -12, 73, -12, 15, -11, 3,
  -11, 59, -11, 2, -10, 3, -10, 5, -10, 1, -9, -69, -9, 38, -9, 75, -9, 15, -7,
  3, -7, 6, -7, 12, -6, -17, -7, 48, -7, 96, -7, -65, -6, 39, -6, 77, -6, -103,
  -5, 31, -5, 62, -5, 123, -4, -11, -4, 49, -4, 98, -4, -60, -3, 4, -2, 79, -3,
  16, -2, 32, -2, 63, -2, 2, -1, 25, 0, 5, 1, 1, 2, 2, 2, 4, 2, 8, 2, 16, 2,
  32, 2, 64, 2, -128, 2, 26, 2, 52, 2, 103, 3, -51, 3, 41, 4, 82, 4, -92, 4,
  33, 4, 66, 4, -124, 5, 27, 5, 53, 5, 105, 6, 21, 6, 42, 6, 84, 6, 17, 7, 34,
  7, 68, 7, 2, 8, 3, 8, 6, 8, 108, 9, -41, 9, 43, 10, 86, 9, -84, 10, 35, 10,
  69, 10, -118, 11, 28, 11, 55, 12, 11, 13, 22, 13, 44, 13, 88, 13, -80, 13,
  36, 13, 71, 13, -115, 14, 29, 14, 57, 14, 113, 15, -30, 15, 46, 15, 91, 15,
  19, 16, 37, 16, 73, 16, 2, 17, 3, 17, 6, 17
};

/* min(2^32-1, 10^e-1) for e in range 0 through 10 */
static uint32_t ndigits_dec_threshold[] = {
  0, 9U, 99U, 999U, 9999U, 99999U, 999999U,
  9999999U, 99999999U, 999999999U, 0xffffffffU
};

/* -- Helper functions ---------------------------------------------------- */

/* Compute the number of digits in the decimal representation of x. */
static MSize ndigits_dec(uint32_t x)
{
  MSize t = ((lj_fls(x | 1) * 77) >> 8) + 1; /* 2^8/77 is roughly log2(10) */
  return t + (x > ndigits_dec_threshold[t]);
}

#define WINT_R(x, sh, sc) \
  { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }

/* Write 9-digit unsigned integer to buffer. */
static char *lj_strfmt_wuint9(char *p, uint32_t u)
{
  uint32_t v = u / 10000, w;
  u -= v * 10000;
  w = v / 10000;
  v -= w * 10000;
  *p++ = (char)('0'+w);
  WINT_R(v, 23, 1000)
  WINT_R(v, 12, 100)
  WINT_R(v, 10, 10)
  *p++ = (char)('0'+v);
  WINT_R(u, 23, 1000)
  WINT_R(u, 12, 100)
  WINT_R(u, 10, 10)
  *p++ = (char)('0'+u);
  return p;
}
#undef WINT_R

/* -- Extended precision arithmetic --------------------------------------- */

/*
** The "nd" format is a fixed-precision decimal representation for numbers. It
** consists of up to 64 uint32_t values, with each uint32_t storing a value
** in the range [0, 1e9). A number in "nd" format consists of three variables:
**
**  uint32_t nd[64];
**  uint32_t ndlo;
**  uint32_t ndhi;
**
** The integral part of the number is stored in nd[0 ... ndhi], the value of
** which is sum{i in [0, ndhi] | nd[i] * 10^(9*i)}. If the fractional part of
** the number is zero, ndlo is zero. Otherwise, the fractional part is stored
** in nd[ndlo ... 63], the value of which is taken to be
** sum{i in [ndlo, 63] | nd[i] * 10^(9*(i-64))}.
**
** If the array part had 128 elements rather than 64, then every double would
** have an exact representation in "nd" format. With 64 elements, all integral
** doubles have an exact representation, and all non-integral doubles have
** enough digits to make both %.99e and %.99f do the right thing.
*/

#if LJ_64
#define ND_MUL2K_MAX_SHIFT	29
#define ND_MUL2K_DIV1E9(val)	((uint32_t)((val) / 1000000000))
#else
#define ND_MUL2K_MAX_SHIFT	11
#define ND_MUL2K_DIV1E9(val)	((uint32_t)((val) >> 9) / 1953125)
#endif

/* Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero). */
static uint32_t nd_mul2k(uint32_t* nd, uint32_t ndhi, uint32_t k,
			 uint32_t carry_in, SFormat sf)
{
  uint32_t i, ndlo = 0, start = 1;
  /* Performance hacks. */
  if (k > ND_MUL2K_MAX_SHIFT*2 && STRFMT_FP(sf) != STRFMT_FP(STRFMT_T_FP_F)) {
    start = ndhi - (STRFMT_PREC(sf) + 17) / 8;
  }
  /* Real logic. */
  while (k >= ND_MUL2K_MAX_SHIFT) {
    for (i = ndlo; i <= ndhi; i++) {
      uint64_t val = ((uint64_t)nd[i] << ND_MUL2K_MAX_SHIFT) | carry_in;
      carry_in = ND_MUL2K_DIV1E9(val);
      nd[i] = (uint32_t)val - carry_in * 1000000000;
    }
    if (carry_in) {
      nd[++ndhi] = carry_in; carry_in = 0;
      if (start++ == ndlo) ++ndlo;
    }
    k -= ND_MUL2K_MAX_SHIFT;
  }
  if (k) {
    for (i = ndlo; i <= ndhi; i++) {
      uint64_t val = ((uint64_t)nd[i] << k) | carry_in;
      carry_in = ND_MUL2K_DIV1E9(val);
      nd[i] = (uint32_t)val - carry_in * 1000000000;
    }
    if (carry_in) nd[++ndhi] = carry_in;
  }
  return ndhi;
}

/* Divide nd by 2^k (ndlo is assumed to be zero). */
static uint32_t nd_div2k(uint32_t* nd, uint32_t ndhi, uint32_t k, SFormat sf)
{
  uint32_t ndlo = 0, stop1 = ~0, stop2 = ~0;
  /* Performance hacks. */
  if (!ndhi) {
    if (!nd[0]) {
      return 0;
    } else {
      uint32_t s = lj_ffs(nd[0]);
      if (s >= k) { nd[0] >>= k; return 0; }
      nd[0] >>= s; k -= s;
    }
  }
  if (k > 18) {
    if (STRFMT_FP(sf) == STRFMT_FP(STRFMT_T_FP_F)) {
      stop1 = 63 - (int32_t)STRFMT_PREC(sf) / 9;
    } else {
      int32_t floorlog2 = ndhi * 29 + lj_fls(nd[ndhi]) - k;
      int32_t floorlog10 = (int32_t)(floorlog2 * 0.30102999566398114);
      stop1 = 62 + (floorlog10 - (int32_t)STRFMT_PREC(sf)) / 9;
      stop2 = 61 + ndhi - (int32_t)STRFMT_PREC(sf) / 8;
    }
  }
  /* Real logic. */
  while (k >= 9) {
    uint32_t i = ndhi, carry = 0;
    for (;;) {
      uint32_t val = nd[i];
      nd[i] = (val >> 9) + carry;
      carry = (val & 0x1ff) * 1953125;
      if (i == ndlo) break;
      i = (i - 1) & 0x3f;
    }
    if (ndlo != stop1 && ndlo != stop2) {
      if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
      if (!nd[ndhi]) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
    } else if (!nd[ndhi]) {
      if (ndhi != ndlo) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
      else return ndlo;
    }
    k -= 9;
  }
  if (k) {
    uint32_t mask = (1U << k) - 1, mul = 1000000000 >> k, i = ndhi, carry = 0;
    for (;;) {
      uint32_t val = nd[i];
      nd[i] = (val >> k) + carry;
      carry = (val & mask) * mul;
      if (i == ndlo) break;
      i = (i - 1) & 0x3f;
    }
    if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
  }
  return ndlo;
}

/* Add m*10^e to nd (assumes ndlo <= e/9 <= ndhi and 0 <= m <= 9). */
static uint32_t nd_add_m10e(uint32_t* nd, uint32_t ndhi, uint8_t m, int32_t e)
{
  uint32_t i, carry;
  if (e >= 0) {
    i = (uint32_t)e/9;
    carry = m * (ndigits_dec_threshold[e - (int32_t)i*9] + 1);
  } else {
    int32_t f = (e-8)/9;
    i = (uint32_t)(64 + f);
    carry = m * (ndigits_dec_threshold[e - f*9] + 1);
  }
  for (;;) {
    uint32_t val = nd[i] + carry;
    if (LJ_UNLIKELY(val >= 1000000000)) {
      val -= 1000000000;
      nd[i] = val;
      if (LJ_UNLIKELY(i == ndhi)) {
	ndhi = (ndhi + 1) & 0x3f;
	nd[ndhi] = 1;
	break;
      }
      carry = 1;
      i = (i + 1) & 0x3f;
    } else {
      nd[i] = val;
      break;
    }
  }
  return ndhi;
}

/* Test whether two "nd" values are equal in their most significant digits. */
static int nd_similar(uint32_t* nd, uint32_t ndhi, uint32_t* ref, MSize hilen,
		      MSize prec)
{
  char nd9[9], ref9[9];
  if (hilen <= prec) {
    if (LJ_UNLIKELY(nd[ndhi] != *ref)) return 0;
    prec -= hilen; ref--; ndhi = (ndhi - 1) & 0x3f;
    if (prec >= 9) {
      if (LJ_UNLIKELY(nd[ndhi] != *ref)) return 0;
      prec -= 9; ref--; ndhi = (ndhi - 1) & 0x3f;
    }
  } else {
    prec -= hilen - 9;
  }
  lj_assertX(prec < 9, "bad precision %d", prec);
  lj_strfmt_wuint9(nd9, nd[ndhi]);
  lj_strfmt_wuint9(ref9, *ref);
  return !memcmp(nd9, ref9, prec) && (nd9[prec] < '5') == (ref9[prec] < '5');
}

/* -- Formatted conversions to buffer ------------------------------------- */

/* Write formatted floating-point number to either sb or p. */
static char *lj_strfmt_wfnum(SBuf *sb, SFormat sf, lua_Number n, char *p)
{
  MSize width = STRFMT_WIDTH(sf), prec = STRFMT_PREC(sf), len;
  TValue t;
  t.n = n;
  if (LJ_UNLIKELY((t.u32.hi << 1) >= 0xffe00000)) {
    /* Handle non-finite values uniformly for %a, %e, %f, %g. */
    int prefix = 0, ch = (sf & STRFMT_F_UPPER) ? 0x202020 : 0;
    if (((t.u32.hi & 0x000fffff) | t.u32.lo) != 0) {
      ch ^= ('n' << 16) | ('a' << 8) | 'n';
      if ((sf & STRFMT_F_SPACE)) prefix = ' ';
    } else {
      ch ^= ('i' << 16) | ('n' << 8) | 'f';
      if ((t.u32.hi & 0x80000000)) prefix = '-';
      else if ((sf & STRFMT_F_PLUS)) prefix = '+';
      else if ((sf & STRFMT_F_SPACE)) prefix = ' ';
    }
    len = 3 + (prefix != 0);
    if (!p) p = lj_buf_more(sb, width > len ? width : len);
    if (!(sf & STRFMT_F_LEFT)) while (width-- > len) *p++ = ' ';
    if (prefix) *p++ = prefix;
    *p++ = (char)(ch >> 16); *p++ = (char)(ch >> 8); *p++ = (char)ch;
  } else if (STRFMT_FP(sf) == STRFMT_FP(STRFMT_T_FP_A)) {
    /* %a */
    const char *hexdig = (sf & STRFMT_F_UPPER) ? "0123456789ABCDEFPX"
					       : "0123456789abcdefpx";
    int32_t e = (t.u32.hi >> 20) & 0x7ff;
    char prefix = 0, eprefix = '+';
    if (t.u32.hi & 0x80000000) prefix = '-';
    else if ((sf & STRFMT_F_PLUS)) prefix = '+';
    else if ((sf & STRFMT_F_SPACE)) prefix = ' ';
    t.u32.hi &= 0xfffff;
    if (e) {
      t.u32.hi |= 0x100000;
      e -= 1023;
    } else if (t.u32.lo | t.u32.hi) {
      /* Non-zero denormal - normalise it. */
      uint32_t shift = t.u32.hi ? 20-lj_fls(t.u32.hi) : 52-lj_fls(t.u32.lo);
      e = -1022 - shift;
      t.u64 <<= shift;
    }
    /* abs(n) == t.u64 * 2^(e - 52) */
    /* If n != 0, bit 52 of t.u64 is set, and is the highest set bit. */
    if ((int32_t)prec < 0) {
      /* Default precision: use smallest precision giving exact result. */
      prec = t.u32.lo ? 13-lj_ffs(t.u32.lo)/4 : 5-lj_ffs(t.u32.hi|0x100000)/4;
    } else if (prec < 13) {
      /* Precision is sufficiently low as to maybe require rounding. */
      t.u64 += (((uint64_t)1) << (51 - prec*4));
    }
    if (e < 0) {
      eprefix = '-';
      e = -e;
    }
    len = 5 + ndigits_dec((uint32_t)e) + prec + (prefix != 0)
	    + ((prec | (sf & STRFMT_F_ALT)) != 0);
    if (!p) p = lj_buf_more(sb, width > len ? width : len);
    if (!(sf & (STRFMT_F_LEFT | STRFMT_F_ZERO))) {
      while (width-- > len) *p++ = ' ';
    }
    if (prefix) *p++ = prefix;
    *p++ = '0';
    *p++ = hexdig[17]; /* x or X */
    if ((sf & (STRFMT_F_LEFT | STRFMT_F_ZERO)) == STRFMT_F_ZERO) {
      while (width-- > len) *p++ = '0';
    }
    *p++ = '0' + (t.u32.hi >> 20); /* Usually '1', sometimes '0' or '2'. */
    if ((prec | (sf & STRFMT_F_ALT))) {
      /* Emit fractional part. */
      char *q = p + 1 + prec;
      *p = '.';
      if (prec < 13) t.u64 >>= (52 - prec*4);
      else while (prec > 13) p[prec--] = '0';
      while (prec) { p[prec--] = hexdig[t.u64 & 15]; t.u64 >>= 4; }
      p = q;
    }
    *p++ = hexdig[16]; /* p or P */
    *p++ = eprefix; /* + or - */
    p = lj_strfmt_wint(p, e);
  } else {
    /* %e or %f or %g - begin by converting n to "nd" format. */
    uint32_t nd[64];
    uint32_t ndhi = 0, ndlo, i;
    int32_t e = (t.u32.hi >> 20) & 0x7ff, ndebias = 0;
    char prefix = 0, *q;
    if (t.u32.hi & 0x80000000) prefix = '-';
    else if ((sf & STRFMT_F_PLUS)) prefix = '+';
    else if ((sf & STRFMT_F_SPACE)) prefix = ' ';
    prec += ((int32_t)prec >> 31) & 7; /* Default precision is 6. */
    if (STRFMT_FP(sf) == STRFMT_FP(STRFMT_T_FP_G)) {
      /* %g - decrement precision if non-zero (to make it like %e). */
      prec--;
      prec ^= (uint32_t)((int32_t)prec >> 31);
    }
    if ((sf & STRFMT_T_FP_E) && prec < 14 && n != 0) {
      /* Precision is sufficiently low that rescaling will probably work. */
      if ((ndebias = rescale_e[e >> 6])) {
	t.n = n * rescale_n[e >> 6];
	if (LJ_UNLIKELY(!e)) t.n *= 1e10, ndebias -= 10;
	t.u64 -= 2; /* Convert 2ulp below (later we convert 2ulp above). */
	nd[0] = 0x100000 | (t.u32.hi & 0xfffff);
	e = ((t.u32.hi >> 20) & 0x7ff) - 1075 - (ND_MUL2K_MAX_SHIFT < 29);
	goto load_t_lo; rescale_failed:
	t.n = n;
	e = (t.u32.hi >> 20) & 0x7ff;
	ndebias = ndhi = 0;
      }
    }
    nd[0] = t.u32.hi & 0xfffff;
    if (e == 0) e++; else nd[0] |= 0x100000;
    e -= 1043;
    if (t.u32.lo) {
      e -= 32 + (ND_MUL2K_MAX_SHIFT < 29); load_t_lo:
#if ND_MUL2K_MAX_SHIFT >= 29
      nd[0] = (nd[0] << 3) | (t.u32.lo >> 29);
      ndhi = nd_mul2k(nd, ndhi, 29, t.u32.lo & 0x1fffffff, sf);
#elif ND_MUL2K_MAX_SHIFT >= 11
      ndhi = nd_mul2k(nd, ndhi, 11, t.u32.lo >> 21, sf);
      ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo >> 10) & 0x7ff, sf);
      ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo <<  1) & 0x7ff, sf);
#else
#error "ND_MUL2K_MAX_SHIFT too small"
#endif
    }
    if (e >= 0) {
      ndhi = nd_mul2k(nd, ndhi, (uint32_t)e, 0, sf);
      ndlo = 0;
    } else {
      ndlo = nd_div2k(nd, ndhi, (uint32_t)-e, sf);
      if (ndhi && !nd[ndhi]) ndhi--;
    }
    /* abs(n) == nd * 10^ndebias (for slightly loose interpretation of ==) */
    if ((sf & STRFMT_T_FP_E)) {
      /* %e or %g - assume %e and start by calculating nd's exponent (nde). */
      char eprefix = '+';
      int32_t nde = -1;
      MSize hilen;
      if (ndlo && !nd[ndhi]) {
	ndhi = 64; do {} while (!nd[--ndhi]);
	nde -= 64 * 9;
      }
      hilen = ndigits_dec(nd[ndhi]);
      nde += ndhi * 9 + hilen;
      if (ndebias) {
	/*
	** Rescaling was performed, but this introduced some error, and might
	** have pushed us across a rounding boundary. We check whether this
	** error affected the result by introducing even more error (2ulp in
	** either direction), and seeing whether a rounding boundary was
	** crossed. Having already converted the -2ulp case, we save off its
	** most significant digits, convert the +2ulp case, and compare them.
	*/
	int32_t eidx = e + 70 + (ND_MUL2K_MAX_SHIFT < 29)
			 + (t.u32.lo >= 0xfffffffe && !(~t.u32.hi << 12));
	const int8_t *m_e = four_ulp_m_e + eidx * 2;
	lj_assertG_(G(sbufL(sb)), 0 <= eidx && eidx < 128, "bad eidx %d", eidx);
	nd[33] = nd[ndhi];
	nd[32] = nd[(ndhi - 1) & 0x3f];
	nd[31] = nd[(ndhi - 2) & 0x3f];
	nd_add_m10e(nd, ndhi, (uint8_t)*m_e, m_e[1]);
	if (LJ_UNLIKELY(!nd_similar(nd, ndhi, nd + 33, hilen, prec + 1))) {
	  goto rescale_failed;
	}
      }
      if ((int32_t)(prec - nde) < (0x3f & -(int32_t)ndlo) * 9) {
	/* Precision is sufficiently low as to maybe require rounding. */
	ndhi = nd_add_m10e(nd, ndhi, 5, nde - prec - 1);
	nde += (hilen != ndigits_dec(nd[ndhi]));
      }
      nde += ndebias;
      if ((sf & STRFMT_T_FP_F)) {
	/* %g */
	if ((int32_t)prec >= nde && nde >= -4) {
	  if (nde < 0) ndhi = 0;
	  prec -= nde;
	  goto g_format_like_f;
	} else if (!(sf & STRFMT_F_ALT) && prec && width > 5) {
	  /* Decrease precision in order to strip trailing zeroes. */
	  char tail[9];
	  uint32_t maxprec = hilen - 1 + ((ndhi - ndlo) & 0x3f) * 9;
	  if (prec >= maxprec) prec = maxprec;
	  else ndlo = (ndhi - (((int32_t)(prec - hilen) + 9) / 9)) & 0x3f;
	  i = prec - hilen - (((ndhi - ndlo) & 0x3f) * 9) + 10;
	  lj_strfmt_wuint9(tail, nd[ndlo]);
	  while (prec && tail[--i] == '0') {
	    prec--;
	    if (!i) {
	      if (ndlo == ndhi) { prec = 0; break; }
	      lj_strfmt_wuint9(tail, nd[++ndlo]);
	      i = 9;
	    }
	  }
	}
      }
      if (nde < 0) {
	/* Make nde non-negative. */
	eprefix = '-';
	nde = -nde;
      }
      len = 3 + prec + (prefix != 0) + ndigits_dec((uint32_t)nde) + (nde < 10)
	      + ((prec | (sf & STRFMT_F_ALT)) != 0);
      if (!p) p = lj_buf_more(sb, (width > len ? width : len) + 5);
      if (!(sf & (STRFMT_F_LEFT | STRFMT_F_ZERO))) {
	while (width-- > len) *p++ = ' ';
      }
      if (prefix) *p++ = prefix;
      if ((sf & (STRFMT_F_LEFT | STRFMT_F_ZERO)) == STRFMT_F_ZERO) {
	while (width-- > len) *p++ = '0';
      }
      q = lj_strfmt_wint(p + 1, nd[ndhi]);
      p[0] = p[1]; /* Put leading digit in the correct place. */
      if ((prec | (sf & STRFMT_F_ALT))) {
	/* Emit fractional part. */
	p[1] = '.'; p += 2;
	prec -= (MSize)(q - p); p = q; /* Account for digits already emitted. */
	/* Then emit chunks of 9 digits (this may emit 8 digits too many). */
	for (i = ndhi; (int32_t)prec > 0 && i != ndlo; prec -= 9) {
	  i = (i - 1) & 0x3f;
	  p = lj_strfmt_wuint9(p, nd[i]);
	}
	if ((sf & STRFMT_T_FP_F) && !(sf & STRFMT_F_ALT)) {
	  /* %g (and not %#g) - strip trailing zeroes. */
	  p += (int32_t)prec & ((int32_t)prec >> 31);
	  while (p[-1] == '0') p--;
	  if (p[-1] == '.') p--;
	} else {
	  /* %e (or %#g) - emit trailing zeroes. */
	  while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
	  p += (int32_t)prec;
	}
      } else {
	p++;
      }
      *p++ = (sf & STRFMT_F_UPPER) ? 'E' : 'e';
      *p++ = eprefix; /* + or - */
      if (nde < 10) *p++ = '0'; /* Always at least two digits of exponent. */
      p = lj_strfmt_wint(p, nde);
    } else {
      /* %f (or, shortly, %g in %f style) */
      if (prec < (MSize)(0x3f & -(int32_t)ndlo) * 9) {
	/* Precision is sufficiently low as to maybe require rounding. */
	ndhi = nd_add_m10e(nd, ndhi, 5, 0 - prec - 1);
      }
      g_format_like_f:
      if ((sf & STRFMT_T_FP_E) && !(sf & STRFMT_F_ALT) && prec && width) {
	/* Decrease precision in order to strip trailing zeroes. */
	if (ndlo) {
	  /* nd has a fractional part; we need to look at its digits. */
	  char tail[9];
	  uint32_t maxprec = (64 - ndlo) * 9;
	  if (prec >= maxprec) prec = maxprec;
	  else ndlo = 64 - (prec + 8) / 9;
	  i = prec - ((63 - ndlo) * 9);
	  lj_strfmt_wuint9(tail, nd[ndlo]);
	  while (prec && tail[--i] == '0') {
	    prec--;
	    if (!i) {
	      if (ndlo == 63) { prec = 0; break; }
	      lj_strfmt_wuint9(tail, nd[++ndlo]);
	      i = 9;
	    }
	  }
	} else {
	  /* nd has no fractional part, so precision goes straight to zero. */
	  prec = 0;
	}
      }
      len = ndhi * 9 + ndigits_dec(nd[ndhi]) + prec + (prefix != 0)
		     + ((prec | (sf & STRFMT_F_ALT)) != 0);
      if (!p) p = lj_buf_more(sb, (width > len ? width : len) + 8);
      if (!(sf & (STRFMT_F_LEFT | STRFMT_F_ZERO))) {
	while (width-- > len) *p++ = ' ';
      }
      if (prefix) *p++ = prefix;
      if ((sf & (STRFMT_F_LEFT | STRFMT_F_ZERO)) == STRFMT_F_ZERO) {
	while (width-- > len) *p++ = '0';
      }
      /* Emit integer part. */
      p = lj_strfmt_wint(p, nd[ndhi]);
      i = ndhi;
      while (i) p = lj_strfmt_wuint9(p, nd[--i]);
      if ((prec | (sf & STRFMT_F_ALT))) {
	/* Emit fractional part. */
	*p++ = '.';
	/* Emit chunks of 9 digits (this may emit 8 digits too many). */
	while ((int32_t)prec > 0 && i != ndlo) {
	  i = (i - 1) & 0x3f;
	  p = lj_strfmt_wuint9(p, nd[i]);
	  prec -= 9;
	}
	if ((sf & STRFMT_T_FP_E) && !(sf & STRFMT_F_ALT)) {
	  /* %g (and not %#g) - strip trailing zeroes. */
	  p += (int32_t)prec & ((int32_t)prec >> 31);
	  while (p[-1] == '0') p--;
	  if (p[-1] == '.') p--;
	} else {
	  /* %f (or %#g) - emit trailing zeroes. */
	  while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
	  p += (int32_t)prec;
	}
      }
    }
  }
  if ((sf & STRFMT_F_LEFT)) while (width-- > len) *p++ = ' ';
  return p;
}

/* Add formatted floating-point number to buffer. */
SBuf *lj_strfmt_putfnum(SBuf *sb, SFormat sf, lua_Number n)
{
  sb->w = lj_strfmt_wfnum(sb, sf, n, NULL);
  return sb;
}

/* -- Conversions to strings ---------------------------------------------- */

/* Convert number to string. */
GCstr * LJ_FASTCALL lj_strfmt_num(lua_State *L, cTValue *o)
{
  char buf[STRFMT_MAXBUF_NUM];
  MSize len = (MSize)(lj_strfmt_wfnum(NULL, STRFMT_G14, o->n, buf) - buf);
  return lj_str_new(L, buf, len);
}

