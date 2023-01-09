/**************************************************************************
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 * Copyright (c) 2008 VMware, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include "util/format/u_format.h"
#include "util/format/u_format_fxt1.h"
#include "util/format/u_format_pack.h"
#include "util/format_srgb.h"
#include "util/u_math.h"

#define RCOMP 0
#define GCOMP 1
#define BCOMP 2
#define ACOMP 3

#define FXT1_BLOCK_SIZE 16

static void
fxt1_encode (uint32_t width, uint32_t height, int32_t comps,
             const void *source, int32_t srcRowStride,
             void *dest, int32_t destRowStride);

static void
fxt1_decode_1 (const void *texture, int32_t stride,
               int32_t i, int32_t j, uint8_t *rgba);

/***************************************************************************\
 * FXT1 encoder
 *
 * The encoder was built by reversing the decoder,
 * and is vaguely based on Texus2 by 3dfx. Note that this code
 * is merely a proof of concept, since it is highly UNoptimized;
 * moreover, it is sub-optimal due to initial conditions passed
 * to Lloyd's algorithm (the interpolation modes are even worse).
\***************************************************************************/


#define MAX_COMP 4 /* ever needed maximum number of components in texel */
#define MAX_VECT 4 /* ever needed maximum number of base vectors to find */
#define N_TEXELS 32 /* number of texels in a block (always 32) */
#define LL_N_REP 50 /* number of iterations in lloyd's vq */
#define LL_RMS_D 10 /* fault tolerance (maximum delta) */
#define LL_RMS_E 255 /* fault tolerance (maximum error) */
#define ALPHA_TS 2 /* alpha threshold: (255 - ALPHA_TS) deemed opaque */
static const uint32_t zero = 0;
#define ISTBLACK(v) (memcmp(&(v), &zero, sizeof(zero)) == 0)

/*
 * Define a 64-bit unsigned integer type and macros
 */
#if 1

#define FX64_NATIVE 1

typedef uint64_t Fx64;

#define FX64_MOV32(a, b) a = b
#define FX64_OR32(a, b)  a |= b
#define FX64_SHL(a, c)   a <<= c

#else

#define FX64_NATIVE 0

typedef struct {
   uint32_t lo, hi;
} Fx64;

#define FX64_MOV32(a, b) a.lo = b
#define FX64_OR32(a, b)  a.lo |= b

#define FX64_SHL(a, c)                                 \
   do {                                                \
       if ((c) >= 32) {                                \
          a.hi = a.lo << ((c) - 32);                   \
          a.lo = 0;                                    \
       } else {                                        \
          a.hi = (a.hi << (c)) | (a.lo >> (32 - (c))); \
          a.lo <<= (c);                                \
       }                                               \
   } while (0)

#endif


#define F(i) (float)1 /* can be used to obtain an oblong metric: 0.30 / 0.59 / 0.11 */
#define SAFECDOT 1 /* for paranoids */

#define MAKEIVEC(NV, NC, IV, B, V0, V1)  \
   do {                                  \
      /* compute interpolation vector */ \
      float d2 = 0.0F;                   \
      float rd2;                         \
                                         \
      for (i = 0; i < NC; i++) {         \
         IV[i] = (V1[i] - V0[i]) * F(i); \
         d2 += IV[i] * IV[i];            \
      }                                  \
      rd2 = (float)NV / d2;              \
      B = 0;                             \
      for (i = 0; i < NC; i++) {         \
         IV[i] *= F(i);                  \
         B -= IV[i] * V0[i];             \
         IV[i] *= rd2;                   \
      }                                  \
      B = B * rd2 + 0.5f;                \
   } while (0)

#define CALCCDOT(TEXEL, NV, NC, IV, B, V)\
   do {                                  \
      float dot = 0.0F;                  \
      for (i = 0; i < NC; i++) {         \
         dot += V[i] * IV[i];            \
      }                                  \
      TEXEL = (int32_t)(dot + B);        \
      if (SAFECDOT) {                    \
         if (TEXEL < 0) {                \
            TEXEL = 0;                   \
         } else if (TEXEL > NV) {        \
            TEXEL = NV;                  \
         }                               \
      }                                  \
   } while (0)


static int32_t
fxt1_bestcol (float vec[][MAX_COMP], int32_t nv,
              uint8_t input[MAX_COMP], int32_t nc)
{
   int32_t i, j, best = -1;
   float err = 1e9; /* big enough */

   for (j = 0; j < nv; j++) {
      float e = 0.0F;
      for (i = 0; i < nc; i++) {
         e += (vec[j][i] - input[i]) * (vec[j][i] - input[i]);
      }
      if (e < err) {
         err = e;
         best = j;
      }
   }

   return best;
}


static int32_t
fxt1_worst (float vec[MAX_COMP],
            uint8_t input[N_TEXELS][MAX_COMP], int32_t nc, int32_t n)
{
   int32_t i, k, worst = -1;
   float err = -1.0F; /* small enough */

   for (k = 0; k < n; k++) {
      float e = 0.0F;
      for (i = 0; i < nc; i++) {
         e += (vec[i] - input[k][i]) * (vec[i] - input[k][i]);
      }
      if (e > err) {
         err = e;
         worst = k;
      }
   }

   return worst;
}


static int32_t
fxt1_variance (uint8_t input[N_TEXELS / 2][MAX_COMP], int32_t nc)
{
   const int n = N_TEXELS / 2;
   int32_t i, k, best = 0;
   int32_t sx, sx2;
   double var, maxvar = -1; /* small enough */
   double teenth = 1.0 / n;

   for (i = 0; i < nc; i++) {
      sx = sx2 = 0;
      for (k = 0; k < n; k++) {
         int32_t t = input[k][i];
         sx += t;
         sx2 += t * t;
      }
      var = sx2 * teenth - sx * sx * teenth * teenth;
      if (maxvar < var) {
         maxvar = var;
         best = i;
      }
   }

   return best;
}


static int32_t
fxt1_choose (float vec[][MAX_COMP], int32_t nv,
             uint8_t input[N_TEXELS][MAX_COMP], int32_t nc, int32_t n)
{
#if 0
   /* Choose colors from a grid.
    */
   int32_t i, j;

   for (j = 0; j < nv; j++) {
      int32_t m = j * (n - 1) / (nv - 1);
      for (i = 0; i < nc; i++) {
         vec[j][i] = input[m][i];
      }
   }
#else
   /* Our solution here is to find the darkest and brightest colors in
    * the 8x4 tile and use those as the two representative colors.
    * There are probably better algorithms to use (histogram-based).
    */
   int32_t i, j, k;
   int32_t minSum = 2000; /* big enough */
   int32_t maxSum = -1; /* small enough */
   int32_t minCol = 0; /* phoudoin: silent compiler! */
   int32_t maxCol = 0; /* phoudoin: silent compiler! */

   struct {
      int32_t flag;
      int32_t key;
      int32_t freq;
      int32_t idx;
   } hist[N_TEXELS];
   int32_t lenh = 0;

   memset(hist, 0, sizeof(hist));

   for (k = 0; k < n; k++) {
      int32_t l;
      int32_t key = 0;
      int32_t sum = 0;
      for (i = 0; i < nc; i++) {
         key <<= 8;
         key |= input[k][i];
         sum += input[k][i];
      }
      for (l = 0; l < n; l++) {
         if (!hist[l].flag) {
            /* alloc new slot */
            hist[l].flag = !0;
            hist[l].key = key;
            hist[l].freq = 1;
            hist[l].idx = k;
            lenh = l + 1;
            break;
         } else if (hist[l].key == key) {
            hist[l].freq++;
            break;
         }
      }
      if (minSum > sum) {
         minSum = sum;
         minCol = k;
      }
      if (maxSum < sum) {
         maxSum = sum;
         maxCol = k;
      }
   }

   if (lenh <= nv) {
      for (j = 0; j < lenh; j++) {
         for (i = 0; i < nc; i++) {
            vec[j][i] = (float)input[hist[j].idx][i];
         }
      }
      for (; j < nv; j++) {
         for (i = 0; i < nc; i++) {
            vec[j][i] = vec[0][i];
         }
      }
      return 0;
   }

   for (j = 0; j < nv; j++) {
      for (i = 0; i < nc; i++) {
         vec[j][i] = ((nv - 1 - j) * input[minCol][i] + j * input[maxCol][i] + (nv - 1) / 2) / (float)(nv - 1);
      }
   }
#endif

   return !0;
}


static int32_t
fxt1_lloyd (float vec[][MAX_COMP], int32_t nv,
            uint8_t input[N_TEXELS][MAX_COMP], int32_t nc, int32_t n)
{
   /* Use the generalized lloyd's algorithm for VQ:
    *     find 4 color vectors.
    *
    *     for each sample color
    *         sort to nearest vector.
    *
    *     replace each vector with the centroid of its matching colors.
    *
    *     repeat until RMS doesn't improve.
    *
    *     if a color vector has no samples, or becomes the same as another
    *     vector, replace it with the color which is farthest from a sample.
    *
    * vec[][MAX_COMP]           initial vectors and resulting colors
    * nv                        number of resulting colors required
    * input[N_TEXELS][MAX_COMP] input texels
    * nc                        number of components in input / vec
    * n                         number of input samples
    */

   int32_t sum[MAX_VECT][MAX_COMP]; /* used to accumulate closest texels */
   int32_t cnt[MAX_VECT]; /* how many times a certain vector was chosen */
   float error, lasterror = 1e9;

   int32_t i, j, k, rep;

   /* the quantizer */
   for (rep = 0; rep < LL_N_REP; rep++) {
      /* reset sums & counters */
      for (j = 0; j < nv; j++) {
         for (i = 0; i < nc; i++) {
            sum[j][i] = 0;
         }
         cnt[j] = 0;
      }
      error = 0;

      /* scan whole block */
      for (k = 0; k < n; k++) {
#if 1
         int32_t best = -1;
         float err = 1e9; /* big enough */
         /* determine best vector */
         for (j = 0; j < nv; j++) {
            float e = (vec[j][0] - input[k][0]) * (vec[j][0] - input[k][0]) +
                      (vec[j][1] - input[k][1]) * (vec[j][1] - input[k][1]) +
                      (vec[j][2] - input[k][2]) * (vec[j][2] - input[k][2]);
            if (nc == 4) {
               e += (vec[j][3] - input[k][3]) * (vec[j][3] - input[k][3]);
            }
            if (e < err) {
               err = e;
               best = j;
            }
         }
#else
         int32_t best = fxt1_bestcol(vec, nv, input[k], nc, &err);
#endif
         assert(best >= 0);
         /* add in closest color */
         for (i = 0; i < nc; i++) {
            sum[best][i] += input[k][i];
         }
         /* mark this vector as used */
         cnt[best]++;
         /* accumulate error */
         error += err;
      }

      /* check RMS */
      if ((error < LL_RMS_E) ||
          ((error < lasterror) && ((lasterror - error) < LL_RMS_D))) {
         return !0; /* good match */
      }
      lasterror = error;

      /* move each vector to the barycenter of its closest colors */
      for (j = 0; j < nv; j++) {
         if (cnt[j]) {
            float div = 1.0F / cnt[j];
            for (i = 0; i < nc; i++) {
               vec[j][i] = div * sum[j][i];
            }
         } else {
            /* this vec has no samples or is identical with a previous vec */
            int32_t worst = fxt1_worst(vec[j], input, nc, n);
            for (i = 0; i < nc; i++) {
               vec[j][i] = input[worst][i];
            }
         }
      }
   }

   return 0; /* could not converge fast enough */
}


static void
fxt1_quantize_CHROMA (uint32_t *cc,
                      uint8_t input[N_TEXELS][MAX_COMP])
{
   const int32_t n_vect = 4; /* 4 base vectors to find */
   const int32_t n_comp = 3; /* 3 components: R, G, B */
   float vec[MAX_VECT][MAX_COMP];
   int32_t i, j, k;
   Fx64 hi; /* high quadword */
   uint32_t lohi, lolo; /* low quadword: hi dword, lo dword */

   if (fxt1_choose(vec, n_vect, input, n_comp, N_TEXELS) != 0) {
      fxt1_lloyd(vec, n_vect, input, n_comp, N_TEXELS);
   }

   FX64_MOV32(hi, 4); /* cc-chroma = "010" + unused bit */
   for (j = n_vect - 1; j >= 0; j--) {
      for (i = 0; i < n_comp; i++) {
         /* add in colors */
         FX64_SHL(hi, 5);
         FX64_OR32(hi, (uint32_t)(vec[j][i] / 8.0F));
      }
   }
   ((Fx64 *)cc)[1] = hi;

   lohi = lolo = 0;
   /* right microtile */
   for (k = N_TEXELS - 1; k >= N_TEXELS/2; k--) {
      lohi <<= 2;
      lohi |= fxt1_bestcol(vec, n_vect, input[k], n_comp);
   }
   /* left microtile */
   for (; k >= 0; k--) {
      lolo <<= 2;
      lolo |= fxt1_bestcol(vec, n_vect, input[k], n_comp);
   }
   cc[1] = lohi;
   cc[0] = lolo;
}


static void
fxt1_quantize_ALPHA0 (uint32_t *cc,
                      uint8_t input[N_TEXELS][MAX_COMP],
                      uint8_t reord[N_TEXELS][MAX_COMP], int32_t n)
{
   const int32_t n_vect = 3; /* 3 base vectors to find */
   const int32_t n_comp = 4; /* 4 components: R, G, B, A */
   float vec[MAX_VECT][MAX_COMP];
   int32_t i, j, k;
   Fx64 hi; /* high quadword */
   uint32_t lohi, lolo; /* low quadword: hi dword, lo dword */

   /* the last vector indicates zero */
   for (i = 0; i < n_comp; i++) {
      vec[n_vect][i] = 0;
   }

   /* the first n texels in reord are guaranteed to be non-zero */
   if (fxt1_choose(vec, n_vect, reord, n_comp, n) != 0) {
      fxt1_lloyd(vec, n_vect, reord, n_comp, n);
   }

   FX64_MOV32(hi, 6); /* alpha = "011" + lerp = 0 */
   for (j = n_vect - 1; j >= 0; j--) {
      /* add in alphas */
      FX64_SHL(hi, 5);
      FX64_OR32(hi, (uint32_t)(vec[j][ACOMP] / 8.0F));
   }
   for (j = n_vect - 1; j >= 0; j--) {
      for (i = 0; i < n_comp - 1; i++) {
         /* add in colors */
         FX64_SHL(hi, 5);
         FX64_OR32(hi, (uint32_t)(vec[j][i] / 8.0F));
      }
   }
   ((Fx64 *)cc)[1] = hi;

   lohi = lolo = 0;
   /* right microtile */
   for (k = N_TEXELS - 1; k >= N_TEXELS/2; k--) {
      lohi <<= 2;
      lohi |= fxt1_bestcol(vec, n_vect + 1, input[k], n_comp);
   }
   /* left microtile */
   for (; k >= 0; k--) {
      lolo <<= 2;
      lolo |= fxt1_bestcol(vec, n_vect + 1, input[k], n_comp);
   }
   cc[1] = lohi;
   cc[0] = lolo;
}


static void
fxt1_quantize_ALPHA1 (uint32_t *cc,
                      uint8_t input[N_TEXELS][MAX_COMP])
{
   const int32_t n_vect = 3; /* highest vector number in each microtile */
   const int32_t n_comp = 4; /* 4 components: R, G, B, A */
   float vec[1 + 1 + 1][MAX_COMP]; /* 1.5 extrema for each sub-block */
   float b, iv[MAX_COMP]; /* interpolation vector */
   int32_t i, j, k;
   Fx64 hi; /* high quadword */
   uint32_t lohi, lolo; /* low quadword: hi dword, lo dword */

   int32_t minSum;
   int32_t maxSum;
   int32_t minColL = 0, maxColL = 0;
   int32_t minColR = 0, maxColR = 0;
   int32_t sumL = 0, sumR = 0;
   int32_t nn_comp;
   /* Our solution here is to find the darkest and brightest colors in
    * the 4x4 tile and use those as the two representative colors.
    * There are probably better algorithms to use (histogram-based).
    */
   nn_comp = n_comp;
   while ((minColL == maxColL) && nn_comp) {
       minSum = 2000; /* big enough */
       maxSum = -1; /* small enough */
       for (k = 0; k < N_TEXELS / 2; k++) {
           int32_t sum = 0;
           for (i = 0; i < nn_comp; i++) {
               sum += input[k][i];
           }
           if (minSum > sum) {
               minSum = sum;
               minColL = k;
           }
           if (maxSum < sum) {
               maxSum = sum;
               maxColL = k;
           }
           sumL += sum;
       }

       nn_comp--;
   }

   nn_comp = n_comp;
   while ((minColR == maxColR) && nn_comp) {
       minSum = 2000; /* big enough */
       maxSum = -1; /* small enough */
       for (k = N_TEXELS / 2; k < N_TEXELS; k++) {
           int32_t sum = 0;
           for (i = 0; i < nn_comp; i++) {
               sum += input[k][i];
           }
           if (minSum > sum) {
               minSum = sum;
               minColR = k;
           }
           if (maxSum < sum) {
               maxSum = sum;
               maxColR = k;
           }
           sumR += sum;
       }

       nn_comp--;
   }

   /* choose the common vector (yuck!) */
   {
      int32_t j1, j2;
      int32_t v1 = 0, v2 = 0;
      float err = 1e9; /* big enough */
      float tv[2 * 2][MAX_COMP]; /* 2 extrema for each sub-block */
      for (i = 0; i < n_comp; i++) {
         tv[0][i] = input[minColL][i];
         tv[1][i] = input[maxColL][i];
         tv[2][i] = input[minColR][i];
         tv[3][i] = input[maxColR][i];
      }
      for (j1 = 0; j1 < 2; j1++) {
         for (j2 = 2; j2 < 4; j2++) {
            float e = 0.0F;
            for (i = 0; i < n_comp; i++) {
               e += (tv[j1][i] - tv[j2][i]) * (tv[j1][i] - tv[j2][i]);
            }
            if (e < err) {
               err = e;
               v1 = j1;
               v2 = j2;
            }
         }
      }
      for (i = 0; i < n_comp; i++) {
         vec[0][i] = tv[1 - v1][i];
         vec[1][i] = (tv[v1][i] * sumL + tv[v2][i] * sumR) / (sumL + sumR);
         vec[2][i] = tv[5 - v2][i];
      }
   }

   /* left microtile */
   cc[0] = 0;
   if (minColL != maxColL) {
      /* compute interpolation vector */
      MAKEIVEC(n_vect, n_comp, iv, b, vec[0], vec[1]);

      /* add in texels */
      lolo = 0;
      for (k = N_TEXELS / 2 - 1; k >= 0; k--) {
         int32_t texel;
         /* interpolate color */
         CALCCDOT(texel, n_vect, n_comp, iv, b, input[k]);
         /* add in texel */
         lolo <<= 2;
         lolo |= texel;
      }

      cc[0] = lolo;
   }

   /* right microtile */
   cc[1] = 0;
   if (minColR != maxColR) {
      /* compute interpolation vector */
      MAKEIVEC(n_vect, n_comp, iv, b, vec[2], vec[1]);

      /* add in texels */
      lohi = 0;
      for (k = N_TEXELS - 1; k >= N_TEXELS / 2; k--) {
         int32_t texel;
         /* interpolate color */
         CALCCDOT(texel, n_vect, n_comp, iv, b, input[k]);
         /* add in texel */
         lohi <<= 2;
         lohi |= texel;
      }

      cc[1] = lohi;
   }

   FX64_MOV32(hi, 7); /* alpha = "011" + lerp = 1 */
   for (j = n_vect - 1; j >= 0; j--) {
      /* add in alphas */
      FX64_SHL(hi, 5);
      FX64_OR32(hi, (uint32_t)(vec[j][ACOMP] / 8.0F));
   }
   for (j = n_vect - 1; j >= 0; j--) {
      for (i = 0; i < n_comp - 1; i++) {
         /* add in colors */
         FX64_SHL(hi, 5);
         FX64_OR32(hi, (uint32_t)(vec[j][i] / 8.0F));
      }
   }
   ((Fx64 *)cc)[1] = hi;
}


static void
fxt1_quantize_HI (uint32_t *cc,
                  uint8_t input[N_TEXELS][MAX_COMP],
                  uint8_t reord[N_TEXELS][MAX_COMP], int32_t n)
{
   const int32_t n_vect = 6; /* highest vector number */
   const int32_t n_comp = 3; /* 3 components: R, G, B */
   float b = 0.0F;       /* phoudoin: silent compiler! */
   float iv[MAX_COMP];   /* interpolation vector */
   int32_t i, k;
   uint32_t hihi; /* high quadword: hi dword */

   int32_t minSum = 2000; /* big enough */
   int32_t maxSum = -1; /* small enough */
   int32_t minCol = 0; /* phoudoin: silent compiler! */
   int32_t maxCol = 0; /* phoudoin: silent compiler! */

   /* Our solution here is to find the darkest and brightest colors in
    * the 8x4 tile and use those as the two representative colors.
    * There are probably better algorithms to use (histogram-based).
    */
   for (k = 0; k < n; k++) {
      int32_t sum = 0;
      for (i = 0; i < n_comp; i++) {
         sum += reord[k][i];
      }
      if (minSum > sum) {
         minSum = sum;
         minCol = k;
      }
      if (maxSum < sum) {
         maxSum = sum;
         maxCol = k;
      }
   }

   hihi = 0; /* cc-hi = "00" */
   for (i = 0; i < n_comp; i++) {
      /* add in colors */
      hihi <<= 5;
      hihi |= reord[maxCol][i] >> 3;
   }
   for (i = 0; i < n_comp; i++) {
      /* add in colors */
      hihi <<= 5;
      hihi |= reord[minCol][i] >> 3;
   }
   cc[3] = hihi;
   cc[0] = cc[1] = cc[2] = 0;

   /* compute interpolation vector */
   if (minCol != maxCol) {
      MAKEIVEC(n_vect, n_comp, iv, b, reord[minCol], reord[maxCol]);
   }

   /* add in texels */
   for (k = N_TEXELS - 1; k >= 0; k--) {
      int32_t t = k * 3;
      uint32_t *kk = (uint32_t *)((char *)cc + t / 8);
      int32_t texel = n_vect + 1; /* transparent black */

      if (!ISTBLACK(input[k])) {
         if (minCol != maxCol) {
            /* interpolate color */
            CALCCDOT(texel, n_vect, n_comp, iv, b, input[k]);
            /* add in texel */
            kk[0] |= texel << (t & 7);
         }
      } else {
         /* add in texel */
         kk[0] |= texel << (t & 7);
      }
   }
}


static void
fxt1_quantize_MIXED1 (uint32_t *cc,
                      uint8_t input[N_TEXELS][MAX_COMP])
{
   const int32_t n_vect = 2; /* highest vector number in each microtile */
   const int32_t n_comp = 3; /* 3 components: R, G, B */
   uint8_t vec[2 * 2][MAX_COMP]; /* 2 extrema for each sub-block */
   float b, iv[MAX_COMP]; /* interpolation vector */
   int32_t i, j, k;
   Fx64 hi; /* high quadword */
   uint32_t lohi, lolo; /* low quadword: hi dword, lo dword */

   int32_t minSum;
   int32_t maxSum;
   int32_t minColL = 0, maxColL = -1;
   int32_t minColR = 0, maxColR = -1;

   /* Our solution here is to find the darkest and brightest colors in
    * the 4x4 tile and use those as the two representative colors.
    * There are probably better algorithms to use (histogram-based).
    */
   minSum = 2000; /* big enough */
   maxSum = -1; /* small enough */
   for (k = 0; k < N_TEXELS / 2; k++) {
      if (!ISTBLACK(input[k])) {
         int32_t sum = 0;
         for (i = 0; i < n_comp; i++) {
            sum += input[k][i];
         }
         if (minSum > sum) {
            minSum = sum;
            minColL = k;
         }
         if (maxSum < sum) {
            maxSum = sum;
            maxColL = k;
         }
      }
   }
   minSum = 2000; /* big enough */
   maxSum = -1; /* small enough */
   for (; k < N_TEXELS; k++) {
      if (!ISTBLACK(input[k])) {
         int32_t sum = 0;
         for (i = 0; i < n_comp; i++) {
            sum += input[k][i];
         }
         if (minSum > sum) {
            minSum = sum;
            minColR = k;
         }
         if (maxSum < sum) {
            maxSum = sum;
            maxColR = k;
         }
      }
   }

   /* left microtile */
   if (maxColL == -1) {
      /* all transparent black */
      cc[0] = ~0u;
      for (i = 0; i < n_comp; i++) {
         vec[0][i] = 0;
         vec[1][i] = 0;
      }
   } else {
      cc[0] = 0;
      for (i = 0; i < n_comp; i++) {
         vec[0][i] = input[minColL][i];
         vec[1][i] = input[maxColL][i];
      }
      if (minColL != maxColL) {
         /* compute interpolation vector */
         MAKEIVEC(n_vect, n_comp, iv, b, vec[0], vec[1]);

         /* add in texels */
         lolo = 0;
         for (k = N_TEXELS / 2 - 1; k >= 0; k--) {
            int32_t texel = n_vect + 1; /* transparent black */
            if (!ISTBLACK(input[k])) {
               /* interpolate color */
               CALCCDOT(texel, n_vect, n_comp, iv, b, input[k]);
            }
            /* add in texel */
            lolo <<= 2;
            lolo |= texel;
         }
         cc[0] = lolo;
      }
   }

   /* right microtile */
   if (maxColR == -1) {
      /* all transparent black */
      cc[1] = ~0u;
      for (i = 0; i < n_comp; i++) {
         vec[2][i] = 0;
         vec[3][i] = 0;
      }
   } else {
      cc[1] = 0;
      for (i = 0; i < n_comp; i++) {
         vec[2][i] = input[minColR][i];
         vec[3][i] = input[maxColR][i];
      }
      if (minColR != maxColR) {
         /* compute interpolation vector */
         MAKEIVEC(n_vect, n_comp, iv, b, vec[2], vec[3]);

         /* add in texels */
         lohi = 0;
         for (k = N_TEXELS - 1; k >= N_TEXELS / 2; k--) {
            int32_t texel = n_vect + 1; /* transparent black */
            if (!ISTBLACK(input[k])) {
               /* interpolate color */
               CALCCDOT(texel, n_vect, n_comp, iv, b, input[k]);
            }
            /* add in texel */
            lohi <<= 2;
            lohi |= texel;
         }
         cc[1] = lohi;
      }
   }

   FX64_MOV32(hi, 9 | (vec[3][GCOMP] & 4) | ((vec[1][GCOMP] >> 1) & 2)); /* chroma = "1" */
   for (j = 2 * 2 - 1; j >= 0; j--) {
      for (i = 0; i < n_comp; i++) {
         /* add in colors */
         FX64_SHL(hi, 5);
         FX64_OR32(hi, vec[j][i] >> 3);
      }
   }
   ((Fx64 *)cc)[1] = hi;
}


static void
fxt1_quantize_MIXED0 (uint32_t *cc,
                      uint8_t input[N_TEXELS][MAX_COMP])
{
   const int32_t n_vect = 3; /* highest vector number in each microtile */
   const int32_t n_comp = 3; /* 3 components: R, G, B */
   uint8_t vec[2 * 2][MAX_COMP]; /* 2 extrema for each sub-block */
   float b, iv[MAX_COMP]; /* interpolation vector */
   int32_t i, j, k;
   Fx64 hi; /* high quadword */
   uint32_t lohi, lolo; /* low quadword: hi dword, lo dword */

   int32_t minColL = 0, maxColL = 0;
   int32_t minColR = 0, maxColR = 0;
#if 0
   int32_t minSum;
   int32_t maxSum;

   /* Our solution here is to find the darkest and brightest colors in
    * the 4x4 tile and use those as the two representative colors.
    * There are probably better algorithms to use (histogram-based).
    */
   minSum = 2000; /* big enough */
   maxSum = -1; /* small enough */
   for (k = 0; k < N_TEXELS / 2; k++) {
      int32_t sum = 0;
      for (i = 0; i < n_comp; i++) {
         sum += input[k][i];
      }
      if (minSum > sum) {
         minSum = sum;
         minColL = k;
      }
      if (maxSum < sum) {
         maxSum = sum;
         maxColL = k;
      }
   }
   minSum = 2000; /* big enough */
   maxSum = -1; /* small enough */
   for (; k < N_TEXELS; k++) {
      int32_t sum = 0;
      for (i = 0; i < n_comp; i++) {
         sum += input[k][i];
      }
      if (minSum > sum) {
         minSum = sum;
         minColR = k;
      }
      if (maxSum < sum) {
         maxSum = sum;
         maxColR = k;
      }
   }
#else
   int32_t minVal;
   int32_t maxVal;
   int32_t maxVarL = fxt1_variance(input, n_comp);
   int32_t maxVarR = fxt1_variance(&input[N_TEXELS / 2], n_comp);

   /* Scan the channel with max variance for lo & hi
    * and use those as the two representative colors.
    */
   minVal = 2000; /* big enough */
   maxVal = -1; /* small enough */
   for (k = 0; k < N_TEXELS / 2; k++) {
      int32_t t = input[k][maxVarL];
      if (minVal > t) {
         minVal = t;
         minColL = k;
      }
      if (maxVal < t) {
         maxVal = t;
         maxColL = k;
      }
   }
   minVal = 2000; /* big enough */
   maxVal = -1; /* small enough */
   for (; k < N_TEXELS; k++) {
      int32_t t = input[k][maxVarR];
      if (minVal > t) {
         minVal = t;
         minColR = k;
      }
      if (maxVal < t) {
         maxVal = t;
         maxColR = k;
      }
   }
#endif

   /* left microtile */
   cc[0] = 0;
   for (i = 0; i < n_comp; i++) {
      vec[0][i] = input[minColL][i];
      vec[1][i] = input[maxColL][i];
   }
   if (minColL != maxColL) {
      /* compute interpolation vector */
      MAKEIVEC(n_vect, n_comp, iv, b, vec[0], vec[1]);

      /* add in texels */
      lolo = 0;
      for (k = N_TEXELS / 2 - 1; k >= 0; k--) {
         int32_t texel;
         /* interpolate color */
         CALCCDOT(texel, n_vect, n_comp, iv, b, input[k]);
         /* add in texel */
         lolo <<= 2;
         lolo |= texel;
      }

      /* funky encoding for LSB of green */
      if ((int32_t)((lolo >> 1) & 1) != (((vec[1][GCOMP] ^ vec[0][GCOMP]) >> 2) & 1)) {
         for (i = 0; i < n_comp; i++) {
            vec[1][i] = input[minColL][i];
            vec[0][i] = input[maxColL][i];
         }
         lolo = ~lolo;
      }

      cc[0] = lolo;
   }

   /* right microtile */
   cc[1] = 0;
   for (i = 0; i < n_comp; i++) {
      vec[2][i] = input[minColR][i];
      vec[3][i] = input[maxColR][i];
   }
   if (minColR != maxColR) {
      /* compute interpolation vector */
      MAKEIVEC(n_vect, n_comp, iv, b, vec[2], vec[3]);

      /* add in texels */
      lohi = 0;
      for (k = N_TEXELS - 1; k >= N_TEXELS / 2; k--) {
         int32_t texel;
         /* interpolate color */
         CALCCDOT(texel, n_vect, n_comp, iv, b, input[k]);
         /* add in texel */
         lohi <<= 2;
         lohi |= texel;
      }

      /* funky encoding for LSB of green */
      if ((int32_t)((lohi >> 1) & 1) != (((vec[3][GCOMP] ^ vec[2][GCOMP]) >> 2) & 1)) {
         for (i = 0; i < n_comp; i++) {
            vec[3][i] = input[minColR][i];
            vec[2][i] = input[maxColR][i];
         }
         lohi = ~lohi;
      }

      cc[1] = lohi;
   }

   FX64_MOV32(hi, 8 | (vec[3][GCOMP] & 4) | ((vec[1][GCOMP] >> 1) & 2)); /* chroma = "1" */
   for (j = 2 * 2 - 1; j >= 0; j--) {
      for (i = 0; i < n_comp; i++) {
         /* add in colors */
         FX64_SHL(hi, 5);
         FX64_OR32(hi, vec[j][i] >> 3);
      }
   }
   ((Fx64 *)cc)[1] = hi;
}


static void
fxt1_quantize (uint32_t *cc, const uint8_t *lines[], int32_t comps)
{
   int32_t trualpha;
   uint8_t reord[N_TEXELS][MAX_COMP];

   uint8_t input[N_TEXELS][MAX_COMP];
   int32_t i, k, l;

   if (comps == 3) {
      /* make the whole block opaque */
      memset(input, -1, sizeof(input));
   }

   /* 8 texels each line */
   for (l = 0; l < 4; l++) {
      for (k = 0; k < 4; k++) {
         for (i = 0; i < comps; i++) {
            input[k + l * 4][i] = *lines[l]++;
         }
      }
      for (; k < 8; k++) {
         for (i = 0; i < comps; i++) {
            input[k + l * 4 + 12][i] = *lines[l]++;
         }
      }
   }

   /* block layout:
    * 00, 01, 02, 03, 08, 09, 0a, 0b
    * 10, 11, 12, 13, 18, 19, 1a, 1b
    * 04, 05, 06, 07, 0c, 0d, 0e, 0f
    * 14, 15, 16, 17, 1c, 1d, 1e, 1f
    */

   /* [dBorca]
    * stupidity flows forth from this
    */
   l = N_TEXELS;
   trualpha = 0;
   if (comps == 4) {
      /* skip all transparent black texels */
      l = 0;
      for (k = 0; k < N_TEXELS; k++) {
         /* test all components against 0 */
         if (!ISTBLACK(input[k])) {
            /* texel is not transparent black */
            memcpy(reord[l], input[k], 4);
            if (reord[l][ACOMP] < (255 - ALPHA_TS)) {
               /* non-opaque texel */
               trualpha = !0;
            }
            l++;
         }
      }
   }

#if 0
   if (trualpha) {
      fxt1_quantize_ALPHA0(cc, input, reord, l);
   } else if (l == 0) {
      cc[0] = cc[1] = cc[2] = -1;
      cc[3] = 0;
   } else if (l < N_TEXELS) {
      fxt1_quantize_HI(cc, input, reord, l);
   } else {
      fxt1_quantize_CHROMA(cc, input);
   }
   (void)fxt1_quantize_ALPHA1;
   (void)fxt1_quantize_MIXED1;
   (void)fxt1_quantize_MIXED0;
#else
   if (trualpha) {
      fxt1_quantize_ALPHA1(cc, input);
   } else if (l == 0) {
      cc[0] = cc[1] = cc[2] = ~0u;
      cc[3] = 0;
   } else if (l < N_TEXELS) {
      fxt1_quantize_MIXED1(cc, input);
   } else {
      fxt1_quantize_MIXED0(cc, input);
   }
   (void)fxt1_quantize_ALPHA0;
   (void)fxt1_quantize_HI;
   (void)fxt1_quantize_CHROMA;
#endif
}



/**
 * Upscale an image by replication, not (typical) stretching.
 * We use this when the image width or height is less than a
 * certain size (4, 8) and we need to upscale an image.
 */
static void
upscale_teximage2d(int32_t inWidth, int32_t inHeight,
                   int32_t outWidth, int32_t outHeight,
                   int32_t comps, const uint8_t *src, int32_t srcRowStride,
                   uint8_t *dest )
{
   int32_t i, j, k;

   assert(outWidth >= inWidth);
   assert(outHeight >= inHeight);
#if 0
   assert(inWidth == 1 || inWidth == 2 || inHeight == 1 || inHeight == 2);
   assert((outWidth & 3) == 0);
   assert((outHeight & 3) == 0);
#endif

   for (i = 0; i < outHeight; i++) {
      const int32_t ii = i % inHeight;
      for (j = 0; j < outWidth; j++) {
         const int32_t jj = j % inWidth;
         for (k = 0; k < comps; k++) {
            dest[(i * outWidth + j) * comps + k]
               = src[ii * srcRowStride + jj * comps + k];
         }
      }
   }
}


static void
fxt1_encode (uint32_t width, uint32_t height, int32_t comps,
             const void *source, int32_t srcRowStride,
             void *dest, int32_t destRowStride)
{
   uint32_t x, y;
   const uint8_t *data;
   uint32_t *encoded = (uint32_t *)dest;
   void *newSource = NULL;

   assert(comps == 3 || comps == 4);

   /* Replicate image if width is not M8 or height is not M4 */
   if ((width & 7) | (height & 3)) {
      int32_t newWidth = (width + 7) & ~7;
      int32_t newHeight = (height + 3) & ~3;
      newSource = malloc(comps * newWidth * newHeight * sizeof(uint8_t));
      if (!newSource)
         return;
      upscale_teximage2d(width, height, newWidth, newHeight,
                         comps, (const uint8_t *) source,
                         srcRowStride, (uint8_t *) newSource);
      source = newSource;
      width = newWidth;
      height = newHeight;
      srcRowStride = comps * newWidth;
   }

   data = (const uint8_t *) source;
   destRowStride = (destRowStride - width * 2) / 4;
   for (y = 0; y < height; y += 4) {
      uint32_t offs = 0 + (y + 0) * srcRowStride;
      for (x = 0; x < width; x += 8) {
         const uint8_t *lines[4];
         lines[0] = &data[offs];
         lines[1] = lines[0] + srcRowStride;
         lines[2] = lines[1] + srcRowStride;
         lines[3] = lines[2] + srcRowStride;
         offs += 8 * comps;
         fxt1_quantize(encoded, lines, comps);
         /* 128 bits per 8x4 block */
         encoded += 4;
      }
      encoded += destRowStride;
   }

   free(newSource);
}


/***************************************************************************\
 * FXT1 decoder
 *
 * The decoder is based on GL_3DFX_texture_compression_FXT1
 * specification and serves as a concept for the encoder.
\***************************************************************************/


/* lookup table for scaling 5 bit colors up to 8 bits */
static const uint8_t _rgb_scale_5[] = {
   0,   8,   16,  25,  33,  41,  49,  58,
   66,  74,  82,  90,  99,  107, 115, 123,
   132, 140, 148, 156, 165, 173, 181, 189,
   197, 206, 214, 222, 230, 239, 247, 255
};

/* lookup table for scaling 6 bit colors up to 8 bits */
static const uint8_t _rgb_scale_6[] = {
   0,   4,   8,   12,  16,  20,  24,  28,
   32,  36,  40,  45,  49,  53,  57,  61,
   65,  69,  73,  77,  81,  85,  89,  93,
   97,  101, 105, 109, 113, 117, 121, 125,
   130, 134, 138, 142, 146, 150, 154, 158,
   162, 166, 170, 174, 178, 182, 186, 190,
   194, 198, 202, 206, 210, 215, 219, 223,
   227, 231, 235, 239, 243, 247, 251, 255
};


#define CC_SEL(cc, which) (((uint32_t *)(cc))[(which) / 32] >> ((which) & 31))
#define UP5(c) _rgb_scale_5[(c) & 31]
#define UP6(c, b) _rgb_scale_6[(((c) & 31) << 1) | ((b) & 1)]
#define LERP(n, t, c0, c1) (((n) - (t)) * (c0) + (t) * (c1) + (n) / 2) / (n)


static void
fxt1_decode_1HI (const uint8_t *code, int32_t t, uint8_t *rgba)
{
   const uint32_t *cc;

   t *= 3;
   cc = (const uint32_t *)(code + t / 8);
   t = (cc[0] >> (t & 7)) & 7;

   if (t == 7) {
      rgba[RCOMP] = rgba[GCOMP] = rgba[BCOMP] = rgba[ACOMP] = 0;
   } else {
      uint8_t r, g, b;
      cc = (const uint32_t *)(code + 12);
      if (t == 0) {
         b = UP5(CC_SEL(cc, 0));
         g = UP5(CC_SEL(cc, 5));
         r = UP5(CC_SEL(cc, 10));
      } else if (t == 6) {
         b = UP5(CC_SEL(cc, 15));
         g = UP5(CC_SEL(cc, 20));
         r = UP5(CC_SEL(cc, 25));
      } else {
         b = LERP(6, t, UP5(CC_SEL(cc, 0)), UP5(CC_SEL(cc, 15)));
         g = LERP(6, t, UP5(CC_SEL(cc, 5)), UP5(CC_SEL(cc, 20)));
         r = LERP(6, t, UP5(CC_SEL(cc, 10)), UP5(CC_SEL(cc, 25)));
      }
      rgba[RCOMP] = r;
      rgba[GCOMP] = g;
      rgba[BCOMP] = b;
      rgba[ACOMP] = 255;
   }
}


static void
fxt1_decode_1CHROMA (const uint8_t *code, int32_t t, uint8_t *rgba)
{
   const uint32_t *cc;
   uint32_t kk;

   cc = (const uint32_t *)code;
   if (t & 16) {
      cc++;
      t &= 15;
   }
   t = (cc[0] >> (t * 2)) & 3;

   t *= 15;
   cc = (const uint32_t *)(code + 8 + t / 8);
   kk = cc[0] >> (t & 7);
   rgba[BCOMP] = UP5(kk);
   rgba[GCOMP] = UP5(kk >> 5);
   rgba[RCOMP] = UP5(kk >> 10);
   rgba[ACOMP] = 255;
}


static void
fxt1_decode_1MIXED (const uint8_t *code, int32_t t, uint8_t *rgba)
{
   const uint32_t *cc;
   uint32_t col[2][3];
   int32_t glsb, selb;

   cc = (const uint32_t *)code;
   if (t & 16) {
      t &= 15;
      t = (cc[1] >> (t * 2)) & 3;
      /* col 2 */
      col[0][BCOMP] = (*(const uint32_t *)(code + 11)) >> 6;
      col[0][GCOMP] = CC_SEL(cc, 99);
      col[0][RCOMP] = CC_SEL(cc, 104);
      /* col 3 */
      col[1][BCOMP] = CC_SEL(cc, 109);
      col[1][GCOMP] = CC_SEL(cc, 114);
      col[1][RCOMP] = CC_SEL(cc, 119);
      glsb = CC_SEL(cc, 126);
      selb = CC_SEL(cc, 33);
   } else {
      t = (cc[0] >> (t * 2)) & 3;
      /* col 0 */
      col[0][BCOMP] = CC_SEL(cc, 64);
      col[0][GCOMP] = CC_SEL(cc, 69);
      col[0][RCOMP] = CC_SEL(cc, 74);
      /* col 1 */
      col[1][BCOMP] = CC_SEL(cc, 79);
      col[1][GCOMP] = CC_SEL(cc, 84);
      col[1][RCOMP] = CC_SEL(cc, 89);
      glsb = CC_SEL(cc, 125);
      selb = CC_SEL(cc, 1);
   }

   if (CC_SEL(cc, 124) & 1) {
      /* alpha[0] == 1 */

      if (t == 3) {
         /* zero */
         rgba[RCOMP] = rgba[BCOMP] = rgba[GCOMP] = rgba[ACOMP] = 0;
      } else {
         uint8_t r, g, b;
         if (t == 0) {
            b = UP5(col[0][BCOMP]);
            g = UP5(col[0][GCOMP]);
            r = UP5(col[0][RCOMP]);
         } else if (t == 2) {
            b = UP5(col[1][BCOMP]);
            g = UP6(col[1][GCOMP], glsb);
            r = UP5(col[1][RCOMP]);
         } else {
            b = (UP5(col[0][BCOMP]) + UP5(col[1][BCOMP])) / 2;
            g = (UP5(col[0][GCOMP]) + UP6(col[1][GCOMP], glsb)) / 2;
            r = (UP5(col[0][RCOMP]) + UP5(col[1][RCOMP])) / 2;
         }
         rgba[RCOMP] = r;
         rgba[GCOMP] = g;
         rgba[BCOMP] = b;
         rgba[ACOMP] = 255;
      }
   } else {
      /* alpha[0] == 0 */
      uint8_t r, g, b;
      if (t == 0) {
         b = UP5(col[0][BCOMP]);
         g = UP6(col[0][GCOMP], glsb ^ selb);
         r = UP5(col[0][RCOMP]);
      } else if (t == 3) {
         b = UP5(col[1][BCOMP]);
         g = UP6(col[1][GCOMP], glsb);
         r = UP5(col[1][RCOMP]);
      } else {
         b = LERP(3, t, UP5(col[0][BCOMP]), UP5(col[1][BCOMP]));
         g = LERP(3, t, UP6(col[0][GCOMP], glsb ^ selb),
                        UP6(col[1][GCOMP], glsb));
         r = LERP(3, t, UP5(col[0][RCOMP]), UP5(col[1][RCOMP]));
      }
      rgba[RCOMP] = r;
      rgba[GCOMP] = g;
      rgba[BCOMP] = b;
      rgba[ACOMP] = 255;
   }
}


static void
fxt1_decode_1ALPHA (const uint8_t *code, int32_t t, uint8_t *rgba)
{
   const uint32_t *cc;
   uint8_t r, g, b, a;

   cc = (const uint32_t *)code;
   if (CC_SEL(cc, 124) & 1) {
      /* lerp == 1 */
      uint32_t col0[4];

      if (t & 16) {
         t &= 15;
         t = (cc[1] >> (t * 2)) & 3;
         /* col 2 */
         col0[BCOMP] = (*(const uint32_t *)(code + 11)) >> 6;
         col0[GCOMP] = CC_SEL(cc, 99);
         col0[RCOMP] = CC_SEL(cc, 104);
         col0[ACOMP] = CC_SEL(cc, 119);
      } else {
         t = (cc[0] >> (t * 2)) & 3;
         /* col 0 */
         col0[BCOMP] = CC_SEL(cc, 64);
         col0[GCOMP] = CC_SEL(cc, 69);
         col0[RCOMP] = CC_SEL(cc, 74);
         col0[ACOMP] = CC_SEL(cc, 109);
      }

      if (t == 0) {
         b = UP5(col0[BCOMP]);
         g = UP5(col0[GCOMP]);
         r = UP5(col0[RCOMP]);
         a = UP5(col0[ACOMP]);
      } else if (t == 3) {
         b = UP5(CC_SEL(cc, 79));
         g = UP5(CC_SEL(cc, 84));
         r = UP5(CC_SEL(cc, 89));
         a = UP5(CC_SEL(cc, 114));
      } else {
         b = LERP(3, t, UP5(col0[BCOMP]), UP5(CC_SEL(cc, 79)));
         g = LERP(3, t, UP5(col0[GCOMP]), UP5(CC_SEL(cc, 84)));
         r = LERP(3, t, UP5(col0[RCOMP]), UP5(CC_SEL(cc, 89)));
         a = LERP(3, t, UP5(col0[ACOMP]), UP5(CC_SEL(cc, 114)));
      }
   } else {
      /* lerp == 0 */

      if (t & 16) {
         cc++;
         t &= 15;
      }
      t = (cc[0] >> (t * 2)) & 3;

      if (t == 3) {
         /* zero */
         r = g = b = a = 0;
      } else {
         uint32_t kk;
         cc = (const uint32_t *)code;
         a = UP5(cc[3] >> (t * 5 + 13));
         t *= 15;
         cc = (const uint32_t *)(code + 8 + t / 8);
         kk = cc[0] >> (t & 7);
         b = UP5(kk);
         g = UP5(kk >> 5);
         r = UP5(kk >> 10);
      }
   }
   rgba[RCOMP] = r;
   rgba[GCOMP] = g;
   rgba[BCOMP] = b;
   rgba[ACOMP] = a;
}


static void
fxt1_decode_1 (const void *texture, int32_t stride, /* in pixels */
               int32_t i, int32_t j, uint8_t *rgba)
{
   static void (*decode_1[]) (const uint8_t *, int32_t, uint8_t *) = {
      fxt1_decode_1HI,     /* cc-high   = "00?" */
      fxt1_decode_1HI,     /* cc-high   = "00?" */
      fxt1_decode_1CHROMA, /* cc-chroma = "010" */
      fxt1_decode_1ALPHA,  /* alpha     = "011" */
      fxt1_decode_1MIXED,  /* mixed     = "1??" */
      fxt1_decode_1MIXED,  /* mixed     = "1??" */
      fxt1_decode_1MIXED,  /* mixed     = "1??" */
      fxt1_decode_1MIXED   /* mixed     = "1??" */
   };

   const uint8_t *code = (const uint8_t *)texture +
                         ((j / 4) * (stride / 8) + (i / 8)) * 16;
   int32_t mode = CC_SEL(code, 125);
   int32_t t = i & 7;

   if (t & 4) {
      t += 12;
   }
   t += (j & 3) * 4;

   decode_1[mode](code, t, rgba);
}

/*
 * Pixel fetch within a block.
 */

void
util_format_fxt1_rgb_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   fxt1_decode_1(src, 0, i, j, dst);
}

void
util_format_fxt1_rgba_fetch_rgba_8unorm(uint8_t *restrict dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   fxt1_decode_1(src, 0, i, j, dst);
   dst[3] = 0xff;
}

void
util_format_fxt1_rgb_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   fxt1_decode_1(src, 0, i, j, tmp);
   dst[0] = ubyte_to_float(tmp[0]);
   dst[1] = ubyte_to_float(tmp[1]);
   dst[2] = ubyte_to_float(tmp[2]);
   dst[3] = 1.0;
}

void
util_format_fxt1_rgba_fetch_rgba(void *restrict in_dst, const uint8_t *restrict src, unsigned i, unsigned j)
{
   float *dst = in_dst;
   uint8_t tmp[4];
   fxt1_decode_1(src, 0, i, j, tmp);
   dst[0] = ubyte_to_float(tmp[0]);
   dst[1] = ubyte_to_float(tmp[1]);
   dst[2] = ubyte_to_float(tmp[2]);
   dst[3] = ubyte_to_float(tmp[3]);
}

/*
 * Block decompression.
 */

static inline void
util_format_fxtn_rgb_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height,
                                        bool rgba)
{
   const unsigned bw = 8, bh = 4, comps = 4;
   unsigned x, y, i, j;
   for (y = 0; y < height; y += bh) {
      const uint8_t *src = src_row;
      for (x = 0; x < width; x += bw) {
         for (j = 0; j < bh; ++j) {
            for (i = 0; i < bw; ++i) {
               uint8_t *dst = dst_row + (y + j) * dst_stride / sizeof(*dst_row) + (x + i) * comps;
               fxt1_decode_1(src, 0, i, j, dst);
               if (!rgba)
                  dst[3] = 0xff;
            }
         }
         src += FXT1_BLOCK_SIZE;
      }
      src_row += src_stride;
   }
}

void
util_format_fxt1_rgb_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   util_format_fxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           false);
}

void
util_format_fxt1_rgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height)
{
   util_format_fxtn_rgb_unpack_rgba_8unorm(dst_row, dst_stride,
                                           src_row, src_stride,
                                           width, height,
                                           true);
}

static inline void
util_format_fxtn_rgb_unpack_rgba_float(float *dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height,
                                       bool rgba)
{
   const unsigned bw = 8, bh = 4, comps = 4;
   unsigned x, y, i, j;
   for (y = 0; y < height; y += 4) {
      const uint8_t *src = src_row;
      for (x = 0; x < width; x += 8) {
         for (j = 0; j < bh; ++j) {
            for (i = 0; i < bw; ++i) {
               float *dst = dst_row + (y + j)*dst_stride/sizeof(*dst_row) + (x + i) * comps;
               uint8_t tmp[4];
               fxt1_decode_1(src, 0, i, j, tmp);
               dst[0] = ubyte_to_float(tmp[0]);
               dst[1] = ubyte_to_float(tmp[1]);
               dst[2] = ubyte_to_float(tmp[2]);
               if (rgba)
                  dst[3] = ubyte_to_float(tmp[3]);
               else
                  dst[3] = 1.0;
            }
         }
         src += FXT1_BLOCK_SIZE;
      }
      src_row += src_stride;
   }
}

void
util_format_fxt1_rgb_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   util_format_fxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          false);
}

void
util_format_fxt1_rgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height)
{
   util_format_fxtn_rgb_unpack_rgba_float(dst_row, dst_stride,
                                          src_row, src_stride,
                                          width, height,
                                          true);
}

/*
 * Block compression.
 */

void
util_format_fxt1_rgb_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                      const uint8_t *restrict src, unsigned src_stride,
                                      unsigned width, unsigned height)
{
   /* The encoder for FXT1_RGB wants 24bpp packed rgb, so make a temporary to do that.
    */
   int temp_stride = width * 3;
   uint8_t *temp = malloc(height * temp_stride);
   if (!temp)
      return;

   for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
         temp[y * temp_stride + x * 3 + 0] = src[x * 4 + 0];
         temp[y * temp_stride + x * 3 + 1] = src[x * 4 + 1];
         temp[y * temp_stride + x * 3 + 2] = src[x * 4 + 2];
      }
      src += src_stride;
   }

   fxt1_encode(width, height, 3, temp, temp_stride, dst_row, dst_stride);

   free(temp);
}

void
util_format_fxt1_rgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const uint8_t *restrict src, unsigned src_stride,
                                       unsigned width, unsigned height)
{
   fxt1_encode(width, height, 4, src, src_stride, dst_row, dst_stride);
}

void
util_format_fxt1_rgb_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                     const float *restrict src, unsigned src_stride,
                                     unsigned width, unsigned height)
{
   int temp_stride = width * 4;
   uint8_t *temp = malloc(height * temp_stride);
   if (!temp)
      return;

   util_format_r8g8b8a8_unorm_pack_rgba_float(temp, temp_stride,
                                              src, src_stride,
                                              width, height);

   util_format_fxt1_rgb_pack_rgba_8unorm(dst_row, dst_stride,
                                         temp, temp_stride,
                                         width, height);

   free(temp);
}

void
util_format_fxt1_rgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                      const float *restrict src, unsigned src_stride,
                                      unsigned width, unsigned height)
{
   int temp_stride = width * 4;
   uint8_t *temp = malloc(height * temp_stride);
   if (!temp)
      return;

   util_format_r8g8b8a8_unorm_pack_rgba_float(temp, temp_stride,
                                              src, src_stride,
                                              width, height);

   util_format_fxt1_rgba_pack_rgba_8unorm(dst_row, dst_stride,
                                          temp, temp_stride,
                                          width, height);

   free(temp);
}
