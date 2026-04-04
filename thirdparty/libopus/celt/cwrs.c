/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2009 Xiph.Org Foundation
   Copyright (c) 2007-2009 Timothy B. Terriberry
   Written by Timothy B. Terriberry and Jean-Marc Valin */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "os_support.h"
#include "cwrs.h"
#include "mathops.h"
#include "arch.h"

#if defined(CUSTOM_MODES) || defined(ENABLE_QEXT)
#define CWRS_EXTRA_ROWS
#endif

#if defined(CUSTOM_MODES)

/*Guaranteed to return a conservatively large estimate of the binary logarithm
   with frac bits of fractional precision.
  Tested for all possible 32-bit inputs with frac=4, where the maximum
   overestimation is 0.06254243 bits.*/
int log2_frac(opus_uint32 val, int frac)
{
  int l;
  l=EC_ILOG(val);
  if(val&(val-1)){
    /*This is (val>>l-16), but guaranteed to round up, even if adding a bias
       before the shift would cause overflow (e.g., for 0xFFFFxxxx).
       Doesn't work for val=0, but that case fails the test above.*/
    if(l>16)val=((val-1)>>(l-16))+1;
    else val<<=16-l;
    l=(l-1)<<frac;
    /*Note that we always need one iteration, since the rounding up above means
       that we might need to adjust the integer part of the logarithm.*/
    do{
      int b;
      b=(int)(val>>16);
      l+=b<<frac;
      val=(val+b)>>b;
      val=(val*val+0x7FFF)>>15;
    }
    while(frac-->0);
    /*If val is not exactly 0x8000, then we have to round up the remainder.*/
    return l+(val>0x8000);
  }
  /*Exact powers of two require no rounding.*/
  else return (l-1)<<frac;
}
#endif

/*Although derived separately, the pulse vector coding scheme is equivalent to
   a Pyramid Vector Quantizer \cite{Fis86}.
  Some additional notes about an early version appear at
   https://people.xiph.org/~tterribe/notes/cwrs.html, but the codebook ordering
   and the definitions of some terms have evolved since that was written.

  The conversion from a pulse vector to an integer index (encoding) and back
   (decoding) is governed by two related functions, V(N,K) and U(N,K).

  V(N,K) = the number of combinations, with replacement, of N items, taken K
   at a time, when a sign bit is added to each item taken at least once (i.e.,
   the number of N-dimensional unit pulse vectors with K pulses).
  One way to compute this is via
    V(N,K) = K>0 ? sum(k=1...K,2**k*choose(N,k)*choose(K-1,k-1)) : 1,
   where choose() is the binomial function.
  A table of values for N<10 and K<10 looks like:
  V[10][10] = {
    {1,  0,   0,    0,    0,     0,     0,      0,      0,       0},
    {1,  2,   2,    2,    2,     2,     2,      2,      2,       2},
    {1,  4,   8,   12,   16,    20,    24,     28,     32,      36},
    {1,  6,  18,   38,   66,   102,   146,    198,    258,     326},
    {1,  8,  32,   88,  192,   360,   608,    952,   1408,    1992},
    {1, 10,  50,  170,  450,  1002,  1970,   3530,   5890,    9290},
    {1, 12,  72,  292,  912,  2364,  5336,  10836,  20256,   35436},
    {1, 14,  98,  462, 1666,  4942, 12642,  28814,  59906,  115598},
    {1, 16, 128,  688, 2816,  9424, 27008,  68464, 157184,  332688},
    {1, 18, 162,  978, 4482, 16722, 53154, 148626, 374274,  864146}
  };

  U(N,K) = the number of such combinations wherein N-1 objects are taken at
   most K-1 at a time.
  This is given by
    U(N,K) = sum(k=0...K-1,V(N-1,k))
           = K>0 ? (V(N-1,K-1) + V(N,K-1))/2 : 0.
  The latter expression also makes clear that U(N,K) is half the number of such
   combinations wherein the first object is taken at least once.
  Although it may not be clear from either of these definitions, U(N,K) is the
   natural function to work with when enumerating the pulse vector codebooks,
   not V(N,K).
  U(N,K) is not well-defined for N=0, but with the extension
    U(0,K) = K>0 ? 0 : 1,
   the function becomes symmetric: U(N,K) = U(K,N), with a similar table:
  U[10][10] = {
    {1, 0,  0,   0,    0,    0,     0,     0,      0,      0},
    {0, 1,  1,   1,    1,    1,     1,     1,      1,      1},
    {0, 1,  3,   5,    7,    9,    11,    13,     15,     17},
    {0, 1,  5,  13,   25,   41,    61,    85,    113,    145},
    {0, 1,  7,  25,   63,  129,   231,   377,    575,    833},
    {0, 1,  9,  41,  129,  321,   681,  1289,   2241,   3649},
    {0, 1, 11,  61,  231,  681,  1683,  3653,   7183,  13073},
    {0, 1, 13,  85,  377, 1289,  3653,  8989,  19825,  40081},
    {0, 1, 15, 113,  575, 2241,  7183, 19825,  48639, 108545},
    {0, 1, 17, 145,  833, 3649, 13073, 40081, 108545, 265729}
  };

  With this extension, V(N,K) may be written in terms of U(N,K):
    V(N,K) = U(N,K) + U(N,K+1)
   for all N>=0, K>=0.
  Thus U(N,K+1) represents the number of combinations where the first element
   is positive or zero, and U(N,K) represents the number of combinations where
   it is negative.
  With a large enough table of U(N,K) values, we could write O(N) encoding
   and O(min(N*log(K),N+K)) decoding routines, but such a table would be
   prohibitively large for small embedded devices (K may be as large as 32767
   for small N, and N may be as large as 200).

  Both functions obey the same recurrence relation:
    V(N,K) = V(N-1,K) + V(N,K-1) + V(N-1,K-1),
    U(N,K) = U(N-1,K) + U(N,K-1) + U(N-1,K-1),
   for all N>0, K>0, with different initial conditions at N=0 or K=0.
  This allows us to construct a row of one of the tables above given the
   previous row or the next row.
  Thus we can derive O(NK) encoding and decoding routines with O(K) memory
   using only addition and subtraction.

  When encoding, we build up from the U(2,K) row and work our way forwards.
  When decoding, we need to start at the U(N,K) row and work our way backwards,
   which requires a means of computing U(N,K).
  U(N,K) may be computed from two previous values with the same N:
    U(N,K) = ((2*N-1)*U(N,K-1) - U(N,K-2))/(K-1) + U(N,K-2)
   for all N>1, and since U(N,K) is symmetric, a similar relation holds for two
   previous values with the same K:
    U(N,K>1) = ((2*K-1)*U(N-1,K) - U(N-2,K))/(N-1) + U(N-2,K)
   for all K>1.
  This allows us to construct an arbitrary row of the U(N,K) table by starting
   with the first two values, which are constants.
  This saves roughly 2/3 the work in our O(NK) decoding routine, but costs O(K)
   multiplications.
  Similar relations can be derived for V(N,K), but are not used here.

  For N>0 and K>0, U(N,K) and V(N,K) take on the form of an (N-1)-degree
   polynomial for fixed N.
  The first few are
    U(1,K) = 1,
    U(2,K) = 2*K-1,
    U(3,K) = (2*K-2)*K+1,
    U(4,K) = (((4*K-6)*K+8)*K-3)/3,
    U(5,K) = ((((2*K-4)*K+10)*K-8)*K+3)/3,
   and
    V(1,K) = 2,
    V(2,K) = 4*K,
    V(3,K) = 4*K*K+2,
    V(4,K) = 8*(K*K+2)*K/3,
    V(5,K) = ((4*K*K+20)*K*K+6)/3,
   for all K>0.
  This allows us to derive O(N) encoding and O(N*log(K)) decoding routines for
   small N (and indeed decoding is also O(N) for N<3).

  @ARTICLE{Fis86,
    author="Thomas R. Fischer",
    title="A Pyramid Vector Quantizer",
    journal="IEEE Transactions on Information Theory",
    volume="IT-32",
    number=4,
    pages="568--583",
    month=Jul,
    year=1986
  }*/

#if !defined(SMALL_FOOTPRINT)

/*U(N,K) = U(K,N) := N>0?K>0?U(N-1,K)+U(N,K-1)+U(N-1,K-1):0:K>0?1:0*/
# define CELT_PVQ_U(_n,_k) (CELT_PVQ_U_ROW[IMIN(_n,_k)][IMAX(_n,_k)])
/*V(N,K) := U(N,K)+U(N,K+1) = the number of PVQ codewords for a band of size N
   with K pulses allocated to it.*/
# define CELT_PVQ_V(_n,_k) (CELT_PVQ_U(_n,_k)+CELT_PVQ_U(_n,(_k)+1))

/*For each V(N,K) supported, we will access element U(min(N,K+1),max(N,K+1)).
  Thus, the number of entries in row I is the larger of the maximum number of
   pulses we will ever allocate for a given N=I (K=128, or however many fit in
   32 bits, whichever is smaller), plus one, and the maximum N for which
   K=I-1 pulses fit in 32 bits.
  The largest band size in an Opus Custom mode is 208.
  Otherwise, we can limit things to the set of N which can be achieved by
   splitting a band from a standard Opus mode: 176, 144, 96, 88, 72, 64, 48,
   44, 36, 32, 24, 22, 18, 16, 8, 4, 2).*/
#if defined(CWRS_EXTRA_ROWS)
static const opus_uint32 CELT_PVQ_U_DATA[1488]={
#else
static const opus_uint32 CELT_PVQ_U_DATA[1272]={
#endif
  /*N=0, K=0...176:*/
  1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#if defined(CWRS_EXTRA_ROWS)
  /*...208:*/
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0,
#endif
  /*N=1, K=1...176:*/
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#if defined(CWRS_EXTRA_ROWS)
  /*...208:*/
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 1, 1, 1, 1,
#endif
  /*N=2, K=2...176:*/
  3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41,
  43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79,
  81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113,
  115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143,
  145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173,
  175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203,
  205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233,
  235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259, 261, 263,
  265, 267, 269, 271, 273, 275, 277, 279, 281, 283, 285, 287, 289, 291, 293,
  295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315, 317, 319, 321, 323,
  325, 327, 329, 331, 333, 335, 337, 339, 341, 343, 345, 347, 349, 351,
#if defined(CWRS_EXTRA_ROWS)
  /*...208:*/
  353, 355, 357, 359, 361, 363, 365, 367, 369, 371, 373, 375, 377, 379, 381,
  383, 385, 387, 389, 391, 393, 395, 397, 399, 401, 403, 405, 407, 409, 411,
  413, 415,
#endif
  /*N=3, K=3...176:*/
  13, 25, 41, 61, 85, 113, 145, 181, 221, 265, 313, 365, 421, 481, 545, 613,
  685, 761, 841, 925, 1013, 1105, 1201, 1301, 1405, 1513, 1625, 1741, 1861,
  1985, 2113, 2245, 2381, 2521, 2665, 2813, 2965, 3121, 3281, 3445, 3613, 3785,
  3961, 4141, 4325, 4513, 4705, 4901, 5101, 5305, 5513, 5725, 5941, 6161, 6385,
  6613, 6845, 7081, 7321, 7565, 7813, 8065, 8321, 8581, 8845, 9113, 9385, 9661,
  9941, 10225, 10513, 10805, 11101, 11401, 11705, 12013, 12325, 12641, 12961,
  13285, 13613, 13945, 14281, 14621, 14965, 15313, 15665, 16021, 16381, 16745,
  17113, 17485, 17861, 18241, 18625, 19013, 19405, 19801, 20201, 20605, 21013,
  21425, 21841, 22261, 22685, 23113, 23545, 23981, 24421, 24865, 25313, 25765,
  26221, 26681, 27145, 27613, 28085, 28561, 29041, 29525, 30013, 30505, 31001,
  31501, 32005, 32513, 33025, 33541, 34061, 34585, 35113, 35645, 36181, 36721,
  37265, 37813, 38365, 38921, 39481, 40045, 40613, 41185, 41761, 42341, 42925,
  43513, 44105, 44701, 45301, 45905, 46513, 47125, 47741, 48361, 48985, 49613,
  50245, 50881, 51521, 52165, 52813, 53465, 54121, 54781, 55445, 56113, 56785,
  57461, 58141, 58825, 59513, 60205, 60901, 61601,
#if defined(CWRS_EXTRA_ROWS)
  /*...208:*/
  62305, 63013, 63725, 64441, 65161, 65885, 66613, 67345, 68081, 68821, 69565,
  70313, 71065, 71821, 72581, 73345, 74113, 74885, 75661, 76441, 77225, 78013,
  78805, 79601, 80401, 81205, 82013, 82825, 83641, 84461, 85285, 86113,
#endif
  /*N=4, K=4...176:*/
  63, 129, 231, 377, 575, 833, 1159, 1561, 2047, 2625, 3303, 4089, 4991, 6017,
  7175, 8473, 9919, 11521, 13287, 15225, 17343, 19649, 22151, 24857, 27775,
  30913, 34279, 37881, 41727, 45825, 50183, 54809, 59711, 64897, 70375, 76153,
  82239, 88641, 95367, 102425, 109823, 117569, 125671, 134137, 142975, 152193,
  161799, 171801, 182207, 193025, 204263, 215929, 228031, 240577, 253575,
  267033, 280959, 295361, 310247, 325625, 341503, 357889, 374791, 392217,
  410175, 428673, 447719, 467321, 487487, 508225, 529543, 551449, 573951,
  597057, 620775, 645113, 670079, 695681, 721927, 748825, 776383, 804609,
  833511, 863097, 893375, 924353, 956039, 988441, 1021567, 1055425, 1090023,
  1125369, 1161471, 1198337, 1235975, 1274393, 1313599, 1353601, 1394407,
  1436025, 1478463, 1521729, 1565831, 1610777, 1656575, 1703233, 1750759,
  1799161, 1848447, 1898625, 1949703, 2001689, 2054591, 2108417, 2163175,
  2218873, 2275519, 2333121, 2391687, 2451225, 2511743, 2573249, 2635751,
  2699257, 2763775, 2829313, 2895879, 2963481, 3032127, 3101825, 3172583,
  3244409, 3317311, 3391297, 3466375, 3542553, 3619839, 3698241, 3777767,
  3858425, 3940223, 4023169, 4107271, 4192537, 4278975, 4366593, 4455399,
  4545401, 4636607, 4729025, 4822663, 4917529, 5013631, 5110977, 5209575,
  5309433, 5410559, 5512961, 5616647, 5721625, 5827903, 5935489, 6044391,
  6154617, 6266175, 6379073, 6493319, 6608921, 6725887, 6844225, 6963943,
  7085049, 7207551,
#if defined(CWRS_EXTRA_ROWS)
  /*...208:*/
  7331457, 7456775, 7583513, 7711679, 7841281, 7972327, 8104825, 8238783,
  8374209, 8511111, 8649497, 8789375, 8930753, 9073639, 9218041, 9363967,
  9511425, 9660423, 9810969, 9963071, 10116737, 10271975, 10428793, 10587199,
  10747201, 10908807, 11072025, 11236863, 11403329, 11571431, 11741177,
  11912575,
#endif
  /*N=5, K=5...176:*/
  321, 681, 1289, 2241, 3649, 5641, 8361, 11969, 16641, 22569, 29961, 39041,
  50049, 63241, 78889, 97281, 118721, 143529, 172041, 204609, 241601, 283401,
  330409, 383041, 441729, 506921, 579081, 658689, 746241, 842249, 947241,
  1061761, 1186369, 1321641, 1468169, 1626561, 1797441, 1981449, 2179241,
  2391489, 2618881, 2862121, 3121929, 3399041, 3694209, 4008201, 4341801,
  4695809, 5071041, 5468329, 5888521, 6332481, 6801089, 7295241, 7815849,
  8363841, 8940161, 9545769, 10181641, 10848769, 11548161, 12280841, 13047849,
  13850241, 14689089, 15565481, 16480521, 17435329, 18431041, 19468809,
  20549801, 21675201, 22846209, 24064041, 25329929, 26645121, 28010881,
  29428489, 30899241, 32424449, 34005441, 35643561, 37340169, 39096641,
  40914369, 42794761, 44739241, 46749249, 48826241, 50971689, 53187081,
  55473921, 57833729, 60268041, 62778409, 65366401, 68033601, 70781609,
  73612041, 76526529, 79526721, 82614281, 85790889, 89058241, 92418049,
  95872041, 99421961, 103069569, 106816641, 110664969, 114616361, 118672641,
  122835649, 127107241, 131489289, 135983681, 140592321, 145317129, 150160041,
  155123009, 160208001, 165417001, 170752009, 176215041, 181808129, 187533321,
  193392681, 199388289, 205522241, 211796649, 218213641, 224775361, 231483969,
  238341641, 245350569, 252512961, 259831041, 267307049, 274943241, 282741889,
  290705281, 298835721, 307135529, 315607041, 324252609, 333074601, 342075401,
  351257409, 360623041, 370174729, 379914921, 389846081, 399970689, 410291241,
  420810249, 431530241, 442453761, 453583369, 464921641, 476471169, 488234561,
  500214441, 512413449, 524834241, 537479489, 550351881, 563454121, 576788929,
  590359041, 604167209, 618216201, 632508801,
#if defined(CWRS_EXTRA_ROWS)
  /*...208:*/
  647047809, 661836041, 676876329, 692171521, 707724481, 723538089, 739615241,
  755958849, 772571841, 789457161, 806617769, 824056641, 841776769, 859781161,
  878072841, 896654849, 915530241, 934702089, 954173481, 973947521, 994027329,
  1014416041, 1035116809, 1056132801, 1077467201, 1099123209, 1121104041,
  1143412929, 1166053121, 1189027881, 1212340489, 1235994241,
#endif
  /*N=6, K=6...96:*/
  1683, 3653, 7183, 13073, 22363, 36365, 56695, 85305, 124515, 177045, 246047,
  335137, 448427, 590557, 766727, 982729, 1244979, 1560549, 1937199, 2383409,
  2908411, 3522221, 4235671, 5060441, 6009091, 7095093, 8332863, 9737793,
  11326283, 13115773, 15124775, 17372905, 19880915, 22670725, 25765455,
  29189457, 32968347, 37129037, 41699767, 46710137, 52191139, 58175189,
  64696159, 71789409, 79491819, 87841821, 96879431, 106646281, 117185651,
  128542501, 140763503, 153897073, 167993403, 183104493, 199284183, 216588185,
  235074115, 254801525, 275831935, 298228865, 322057867, 347386557, 374284647,
  402823977, 433078547, 465124549, 499040399, 534906769, 572806619, 612825229,
  655050231, 699571641, 746481891, 795875861, 847850911, 902506913, 959946283,
  1020274013, 1083597703, 1150027593, 1219676595, 1292660325, 1369097135,
  1449108145, 1532817275, 1620351277, 1711839767, 1807415257, 1907213187,
  2011371957, 2120032959,
#if defined(CWRS_EXTRA_ROWS)
  /*...109:*/
  2233340609U, 2351442379U, 2474488829U, 2602633639U, 2736033641U, 2874848851U,
  3019242501U, 3169381071U, 3325434321U, 3487575323U, 3655980493U, 3830829623U,
  4012305913U,
#endif
  /*N=7, K=7...54*/
  8989, 19825, 40081, 75517, 134245, 227305, 369305, 579125, 880685, 1303777,
  1884961, 2668525, 3707509, 5064793, 6814249, 9041957, 11847485, 15345233,
  19665841, 24957661, 31388293, 39146185, 48442297, 59511829, 72616013,
  88043969, 106114625, 127178701, 151620757, 179861305, 212358985, 249612805,
  292164445, 340600625, 395555537, 457713341, 527810725, 606639529, 695049433,
  793950709, 904317037, 1027188385, 1163673953, 1314955181, 1482288821,
  1667010073, 1870535785, 2094367717,
#if defined(CWRS_EXTRA_ROWS)
  /*...60:*/
  2340095869U, 2609401873U, 2904062449U, 3225952925U, 3577050821U, 3959439497U,
#endif
  /*N=8, K=8...37*/
  48639, 108545, 224143, 433905, 795455, 1392065, 2340495, 3800305, 5984767,
  9173505, 13726991, 20103025, 28875327, 40754369, 56610575, 77500017,
  104692735, 139703809, 184327311, 240673265, 311207743, 398796225, 506750351,
  638878193, 799538175, 993696769, 1226990095, 1505789553, 1837271615,
  2229491905U,
#if defined(CWRS_EXTRA_ROWS)
  /*...40:*/
  2691463695U, 3233240945U, 3866006015U,
#endif
  /*N=9, K=9...28:*/
  265729, 598417, 1256465, 2485825, 4673345, 8405905, 14546705, 24331777,
  39490049, 62390545, 96220561, 145198913, 214828609, 312193553, 446304145,
  628496897, 872893441, 1196924561, 1621925137, 2173806145U,
#if defined(CWRS_EXTRA_ROWS)
  /*...29:*/
  2883810113U,
#endif
  /*N=10, K=10...24:*/
  1462563, 3317445, 7059735, 14218905, 27298155, 50250765, 89129247, 152951073,
  254831667, 413442773, 654862247, 1014889769, 1541911931, 2300409629U,
  3375210671U,
  /*N=11, K=11...19:*/
  8097453, 18474633, 39753273, 81270333, 158819253, 298199265, 540279585,
  948062325, 1616336765,
#if defined(CWRS_EXTRA_ROWS)
  /*...20:*/
  2684641785U,
#endif
  /*N=12, K=12...18:*/
  45046719, 103274625, 224298231, 464387817, 921406335, 1759885185,
  3248227095U,
  /*N=13, K=13...16:*/
  251595969, 579168825, 1267854873, 2653649025U,
  /*N=14, K=14:*/
  1409933619
};

#if defined(CWRS_EXTRA_ROWS)
static const opus_uint32 *const CELT_PVQ_U_ROW[15]={
  CELT_PVQ_U_DATA+   0,CELT_PVQ_U_DATA+ 208,CELT_PVQ_U_DATA+ 415,
  CELT_PVQ_U_DATA+ 621,CELT_PVQ_U_DATA+ 826,CELT_PVQ_U_DATA+1030,
  CELT_PVQ_U_DATA+1233,CELT_PVQ_U_DATA+1336,CELT_PVQ_U_DATA+1389,
  CELT_PVQ_U_DATA+1421,CELT_PVQ_U_DATA+1441,CELT_PVQ_U_DATA+1455,
  CELT_PVQ_U_DATA+1464,CELT_PVQ_U_DATA+1470,CELT_PVQ_U_DATA+1473
};
#else
static const opus_uint32 *const CELT_PVQ_U_ROW[15]={
  CELT_PVQ_U_DATA+   0,CELT_PVQ_U_DATA+ 176,CELT_PVQ_U_DATA+ 351,
  CELT_PVQ_U_DATA+ 525,CELT_PVQ_U_DATA+ 698,CELT_PVQ_U_DATA+ 870,
  CELT_PVQ_U_DATA+1041,CELT_PVQ_U_DATA+1131,CELT_PVQ_U_DATA+1178,
  CELT_PVQ_U_DATA+1207,CELT_PVQ_U_DATA+1226,CELT_PVQ_U_DATA+1240,
  CELT_PVQ_U_DATA+1248,CELT_PVQ_U_DATA+1254,CELT_PVQ_U_DATA+1257
};
#endif

#if defined(CUSTOM_MODES)
void get_required_bits(opus_int16 *_bits,int _n,int _maxk,int _frac){
  int k;
  /*_maxk==0 => there's nothing to do.*/
  celt_assert(_maxk>0);
  _bits[0]=0;
  for(k=1;k<=_maxk;k++)_bits[k]=log2_frac(CELT_PVQ_V(_n,k),_frac);
}
#endif

static opus_uint32 icwrs(int _n,const int *_y){
  opus_uint32 i;
  int         j;
  int         k;
  celt_assert(_n>=2);
  j=_n-1;
  i=_y[j]<0;
  k=abs(_y[j]);
  do{
    j--;
    i+=CELT_PVQ_U(_n-j,k);
    k+=abs(_y[j]);
    if(_y[j]<0)i+=CELT_PVQ_U(_n-j,k+1);
  }
  while(j>0);
  return i;
}

void encode_pulses(const int *_y,int _n,int _k,ec_enc *_enc){
  celt_assert(_k>0);
  ec_enc_uint(_enc,icwrs(_n,_y),CELT_PVQ_V(_n,_k));
}

static opus_val32 cwrsi(int _n,int _k,opus_uint32 _i,int *_y){
  opus_uint32 p;
  int         s;
  int         k0;
  opus_int16  val;
  opus_val32  yy=0;
  celt_assert(_k>0);
  celt_assert(_n>1);
  while(_n>2){
    opus_uint32 q;
    /*Lots of pulses case:*/
    if(_k>=_n){
      const opus_uint32 *row;
      row=CELT_PVQ_U_ROW[_n];
      /*Are the pulses in this dimension negative?*/
      p=row[_k+1];
      s=-(_i>=p);
      _i-=p&s;
      /*Count how many pulses were placed in this dimension.*/
      k0=_k;
      q=row[_n];
      if(q>_i){
        celt_sig_assert(p>q);
        _k=_n;
        do p=CELT_PVQ_U_ROW[--_k][_n];
        while(p>_i);
      }
      else for(p=row[_k];p>_i;p=row[_k])_k--;
      _i-=p;
      val=(k0-_k+s)^s;
      *_y++=val;
      yy=MAC16_16(yy,val,val);
    }
    /*Lots of dimensions case:*/
    else{
      /*Are there any pulses in this dimension at all?*/
      p=CELT_PVQ_U_ROW[_k][_n];
      q=CELT_PVQ_U_ROW[_k+1][_n];
      if(p<=_i&&_i<q){
        _i-=p;
        *_y++=0;
      }
      else{
        /*Are the pulses in this dimension negative?*/
        s=-(_i>=q);
        _i-=q&s;
        /*Count how many pulses were placed in this dimension.*/
        k0=_k;
        do p=CELT_PVQ_U_ROW[--_k][_n];
        while(p>_i);
        _i-=p;
        val=(k0-_k+s)^s;
        *_y++=val;
        yy=MAC16_16(yy,val,val);
      }
    }
    _n--;
  }
  /*_n==2*/
  p=2*_k+1;
  s=-(_i>=p);
  _i-=p&s;
  k0=_k;
  _k=(_i+1)>>1;
  if(_k)_i-=2*_k-1;
  val=(k0-_k+s)^s;
  *_y++=val;
  yy=MAC16_16(yy,val,val);
  /*_n==1*/
  s=-(int)_i;
  val=(_k+s)^s;
  *_y=val;
  yy=MAC16_16(yy,val,val);
  return yy;
}

opus_val32 decode_pulses(int *_y,int _n,int _k,ec_dec *_dec){
  return cwrsi(_n,_k,ec_dec_uint(_dec,CELT_PVQ_V(_n,_k)),_y);
}

#else /* SMALL_FOOTPRINT */

/*Computes the next row/column of any recurrence that obeys the relation
   u[i][j]=u[i-1][j]+u[i][j-1]+u[i-1][j-1].
  _ui0 is the base case for the new row/column.*/
static OPUS_INLINE void unext(opus_uint32 *_ui,unsigned _len,opus_uint32 _ui0){
  opus_uint32 ui1;
  unsigned      j;
  /*This do-while will overrun the array if we don't have storage for at least
     2 values.*/
  j=1; do {
    ui1=UADD32(UADD32(_ui[j],_ui[j-1]),_ui0);
    _ui[j-1]=_ui0;
    _ui0=ui1;
  } while (++j<_len);
  _ui[j-1]=_ui0;
}

/*Computes the previous row/column of any recurrence that obeys the relation
   u[i-1][j]=u[i][j]-u[i][j-1]-u[i-1][j-1].
  _ui0 is the base case for the new row/column.*/
static OPUS_INLINE void uprev(opus_uint32 *_ui,unsigned _n,opus_uint32 _ui0){
  opus_uint32 ui1;
  unsigned      j;
  /*This do-while will overrun the array if we don't have storage for at least
     2 values.*/
  j=1; do {
    ui1=USUB32(USUB32(_ui[j],_ui[j-1]),_ui0);
    _ui[j-1]=_ui0;
    _ui0=ui1;
  } while (++j<_n);
  _ui[j-1]=_ui0;
}

/*Compute V(_n,_k), as well as U(_n,0..._k+1).
  _u: On exit, _u[i] contains U(_n,i) for i in [0..._k+1].*/
static opus_uint32 ncwrs_urow(unsigned _n,unsigned _k,opus_uint32 *_u){
  opus_uint32 um2;
  unsigned      len;
  unsigned      k;
  len=_k+2;
  /*We require storage at least 3 values (e.g., _k>0).*/
  celt_assert(len>=3);
  _u[0]=0;
  _u[1]=um2=1;
  /*If _n==0, _u[0] should be 1 and the rest should be 0.*/
  /*If _n==1, _u[i] should be 1 for i>1.*/
  celt_assert(_n>=2);
  /*If _k==0, the following do-while loop will overflow the buffer.*/
  celt_assert(_k>0);
  k=2;
  do _u[k]=(k<<1)-1;
  while(++k<len);
  for(k=2;k<_n;k++)unext(_u+1,_k+1,1);
  return _u[_k]+_u[_k+1];
}

/*Returns the _i'th combination of _k elements chosen from a set of size _n
   with associated sign bits.
  _y: Returns the vector of pulses.
  _u: Must contain entries [0..._k+1] of row _n of U() on input.
      Its contents will be destructively modified.*/
static opus_val32 cwrsi(int _n,int _k,opus_uint32 _i,int *_y,opus_uint32 *_u){
  int j;
  opus_int16 val;
  opus_val32 yy=0;
  celt_assert(_n>0);
  j=0;
  do{
    opus_uint32 p;
    int           s;
    int           yj;
    p=_u[_k+1];
    s=-(_i>=p);
    _i-=p&s;
    yj=_k;
    p=_u[_k];
    while(p>_i)p=_u[--_k];
    _i-=p;
    yj-=_k;
    val=(yj+s)^s;
    _y[j]=val;
    yy=MAC16_16(yy,val,val);
    uprev(_u,_k+2,0);
  }
  while(++j<_n);
  return yy;
}

/*Returns the index of the given combination of K elements chosen from a set
   of size 1 with associated sign bits.
  _y: The vector of pulses, whose sum of absolute values is K.
  _k: Returns K.*/
static OPUS_INLINE opus_uint32 icwrs1(const int *_y,int *_k){
  *_k=abs(_y[0]);
  return _y[0]<0;
}

/*Returns the index of the given combination of K elements chosen from a set
   of size _n with associated sign bits.
  _y:  The vector of pulses, whose sum of absolute values must be _k.
  _nc: Returns V(_n,_k).*/
static OPUS_INLINE opus_uint32 icwrs(int _n,int _k,opus_uint32 *_nc,const int *_y,
 opus_uint32 *_u){
  opus_uint32 i;
  int         j;
  int         k;
  /*We can't unroll the first two iterations of the loop unless _n>=2.*/
  celt_assert(_n>=2);
  _u[0]=0;
  for(k=1;k<=_k+1;k++)_u[k]=(k<<1)-1;
  i=icwrs1(_y+_n-1,&k);
  j=_n-2;
  i+=_u[k];
  k+=abs(_y[j]);
  if(_y[j]<0)i+=_u[k+1];
  while(j-->0){
    unext(_u,_k+2,0);
    i+=_u[k];
    k+=abs(_y[j]);
    if(_y[j]<0)i+=_u[k+1];
  }
  *_nc=_u[k]+_u[k+1];
  return i;
}

#if defined(CUSTOM_MODES)
void get_required_bits(opus_int16 *_bits,int _n,int _maxk,int _frac){
  int k;
  /*_maxk==0 => there's nothing to do.*/
  celt_assert(_maxk>0);
  _bits[0]=0;
  if (_n==1)
  {
    for (k=1;k<=_maxk;k++)
      _bits[k] = 1<<_frac;
  }
  else {
    VARDECL(opus_uint32,u);
    SAVE_STACK;
    ALLOC(u,_maxk+2U,opus_uint32);
    ncwrs_urow(_n,_maxk,u);
    for(k=1;k<=_maxk;k++)
      _bits[k]=log2_frac(u[k]+u[k+1],_frac);
    RESTORE_STACK;
  }
}
#endif /* CUSTOM_MODES */

void encode_pulses(const int *_y,int _n,int _k,ec_enc *_enc){
  opus_uint32 i;
  VARDECL(opus_uint32,u);
  opus_uint32 nc;
  SAVE_STACK;
  celt_assert(_k>0);
  ALLOC(u,_k+2U,opus_uint32);
  i=icwrs(_n,_k,&nc,_y,u);
  ec_enc_uint(_enc,i,nc);
  RESTORE_STACK;
}

opus_val32 decode_pulses(int *_y,int _n,int _k,ec_dec *_dec){
  VARDECL(opus_uint32,u);
  int ret;
  SAVE_STACK;
  celt_assert(_k>0);
  ALLOC(u,_k+2U,opus_uint32);
  ret = cwrsi(_n,_k,ec_dec_uint(_dec,ncwrs_urow(_n,_k,u)),_y,u);
  RESTORE_STACK;
  return ret;
}

#endif /* SMALL_FOOTPRINT */
