/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************

  function:
    last mod: $Id: sse2trans.h 15675 2009-02-06 09:43:27Z tterribe $

 ********************************************************************/

#if !defined(_x86_x86zigzag_H)
# define _x86_x86zigzag_H (1)
# include "x86enc.h"


/*Converts DCT coefficients from transposed order into zig-zag scan order and
   stores them in %[y].
  This relies on two macros to load the contents of each row:
   OC_ZZ_LOAD_ROW_LO(row,"reg") and OC_ZZ_LOAD_ROW_HI(row,"reg"), which load
   the first four and second four entries of each row into the specified
   register, respectively.
  OC_ZZ_LOAD_ROW_LO must be called before OC_ZZ_LOAD_ROW_HI for the same row
   (because when the rows are already in SSE2 registers, loading the high half
   destructively modifies the register).
  The index of each output element in the original 64-element array should wind
   up in the following 8x8 matrix (the letters indicate the order we compute
   each 4-tuple below):
    A  0  8  1  2   9 16 24 17 B
    C 10  3  4 11  18 25 32 40 E
    F 33 26 19 12   5  6 13 20 D
    G 27 34 41 48  56 49 42 35 I
    L 28 21 14  7  15 22 29 36 M
    H 43 50 57 58  51 44 37 30 O
    N 23 31 38 45  52 59 60 53 J
    P 46 39 47 54  61 62 55 63 K
  The order of the coefficients within each tuple is reversed in the comments
   below to reflect the usual MSB to LSB notation.*/
#define OC_TRANSPOSE_ZIG_ZAG_MMXEXT \
  OC_ZZ_LOAD_ROW_LO(0,"%%mm0")   /*mm0=03 02 01 00*/ \
  OC_ZZ_LOAD_ROW_LO(1,"%%mm1")   /*mm1=11 10 09 08*/ \
  OC_ZZ_LOAD_ROW_LO(2,"%%mm2")   /*mm2=19 18 17 16*/ \
  OC_ZZ_LOAD_ROW_LO(3,"%%mm3")   /*mm3=27 26 25 24*/ \
  OC_ZZ_LOAD_ROW_HI(0,"%%mm4")   /*mm4=07 06 05 04*/ \
  OC_ZZ_LOAD_ROW_HI(1,"%%mm5")   /*mm5=15 14 13 12*/ \
  OC_ZZ_LOAD_ROW_HI(2,"%%mm6")   /*mm6=23 22 21 20*/ \
  "movq %%mm0,%%mm7\n\t"         /*mm7=03 02 01 00*/ \
  "punpckhdq %%mm1,%%mm0\n\t"    /*mm0=11 10 03 02*/ \
  "pshufw $0x39,%%mm4,%%mm4\n\t" /*mm4=04 07 06 05*/ \
  "punpcklwd %%mm0,%%mm1\n\t"    /*mm1=03 09 02 08*/ \
  "pshufw $0x39,%%mm5,%%mm5\n\t" /*mm5=12 15 14 13*/ \
  "punpcklwd %%mm1,%%mm7\n\t"    /*mm7=02 01 08 00 *A*/ \
  "movq %%mm7,0x00(%[y])\n\t" \
  "punpckhwd %%mm4,%%mm1\n\t"    /*mm1=04 03 07 09*/ \
  "movq %%mm2,%%mm7\n\t"         /*mm7=19 18 17 16*/ \
  "punpckhdq %%mm1,%%mm0\n\t"    /*mm0=04 03 11 10*/ \
  "punpckhwd %%mm5,%%mm7\n\t"    /*mm7=12 19 15 18*/ \
  "punpcklwd %%mm3,%%mm1\n\t"    /*mm1=25 07 24 09*/ \
  "punpcklwd %%mm6,%%mm5\n\t"    /*mm5=21 14 20 13*/ \
  "punpcklwd %%mm2,%%mm1\n\t"    /*mm1=17 24 16 09 *B*/ \
  OC_ZZ_LOAD_ROW_LO(4,"%%mm2")   /*mm2=35 34 33 32*/ \
  "movq %%mm1,0x08(%[y])\n\t" \
  OC_ZZ_LOAD_ROW_LO(5,"%%mm1")   /*mm1=43 42 41 40*/ \
  "pshufw $0x78,%%mm0,%%mm0\n\t" /*mm0=11 04 03 10 *C*/ \
  "movq %%mm0,0x10(%[y])\n\t" \
  "punpckhdq %%mm4,%%mm6\n\t"    /*mm6=?? 07 23 22*/ \
  "punpckldq %%mm5,%%mm4\n\t"    /*mm4=20 13 06 05 *D*/ \
  "movq %%mm4,0x28(%[y])\n\t" \
  "psrlq $16,%%mm3\n\t"          /*mm3=.. 27 26 25*/ \
  "pshufw $0x0E,%%mm2,%%mm0\n\t" /*mm0=?? ?? 35 34*/ \
  "movq %%mm7,%%mm4\n\t"         /*mm4=12 19 15 18*/ \
  "punpcklwd %%mm3,%%mm2\n\t"    /*mm2=26 33 25 32*/ \
  "punpcklwd %%mm1,%%mm4\n\t"    /*mm4=41 15 40 18*/ \
  "punpckhwd %%mm1,%%mm3\n\t"    /*mm3=43 .. 42 27*/ \
  "punpckldq %%mm2,%%mm4\n\t"    /*mm4=25 32 40 18*/ \
  "punpcklwd %%mm0,%%mm3\n\t"    /*mm3=35 42 34 27*/ \
  OC_ZZ_LOAD_ROW_LO(6,"%%mm0")   /*mm0=51 50 49 48*/ \
  "pshufw $0x6C,%%mm4,%%mm4\n\t" /*mm4=40 32 25 18 *E*/ \
  "movq %%mm4,0x18(%[y])\n\t" \
  OC_ZZ_LOAD_ROW_LO(7,"%%mm4")   /*mm4=59 58 57 56*/ \
  "punpckhdq %%mm7,%%mm2\n\t"    /*mm2=12 19 26 33 *F*/ \
  "movq %%mm2,0x20(%[y])\n\t" \
  "pshufw $0xD0,%%mm1,%%mm1\n\t" /*mm1=43 41 ?? ??*/ \
  "pshufw $0x87,%%mm0,%%mm0\n\t" /*mm0=50 48 49 51*/ \
  "movq %%mm3,%%mm2\n\t"         /*mm2=35 42 34 27*/ \
  "punpckhwd %%mm0,%%mm1\n\t"    /*mm1=50 43 48 41*/ \
  "pshufw $0x93,%%mm4,%%mm4\n\t" /*mm4=58 57 56 59*/ \
  "punpckldq %%mm1,%%mm3\n\t"    /*mm3=48 41 34 27 *G*/ \
  "movq %%mm3,0x30(%[y])\n\t" \
  "punpckhdq %%mm4,%%mm1\n\t"    /*mm1=58 57 50 43 *H*/ \
  "movq %%mm1,0x50(%[y])\n\t" \
  OC_ZZ_LOAD_ROW_HI(7,"%%mm1")   /*mm1=63 62 61 60*/ \
  "punpcklwd %%mm0,%%mm4\n\t"    /*mm4=49 56 51 59*/ \
  OC_ZZ_LOAD_ROW_HI(6,"%%mm0")   /*mm0=55 54 53 52*/ \
  "psllq $16,%%mm6\n\t"          /*mm6=07 23 22 ..*/ \
  "movq %%mm4,%%mm3\n\t"         /*mm3=49 56 51 59*/ \
  "punpckhdq %%mm2,%%mm4\n\t"    /*mm4=35 42 49 56 *I*/ \
  OC_ZZ_LOAD_ROW_HI(3,"%%mm2")   /*mm2=31 30 29 28*/ \
  "movq %%mm4,0x38(%[y])\n\t" \
  "punpcklwd %%mm1,%%mm3\n\t"    /*mm3=61 51 60 59*/ \
  "punpcklwd %%mm6,%%mm7\n\t"    /*mm7=22 15 .. ??*/ \
  "movq %%mm3,%%mm4\n\t"         /*mm4=61 51 60 59*/ \
  "punpcklwd %%mm0,%%mm3\n\t"    /*mm3=53 60 52 59*/ \
  "punpckhwd %%mm0,%%mm4\n\t"    /*mm4=55 61 54 51*/ \
  OC_ZZ_LOAD_ROW_HI(4,"%%mm0")   /*mm0=39 38 37 36*/ \
  "pshufw $0xE1,%%mm3,%%mm3\n\t" /*mm3=53 60 59 52 *J*/ \
  "movq %%mm3,0x68(%[y])\n\t" \
  "movq %%mm4,%%mm3\n\t"         /*mm3=?? ?? 54 51*/ \
  "pshufw $0x39,%%mm2,%%mm2\n\t" /*mm2=28 31 30 29*/ \
  "punpckhwd %%mm1,%%mm4\n\t"    /*mm4=63 55 62 61 *K*/ \
  OC_ZZ_LOAD_ROW_HI(5,"%%mm1")   /*mm1=47 46 45 44*/ \
  "movq %%mm4,0x78(%[y])\n\t" \
  "punpckhwd %%mm2,%%mm6\n\t"    /*mm6=28 07 31 23*/ \
  "punpcklwd %%mm0,%%mm2\n\t"    /*mm2=37 30 36 29*/ \
  "punpckhdq %%mm6,%%mm5\n\t"    /*mm5=28 07 21 14*/ \
  "pshufw $0x4B,%%mm2,%%mm2\n\t" /*mm2=36 29 30 37*/ \
  "pshufw $0x87,%%mm5,%%mm5\n\t" /*mm5=07 14 21 28 *L*/ \
  "movq %%mm5,0x40(%[y])\n\t" \
  "punpckhdq %%mm2,%%mm7\n\t"    /*mm7=36 29 22 15 *M*/ \
  "movq %%mm7,0x48(%[y])\n\t" \
  "pshufw $0x9C,%%mm1,%%mm1\n\t" /*mm1=46 45 47 44*/ \
  "punpckhwd %%mm1,%%mm0\n\t"    /*mm0=46 39 45 38*/ \
  "punpcklwd %%mm1,%%mm3\n\t"    /*mm3=47 54 44 51*/ \
  "punpckldq %%mm0,%%mm6\n\t"    /*mm6=45 38 31 23 *N*/ \
  "movq %%mm6,0x60(%[y])\n\t" \
  "punpckhdq %%mm3,%%mm0\n\t"    /*mm0=47 54 46 39*/ \
  "punpckldq %%mm2,%%mm3\n\t"    /*mm3=30 37 44 51 *O*/ \
  "movq %%mm3,0x58(%[y])\n\t" \
  "pshufw $0xB1,%%mm0,%%mm0\n\t" /*mm0=54 47 39 46 *P*/ \
  "movq %%mm0,0x70(%[y])\n\t" \

/*Converts DCT coefficients in %[dct] from natural order into zig-zag scan
   order and stores them in %[qdct].
  The index of each output element in the original 64-element array should wind
   up in the following 8x8 matrix (the letters indicate the order we compute
   each 4-tuple below):
    A  0  1  8 16   9  2  3 10 B
    C 17 24 32 25  18 11  4  5 D
    E 12 19 26 33  40 48 41 34 I
    H 27 20 13  6   7 14 21 28 G
    K 35 42 49 56  57 50 43 36 J
    F 29 22 15 23  30 37 44 51 M
    P 58 59 52 45  38 31 39 46 L
    N 53 60 61 54  47 55 62 63 O
  The order of the coefficients within each tuple is reversed in the comments
   below to reflect the usual MSB to LSB notation.*/
#define OC_ZIG_ZAG_MMXEXT \
  "movq 0x00(%[dct]),%%mm0\n\t"  /*mm0=03 02 01 00*/ \
  "movq 0x08(%[dct]),%%mm1\n\t"  /*mm1=07 06 05 04*/ \
  "movq 0x10(%[dct]),%%mm2\n\t"  /*mm2=11 10 09 08*/ \
  "movq 0x20(%[dct]),%%mm3\n\t"  /*mm3=19 18 17 16*/ \
  "movq 0x30(%[dct]),%%mm4\n\t"  /*mm4=27 26 25 24*/ \
  "movq 0x40(%[dct]),%%mm5\n\t"  /*mm5=35 34 33 32*/ \
  "movq %%mm2,%%mm7\n\t"         /*mm7=11 10 09 08*/ \
  "punpcklwd %%mm3,%%mm2\n\t"    /*mm2=17 09 16 08*/ \
  "movq %%mm0,%%mm6\n\t"         /*mm6=03 02 01 00*/ \
  "punpckldq %%mm2,%%mm0\n\t"    /*mm0=16 08 01 00 *A*/ \
  "movq %%mm0,0x00(%[qdct])\n\t" \
  "movq 0x18(%[dct]),%%mm0\n\t"  /*mm0=15 14 13 12*/ \
  "punpckhdq %%mm6,%%mm6\n\t"    /*mm6=03 02 03 02*/ \
  "psrlq $16,%%mm7\n\t"          /*mm7=.. 11 10 09*/ \
  "punpckldq %%mm7,%%mm6\n\t"    /*mm6=10 09 03 02*/ \
  "punpckhwd %%mm7,%%mm3\n\t"    /*mm3=.. 19 11 18*/ \
  "pshufw $0xD2,%%mm6,%%mm6\n\t" /*mm6=10 03 02 09 *B*/ \
  "movq %%mm6,0x08(%[qdct])\n\t" \
  "psrlq $48,%%mm2\n\t"          /*mm2=.. .. .. 17*/ \
  "movq %%mm1,%%mm6\n\t"         /*mm6=07 06 05 04*/ \
  "punpcklwd %%mm5,%%mm2\n\t"    /*mm2=33 .. 32 17*/ \
  "movq %%mm3,%%mm7\n\t"         /*mm7=.. 19 11 18*/ \
  "punpckldq %%mm1,%%mm3\n\t"    /*mm3=05 04 11 18 *C*/ \
  "por %%mm2,%%mm7\n\t"          /*mm7=33 19 ?? ??*/ \
  "punpcklwd %%mm4,%%mm2\n\t"    /*mm2=25 32 24 17 *D**/ \
  "movq %%mm2,0x10(%[qdct])\n\t" \
  "movq %%mm3,0x18(%[qdct])\n\t" \
  "movq 0x28(%[dct]),%%mm2\n\t"  /*mm2=23 22 21 20*/ \
  "movq 0x38(%[dct]),%%mm1\n\t"  /*mm1=31 30 29 28*/ \
  "pshufw $0x9C,%%mm0,%%mm3\n\t" /*mm3=14 13 15 12*/ \
  "punpckhdq %%mm7,%%mm7\n\t"    /*mm7=33 19 33 19*/ \
  "punpckhwd %%mm3,%%mm6\n\t"    /*mm6=14 07 13 06*/ \
  "punpckldq %%mm0,%%mm0\n\t"    /*mm0=13 12 13 12*/ \
  "punpcklwd %%mm1,%%mm3\n\t"    /*mm3=29 15 28 12*/ \
  "punpckhwd %%mm4,%%mm0\n\t"    /*mm0=27 13 26 12*/ \
  "pshufw $0xB4,%%mm3,%%mm3\n\t" /*mm3=15 29 28 12*/ \
  "psrlq $48,%%mm4\n\t"          /*mm4=.. .. .. 27*/ \
  "punpcklwd %%mm7,%%mm0\n\t"    /*mm0=33 26 19 12 *E*/ \
  "punpcklwd %%mm1,%%mm4\n\t"    /*mm4=29 .. 28 27*/ \
  "punpckhwd %%mm2,%%mm3\n\t"    /*mm3=23 15 22 29 *F*/ \
  "movq %%mm0,0x20(%[qdct])\n\t" \
  "movq %%mm3,0x50(%[qdct])\n\t" \
  "movq 0x60(%[dct]),%%mm3\n\t"  /*mm3=51 50 49 48*/ \
  "movq 0x70(%[dct]),%%mm7\n\t"  /*mm7=59 58 57 56*/ \
  "movq 0x50(%[dct]),%%mm0\n\t"  /*mm0=43 42 41 40*/ \
  "punpcklwd %%mm4,%%mm2\n\t"    /*mm2=28 21 27 20*/ \
  "psrlq $32,%%mm5\n\t"          /*mm5=.. .. 35 34*/ \
  "movq %%mm2,%%mm4\n\t"         /*mm4=28 21 27 20*/ \
  "punpckldq %%mm6,%%mm2\n\t"    /*mm2=13 06 27 20*/ \
  "punpckhdq %%mm4,%%mm6\n\t"    /*mm6=28 21 14 07 *G*/ \
  "movq %%mm3,%%mm4\n\t"         /*mm4=51 50 49 48*/ \
  "pshufw $0xB1,%%mm2,%%mm2\n\t" /*mm2=06 13 20 27 *H*/ \
  "movq %%mm2,0x30(%[qdct])\n\t" \
  "movq %%mm6,0x38(%[qdct])\n\t" \
  "movq 0x48(%[dct]),%%mm2\n\t"  /*mm2=39 38 37 36*/ \
  "punpcklwd %%mm5,%%mm4\n\t"    /*mm4=35 49 34 48*/ \
  "movq 0x58(%[dct]),%%mm5\n\t"  /*mm5=47 46 45 44*/ \
  "punpckldq %%mm7,%%mm6\n\t"    /*mm6=57 56 14 07*/ \
  "psrlq $32,%%mm3\n\t"          /*mm3=.. .. 51 50*/ \
  "punpckhwd %%mm0,%%mm6\n\t"    /*mm6=43 57 42 56*/ \
  "punpcklwd %%mm4,%%mm0\n\t"    /*mm0=34 41 48 40 *I*/ \
  "pshufw $0x4E,%%mm6,%%mm6\n\t" /*mm6=42 56 43 57*/ \
  "movq %%mm0,0x28(%[qdct])\n\t" \
  "punpcklwd %%mm2,%%mm3\n\t"    /*mm3=37 51 36 50*/ \
  "punpckhwd %%mm6,%%mm4\n\t"    /*mm4=42 35 56 49*/ \
  "punpcklwd %%mm3,%%mm6\n\t"    /*mm6=36 43 50 57 *J*/ \
  "pshufw $0x4E,%%mm4,%%mm4\n\t" /*mm4=56 49 42 35 *K*/ \
  "movq %%mm4,0x40(%[qdct])\n\t" \
  "movq %%mm6,0x48(%[qdct])\n\t" \
  "movq 0x68(%[dct]),%%mm6\n\t"  /*mm6=55 54 53 52*/ \
  "movq 0x78(%[dct]),%%mm0\n\t"  /*mm0=63 62 61 60*/ \
  "psrlq $32,%%mm1\n\t"          /*mm1=.. .. 31 30*/ \
  "pshufw $0xD8,%%mm5,%%mm5\n\t" /*mm5=47 45 46 44*/ \
  "pshufw $0x0B,%%mm3,%%mm3\n\t" /*mm3=50 50 51 37*/ \
  "punpcklwd %%mm5,%%mm1\n\t"    /*mm1=46 31 44 30*/ \
  "pshufw $0xC9,%%mm6,%%mm6\n\t" /*mm6=55 52 54 53*/ \
  "punpckhwd %%mm1,%%mm2\n\t"    /*mm2=46 39 31 38 *L*/ \
  "punpcklwd %%mm3,%%mm1\n\t"    /*mm1=51 44 37 30 *M*/ \
  "movq %%mm2,0x68(%[qdct])\n\t" \
  "movq %%mm1,0x58(%[qdct])\n\t" \
  "punpckhwd %%mm6,%%mm5\n\t"    /*mm5=55 47 52 45*/ \
  "punpckldq %%mm0,%%mm6\n\t"    /*mm6=61 60 54 53*/ \
  "pshufw $0x10,%%mm5,%%mm4\n\t" /*mm4=45 52 45 45*/ \
  "pshufw $0x78,%%mm6,%%mm6\n\t" /*mm6=53 60 61 54 *N*/ \
  "punpckhdq %%mm0,%%mm5\n\t"    /*mm5=63 62 55 47 *O*/ \
  "punpckhdq %%mm4,%%mm7\n\t"    /*mm7=45 52 59 58 *P*/ \
  "movq %%mm6,0x70(%[qdct])\n\t" \
  "movq %%mm5,0x78(%[qdct])\n\t" \
  "movq %%mm7,0x60(%[qdct])\n\t" \

#endif
