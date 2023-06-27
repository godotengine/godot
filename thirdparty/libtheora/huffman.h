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
    last mod: $Id$

 ********************************************************************/

#if !defined(_huffman_H)
# define _huffman_H (1)
# include "theora/codec.h"
# include "ocintrin.h"

/*The range of valid quantized DCT coefficient values.
  VP3 used 511 in the encoder, but the bitstream is capable of 580.*/
#define OC_DCT_VAL_RANGE         (580)

#define OC_NDCT_TOKEN_BITS       (5)

#define OC_DCT_EOB1_TOKEN        (0)
#define OC_DCT_EOB2_TOKEN        (1)
#define OC_DCT_EOB3_TOKEN        (2)
#define OC_DCT_REPEAT_RUN0_TOKEN (3)
#define OC_DCT_REPEAT_RUN1_TOKEN (4)
#define OC_DCT_REPEAT_RUN2_TOKEN (5)
#define OC_DCT_REPEAT_RUN3_TOKEN (6)

#define OC_DCT_SHORT_ZRL_TOKEN   (7)
#define OC_DCT_ZRL_TOKEN         (8)

#define OC_ONE_TOKEN             (9)
#define OC_MINUS_ONE_TOKEN       (10)
#define OC_TWO_TOKEN             (11)
#define OC_MINUS_TWO_TOKEN       (12)

#define OC_DCT_VAL_CAT2          (13)
#define OC_DCT_VAL_CAT3          (17)
#define OC_DCT_VAL_CAT4          (18)
#define OC_DCT_VAL_CAT5          (19)
#define OC_DCT_VAL_CAT6          (20)
#define OC_DCT_VAL_CAT7          (21)
#define OC_DCT_VAL_CAT8          (22)

#define OC_DCT_RUN_CAT1A         (23)
#define OC_DCT_RUN_CAT1B         (28)
#define OC_DCT_RUN_CAT1C         (29)
#define OC_DCT_RUN_CAT2A         (30)
#define OC_DCT_RUN_CAT2B         (31)

#define OC_NDCT_EOB_TOKEN_MAX    (7)
#define OC_NDCT_ZRL_TOKEN_MAX    (9)
#define OC_NDCT_VAL_MAX          (23)
#define OC_NDCT_VAL_CAT1_MAX     (13)
#define OC_NDCT_VAL_CAT2_MAX     (17)
#define OC_NDCT_VAL_CAT2_SIZE    (OC_NDCT_VAL_CAT2_MAX-OC_DCT_VAL_CAT2)
#define OC_NDCT_RUN_MAX          (32)
#define OC_NDCT_RUN_CAT1A_MAX    (28)

extern const unsigned char OC_DCT_TOKEN_EXTRA_BITS[TH_NDCT_TOKENS];

#endif
