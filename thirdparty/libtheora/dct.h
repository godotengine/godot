/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2009                *
 * by the Xiph.Org Foundation and contributors                      *
 * https://www.xiph.org/                                            *
 *                                                                  *
 ********************************************************************

  function:

 ********************************************************************/

/*Definitions shared by the forward and inverse DCT transforms.*/
#if !defined(_dct_H)
# define _dct_H (1)

/*cos(n*pi/16) (resp. sin(m*pi/16)) scaled by 65536.*/
#define OC_C1S7 ((ogg_int32_t)64277)
#define OC_C2S6 ((ogg_int32_t)60547)
#define OC_C3S5 ((ogg_int32_t)54491)
#define OC_C4S4 ((ogg_int32_t)46341)
#define OC_C5S3 ((ogg_int32_t)36410)
#define OC_C6S2 ((ogg_int32_t)25080)
#define OC_C7S1 ((ogg_int32_t)12785)

#endif
